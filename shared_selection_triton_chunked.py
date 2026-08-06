#!/usr/bin/env python3
"""
shared_selection_triton_chunked.py

CHUNKED kernel ("chunk_exact"): selection at CHUNK granularity instead of per key.

    Kernel A  (unchanged, imported)  -> per-key votes + weighted scores
    _build_sets_chunk (NEW)          -> aggregate scores into fixed chunks, keep the top n_chunk
                                        plus forced global/local chunks, dense-rank
    Kernel C  (unchanged, imported)  -> gather + base RoPE at dense ranks + GEMM

NO NEW TRITON. Kernel A and Kernel C are imported verbatim from shared_selection_triton_perquery --
this is deliberate. The five existing kernels are near-duplicate files, and every optimisation has had
to be applied five times (the sort->scatter compaction, then the sort->topk order statistics). Sharing
the kernels means fixes propagate, and this module inherits both of those memory wins for free.
Kernel C is used in POOLED mode (Mask = -1 on every valid slot => all 32 queries attend the whole
selected set), which is exactly how the pq decode path already drives it.

WHY CHUNKS
----------
The per-key union is data-dependent: measured at ~8.5*L^0.696, crossing k_max=4096 at ~7.1k tokens,
overflowing on 68% of deep tiles at 12k. Every overflow silently clips someone's keys. With chunk
selection the budget is EXACT and known before any data is seen:

    kept <= (n_forced_chunks + n_chunk) * chunk_size          (equality except at the causal edge)

so k_max stops being a guess that gets validated afterwards. Secondary wins: the top-k runs over
N/C chunks instead of N keys (1536 vs 24576 at 24k with C=16), and keys inside a chunk are
contiguous, so dense ranks preserve intra-chunk relative distance EXACTLY -- only inter-chunk gaps
compress.

WHAT IT COSTS (measured, measure_chunk_recall.py, PG-19, geom=pi, expmass)
    recall of threshold-passers' softmax mass, at ~4096 slots:
        L=8192   per-key 97.4%   chunked C=16 93.7%    (-3.7pp)
        L=16384  per-key 88.3%   chunked C=16 78.7%    (-9.7pp)
        L=24576  per-key 78.7%   chunked C=16 67.6%   (-11.1pp)
    Chunk size barely matters ABOVE 16 (67.6 / 66.9 / 66.6 / 66.4 for C=16/32/64/128 at 24k), so
    C=16 is the default: it is the finest measured, and it also spends the least on forced chunks
    (7 chunks = 112 slots, vs 3 chunks = 384 slots at C=128, because global/local round UP to whole
    chunks). `sum` beat `mean` aggregation at every measured point.

Batch size 1 (q,k,v are (1,H,N,D)). Prefill + decode.
"""
import math
import torch

from chunk_recall_core import forced_chunk_mask          # shared with the Phase-0 measurement, so the
                                                         # kernel and the sweep mask identically
try:
    from transformers.models.llama.modeling_llama import repeat_kv
except Exception:
    repeat_kv = None

from shared_selection_triton_perquery import (HAVE_TRITON, _rope_rotate,
                                               _decode_scores_kernel)

# Kernel A and Kernel C are defined INSIDE `if HAVE_TRITON:` in the source module, so importing them
# unconditionally would make this module unimportable on any machine without Triton -- including for
# the CPU-only parts (budget arithmetic, _build_sets_chunk, the golden reference), which are exactly
# what the unit tests exercise. Guard the import the same way the source guards the definition.
if HAVE_TRITON:
    from shared_selection_triton_perquery import (
        _select_bits_kernel,      # Kernel A  -- selection + votes + weighted scores
        _gather_attn_pq_kernel,   # Kernel C  -- gather + dense-rank RoPE + online-softmax GEMM
    )


# ──────────────────────────────────────────────────────────────────────────────
# MEASUREMENT-ONLY chunk-quality counters (do NOT affect execution).
# Accumulated on device (no host sync during the run); read once via get_chunk_stats().
# ──────────────────────────────────────────────────────────────────────────────
def _fresh():
    return {"cells": 0, "cap_cnt": None, "tot_cnt": None, "slots": None}


_CHUNK_STATS = {"prefill": _fresh(), "decode": _fresh()}


def reset_chunk_stats():
    _CHUNK_STATS["prefill"] = _fresh()
    _CHUNK_STATS["decode"] = _fresh()


def _phase(d):
    tot = d["cells"]
    cap = float(d["cap_cnt"].item()) if d["cap_cnt"] is not None else 0.0
    all_ = float(d["tot_cnt"].item()) if d["tot_cnt"] is not None else 0.0
    slt = float(d["slots"].item()) if d["slots"] is not None else 0.0
    return {
        "cells": tot,
        "recall_count": (cap / all_) if all_ else 0.0,   # passers kept / passers available
        "utilisation": (cap / slt) if slt else 0.0,      # passers kept / slots gathered
        "slots_per_cell": (slt / tot) if tot else 0.0,
    }


def get_chunk_stats():
    """How well chunk selection is doing, per phase. recall_count = fraction of threshold-passing
    keys that landed inside a selected chunk; utilisation = fraction of gathered slots that are
    passers (the rest are passengers riding along inside a selected chunk)."""
    return {"prefill": _phase(_CHUNK_STATS["prefill"]), "decode": _phase(_CHUNK_STATS["decode"])}


def _stat_acc(cap, tot, slots, cells, phase="prefill"):
    d = _CHUNK_STATS[phase]
    d["cells"] += cells
    for k, v in (("cap_cnt", cap), ("tot_cnt", tot), ("slots", slots)):
        d[k] = v if d[k] is None else d[k] + v


# ──────────────────────────────────────────────────────────────────────────────
# Budget arithmetic -- the whole point of this kernel is that these are EXACT.
# ──────────────────────────────────────────────────────────────────────────────
def n_forced_chunks(chunk_size, n_global, n_local, tile):
    """Worst-case number of chunks the forced global+local sets occupy.

    global occupies ceil(n_global / C) chunks. The local span is [tile_start - n_local + 1,
    tile_start + tile - 1], i.e. n_local + tile - 1 keys, which straddles at most
    ceil((n_local + tile - 1)/C) + 1 chunks depending on where the tile falls against the chunk grid
    (the +1 is the misalignment). Worst case assumes global and local do not overlap, which holds for
    every tile past the start of the sequence.
    """
    C = chunk_size
    g = (n_global + C - 1) // C
    l = (n_local + tile - 1 + C - 1) // C + 1
    return g + l


def derive_n_chunk(k_max, chunk_size, n_global, n_local, tile):
    """How many chunks to SELECT so that forced + selected fills k_max exactly.

        (n_forced + n_chunk) * C = k_max      ->   n_chunk = k_max/C - n_forced

    Note the forced part is counted in whole CHUNKS, not raw tokens: at C=16 the 32 global + 63 local
    keys round up to 7 chunks = 112 slots, so n_chunk = 4096/16 - 7 = 249, not (4096-95)/16.
    """
    C = chunk_size
    assert k_max % C == 0, f"k_max={k_max} must be divisible by chunk_size={C}"
    nf = n_forced_chunks(C, n_global, n_local, tile)
    n = k_max // C - nf
    assert n > 0, (f"chunk_size={C} leaves no budget: forced needs {nf} chunks ({nf*C} slots) of "
                   f"k_max={k_max}. Raise k_max or lower chunk_size/n_global/n_local.")
    return n


# ──────────────────────────────────────────────────────────────────────────────
# HOST: chunk-granular set construction (the only new logic in this file)
# ──────────────────────────────────────────────────────────────────────────────
def _build_sets_chunk(votes, wvotes, n_global, n_local, T, k_max, chunk_size, n_chunk,
                      q_pos_base=0, select_score="expmass", chunk_agg="sum", measure=False):
    """votes (H,nt,N) int, wvotes (H,nt,N) fp32. Returns:
      idx      (H,nt,k_max) int32  kept-key positions ascending, sentinel=N
      newpos_k (H,nt,k_max) fp32   dense rank of each kept key
      newpos_q (H,nt,T)     fp32   dense rank of each query's own position
      mask     (H,nt,k_max) int32  -1 on valid slots (POOLED: every query attends the whole set)
      usize    (H,nt) int32        threshold-passers visible to the tile (pre-selection)
      kept_size(H,nt) int32        |selected keys| -- deterministic, <= k_max BY CONSTRUCTION
    """
    H, nt, N = votes.shape
    dev = votes.device
    C = chunk_size
    nC = (N + C - 1) // C
    Np = nC * C
    NEG = torch.finfo(torch.float32).min
    kj = torch.arange(N, device=dev)

    # ── per-key score, zeroed on non-passers: a chunk is ranked by what its THRESHOLD-PASSING keys
    #    contribute, not by every key it happens to contain ──
    cand = votes > 0
    rank_val = votes.float() if (select_score == "freq" or wvotes is None
                                 or wvotes.shape[-1] != N) else wvotes.float()
    s_masked = torch.where(cand, rank_val, torch.zeros_like(rank_val))

    pad = Np - N
    sm = torch.nn.functional.pad(s_masked, (0, pad)) if pad else s_masked
    mass_c = sm.view(H, nt, nC, C).sum(-1)                                  # (H,nt,nC)
    if chunk_agg == "mean":
        cp = torch.nn.functional.pad(cand.float(), (0, pad)) if pad else cand.float()
        cnt_c = cp.view(H, nt, nC, C).sum(-1)
        rank_c = mass_c / cnt_c.clamp(min=1)
    else:
        rank_c = mass_c

    # ── forced (global ∪ local) and causal visibility, both at CHUNK granularity ──
    forced, visible = forced_chunk_mask(nt, nC, C, N, T, n_global, n_local, q_pos_base, dev)

    # ── rank the non-forced visible chunks; forced ones are kept regardless so they must not
    #    occupy ranking slots ──
    F = forced[None].expand(H, nt, nC)
    V = visible[None].expand(H, nt, nC)
    rank_c = torch.where(F | ~V, torch.full_like(rank_c, NEG), rank_c)

    n_take = min(n_chunk, nC)
    tv, ti = rank_c.topk(n_take, dim=-1)
    ok = tv > NEG                                                           # a real candidate
    sel = torch.zeros((H, nt, nC), dtype=torch.bool, device=dev)
    sel.scatter_(-1, ti, ok)
    sel |= F & V

    # ── expand chunks -> keys, then TRIM the causal edge. A visible chunk may extend past the
    #    tile's last query; those keys are non-causal for every query in the tile. Kernel C would
    #    mask them anyway, but keeping them would burn slots and push the dense ranks of nothing. ──
    kept = sel.unsqueeze(-1).expand(H, nt, nC, C).reshape(H, nt, Np)[:, :, :N]
    last_q = q_pos_base + torch.arange(nt, device=dev) * T + (T - 1)         # (nt,)
    kept = kept & (kj[None, None, :] <= last_q[None, :, None].clamp(max=N - 1))
    kept_size = kept.sum(-1, dtype=torch.int32)
    usize = (cand & (kj[None, None, :] <= last_q[None, :, None])).sum(-1, dtype=torch.int32)

    if measure:
        _stat_acc((cand & kept).sum(dtype=torch.float64),
                  usize.sum(dtype=torch.float64),
                  kept_size.sum(dtype=torch.float64), H * nt, "prefill")

    # ── dense ranks over the kept set: contiguous WITHIN a chunk, so intra-chunk relative
    #    distances survive exactly; only inter-chunk gaps compress ──
    rank0 = (kept.int().cumsum(-1) - 1).float()

    # Compaction by SCATTER (same as the pq kernel): rank0 already holds each kept key's destination
    # slot, so this is O(N) and never materialises an (H,nt,N) int64 tensor the way sort did.
    # kept_size <= k_max holds by construction here, so the rank0 < k_max guard is belt-and-braces.
    slot = torch.where(kept & (rank0 < k_max), rank0.long(),
                       torch.full_like(rank0, k_max, dtype=torch.long))
    idx = torch.full((H, nt, k_max + 1), N, device=dev, dtype=torch.long)
    idx.scatter_(-1, slot, kj[None, None, :].expand(H, nt, N))
    idx = idx[..., :k_max]                                                  # drop the dump slot

    idx_clamp = idx.clamp(max=N - 1)
    valid = idx < N
    newpos_k = torch.gather(rank0, -1, idx_clamp)
    newpos_k = torch.where(valid, newpos_k, torch.zeros_like(newpos_k))

    # POOLED: every query attends every selected key (subject to Kernel C's causal test).
    mask = torch.where(valid, torch.full((H, nt, k_max), -1, dtype=torch.int32, device=dev),
                       torch.zeros((H, nt, k_max), dtype=torch.int32, device=dev))

    qpos = q_pos_base + (torch.arange(nt, device=dev)[:, None] * T
                         + torch.arange(T, device=dev)[None, :])
    qpos = qpos.clamp(max=N - 1)              # ragged last tile: padded query slots stay in range
    newpos_q = torch.gather(rank0, -1, qpos[None].expand(H, nt, T))

    return (idx.to(torch.int32).contiguous(), newpos_k.float().contiguous(),
            newpos_q.float().contiguous(), mask.contiguous(),
            usize.contiguous(), kept_size.contiguous())


# ──────────────────────────────────────────────────────────────────────────────
# PREFILL orchestration
# ──────────────────────────────────────────────────────────────────────────────
def shared_chunk_attention(q, k, v, tau, inv_freq_a, inv_freq_c, pos_scale_a=1.0, rope_scale_a=1.0,
                           sm_scale=None, n_global=32, n_local=32, tile=32, k_max=4096,
                           chunk_size=16, n_chunk=None, block_n=64, block_k=64, q_pos_base=0,
                           select_score="expmass", chunk_agg="sum", measure=False):
    """q,k,v: (1,H,N,D) RAW. Kernel A selects under the a-geometry; Kernel C attends under BASE RoPE
    at dense ranks (always in-window). Returns (out, usize, kept_size)."""
    assert HAVE_TRITON, "Triton not available"
    Z, H, N, D = q.shape
    assert Z == 1
    if n_chunk is None:
        n_chunk = derive_n_chunk(k_max, chunk_size, n_global, n_local, tile)
    # Budget is exact and STATIC -- assert it before any data is seen. This is the property the
    # whole design exists for; if it fails the scatter would drop keys silently.
    nf = n_forced_chunks(chunk_size, n_global, n_local, tile)
    nC_total = (N + chunk_size - 1) // chunk_size
    # A tile keeps at most min(all chunks, forced + selected) chunks: you cannot select more chunks
    # than exist, and forced/selected overlap (sel |= forced). Bounding by (nf + n_chunk) alone would
    # reject perfectly valid configs where n_chunk approaches nC -- e.g. "select every chunk".
    bound = min(nC_total, nf + n_chunk) * chunk_size
    assert bound <= k_max, (
        f"budget overflow: min(nC={nC_total}, {nf} forced + {n_chunk} selected) * {chunk_size} = "
        f"{bound} > k_max={k_max}")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    nt = (N + tile - 1) // tile
    qh, kh, vh = q[0].contiguous(), k[0].contiguous(), v[0].contiguous()
    tau = tau.float().contiguous()
    invfa = inv_freq_a.float().contiguous()
    invfc = inv_freq_c.float().contiguous()

    votes = torch.zeros((H, nt, N), dtype=torch.int32, device=q.device)
    row_counts = torch.zeros((H, N), dtype=torch.int32, device=q.device)
    bits = torch.zeros((H, nt, N), dtype=torch.int32, device=q.device)   # unused (pooled), Kernel A
    WMODE = {"freq": 0, "margin": 1, "expmass": 2}[select_score]         # writes it regardless
    wvotes = torch.zeros((H, nt, N) if WMODE > 0 else (H, 1, 1),
                         dtype=torch.float32, device=q.device)
    gridA = (nt, H)
    _select_bits_kernel[gridA](
        qh, kh, tau, invfa, votes, row_counts, wvotes, bits,
        sm_scale, float(pos_scale_a), float(rope_scale_a), int(q_pos_base),
        qh.stride(0), qh.stride(1), qh.stride(2),
        kh.stride(0), kh.stride(1), kh.stride(2),
        votes.stride(0), votes.stride(1), votes.stride(2),
        row_counts.stride(0), row_counts.stride(1),
        wvotes.stride(0), wvotes.stride(1), wvotes.stride(2),
        bits.stride(0), bits.stride(1), bits.stride(2),
        H, N, BLOCK_M=tile, BLOCK_N=block_n, HEAD_DIM=D, HALF_DIM=D // 2, WMODE=WMODE,
        num_warps=4, num_stages=1,
    )
    del bits

    idx, newpos_k, newpos_q, mask, usize, kept_size = _build_sets_chunk(
        votes, wvotes, n_global, n_local, tile, k_max, chunk_size, n_chunk,
        q_pos_base=q_pos_base, select_score=select_score, chunk_agg=chunk_agg, measure=measure)

    out = torch.empty((H, N, D), dtype=q.dtype, device=q.device)
    gridC = (nt, H)
    _gather_attn_pq_kernel[gridC](
        qh, kh, vh, invfc, idx, newpos_k, newpos_q, mask, out,
        sm_scale, 1.0, int(q_pos_base),
        qh.stride(0), qh.stride(1), qh.stride(2),
        kh.stride(0), kh.stride(1), kh.stride(2),
        vh.stride(0), vh.stride(1), vh.stride(2),
        idx.stride(0), idx.stride(1), idx.stride(2),
        newpos_k.stride(0), newpos_k.stride(1), newpos_k.stride(2),
        newpos_q.stride(0), newpos_q.stride(1), newpos_q.stride(2),
        mask.stride(0), mask.stride(1), mask.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H, N, BLOCK_M=tile, BLOCK_K=block_k, K_MAX=k_max, HEAD_DIM=D, HALF_DIM=D // 2,
        num_warps=4, num_stages=1,
    )
    return out.unsqueeze(0), usize, kept_size


# ──────────────────────────────────────────────────────────────────────────────
# GOLDEN REFERENCE (pure torch, independent of Triton and of _build_sets_chunk).
# O(nt * N^2) -- small N only.
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def chunk_attention_torch(q, k, v, tau, inv_freq_a, inv_freq_c, pos_scale_a=1.0, rope_scale_a=1.0,
                          sm_scale=None, n_global=32, n_local=32, tile=32, k_max=4096,
                          chunk_size=16, n_chunk=None, select_score="expmass", chunk_agg="sum"):
    """Reference forward. q,k,v (1,H,N,D). Returns (1,H,N,D) in q.dtype."""
    Z, H, N, D = q.shape
    assert Z == 1
    T, C = tile, chunk_size
    nt = (N + T - 1) // T
    if n_chunk is None:
        n_chunk = derive_n_chunk(k_max, C, n_global, n_local, T)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    dev = q.device
    NEG = float("-inf")
    qf, kf, vf = q[0].float(), k[0].float(), v[0].float()
    invfa, invfc = inv_freq_a.to(dev).float(), inv_freq_c.to(dev).float()
    tau = tau.to(dev).float()
    out = torch.zeros((H, N, D), device=dev, dtype=torch.float32)

    for t in range(nt):
        r0, r1 = t * T, min((t + 1) * T, N)
        qpos = torch.arange(r0, r1, device=dev)
        end = min((t + 1) * T, N)
        ks = torch.arange(end, device=dev)

        qs = _rope_rotate(qf[:, r0:r1], qpos.float() * pos_scale_a, invfa, rope_scale_a)
        kks = _rope_rotate(kf[:, :end], ks.float() * pos_scale_a, invfa, rope_scale_a)
        s = torch.einsum("hmd,hnd->hmn", qs, kks) * sm_scale
        causal = qpos[:, None] >= ks[None, :]
        s = torch.where(causal[None], s, torch.full_like(s, NEG))
        mmax = s.max(-1).values
        keep = (s >= (mmax[..., None] - tau[:, None, None])) & causal[None]
        votes = keep.sum(1)                                                   # (H,end)
        w = torch.where(keep, torch.exp(s - mmax[..., None]), torch.zeros_like(s))
        wv = w.sum(1)                                                         # (H,end)

        # ── chunk aggregation over the FULL key axis (chunk grid is absolute, not per-tile) ──
        nC = (N + C - 1) // C
        rv = votes.float() if select_score == "freq" else wv
        rv = torch.where(votes > 0, rv, torch.zeros_like(rv))
        full = torch.zeros((H, nC * C), device=dev, dtype=torch.float32)
        full[:, :end] = rv
        mass_c = full.view(H, nC, C).sum(-1)                                  # (H,nC)
        if chunk_agg == "mean":
            cf = torch.zeros((H, nC * C), device=dev)
            cf[:, :end] = (votes > 0).float()
            rank_c = mass_c / cf.view(H, nC, C).sum(-1).clamp(min=1)
        else:
            rank_c = mass_c

        last_q = r1 - 1
        c_idx = torch.arange(nC, device=dev)
        vis = c_idx * C <= last_q
        glob_c = c_idx < (n_global + C - 1) // C
        lo, hi = max(0, r0 - n_local + 1) // C, last_q // C
        loc_c = (c_idx >= lo) & (c_idx <= hi)
        forced_c = (glob_c | loc_c) & vis

        rk = torch.where(forced_c[None] | ~vis[None], torch.full_like(rank_c, NEG), rank_c)
        n_take = min(n_chunk, nC)
        tv, ti = rk.topk(n_take, dim=-1)
        sel_c = torch.zeros((H, nC), dtype=torch.bool, device=dev)
        sel_c.scatter_(-1, ti, tv > NEG)
        sel_c |= forced_c[None]

        kept_full = sel_c.unsqueeze(-1).expand(H, nC, C).reshape(H, nC * C)
        kept = kept_full[:, :end] & (ks <= last_q)[None]                      # causal-edge trim

        rank0 = (kept.int().cumsum(-1) - 1).float()
        krc = _rope_rotate(kf[:, :end], rank0, invfc, 1.0)
        npq = torch.gather(rank0, -1, qpos[None].expand(H, r1 - r0))
        qrc = _rope_rotate(qf[:, r0:r1], npq, invfc, 1.0)
        sc = torch.einsum("hmd,hnd->hmn", qrc, krc) * sm_scale
        sel_q = kept[:, None, :].expand(H, r1 - r0, end) & causal[None]        # POOLED
        sc = torch.where(sel_q, sc, torch.full_like(sc, NEG))
        a = torch.softmax(sc, dim=-1)
        out[:, r0:r1] = torch.einsum("hmn,hnd->hmd", a, vf[:, :end])
    return out.unsqueeze(0).to(q.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# DECODE -- one query, chunk selection over the whole cache.
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _decode_build_sets_chunk(q, K, tau, inv_freq_a, q_pos, pos_scale_a, rope_scale_a, sm_scale,
                             n_global, n_local, k_max, chunk_size, n_chunk, scores=None,
                             select_score="expmass", chunk_agg="sum", measure=False):
    """Chunk selection for a single query. q:(H,D), K:(H,M,D), M=q_pos+1.
    Returns idx (H,k_max) long (sentinel=M), newpos_k (H,k_max) fp32, newpos_q (H,) fp32."""
    H, M, D = K.shape
    dev = K.device
    C = chunk_size
    nC = (M + C - 1) // C
    NEG = torch.finfo(torch.float32).min
    inv_freq_a = inv_freq_a.to(dev).float()
    tau = tau.to(dev).float()

    if scores is None:
        pos = torch.arange(M, device=dev)
        qs = _rope_rotate(q, torch.tensor(float(q_pos) * pos_scale_a, device=dev),
                          inv_freq_a, rope_scale_a)
        Ks = _rope_rotate(K, pos.float() * pos_scale_a, inv_freq_a, rope_scale_a)
        score = torch.einsum("hd,hmd->hm", qs, Ks) * sm_scale
    else:
        score = scores.to(dev).float()

    m = score.max(dim=-1).values
    keep = score >= (m - tau)[:, None]                                        # (H,M)
    # For a SINGLE query expmass = exp(s-m) is strictly monotone in s, so ranking chunks by summed
    # exp(s-m) is the decode analogue of the prefill expmass aggregation.
    rv = keep.float() if select_score == "freq" else torch.where(
        keep, torch.exp(score - m[:, None]), torch.zeros_like(score))

    pad = nC * C - M
    rvp = torch.nn.functional.pad(rv, (0, pad)) if pad else rv
    mass_c = rvp.view(H, nC, C).sum(-1)
    if chunk_agg == "mean":
        kp = torch.nn.functional.pad(keep.float(), (0, pad)) if pad else keep.float()
        rank_c = mass_c / kp.view(H, nC, C).sum(-1).clamp(min=1)
    else:
        rank_c = mass_c

    c_idx = torch.arange(nC, device=dev)
    glob_c = c_idx < (n_global + C - 1) // C
    lo, hi = max(0, q_pos - n_local + 1) // C, q_pos // C
    forced_c = glob_c | ((c_idx >= lo) & (c_idx <= hi))
    forced_c = forced_c & (c_idx * C <= q_pos)

    rk = torch.where(forced_c[None] | (c_idx * C > q_pos)[None],
                     torch.full_like(rank_c, NEG), rank_c)
    n_take = min(n_chunk, nC)
    tv, ti = rk.topk(n_take, dim=-1)
    sel_c = torch.zeros((H, nC), dtype=torch.bool, device=dev)
    sel_c.scatter_(-1, ti, tv > NEG)
    sel_c |= forced_c[None]

    kj = torch.arange(M, device=dev)
    kept = sel_c.unsqueeze(-1).expand(H, nC, C).reshape(H, nC * C)[:, :M] & (kj <= q_pos)[None]
    kept[:, q_pos] = True                       # a query must always be able to attend itself

    if measure:
        _stat_acc((keep & kept).sum(dtype=torch.float64), keep.sum(dtype=torch.float64),
                  kept.sum(dtype=torch.float64), H, "decode")

    rank0 = (kept.int().cumsum(-1) - 1).float()
    slot = torch.where(kept & (rank0 < k_max), rank0.long(),
                       torch.full_like(rank0, k_max, dtype=torch.long))
    idx = torch.full((H, k_max + 1), M, device=dev, dtype=torch.long)
    idx.scatter_(-1, slot, kj[None].expand(H, M))
    idx = idx[..., :k_max]

    idx_clamp = idx.clamp(max=M - 1)
    newpos_k = torch.gather(rank0, -1, idx_clamp)
    newpos_k = torch.where(idx < M, newpos_k, torch.zeros_like(newpos_k))
    return idx, newpos_k.float(), rank0[:, q_pos].float()


# ──────────────────────────────────────────────────────────────────────────────
# INSTALL
# ──────────────────────────────────────────────────────────────────────────────
def install_shared_pi_forward(model, pct=30.0, context_window=4096, n_global=32, n_local=32,
                              tile=32, k_max=4096, block_n=32, block_k=32, max_len=None,
                              # Triton is the default now that verify_chunk_decode.py has checked the
                              # integration end to end: the chunk-built (idx, newpos_k, newpos_q)
                              # handed to the SHARED perquery attend kernels agrees with the torch
                              # path across padding / boundary / ragged / full-budget shapes (D2),
                              # matches dense attention when tau=-inf (D3), and produces token-for-
                              # token identical greedy output on real weights (D5). run_ruler.py
                              # already passed --decode_backend triton explicitly; this just stops
                              # the module default from disagreeing with the verified path.
                              pct_decode=100.0, decode_backend="triton", select_backend="triton",
                              prefill_mode="shared", measure=False, select_score="expmass",
                              select_geom="pi", apply_mscale=True,
                              chunk_size=16, n_chunk=None, chunk_agg="sum"):
    """Patch LlamaAttention.forward to route through the CHUNKED kernels.

    chunk_size : keys per chunk (default 16 -- the finest measured, and the cheapest in forced slots)
    n_chunk    : chunks to SELECT. None -> derived so forced+selected fills k_max exactly.
    chunk_agg  : "sum" (default, best at every measured point) | "mean"
    select_geom / apply_mscale / select_score: identical meaning to the pq kernel.
    pct / pct_decode accepted for API compatibility and IGNORED (the budget is n_chunk, not a pct).
    """
    import transformers.models.llama.modeling_llama as ml
    from shared_selection_triton_perquery import (_decode_attend_torch_pq,
                                                  _decode_attend_kernelC_pq)
    assert select_geom in ("pi", "yarn"), f"select_geom must be pi|yarn, got {select_geom!r}"
    assert chunk_agg in ("sum", "mean"), f"chunk_agg must be sum|mean, got {chunk_agg!r}"
    if n_chunk is None:
        n_chunk = derive_n_chunk(k_max, chunk_size, n_global, n_local, tile)

    rotary = model.model.rotary_emb
    inv_freq_rot = rotary.inv_freq.detach().float()
    attn_scaling = float(getattr(rotary, "attention_scaling", 1.0))

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    theta = 10000.0
    _rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None)
    if isinstance(_rp, dict) and _rp.get("rope_theta"):
        theta = float(_rp["rope_theta"])
    elif getattr(cfg, "rope_theta", None):
        theta = float(cfg.rope_theta)
    inv_freq_base = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    if select_geom == "yarn":
        if attn_scaling == 1.0:
            print("WARNING [chunk kernel]: rotary.attention_scaling == 1.0 — the model does NOT look "
                  "yarn-loaded (mscale should be >1).", flush=True)
        inv_freq_a = inv_freq_rot
        rope_scale_a = attn_scaling if apply_mscale else 1.0
    else:
        inv_freq_a = inv_freq_base
        rope_scale_a = 1.0

    nf = n_forced_chunks(chunk_size, n_global, n_local, tile)
    print(f"chunk kernel: A={select_geom}(rope_scale_A={rope_scale_a:.4f}), "
          f"C=base RoPE(theta={theta:.0f}) at DENSE-RANK positions; "
          f"chunk_size={chunk_size} n_chunk={n_chunk} agg={chunk_agg} rank={select_score}; "
          f"budget=({nf} forced + {n_chunk}) x {chunk_size} = {(nf+n_chunk)*chunk_size} "
          f"<= k_max={k_max}  [EXACT, no overflow path]", flush=True)

    for li, layer in enumerate(model.model.layers):
        sa = layer.self_attn
        dev = sa.q_proj.weight.device
        sa.spi_inv_freq_a = inv_freq_a.to(dev)
        sa.spi_rope_scale_a = rope_scale_a
        sa.spi_inv_freq_c = inv_freq_base.to(dev)
        sa.spi_geom = select_geom
        sa.spi_W = context_window
        sa.spi_ng = n_global; sa.spi_nl = n_local
        sa.spi_tile = tile; sa.spi_kmax = k_max
        sa.spi_block_n = block_n; sa.spi_block_k = block_k
        sa.spi_layer_idx = getattr(sa, "layer_idx", li)
        sa.spi_max_len = max_len
        sa.spi_pos_scale_a = None
        sa.spi_select_score = select_score
        sa.spi_decode_backend = decode_backend
        sa.spi_select_backend = select_backend
        sa.spi_prefill_mode = prefill_mode
        sa.spi_chunk_size = chunk_size
        sa.spi_n_chunk = n_chunk
        sa.spi_chunk_agg = chunk_agg
        sa.spi_measure = measure

    def spi_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, **kwargs):
        B, N, _ = hidden_states.shape
        hs = (B, N, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hs).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hs).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hs).transpose(1, 2)

        cache = past_key_values if past_key_values is not None else kwargs.get("past_key_value", None)
        past_len = 0
        if cache is not None:
            try:
                past_len = int(cache.get_seq_length(self.spi_layer_idx))
            except TypeError:
                past_len = int(cache.get_seq_length())
            k, v = cache.update(k, v, self.spi_layer_idx)

        if repeat_kv is not None:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        if past_len == 0:
            if self.spi_geom == "yarn":
                self.spi_pos_scale_a = 1.0
            else:
                L_target = self.spi_max_len if self.spi_max_len is not None else N
                self.spi_pos_scale_a = min(1.0, self.spi_W / float(L_target))

        if past_len == 0 and getattr(self, "spi_prefill_mode", "shared") == "shared":
            out, usize, kept_size = shared_chunk_attention(
                q, k, v, self.tau_head_vec, self.spi_inv_freq_a, self.spi_inv_freq_c,
                pos_scale_a=self.spi_pos_scale_a, rope_scale_a=self.spi_rope_scale_a,
                sm_scale=self.scaling, n_global=self.spi_ng, n_local=self.spi_nl,
                tile=self.spi_tile, k_max=self.spi_kmax, chunk_size=self.spi_chunk_size,
                n_chunk=self.spi_n_chunk, block_n=self.spi_block_n, block_k=self.spi_block_k,
                q_pos_base=past_len, select_score=self.spi_select_score,
                chunk_agg=self.spi_chunk_agg, measure=self.spi_measure)
            self.spi_last_usize = usize.detach()
            self.spi_last_kept = kept_size.detach()
        else:
            select_backend = getattr(self, "spi_select_backend", "torch")
            attend_backend = getattr(self, "spi_decode_backend", "torch")
            Kf, Vf, qf = k[0], v[0], q[0]
            outs = []
            for i in range(N):
                P = past_len + i
                Ki, Vi, qi = Kf[:, :P + 1, :], Vf[:, :P + 1, :], qf[:, i, :]
                scores = None
                if select_backend == "triton":
                    scores = _decode_scores_kernel(qi, Ki, self.spi_inv_freq_a, P,
                                                   self.spi_pos_scale_a, self.spi_rope_scale_a,
                                                   self.scaling, block_n=self.spi_block_n)
                idx, npk, npq = _decode_build_sets_chunk(
                    qi, Ki, self.tau_head_vec, self.spi_inv_freq_a, P, self.spi_pos_scale_a,
                    self.spi_rope_scale_a, self.scaling, self.spi_ng, self.spi_nl,
                    self.spi_kmax, self.spi_chunk_size, self.spi_n_chunk, scores=scores,
                    select_score=self.spi_select_score, chunk_agg=self.spi_chunk_agg,
                    measure=self.spi_measure)
                if attend_backend == "triton":
                    oh = _decode_attend_kernelC_pq(qi, Ki, Vi, idx, npk, npq, self.spi_inv_freq_c,
                                                   self.scaling, self.spi_kmax, q_pos=P,
                                                   block_k=self.spi_block_k)
                else:
                    oh = _decode_attend_torch_pq(qi, Ki, Vi, idx, npk, npq, self.spi_inv_freq_c,
                                                 self.scaling)
                outs.append(oh)
            out = torch.stack(outs, dim=1).unsqueeze(0)

        attn_output = out.transpose(1, 2).reshape(B, N, -1).contiguous()
        return self.o_proj(attn_output), None

    ml.LlamaAttention.forward = spi_forward
