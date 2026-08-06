#!/usr/bin/env python3
"""
shared_selection_triton_perquery.py

PQ-UNION kernel ("pq_exact"): PER-QUERY key sets on tensor cores, via union superset + dense-rank
positions + per-query bitmask. Fork of shared_selection_triton_exact_2.py.

Design (frozen spec):
  * SELECTION (Kernel A): exact per-query max thresholding, unchanged math. Geometry is host-chosen
    (select_geom): "pi" scores at compressed positions (base inv_freq, pos_scale=W/N), "yarn" scores
    at raw positions (YaRN inv_freq + mscale). NEW output: packed per-query keep bits — with tile=32
    the 32 queries' bits for key j fit ONE int32:  Bits[h,t,j] = sum_i keep(i,j) << i.
  * SET (host _build_sets_pq): the FULL UNION of the tile's per-query keeps (votes>0), plus forced
    global (first n_global) + local (last n_local) for every query. NO top-pct cut (pct is accepted
    for API compat and ignored). k_max default 4096; if |union|+forced > k_max, keep the top-k_max
    ranked by expmass (truncation of last resort, logged via `overflow`).
  * POSITIONS: dense consecutive ranks — kept keys in sequence order get 0,1,2,... (= cumsum-1 of the
    kept mask; equivalently base_position advancing by the union count per tile). Monotone, collision-
    free, and bounded by |kept| <= k_max << 4096, so KERNEL C ALWAYS RUNS NATIVE RoPE IN-WINDOW:
    base inv_freq, rope_scale=1, no PI compression, no YaRN, no mscale — at ANY context length.
  * ATTENTION (Kernel C): identical gather + RoPE(rank) + online-softmax GEMM (tensor cores intact),
    plus a per-query bit test folded into the keep condition:
        keep(row i, slot s) = causal & valid & ((Mask[s] >> i) & 1)
    Mask = Bits gathered at idx, with forced global/local columns set to -1 (all 32 queries attend).
    So query i attends exactly (its own threshold passers ∩ kept) ∪ global ∪ local — a true per-query
    set — while the QK matmul stays one shared (32 x d)(d x k_max) GEMM.

  * DECODE: one query at a time -> the "union" IS that query's own kept set, so no bitmask is
    needed (mask = all-ones over its slots) and the dense ranks are already per-query dense. Same
    two-geometry split; k_max cap by score (for a single query exp(s-m) is monotone in s, so
    score-ranking == the expmass ranking used at prefill).

Batch size 1 (q,k,v are (1,H,N,D)). Prefill + decode.
"""
import math
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
except Exception:
    repeat_kv = None


# ──────────────────────────────────────────────────────────────────────────────
# MEASUREMENT-ONLY overflow counters (do NOT affect kernel execution).
# Accumulated inside _build_sets_pq as lazy on-device tensors (no host sync during the
# run); get_overflow_stats() syncs once when read. Counts (head,tile) cells.
# ──────────────────────────────────────────────────────────────────────────────
def _fresh():
    return {"cells": 0, "overflow": None, "union_sum": None, "union_max": None}


# tracked separately: a PREFILL cell is a (head, 32-query tile) — the "did this tile stay pure
# per-query, or get clipped to top-k_max" flag. A DECODE cell is a (head, single query). Mixed
# together the decode cells would swamp the tile metric during generation, so they are kept apart.
_OVF_STATS = {"prefill": _fresh(), "decode": _fresh()}


def reset_overflow_stats():
    """Zero the cumulative overflow/union counters (e.g. between benchmark runs)."""
    _OVF_STATS["prefill"] = _fresh()
    _OVF_STATS["decode"] = _fresh()


def _phase_stats(d):
    tot = d["cells"]
    ovf = int(d["overflow"].item()) if d["overflow"] is not None else 0
    usum = int(d["union_sum"].item()) if d["union_sum"] is not None else 0
    umax = int(d["union_max"].item()) if d["union_max"] is not None else 0
    return {
        "total_cells": tot,                                   # selection cells processed
        "overflow_cells": ovf,                                # cells where |union∪forced| > k_max
        "overflow_frac": (ovf / tot) if tot else 0.0,         # the frequency you care about
        "union_mean": (usum / tot) if tot else 0.0,           # avg union size (pre-forced)
        "union_max": umax,                                    # largest union seen
    }


def get_overflow_stats():
    """Cumulative since last reset, per phase: how often the full union exceeded k_max (-> the
    expmass top-k_max clip), plus union-size stats. NOTE: overflow does NOT revert to pooled
    shared selection — each query still applies its own mask against the clipped candidate pool;
    it only means some queries lost their lowest-mass keys. Read `union_mean`/`union_max` next to
    `overflow_frac` to judge severity. One host sync here, none during the run."""
    return {"prefill": _phase_stats(_OVF_STATS["prefill"]),
            "decode": _phase_stats(_OVF_STATS["decode"])}


def _ovf_accumulate(overflow, usize, phase="prefill"):
    """Measurement only. overflow (H,n) bool, usize (H,n) int32 — accumulated on device."""
    d = _OVF_STATS[phase]
    d["cells"] += overflow.numel()
    ov = overflow.sum()
    us = usize.long().sum()
    um = usize.max()
    if d["overflow"] is None:
        d["overflow"] = ov
        d["union_sum"] = us
        d["union_max"] = um
    else:
        d["overflow"] = d["overflow"] + ov
        d["union_sum"] = d["union_sum"] + us
        d["union_max"] = torch.maximum(d["union_max"], um)


if HAVE_TRITON:

    # ══════════════════════════════════════════════════════════════════════════
    # KERNEL A : selection + voting + PER-QUERY BITS  (one program per (tile, head))
    # EXACT MAX, 2 passes. Identical to exact_2 except the packed `Bits` output.
    # ══════════════════════════════════════════════════════════════════════════
    @triton.jit
    def _select_bits_kernel(
        Q, K, Tau, InvFreq, Votes, RowCounts, WVotes, Bits,
        sm_scale, pos_scale, rope_scale, q_pos_base,
        stride_qh, stride_qn, stride_qd,
        stride_kh, stride_kn, stride_kd,
        stride_vh, stride_vt, stride_vn,
        stride_rh, stride_rn,
        stride_wh, stride_wt, stride_wn,
        stride_bh, stride_bt, stride_bn,
        H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr, HALF_DIM: tl.constexpr, WMODE: tl.constexpr,
    ):
        start_m = tl.program_id(0)          # tile index
        off_h   = tl.program_id(1)          # head (batch=1)
        q_base = Q + off_h * stride_qh
        k_base = K + off_h * stride_kh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)      # row index into Q
        qpos   = q_pos_base + offs_m                            # ABSOLUTE query position
        offs_h = tl.arange(0, HALF_DIM)
        invf = tl.load(InvFreq + offs_h)
        tau  = tl.load(Tau + off_h)

        # per-row bit values for packing: row i -> 1<<i (row 31 wraps to int32 sign bit — exact packing)
        rows = tl.arange(0, BLOCK_M)
        pw = tl.full([BLOCK_M], 1, tl.int32) << rows            # (BLOCK_M,) int32

        # raw Q halves -> rotate at select-geometry positions (pos * pos_scale)
        q1 = tl.load(q_base + offs_m[:, None] * stride_qn + offs_h[None, :] * stride_qd,
                     mask=offs_m[:, None] < N_CTX, other=0.0)
        q2 = tl.load(q_base + offs_m[:, None] * stride_qn + (HALF_DIM + offs_h)[None, :] * stride_qd,
                     mask=offs_m[:, None] < N_CTX, other=0.0)
        angq = (qpos[:, None].to(tl.float32) * pos_scale) * invf[None, :]
        cosq = tl.cos(angq) * rope_scale
        sinq = tl.sin(angq) * rope_scale
        qr1 = q1 * cosq - q2 * sinq
        qr2 = q2 * cosq + q1 * sinq

        hi = q_pos_base + (start_m + 1) * BLOCK_M

        # ── pass 1: EXACT per-query max over all causal keys ──
        m = tl.full([BLOCK_M], float("-inf"), tl.float32)
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k1 = tl.load(k_base + offs_n[:, None] * stride_kn + offs_h[None, :] * stride_kd,
                         mask=offs_n[:, None] < N_CTX, other=0.0)
            k2 = tl.load(k_base + offs_n[:, None] * stride_kn + (HALF_DIM + offs_h)[None, :] * stride_kd,
                         mask=offs_n[:, None] < N_CTX, other=0.0)
            angk = (offs_n[:, None].to(tl.float32) * pos_scale) * invf[None, :]
            cosk = tl.cos(angk) * rope_scale
            sink = tl.sin(angk) * rope_scale
            kr1 = k1 * cosk - k2 * sink
            kr2 = k2 * cosk + k1 * sink
            qk = (tl.dot(qr1, tl.trans(kr1)) + tl.dot(qr2, tl.trans(kr2))) * sm_scale
            causal = qpos[:, None] >= offs_n[None, :]
            qk = tl.where(causal, qk, float("-inf"))
            m = tl.maximum(m, tl.max(qk, 1))

        # ── pass 2: keep + vote + PACKED PER-QUERY BITS against the EXACT max ──
        row_counts = tl.zeros([BLOCK_M], tl.int32)
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k1 = tl.load(k_base + offs_n[:, None] * stride_kn + offs_h[None, :] * stride_kd,
                         mask=offs_n[:, None] < N_CTX, other=0.0)
            k2 = tl.load(k_base + offs_n[:, None] * stride_kn + (HALF_DIM + offs_h)[None, :] * stride_kd,
                         mask=offs_n[:, None] < N_CTX, other=0.0)
            angk = (offs_n[:, None].to(tl.float32) * pos_scale) * invf[None, :]
            cosk = tl.cos(angk) * rope_scale
            sink = tl.sin(angk) * rope_scale
            kr1 = k1 * cosk - k2 * sink
            kr2 = k2 * cosk + k1 * sink
            qk = (tl.dot(qr1, tl.trans(kr1)) + tl.dot(qr2, tl.trans(kr2))) * sm_scale
            causal = qpos[:, None] >= offs_n[None, :]
            # exclude padded query rows (ragged last tile when N % BLOCK_M != 0) from votes/bits
            keep = (qk >= (m[:, None] - tau)) & causal & (offs_m[:, None] < N_CTX)
            row_counts += tl.sum(keep.to(tl.int32), 1)           # per-query kept count
            votes = tl.sum(keep.to(tl.int32), 0)                 # (BLOCK_N,)  frequency
            tl.store(Votes + off_h * stride_vh + start_m * stride_vt + offs_n * stride_vn,
                     votes, mask=offs_n < N_CTX)
            # packed per-query bits: bit i of column j = keep(query i, key j)
            bits = tl.sum(tl.where(keep, pw[:, None], 0), 0)     # (BLOCK_N,) int32
            tl.store(Bits + off_h * stride_bh + start_m * stride_bt + offs_n * stride_bn,
                     bits, mask=offs_n < N_CTX)
            # WMODE>0: weighted score per key over the tile's queries (ranking for overflow truncation)
            #   1 = margin   delta = s - (m - tau)  in [0, tau]
            #   2 = expmass  exp(s - m)             in (0, 1]  (max-normalized attention weight)
            if WMODE == 1:
                contr = tl.where(keep, qk - (m[:, None] - tau), 0.0)
                wv = tl.sum(contr, 0)                            # (BLOCK_N,) fp32
                tl.store(WVotes + off_h * stride_wh + start_m * stride_wt + offs_n * stride_wn,
                         wv, mask=offs_n < N_CTX)
            elif WMODE == 2:
                contr = tl.where(keep, tl.exp(qk - m[:, None]), 0.0)
                wv = tl.sum(contr, 0)                            # (BLOCK_N,) fp32
                tl.store(WVotes + off_h * stride_wh + start_m * stride_wt + offs_n * stride_wn,
                         wv, mask=offs_n < N_CTX)
        tl.store(RowCounts + off_h * stride_rh + offs_m * stride_rn,
                 row_counts, mask=offs_m < N_CTX)


    # ══════════════════════════════════════════════════════════════════════════
    # KERNEL C : gather + rank-positioned attention + PER-QUERY BITMASK
    # (one program per (tile, head)). GEMM identical to exact_2 — tensor cores intact;
    # the mask is one extra int32 load per key slot + a bit test folded into `keep`.
    # ══════════════════════════════════════════════════════════════════════════
    @triton.jit
    def _gather_attn_pq_kernel(
        Q, Kc, Vc, InvFreq, Idx, NewposK, NewposQ, Mask, Out,
        sm_scale, rope_scale, q_pos_base,
        stride_qh, stride_qn, stride_qd,
        stride_kh, stride_kn, stride_kd,
        stride_vh, stride_vn, stride_vd,
        stride_ih, stride_it, stride_ik,
        stride_nkh, stride_nkt, stride_nkk,
        stride_nqh, stride_nqt, stride_nqm,
        stride_mh, stride_mt, stride_mk,
        stride_oh, stride_on, stride_od,
        H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, K_MAX: tl.constexpr,
        HEAD_DIM: tl.constexpr, HALF_DIM: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_h   = tl.program_id(1)
        q_base = Q  + off_h * stride_qh
        k_base = Kc + off_h * stride_kh
        v_base = Vc + off_h * stride_vh
        o_base = Out + off_h * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)      # row index into Q / Out
        qpos   = q_pos_base + offs_m                            # ABSOLUTE query position (for causal)
        offs_h = tl.arange(0, HALF_DIM)
        offs_d = tl.arange(0, HEAD_DIM)
        invf = tl.load(InvFreq + offs_h)
        rows = tl.arange(0, BLOCK_M)                            # tile-local query row (bit index)

        # raw Q -> rotate at NEWPOS (dense rank; one position per query)
        npq = tl.load(NewposQ + off_h * stride_nqh + start_m * stride_nqt
                      + tl.arange(0, BLOCK_M) * stride_nqm)               # (BLOCK_M,)
        q1 = tl.load(q_base + offs_m[:, None] * stride_qn + offs_h[None, :] * stride_qd,
                     mask=offs_m[:, None] < N_CTX, other=0.0)
        q2 = tl.load(q_base + offs_m[:, None] * stride_qn + (HALF_DIM + offs_h)[None, :] * stride_qd,
                     mask=offs_m[:, None] < N_CTX, other=0.0)
        angq = npq[:, None] * invf[None, :]
        cosq = tl.cos(angq) * rope_scale
        sinq = tl.sin(angq) * rope_scale
        qr1 = q1 * cosq - q2 * sinq
        qr2 = q2 * cosq + q1 * sinq

        mR  = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

        for start_k in range(0, K_MAX, BLOCK_K):
            slots = start_k + tl.arange(0, BLOCK_K)
            idx = tl.load(Idx + off_h * stride_ih + start_m * stride_it + slots * stride_ik,
                          mask=slots < K_MAX, other=N_CTX)               # padded with sentinel N_CTX
            valid = idx < N_CTX
            npk = tl.load(NewposK + off_h * stride_nkh + start_m * stride_nkt + slots * stride_nkk,
                          mask=valid, other=0.0)
            # per-query bitmask for these slots (int32; bit i = query row i attends this key)
            mval = tl.load(Mask + off_h * stride_mh + start_m * stride_mt + slots * stride_mk,
                           mask=slots < K_MAX, other=0)
            # gather k of the kept keys
            k1 = tl.load(k_base + idx[:, None] * stride_kn + offs_h[None, :] * stride_kd,
                         mask=valid[:, None], other=0.0)
            k2 = tl.load(k_base + idx[:, None] * stride_kn + (HALF_DIM + offs_h)[None, :] * stride_kd,
                         mask=valid[:, None], other=0.0)
            angk = npk[:, None] * invf[None, :]
            cosk = tl.cos(angk) * rope_scale
            sink = tl.sin(angk) * rope_scale
            kr1 = k1 * cosk - k2 * sink
            kr2 = k2 * cosk + k1 * sink

            qk = (tl.dot(qr1, tl.trans(kr1)) + tl.dot(qr2, tl.trans(kr2))) * sm_scale  # (BLOCK_M, BLOCK_K)
            # causal (orig pos) + not padding + PER-QUERY bit test (arith >> then &1 extracts bit i)
            bit = (mval[None, :] >> rows[:, None]) & 1
            keep = (qpos[:, None] >= idx[None, :]) & valid[None, :] & (bit > 0)
            qk = tl.where(keep, qk, float("-inf"))

            mR_new = tl.maximum(mR, tl.max(qk, 1))
            mR_safe = tl.where(mR_new == float("-inf"), 0.0, mR_new)
            p = tl.exp(qk - mR_safe[:, None])
            alpha = tl.exp(mR - mR_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            vv = tl.load(v_base + idx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                         mask=valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)
            mR = mR_new

        acc = acc / l_i[:, None]
        tl.store(o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
                 acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


    # ══════════════════════════════════════════════════════════════════════════
    # DECODE-SELECT KERNEL : single query -> per-key scores (one program per (key-block, head)).
    # Counterpart to Kernel A for decode, under the SELECT geometry (inv_freq_a/pos_scale_a/
    # rope_scale_a). Host takes the exact max + threshold + cap + dense ranks. Identical to the
    # exact_2 decode-select kernel (no bits needed: one query -> no per-query packing).
    # ══════════════════════════════════════════════════════════════════════════
    @triton.jit
    def _decode_select_kernel(
        Q, K, InvFreq, Scores,
        sm_scale, pos_scale, rope_scale, q_pos,
        stride_qh, stride_qd,
        stride_kh, stride_kn, stride_kd,
        stride_sh, stride_sn,
        H, M,
        BLOCK_N: tl.constexpr, HALF_DIM: tl.constexpr,
    ):
        kb = tl.program_id(0)               # key-block index
        off_h = tl.program_id(1)            # head
        offs_n = kb * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_h = tl.arange(0, HALF_DIM)
        invf = tl.load(InvFreq + offs_h)

        q1 = tl.load(Q + off_h * stride_qh + offs_h * stride_qd)
        q2 = tl.load(Q + off_h * stride_qh + (HALF_DIM + offs_h) * stride_qd)
        angq = (q_pos * pos_scale) * invf
        cosq = tl.cos(angq) * rope_scale
        sinq = tl.sin(angq) * rope_scale
        qr1 = q1 * cosq - q2 * sinq
        qr2 = q2 * cosq + q1 * sinq

        k1 = tl.load(K + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_h[None, :] * stride_kd,
                     mask=offs_n[:, None] < M, other=0.0)
        k2 = tl.load(K + off_h * stride_kh + offs_n[:, None] * stride_kn + (HALF_DIM + offs_h)[None, :] * stride_kd,
                     mask=offs_n[:, None] < M, other=0.0)
        angk = (offs_n[:, None].to(tl.float32) * pos_scale) * invf[None, :]
        cosk = tl.cos(angk) * rope_scale
        sink = tl.sin(angk) * rope_scale
        kr1 = k1 * cosk - k2 * sink
        kr2 = k2 * cosk + k1 * sink

        score = tl.sum(qr1[None, :] * kr1 + qr2[None, :] * kr2, axis=1) * sm_scale   # (BLOCK_N,)
        tl.store(Scores + off_h * stride_sh + offs_n * stride_sn, score, mask=offs_n < M)


# ──────────────────────────────────────────────────────────────────────────────
# HOST : votes/bits -> FULL UNION (+forced global/local) -> dense-rank positions
#        + per-query slot bitmask. k_max overflow -> top-k_max by expmass rank.
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _build_sets_pq(votes, bits, wvotes, n_global, n_local, T, k_max, q_pos_base=0,
                   select_score="expmass"):
    """votes (H,nt,N) int, bits (H,nt,N) int32 (bit i of [h,t,j] = keep(query i, key j)),
    wvotes (H,nt,N) fp32 (or dummy). Returns:
      idx      (H,nt,k_max) int32  kept-key positions ascending, sentinel=N
      newpos_k (H,nt,k_max) fp32   dense rank of each kept key (0,1,2,... over the kept set)
      newpos_q (H,nt,T)     fp32   dense rank of each query's own position
      mask     (H,nt,k_max) int32  per-query bits per slot; forced cols = -1 (all queries)
      usize    (H,nt) int32        true union size (votes>0, pre-forced)
      kept_size(H,nt) int32        |union ∪ forced| after any truncation
      overflow (H,nt) bool         tile needed the top-k_max expmass truncation
    Positions are ranks over the FINAL kept set -> monotone, collision-free, max < k_max."""
    H, nt, N = votes.shape
    dev = votes.device
    NEG = float("-inf")
    kj = torch.arange(N, device=dev)
    glob = kj < n_global                                                 # (N,)
    tstart = q_pos_base + torch.arange(nt, device=dev) * T               # absolute tile-start positions
    lo = tstart - n_local + 1
    hi = tstart + T - 1
    local_block = (kj[None, :] >= lo[:, None]) & (kj[None, :] <= hi[:, None])   # (nt,N)
    forced_tn = glob[None, :] | local_block                              # (nt,N) forced for EVERY query

    union = votes > 0                                                    # (H,nt,N) any query kept it
    usize = union.sum(-1, dtype=torch.int32)                             # (H,nt)
    kept_full = union | forced_tn[None]
    overflow = kept_full.sum(-1) > k_max                                 # (H,nt)
    _ovf_accumulate(overflow, usize)                                     # measurement only (no effect)

    # overflow truncation: keep forced + top-(k_max - n_forced) non-forced union keys by rank score.
    # Exact budget via topk+scatter (no tie overshoot -> position-sort truncation can never drop
    # the forced local block / the queries themselves).
    if select_score == "freq" or wvotes is None or wvotes.shape[-1] != N:
        rank_val = votes.float()
    else:
        rank_val = wvotes.float() + 1e-6 * votes.float()                 # freq as deterministic tiebreak
    cand = union & ~forced_tn[None]
    rv = torch.where(cand, rank_val, torch.full_like(rank_val, NEG))
    n_forced = forced_tn.sum(-1)                                         # (nt,)
    budget = (k_max - n_forced).clamp(min=0)                             # (nt,)
    Ktop = int(min(k_max, N))
    tv, ti = rv.topk(Ktop, dim=-1)                                       # (H,nt,Ktop)
    ok = (torch.arange(Ktop, device=dev)[None, None, :] < budget[None, :, None]) & (tv > NEG)
    sel_trunc = torch.zeros_like(union)
    sel_trunc.scatter_(-1, ti, ok)
    kept = torch.where(overflow.unsqueeze(-1), sel_trunc | forced_tn[None], kept_full)
    kept_size = kept.sum(-1, dtype=torch.int32)

    # dense-rank positions over the FINAL kept set (0,1,2,...): rank = (# kept keys before it).
    # This IS the base-position scheme: bp advances by exactly the tile's kept count.
    rank0 = (kept.int().cumsum(-1) - 1).float()                          # (H,nt,N)

    # Compaction by SCATTER, not sort. rank0 already holds each kept key's destination slot, so this
    # is O(N) instead of O(N log N) -- and, more importantly, it never materialises an (H,nt,N) int64
    # tensor. The old `sort(where(kept, kj, N))` needed three of them live at once (masked_pos, plus
    # sort's values AND indices): ~14.5 GB at N=24576 vs ~0.8 GB for the buffer below.
    # Non-kept keys all scatter into the dump slot k_max, which is then sliced off; they collide, but
    # every writer stores to a slot we discard, so the nondeterminism is harmless. Kept keys have
    # unique ranks (kept_size <= k_max is guaranteed by the truncation above), so they never collide.
    # `kj.expand(...)` stays a stride-0 view -- scatter_ reads it without copying.
    slot = torch.where(kept & (rank0 < k_max), rank0.long(),
                       torch.full_like(rank0, k_max, dtype=torch.long))
    idx_sorted = torch.full((H, nt, k_max + 1), N, device=dev, dtype=torch.long)
    idx_sorted.scatter_(-1, slot, kj[None, None, :].expand(H, nt, N))
    idx_sorted = idx_sorted[..., :k_max]                                 # drop the dump slot
    idx_clamp = idx_sorted.clamp(max=N - 1)
    valid = idx_sorted < N
    newpos_k = torch.gather(rank0, -1, idx_clamp)
    newpos_k = torch.where(valid, newpos_k, torch.zeros_like(newpos_k))

    # per-query slot bitmask: bits gathered at idx; forced columns -> -1 (all 32 bits set);
    # padding slots -> 0 (no query; Kernel C's `valid` also excludes them).
    mask_g = torch.gather(bits, -1, idx_clamp)                           # (H,nt,k_max) int32
    gl = idx_clamp < n_global
    lc = (idx_clamp >= lo[None, :, None]) & (idx_clamp <= hi[None, :, None])
    forced_g = gl | lc
    mask = torch.where(forced_g, torch.full_like(mask_g, -1), mask_g)
    mask = torch.where(valid, mask, torch.zeros_like(mask))

    qpos = q_pos_base + (torch.arange(nt, device=dev)[:, None] * T + torch.arange(T, device=dev)[None, :])
    qpos = qpos.clamp(max=N - 1)            # ragged last tile: padded query slots (unused) stay in-range
    newpos_q = torch.gather(rank0, -1, qpos[None].expand(H, nt, T))      # (H,nt,T)

    return (idx_sorted.to(torch.int32).contiguous(),
            newpos_k.float().contiguous(),
            newpos_q.float().contiguous(),
            mask.to(torch.int32).contiguous(),
            usize.contiguous(),
            kept_size.contiguous(),
            overflow.contiguous())


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration (prefill)
# ──────────────────────────────────────────────────────────────────────────────
def shared_pq_attention(q, k, v, tau, inv_freq_a, inv_freq_c, pos_scale_a=1.0, rope_scale_a=1.0,
                        sm_scale=None, n_global=32, n_local=32, tile=32, k_max=4096,
                        block_n=64, block_k=64, q_pos_base=0, select_score="expmass"):
    """q,k,v: (1,H,N,D) RAW. Two-geometry split:
      Kernel A (select): inv_freq_a at pos*pos_scale_a, cos/sin scaled by rope_scale_a
                          ("pi": base freqs + compression; "yarn": yarn freqs + mscale, pos_scale_a=1)
      Kernel C (attend): inv_freq_c (BASE RoPE) at dense-rank positions, rope_scale=1 — always in-window.
    tile MUST be 32 (per-query bits pack into int32). Returns (out, usize, kept_size, overflow)."""
    assert HAVE_TRITON, "Triton not available"
    Z, H, N, D = q.shape
    assert Z == 1
    assert tile == 32, "pq_exact packs per-query keep bits into int32 -> tile must be 32"
    # The forced set (global ∪ local) is kept unconditionally, so it must FIT in k_max — otherwise
    # budget clamps to 0, kept stays > k_max, and the position-sort silently drops forced keys.
    # forced <= n_global + (n_local + tile - 1); check statically (no host sync).
    assert n_global + n_local + tile - 1 <= k_max, (
        f"forced set (n_global={n_global} + n_local={n_local} + tile-1={tile-1} = "
        f"{n_global + n_local + tile - 1}) exceeds k_max={k_max}: forced keys would be dropped")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    nt = (N + tile - 1) // tile            # ceil: last tile may be ragged
    qh, kh, vh = q[0].contiguous(), k[0].contiguous(), v[0].contiguous()
    tau = tau.float().contiguous()
    invfa = inv_freq_a.float().contiguous()
    invfc = inv_freq_c.float().contiguous()

    votes = torch.zeros((H, nt, N), dtype=torch.int32, device=q.device)
    row_counts = torch.zeros((H, N), dtype=torch.int32, device=q.device)
    bits = torch.zeros((H, nt, N), dtype=torch.int32, device=q.device)
    WMODE = {"freq": 0, "margin": 1, "expmass": 2}[select_score]
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

    idx, newpos_k, newpos_q, mask, usize, kept_size, overflow = _build_sets_pq(
        votes, bits, wvotes, n_global, n_local, tile, k_max,
        q_pos_base=q_pos_base, select_score=select_score)

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
    return out.unsqueeze(0), usize, kept_size, overflow


# ──────────────────────────────────────────────────────────────────────────────
# GOLDEN REFERENCE (pure torch, independent of the Triton path and of _build_sets_pq).
# O(nt * N^2) — use small N. Same rules: exact-max threshold under a-geometry, full union
# + forced, expmass top-k_max truncation, dense ranks, per-query masked softmax under
# c-geometry (base RoPE, rope_scale=1).
# ──────────────────────────────────────────────────────────────────────────────
def _rope_rotate(x, pos, inv_freq, rope_scale):
    """Rotate x (..., D) by RoPE at `pos` (broadcastable to x.shape[:-1]); half-split (Llama) form."""
    half = x.shape[-1] // 2
    ang = pos[..., None].to(torch.float32) * inv_freq.to(torch.float32)      # (..., half)
    cos = torch.cos(ang) * rope_scale
    sin = torch.sin(ang) * rope_scale
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


@torch.no_grad()
def pq_attention_torch(q, k, v, tau, inv_freq_a, inv_freq_c, pos_scale_a=1.0, rope_scale_a=1.0,
                       sm_scale=None, n_global=32, n_local=32, tile=32, k_max=4096,
                       select_score="expmass"):
    """Reference forward. q,k,v (1,H,N,D). Returns (1,H,N,D) in q.dtype."""
    Z, H, N, D = q.shape
    assert Z == 1
    T = tile
    nt = (N + T - 1) // T
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    dev = q.device
    NEG = float("-inf")
    qf, kf, vf = q[0].float(), k[0].float(), v[0].float()
    invfa = inv_freq_a.to(dev).float()
    invfc = inv_freq_c.to(dev).float()
    tau = tau.to(dev).float()
    out = torch.zeros((H, N, D), device=dev, dtype=torch.float32)

    for t in range(nt):
        r0, r1 = t * T, min((t + 1) * T, N)
        m_rows = r1 - r0
        qpos = torch.arange(r0, r1, device=dev)
        end = min((t + 1) * T, N)                       # keys visible to this tile
        ks = torch.arange(end, device=dev)

        # ── selection under a-geometry ──
        qs = _rope_rotate(qf[:, r0:r1], qpos.float() * pos_scale_a, invfa, rope_scale_a)   # (H,m,D)
        kks = _rope_rotate(kf[:, :end], ks.float() * pos_scale_a, invfa, rope_scale_a)     # (H,end,D)
        s = torch.einsum("hmd,hnd->hmn", qs, kks) * sm_scale                               # (H,m,end)
        causal = qpos[:, None] >= ks[None, :]                                              # (m,end)
        s = torch.where(causal[None], s, torch.full_like(s, NEG))
        mmax = s.max(-1).values                                                            # (H,m)
        keep = (s >= (mmax[..., None] - tau[:, None, None])) & causal[None]                # (H,m,end)
        votes = keep.sum(1)                                                                # (H,end)
        w = torch.where(keep, torch.exp(s - mmax[..., None]), torch.zeros_like(s))
        wv = w.sum(1)                                                                      # (H,end)

        # ── full union + forced global/local ──
        forced = (ks < n_global) | ((ks >= r0 - n_local + 1) & (ks <= r0 + T - 1))         # (end,)
        union = votes > 0
        kept = union | forced[None]                                                        # (H,end)

        # ── k_max overflow: forced + top-budget non-forced union keys by rank score ──
        over = kept.sum(-1) > k_max                                                        # (H,)
        if bool(over.any()):
            if select_score == "freq":
                rankv = votes.float()
            else:
                rankv = wv + 1e-6 * votes.float()
            cand = union & ~forced[None]
            rvv = torch.where(cand, rankv, torch.full_like(rankv, NEG))
            budget = max(int(k_max - int(forced.sum())), 0)
            Ktop = max(min(budget, end), 1)
            tv, ti = rvv.topk(Ktop, dim=-1)
            okk = (torch.arange(Ktop, device=dev)[None, :] < budget) & (tv > NEG)
            sel = torch.zeros_like(union)
            sel.scatter_(-1, ti, okk)
            kept = torch.where(over[:, None], sel | forced[None], kept)

        # ── dense ranks + per-query masked attention under c-geometry ──
        rank0 = (kept.int().cumsum(-1) - 1).float()                                        # (H,end)
        krc = _rope_rotate(kf[:, :end], rank0, invfc, 1.0)                                 # (H,end,D)
        npq = torch.gather(rank0, -1, qpos[None].expand(H, m_rows))                        # (H,m)
        qrc = _rope_rotate(qf[:, r0:r1], npq, invfc, 1.0)                                  # (H,m,D)
        sc = torch.einsum("hmd,hnd->hmn", qrc, krc) * sm_scale                             # (H,m,end)
        sel_q = (keep & kept[:, None, :]) | forced[None, None, :]      # per-query set (union∩own)∪forced
        sel_q = sel_q & causal[None]                                    # forced local includes future keys
        sc = torch.where(sel_q, sc, torch.full_like(sc, NEG))
        a = torch.softmax(sc, dim=-1)
        out[:, r0:r1] = torch.einsum("hmn,hnd->hmd", a, vf[:, :end])
    return out.unsqueeze(0).to(q.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# DECODE (Phase 4) — one query at a time. With a single query the "union" IS that query's own
# kept set, so NO per-query bitmask is needed (mask = all-ones over the selected slots) and the
# dense ranks are already per-query dense. Same two-geometry split as prefill: select under
# a-geometry, attend under base RoPE at dense ranks (always in-window).
#
# NOTE (expected, same property as exact_2): shared-tile prefill and decode are NOT bitwise
# equal for the same query — in prefill a query's key positions are ranks over its TILE's union
# (so they can have gaps), while at decode they are ranks over its OWN set (gapless). Both obey
# the same rule (rank over the selection unit's kept set); the selection unit differs. The
# apples-to-apples decode check is against prefill_mode="perquery", which uses this same path.
# ──────────────────────────────────────────────────────────────────────────────
def _decode_scores_kernel(q, K, inv_freq_a, q_pos, pos_scale_a, rope_scale_a, sm_scale, block_n=64):
    """Per-key scores (H,M) for one query via Triton _decode_select_kernel. q:(H,D), K:(H,M,D)."""
    assert HAVE_TRITON, "Triton not available"
    H, M, D = K.shape
    dev = K.device
    q = q.contiguous(); K = K.contiguous()
    invf = inv_freq_a.to(dev).float().contiguous()
    scores = torch.empty((H, M), dtype=torch.float32, device=dev)
    grid = (triton.cdiv(M, block_n), H)
    _decode_select_kernel[grid](
        q, K, invf, scores,
        sm_scale, float(pos_scale_a), float(rope_scale_a), int(q_pos),
        q.stride(0), q.stride(1),
        K.stride(0), K.stride(1), K.stride(2),
        scores.stride(0), scores.stride(1),
        H, M, BLOCK_N=block_n, HALF_DIM=D // 2,
        num_warps=4, num_stages=1,
    )
    return scores


@torch.no_grad()
def _decode_build_sets_pq(q, K, tau, inv_freq_a, q_pos, pos_scale_a, rope_scale_a, sm_scale,
                          n_global, n_local, k_max, scores=None):
    """Per-query selection + DENSE-RANK positions for one query (host, torch), vectorized over heads.
    q:(H,D), K:(H,M,D), M=q_pos+1. Returns idx (H,k_max) long (sentinel=M), newpos_k (H,k_max) fp32,
    newpos_q (H,) fp32.
    Rule: keep = score >= max - tau, plus forced global+local+self; FULL set (no top-pct — that is
    the point of pq_exact). If |kept| > k_max, keep forced + top-(k_max-n_forced) non-forced BY SCORE
    — for a single query expmass = exp(s-m) is strictly monotone in s, so score-ranking is IDENTICAL
    to the expmass ranking used at prefill. Positions = dense ranks over the final kept set."""
    H, M, D = K.shape
    dev = K.device
    # forced (global ∪ local ∪ self) is kept unconditionally -> it must fit in k_max, else budget
    # clamps to 0 and the position-sort would silently drop forced keys. Static check, no sync.
    assert n_global + n_local <= k_max, (
        f"decode forced set (n_global={n_global} + n_local={n_local}) exceeds k_max={k_max}")
    inv_freq_a = inv_freq_a.to(dev).float()
    tau = tau.to(dev).float()
    NEG = float("-inf")

    if scores is None:
        pos = torch.arange(M, device=dev)
        qs = _rope_rotate(q, torch.tensor(float(q_pos) * pos_scale_a, device=dev), inv_freq_a, rope_scale_a)
        Ks = _rope_rotate(K, pos.float() * pos_scale_a, inv_freq_a, rope_scale_a)
        score = torch.einsum("hd,hmd->hm", qs, Ks) * sm_scale                 # (H,M)
    else:
        score = scores.to(dev).float()

    m = score.max(dim=-1).values
    keep = score >= (m - tau)[:, None]                                        # (H,M) threshold passers

    forced = torch.zeros(M, dtype=torch.bool, device=dev)
    forced[:min(n_global, M)] = True                          # global / sink
    forced[max(0, q_pos - n_local + 1): q_pos + 1] = True     # local window
    forced[q_pos] = True                                      # always attend to self
    forced_hm = forced[None].expand(H, M)

    kp = keep | forced_hm                                                     # (H,M) full set
    usize = keep.sum(-1, dtype=torch.int32)[:, None]                          # (H,1) union (pre-forced)
    over = kp.sum(-1) > k_max                                                 # (H,)
    _ovf_accumulate(over[:, None], usize, phase="decode")                     # measurement only

    # k_max cap: forced + top-budget non-forced by score (exact budget via topk -> no tie overshoot,
    # so the position-sort truncation below can never drop a forced key / the query itself).
    n_forced = int(forced.sum())
    budget = max(k_max - n_forced, 0)
    cand = kp & (~forced_hm)
    sc = torch.where(cand, score, torch.full_like(score, NEG))
    Ktop = max(min(budget, M), 1)
    tv, ti = sc.topk(Ktop, dim=-1)
    ok = (torch.arange(Ktop, device=dev)[None, :] < budget) & (tv > NEG)
    sel = torch.zeros_like(kp)
    sel.scatter_(-1, ti, ok)
    final_kp = torch.where(over[:, None], sel | forced_hm, kp)                # (H,M)

    # ── dense ranks over the final kept set (same rule as prefill) ──
    kj = torch.arange(M, device=dev)
    rank0 = (final_kp.int().cumsum(-1) - 1).float()                           # (H,M)

    masked_pos = torch.where(final_kp, kj[None].expand(H, M),
                             torch.full((H, M), M, device=dev, dtype=torch.long))
    idx_sorted = masked_pos.sort(-1).values[:, :k_max]                        # (H, min(M,k_max))
    if idx_sorted.shape[-1] < k_max:
        pad = torch.full((H, k_max - idx_sorted.shape[-1]), M, device=dev, dtype=idx_sorted.dtype)
        idx_sorted = torch.cat([idx_sorted, pad], dim=-1)
    idx_clamp = idx_sorted.clamp(max=M - 1)
    newpos_k = torch.gather(rank0, -1, idx_clamp)
    newpos_k = torch.where(idx_sorted < M, newpos_k, torch.zeros_like(newpos_k))
    newpos_q = rank0[:, q_pos]                                                # q_pos is forced (self)
    return idx_sorted, newpos_k.float(), newpos_q.float()


@torch.no_grad()
def _decode_attend_torch_pq(q, K, V, idx, newpos_k, newpos_q, inv_freq_c, sm_scale):
    """Golden torch attend over the selected set, base RoPE at dense ranks (rope_scale=1).
    q:(H,D), K,V:(H,M,D), idx/newpos_k:(H,k_max). -> (H,D)."""
    H, M, D = K.shape
    dev = K.device
    inv_freq_c = inv_freq_c.to(dev).float()
    outs = []
    for h in range(H):
        valid = idx[h] < M
        kk = idx[h][valid]
        npk = newpos_k[h][valid]
        qr = _rope_rotate(q[h], newpos_q[h], inv_freq_c, 1.0)               # (D,)
        kr = _rope_rotate(K[h, kk], npk, inv_freq_c, 1.0)                   # (k,D)
        s = (qr[None, :] * kr).sum(-1) * sm_scale
        a = torch.softmax(s, dim=-1)
        outs.append((a[:, None] * V[h, kk].float()).sum(0))
    return torch.stack(outs, dim=0).to(q.dtype)                              # (H,D)


def _decode_attend_kernelC_pq(q, K, V, idx, newpos_k, newpos_q, inv_freq_c, sm_scale,
                              k_max, q_pos, block_m=16, block_k=32):
    """Attend via the PQ Kernel C: the single query is padded to `block_m` rows (only row 0 real),
    N_CTX = cache length M, q_pos_base = q_pos, rope_scale = 1 (base RoPE at dense ranks).
    Mask = -1 (all bits set) on valid slots -> row 0 attends its whole selected set; padded rows
    are computed then discarded."""
    assert HAVE_TRITON, "Triton not available"
    H, M, D = K.shape
    dev = K.device
    dt = q.dtype
    qh = torch.zeros((H, block_m, D), dtype=dt, device=dev); qh[:, 0, :] = q
    kh = K.contiguous(); vh = V.contiguous()
    Idx = idx.to(torch.int32).unsqueeze(1).contiguous()                      # (H,1,k_max), sentinel=M
    NewposK = newpos_k.float().unsqueeze(1).contiguous()                     # (H,1,k_max)
    NewposQ = torch.zeros((H, 1, block_m), dtype=torch.float32, device=dev)
    NewposQ[:, 0, 0] = newpos_q                                              # row 0 = real query
    valid = (idx < M)
    Mask = torch.where(valid, torch.full_like(idx, -1, dtype=torch.int32),
                       torch.zeros_like(idx, dtype=torch.int32)).unsqueeze(1).contiguous()
    Out = torch.empty((H, block_m, D), dtype=dt, device=dev)
    invfc = inv_freq_c.to(dev).float().contiguous()
    _gather_attn_pq_kernel[(1, H)](
        qh, kh, vh, invfc, Idx, NewposK, NewposQ, Mask, Out,
        sm_scale, 1.0, int(q_pos),
        qh.stride(0), qh.stride(1), qh.stride(2),
        kh.stride(0), kh.stride(1), kh.stride(2),
        vh.stride(0), vh.stride(1), vh.stride(2),
        Idx.stride(0), Idx.stride(1), Idx.stride(2),
        NewposK.stride(0), NewposK.stride(1), NewposK.stride(2),
        NewposQ.stride(0), NewposQ.stride(1), NewposQ.stride(2),
        Mask.stride(0), Mask.stride(1), Mask.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        H, M, BLOCK_M=block_m, BLOCK_K=block_k, K_MAX=k_max, HEAD_DIM=D, HALF_DIM=D // 2,
        num_warps=4, num_stages=1,
    )
    return Out[:, 0, :]                                                      # (H,D)


# ──────────────────────────────────────────────────────────────────────────────
# Model install (prefill + decode)
# ──────────────────────────────────────────────────────────────────────────────
def install_shared_pi_forward(model, pct=30.0, context_window=4096, n_global=32, n_local=32,
                              tile=32, k_max=4096, block_n=32, block_k=32, max_len=None,
                              pct_decode=100.0, decode_backend="torch", select_backend="torch",
                              prefill_mode="shared", measure=False, select_score="expmass",
                              select_geom="pi", apply_mscale=True):
    """Patch LlamaAttention.forward to route through the PQ-UNION Triton kernels.

    select_geom: geometry Kernel A uses to SELECT (Kernel C is always base RoPE at dense ranks):
      "pi"   -> plain-loaded model; base inv_freq at compressed positions (pos_scale=min(1,W/L)).
      "yarn" -> YARN-LOADED model required (rope_type="yarn", factor=N/W, orig_max=W); yarn inv_freq
                at raw positions; apply_mscale toggles the YaRN temperature in Kernel A only.
    pct / pct_decode are accepted for API compatibility and IGNORED (full union / full per-query
    set — that is the point of pq_exact).
    prefill_mode: "shared" (tile-32 union, the real kernel) | "perquery" (per-position decode-style
    selection — the reference mode the decode path is verified against).
    decode_backend / select_backend: "torch" (golden reference) | "triton"."""
    import transformers.models.llama.modeling_llama as ml
    assert select_geom in ("pi", "yarn"), f"select_geom must be pi|yarn, got {select_geom!r}"
    rotary = model.model.rotary_emb
    inv_freq_rot = rotary.inv_freq.detach().float()
    attn_scaling = float(getattr(rotary, "attention_scaling", 1.0))

    # base RoPE inv_freq for Kernel C (and for "pi" selection). Under "yarn" the rotary freqs are the
    # YaRN ones, so recompute base theta^{-2i/D} from config exactly like the hybrid kernel does.
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
            print("WARNING [pq kernel]: rotary.attention_scaling == 1.0 — the model does NOT look "
                  "yarn-loaded (mscale should be >1). Load it with rope_scaling rope_type='yarn'.",
                  flush=True)
        inv_freq_a = inv_freq_rot                          # YaRN NTK-by-parts freqs
        rope_scale_a = attn_scaling if apply_mscale else 1.0
    else:
        inv_freq_a = inv_freq_base                         # base freqs; PI compression via pos_scale
        rope_scale_a = 1.0
    assert inv_freq_base.numel() == inv_freq_a.numel(), \
        f"inv_freq length mismatch: {inv_freq_base.numel()} vs {inv_freq_a.numel()}"
    print(f"pq kernel: A={select_geom}(rope_scale_A={rope_scale_a:.4f}), "
          f"C=base RoPE(theta={theta:.0f}) at DENSE-RANK positions (always in-window); "
          f"FULL-UNION sets, k_max={k_max}, overflow->top-k_max by {select_score}", flush=True)

    for li, layer in enumerate(model.model.layers):
        sa = layer.self_attn
        dev = sa.q_proj.weight.device
        sa.spi_inv_freq_a = inv_freq_a.to(dev)            # select-geometry freqs (Kernel A)
        sa.spi_rope_scale_a = rope_scale_a                # mscale (yarn) or 1.0 (pi)
        sa.spi_inv_freq_c = inv_freq_base.to(dev)         # base RoPE freqs (Kernel C)
        sa.spi_geom = select_geom
        sa.spi_W = context_window
        sa.spi_ng = n_global; sa.spi_nl = n_local
        sa.spi_tile = tile; sa.spi_kmax = k_max
        sa.spi_block_n = block_n; sa.spi_block_k = block_k
        sa.spi_layer_idx = getattr(sa, "layer_idx", li)   # which slot in the KV cache
        sa.spi_max_len = max_len                          # target length for frozen pos_scale_a (pi)
        sa.spi_pos_scale_a = None                         # set once at prefill, reused at decode
        sa.spi_select_score = select_score                # overflow-truncation ranking metric
        sa.spi_decode_backend = decode_backend            # attend: "torch" (reference) or "triton"
        sa.spi_select_backend = select_backend            # select: "torch" (reference) or "triton"
        sa.spi_prefill_mode = prefill_mode                # "shared" (tile-32 union) or "perquery"

    def spi_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, **kwargs):
        B, N, _ = hidden_states.shape
        hs = (B, N, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hs).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hs).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hs).transpose(1, 2)

        # ── KV cache: store RAW k,v (kernels rotate internally); GQA-repeat AFTER caching. ──
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

        # ── frozen select pos_scale: pi -> min(1, W/L) once at prefill; yarn -> raw positions (1.0) ──
        if past_len == 0:
            if self.spi_geom == "yarn":
                self.spi_pos_scale_a = 1.0
            else:
                L_target = self.spi_max_len if self.spi_max_len is not None else N
                self.spi_pos_scale_a = min(1.0, self.spi_W / float(L_target))

        if past_len == 0 and getattr(self, "spi_prefill_mode", "shared") == "shared":
            # ── PREFILL path (tile-32 per-query union; 3-stage Triton pipeline) ──
            out, usize, kept_size, overflow = shared_pq_attention(
                q, k, v, self.tau_head_vec, self.spi_inv_freq_a, self.spi_inv_freq_c,
                pos_scale_a=self.spi_pos_scale_a, rope_scale_a=self.spi_rope_scale_a,
                sm_scale=self.scaling, n_global=self.spi_ng, n_local=self.spi_nl,
                tile=self.spi_tile, k_max=self.spi_kmax,
                block_n=self.spi_block_n, block_k=self.spi_block_k, q_pos_base=past_len,
                select_score=self.spi_select_score)
            self.spi_last_usize = usize.detach()          # true union sizes (log R = usize / top-pct)
            self.spi_last_kept = kept_size.detach()
            self.spi_last_overflow = overflow.detach()
        else:
            # ── PER-QUERY path: each query selects its own set. Serves both real DECODE
            #    (past_len>0, q_len=1) AND per-query prefill (past_len==0, prefill_mode="perquery",
            #    the reference mode the decode path is verified against).
            #    q: (1,H,q_len,D); k,v: (1,H,Ntot,D) full raw cache. Query at position past_len+i
            #    attends keys [0 .. past_len+i].
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
                idx, npk, npq = _decode_build_sets_pq(
                    qi, Ki, self.tau_head_vec, self.spi_inv_freq_a, P, self.spi_pos_scale_a,
                    self.spi_rope_scale_a, self.scaling, self.spi_ng, self.spi_nl,
                    self.spi_kmax, scores=scores)
                if attend_backend == "triton":
                    oh = _decode_attend_kernelC_pq(qi, Ki, Vi, idx, npk, npq, self.spi_inv_freq_c,
                                                   self.scaling, self.spi_kmax, q_pos=P,
                                                   block_k=self.spi_block_k)
                else:
                    oh = _decode_attend_torch_pq(qi, Ki, Vi, idx, npk, npq, self.spi_inv_freq_c,
                                                 self.scaling)
                outs.append(oh)
            out = torch.stack(outs, dim=1).unsqueeze(0)       # (1,H,q_len,D)

        attn_output = out.transpose(1, 2).reshape(B, N, -1).contiguous()
        return self.o_proj(attn_output), None

    ml.LlamaAttention.forward = spi_forward
