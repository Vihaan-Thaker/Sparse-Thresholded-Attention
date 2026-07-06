#!/usr/bin/env python3
"""
shared_selection_triton_exact.py

EXACT-MAX variant of shared_selection_triton.py. Identical Stage 2 (host) and Stage 3 (Kernel C);
the ONLY difference is Kernel A uses TWO passes over the keys to get the EXACT per-query max before
deciding keep (instead of the 1-pass running-max approximation). Use this alongside the 1-pass
version to measure the time/accuracy consequences of running-max vs exact-max selection.

  KERNEL A  _select_vote_kernel   (per head, per query-tile)  -- O(N), tensor-core dot, 2 passes
      pass 1: compressed-position RoPE scores -> EXACT per-query max over all causal keys
      pass 2: keep = score >= exact_max - tau ; votes(j) = sum over the tile's queries ; -> HBM
  HOST      _build_sets()         votes -> top-X% v_cut (+global/local) -> gather index + newpos
  KERNEL C  _gather_attn_kernel   (per head, per query-tile)  -- O(k), tensor-core dot
      gather K,V of the k shared keys -> repositioned attention (RoPE at newpos) -> online softmax

Batch size 1 (q,k,v are (1,H,N,D)). Prefill only.
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


if HAVE_TRITON:

    # ══════════════════════════════════════════════════════════════════════════
    # KERNEL A : selection + voting   (one program per (tile, head)) -- EXACT MAX, 2 passes
    # ══════════════════════════════════════════════════════════════════════════
    @triton.jit
    def _select_vote_kernel(
        Q, K, Tau, InvFreq, Votes, RowCounts,
        sm_scale, pos_scale, rope_scale, q_pos_base,
        stride_qh, stride_qn, stride_qd,
        stride_kh, stride_kn, stride_kd,
        stride_vh, stride_vt, stride_vn,
        stride_rh, stride_rn,
        H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr, HALF_DIM: tl.constexpr,
    ):
        start_m = tl.program_id(0)          # tile index
        off_h   = tl.program_id(1)          # head (batch=1)
        q_base = Q + off_h * stride_qh
        k_base = K + off_h * stride_kh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)      # row index into Q
        qpos   = q_pos_base + offs_m                            # ABSOLUTE query position (row != pos at decode)
        offs_h = tl.arange(0, HALF_DIM)
        invf = tl.load(InvFreq + offs_h)
        tau  = tl.load(Tau + off_h)

        # raw Q halves -> rotate at COMPRESSED positions (cpos = pos*pos_scale)
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

        # ── pass 2: keep + vote against the EXACT max ──
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
            # exclude padded query rows (ragged last tile when N % BLOCK_M != 0) from the vote/counts
            keep = (qk >= (m[:, None] - tau)) & causal & (offs_m[:, None] < N_CTX)
            row_counts += tl.sum(keep.to(tl.int32), 1)           # per-query kept count
            votes = tl.sum(keep.to(tl.int32), 0)                 # (BLOCK_N,)
            tl.store(Votes + off_h * stride_vh + start_m * stride_vt + offs_n * stride_vn,
                     votes, mask=offs_n < N_CTX)
        tl.store(RowCounts + off_h * stride_rh + offs_m * stride_rn,
                 row_counts, mask=offs_m < N_CTX)


    # ══════════════════════════════════════════════════════════════════════════
    # KERNEL C : gather + repositioned attention   (one program per (tile, head))
    # ══════════════════════════════════════════════════════════════════════════
    @triton.jit
    def _gather_attn_kernel(
        Q, Kc, Vc, InvFreq, Idx, NewposK, NewposQ, Out,
        sm_scale, rope_scale, q_pos_base,
        stride_qh, stride_qn, stride_qd,
        stride_kh, stride_kn, stride_kd,
        stride_vh, stride_vn, stride_vd,
        stride_ih, stride_it, stride_ik,
        stride_nkh, stride_nkt, stride_nkk,
        stride_nqh, stride_nqt, stride_nqm,
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

        # raw Q -> rotate at NEWPOS (one position per query)
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
            keep = (qpos[:, None] >= idx[None, :]) & valid[None, :]      # causal (orig pos) + not padding
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
    # DECODE-SELECT KERNEL : single query -> per-key scores  (one program per (key-block, head))
    # Counterpart to Kernel A for decode: scores the one query at COMPRESSED positions and writes
    # the raw scores (H, M) to HBM. The host takes the (exact) max + threshold + cap + reposition.
    # Score vector is tiny (one query), so nothing is fused -> always exact (no running-max variant).
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

        # single query, rotate at compressed position q_pos*pos_scale
        q1 = tl.load(Q + off_h * stride_qh + offs_h * stride_qd)
        q2 = tl.load(Q + off_h * stride_qh + (HALF_DIM + offs_h) * stride_qd)
        angq = (q_pos * pos_scale) * invf
        cosq = tl.cos(angq) * rope_scale
        sinq = tl.sin(angq) * rope_scale
        qr1 = q1 * cosq - q2 * sinq
        qr2 = q2 * cosq + q1 * sinq

        # key block, rotate at compressed positions
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
# HOST : votes -> top-X% (+global/local) -> gather index list + PI newpos
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _build_sets(votes, pos_scale, n_global, n_local, pct, T, k_max, q_pos_base=0, measure=False):
    """votes (H,nt,N) int -> idx (H,nt,k_max) int32 (sentinel=N), newpos_k (H,nt,k_max),
    newpos_q (H,nt,T). Reuses the validated reference logic (vote top-X%, PI reposition).
    q_pos_base = absolute position of the first query row (0 at prefill; past_len at decode)."""
    H, nt, N = votes.shape
    dev = votes.device
    kj = torch.arange(N, device=dev)
    glob = kj < n_global                                                # (N,) absolute key positions
    tstart = q_pos_base + torch.arange(nt, device=dev) * T              # absolute tile-start positions
    lo = tstart - n_local + 1; hi = tstart + T - 1
    local_block = (kj[None, :] >= lo[:, None]) & (kj[None, :] <= hi[:, None])   # (nt,N)

    vf = votes.float()
    Usize = (vf > 0).sum(-1)                                            # (H,nt)
    sv = torch.sort(vf, dim=-1, descending=True).values                # (H,nt,N)
    glob_loc = glob[None, :] | local_block                             # (nt,N) forced global+local

    # Per (head,tile), drop the X% only when the requested set exceeds k_max, in coarse steps:
    # 30% -> 25% -> 20% (stop at 20; do NOT go lower for now). Empirically ~25% suffices in most
    # overflow cases. `chosen` records the largest X% that fits; tiles that already fit keep the
    # requested pct. Tiles still over k_max at 20% stay unresolved (chosen=0; gather truncates them).
    start_pct = max(0, int(round(pct)))
    pct_steps = [start_pct] + [c for c in (25, 20) if c < start_pct]    # e.g. [30, 25, 20]
    shared = torch.zeros((H, nt, N), dtype=torch.bool, device=dev)
    kept   = torch.zeros((H, nt, N), dtype=torch.bool, device=dev)
    chosen = torch.zeros((H, nt), dtype=torch.int32, device=dev)
    resolved = torch.zeros((H, nt), dtype=torch.bool, device=dev)
    orig_overflow = torch.zeros((H, nt), dtype=torch.bool, device=dev)
    shared_p = kept_p = None
    for p in pct_steps:
        budget_p = torch.ceil((p / 100.0) * Usize).clamp(min=1).long()
        v_cut_p = torch.gather(sv, -1, (budget_p - 1).clamp(0, N - 1).unsqueeze(-1))
        shared_p = vf >= v_cut_p                                        # (H,nt,N) tie-inclusive
        kept_p = shared_p | glob_loc[None]                             # (H,nt,N)
        if p == start_pct:
            orig_overflow = kept_p.sum(-1) > k_max
        newly = (kept_p.sum(-1) <= k_max) & (~resolved)                # (H,nt)
        sel = newly.unsqueeze(-1)
        shared = torch.where(sel, shared_p, shared)
        kept   = torch.where(sel, kept_p, kept)
        chosen = torch.where(newly, torch.full_like(chosen, p), chosen)
        resolved = resolved | newly
        if bool(resolved.all()):
            break
    if start_pct == 0:
        shared_p = torch.zeros((H, nt, N), dtype=torch.bool, device=dev)
        kept_p = glob_loc[None].expand(H, nt, N)
        orig_overflow = kept_p.sum(-1) > k_max
    # tiles that never fit (even at 20%): use the smallest tried set; gather still truncates those.
    sel = (~resolved).unsqueeze(-1)
    shared = torch.where(sel, shared_p, shared)
    kept   = torch.where(sel, kept_p, kept)

    # `overflow20` = "had to go below the floor% (20%)" mask — cheap, no host sync. Always computed.
    floor_pct = pct_steps[-1]
    overflow20 = orig_overflow & (~resolved)                            # top-floor% still over k_max
    kpct_meas = chosen.clone()                                          # refined below only when measure=True

    if measure:
        # ── MEASUREMENT ONLY (instrumentation; does NOT alter selection/gather; adds per-iteration
        #    .all()/.any() host syncs, so it is skipped for benchmarking). For tiles whose top-20%
        #    still exceeds k_max, step 1% at a time (19..1) to record the true minimal fitting pct,
        #    plus the "adjusted"/"truncated" overflow prints.
        meas_resolved = resolved.clone()
        for p in range(floor_pct - 1, 0, -1):                          # 19,18,...,1
            if bool(meas_resolved.all()):
                break
            budget_m = torch.ceil((p / 100.0) * Usize).clamp(min=1).long()
            v_cut_m = torch.gather(sv, -1, (budget_m - 1).clamp(0, N - 1).unsqueeze(-1))
            kept_m = (vf >= v_cut_m) | glob_loc[None]                   # (H,nt,N)
            newly_m = (kept_m.sum(-1) <= k_max) & (~meas_resolved)     # (H,nt)
            kpct_meas = torch.where(newly_m, torch.full_like(kpct_meas, p), kpct_meas)
            meas_resolved = meas_resolved | newly_m

        adjusted = orig_overflow & resolved
        if bool(adjusted.any()):
            vals, counts = torch.unique(chosen[adjusted].detach().cpu(), return_counts=True)
            summary = ", ".join(f"{int(v)}%:{int(c)}" for v, c in zip(vals, counts))
            print(f"Adjusted top-pct to fit k_max={k_max}: {summary} (requested {pct:g}%).", flush=True)
        unresolved = orig_overflow & (~resolved)
        if bool(unresolved.any()):
            print(f"WARNING: {int(unresolved.sum().item())} (head,tile) sets still exceed k_max={k_max} "
                  f"even at top {pct_steps[-1]}%; gather will truncate those tiles.", flush=True)

    rank0 = (kept.int().cumsum(-1) - 1).float()
    cpos = kj.float() * pos_scale
    cr = torch.where(kept, cpos[None, None, :] - rank0, torch.full_like(rank0, float("-inf")))
    newpos_all = rank0 + torch.cummax(cr, dim=-1).values               # (H,nt,N)

    masked_pos = torch.where(kept, kj[None, None, :].expand(H, nt, N),
                             torch.full((H, nt, N), N, device=dev, dtype=torch.long))
    idx_sorted = masked_pos.sort(-1).values[:, :, :k_max]              # (H,nt, min(N,k_max))
    if idx_sorted.shape[-1] < k_max:                                   # short prefill (N < k_max):
        pad = torch.full((H, nt, k_max - idx_sorted.shape[-1]), N,     # pad out to k_max with sentinel N
                         device=dev, dtype=idx_sorted.dtype)           # so Kernel C's K_MAX matches idx width
        idx_sorted = torch.cat([idx_sorted, pad], dim=-1)
    idx_clamp = idx_sorted.clamp(max=N - 1)
    newpos_k = torch.gather(newpos_all, -1, idx_clamp)
    newpos_k = torch.where(idx_sorted < N, newpos_k, torch.zeros_like(newpos_k))

    qpos = q_pos_base + (torch.arange(nt, device=dev)[:, None] * T + torch.arange(T, device=dev)[None, :])  # (nt,T) absolute
    qpos = qpos.clamp(max=N - 1)                                # ragged last tile: padded query slots (unused) stay in-range
    newpos_q = torch.gather(newpos_all, -1, qpos[None].expand(H, nt, T))                       # (H,nt,T)

    # per-tile sizes: |top-X% voted set| (the "top 30%") and |full kept set| (= what k_max caps).
    shared_size = shared.to(torch.int32).sum(-1)                                               # (H,nt)
    kept_size   = kept.to(torch.int32).sum(-1)                                                 # (H,nt)

    return (idx_sorted.to(torch.int32).contiguous(),
            newpos_k.float().contiguous(),
            newpos_q.float().contiguous(),
            shared_size.contiguous(),
            kept_size.contiguous(),
            chosen.contiguous(),
            orig_overflow.contiguous(),
            kpct_meas.contiguous(),
            overflow20.contiguous())


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────
def shared_pi_attention(q, k, v, tau, inv_freq, pos_scale=1.0, pct=30.0, sm_scale=None,
                        rope_scale=1.0, n_global=32, n_local=32, tile=32, k_max=1024,
                        block_n=64, block_k=64, q_pos_base=0, measure=False):
    """q,k,v: (1,H,N,D) RAW. q_pos_base = absolute position of the first query (0 at prefill).
    measure=True runs the below-20% instrumentation (host syncs); leave False for speed.
    Returns out plus per-tile shared/kept/k' stats."""
    assert HAVE_TRITON, "Triton not available"
    Z, H, N, D = q.shape
    assert Z == 1
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    nt = (N + tile - 1) // tile            # ceil: last tile may be ragged (real prompts, N % tile != 0)
    qh, kh, vh = q[0].contiguous(), k[0].contiguous(), v[0].contiguous()
    tau = tau.float().contiguous(); inv_freq = inv_freq.float().contiguous()

    votes = torch.zeros((H, nt, N), dtype=torch.int32, device=q.device)
    row_counts = torch.zeros((H, N), dtype=torch.int32, device=q.device)
    gridA = (nt, H)
    _select_vote_kernel[gridA](
        qh, kh, tau, inv_freq, votes, row_counts,
        sm_scale, float(pos_scale), float(rope_scale), int(q_pos_base),
        qh.stride(0), qh.stride(1), qh.stride(2),
        kh.stride(0), kh.stride(1), kh.stride(2),
        votes.stride(0), votes.stride(1), votes.stride(2),
        row_counts.stride(0), row_counts.stride(1),
        H, N, BLOCK_M=tile, BLOCK_N=block_n, HEAD_DIM=D, HALF_DIM=D // 2,
        num_warps=4, num_stages=1,
    )

    idx, newpos_k, newpos_q, shared_size, kept_size, kpct, overflow, kpct_meas, overflow20 = _build_sets(
        votes, float(pos_scale), n_global, n_local, pct, tile, k_max, q_pos_base=q_pos_base, measure=measure)

    out = torch.empty((H, N, D), dtype=q.dtype, device=q.device)
    gridC = (nt, H)
    _gather_attn_kernel[gridC](
        qh, kh, vh, inv_freq, idx, newpos_k, newpos_q, out,
        sm_scale, float(rope_scale), int(q_pos_base),
        qh.stride(0), qh.stride(1), qh.stride(2),
        kh.stride(0), kh.stride(1), kh.stride(2),
        vh.stride(0), vh.stride(1), vh.stride(2),
        idx.stride(0), idx.stride(1), idx.stride(2),
        newpos_k.stride(0), newpos_k.stride(1), newpos_k.stride(2),
        newpos_q.stride(0), newpos_q.stride(1), newpos_q.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H, N, BLOCK_M=tile, BLOCK_K=block_k, K_MAX=k_max, HEAD_DIM=D, HALF_DIM=D // 2,
        num_warps=4, num_stages=1,
    )
    return out.unsqueeze(0), shared_size, kept_size, kpct, overflow, kpct_meas, overflow20


# ──────────────────────────────────────────────────────────────────────────────
# DECODE — step 3 torch reference: per-query sparse-PI attention for new token(s).
# Pure PyTorch (no Triton); this is the golden reference the Triton decode path
# (steps 4-5) will be validated against. Same selection/reposition rules as prefill,
# but per single query (tile-of-1) with a SCORE-based k_max cap (no voting).
# ──────────────────────────────────────────────────────────────────────────────
def _rope_rotate(x, pos, inv_freq, rope_scale):
    """Rotate x (..., D) by RoPE at `pos` (broadcastable to x.shape[:-1]); half-split (Llama) form."""
    half = x.shape[-1] // 2
    ang = pos[..., None].to(torch.float32) * inv_freq.to(torch.float32)      # (..., half)
    cos = torch.cos(ang) * rope_scale
    sin = torch.sin(ang) * rope_scale
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _decode_scores_kernel(q, K, inv_freq, q_pos, pos_scale, rope_scale, sm_scale, block_n=64):
    """Per-key scores (H,M) for one query via the Triton _decode_select_kernel. q:(H,D), K:(H,M,D)."""
    assert HAVE_TRITON, "Triton not available"
    H, M, D = K.shape
    dev = K.device
    q = q.contiguous(); K = K.contiguous()
    invf = inv_freq.to(dev).float().contiguous()
    scores = torch.empty((H, M), dtype=torch.float32, device=dev)
    grid = (triton.cdiv(M, block_n), H)
    _decode_select_kernel[grid](
        q, K, invf, scores,
        sm_scale, float(pos_scale), float(rope_scale), int(q_pos),
        q.stride(0), q.stride(1),
        K.stride(0), K.stride(1), K.stride(2),
        scores.stride(0), scores.stride(1),
        H, M, BLOCK_N=block_n, HALF_DIM=D // 2,
        num_warps=4, num_stages=1,
    )
    return scores


@torch.no_grad()
def _decode_build_sets(q, K, tau, inv_freq, q_pos, pos_scale, rope_scale, sm_scale,
                       n_global, n_local, k_max, pct_decode, scores=None):
    """Per-query SELECTION + PI reposition (host, torch), VECTORIZED over heads (no Python loop).
    q:(H,D), K:(H,M,D), M=q_pos+1. Returns idx (H,k_max) long padded with sentinel M,
    newpos_k (H,k_max) float, newpos_q (H,) float. Heads are independent (vectorized, no Python loop).
    Rule: threshold keep (score>=max-tau) + global + local + self, optional top-pct_decode% of
    passers (by score), then k_max cap by score; newpos = rank + cummax(cpos - rank).
    scores: precomputed per-key scores (H,M); if None they are computed here in torch."""
    H, M, D = K.shape
    dev = K.device
    inv_freq = inv_freq.to(dev).float()
    tau = tau.to(dev).float()
    NEG = float("-inf")

    if scores is None:
        pos = torch.arange(M, device=dev)
        cpos_sel = pos.float() * pos_scale
        qs = _rope_rotate(q, torch.tensor(float(q_pos) * pos_scale, device=dev), inv_freq, rope_scale)  # (H,D)
        Ks = _rope_rotate(K, cpos_sel, inv_freq, rope_scale)                                            # (H,M,D)
        score = torch.einsum("hd,hmd->hm", qs, Ks) * sm_scale                                           # (H,M)
    else:
        score = scores.to(dev).float()
    m = score.max(dim=-1).values
    keep = score >= (m - tau)[:, None]                                     # (H,M) threshold passers

    forced = torch.zeros(M, dtype=torch.bool, device=dev)
    forced[:min(n_global, M)] = True                          # global / sink
    forced[max(0, q_pos - n_local + 1): q_pos + 1] = True     # local window
    forced[q_pos] = True                                      # always attend to self
    forced_hm = forced[None].expand(H, M)

    # ── top-pct_decode% of threshold-passers, per head, by score (vectorized) ──
    if pct_decode < 100.0:
        n_pass = keep.sum(-1)                                              # (H,)
        budget_p = torch.ceil(pct_decode / 100.0 * n_pass).clamp(min=1).long()
        score_p = torch.where(keep, score, torch.full_like(score, NEG))
        sorted_p = torch.sort(score_p, dim=-1, descending=True).values     # (H,M)
        cut_p = torch.gather(sorted_p, -1, (budget_p - 1).clamp(0, M - 1)[:, None])   # (H,1)
        kept_pct = keep & (score >= cut_p)
    else:
        kept_pct = keep

    kp = kept_pct | forced_hm                                              # (H,M)

    # ── k_max cap by score: keep forced (global+local+self), fill remaining budget by highest score ──
    count = kp.sum(-1)                                                     # (H,)
    n_forced = int(forced.sum())
    budget_c = k_max - n_forced
    non_forced = kp & (~forced_hm)
    score_nf = torch.where(non_forced, score, torch.full_like(score, NEG))
    sorted_c = torch.sort(score_nf, dim=-1, descending=True).values       # (H,M)
    ci = min(max(budget_c - 1, 0), M - 1)
    cut_c = sorted_c[:, ci:ci + 1]                                         # (H,1) budget_c-th largest non-forced
    kept_nf = non_forced & (score >= cut_c)
    if budget_c <= 0:
        kept_nf = torch.zeros_like(kept_nf)
    capped = forced_hm | kept_nf
    over = (count > k_max)[:, None]
    final_kp = torch.where(over, capped, kp)                              # (H,M)

    # ── PI reposition (vectorized over heads; nt=1) — same formula as prefill _build_sets ──
    kj = torch.arange(M, device=dev)
    rank0 = (final_kp.int().cumsum(-1) - 1).float()                       # (H,M)
    cpos = kj.float() * pos_scale
    cr = torch.where(final_kp, cpos[None] - rank0, torch.full_like(rank0, NEG))
    newpos_all = rank0 + torch.cummax(cr, dim=-1).values                  # (H,M)

    masked_pos = torch.where(final_kp, kj[None].expand(H, M),
                             torch.full((H, M), M, device=dev, dtype=torch.long))
    idx_sorted = masked_pos.sort(-1).values[:, :k_max]                    # (H, min(M,k_max)) ascending, sentinel=M
    if idx_sorted.shape[-1] < k_max:
        pad = torch.full((H, k_max - idx_sorted.shape[-1]), M, device=dev, dtype=idx_sorted.dtype)
        idx_sorted = torch.cat([idx_sorted, pad], dim=-1)
    idx_clamp = idx_sorted.clamp(max=M - 1)
    newpos_k = torch.gather(newpos_all, -1, idx_clamp)
    newpos_k = torch.where(idx_sorted < M, newpos_k, torch.zeros_like(newpos_k))
    newpos_q = newpos_all[:, q_pos]                                       # (H,) q_pos is forced (self)
    return idx_sorted, newpos_k.float(), newpos_q.float()


@torch.no_grad()
def _decode_attend_torch(q, K, V, idx, newpos_k, newpos_q, inv_freq, rope_scale, sm_scale):
    """Torch attend over the selected set. q:(H,D), K,V:(H,M,D), idx/newpos_k:(H,k_max). -> (H,D)."""
    H, M, D = K.shape
    dev = K.device
    inv_freq = inv_freq.to(dev).float()
    outs = []
    for h in range(H):
        valid = idx[h] < M
        kk = idx[h][valid]
        npk = newpos_k[h][valid]
        qr = _rope_rotate(q[h], newpos_q[h], inv_freq, rope_scale)         # (D,)
        kr = _rope_rotate(K[h, kk], npk, inv_freq, rope_scale)             # (k,D)
        s = (qr[None, :] * kr).sum(-1) * sm_scale
        a = torch.softmax(s, dim=-1)
        outs.append((a[:, None] * V[h, kk].float()).sum(0))
    return torch.stack(outs, dim=0).to(q.dtype)                            # (H,D)


def _decode_attend_kernelC(q, K, V, idx, newpos_k, newpos_q, inv_freq, rope_scale, sm_scale,
                           k_max, q_pos, block_m=16, block_k=32):
    """Attend via the EXISTING Triton Kernel C (unmodified): the single query is padded to a
    block of `block_m` rows (only row 0 real), N_CTX = cache length M, q_pos_base = q_pos."""
    assert HAVE_TRITON, "Triton not available"
    H, M, D = K.shape
    dev = K.device
    dt = q.dtype
    qh = torch.zeros((H, block_m, D), dtype=dt, device=dev); qh[:, 0, :] = q
    kh = K.contiguous(); vh = V.contiguous()
    Idx = idx.to(torch.int32).unsqueeze(1).contiguous()                    # (H,1,k_max), sentinel=M
    NewposK = newpos_k.float().unsqueeze(1).contiguous()                   # (H,1,k_max)
    NewposQ = torch.zeros((H, 1, block_m), dtype=torch.float32, device=dev)
    NewposQ[:, 0, 0] = newpos_q                                            # row 0 = real query
    Out = torch.empty((H, block_m, D), dtype=dt, device=dev)
    invf = inv_freq.to(dev).float().contiguous()
    _gather_attn_kernel[(1, H)](
        qh, kh, vh, invf, Idx, NewposK, NewposQ, Out,
        sm_scale, float(rope_scale), int(q_pos),
        qh.stride(0), qh.stride(1), qh.stride(2),
        kh.stride(0), kh.stride(1), kh.stride(2),
        vh.stride(0), vh.stride(1), vh.stride(2),
        Idx.stride(0), Idx.stride(1), Idx.stride(2),
        NewposK.stride(0), NewposK.stride(1), NewposK.stride(2),
        NewposQ.stride(0), NewposQ.stride(1), NewposQ.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        H, M, BLOCK_M=block_m, BLOCK_K=block_k, K_MAX=k_max, HEAD_DIM=D, HALF_DIM=D // 2,
        num_warps=4, num_stages=1,
    )
    return Out[:, 0, :]                                                    # (H,D)


def install_shared_pi_forward(model, pct=30.0, context_window=4096, n_global=32, n_local=32,
                              tile=32, k_max=1024, block_n=32, block_k=32, max_len=None,
                              pct_decode=100.0, decode_backend="torch", select_backend="torch",
                              prefill_mode="shared", measure=False):
    """Patch LlamaAttention.forward to route through the EXACT-max shared-PI Triton kernels.

    max_len: freeze pos_scale = min(1, context_window/max_len) at prefill (for decode consistency).
             None -> use the prefill length N (identical to the old per-forward behavior)."""
    import transformers.models.llama.modeling_llama as ml
    rotary = model.model.rotary_emb
    inv_freq = rotary.inv_freq.detach().float()
    rope_scale = float(getattr(rotary, "attention_scaling", 1.0))
    for li, layer in enumerate(model.model.layers):
        sa = layer.self_attn
        dev = sa.q_proj.weight.device
        sa.spi_inv_freq = inv_freq.to(dev)
        sa.spi_rope_scale = rope_scale
        sa.spi_pct = pct; sa.spi_W = context_window
        sa.spi_ng = n_global; sa.spi_nl = n_local
        sa.spi_tile = tile; sa.spi_kmax = k_max
        sa.spi_block_n = block_n; sa.spi_block_k = block_k
        sa.spi_layer_idx = getattr(sa, "layer_idx", li)   # which slot in the KV cache
        sa.spi_max_len = max_len                          # target length for frozen pos_scale
        sa.spi_pos_scale = None                           # set once at prefill, reused at decode
        sa.spi_pct_decode = pct_decode                    # top-% of threshold-passers kept at decode
        sa.spi_decode_backend = decode_backend            # attend: "torch" or "triton" (Kernel C)
        sa.spi_select_backend = select_backend            # select: "torch" or "triton" (_decode_select_kernel)
        sa.spi_prefill_mode = prefill_mode                # "shared" (tile-32) or "perquery" (per-position)
        sa.spi_measure = measure                          # below-20% instrumentation (host syncs; off for speed)

    def spi_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, **kwargs):
        B, N, _ = hidden_states.shape
        hs = (B, N, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hs).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hs).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hs).transpose(1, 2)

        # ── KV cache: store RAW k,v (kernels rotate internally); GQA-repeat AFTER caching. ──
        # HF may pass the cache as `past_key_values` or (older) `past_key_value`.
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

        # ── frozen pos_scale: set once at prefill from spi_max_len (or N), reused at decode ──
        if past_len == 0:
            L_target = self.spi_max_len if self.spi_max_len is not None else N
            self.spi_pos_scale = min(1.0, self.spi_W / float(L_target))
        pos_scale = self.spi_pos_scale

        if past_len == 0 and getattr(self, "spi_prefill_mode", "shared") == "shared":
            # ── PREFILL path (tile-32 shared selection; 3-stage Triton pipeline) ──
            out, shared_size, kept_size, kpct, overflow, kpct_meas, overflow20 = shared_pi_attention(
                q, k, v, self.tau_head_vec, self.spi_inv_freq,
                pos_scale=pos_scale, pct=self.spi_pct, sm_scale=self.scaling,
                rope_scale=self.spi_rope_scale, n_global=self.spi_ng,
                n_local=self.spi_nl, tile=self.spi_tile, k_max=self.spi_kmax,
                block_n=self.spi_block_n, block_k=self.spi_block_k, q_pos_base=past_len,
                measure=getattr(self, "spi_measure", False))
            self.spi_last_shared = shared_size.detach()
            self.spi_last_kept = kept_size.detach()
            self.spi_last_kpct = kpct.detach()
            self.spi_last_overflow = overflow.detach()
            self.spi_last_kpct_meas = kpct_meas.detach()      # true minimal fitting pct (fine below floor)
            self.spi_last_overflow20 = overflow20.detach()    # had to go below the floor (20%)
        else:
            # ── PER-QUERY path: each query selects its own set (decode behavior). Serves both real
            #    decode (past_len>0, q_len=1) AND per-query prefill (past_len==0, prefill_mode="perquery").
            #    q: (1,H,q_len,D); k,v: (1,H,Ntot,D) full raw cache. Query at position past_len+i
            #    attends to keys [0 .. past_len+i].
            select_backend = getattr(self, "spi_select_backend", "torch")
            attend_backend = getattr(self, "spi_decode_backend", "torch")
            Kf, Vf, qf = k[0], v[0], q[0]
            outs = []
            for i in range(N):
                P = past_len + i
                Ki, Vi, qi = Kf[:, :P + 1, :], Vf[:, :P + 1, :], qf[:, i, :]
                scores = None
                if select_backend == "triton":
                    scores = _decode_scores_kernel(qi, Ki, self.spi_inv_freq, P, pos_scale,
                                                   self.spi_rope_scale, self.scaling, block_n=self.spi_block_n)
                idx, npk, npq = _decode_build_sets(
                    qi, Ki, self.tau_head_vec, self.spi_inv_freq, P, pos_scale,
                    self.spi_rope_scale, self.scaling, self.spi_ng, self.spi_nl, self.spi_kmax,
                    getattr(self, "spi_pct_decode", 100.0), scores=scores)
                if attend_backend == "triton":
                    oh = _decode_attend_kernelC(qi, Ki, Vi, idx, npk, npq, self.spi_inv_freq,
                                                self.spi_rope_scale, self.scaling, self.spi_kmax,
                                                q_pos=P, block_k=self.spi_block_k)
                else:
                    oh = _decode_attend_torch(qi, Ki, Vi, idx, npk, npq, self.spi_inv_freq,
                                              self.spi_rope_scale, self.scaling)
                outs.append(oh)
            out = torch.stack(outs, dim=1).unsqueeze(0)       # (1,H,q_len,D)

        attn_output = out.transpose(1, 2).reshape(B, N, -1).contiguous()
        return self.o_proj(attn_output), None

    ml.LlamaAttention.forward = spi_forward
