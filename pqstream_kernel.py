#!/usr/bin/env python3
"""
pqstream_kernel.py

PQ-STREAM kernel ("pq_stream"): per-query thresholding with STREAMING TILE POSITIONS, fused into a
single Triton launch. Fork of shared_selection_triton_perquery.py.

WHY THIS IS NOT IN THE shared_selection_triton_* NAMESPACE
-----------------------------------------------------------
That prefix names the defining trait of those kernels -- one SHARED key set per 32-query tile. This
kernel removes exactly that: every query keeps its own set, enforced by its own threshold mask. What
is shared here is only the POSITION assignment (a per-block base `d`, and one position per key slot),
never the key set. Naming follows dca_kernel.py, the existing precedent for a standalone module.

THE POSITION RULE (see PQSTREAM_KERNEL_PLAN.md §0)
--------------------------------------------------
Key blocks are absolute-aligned and visited BACKWARD from the query tile's diagonal:

    P_top   = min(tile's last query, N-1)            # last REAL query (ragged tiles!)
    nblk    = cdiv(P_top + 1, BN)
    start_n = (nblk - 1 - t) * BN                     # t = 0 .. nblk-1  (descending)
    hi_t    = min(P_top, start_n + BN - 1)            # binds only at t = 0
    o_j     = hi_t - j                                # natural offset inside the block, in [0, BN)
    dist(j) = d_t + o_j       d_0 = 0,  d_{t+1} = d_t + adv_t,  adv_t = max_i c_i

Rotation split -- do NOT reformulate:

    npq_i = qpos_i - P_top        (<= 0, block-INVARIANT -> Q rotated ONCE, outside the loop)
    npk_j = -(d_t + o_j)          (per key slot, shared across the tile's queries)
    distance = npq_i - npk_j = d_t + o_j - (P_top - qpos_i)

Putting d_t on the KEY side is what keeps npq block-invariant. On the query side it would force a Q
re-rotation every block (that is what `advance=self` would need, hence its deferral).

The GEMM can only express distances that factor as a_i - b_s (Q rotated once per row, K once per
slot), so a key's position may not depend on which query looks at it. dist(j) depends only on j and
the per-block scalar d_t -- it factors EXACTLY. A true per-query rank does not, and would need 32
GEMVs instead of one GEMM (the ~50 s/sequence path shared-tile beat by 20x).

ACCEPTED DISTORTION: within-block spread is BN but the base advances by max_i c_i <= BN, so
consecutive blocks' position ranges overlap by BN - max_i c_i and an older key can land closer than a
newer one. In exchange, INSIDE a block the geometry is exact -- adjacent keys stay 1 apart, as in
native RoPE. All compression is pushed to block boundaries, which makes BN the compression
granularity rather than a tuning constant.

WHAT IS GONE vs pq_exact: Votes, Bits, WVotes, Idx, NewposK, NewposQ, Mask, the whole
_build_sets_pq host function, k_max, the overflow truncation, and select_score. Per (query tile,
head) the entire HBM footprint is K read x2, V read x1, Out written x1, plus the tiny diagnostics.
Two kernel LAUNCHES can only communicate through HBM -- that is why pq_exact must materialise those
arrays; fusing lets the keep mask and `d` live and die in registers.

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
    from transformers.models.llama.modeling_llama import repeat_kv
except Exception:
    repeat_kv = None


# advance rule -> constexpr. `self` (per-query base) is deliberately absent: it makes d a vector,
# which cannot go on the key side, so it needs a per-block Q re-rotation and a different code path.
#
# HOST-SIDE ONLY. These names must NOT be referenced inside a @triton.jit body -- Triton >= 3.6
# rejects reads of plain module globals from a kernel. The kernel compares against the bare literals
# and says so in a comment; keep the two in sync if a third rule is ever added.
ADV_MAX = 0     # d += max_i c_i                          (the scheme)
ADV_FULL = 1    # d += slots the block occupies           (no compression -> native RoPE; control)
_ADV = {"max": ADV_MAX, "full": ADV_FULL}


# ──────────────────────────────────────────────────────────────────────────────
# MEASUREMENT-ONLY counters (do NOT affect kernel execution).
# k_max is gone, so overflow is gone -- but the accessor NAMES are kept, because callers probe
# them. What they report changes: the headline number is now the position SPAN (final d), i.e.
# "does the furthest repositioned key stay inside the 4096 RoPE window?", which is the entire
# justification for repositioning. Accumulated as lazy on-device tensors (no host sync during the
# run); get_overflow_stats() syncs once when read.
# ──────────────────────────────────────────────────────────────────────────────
def _fresh():
    return {"cells": 0, "span_sum": None, "span_max": None,
            "kept_sum": None, "rows": 0, "blocks": None, "skipped": None}


_STATS = {"prefill": _fresh(), "decode": _fresh()}


def reset_overflow_stats():
    """Zero the cumulative span/kept/skip counters (e.g. between benchmark runs)."""
    _STATS["prefill"] = _fresh()
    _STATS["decode"] = _fresh()


reset_span_stats = reset_overflow_stats          # honest alias


def _phase_stats(d):
    tot = d["cells"]
    ssum = float(d["span_sum"].item()) if d["span_sum"] is not None else 0.0
    smax = float(d["span_max"].item()) if d["span_max"] is not None else 0.0
    ksum = float(d["kept_sum"].item()) if d["kept_sum"] is not None else 0.0
    blk = int(d["blocks"].item()) if d["blocks"] is not None else 0
    skp = int(d["skipped"].item()) if d["skipped"] is not None else 0
    return {
        "total_cells": tot,                                  # (head, tile) cells processed
        "span_mean": (ssum / tot) if tot else 0.0,           # mean final d
        "span_max": smax,                                    # MAX final d -- must stay < 4096
        "kept_mean": (ksum / d["rows"]) if d["rows"] else 0.0,   # mean keys kept per query
        "blocks": blk,
        "skipped": skp,
        "skip_frac": (skp / blk) if blk else 0.0,            # the `s` of PQSTREAM_KERNEL_PLAN.md §3
        # kept for call-site compatibility with the pq_exact API; pq_stream cannot overflow.
        "overflow_cells": 0,
        "overflow_frac": 0.0,
    }


def get_overflow_stats():
    """Cumulative since last reset, per phase. NOTE: pq_stream has no k_max and therefore no
    overflow -- those two fields are hard 0 and exist only so callers written against the pq_exact
    API keep working. The numbers that matter here are `span_max` (must stay < the RoPE window,
    else the model extrapolates) and `skip_frac`. One host sync here, none during the run."""
    return {"prefill": _phase_stats(_STATS["prefill"]),
            "decode": _phase_stats(_STATS["decode"])}


def _acc(dst, key, val, reduce="sum"):
    if dst[key] is None:
        dst[key] = val
    elif reduce == "max":
        dst[key] = torch.maximum(dst[key], val)
    else:
        dst[key] = dst[key] + val


def _stats_accumulate(span, kept, blocks, skipped, phase="prefill"):
    """Measurement only. span (H,nt) fp32, kept (H,N) int32, blocks/skipped (H,nt) int32."""
    d = _STATS[phase]
    d["cells"] += span.numel()
    d["rows"] += kept.numel()
    _acc(d, "span_sum", span.double().sum())
    _acc(d, "span_max", span.max(), reduce="max")
    _acc(d, "kept_sum", kept.double().sum())
    _acc(d, "blocks", blocks.long().sum())
    _acc(d, "skipped", skipped.long().sum())


if HAVE_TRITON:

    # ══════════════════════════════════════════════════════════════════════════
    # THE FUSED KERNEL : pass 1 (exact per-query max) + pass 2 (select + position + attend)
    # in ONE launch, one program per (query tile, head).
    #
    # Pass 1 must exist and must be separate: the threshold `s >= m_i - tau` needs the EXACT global
    # row max. A running max (one-pass FlashAttention style) admits/rejects keys against a max that
    # is still wrong early in the scan -- settled empirically, ppl 31.7 vs 7.8 at 6k
    # (RESEARCH_SUMMARY.md §2.3). Pass 1 is K-only: no V, no accumulation.
    #
    # REGISTERS (PQSTREAM_KERNEL_PLAN.md §8): both Q rotations are hoisted (Option 1) AND the two
    # K rotations are SEQUENCED so krs dies before kra is born (Option 3, a precondition not an
    # option). Peak is at the attend-side K rotation, where k1,k2,cos,sin,kra1,kra2 are all live --
    # six K tiles, irreducible. ~206 regs/thread at BN=32/nw=4; BN=64 needs nw=8 (~310 otherwise,
    # which spills). num_warps is derived in pqstream_attention(), not passed in.
    # ══════════════════════════════════════════════════════════════════════════
    @triton.jit
    def _pqstream_kernel(
        Q, K, V, InvFreqA, InvFreqC, Tau, Out, Span, Kept, Blocks, Skipped,
        sm_scale, pos_scale_a, rope_scale_a, n_global, n_local,
        stride_qh, stride_qn, stride_qd,
        stride_kh, stride_kn, stride_kd,
        stride_vh, stride_vn, stride_vd,
        stride_oh, stride_on, stride_od,
        stride_sh, stride_st,
        stride_ch, stride_cn,
        N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr, HALF_DIM: tl.constexpr,
        ADV: tl.constexpr, SKIP: tl.constexpr,
    ):
        start_m = tl.program_id(0)          # query tile index
        off_h = tl.program_id(1)            # head (batch = 1)
        q_base = Q + off_h * stride_qh
        k_base = K + off_h * stride_kh
        v_base = V + off_h * stride_vh
        o_base = Out + off_h * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)       # row index into Q / Out
        qpos = offs_m                                            # prefill: q_pos_base == 0 (asserted)
        offs_h = tl.arange(0, HALF_DIM)
        offs_d = tl.arange(0, HEAD_DIM)
        invfa = tl.load(InvFreqA + offs_h)
        invfc = tl.load(InvFreqC + offs_h)
        tau = tl.load(Tau + off_h)
        row_ok = offs_m < N_CTX                                  # ragged last tile

        # CHECKLIST #1: P_top is the last REAL query in the tile, not the last ROW. On a ragged
        # final tile the padded rows contribute c_i = 0, so adv_0 is set by the last real query;
        # anchoring on a padded row would shift every distance in that tile down by the pad count.
        tile_last = start_m * BLOCK_M + BLOCK_M - 1
        P_top = tl.minimum(tile_last, N_CTX - 1)
        hi = P_top + 1                                           # causal key count for this tile
        nblk = tl.cdiv(hi, BLOCK_N)

        # ── prologue: raw Q loaded ONCE, rotated into BOTH geometries and hoisted (§8 Option 1) ──
        q1 = tl.load(q_base + offs_m[:, None] * stride_qn + offs_h[None, :] * stride_qd,
                     mask=row_ok[:, None], other=0.0)
        q2 = tl.load(q_base + offs_m[:, None] * stride_qn + (HALF_DIM + offs_h)[None, :] * stride_qd,
                     mask=row_ok[:, None], other=0.0)
        # select geometry: "pi" = base freqs at compressed positions, "yarn" = yarn freqs + mscale
        angqs = (qpos[:, None].to(tl.float32) * pos_scale_a) * invfa[None, :]
        cosqs = tl.cos(angqs) * rope_scale_a
        sinqs = tl.sin(angqs) * rope_scale_a
        qsel1 = q1 * cosqs - q2 * sinqs
        qsel2 = q2 * cosqs + q1 * sinqs
        # attend geometry: BASE RoPE at npq_i = qpos_i - P_top (<= 0), block-invariant. CHECKLIST #17
        npq = (qpos - P_top).to(tl.float32)
        angqa = npq[:, None] * invfc[None, :]
        cosqa = tl.cos(angqa)
        sinqa = tl.sin(angqa)
        qatt1 = q1 * cosqa - q2 * sinqa
        qatt2 = q2 * cosqa + q1 * sinqa

        # ══ PASS 1 : EXACT per-query max over all causal keys, select geometry, forward+aligned ══
        m = tl.full([BLOCK_M], float("-inf"), tl.float32)
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_ok = offs_n < N_CTX
            k1 = tl.load(k_base + offs_n[:, None] * stride_kn + offs_h[None, :] * stride_kd,
                         mask=n_ok[:, None], other=0.0)
            k2 = tl.load(k_base + offs_n[:, None] * stride_kn + (HALF_DIM + offs_h)[None, :] * stride_kd,
                         mask=n_ok[:, None], other=0.0)
            angk = (offs_n[:, None].to(tl.float32) * pos_scale_a) * invfa[None, :]
            cosk = tl.cos(angk) * rope_scale_a
            sink = tl.sin(angk) * rope_scale_a
            kr1 = k1 * cosk - k2 * sink
            kr2 = k2 * cosk + k1 * sink
            qk = (tl.dot(qsel1, tl.trans(kr1)) + tl.dot(qsel2, tl.trans(kr2))) * sm_scale
            causal = qpos[:, None] >= offs_n[None, :]
            qk = tl.where(causal & n_ok[None, :], qk, float("-inf"))
            m = tl.maximum(m, tl.max(qk, 1))

        # ══ PASS 2 : select + position + attend, DESCENDING blocks. Body order is load-bearing ══
        d = 0.0                                          # running base (loop-carried scalar)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
        mR = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        kept_rows = tl.zeros([BLOCK_M], tl.int32)
        # Counted as "blocks USED", then subtracted at the end. Tracking skips in an `else` branch
        # would put a second loop-carried variable on the other side of a runtime conditional; only
        # the taken branch mutates state this way, which is the shape Triton lowers most reliably.
        n_used = 0.0

        # forced-local lower bound, matching _build_sets_pq's local_block exactly
        loc_lo = start_m * BLOCK_M - n_local + 1
        loc_hi = start_m * BLOCK_M + BLOCK_M - 1

        # CHECKLIST #15: forward `t` with a computed descending start_n, NOT a negative-step range.
        for t in range(0, nblk):
            start_n = (nblk - 1 - t) * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_ok = offs_n < N_CTX
            # CHECKLIST #2: aligned blocks, offset measured from hi_t. min() binds only at t == 0.
            hi_t = tl.minimum(P_top, start_n + BLOCK_N - 1)
            o = (hi_t - offs_n).to(tl.float32)           # < 0 only for lanes above P_top (masked)

            # ── 1. load K once; k1,k2 must survive BOTH rotations (CHECKLIST #11) ──
            k1 = tl.load(k_base + offs_n[:, None] * stride_kn + offs_h[None, :] * stride_kd,
                         mask=n_ok[:, None], other=0.0)
            k2 = tl.load(k_base + offs_n[:, None] * stride_kn + (HALF_DIM + offs_h)[None, :] * stride_kd,
                         mask=n_ok[:, None], other=0.0)

            # ── 2-3. select rotation + select GEMM ──
            angks = (offs_n[:, None].to(tl.float32) * pos_scale_a) * invfa[None, :]
            cosks = tl.cos(angks) * rope_scale_a
            sinks = tl.sin(angks) * rope_scale_a
            krs1 = k1 * cosks - k2 * sinks
            krs2 = k2 * cosks + k1 * sinks
            qk_sel = (tl.dot(qsel1, tl.trans(krs1)) + tl.dot(qsel2, tl.trans(krs2))) * sm_scale

            # ── 4. per-query keep mask. CHECKLIST #4 (forced counted), #5 (AND causal, never OR
            #      over it -- forcing a non-causal key would leak the future), #6 (padded rows out:
            #      without it m_i = -inf and qk = -inf make (-inf >= -inf) True) ──
            causal = qpos[:, None] >= offs_n[None, :]
            forced = ((offs_n < n_global) | ((offs_n >= loc_lo) & (offs_n <= loc_hi)))
            keep = (((qk_sel >= (m[:, None] - tau)) | forced[None, :])
                    & causal & row_ok[:, None] & n_ok[None, :])
            c_i = tl.sum(keep.to(tl.int32), 1)
            # `adv` is kept fp32 in BOTH branches so the loop-carried type of `d` never changes
            # across iterations (a Triton loop-carried scalar must keep one type). ADV is a
            # constexpr, so only one branch is ever traced.
            #
            # ADV_FULL advances by the number of position slots this block actually OCCUPIES,
            # hi_t - start_n + 1, NOT by BLOCK_N. They differ only on the first (topmost) block,
            # which spans P_top - start_n + 1 <= BN slots -- but advancing by BN there shifts every
            # older block by (BN - 1 - P_top mod BN) and breaks the native-RoPE oracle for every
            # tile whose P_top+1 is not a multiple of BN. At tau = inf ADV_MAX agrees with this
            # exactly: max_i c_i is attained by the LAST real query, which sees hi_t - start_n + 1
            # causal keys on the top block and all BN on every block below it.
            #
            # The literal 1 is ADV_FULL, spelled out on purpose: Triton >= 3.6 raises
            # NameError("Cannot access global variable ... from within @jit'ed function") for any
            # plain module-level global read inside a kernel, and ADV_FULL is a plain int. Wrapping
            # it in tl.constexpr would also work, but a kernel that reaches outside itself for a
            # value is the thing that broke -- keep the comparison self-contained. (The env escape
            # TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 that the error suggests is documented as not
            # supported forever; do not rely on it.)
            if ADV == 1:                                     # 1 == ADV_FULL
                adv = (hi_t - start_n + 1).to(tl.float32)
            else:                                            # 0 == ADV_MAX
                adv = tl.max(c_i).to(tl.float32)
            # krs1, krs2, qk_sel are dead from here -- this is §8 Option 3, the sequencing that
            # keeps the six-K-tile peak from becoming eight.

            # CHECKLIST #13: adv == 0 <=> no query kept anything <=> the attend GEMM is a no-op AND
            # d must not advance. Both follow from the same condition.
            # ONE constexpr condition only. `SKIP and ADV == ADV_MAX` would mix a constexpr bool with
            # a constexpr comparison in a Python `and`, which is more trace-time machinery than this
            # needs; under ADV_FULL adv = hi_t - start_n + 1 >= 1 always (hi_t >= start_n for every
            # in-range block), so the runtime test is simply never taken there.
            if SKIP:
                do_blk = adv > 0.0
            else:
                do_blk = True
            if do_blk:
                # ── 6. positions: npk uses d BEFORE the update (CHECKLIST #3) ──
                npk = -(d + o)
                # ── 7. attend rotation, BASE RoPE, from the SAME k1,k2 -- no reload ──
                angka = npk[:, None] * invfc[None, :]
                coska = tl.cos(angka)
                sinka = tl.sin(angka)
                kra1 = k1 * coska - k2 * sinka
                kra2 = k2 * coska + k1 * sinka
                # ── 8-9. attend GEMM + online softmax (CHECKLIST #18 sm_scale, #12 keep survives) ──
                qk_att = (tl.dot(qatt1, tl.trans(kra1)) + tl.dot(qatt2, tl.trans(kra2))) * sm_scale
                qk_att = tl.where(keep, qk_att, float("-inf"))

                mR_new = tl.maximum(mR, tl.max(qk_att, 1))
                # CHECKLIST #7: a query that keeps nothing in this block hits -inf on its first block
                mR_safe = tl.where(mR_new == float("-inf"), 0.0, mR_new)
                p = tl.exp(qk_att - mR_safe[:, None])
                alpha = tl.exp(mR - mR_safe)
                l_i = l_i * alpha + tl.sum(p, 1)
                vv = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                             mask=n_ok[:, None], other=0.0)
                acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)     # CHECKLIST #19
                mR = mR_new
                kept_rows += c_i
                d += adv
                n_used += 1.0

        # CHECKLIST #8: l_i > 0 is guaranteed -- every real query's own position lies in its forced
        # local window, so it always keeps at least one key.
        acc = acc / l_i[:, None]
        tl.store(o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
                 acc.to(Out.dtype.element_ty), mask=row_ok[:, None])
        tl.store(Span + off_h * stride_sh + start_m * stride_st, d)
        tl.store(Blocks + off_h * stride_sh + start_m * stride_st, nblk.to(tl.float32))
        tl.store(Skipped + off_h * stride_sh + start_m * stride_st, nblk.to(tl.float32) - n_used)
        tl.store(Kept + off_h * stride_ch + offs_m * stride_cn, kept_rows, mask=row_ok)

    # ══════════════════════════════════════════════════════════════════════════
    # DECODE-SELECT KERNEL : single query -> per-key scores, under the SELECT geometry.
    # Copied verbatim from shared_selection_triton_perquery.py (no bits: one query, no packing).
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
        kb = tl.program_id(0)
        off_h = tl.program_id(1)
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

        score = tl.sum(qr1[None, :] * kr1 + qr2[None, :] * kr2, axis=1) * sm_scale
        tl.store(Scores + off_h * stride_sh + offs_n * stride_sn, score, mask=offs_n < M)

    # ══════════════════════════════════════════════════════════════════════════
    # DECODE-ATTEND KERNEL : gather + attend at the streamed positions.
    # This is pq_exact's _gather_attn_pq_kernel MINUS the per-query bitmask -- with one query there
    # is no mask to apply. The gather survives HERE (and only here): prefill streams, but a single
    # query's kept set really is sparse and worth gathering.
    # ══════════════════════════════════════════════════════════════════════════
    @triton.jit
    def _decode_attend_kernel(
        Q, Kc, Vc, InvFreq, Idx, NewposK, NewposQ, Out,
        sm_scale, q_pos_base,
        stride_qh, stride_qn, stride_qd,
        stride_kh, stride_kn, stride_kd,
        stride_vh, stride_vn, stride_vd,
        stride_ih, stride_it, stride_ik,
        stride_nkh, stride_nkt, stride_nkk,
        stride_nqh, stride_nqt, stride_nqm,
        stride_oh, stride_on, stride_od,
        N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, K_MAX: tl.constexpr,
        HEAD_DIM: tl.constexpr, HALF_DIM: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_h = tl.program_id(1)
        q_base = Q + off_h * stride_qh
        k_base = Kc + off_h * stride_kh
        v_base = Vc + off_h * stride_vh
        o_base = Out + off_h * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        qpos = q_pos_base + offs_m
        offs_h = tl.arange(0, HALF_DIM)
        offs_d = tl.arange(0, HEAD_DIM)
        invf = tl.load(InvFreq + offs_h)

        npq = tl.load(NewposQ + off_h * stride_nqh + start_m * stride_nqt
                      + tl.arange(0, BLOCK_M) * stride_nqm)
        q1 = tl.load(q_base + offs_m[:, None] * stride_qn + offs_h[None, :] * stride_qd)
        q2 = tl.load(q_base + offs_m[:, None] * stride_qn + (HALF_DIM + offs_h)[None, :] * stride_qd)
        angq = npq[:, None] * invf[None, :]
        cosq = tl.cos(angq)
        sinq = tl.sin(angq)
        qr1 = q1 * cosq - q2 * sinq
        qr2 = q2 * cosq + q1 * sinq

        mR = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

        for start_k in range(0, K_MAX, BLOCK_K):
            slots = start_k + tl.arange(0, BLOCK_K)
            idx = tl.load(Idx + off_h * stride_ih + start_m * stride_it + slots * stride_ik,
                          mask=slots < K_MAX, other=N_CTX)
            valid = idx < N_CTX
            npk = tl.load(NewposK + off_h * stride_nkh + start_m * stride_nkt + slots * stride_nkk,
                          mask=valid, other=0.0)
            k1 = tl.load(k_base + idx[:, None] * stride_kn + offs_h[None, :] * stride_kd,
                         mask=valid[:, None], other=0.0)
            k2 = tl.load(k_base + idx[:, None] * stride_kn + (HALF_DIM + offs_h)[None, :] * stride_kd,
                         mask=valid[:, None], other=0.0)
            angk = npk[:, None] * invf[None, :]
            cosk = tl.cos(angk)
            sink = tl.sin(angk)
            kr1 = k1 * cosk - k2 * sink
            kr2 = k2 * cosk + k1 * sink

            qk = (tl.dot(qr1, tl.trans(kr1)) + tl.dot(qr2, tl.trans(kr2))) * sm_scale
            keep = (qpos[:, None] >= idx[None, :]) & valid[None, :]
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
                 acc.to(Out.dtype.element_ty))


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration (prefill) — ONE launch
# ──────────────────────────────────────────────────────────────────────────────
def pqstream_attention(q, k, v, tau, inv_freq_a, inv_freq_c, pos_scale_a=1.0, rope_scale_a=1.0,
                       sm_scale=None, n_global=32, n_local=32, tile=32, block_n=32,
                       pos_advance="max", q_pos_base=0, use_skip=True, num_warps=None):
    """q,k,v: (1,H,N,D) RAW. Two-geometry split, identical to pq_exact so the calibrated tau tables
    transfer unchanged:
      select : inv_freq_a at pos*pos_scale_a, cos/sin scaled by rope_scale_a
               ("pi" -> base freqs + compression; "yarn" -> yarn freqs + mscale, pos_scale_a=1)
      attend : inv_freq_c (BASE RoPE) at the streamed positions, rope_scale = 1.
    Returns (out, span, kept, blocks, skipped)."""
    assert HAVE_TRITON, "Triton not available"
    Z, H, N, D = q.shape
    assert Z == 1
    assert tile == 32, "BLOCK_M is fixed at 32 by the tile convention"
    # Prefill only ever runs at past_len == 0 (see install_pqstream_forward), and the kernel indexes
    # keys by absolute position with qpos == offs_m. Asserting it beats carrying a q_pos_base that
    # is silently wrong the day someone adds chunked prefill.
    assert q_pos_base == 0, f"pqstream prefill expects q_pos_base == 0, got {q_pos_base}"
    # PQSTREAM_KERNEL_PLAN.md §5 oracle 2 / CHECKLIST #16: every sub-diagonal block must be fully
    # causal for all queries, so that max_i c_i == BLOCK_N at tau = inf.
    assert block_n >= tile - 1, (
        f"block_n={block_n} < tile-1={tile - 1}: the tau=inf oracle would not hold")
    adv = _ADV[pos_advance]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    nt = (N + tile - 1) // tile
    qh, kh, vh = q[0].contiguous(), k[0].contiguous(), v[0].contiguous()
    tau = tau.float().contiguous()
    invfa = inv_freq_a.float().contiguous()
    invfc = inv_freq_c.float().contiguous()

    out = torch.empty((H, N, D), dtype=q.dtype, device=q.device)
    span = torch.zeros((H, nt), dtype=torch.float32, device=q.device)
    blocks = torch.zeros((H, nt), dtype=torch.float32, device=q.device)
    skipped = torch.zeros((H, nt), dtype=torch.float32, device=q.device)
    kept = torch.zeros((H, N), dtype=torch.int32, device=q.device)

    # num_warps sets how many 32-thread warps staff one program, and therefore how finely each tile
    # is spread across threads: a (BN,64) fp32 tile costs BN*64/(32*num_warps) registers PER THREAD,
    # so doubling num_warps halves per-thread register demand. Derived here so the rule lives in ONE
    # place and no SLURM script can get it wrong; `num_warps=` overrides it for the V5 sweep.
    #
    # MEASURED on A100 / Triton 3.6 (job 274761), NOT the §8 estimate, which was badly low:
    #     BN=32 nw=4 -> n_regs=243  n_spills=8   (estimate said ~206)
    #     BN=64 nw=8 -> n_regs=246  n_spills=8   (estimate said ~158)
    # Both sit within 12 of the 255/thread ceiling, i.e. ptxas is using everything available and
    # parking the remainder in local memory. 8 bytes is 2 registers of per-thread scratch, which
    # stays L1-resident -- a sign the kernel is at the edge of the register file, not a bug.
    if num_warps is None:
        num_warps = 8 if block_n >= 64 else 4

    _pqstream_kernel[(nt, H)](
        qh, kh, vh, invfa, invfc, tau, out, span, kept, blocks, skipped,
        sm_scale, float(pos_scale_a), float(rope_scale_a), int(n_global), int(n_local),
        qh.stride(0), qh.stride(1), qh.stride(2),
        kh.stride(0), kh.stride(1), kh.stride(2),
        vh.stride(0), vh.stride(1), vh.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        span.stride(0), span.stride(1),
        kept.stride(0), kept.stride(1),
        N,
        BLOCK_M=tile, BLOCK_N=block_n, HEAD_DIM=D, HALF_DIM=D // 2,
        ADV=adv, SKIP=bool(use_skip),
        num_warps=num_warps, num_stages=1,
    )
    _stats_accumulate(span, kept, blocks, skipped, phase="prefill")
    return out.unsqueeze(0), span, kept, blocks, skipped


# ──────────────────────────────────────────────────────────────────────────────
# GOLDEN REFERENCE (pure torch, independent of the Triton path). O(nt * N^2) — small N only.
# Same rule: exact-max threshold under the select geometry, descending aligned blocks, offset
# inside the block, base advancing by max_i c_i (or BN), attend under base RoPE.
# ──────────────────────────────────────────────────────────────────────────────
def _rope_rotate(x, pos, inv_freq, rope_scale):
    """Rotate x (..., D) by RoPE at `pos` (broadcastable to x.shape[:-1]); half-split (Llama) form."""
    half = x.shape[-1] // 2
    ang = pos[..., None].to(torch.float32) * inv_freq.to(torch.float32)
    cos = torch.cos(ang) * rope_scale
    sin = torch.sin(ang) * rope_scale
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


@torch.no_grad()
def pqstream_attention_torch(q, k, v, tau, inv_freq_a, inv_freq_c, pos_scale_a=1.0,
                             rope_scale_a=1.0, sm_scale=None, n_global=32, n_local=32,
                             tile=32, block_n=32, pos_advance="max"):
    """Reference forward. q,k,v (1,H,N,D). Returns (1,H,N,D) in q.dtype."""
    Z, H, N, D = q.shape
    assert Z == 1
    T, BN = tile, block_n
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
        qpos = torch.arange(r0, r1, device=dev)
        P_top = min(t * T + T - 1, N - 1)
        hi = P_top + 1
        nblk = (hi + BN - 1) // BN
        ks = torch.arange(hi, device=dev)
        causal = qpos[:, None] >= ks[None, :]                                  # (m, hi)

        # ── exact per-query max under the select geometry (pass 1) ──
        qs = _rope_rotate(qf[:, r0:r1], qpos.float() * pos_scale_a, invfa, rope_scale_a)
        kks = _rope_rotate(kf[:, :hi], ks.float() * pos_scale_a, invfa, rope_scale_a)
        s = torch.einsum("hmd,hnd->hmn", qs, kks) * sm_scale
        s = torch.where(causal[None], s, torch.full_like(s, NEG))
        mmax = s.max(-1).values                                                # (H, m)

        forced = (ks < n_global) | ((ks >= r0 - n_local + 1) & (ks <= r0 + T - 1))
        keep = (((s >= (mmax[..., None] - tau[:, None, None])) | forced[None, None, :])
                & causal[None])                                                # (H, m, hi)

        # ── descending blocks: positions + per-query masked attention (pass 2) ──
        # blk_used mirrors the kernel's `adv == 0` skip. It is redundant (a skipped block has
        # keep all-False anyway, so `sel` below would mask it) but it is kept so the reference
        # fails loudly if that invariant ever stops holding.
        blk_used = torch.zeros((H, hi), dtype=torch.bool, device=dev)
        npk_all = torch.zeros((H, hi), device=dev, dtype=torch.float32)
        for h in range(H):
            d = 0.0
            for bt in range(nblk):
                start_n = (nblk - 1 - bt) * BN
                end_n = min(start_n + BN, hi)
                hi_t = min(P_top, start_n + BN - 1)
                o = (hi_t - torch.arange(start_n, end_n, device=dev)).float()
                c_i = keep[h, :, start_n:end_n].sum(-1)                        # (m,)
                # `full` advances by the slots the block OCCUPIES (hi_t-start_n+1), not by BN --
                # see the kernel comment; using BN breaks the native-RoPE oracle on ragged P_top.
                adv = (float(hi_t - start_n + 1) if pos_advance == "full"
                       else float(c_i.max().item()))
                if adv > 0:
                    npk_all[h, start_n:end_n] = -(d + o)                       # d BEFORE update
                    blk_used[h, start_n:end_n] = True
                    d += adv

        krc = _rope_rotate(kf[:, :hi], npk_all, invfc, 1.0)                    # (H, hi, D)
        npq = (qpos - P_top).float()[None].expand(H, r1 - r0)
        qrc = _rope_rotate(qf[:, r0:r1], npq, invfc, 1.0)                      # (H, m, D)
        sc = torch.einsum("hmd,hnd->hmn", qrc, krc) * sm_scale
        sel = keep & blk_used[:, None, :]
        sc = torch.where(sel, sc, torch.full_like(sc, NEG))
        a = torch.softmax(sc, dim=-1)
        out[:, r0:r1] = torch.einsum("hmn,hnd->hmd", a, vf[:, :hi])
    return out.unsqueeze(0).to(q.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# DECODE — the same position RULE as prefill (PQSTREAM_KERNEL_PLAN.md §4).
#
# This is the one place pq_exact's design is deliberately NOT reused. pq_exact decodes at global
# dense ranks while its prefill uses tile-union ranks -- two different rules, so the geometry jumps
# at the prefill->decode boundary. Here both ends run the identical rule: aligned blocks descending
# from the query, natural offset inside the block, base advancing by a kept count.
#
# BE PRECISE ABOUT WHAT THAT DOES AND DOES NOT FIX. The rule discontinuity is gone. A residual seam
# in the ADVANCE remains and is not removable here: prefill advances by max_i c_i over the 32-query
# tile, decode has one query and can only advance by its own c. The two coincide exactly when the
# query is the tile's argmax and diverge by (max_i c_i - c_i) per block otherwise -- i.e. bounded by
# the same tile-sharing term the whole design trades on, not by a second, independent mismatch.
# advance=self (deferred, see PQSTREAM_KERNEL_PLAN.md §2) would close it completely, since prefill
# would then also advance per query; that is an argument for eventually building it.
#
# With one query max_i c_i == c, so `max` and `full` collapse to one decode path.
# ──────────────────────────────────────────────────────────────────────────────
def _decode_scores_kernel(q, K, inv_freq_a, q_pos, pos_scale_a, rope_scale_a, sm_scale, block_n=64):
    """Per-key scores (H,M) for one query via Triton. q:(H,D), K:(H,M,D)."""
    assert HAVE_TRITON, "Triton not available"
    H, M, D = K.shape
    dev = K.device
    q = q.contiguous(); K = K.contiguous()
    invf = inv_freq_a.to(dev).float().contiguous()
    scores = torch.empty((H, M), dtype=torch.float32, device=dev)
    _decode_select_kernel[(triton.cdiv(M, block_n), H)](
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
def _decode_build_sets_pqs(q, K, tau, inv_freq_a, q_pos, pos_scale_a, rope_scale_a, sm_scale,
                           n_global, n_local, block_n, scores=None, block_k=32):
    """Per-query selection + STREAMED positions for one query (host, torch), vectorised over heads.
    q:(H,D), K:(H,M,D), M = q_pos+1. Returns idx (H,kbuf) long (sentinel=M), newpos_k (H,kbuf) fp32,
    newpos_q (H,) fp32.

    No k_max: the buffer is M padded up to a multiple of block_k. One query at 12k is a 1.5 MB
    (H,M) int32 -- capping it would reintroduce exactly the truncation pq_stream exists to remove."""
    H, M, D = K.shape
    dev = K.device
    BN = block_n
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
    keep = score >= (m - tau)[:, None]                                    # (H,M)

    forced = torch.zeros(M, dtype=torch.bool, device=dev)
    forced[:min(n_global, M)] = True                                      # global / sink
    forced[max(0, q_pos - n_local + 1): q_pos + 1] = True                 # local window
    forced[q_pos] = True                                                  # always attend to self
    kp = keep | forced[None]                                              # (H,M) final set

    # ── streamed positions: aligned blocks, descending, base = kept count of MORE RECENT blocks ──
    nblk = (M + BN - 1) // BN
    pad = nblk * BN - M
    kpp = torch.nn.functional.pad(kp, (0, pad), value=False).view(H, nblk, BN)
    c = kpp.sum(-1)                                                       # (H,nblk) kept per block
    # exclusive reverse cumsum: d[b] = sum of c[b'] for b' > b (blocks nearer the query)
    d = c.flip(-1).cumsum(-1).flip(-1) - c                                # (H,nblk)

    b = torch.arange(nblk, device=dev)
    hi_b = torch.minimum(torch.full_like(b, q_pos), b * BN + BN - 1)      # (nblk,)
    j = b[:, None] * BN + torch.arange(BN, device=dev)[None, :]           # (nblk,BN) absolute key
    o = (hi_b[:, None] - j).float()                                       # (nblk,BN)
    dist = d[:, :, None].float() + o[None]                                # (H,nblk,BN)
    dist = dist.reshape(H, nblk * BN)[:, :M]
    newpos_k_full = -dist                                                 # (H,M)

    # ── compact the kept set (sort is fine: one query, M is small) ──
    # kbuf is passed to _decode_attend_kernel as the CONSTEXPR K_MAX, so every distinct value forces
    # a fresh Triton COMPILE. Rounding to block_k would give a new value every block_k tokens of
    # generation -- dozens of ~20 s compiles across a RULER run, all of it wall-clock inside the
    # decode loop. pq_exact never hit this because its K_MAX was the fixed k_max=4096; removing the
    # budget is what exposed it. Quantise to 1024 (a multiple of both block_k=32 and 64) so the whole
    # 1k-16k range yields ~16 distinct values, compiled once each and cached for the rest of the run.
    # The extra slots hold the sentinel M and are masked out, so this costs a little work, not
    # correctness.
    _Q = 1024
    kbuf = max(_Q, ((M + _Q - 1) // _Q) * _Q)
    assert kbuf % block_k == 0, f"kbuf={kbuf} must be a multiple of block_k={block_k}"
    kj = torch.arange(M, device=dev)
    masked_pos = torch.where(kp, kj[None].expand(H, M),
                             torch.full((H, M), M, device=dev, dtype=torch.long))
    idx_sorted = masked_pos.sort(-1).values
    if idx_sorted.shape[-1] < kbuf:
        pad_i = torch.full((H, kbuf - idx_sorted.shape[-1]), M, device=dev, dtype=idx_sorted.dtype)
        idx_sorted = torch.cat([idx_sorted, pad_i], dim=-1)
    else:
        idx_sorted = idx_sorted[:, :kbuf]
    idx_clamp = idx_sorted.clamp(max=M - 1)
    newpos_k = torch.gather(newpos_k_full, -1, idx_clamp)
    newpos_k = torch.where(idx_sorted < M, newpos_k, torch.zeros_like(newpos_k))
    newpos_q = torch.zeros((H,), device=dev, dtype=torch.float32)          # npq = 0 at decode

    # span per head = furthest streamed distance among the keys actually ATTENDED. Taking the max
    # over all of `dist` would report the oldest key's slot whether or not it was kept, which
    # over-states the span and would make the "< RoPE window" check fire spuriously.
    _stats_accumulate(torch.where(kp, dist, torch.zeros_like(dist)).max(-1).values[:, None].float(),
                      kp.sum(-1, dtype=torch.int32),
                      torch.full((H, 1), float(nblk), device=dev),
                      (c == 0).sum(-1, keepdim=True).float(),
                      phase="decode")
    return idx_sorted, newpos_k.float(), newpos_q


@torch.no_grad()
def _decode_attend_torch(q, K, V, idx, newpos_k, newpos_q, inv_freq_c, sm_scale):
    """Golden torch attend over the selected set, base RoPE at the streamed positions.
    q:(H,D), K,V:(H,M,D), idx/newpos_k:(H,kbuf). -> (H,D)."""
    H, M, D = K.shape
    inv_freq_c = inv_freq_c.to(K.device).float()
    outs = []
    for h in range(H):
        valid = idx[h] < M
        kk = idx[h][valid]
        npk = newpos_k[h][valid]
        qr = _rope_rotate(q[h], newpos_q[h], inv_freq_c, 1.0)
        kr = _rope_rotate(K[h, kk], npk, inv_freq_c, 1.0)
        s = (qr[None, :] * kr).sum(-1) * sm_scale
        a = torch.softmax(s, dim=-1)
        outs.append((a[:, None] * V[h, kk].float()).sum(0))
    return torch.stack(outs, dim=0).to(q.dtype)


def _decode_attend_triton(q, K, V, idx, newpos_k, newpos_q, inv_freq_c, sm_scale,
                          q_pos, block_m=16, block_k=32):
    """Attend via _decode_attend_kernel: the single query is padded to `block_m` rows (only row 0
    real), N_CTX = cache length M, rope_scale = 1. Padded rows are computed then discarded."""
    assert HAVE_TRITON, "Triton not available"
    H, M, D = K.shape
    dev = K.device
    dt = q.dtype
    kbuf = idx.shape[-1]
    qh = torch.zeros((H, block_m, D), dtype=dt, device=dev); qh[:, 0, :] = q
    kh = K.contiguous(); vh = V.contiguous()
    Idx = idx.to(torch.int32).unsqueeze(1).contiguous()
    NewposK = newpos_k.float().unsqueeze(1).contiguous()
    NewposQ = torch.zeros((H, 1, block_m), dtype=torch.float32, device=dev)
    NewposQ[:, 0, 0] = newpos_q
    Out = torch.empty((H, block_m, D), dtype=dt, device=dev)
    invfc = inv_freq_c.to(dev).float().contiguous()
    _decode_attend_kernel[(1, H)](
        qh, kh, vh, invfc, Idx, NewposK, NewposQ, Out,
        sm_scale, int(q_pos),
        qh.stride(0), qh.stride(1), qh.stride(2),
        kh.stride(0), kh.stride(1), kh.stride(2),
        vh.stride(0), vh.stride(1), vh.stride(2),
        Idx.stride(0), Idx.stride(1), Idx.stride(2),
        NewposK.stride(0), NewposK.stride(1), NewposK.stride(2),
        NewposQ.stride(0), NewposQ.stride(1), NewposQ.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        M, BLOCK_M=block_m, BLOCK_K=block_k, K_MAX=kbuf, HEAD_DIM=D, HALF_DIM=D // 2,
        num_warps=4, num_stages=1,
    )
    return Out[:, 0, :]


# ──────────────────────────────────────────────────────────────────────────────
# Model install (prefill + decode)
# ──────────────────────────────────────────────────────────────────────────────
def install_pqstream_forward(model, context_window=4096, n_global=32, n_local=32,
                             tile=32, block_n=32, block_k=32, max_len=None,
                             decode_backend="torch", select_backend="torch",
                             prefill_mode="stream", select_geom="pi", apply_mscale=True,
                             pos_advance="max", use_skip=True, num_warps=None,
                             # ---- accepted and IGNORED, for harness compatibility ----
                             pct=None, pct_decode=None, k_max=None, select_score=None,
                             measure=False, **_unused):
    """Patch LlamaAttention.forward to route through the PQ-STREAM kernel.

    select_geom: geometry used to SELECT (attend is ALWAYS base RoPE at streamed positions):
      "pi"   -> plain-loaded model; base inv_freq at compressed positions (pos_scale=min(1,W/L)).
      "yarn" -> YARN-LOADED model required; yarn inv_freq at raw positions; apply_mscale toggles the
                YaRN temperature in the SELECT geometry only.
    pos_advance: "max" (d += max_i c_i, the scheme) | "full" (d += block_n, no compression -> the
                 control that isolates repositioning from sparsity).

    pct / pct_decode / k_max / select_score / measure are accepted and IGNORED. run_ruler.py:128 and
    run_kernel_benchmark.py:180 build `inst` unconditionally, so a narrower signature would raise
    TypeError at install time -- and pq_stream has no budget to cap and no ranking to choose."""
    import transformers.models.llama.modeling_llama as ml
    assert select_geom in ("pi", "yarn"), f"select_geom must be pi|yarn, got {select_geom!r}"
    assert pos_advance in _ADV, f"pos_advance must be max|full, got {pos_advance!r}"
    if prefill_mode == "shared":            # back-compat: nothing is shared here
        prefill_mode = "stream"
    if k_max is not None:
        print(f"pq_stream: k_max={k_max} IGNORED (no budget — sets are exact at every length)",
              flush=True)

    rotary = model.model.rotary_emb
    inv_freq_rot = rotary.inv_freq.detach().float()
    attn_scaling = float(getattr(rotary, "attention_scaling", 1.0))

    # base RoPE inv_freq for the attend geometry (and for "pi" selection). Under "yarn" the rotary
    # freqs are the YaRN ones, so recompute base theta^{-2i/D} from config, as the hybrid kernel does.
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
            print("WARNING [pq_stream]: rotary.attention_scaling == 1.0 — the model does NOT look "
                  "yarn-loaded (mscale should be >1). Load it with rope_scaling rope_type='yarn'.",
                  flush=True)
        inv_freq_a = inv_freq_rot
        rope_scale_a = attn_scaling if apply_mscale else 1.0
    else:
        inv_freq_a = inv_freq_base
        rope_scale_a = 1.0
    assert inv_freq_base.numel() == inv_freq_a.numel(), \
        f"inv_freq length mismatch: {inv_freq_base.numel()} vs {inv_freq_a.numel()}"
    assert block_n >= tile - 1, f"block_n={block_n} must be >= tile-1={tile - 1}"

    print(f"pq_stream: select={select_geom}(rope_scale={rope_scale_a:.4f}), attend=base RoPE"
          f"(theta={theta:.0f}) at STREAMED positions; advance={pos_advance}, BN={block_n}, "
          f"num_warps={num_warps if num_warps else (8 if block_n >= 64 else 4)}"
          f"{'' if num_warps else ' (derived)'}, no k_max", flush=True)

    for li, layer in enumerate(model.model.layers):
        sa = layer.self_attn
        dev = sa.q_proj.weight.device
        sa.pqs_inv_freq_a = inv_freq_a.to(dev)          # select-geometry freqs
        sa.pqs_rope_scale_a = rope_scale_a              # mscale (yarn) or 1.0 (pi)
        sa.pqs_inv_freq_c = inv_freq_base.to(dev)       # base RoPE freqs (attend)
        sa.pqs_geom = select_geom
        sa.pqs_W = context_window
        sa.pqs_ng = n_global; sa.pqs_nl = n_local
        sa.pqs_tile = tile
        sa.pqs_block_n = block_n; sa.pqs_block_k = block_k
        sa.pqs_layer_idx = getattr(sa, "layer_idx", li)
        sa.pqs_max_len = max_len                        # target length for the frozen pos_scale (pi)
        sa.pqs_pos_scale_a = None                       # set once at prefill, reused at decode
        sa.pqs_advance = pos_advance
        sa.pqs_use_skip = bool(use_skip)
        sa.pqs_num_warps = num_warps            # None -> derived (8 if BN>=64 else 4)
        sa.pqs_decode_backend = decode_backend
        sa.pqs_select_backend = select_backend
        sa.pqs_prefill_mode = prefill_mode

    def pqs_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, **kwargs):
        B, N, _ = hidden_states.shape
        hs = (B, N, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hs).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hs).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hs).transpose(1, 2)

        # ── KV cache: store RAW k,v (the kernel rotates internally); GQA-repeat AFTER caching ──
        cache = past_key_values if past_key_values is not None else kwargs.get("past_key_value", None)
        past_len = 0
        if cache is not None:
            try:
                past_len = int(cache.get_seq_length(self.pqs_layer_idx))
            except TypeError:
                past_len = int(cache.get_seq_length())
            k, v = cache.update(k, v, self.pqs_layer_idx)

        if repeat_kv is not None:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # ── frozen select pos_scale: pi -> min(1, W/L) once at prefill; yarn -> raw positions ──
        if past_len == 0:
            if self.pqs_geom == "yarn":
                self.pqs_pos_scale_a = 1.0
            else:
                L_target = self.pqs_max_len if self.pqs_max_len is not None else N
                self.pqs_pos_scale_a = min(1.0, self.pqs_W / float(L_target))

        if past_len == 0 and getattr(self, "pqs_prefill_mode", "stream") == "stream":
            # ── PREFILL: one fused launch ──
            out, span, kept, blocks, skipped = pqstream_attention(
                q, k, v, self.tau_head_vec, self.pqs_inv_freq_a, self.pqs_inv_freq_c,
                pos_scale_a=self.pqs_pos_scale_a, rope_scale_a=self.pqs_rope_scale_a,
                sm_scale=self.scaling, n_global=self.pqs_ng, n_local=self.pqs_nl,
                tile=self.pqs_tile, block_n=self.pqs_block_n,
                pos_advance=self.pqs_advance, q_pos_base=past_len, use_skip=self.pqs_use_skip,
                num_warps=getattr(self, "pqs_num_warps", None))
            self.pqs_last_span = span.detach()          # THE diagnostic: max must stay < W
            self.pqs_last_kept = kept.detach()
            self.pqs_last_skipped = skipped.detach()
        else:
            # ── PER-QUERY path: real DECODE (past_len>0, q_len=1) and per-query prefill
            #    (prefill_mode="perquery", the reference mode decode is verified against).
            select_backend = getattr(self, "pqs_select_backend", "torch")
            attend_backend = getattr(self, "pqs_decode_backend", "torch")
            Kf, Vf, qf = k[0], v[0], q[0]
            outs = []
            for i in range(N):
                P = past_len + i
                Ki, Vi, qi = Kf[:, :P + 1, :], Vf[:, :P + 1, :], qf[:, i, :]
                scores = None
                if select_backend == "triton":
                    scores = _decode_scores_kernel(qi, Ki, self.pqs_inv_freq_a, P,
                                                   self.pqs_pos_scale_a, self.pqs_rope_scale_a,
                                                   self.scaling, block_n=self.pqs_block_n)
                idx, npk, npq = _decode_build_sets_pqs(
                    qi, Ki, self.tau_head_vec, self.pqs_inv_freq_a, P, self.pqs_pos_scale_a,
                    self.pqs_rope_scale_a, self.scaling, self.pqs_ng, self.pqs_nl,
                    self.pqs_block_n, scores=scores, block_k=self.pqs_block_k)
                if attend_backend == "triton":
                    oh = _decode_attend_triton(qi, Ki, Vi, idx, npk, npq, self.pqs_inv_freq_c,
                                               self.scaling, q_pos=P, block_k=self.pqs_block_k)
                else:
                    oh = _decode_attend_torch(qi, Ki, Vi, idx, npk, npq, self.pqs_inv_freq_c,
                                              self.scaling)
                outs.append(oh)
            out = torch.stack(outs, dim=1).unsqueeze(0)       # (1,H,q_len,D)

        attn_output = out.transpose(1, 2).reshape(B, N, -1).contiguous()
        return self.o_proj(attn_output), None

    ml.LlamaAttention.forward = pqs_forward


# All three harnesses call install_shared_pi_forward (run_ruler.py:143, run_longbench.py:107,
# run_kernel_benchmark.py:200). The alias keeps the accurate name the real one while costing zero
# harness edits for the rename.
install_shared_pi_forward = install_pqstream_forward
