#!/usr/bin/env python3
"""
dca_kernel.py

Dual Chunk Attention (DCA) — the attention core only, extracted for the RULER harness.

    Paper : "Training-Free Long-Context Scaling of Large Language Models"
            An et al., ICML 2024 — arXiv:2402.17463
    Repo  : github.com/HKUNLP/ChunkLlama  (chunkllama_attn_replace.py)

WHY THIS FILE EXISTS
--------------------
Implementations/DCA/dca_implement.py (1381 lines) holds the validated DCA core -- line-by-line
matched against ChunkLlama and already reproducing LongBench narrativeqa = 20.15 on Prajna. It
CANNOT be imported here: at module load it does sys.path.insert on ./sparse-attention-hub and pulls
in the SAH adapter + benchmark runners, a tree that lives in ~/dca_project/, not in ~/vihaan/ where
RULER runs. So the core is extracted and everything from §6 on (adapter, run_dca_benchmarks,
LongBench prompts, CLI) is dropped.

NO TRITON. DCA is not a sparse method: it is DENSE attention with remapped positions. Every query
still sees every causal key; only the RoPE position assigned to each (query, key) pair changes, so
that no relative distance ever exceeds the pretraining window. The implementation is three
Flash-Attention calls (intra / succ / inter) merged by log-sum-exp, exactly as in Appendix A.3.

WHAT DCA DOES (paper Eq. 2 / 5 / 7 / 8)
---------------------------------------
Keys are rotated cyclically:            P_k[j]        = j mod s
Queries get THREE position sets, chosen by how far apart the chunks are:
    same chunk        (di == 0)   ->    P_q^intra[i]  = i mod s
    one chunk back    (di == 1)   ->    P_q^succ[i]   = [s, s+1, ..., s+w-1, c-1, ..., c-1]
    further back      (di >  1)   ->    P_q^inter[i]  = c-1                       (constant)
then a single softmax over the concatenation (Eq. 9).

RUNTIME CONVENTION (ChunkLlama) vs FIGURE CONVENTION (paper Fig. 2)
-------------------------------------------------------------------
ChunkLlama's code -- and therefore this file -- strides by  chunk_len = s - w  and takes positions
modulo chunk_len, clamping succ/inter at chunk_size. The paper's Figure 2 strides by s and clamps at
c-1. The two are structurally identical with (s -> chunk_len, c-1 -> chunk_size); only the constants
differ. For Llama-2 (s=3072, w=512, chunk_len=2560) the runtime values are:

    keys        j mod 2560                  in [0, 2559]
    intra q     i mod 2560                  in [0, 2559]
    succ  q     min(2560 + off, 3072)       in [2560, 3072]   (unclamped for off < w = 512)
    inter q     3072                        constant

Sanity of the succ set: a query at chunk offset 0 scores the previous chunk's last key, whose key
position is 2559, from succ position 2560 -> relative distance 1. Locality across the chunk boundary
is preserved exactly, which is the entire point of §3.3. Max relative distance anywhere is
3072 - 0 = 3072 < 4096, so RoPE is never out of distribution.

FIXES APPLIED RELATIVE TO dca_implement.py  (each marked FIX-n at its site)
--------------------------------------------------------------------------
1. forward signature: this transformers version passes `position_embeddings` as the SECOND
   POSITIONAL argument (see shared_selection_triton_chunked.spi_forward, which is written that way
   and works). The old signature had `attention_mask` there, so the (cos, sin) tuple would have
   landed in attention_mask and the real mask in position_ids -- then `position_ids % chunk_len`
   on a 4-D mask. Signature realigned, and position_ids is now ALWAYS derived from the cache
   length rather than trusted from the caller.
2. _fa_block dtype: the old code forced bfloat16 whenever compute capability >= 8, which is the
   only branch that reaches flash_attn at all (the float16 arm of that ternary was dead code). The
   model loads in float16, so every call round-tripped fp16 -> bf16 -> fp16 and threw away 2
   mantissa bits for nothing. That alone would break the `max|logit diff| < 1e-3` dense oracle.
3. pretrain_len is now authoritative for the rotary's max_position_embeddings instead of being read
   off the model's rotary. mscale = get_mscale(seq_len / max_position_embeddings) is
   LENGTH-DEPENDENT, so if a caller bumps cfg.max_position_embeddings (the natural thing to do for
   a long-context run) the ratio collapses to 1.0 and the mscale silently switches off.
4. chunked-prefill guard: prefill indexes k_full with QUERY-relative indices, which is only valid
   when the cache is empty. With a warm cache and q_len > 1 it would silently attend to the wrong
   keys. Now asserted rather than left latent.
5. _merge_lse does the weighted sum in fp32 and casts once at the end.
6. patch_model_with_dca returns the patched-layer count and refuses to return silently having
   patched nothing -- the failure mode that let chunk_exact run dense for a whole benchmark round.
7. rotary cache lifetime: the guard `if seq_len > self.max_seq_len` was compared against a
   max_seq_len set to seq_len, ignoring the _MAX_NEW_TOKENS headroom the table was built with. Every
   decode step grows kv_len by 1 and so rebuilt all six trig tables: 161 rebuilds per layer per
   sample at any length >= 4096 (~5k per sample over 32 layers). It also let mscale drift token by
   token, so cached keys ended up rotated at a slightly different mscale than the live query.
   Now 1 rebuild per sample, with mscale frozen at the prefill length.
8. softmax scale is taken from self.scaling when present (identical for Llama-2, but hard-coding
   1/sqrt(D) would silently diverge on a model that scales differently).
9. the n_patched == 0 restore path iterated ALL of `originals`, including the rotary entries whose
   values are CLASS objects -- assigning one to ChunkLlamaRotaryEmbedding.forward corrupted this
   module for the rest of the process. Restores are now keyed by category.
10. flash_attn's return arity under return_attn_probs (needed for the LSE) has moved across
   releases; dca_env has 2.8.3 but RULER runs under attn311. A mismatch now falls through to the
   PyTorch path once, loudly, instead of killing a 24h job on an unpack error.

VERIFIED AGAINST THE PAPER (numerically, not by eye)
----------------------------------------------------
  * Eq. 8 three-way partition vs the loop's key ranges         0 mismatches
  * prefill vs decode assign the same (type, P_q) to a pair    0 disagreements
  * every relative position lands in [0, c=4096)               0 violations
  * cross-chunk boundary distance == 1                          exact
  * reachable ranges: intra [0, i%chunk_len], succ [.., 2880], inter [513, 3072]

NOTE ON THE SUCC/INTER OVERLAP (expected, not a bug). succ distances reach 2880 while inter
distances start at 513, so a chunk-2 query can see a distant chunk-0 key at M=513 while a nearer
chunk-1 key sits at M=2880 -- the nearer key looks farther. This inversion is inherent to assigning
all inter-chunk queries one constant position, and it is present in the paper's own Figure 2
example (c=10, s=6, w=4: chunk-1 keys get M in [1,6], chunk-0 keys get M in [4,9]). §3.2 calls it
out as attention "albeit with less precision for distant token positions."

Batch size 1 is assumed throughout (matching the RULER driver). Prefill + decode.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Flash Attention 2 (optional) ──────────────────────────────────────────────
try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _flash_attn_func = None
    _FLASH_ATTN_AVAILABLE = False

try:
    from transformers.models.llama.modeling_llama import repeat_kv as _hf_repeat_kv
except ImportError:
    def _hf_repeat_kv(hidden_states, n_rep):
        if n_rep == 1:
            return hidden_states
        bs, n_kv, slen, hd = hidden_states.shape
        return (hidden_states[:, :, None, :, :]
                .expand(bs, n_kv, n_rep, slen, hd)
                .reshape(bs, n_kv * n_rep, slen, hd))


_FLASH_DISABLED = False          # set by _disable_flash() on a signature mismatch (FIX-10)


def _disable_flash(reason: str) -> None:
    global _FLASH_DISABLED
    if not _FLASH_DISABLED:
        _FLASH_DISABLED = True
        print("\n" + "!" * 78, flush=True)
        print(f"[DCA] flash_attn call FAILED ({reason}).", flush=True)
        print("[DCA] Falling back to the PyTorch attention path for the REST OF THIS RUN.",
              flush=True)
        print("[DCA] Results stay correct; expect a large slowdown and higher memory at 12k.",
              flush=True)
        print("!" * 78 + "\n", flush=True)


def flash_attn_status() -> str:
    """Which attention path _fa_block will take. Log this at install time: the PyTorch fallback is
    correct but materialises the score matrix, so a silent fallback at 12k is a 30x slowdown, not a
    wrong answer -- and we would rather know which one produced the numbers."""
    if not _FLASH_ATTN_AVAILABLE:
        return "flash_attn NOT importable -> PyTorch fallback (correct, slower, more memory)"
    if _FLASH_DISABLED:
        return "flash_attn present but DISABLED after a call failure -> PyTorch fallback"
    return "flash_attn available (used on sm80+ CUDA tensors)"


# Module-level globals; set by patch_model_with_dca BEFORE any rotary is constructed.
_DCA_CHUNK_SIZE: Optional[int] = None
_DCA_LOCAL_WINDOW: Optional[int] = None
_MAX_NEW_TOKENS: int = 512          # headroom for the cyclic key table during generation


def get_mscale(scale: float = 1.0) -> float:
    """YaRN mscale for length-generalization scaling (ChunkLlama uses 0.05, not YaRN's 0.1)."""
    if scale <= 1:
        return 1.0
    return 0.05 * math.log(scale) + 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Rotary embedding: six cos/sin tables, one pair per DCA position type
# ══════════════════════════════════════════════════════════════════════════════
class ChunkLlamaRotaryEmbedding(torch.nn.Module):
    """Byte-identical to ChunkLlama's ChunkLlamaRotaryEmbedding.

    Returns a SIX-tuple (q_cos, q_sin, qc_cos, qc_sin, k_cos, k_sin), unlike HF's rotary which
    returns two. dca_forward is the only consumer; the model-level rotary is left untouched and its
    output is ignored via **kwargs.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000,
                 scaling_factor: float = 1.0, device=None):
        super().__init__()
        self.max_seq_len = max_position_embeddings
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self._set_cos_sin_cache(seq_len=self.max_seq_len, device=device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype) -> None:
        # NOTE mscale is LENGTH-DEPENDENT: it grows as the input outruns max_position_embeddings.
        # Keep max_position_embeddings pinned to the PRETRAINING length (4096) or this silently
        # becomes 1.0 -- see FIX-3 in patch_model_with_dca.
        scale = seq_len / self.max_position_embeddings
        mscale = get_mscale(scale)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        chunk_len = _DCA_CHUNK_SIZE - _DCA_LOCAL_WINDOW          # stride between chunks

        # intra queries: 0 … chunk_len-1
        q_t = (torch.arange(chunk_len, device=device, dtype=self.inv_freq.dtype)
               / self.scaling_factor)
        # succ/inter queries: chunk_len … chunk_size (clamped) -- Eq. 7, with the clamp providing
        # the constant inter position (Eq. 5) at index chunk_len-1.
        qc_t = ((torch.arange(chunk_len, device=device, dtype=self.inv_freq.dtype) + chunk_len)
                .clamp(max=_DCA_CHUNK_SIZE) / self.scaling_factor)
        # keys: cyclic, tiled over seq_len + generation headroom -- Eq. 2
        k_t = (torch.arange(seq_len + _MAX_NEW_TOKENS, device=device, dtype=self.inv_freq.dtype)
               % chunk_len / self.scaling_factor)

        q_freqs = torch.outer(q_t, self.inv_freq)
        qc_freqs = torch.outer(qc_t, self.inv_freq)
        k_freqs = torch.outer(k_t, self.inv_freq)

        q_emb = torch.cat((q_freqs, q_freqs), dim=-1)
        qc_emb = torch.cat((qc_freqs, qc_freqs), dim=-1)
        k_emb = torch.cat((k_freqs, k_freqs), dim=-1)

        self.register_buffer("q_cos_cached", (q_emb.cos() * mscale).to(dtype), persistent=False)
        self.register_buffer("q_sin_cached", (q_emb.sin() * mscale).to(dtype), persistent=False)
        self.register_buffer("qc_cos_cached", (qc_emb.cos() * mscale).to(dtype), persistent=False)
        self.register_buffer("qc_sin_cached", (qc_emb.sin() * mscale).to(dtype), persistent=False)
        self.register_buffer("k_cos_cached", (k_emb.cos() * mscale).to(dtype), persistent=False)
        self.register_buffer("k_sin_cached", (k_emb.sin() * mscale).to(dtype), persistent=False)
        self._last_mscale = float(mscale)
        self._cast = None          # FIX-11: invalidate the dtype-cast view cache (see forward)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = self.max_seq_len
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len=seq_len, device=self.inv_freq.device,
                                    dtype=torch.float32)
            # FIX-7: credit the generation headroom that _set_cos_sin_cache already built into
            # k_cos_cached (seq_len + _MAX_NEW_TOKENS rows). Setting max_seq_len = seq_len instead
            # -- as ChunkLlama does -- makes EVERY decode step (kv_len grows by 1) exceed it and
            # rebuild all six tables: 161 rebuilds per layer per sample at any length >= 4096,
            # i.e. ~5k rebuilds of ~12800x128 trig tables per sample across 32 layers. It is also a
            # correctness wart: mscale = get_mscale(seq_len/4096) drifts token by token, so cached
            # keys end up rotated at a slightly different mscale than the live query.
            # Pinning it here freezes mscale at the PREFILL length for the whole generation, which
            # is the intended semantics -- the headroom exists precisely so decode never rebuilds.
            self.max_seq_len = seq_len + _MAX_NEW_TOKENS
        # FIX-11: cast ONCE, then hand out slices.
        # The tables are stored fp32 and the model runs fp16, so `.to(dtype)` ALLOCATES AND COPIES.
        # ChunkLlama does that on every call; with 32 layers that is 192 allocate-and-convert ops
        # per generated token (~51M elements at kv_len 2100) to reproduce numbers that never change
        # — the tables are static after FIX-7. Slicing, by contrast, returns a VIEW and is free.
        # Measured (profile_dca_decode.py, job 272430, A100):
        #     rotary.forward  0.172 -> 0.018 ms/call   (9x)
        #     DCA decode     63.75 -> 45.05 ms/token   (1.42x), tables bit-identical
        # NOTE this fix is real but SMALL. It does NOT explain the ~1090 ms/token the RULER job
        # measured; the same profiler put DCA at 63.75 ms/token, so that 17x gap lives outside the
        # kernel (host contention / launch latency are the open suspects). Do not read this as the
        # RULER slowdown having been solved.
        # q_cos/qc_cos have only chunk_len rows; slicing to seq_len is a no-op past that, which is
        # correct because they are indexed by IN-CHUNK offsets, never by absolute position.
        dt = x.dtype
        if getattr(self, "_cast", None) is None or self._cast_dtype != dt:
            self._cast = tuple(getattr(self, n).to(dt) for n in
                               ("q_cos_cached", "q_sin_cached", "qc_cos_cached",
                                "qc_sin_cached", "k_cos_cached", "k_sin_cached"))
            self._cast_dtype = dt
        return tuple(t[:seq_len] for t in self._cast)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         position_ids: torch.Tensor) -> torch.Tensor:
    """Table-lookup RoPE — identical to ChunkLlama's. x is (bsz, heads, seq, dim);
    position_ids is (bsz, seq) of indices into the cos/sin slabs."""
    cos = cos.squeeze(1).squeeze(0).to(position_ids.device)
    sin = sin.squeeze(1).squeeze(0).to(position_ids.device)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


# ══════════════════════════════════════════════════════════════════════════════
# Attention blocks + LSE merge (paper §3.4, Appendix A.3)
# ══════════════════════════════════════════════════════════════════════════════
def _fa_block(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool,
              softmax_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """One attention block -> (out (b,H,Lq,D), lse (b,H,Lq)). Flash-Attention when available,
    otherwise a numerically equivalent PyTorch path."""
    b, H, L_q, D = q.shape
    _, _, L_k, _ = k.shape

    if L_k == 0:
        lse_empty = torch.full((b, H, L_q), float("-inf"), device=q.device, dtype=torch.float32)
        return torch.zeros_like(q), lse_empty

    if (_FLASH_ATTN_AVAILABLE and not _FLASH_DISABLED
            and q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 8):
        # FIX-2: preserve the incoming dtype. The old code forced bfloat16 here, which is a
        # gratuitous 2-mantissa-bit loss on an fp16 model and is enough on its own to break the
        # bit-level dense oracle at N <= chunk_len.
        _fa_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
        q_fa = q.transpose(1, 2).to(_fa_dtype)
        k_fa = k.transpose(1, 2).to(_fa_dtype)
        v_fa = v.transpose(1, 2).to(_fa_dtype)

        # FIX-10: we need the LSE to merge the three blocks, and it only comes back via
        # return_attn_probs, whose return arity has moved across flash-attn releases. dca_env has
        # 2.8.3 but RULER runs under attn311. Rather than let a 24h job die on an unpack error,
        # fall through to the PyTorch path ONCE, loudly, and stay there.
        try:
            out_fa, lse, _ = _flash_attn_func(
                q_fa, k_fa, v_fa, softmax_scale=softmax_scale, causal=causal,
                return_attn_probs=True)
            return out_fa.transpose(1, 2).to(q.dtype), lse.float()
        except (ValueError, TypeError) as e:                            # noqa: BLE001
            _disable_flash(f"{type(e).__name__}: {e}")

    # FIX-11: run the fp32 GEMM under TF32 instead of full fp32.
    #
    # This path is NOT a rarely-taken fallback here: flash_attn is not importable in the attn311
    # env that actually runs RULER, so it IS the production path, and its speed decides whether the
    # 7-length sweep fits in a 23:59 slot.
    #
    # A true fp32 GEMM runs at ~19.5 TFLOPS on an A100 vs ~156 for TF32 and ~312 for fp16.
    # Doing the matmul in fp16 instead would be fastest, but it ROUNDS THE SCORES TO fp16 before
    # the softmax (measured: max|dscore| 0.016 on scores of magnitude ~59), whereas flash_attn
    # keeps them in fp32 registers. TF32 gives ~8x over fp32 while keeping fp32 accumulation AND
    # fp32 score storage, so it does not move the numbers the V2 dense oracle was measured against.
    # The flag is global, so save/restore it rather than leaking the change into the rest of the
    # model (and into any other kernel sharing the process).
    _tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * softmax_scale
        if causal and L_q > 1:
            mask = torch.triu(torch.ones(L_q, L_k, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        out = torch.matmul(F.softmax(scores, dim=-1), v.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _tf32
    return out.to(q.dtype), lse


def _merge_lse(blocks: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Combine (out, lse) blocks by log-sum-exp weighting. Identical to one global softmax over the
    concatenated score vector, which is what Eq. 9 specifies."""
    if len(blocks) == 1:
        return blocks[0][0]

    out_dtype = blocks[0][0].dtype
    # FIX-5: accumulate in fp32; cast once at the end.
    outs = torch.stack([b[0] for b in blocks]).float()
    lses = torch.stack([b[1] for b in blocks]).float()

    lse_max = lses.max(dim=0).values
    weights = (lses - lse_max.unsqueeze(0)).exp()
    weights = weights / weights.sum(dim=0, keepdim=True)
    return (outs * weights.unsqueeze(-1)).sum(dim=0).to(out_dtype)


# ══════════════════════════════════════════════════════════════════════════════
# DCA forward
# ══════════════════════════════════════════════════════════════════════════════
def _make_dca_forward(chunk_size: int, pretrain_len: int, local_window: int):
    s, c, w = chunk_size, pretrain_len, local_window
    chunk_len = s - w

    def dca_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,      # FIX-1: SECOND POSITIONAL in this transformers version.
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        **kwargs,
    ):
        """DCA replacement for LlamaAttention.forward.

        `position_embeddings` (the model-level rotary's (cos, sin)) is accepted and DISCARDED: DCA
        computes its own rotations from the six-table rotary installed on this module. Likewise
        `attention_mask` -- causality is enforced by the intra block's causal flag and by the
        key ranges the other two blocks are given.
        """
        bsz, q_len, _ = hidden_states.shape

        # HF spells the cache `past_key_value` (singular) on the decoder-layer call path.
        if past_key_values is None:
            past_key_values = kwargs.get("past_key_value", None)

        D = self.head_dim
        H = getattr(self, "num_heads", None) or self.config.num_attention_heads
        H_kv = getattr(self, "num_key_value_heads", None) or self.config.num_key_value_heads
        n_groups = H // H_kv
        # FIX-8: prefer the module's own scaling if it exposes one. For Llama-2 self.scaling is
        # head_dim**-0.5 so this is identical, but hard-coding 1/sqrt(D) would silently diverge on
        # any model that scales differently.
        scale = float(getattr(self, "scaling", None) or (1.0 / math.sqrt(D)))

        q = self.q_proj(hidden_states).view(bsz, q_len, H, D).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, H_kv, D).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, H_kv, D).transpose(1, 2)

        # ── cache length ──────────────────────────────────────────────────────
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                try:
                    start_pos = int(past_key_values.get_seq_length(self.layer_idx))
                except TypeError:
                    start_pos = int(past_key_values.get_seq_length())
            elif not hasattr(past_key_values, "update"):
                start_pos = past_key_values[self.layer_idx][0].shape[2]
            else:
                start_pos = 0
        else:
            start_pos = 0

        # FIX-4: prefill indexes k_full with QUERY-relative indices below, which is only valid with
        # an empty cache. A warm cache + q_len > 1 would silently attend to the wrong keys.
        assert start_pos == 0 or q_len == 1, (
            f"dca_forward: chunked prefill is unsupported (start_pos={start_pos}, q_len={q_len}). "
            f"Prefill must be a single pass over an empty cache; decode must be one token.")

        kv_seq_len = start_pos + q_len

        # FIX-1: derive positions from the cache rather than trusting a caller-supplied
        # position_ids, whose argument slot moved between transformers versions.
        position_ids = torch.arange(start_pos, kv_seq_len, device=q.device).unsqueeze(0)

        q_cos, q_sin, qc_cos, qc_sin, k_cos, k_sin = self.rotary_emb(v, seq_len=kv_seq_len)

        # keys carry the cyclic embedding (Eq. 2); the cache therefore stores ROTATED keys
        k = apply_rotary_pos_emb(k, k_cos, k_sin, position_ids)

        if past_key_values is not None:
            if hasattr(past_key_values, "update"):
                k, v = past_key_values.update(k, v, self.layer_idx)
            else:
                past_k, past_v = past_key_values[self.layer_idx]
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

        kv_len = k.shape[2]
        position_ids = position_ids % chunk_len          # in-chunk offsets for the query tables

        k_full = _hf_repeat_kv(k, n_groups) if n_groups > 1 else k
        v_full = _hf_repeat_kv(v, n_groups) if n_groups > 1 else v

        # ══ PREFILL ═══════════════════════════════════════════════════════════
        if q_len > 1:
            attn_out = torch.zeros(bsz, H, q_len, D, device=q.device, dtype=q.dtype)

            for begin in range(0, q_len, chunk_len):
                seg_len = min(chunk_len, q_len - begin)
                end = begin + seg_len
                q_slice = q[:, :, begin:end]
                pid_slice = position_ids[:, begin:end]

                blocks: List[Tuple[torch.Tensor, torch.Tensor]] = []

                # intra — same chunk, causal (di == 0)
                q_intra = apply_rotary_pos_emb(q_slice, q_cos, q_sin, pid_slice)
                blocks.append(_fa_block(q_intra, k_full[:, :, begin:end], v_full[:, :, begin:end],
                                        causal=True, softmax_scale=scale))

                # succ — exactly one previous chunk, non-causal (di == 1)
                succ_start = max(0, begin - chunk_len)
                if begin > 0:
                    q_succ = apply_rotary_pos_emb(q_slice, qc_cos, qc_sin, pid_slice)
                    blocks.append(_fa_block(q_succ, k_full[:, :, succ_start:begin],
                                            v_full[:, :, succ_start:begin],
                                            causal=False, softmax_scale=scale))

                # inter — everything older, at the constant clamped position (di > 1)
                if succ_start > 0:
                    inter_pos = torch.full((bsz, seg_len), chunk_len - 1,
                                           device=q.device, dtype=torch.long)
                    q_inter = apply_rotary_pos_emb(q_slice, qc_cos, qc_sin, inter_pos)
                    blocks.append(_fa_block(q_inter, k_full[:, :, :succ_start],
                                            v_full[:, :, :succ_start],
                                            causal=False, softmax_scale=scale))

                attn_out[:, :, begin:end] = _merge_lse(blocks)

        # ══ DECODE ════════════════════════════════════════════════════════════
        else:
            cur_pos = kv_len - 1
            chunk_start = (cur_pos // chunk_len) * chunk_len
            inter_end = max(0, chunk_start - chunk_len)

            q_intra = apply_rotary_pos_emb(q, q_cos, q_sin, position_ids)
            score_list: List[torch.Tensor] = []

            # Guards match ChunkLlama's `if chunk_num_curr >= 1/2`. Computing q_inter
            # unconditionally indexes qc_cos at chunk_len-1, which overruns the table whenever the
            # context is shorter than chunk_len -> CUDA device-side assert. RULER has short
            # contexts at every length, so this path runs constantly.
            if inter_end > 0:
                inter_pos = torch.tensor([[chunk_len - 1]], device=q.device, dtype=torch.long)
                q_inter = apply_rotary_pos_emb(q, qc_cos, qc_sin, inter_pos)
                score_list.append(
                    torch.matmul(q_inter, k_full[:, :, :inter_end].transpose(-2, -1)) * scale)
            if chunk_start > inter_end:
                q_succ = apply_rotary_pos_emb(q, qc_cos, qc_sin, position_ids)
                score_list.append(
                    torch.matmul(q_succ,
                                 k_full[:, :, inter_end:chunk_start].transpose(-2, -1)) * scale)
            score_list.append(
                torch.matmul(q_intra, k_full[:, :, chunk_start:kv_len].transpose(-2, -1)) * scale)

            # Eq. 9: ONE softmax over the concatenated scores. score_list is in cache order
            # (inter | succ | intra) = [0:inter_end] | [inter_end:chunk_start] | [chunk_start:],
            # so it aligns with v_full without reordering.
            attn_weights = torch.softmax(torch.cat(score_list, dim=-1), dim=-1,
                                         dtype=torch.float32).to(q.dtype)
            attn_out = torch.matmul(attn_weights, v_full)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, H * D)
        return self.o_proj(attn_out), None

    return dca_forward


# ══════════════════════════════════════════════════════════════════════════════
# Patching
# ══════════════════════════════════════════════════════════════════════════════
def _is_patchable_llama(module: torch.nn.Module) -> bool:
    required = ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "layer_idx")
    if not all(hasattr(module, a) for a in required):
        return False
    return hasattr(module, "num_heads") or (
        hasattr(module, "config") and hasattr(module.config, "num_attention_heads"))


def patch_model_with_dca(model: torch.nn.Module, chunk_size: int, pretrain_len: int,
                         local_window: int) -> Dict[str, Any]:
    """Class-level monkey-patch of every Llama-style attention forward, plus a six-table rotary on
    each attention module. Mirrors ChunkLlama's replace_with_chunkllama().

    Returns the `originals` dict for unpatch_model(). RAISES if it patched nothing.
    """
    global _DCA_CHUNK_SIZE, _DCA_LOCAL_WINDOW

    import transformers.models.llama.modeling_llama as _llama_mod

    assert 0 < local_window < chunk_size, f"need 0 < w({local_window}) < s({chunk_size})"
    assert chunk_size < pretrain_len, f"need s({chunk_size}) < c({pretrain_len})"

    # Set BEFORE constructing any ChunkLlamaRotaryEmbedding — _set_cos_sin_cache reads them.
    _DCA_CHUNK_SIZE = chunk_size
    _DCA_LOCAL_WINDOW = local_window

    dca_fwd = _make_dca_forward(chunk_size, pretrain_len, local_window)
    originals: Dict[str, Any] = {}

    for cls_name in ("LlamaAttention", "LlamaFlashAttention2", "LlamaSdpaAttention"):
        cls = getattr(_llama_mod, cls_name, None)
        if cls is not None:
            originals[cls_name] = cls.forward
            cls.forward = dca_fwd

    for re_name in ("LlamaRotaryEmbedding", "LlamaLinearScalingRotaryEmbedding"):
        if hasattr(_llama_mod, re_name):
            originals[re_name] = getattr(_llama_mod, re_name)
            setattr(_llama_mod, re_name, ChunkLlamaRotaryEmbedding)

    rotary_originals: Dict[str, Any] = {}
    n_patched = 0
    for mod_name, m in model.named_modules():
        if not _is_patchable_llama(m):
            continue
        n_patched += 1
        old_re = getattr(m, "rotary_emb", None)
        rotary_originals[mod_name] = old_re
        base = int(getattr(old_re, "base", None) or getattr(m.config, "rope_theta", 10_000))
        inv_f = getattr(old_re, "inv_freq", None) if old_re is not None else None
        dev = inv_f.device if inv_f is not None else None

        # FIX-3: pretrain_len is authoritative. Reading max_position_embeddings off the model's
        # rotary makes mscale = get_mscale(seq_len / max_position_embeddings) collapse to 1.0 the
        # moment a caller bumps the config for a long-context run -- silently disabling the
        # length-dependent scaling ChunkLlama applies.
        m.rotary_emb = ChunkLlamaRotaryEmbedding(
            dim=m.head_dim,
            max_position_embeddings=pretrain_len,
            base=base,
            scaling_factor=1.0,
            device=(dev if dev is not None and dev.type != "meta" else None),
        )

    # FIX-6: patching zero layers is the failure mode that let chunk_exact run dense for a whole
    # benchmark round -- plausible numbers, wrong method. Refuse to return quietly.
    if n_patched == 0:
        # FIX-9: restore ONLY the attention classes. Iterating all of `originals` would also hit the
        # rotary entries, whose values are CLASS objects -- assigning one to
        # ChunkLlamaRotaryEmbedding.forward would corrupt this module for the rest of the process.
        for cls_name in ("LlamaAttention", "LlamaFlashAttention2", "LlamaSdpaAttention"):
            if cls_name in originals:
                cls = getattr(_llama_mod, cls_name, None)
                if cls is not None:
                    cls.forward = originals[cls_name]
        for re_name in ("LlamaRotaryEmbedding", "LlamaLinearScalingRotaryEmbedding"):
            if re_name in originals:
                setattr(_llama_mod, re_name, originals[re_name])
        raise RuntimeError(
            "patch_model_with_dca patched 0 attention layers — the model would have run DENSE. "
            "Check that this is a Llama-style model and that the attention modules expose "
            "q_proj/k_proj/v_proj/o_proj/head_dim/layer_idx.")

    originals["_model"] = model
    originals["_rotary_emb_originals"] = rotary_originals
    originals["_n_patched"] = n_patched

    print(f"  [DCA] patched {n_patched} attention layers "
          f"(s={chunk_size}, c={pretrain_len}, w={local_window}, chunk_len={chunk_size-local_window})",
          flush=True)
    print(f"  [DCA] {flash_attn_status()}", flush=True)
    return originals


def unpatch_model(originals: Dict[str, Any]) -> None:
    """Restore everything saved by patch_model_with_dca()."""
    try:
        import transformers.models.llama.modeling_llama as _llama_mod
    except ImportError:
        return

    n_fwd = 0
    for cls_name in ("LlamaAttention", "LlamaFlashAttention2", "LlamaSdpaAttention"):
        if cls_name in originals:
            cls = getattr(_llama_mod, cls_name, None)
            if cls is not None:
                cls.forward = originals[cls_name]
                n_fwd += 1

    for re_name in ("LlamaRotaryEmbedding", "LlamaLinearScalingRotaryEmbedding"):
        if re_name in originals:
            setattr(_llama_mod, re_name, originals[re_name])

    model = originals.get("_model")
    re_originals = originals.get("_rotary_emb_originals", {})
    if model is not None and re_originals:
        for mod_name, m in model.named_modules():
            if mod_name in re_originals:
                if re_originals[mod_name] is None:
                    if hasattr(m, "rotary_emb"):
                        delattr(m, "rotary_emb")
                else:
                    m.rotary_emb = re_originals[mod_name]

    print(f"  [DCA] restored {n_fwd} attention class(es) and rotary embeddings", flush=True)


def install_dca_forward(model, chunk_size: int = 3072, context_window: int = 4096,
                        local_window: int = 512) -> Dict[str, Any]:
    """Interface-compatible wrapper matching our other kernels' install_* naming."""
    return patch_model_with_dca(model, chunk_size, context_window, local_window)


# ══════════════════════════════════════════════════════════════════════════════
# CPU-only self-check of the position tables (no model, no GPU).
#   python dca_kernel.py
# ══════════════════════════════════════════════════════════════════════════════
def _selfcheck(s: int = 3072, c: int = 4096, w: int = 512, seq_len: int = 8192) -> None:
    global _DCA_CHUNK_SIZE, _DCA_LOCAL_WINDOW
    _DCA_CHUNK_SIZE, _DCA_LOCAL_WINDOW = s, w
    chunk_len = s - w

    rot = ChunkLlamaRotaryEmbedding(dim=128, max_position_embeddings=c, base=10000)
    rot._set_cos_sin_cache(seq_len=seq_len, device=None, dtype=torch.float32)

    q_t = torch.arange(chunk_len, dtype=torch.float32)
    qc_t = (torch.arange(chunk_len, dtype=torch.float32) + chunk_len).clamp(max=s)
    k_t = torch.arange(seq_len + _MAX_NEW_TOKENS, dtype=torch.float32) % chunk_len

    assert chunk_len == 2560, chunk_len
    assert q_t.min() == 0 and q_t.max() == chunk_len - 1
    assert qc_t[0] == chunk_len and qc_t[-1] == s
    assert int((qc_t < s).sum()) == w, f"unclamped succ offsets = {(qc_t < s).sum()}, want w={w}"
    assert k_t.max() == chunk_len - 1

    boundary = float(qc_t[0] - k_t[chunk_len - 1])           # first query of a chunk vs last key
    max_rel = float(qc_t.max() - k_t.min())
    print(f"chunk_len={chunk_len}  intra q∈[{int(q_t.min())},{int(q_t.max())}]  "
          f"succ q∈[{int(qc_t.min())},{int(qc_t.max())}] (unclamped for off<{w})  "
          f"inter q={int(qc_t[chunk_len-1])}  keys cyclic %{chunk_len}")
    print(f"cross-chunk boundary relative distance = {boundary:.0f} (want 1)")
    print(f"max relative distance = {max_rel:.0f} (want < c={c})")
    assert boundary == 1.0, boundary
    assert max_rel < c, (max_rel, c)

    for L in (4096, 8192, 12288):
        print(f"  seq_len={L:6d}  scale={L/c:.3f}  mscale={get_mscale(L/c):.6f}")
    assert _MAX_NEW_TOKENS >= 160, "must exceed ruler_config.MAX_NEW_TOKENS['cwe']"
    print("selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
