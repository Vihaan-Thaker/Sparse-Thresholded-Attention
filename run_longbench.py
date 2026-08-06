#!/usr/bin/env python3
"""
run_longbench.py

Run LongBench (English QA subset) on Llama-2-7b-CHAT with our sparse-selection kernels or dense
baselines, generating predictions to JSONL. Scoring is done separately by score_longbench.py.

One (kernel, task-set) per invocation. Kernels reuse install_shared_pi_forward exactly as in the
verified PG-19 driver; only the MODEL (chat) and TAU (chat thresholds) differ.

Positioning: context is EXTENDED (not truncated to 4k). dense_pi/dense_yarn use a fixed rope factor
= max(1, max_length/W). Sparse kernels adapt pos_scale per-sample (installed with max_len=None, so
pos_scale = min(1, W/N) uses each sample's actual length N); yarn/hybrid inv_freq is baked at load
with the same fixed factor.

Decode backend: default "triton" (verified against the torch reference by verify_triton_decode.py for
exact2/yarn_exact/hybrid_exact — both the select-scores and attend Kernel-C toggles). "torch" remains
selectable as the slower reference path.
"""
import os, json, time, argparse
import numpy as np
import torch
from transformers import LlamaForCausalLM, AutoConfig, AutoTokenizer

import longbench_config as lb

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
TAU_PATH = "/home/rethinkingai-self/vihaan/attn_threshold_results/analysis_seqlen4096_chat/stats_H_ws.npz"

# kernel id -> module name (sparse kernels only)
SPARSE_MODS = {
    "exact":        "shared_selection_triton_exact",
    "exact2":       "shared_selection_triton_exact_2",
    "yarn_exact":   "shared_selection_triton_YaRN_exact",
    "hybrid_exact": "shared_selection_triton_hybrid_exact",
    "pq_stream":    "pqstream_kernel",              # per-query + streamed tile positions
}
YARN_LOAD = {"yarn_exact", "hybrid_exact"}          # need a yarn-loaded model for Kernel A
# Kernels whose selection geometry is a free choice. Under geom="yarn" they read YaRN inv_freq +
# mscale off rotary_emb, so the MODEL must be yarn-loaded too -- `yarn` below folds that in. Without
# it, geom=yarn would install against a plain-loaded model, the kernel would print its
# attention_scaling==1.0 warning, and the run would quietly select under the wrong geometry.
# (pq_exact/chunk_exact are deliberately absent: this harness has never carried them.)
GEOM_KERNELS = {"pq_stream"}


def load_tau(H, ki):
    return np.nan_to_num(H["seq_mean"][:, :, ki].astype(np.float32), nan=np.inf)


def _mutate_rope(cfg, rope_type, factor, seq_len, context_window):
    cfg.max_position_embeddings = max(int(cfg.max_position_embeddings), seq_len + 8)
    base = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) \
           or {"rope_theta": float(getattr(cfg, "rope_theta", 10000.0))}
    rp = dict(base); rp["rope_type"] = rope_type; rp["factor"] = factor
    if rope_type == "yarn":
        rp["original_max_position_embeddings"] = int(context_window)
    for attr in ("rope_parameters", "rope_scaling"):
        try:
            setattr(cfg, attr, dict(rp))
        except Exception:
            pass
    return cfg


def load_model(kernel, max_length, args, device):
    """Load chat model + (for sparse) install our kernel with chat thresholds."""
    W = float(args.context_window)
    factor = max(1.0, max_length / W)               # fixed rope factor for the whole run

    if kernel in ("dense_raw", "dense_pi", "dense_yarn"):
        cfg = AutoConfig.from_pretrained(MODEL_NAME)
        if kernel == "dense_pi":
            cfg = _mutate_rope(cfg, "linear", factor, max_length, args.context_window)
        elif kernel == "dense_yarn":
            cfg = _mutate_rope(cfg, "yarn", factor, max_length, args.context_window)
        else:
            cfg.max_position_embeddings = max(int(cfg.max_position_embeddings), max_length + 8)
        m = LlamaForCausalLM.from_pretrained(
            MODEL_NAME, config=cfg, torch_dtype=torch.float16,
            attn_implementation=args.attn_impl, device_map=device).eval()
        print(f"{kernel}: dense chat model, factor={factor:.3f}", flush=True)
        return m

    # ---- sparse: load (yarn or plain) chat model, set chat tau, install kernel ----
    yarn = kernel in YARN_LOAD or (kernel in GEOM_KERNELS and args.select_geom == "yarn")
    if yarn:
        cfg = _mutate_rope(AutoConfig.from_pretrained(MODEL_NAME), "yarn", factor, max_length, args.context_window)
        m = LlamaForCausalLM.from_pretrained(
            MODEL_NAME, config=cfg, torch_dtype=torch.float16,
            attn_implementation="eager", device_map=device).eval()
        print(f"{kernel}: yarn-loaded chat model, factor={factor:.3f}", flush=True)
    else:
        m = LlamaForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, attn_implementation="eager", device_map=device).eval()
        print(f"{kernel}: plain chat model", flush=True)

    Hf = np.load(TAU_PATH)
    ki = int(np.where(Hf["k_values"] == args.k)[0][0])
    tau_t = torch.from_numpy(load_tau(Hf, ki)).to(device)
    for li, layer in enumerate(m.model.layers):
        layer.self_attn.tau_head_vec = tau_t[li].contiguous()

    st = __import__(SPARSE_MODS[kernel])
    inst = dict(pct=30.0, context_window=args.context_window,
                n_global=args.n_global, n_local=args.n_local, tile=args.tile,
                k_max=args.k_max, block_n=args.block_n, block_k=args.block_k,
                max_len=None,                                   # per-sample pos_scale (min(1,W/N))
                decode_backend=args.decode_backend, select_backend=args.decode_backend)
    if kernel in ("exact2", "yarn_exact", "hybrid_exact"):
        inst["select_score"] = args.select_score
    if kernel in GEOM_KERNELS:
        inst["select_geom"] = args.select_geom
    if kernel == "pq_stream":
        inst["pos_advance"] = args.pos_advance
        # k_max stays in `inst` and is accepted-and-ignored by install_pqstream_forward: pq_stream
        # has no gather budget, so there is nothing to cap.
    if yarn:
        inst["apply_mscale"] = bool(args.apply_mscale)
    st.install_shared_pi_forward(m, **inst)
    if kernel == "pq_stream":
        print(f"{kernel}/geom={args.select_geom}/adv={args.pos_advance}/bn={args.block_n}: "
              f"installed (NO k_max, decode={args.decode_backend})", flush=True)
        return m
    print(f"{kernel}/{args.select_score}: installed (k_max={args.k_max}, decode={args.decode_backend})", flush=True)
    return m


@torch.no_grad()
def run_task(model, tok, task, examples, max_length, device):
    preds = []
    max_gen = lb.MAX_NEW_TOKENS[task]
    for i, ex in enumerate(examples):
        prompt = lb.build_prompt(task, ex)
        prompt = lb.truncate_middle(tok, prompt, max_length)
        if task in lb.CHAT_TASKS:
            prompt = lb.build_chat_llama2(prompt)
        ids = tok(prompt, truncation=False, return_tensors="pt").input_ids.to(device)
        ctx_len = ids.shape[1]
        out = model.generate(ids, max_new_tokens=max_gen, do_sample=False, num_beams=1,
                             use_cache=True, pad_token_id=tok.eos_token_id)
        pred = tok.decode(out[0, ctx_len:], skip_special_tokens=True)
        preds.append(dict(predicted_answer=pred, answers=list(ex["answers"]),
                          all_classes=list(ex["all_classes"]) if ex["all_classes"] is not None else [],
                          length=int(ex["length"]), task=task, ctx_len=ctx_len))
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{task}] {i+1}/{len(examples)}  ctx={ctx_len}  pred[:60]={pred[:60]!r}", flush=True)
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True,
                   choices=["dense_raw", "dense_pi", "dense_yarn", "exact", "exact2",
                            "yarn_exact", "hybrid_exact", "pq_stream"])
    p.add_argument("--select_geom", default="pi", choices=["pi", "yarn"],
                   help="pq_stream: selection geometry (the attend stage always uses base RoPE at "
                        "streamed positions). geom=yarn also forces a yarn-loaded model.")
    p.add_argument("--pos_advance", default="max", choices=["max", "full"],
                   help="pq_stream: per-block base advance. 'max' = max_i c_i (the scheme); "
                        "'full' = no compression -> native RoPE (the control).")
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_samples", type=int, default=0, help="0 = all samples in the task")
    p.add_argument("--max_length", type=int, default=24576)     # 24k: extend, let kernels reposition
    p.add_argument("--context_window", type=int, default=4096)
    p.add_argument("--k_max", type=int, default=2048)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--tile", type=int, default=32)
    p.add_argument("--n_global", type=int, default=32)
    p.add_argument("--n_local", type=int, default=32)
    p.add_argument("--block_n", type=int, default=32)
    p.add_argument("--block_k", type=int, default=32)
    p.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--select_score", default="expmass", choices=["freq", "margin", "expmass"])
    p.add_argument("--apply_mscale", type=int, default=1, choices=[0, 1])
    p.add_argument("--decode_backend", default="triton", choices=["torch", "triton"],
                   help="decode select+attend backend; triton (default, verified) is fast, torch is the reference")
    p.add_argument("--dataset_id", default="THUDM/LongBench")
    args = p.parse_args()
    device = "cuda"

    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    outdir = os.path.join(args.out_dir, args.kernel if args.kernel.startswith("dense")
                          else f"{args.kernel}_{args.select_score}")
    os.makedirs(outdir, exist_ok=True)

    print(f"loading model: kernel={args.kernel} max_length={args.max_length}", flush=True)
    model = load_model(args.kernel, args.max_length, args, device)

    for task in args.tasks:
        print(f"\n=============== TASK {task} ===============", flush=True)
        ds = load_dataset(args.dataset_id, task, split="test", trust_remote_code=True)
        examples = list(ds) if args.n_samples <= 0 else list(ds)[:args.n_samples]
        t0 = time.time()
        preds = run_task(model, tok, task, examples, args.max_length, device)
        path = os.path.join(outdir, f"{task}.jsonl")
        with open(path, "w") as f:
            for r in preds:
                f.write(json.dumps(r) + "\n")
        print(f"-> {task}: {len(preds)} preds in {time.time()-t0:.0f}s  saved {path}", flush=True)

    print(f"\nDONE. predictions in {outdir}", flush=True)


if __name__ == "__main__":
    main()
