#!/usr/bin/env python3
"""
run_ruler.py

Run the (pre-generated) RULER dataset on Llama-2-7B with our sparse-selection kernels or dense
baselines, writing predictions to JSONL. Scoring is separate (score_ruler.py).

Mirrors run_longbench.py EXACTLY for model loading / kernel install / decode backend — the only
differences: (1) --model {base,chat} + --tau_path so it pairs with our bench7/8 base-6k setup,
(2) the prompt is READ from ruler_data (already length-fitted; NO truncation), (3) one --length per
invocation so the rope factor = max(1, length/W) is exact (the SLURM script loops lengths).

Positioning identical to run_longbench: dense_pi/dense_yarn use fixed rope factor; sparse kernels
install with max_len=None (per-sample pos_scale = min(1, W/N)); yarn/hybrid inv_freq baked at load.
Decode backend defaults to the verified "triton" path.
"""
import os, re, json, time, argparse
import numpy as np
import torch
from transformers import LlamaForCausalLM, AutoConfig, AutoTokenizer

import ruler_config as R

MODELS = {"base": "meta-llama/Llama-2-7b-hf", "chat": "meta-llama/Llama-2-7b-chat-hf"}
DEFAULT_TAU = "/home/rethinkingai-self/vihaan/attn_threshold_results/analysis_seqlen6144_yarn/stats_H_ws.npz"

SPARSE_MODS = {
    "exact":        "shared_selection_triton_exact",
    "exact2":       "shared_selection_triton_exact_2",
    "yarn_exact":   "shared_selection_triton_YaRN_exact",
    "hybrid_exact": "shared_selection_triton_hybrid_exact",
    "pq_exact":     "shared_selection_triton_perquery",   # per-query union + dense-rank + bitmask
    "chunk_exact":  "shared_selection_triton_chunked",    # chunk-granular, exact budget
    "pq_stream":    "pqstream_kernel",                    # per-query + streamed tile positions
}
YARN_LOAD = {"yarn_exact", "hybrid_exact"}
# pq_stream is deliberately ABSENT: it has no k_max, so no overflow, so nothing for select_score
# to rank. Its install accepts and ignores the argument, but passing it would be a lie in the log.
SEL_SCORE_KERNELS = {"exact2", "yarn_exact", "hybrid_exact", "pq_exact", "chunk_exact"}
GEOM_KERNELS = {"pq_exact", "chunk_exact", "pq_stream"}


def needs_yarn_load(kernel, select_geom):
    """pq_exact/pq_stream select under either geometry (--select_geom); yarn-select needs a
    yarn-loaded model. Mirrors run_kernel_benchmark.py so RULER and the ppl benchmark install
    identically."""
    return kernel in YARN_LOAD or (kernel in GEOM_KERNELS and select_geom == "yarn")


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


def load_model(kernel, model_name, seq_len, args, device, dtype=torch.float16):
    """dtype defaults to float16 -- what every RULER/bench result in this project was produced
    with. It is a parameter only so the SCROLLS harness can reproduce ChunkLlama's bfloat16 path
    for its published-number gate; do NOT change the default."""
    W = float(args.context_window)
    factor = max(1.0, seq_len / W)

    if kernel == "dca":
        # DCA (An et al., ICML 2024) is DENSE attention with remapped positions -- no tau, no
        # k_max, no selection. Returns early so the sparse path's tau load never runs.
        #
        # DELIBERATELY NOT mutating rope, and deliberately NOT bumping max_position_embeddings:
        #   * DCA already keeps every relative distance < c by construction (max 3072 for
        #     s=3072/w=512), so PI/YaRN on top would double-scale.
        #   * dca_kernel's mscale = get_mscale(seq_len / max_position_embeddings) is LENGTH
        #     DEPENDENT. patch_model_with_dca pins the rotary to pretrain_len (FIX-3), but leaving
        #     the config alone keeps the two consistent. The model-level rotary computes cos/sin
        #     from position_ids on the fly (no fixed table), and dca_forward discards its output
        #     anyway, so nothing needs the larger window.
        cfg = AutoConfig.from_pretrained(model_name)
        m = LlamaForCausalLM.from_pretrained(
            model_name, config=cfg, torch_dtype=dtype,
            attn_implementation="eager", device_map=device).eval()
        import dca_kernel as dk
        # eager: the patch replaces LlamaAttention.forward plus the Sdpa/FA2 subclasses IF they
        # still exist in this transformers version. eager guarantees we land on a covered class.
        # patch_model_with_dca raises if it patched 0 layers (would silently run dense).
        dk.patch_model_with_dca(m, args.dca_chunk_size, args.context_window, args.dca_local_window)
        print(f"dca: {args.model} model, s={args.dca_chunk_size} w={args.dca_local_window} "
              f"chunk_len={args.dca_chunk_size - args.dca_local_window} "
              f"(no rope mutation, no tau)", flush=True)
        return m

    if kernel in ("dense_raw", "dense_pi", "dense_yarn"):
        cfg = AutoConfig.from_pretrained(model_name)
        if kernel == "dense_pi":
            cfg = _mutate_rope(cfg, "linear", factor, seq_len, args.context_window)
        elif kernel == "dense_yarn":
            cfg = _mutate_rope(cfg, "yarn", factor, seq_len, args.context_window)
        else:
            cfg.max_position_embeddings = max(int(cfg.max_position_embeddings), seq_len + 8)
        m = LlamaForCausalLM.from_pretrained(
            model_name, config=cfg, torch_dtype=dtype,
            attn_implementation=args.attn_impl, device_map=device).eval()
        print(f"{kernel}: dense {args.model} model, factor={factor:.3f}", flush=True)
        return m

    yarn = needs_yarn_load(kernel, getattr(args, "select_geom", "pi"))
    if yarn:
        cfg = _mutate_rope(AutoConfig.from_pretrained(model_name), "yarn", factor, seq_len, args.context_window)
        m = LlamaForCausalLM.from_pretrained(
            model_name, config=cfg, torch_dtype=dtype,
            attn_implementation="eager", device_map=device).eval()
        print(f"{kernel}: yarn-loaded {args.model} model, factor={factor:.3f}", flush=True)
    else:
        m = LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="eager", device_map=device).eval()
        print(f"{kernel}: plain {args.model} model", flush=True)

    print(f"tau set: {args.tau_path}", flush=True)
    Hf = np.load(args.tau_path)
    ki = int(np.where(Hf["k_values"] == args.k)[0][0])
    tau_t = torch.from_numpy(load_tau(Hf, ki)).to(device)
    for li, layer in enumerate(m.model.layers):
        layer.self_attn.tau_head_vec = tau_t[li].contiguous()

    st = __import__(SPARSE_MODS[kernel])
    inst = dict(pct=30.0, context_window=args.context_window,
                n_global=args.n_global, n_local=args.n_local, tile=args.tile,
                k_max=args.k_max, block_n=args.block_n, block_k=args.block_k,
                max_len=None, decode_backend=args.decode_backend, select_backend=args.decode_backend)
    if kernel in SEL_SCORE_KERNELS:
        inst["select_score"] = args.select_score          # freq/margin/expmass (pq: overflow ranking)
    if kernel in GEOM_KERNELS:
        inst["select_geom"] = args.select_geom            # pi | yarn  (selection geometry)
    if kernel == "chunk_exact":
        inst["chunk_size"] = args.chunk_size
        inst["n_chunk"] = args.n_chunk if args.n_chunk > 0 else None
        inst["chunk_agg"] = args.chunk_agg
    if kernel == "pq_stream":
        inst["pos_advance"] = args.pos_advance            # max | full  (block base advance rule)
        # k_max/select_score stay in `inst` and are accepted-and-ignored by install_pqstream_forward
        # (it prints a notice). A narrower signature there would raise TypeError at install.
    if yarn:
        inst["apply_mscale"] = bool(args.apply_mscale)
    st.install_shared_pi_forward(m, **inst)
    extra = f"/geom={args.select_geom}" if kernel in GEOM_KERNELS else ""
    if kernel == "chunk_exact":
        extra += f"/C={args.chunk_size}/agg={args.chunk_agg}"
    if kernel == "pq_stream":
        extra += f"/adv={args.pos_advance}/bn={args.block_n}"
        print(f"{kernel}{extra}: installed (NO k_max, decode={args.decode_backend})", flush=True)
        return m
    print(f"{kernel}/{args.select_score}{extra}: installed "
          f"(k_max={args.k_max}, decode={args.decode_backend})", flush=True)
    return m


def variant_name(args):
    if args.kernel.startswith("dense"):
        return args.kernel
    if args.kernel == "dca":
        # s and w MUST both appear: w=512 (ChunkLlama code) vs w=1024 (paper prose) is a planned
        # ablation, and without w in the name the two would share one output dir.
        return f"dca_s{args.dca_chunk_size}_w{args.dca_local_window}"
    tag = "tau"
    mm = re.search(r"seqlen(\d+)(?:_(\w+?))?/stats", args.tau_path)
    if mm:
        L = int(mm.group(1)); rest = mm.group(2) or ""
        tag = f"{round(L/1024)}k" + (f"-{rest}" if rest else "")
    if args.kernel == "chunk_exact":
        # C and agg MUST appear: a chunk sweep varies them while everything else stays equal, so
        # without them every config would write to the same directory and silently overwrite.
        return (f"chunk_exact_{args.select_score}_geom{args.select_geom}_C{args.chunk_size}"
                f"_{args.chunk_agg}_km{args.k_max}_{tag}")
    if args.kernel == "pq_exact":
        # geom AND k_max must both appear: the bench9 sweep runs 4 configs that are otherwise identical,
        # and without these they would all share one output dir and silently overwrite each other.
        return f"pq_exact_{args.select_score}_geom{args.select_geom}_km{args.k_max}_{tag}"
    if args.kernel == "pq_stream":
        # adv AND bn must both appear. The sweep varies exactly those two (advance=max vs the
        # `full` control; BN=32 vs 64, which IS the compression granularity here, not a perf knob),
        # so without them the four configs would share one output dir and silently overwrite.
        # No select_score / no k_max — pq_stream has neither.
        return (f"pq_stream_geom{args.select_geom}_adv{args.pos_advance}"
                f"_bn{args.block_n}_{tag}")
    return f"{args.kernel}_{args.select_score}_ms{args.apply_mscale}_{tag}"


@torch.no_grad()
def run_task(model, tok, task, examples, device, chat):
    preds = []
    # RULER's `tokens_to_generate` (their scripts/data/synthetic/constants.py), NOT a value of
    # ours: the data was fitted so len(input) + tokens_to_generate <= nominal length, so the
    # generation budget and the prompt length are two halves of the same constant.
    max_gen = R.MAX_NEW_TOKENS[task]
    for i, ex in enumerate(examples):
        prompt = ex["input"]
        if chat:
            prompt = f"[INST]{prompt}[/INST]"       # chat model only
        ids = tok(prompt, truncation=False, return_tensors="pt").input_ids.to(device)
        ctx_len = ids.shape[1]
        out = model.generate(ids, max_new_tokens=max_gen, do_sample=False, num_beams=1,
                             use_cache=True, pad_token_id=tok.eos_token_id)
        pred = tok.decode(out[0, ctx_len:], skip_special_tokens=True)
        preds.append(dict(index=ex["index"], task=task, length=ex["length"],
                          predicted=pred, outputs=ex["outputs"], ctx_len=ctx_len))
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{task}] {i+1}/{len(examples)}  ctx={ctx_len}  pred[:50]={pred[:50]!r}", flush=True)
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--print_variant", action="store_true",
                   help="print the output directory name variant_name() would use, then exit. "
                        "The SLURM scripts call this to build their resume check, so the check and "
                        "the driver can never disagree about where predictions land.")
    p.add_argument("--kernel", required=True,
                   choices=["dense_raw", "dense_pi", "dense_yarn", "exact", "exact2",
                            "yarn_exact", "hybrid_exact", "pq_exact", "chunk_exact", "dca",
                            "pq_stream"])
    p.add_argument("--dca_chunk_size", type=int, default=3072,
                   help="DCA s; paper/ChunkLlama use 3*context_window//4")
    p.add_argument("--dca_local_window", type=int, default=512,
                   help="DCA w; ChunkLlama CODE uses context_window//8=512 (paper prose implies "
                        "c-s=1024). chunk_len = s - w is the stride.")
    p.add_argument("--model", default="base", choices=["base", "chat"])
    p.add_argument("--tau_path", default=DEFAULT_TAU)
    p.add_argument("--length", type=int, required=True, help="single context length; must match a <data_dir>/len<L> dir (v2: ruler_data_v2_flat)")
    p.add_argument("--tasks", nargs="+", default=list(R.ALL_TASKS))   # copy: argparse holds
                   # the default object itself, and mutating args.tasks would corrupt the module list
    p.add_argument("--data_dir", default="ruler_data_v2_flat",
                   help="FAITHFUL RULER v2 (built by gen_ruler_faithful.py from ruler_ref/). "
                        "The v1 tree `ruler_data` was a re-implementation and its numbers are "
                        "NOT comparable to published RULER results -- see RULER_FAITHFUL_PLAN.md")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_samples", type=int, default=0, help="0 = all in the file")
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
    p.add_argument("--chunk_size", type=int, default=16, help="chunk_exact: keys per chunk")
    p.add_argument("--n_chunk", type=int, default=0,
                   help="chunk_exact: chunks to select; 0 = derive to fill k_max exactly")
    p.add_argument("--chunk_agg", default="sum", choices=["sum", "mean"])
    p.add_argument("--select_geom", default="pi", choices=["pi", "yarn"],
                   help="pq_exact/chunk_exact/pq_stream: selection geometry (the attend stage always "
                        "uses base RoPE at repositioned coordinates)")
    p.add_argument("--pos_advance", default="max", choices=["max", "full"],
                   help="pq_stream only: how the per-block base advances. 'max' = max_i c_i (the "
                        "scheme). 'full' = the slots the block occupies, i.e. NO compression -> "
                        "native RoPE; that is the control isolating repositioning from sparsity.")
    p.add_argument("--apply_mscale", type=int, default=1, choices=[0, 1])
    p.add_argument("--decode_backend", default="triton", choices=["torch", "triton"])
    p.add_argument("--chat_template", type=int, default=0, choices=[0, 1],
                   help="wrap prompt in [INST]..[/INST] (use with --model chat)")
    args = p.parse_args()

    # Answer --print_variant before touching the tokenizer or the model: the SLURM scripts call this
    # once per (config, length) purely to locate the output dir, and it must be instant.
    if args.print_variant:
        print(variant_name(args))
        return

    device = "cuda"

    model_name = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(model_name)
    variant = variant_name(args)
    L = args.length
    outdir = os.path.join(args.out_dir, variant, f"len{L}")
    os.makedirs(outdir, exist_ok=True)

    print(f"loading model: kernel={args.kernel} model={args.model} length={L} variant={variant}", flush=True)
    model = load_model(args.kernel, model_name, L, args, device)

    for task in args.tasks:
        path_in = os.path.join(args.data_dir, f"len{L}", f"{task}.jsonl")
        if not os.path.exists(path_in):
            print(f"!! MISSING data {path_in} — run generate_ruler.py first; skipping {task}", flush=True)
            continue
        examples = [json.loads(l) for l in open(path_in) if l.strip()]
        if args.n_samples > 0:
            examples = examples[:args.n_samples]
        print(f"\n=============== TASK {task}  (len {L}, n={len(examples)}) ===============", flush=True)
        t0 = time.time()
        preds = run_task(model, tok, task, examples, device, chat=bool(args.chat_template))
        path_out = os.path.join(outdir, f"{task}.jsonl")
        with open(path_out, "w") as f:
            for r in preds:
                f.write(json.dumps(r) + "\n")
        print(f"-> {task}: {len(preds)} preds in {time.time()-t0:.0f}s  saved {path_out}", flush=True)

        # Kernel diagnostics, per task. For pq_stream this is the ONLY place the position SPAN
        # becomes visible, and span is the entire justification for repositioning: if the furthest
        # streamed key exceeds --context_window the model is extrapolating past its RoPE window,
        # exactly the failure the kernel exists to prevent. Without this a 24 h sweep would finish
        # with no way to tell whether compression happened at all. Also reports skip_frac, which is
        # the `s` the speed argument turns on. Best-effort: kernels without the accessor are silent.
        try:
            st = __import__(SPARSE_MODS[args.kernel]).get_overflow_stats()["prefill"]
            if st.get("total_cells"):
                warn = "  !! SPAN EXCEEDS CONTEXT WINDOW" if st["span_max"] > args.context_window else ""
                print(f"   [kernel] span_max={st['span_max']:.0f} (window={args.context_window})"
                      f"  span_mean={st['span_mean']:.1f}  kept_mean={st['kept_mean']:.1f}"
                      f"  skip={st['skip_frac']*100:.1f}%"
                      f"  overflow={st['overflow_frac']*100:.2f}%{warn}", flush=True)
        except Exception:
            pass

    print(f"\nDONE. predictions in {os.path.join(args.out_dir, variant)}", flush=True)


if __name__ == "__main__":
    main()
