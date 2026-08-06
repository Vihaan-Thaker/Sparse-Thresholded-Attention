#!/usr/bin/env python3
"""
longbench_lengths.py  --  measure the token-length distribution of the LongBench prompts we run,
to quantify how much context truncation to --max_length actually drops. CPU-only (tokenizer, no GPU).

Builds each prompt exactly as run_longbench.py does (longbench_config.build_prompt), tokenizes with the
Llama-2 tokenizer, and reports per task: n, min / median / mean / p95 / MAX token length, #samples over
the cap, and mean tokens dropped (over the truncated samples and over all). Also prints the single
longest prompt across the subset.
"""
import argparse
import numpy as np
from transformers import AutoTokenizer
import longbench_config as lb

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"])
    ap.add_argument("--max_length", type=int, default=24576)
    ap.add_argument("--dataset_id", default="THUDM/LongBench")
    args = ap.parse_args()

    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    cap = args.max_length

    print(f"{'task':<16}{'n':>5}{'min':>8}{'med':>8}{'mean':>8}{'p95':>9}{'MAX':>9}"
          f"{'>cap':>7}{'drop/trunc':>12}{'drop/all':>10}", flush=True)
    glob_max = (0, None, None)
    all_lens = []
    for task in args.tasks:
        ds = load_dataset(args.dataset_id, task, split="test", trust_remote_code=True)
        lens = []
        for ex in ds:
            prompt = lb.build_prompt(task, ex)
            n = len(tok(prompt, truncation=False).input_ids)
            lens.append(n)
            if n > glob_max[0]:
                glob_max = (n, task, prompt[:120])
        lens = np.array(lens); all_lens.extend(lens.tolist())
        over = lens[lens > cap]
        drop_trunc = float((over - cap).mean()) if over.size else 0.0
        drop_all = float(np.clip(lens - cap, 0, None).mean())
        print(f"{task:<16}{len(lens):>5}{lens.min():>8}{int(np.median(lens)):>8}{int(lens.mean()):>8}"
              f"{int(np.percentile(lens,95)):>9}{lens.max():>9}"
              f"{int((lens>cap).sum()):>7}{drop_trunc:>12.0f}{drop_all:>10.0f}", flush=True)

    all_lens = np.array(all_lens)
    print(f"\nSUBSET: {len(all_lens)} prompts | over {cap}: {int((all_lens>cap).sum())} "
          f"({100*(all_lens>cap).mean():.1f}%) | overall max = {all_lens.max()} tokens", flush=True)
    print(f"longest prompt: {glob_max[0]} tokens in '{glob_max[1]}'\n  head: {glob_max[2]!r}", flush=True)


if __name__ == "__main__":
    main()
