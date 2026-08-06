#!/usr/bin/env python3
"""
gen_ruler_faithful.py — generate RULER data by running NVIDIA's OWN generators.

WHY THIS EXISTS (and why we do NOT port their logic)
-----------------------------------------------------
Our previous generate_ruler.py was a re-implementation from the paper's tables and diverged from
RULER in ways that made our numbers incomparable to published results:

  * needle DEPTH was fixed (random.Random(12345)) -> niah_single put its needle at exactly 50% depth
    in all 100 examples; RULER samples a fresh depth per example from a 40-point percentile grid
  * haystack was the noise sentence everywhere; RULER uses Paul Graham essays for 10 of 13 tasks
  * VT had 3 distractor chains and no in-context example; RULER has 1 chain, 4 hops, and an ICL demo
  * CWE used common x12 vs singletons; RULER uses 30x vs 3x, from REAL words (wonderwords)
  * FWE used fixed counts (30,24,18); RULER samples from a Zeta/Zipf distribution, alpha=2.0
  * tokens_to_generate was wrong for EVERY task (see below), which shifts every input length

Rather than re-implement ~500 lines and risk a silent constant being wrong, this driver invokes
their scripts unchanged. The generator IS theirs; we only choose the lengths and convert the output
format afterwards (convert_ruler_ref.py).

tokens_to_generate, from their scripts/data/synthetic/constants.py -- ALL of ours were wrong:
    niah 128 (ours 32/96) | variable_tracking 30 (ours 48)
    common_words_extraction 120 (ours 160) | freq_words_extraction 50 (ours 48)

TASK NAMES ARE THEIRS. Our old `niah_single` is their `niah_single_1`, our `niah_multikey` is their
`niah_multikey_1`. Using their names makes the mapping to their published per-category tables
unambiguous. Downstream (run_ruler.py TASKS lists, the notebook FAMILY map, score tables) must be
updated to match -- see RULER_FAITHFUL_PLAN.md.

Run on Prajna, where ruler_ref/ is cloned and PaulGrahamEssays.json is downloaded.
"""
import argparse
import json
import os
import subprocess
import sys

# Our 7 tasks, expressed as RULER's own task names + the exact args from their scripts/synthetic.yaml.
# `script` is the generator to invoke; `key` indexes their constants.TASKS for template/tokens.
TASKS = {
    "niah_single_1":   dict(script="niah.py", key="niah",
                            args=["--type_haystack", "noise", "--type_needle_k", "words",
                                  "--type_needle_v", "numbers",
                                  "--num_needle_k", "1", "--num_needle_v", "1", "--num_needle_q", "1"]),
    "niah_multikey_1": dict(script="niah.py", key="niah",
                            args=["--type_haystack", "essay", "--type_needle_k", "words",
                                  "--type_needle_v", "numbers",
                                  "--num_needle_k", "4", "--num_needle_v", "1", "--num_needle_q", "1"]),
    "niah_multivalue": dict(script="niah.py", key="niah",
                            args=["--type_haystack", "essay", "--type_needle_k", "words",
                                  "--type_needle_v", "numbers",
                                  "--num_needle_k", "1", "--num_needle_v", "4", "--num_needle_q", "1"]),
    # num_needle_k=1 with num_needle_q=4 is NOT a typo -- niah.py line 77 does
    #     args.num_needle_k = max(args.num_needle_k, args.num_needle_q)
    # so k becomes 4. Copied verbatim from their synthetic.yaml.
    "niah_multiquery": dict(script="niah.py", key="niah",
                            args=["--type_haystack", "essay", "--type_needle_k", "words",
                                  "--type_needle_v", "numbers",
                                  "--num_needle_k", "1", "--num_needle_v", "1", "--num_needle_q", "4"]),
    "vt":              dict(script="variable_tracking.py", key="variable_tracking",
                            args=["--type_haystack", "noise", "--num_chains", "1", "--num_hops", "4",
                                  "--add_fewshot"]),
    "cwe":             dict(script="common_words_extraction.py", key="common_words_extraction",
                            args=["--freq_cw", "30", "--freq_ucw", "3", "--num_cw", "10"]),
    "fwe":             dict(script="freq_words_extraction.py", key="freq_words_extraction",
                            args=["--alpha", "2.0"]),
}

# Single source of truth. TASKS above carries the CLI ARGS (which are ours to choose, copied from
# their synthetic.yaml); the task NAME SET must agree with ruler_config or run_ruler.py will be
# handed a task it has no generation budget for.
import ruler_config as R
FAMILY = R.FAMILY
assert set(TASKS) == set(R.ALL_TASKS), (
    f"task names disagree: gen_ruler_faithful has {sorted(TASKS)}, "
    f"ruler_config.ALL_TASKS has {sorted(R.ALL_TASKS)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler_ref", required=True,
                    help="path to the cloned RULER repo (contains scripts/data/synthetic/)")
    ap.add_argument("--out_dir", default="ruler_data_v2")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[1024, 2048, 4096, 6144, 8192, 10240, 12288])
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--tokenizer_type", default="hf")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    syn = os.path.join(args.ruler_ref, "scripts", "data", "synthetic")
    if not os.path.isdir(syn):
        sys.exit(f"!! not found: {syn}  (is --ruler_ref pointing at the cloned repo?)")
    essay = os.path.join(syn, "json", "PaulGrahamEssays.json")
    if not os.path.exists(essay):
        sys.exit(f"!! missing {essay}\n   run json/download_paulgraham_essay.py on the LOGIN node "
                 f"(compute nodes are offline). Three tasks need the essay haystack.")

    # their templates + tokens_to_generate, read from THEIR constants.py (never hard-coded here)
    sys.path.insert(0, syn)
    from constants import TASKS as RTASKS          # noqa: E402

    os.makedirs(args.out_dir, exist_ok=True)
    failed = []

    for L in args.lengths:
        for task in args.tasks:
            spec = TASKS[task]
            k = spec["key"]
            # Templates['base'] == "{task_template}", i.e. no chat wrapper for a base model, so the
            # prompt is exactly template + answer_prefix. niah.py splits the prefix back out into its
            # own field; convert_ruler_ref.py re-joins them.
            template = RTASKS[k]["template"] + RTASKS[k]["answer_prefix"]
            save_dir = os.path.join(args.out_dir, f"len{L}")
            out_jsonl = os.path.join(save_dir, task, "validation.jsonl")
            if os.path.exists(out_jsonl) and not args.overwrite:
                print(f"skip (exists) {out_jsonl}", flush=True)
                continue

            cmd = [sys.executable, os.path.join(syn, spec["script"]),
                   "--save_dir", save_dir,
                   "--save_name", task,
                   "--subset", "validation",
                   "--tokenizer_path", args.tokenizer,
                   "--tokenizer_type", args.tokenizer_type,
                   "--max_seq_length", str(L),
                   "--tokens_to_generate", str(RTASKS[k]["tokens_to_generate"]),
                   "--num_samples", str(args.num_samples),
                   "--random_seed", "42",
                   "--template", template] + spec["args"]

            print(f"\n===== len{L} / {task} "
                  f"(tokens_to_generate={RTASKS[k]['tokens_to_generate']}) =====", flush=True)
            r = subprocess.run(cmd, cwd=syn)
            if r.returncode != 0 or not os.path.exists(out_jsonl):
                print(f"!! FAILED len{L}/{task} (rc={r.returncode})", flush=True)
                failed.append(f"len{L}/{task}")

    print("\n" + "=" * 70)
    if failed:
        print(f"  {len(failed)} FAILED: " + ", ".join(failed))
    else:
        print(f"  all {len(args.lengths) * len(args.tasks)} (length, task) pairs generated")
    print(f"  raw RULER output -> {args.out_dir}/len<L>/<task>/validation.jsonl")
    print("  next: convert_ruler_ref.py to turn these into our jsonl format")
    print("=" * 70, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
