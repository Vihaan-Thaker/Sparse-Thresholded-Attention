#!/usr/bin/env python3
"""
validate_ruler_v2.py — does our faithful RULER reproduce PUBLISHED numbers?

THE GATE. Run dense_raw at len4096 ONLY (~1 h), score it, and compare against RULER's published
Llama-2-7B (base) @ 4K. Nothing else re-runs until this is green — re-running 20 kernels on data we
have not validated is how a week of GPU time gets wasted.

WHY len4096 AND dense_raw SPECIFICALLY
    dense_raw at 4096 IS Llama-2-7B base at its native window: no rope mutation, no selection, no
    tau. That is exactly the row RULER reports. Any other kernel or length is not comparable to a
    published number.

TARGETS (ruler_config.PUBLISHED_LLAMA2_7B_BASE_4K, from their Tables 12-15)
    multi-hop  (vt)        58.8    <- single task, exact comparison
    aggregation(cwe,fwe)   73.1    <- two tasks, exact comparison
    retrieval  (4 NIAH)    90.9    <- INDICATIVE ONLY: theirs averages EIGHT NIAH subtasks and we
                                      run four, so a gap here is expected and is not a failure

Tolerance is +/-5pp: at n=100 the per-task sampling error is roughly sqrt(p(1-p)/100) ~ 5pp near
50% accuracy. Their 500 samples give ~2pp. A construction bug is far larger than this -- our v1
implementation missed VT by 22.3pp and aggregation by 9.1pp.

    python validate_ruler_v2.py --pred_root ruler_results_v2
"""
import argparse
import json
import os
import sys

import ruler_config as R


def recall(pred, outputs):
    """RULER's string_match_all, per example. Identical to score_ruler.recall and to
    ruler_ref/scripts/eval/synthetic/constants.py (verified: both return 68.750000 on a shared
    fixture). Duplicated here so the gate does not depend on the scorer it is validating."""
    if not outputs:
        return 0.0
    p = str(pred).lower()
    return sum(1 for o in outputs if str(o).lower().strip() in p) / len(outputs)


def score_task(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return float("nan"), 0
    return 100.0 * sum(recall(r.get("predicted", ""), r.get("outputs", [])) for r in rows) / len(rows), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", default="ruler_results_v2")
    ap.add_argument("--variant", default="dense_raw")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--tol", type=float, default=R.VALIDATION_TOLERANCE_PP)
    args = ap.parse_args()

    d = os.path.join(args.pred_root, args.variant, f"len{args.length}")
    if not os.path.isdir(d):
        sys.exit(f"!! not found: {d}\n   Run dense_raw at len{args.length} on the v2 data first "
                 f"(sbatch validate_ruler_v2.sh).")

    print(f"variant={args.variant}  length={args.length}  root={args.pred_root}\n")
    scores, n_ex = {}, {}
    missing = []
    for t in R.ALL_TASKS:
        p = os.path.join(d, f"{t}.jsonl")
        if not os.path.exists(p):
            missing.append(t); continue
        scores[t], n_ex[t] = score_task(p)

    if missing:
        sys.exit(f"!! missing task files: {', '.join(missing)}")

    print("per task:")
    for t in R.ALL_TASKS:
        print(f"  {t:<18} {scores[t]:6.2f}   (n={n_ex[t]})")

    fam = {}
    for f in R.FAM_ORDER:
        ts = [t for t in R.ALL_TASKS if R.FAMILY[t] == f]
        fam[f] = sum(scores[t] for t in ts) / len(ts)

    pub = R.PUBLISHED_LLAMA2_7B_BASE_4K
    print("\nper family vs RULER published Llama-2-7B base @ 4K:")
    hard_fail = []
    for f in R.FAM_ORDER:
        delta = fam[f] - pub[f]
        indicative = (f == "retrieval")          # 4 of their 8 NIAH subtasks
        ok = abs(delta) <= args.tol
        tag = "INDICATIVE" if indicative else ("PASS" if ok else "FAIL")
        print(f"  {f:<12} ours {fam[f]:6.2f}   published {pub[f]:5.1f}   "
              f"delta {delta:+6.2f}   [{tag}]")
        if not ok and not indicative:
            hard_fail.append(f)

    print(f"\n  {'ALL (our 7)':<12} {sum(scores.values())/7:6.2f}   "
          f"(their 13-task avg {pub['all13']}, not directly comparable)")

    print("\n" + "=" * 72)
    if hard_fail:
        print(f"  GATE FAILED on: {', '.join(hard_fail)}  (tolerance +/-{args.tol}pp)")
        print("  Do NOT re-run the other kernels. Check, in order:")
        print("    1. did gen_ruler_faithful.sh use RULER's scripts, or did the old data leak in?")
        print("       (--data_dir must be ruler_data_v2_flat, not ruler_data)")
        print("    2. did convert_ruler_ref.py re-join input + answer_prefix? Without the answer")
        print("       cue a base model continues the text instead of answering, and every score")
        print("       collapses -- this is the single most likely cause of a large uniform miss.")
        print("    3. is MAX_NEW_TOKENS RULER's tokens_to_generate (niah 128 / vt 30 / cwe 120 /")
        print("       fwe 50)? A short budget truncates multi-answer tasks.")
        print("    4. essay haystack present? nltk punkt resolves on the compute node?")
    else:
        print(f"  GATE PASSED (tolerance +/-{args.tol}pp) — the v2 data reproduces published RULER.")
        print("  Safe to re-run the kernel sweep against ruler_data_v2_flat.")
    print("=" * 72, flush=True)
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
