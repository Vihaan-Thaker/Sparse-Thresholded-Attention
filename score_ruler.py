#!/usr/bin/env python3
"""
score_ruler.py  --  score run_ruler.py predictions with RULER's string-match accuracy.

Self-contained (no hub dependency): RULER scoring is recall-based substring matching — for each
example, the fraction of gold `outputs` that appear (case-insensitive) in the prediction; the task
score is 100 x mean recall over examples. For single-answer tasks this is exactly 0/100 per example.

Run LOCALLY after downloading the prediction tree. Layout expected:
  <pred_root>/<variant>/len<L>/<task>.jsonl

  python score_ruler.py --pred_root ruler_results
  # prints accuracy per (variant, length, task) + a per-length mean, and writes metrics.json per variant.
"""
import os, json, glob, argparse, re
from collections import defaultdict


def recall(pred, outputs):
    if not outputs:
        return 0.0
    p = pred.lower()
    hit = sum(1 for o in outputs if str(o).lower().strip() in p)
    return hit / len(outputs)


def score_file(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return float("nan"), 0
    scores = [recall(r.get("predicted", ""), r.get("outputs", [])) for r in rows]
    return 100.0 * sum(scores) / len(scores), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True, help="dir containing <variant>/len<L>/<task>.jsonl")
    args = ap.parse_args()

    variants = sorted(d for d in os.listdir(args.pred_root)
                      if os.path.isdir(os.path.join(args.pred_root, d)))
    # table[variant][length][task] = acc
    table = defaultdict(lambda: defaultdict(dict))
    all_lengths, all_tasks = set(), set()
    for var in variants:
        for jf in glob.glob(os.path.join(args.pred_root, var, "len*", "*.jsonl")):
            m = re.search(r"len(\d+)[/\\]([^/\\]+)\.jsonl$", jf)
            if not m:
                continue
            L, task = int(m.group(1)), m.group(2)
            acc, n = score_file(jf)
            table[var][L][task] = acc
            all_lengths.add(L); all_tasks.add(task)

    lengths = sorted(all_lengths); tasks = sorted(all_tasks)
    for var in variants:
        metrics = {}
        for L in lengths:
            per = table[var].get(L, {})
            if per:
                metrics[f"len{L}"] = {**{t: round(per.get(t, float('nan')), 2) for t in tasks},
                                      "MEAN": round(sum(per.values()) / len(per), 2)}
        with open(os.path.join(args.pred_root, var, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # print per-length MEAN table (variant x length) then a detailed per-task block
    print("\n===== RULER accuracy — per-length MEAN (variant x length) =====")
    print("%-40s " % "variant" + " ".join(f"{('len'+str(L)):>9}" for L in lengths))
    for var in variants:
        cells = []
        for L in lengths:
            per = table[var].get(L, {})
            cells.append(f"{(sum(per.values())/len(per)):>9.2f}" if per else f"{'-':>9}")
        print("%-40s " % var[:40] + " ".join(cells))

    for L in lengths:
        print(f"\n----- per-task accuracy @ len{L} -----")
        print("%-40s " % "variant" + " ".join(f"{t[:11]:>11}" for t in tasks))
        for var in variants:
            per = table[var].get(L, {})
            if not per:
                continue
            print("%-40s " % var[:40] + " ".join(f"{per.get(t, float('nan')):>11.2f}" for t in tasks))

    print("\nwrote metrics.json into each variant dir under", args.pred_root)


if __name__ == "__main__":
    main()
