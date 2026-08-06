#!/usr/bin/env python3
"""
convert_ruler_ref.py — RULER's native output -> the jsonl format run_ruler.py expects.

RULER writes  <out>/len<L>/<task>/validation.jsonl  with fields:
    index, input, outputs, length, length_w_model_temp, answer_prefix, token_position_answer

run_ruler.py reads  ruler_data_v2_flat/len<L>/<task>.jsonl  and uses:
    input    -- the FULL prompt, answer cue included
    outputs  -- list of gold answer strings
    index, length -- carried into the prediction records

THE ONE SUBSTANTIVE STEP: niah.py (and the others) SPLIT the answer prefix back out of the prompt
after building it:

    answer_prefix_index = input_text.rfind(TASKS['niah']['answer_prefix'][:10])
    answer_prefix = input_text[answer_prefix_index:]
    input_text = input_text[:answer_prefix_index]

so `input` alone is missing the cue that makes a BASE model answer instead of continuing the text.
We re-join them: prompt = input + answer_prefix. Dropping the prefix would tank every score and look
like a kernel failure.

Also re-checks the token budget: RULER asserts len(tokens(input)) + tokens_to_generate <=
max_seq_length, where `length` in their record is exactly that sum. We verify the re-joined prompt
still fits, because the prefix is part of what they counted.
"""
import argparse
import glob
import json
import os
import sys

# Single source of truth -- do NOT keep a local copy. Three divergent FAMILY dicts is exactly how
# a task silently lands in the wrong category and the family averages stop matching the notebook.
import ruler_config as R
FAMILY = R.FAMILY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="ruler_data_v2",
                    help="dir holding len<L>/<task>/validation.jsonl from gen_ruler_faithful.py")
    ap.add_argument("--out_dir", default="ruler_data_v2_flat",
                    help="dir to write len<L>/<task>.jsonl in our format")
    ap.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--check_tokens", action="store_true",
                    help="tokenize each prompt and report the true length (slow, worth it once)")
    args = ap.parse_args()

    tok = None
    if args.check_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    srcs = sorted(glob.glob(os.path.join(args.raw_dir, "len*", "*", "validation.jsonl")))
    if not srcs:
        sys.exit(f"!! no validation.jsonl under {args.raw_dir}/len*/<task>/ — run "
                 f"gen_ruler_faithful.py first")

    n_files = n_rows = 0
    problems = []
    for src in srcs:
        parts = src.split(os.sep)
        task, lentag = parts[-2], parts[-3]
        L = int(lentag.replace("len", ""))
        rows = [json.loads(l) for l in open(src) if l.strip()]
        if not rows:
            problems.append(f"{lentag}/{task}: EMPTY"); continue

        out_rows, tlens = [], []
        for i, r in enumerate(rows):
            prefix = r.get("answer_prefix", "")
            if not prefix:
                problems.append(f"{lentag}/{task}[{i}]: no answer_prefix field")
            prompt = r["input"] + prefix          # <- the re-join; see module docstring
            outs = r["outputs"]
            if not outs:
                problems.append(f"{lentag}/{task}[{i}]: empty outputs")
            # index=i FROM ENUMERATE, deliberately NOT r["index"].
            # RULER's own `index` field is NOT a sample index: niah.py shadows its loop variable --
            #     for index in tqdm(range(num_samples)):
            #         ...
            #         index = input_text.find(answer[0])      <- rebound to a CHARACTER OFFSET
            #         formatted_output = {'index': index, ...}
            # so their field is the char position of the answer in the prompt. Using it here would
            # give duplicate/meaningless indices in the prediction records.
            rec = dict(index=i, task=task, length=L, input=prompt, outputs=outs,
                       family=FAMILY.get(task, "?"),
                       ruler_length=r.get("length"),
                       token_position_answer=r.get("token_position_answer"))
            if tok is not None:
                n = len(tok(prompt, add_special_tokens=False).input_ids)
                rec["input_tokens"] = n
                tlens.append(n)
                if n >= L:
                    problems.append(f"{lentag}/{task}[{i}]: prompt {n} tok >= nominal {L}")
            out_rows.append(rec)

        dst_dir = os.path.join(args.out_dir, lentag)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{task}.jsonl")
        with open(dst, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        med = f" median {sorted(tlens)[len(tlens)//2]} tok" if tlens else ""
        print(f"{dst}  ({len(out_rows)} ex{med})", flush=True)
        n_files += 1
        n_rows += len(out_rows)

    print("\n" + "=" * 70)
    print(f"  wrote {n_files} task files, {n_rows} examples -> {args.out_dir}")
    if problems:
        print(f"  !! {len(problems)} problems:")
        for p in problems[:20]:
            print("     " + p)
        if len(problems) > 20:
            print(f"     ... and {len(problems)-20} more")
    else:
        print("  no problems found")
    print("=" * 70, flush=True)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
