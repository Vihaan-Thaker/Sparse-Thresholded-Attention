#!/usr/bin/env python3
"""
ruler_config.py  --  metadata for the FAITHFUL RULER benchmark (v2).

WHAT CHANGED IN v2
------------------
This file used to hold task specs, prompt templates and word pools for a RULER *re-implementation*
(generate_ruler.py). That re-implementation diverged from the real benchmark badly enough that our
numbers were not comparable to anything published:

  * needle DEPTH was fixed -- niah_single put its needle at exactly 50% depth in all 100 examples;
    RULER samples a fresh depth per example from a 40-point percentile grid
  * the noise sentence was the haystack for every task; RULER uses Paul Graham ESSAYS for 10 of 13
  * VT had 3 distractor chains and no in-context demo; RULER has 1 chain, 4 hops, and an ICL demo
  * CWE used common x12 against singletons, from pseudo-words; RULER uses 30x vs 3x, from REAL words
  * FWE used fixed counts (30,24,18); RULER samples a Zeta/Zipf distribution, alpha=2.0
  * tokens_to_generate was wrong for EVERY task, which shifts every input length

Generation is now done by NVIDIA's OWN scripts (gen_ruler_faithful.py -> ruler_ref/), so this module
no longer describes HOW data is built -- only what the tasks are, what generation budget each gets,
and what the published reference numbers are. There are NO templates here any more; theirs live in
ruler_ref/scripts/data/synthetic/constants.py and must not be duplicated.

The v1 file is preserved as ruler_config_v1_legacy.py purely so old results stay interpretable.

TASK NAMES ARE RULER'S. Our old `niah_single` is their `niah_single_1`; our old `niah_multikey` is
their `niah_multikey_1`. Using their names keeps the mapping to their published per-category tables
unambiguous.
"""

# ── the 7 tasks we run, out of RULER's 13 ────────────────────────────────────
# Omitted: niah_single_2/3 and niah_multikey_2/3 (extra NIAH subtasks, incl. the UUID ones) and
# qa_1/qa_2 (need SQuAD/HotpotQA, i.e. external corpora, and the compute nodes are offline).
# We therefore cover 3 of RULER's 4 capability families.
ALL_TASKS = [
    "niah_single_1",     # S-NIAH-1  noise haystack, word key, number value   (~passkey retrieval)
    "niah_multikey_1",   # MK-NIAH-1 essay haystack, 4 keys (1 target + 3 distractors)
    "niah_multivalue",   # MV-NIAH   essay haystack, 1 key with 4 values
    "niah_multiquery",   # MQ-NIAH   essay haystack, 4 keys all queried
    "vt",                # VT        noise haystack, 1 chain, 4 hops -> 5 variables
    "cwe",               # CWE       10 common words x30 vs uncommon x3
    "fwe",               # FWE       Zipf alpha=2.0 over synthetic 6-char words
]

FAMILY = {
    "niah_single_1": "retrieval", "niah_multikey_1": "retrieval",
    "niah_multivalue": "retrieval", "niah_multiquery": "retrieval",
    "vt": "multi-hop",
    "cwe": "aggregation", "fwe": "aggregation",
}
FAM_ORDER = ["retrieval", "multi-hop", "aggregation"]

# ── generation budget: RULER's `tokens_to_generate` ──────────────────────────
# Copied from ruler_ref/scripts/data/synthetic/constants.py. EVERY ONE of these differs from the
# v1 values we used (niah 32/96, vt 48, cwe 160, fwe 48). It is not just the answer budget: RULER
# fits the prompt so that len(input_tokens) + tokens_to_generate <= max_seq_length, so changing it
# changes the input length too. Do not "tidy" these.
MAX_NEW_TOKENS = {
    "niah_single_1": 128, "niah_multikey_1": 128,
    "niah_multivalue": 128, "niah_multiquery": 128,   # all four are RULER's 'niah' task
    "vt": 30,                                          # 'variable_tracking'
    "cwe": 120,                                        # 'common_words_extraction'
    "fwe": 50,                                         # 'freq_words_extraction'
}

# ── published reference numbers: Llama-2-7B (base) at 4K ─────────────────────
# RULER (Hsieh et al., COLM 2024) Tables 12-15. Our dense_raw at len4096 IS that model at its native
# window, so these are direct validation targets. Their Retrieval row averages EIGHT NIAH subtasks
# and we run four of them, so that one is indicative rather than exact.
#
# Nothing below 4K is validatable: their §8 says "we did not include these results in this paper."
# len1024/len2048 are ours alone -- fine as internal oracles (the DCA/chunk dense-reduction checks),
# never as an external claim.
PUBLISHED_LLAMA2_7B_BASE_4K = {
    "retrieval":   90.9,   # Table 13, 8 NIAH tasks
    "multi-hop":   58.8,   # Table 14, VT
    "aggregation": 73.1,   # Table 15, CWE + FWE
    "all13":       79.4,   # Table 12, all 13 tasks
}

# n=100 (we keep RULER's protocol but not their 500 samples) gives roughly +/-5pp of sampling noise
# per task; their 500 gives ~+/-2pp. Used by the validation gate.
VALIDATION_TOLERANCE_PP = 5.0

# CWE IS LENGTH-DEPENDENT IN RULER'S OWN CODE -- not a defect of ours.
# common_words_extraction.py:
#     if args.max_seq_length < 4096: context, answer = get_example(num_words, 6, 1, num_cw)
#     else:                          context, answer = get_example(num_words, freq_cw, freq_ucw, num_cw)
# so at 1k/2k the frequencies are 6x/1x, not 30x/3x. CWE at those lengths is a DIFFERENT task from
# CWE at 4k+, and only the 4k point is comparable to the published 73.1.
CWE_SHORT_CONTEXT_THRESHOLD = 4096
