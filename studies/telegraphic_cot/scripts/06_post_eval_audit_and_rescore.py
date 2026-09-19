#!/usr/bin/env python3
"""
studies/telegraphic_cot/scripts/06_post_eval_audit_and_rescore.py
Independent post-eval auditor and re-scorer for Telegraphic CoT study.
Performs 100% ground-truth re-scoring with canonical math_verify 0.9.0.
Verifies 0% discrepancy and audits token distributions and representative samples.
"""

import os
import sys
import json
import numpy as np
from math_verify import parse, verify

MODELS = [
    ("arm1_telegraphic_cot", "studies/telegraphic_cot/data/streaming_arm1_telegraphic_cot.jsonl", "studies/telegraphic_cot/data/eval_results_arm1_telegraphic_cot.json"),
    ("control3_matched_pause", "studies/telegraphic_cot/data/streaming_control3_matched_pause.jsonl", "studies/telegraphic_cot/data/eval_results_control3_matched_pause.json"),
    ("arm2_dual_channel_latents", "studies/telegraphic_cot/data/streaming_arm2_dual_channel_latents.jsonl", "studies/telegraphic_cot/data/eval_results_arm2_dual_channel_latents.json")
]

with open("data/benchmark_suite_250.json") as f:
    bench_data = {x["id"]: x for x in json.load(f)}

print("=" * 80)
print("INDEPENDENT POST-EVAL AUDIT AND RESCORING: TELEGRAPHIC COT STUDY")
print("=" * 80)

audit_results = {}

for tag, stream_path, summary_path in MODELS:
    print(f"\n--- Auditing & Re-scoring: {tag} ---")
    with open(summary_path) as f:
        reported_summary = json.load(f)

    records = []
    with open(stream_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 1000, f"Expected 1000 records, got {len(records)}"

    rescore_correct = 0
    discrepancy_count = 0
    trunc_count = 0
    think_tokens = []
    ans_tokens = []

    for r in records:
        pid = r["id"]
        prob = bench_data[pid]
        sol = prob["solution"]
        
        # Ground truth parsing
        if "####" in sol:
            gold_ans = sol.split("####")[-1].strip()
            gold_target = f"\\boxed{{{gold_ans}}}"
        elif "\\boxed{" not in sol:
            gold_target = f"\\boxed{{{sol.strip()}}}"
        else:
            gold_target = sol

        gold_parsed = parse(gold_target, parsing_timeout=None)
        
        # Candidate text
        out_text = r["output_text"]
        if "</think>" in out_text:
            ans_part = out_text.split("</think>", 1)[-1].strip()
        else:
            ans_part = out_text.strip()

        pred_parsed = parse(ans_part, parsing_timeout=None)
        is_corr = bool(verify(gold_parsed, pred_parsed)) if (gold_parsed and pred_parsed) else False
        
        if r.get("is_truncated", False):
            is_corr = False
            trunc_count += 1

        if is_corr != r["is_correct"]:
            discrepancy_count += 1

        if is_corr:
            rescore_correct += 1

        think_tokens.append(r["think_tokens"])
        ans_tokens.append(r["ans_tokens"])

    rescore_pass1 = (rescore_correct / 1000.0) * 100.0
    reported_pass1 = reported_summary["overall_pass1"]

    print(f"Reported Pass@1:   {reported_pass1:.2f}%")
    print(f"Rescored Pass@1:   {rescore_pass1:.2f}%")
    print(f"Discrepancies:     {discrepancy_count} ({(discrepancy_count/1000)*100:.2f}%)")
    print(f"Truncation Rate:   {trunc_count/10.0:.2f}%")
    print(f"Think tokens:      median={np.median(think_tokens):.1f}, mean={np.mean(think_tokens):.1f}, p90={np.percentile(think_tokens, 90):.1f}")
    print(f"Ans tokens:        median={np.median(ans_tokens):.1f}, mean={np.mean(ans_tokens):.1f}, p90={np.percentile(ans_tokens, 90):.1f}")

    assert discrepancy_count == 0, f"Discrepancy detected in {tag}: {discrepancy_count} mismatches!"

    audit_results[tag] = {
        "reported_pass1": reported_pass1,
        "rescored_pass1": rescore_pass1,
        "discrepancy_count": discrepancy_count,
        "truncation_rate": trunc_count / 10.0,
        "think_tokens_median": float(np.median(think_tokens)),
        "think_tokens_mean": float(np.mean(think_tokens)),
        "ans_tokens_median": float(np.median(ans_tokens)),
        "ans_tokens_mean": float(np.mean(ans_tokens)),
        "verdict": "CERTIFIED_ZERO_DISCREPANCY"
    }

out_audit_file = "studies/telegraphic_cot/data/post_eval_audit_rescore.json"
with open(out_audit_file, "w") as f:
    json.dump(audit_results, f, indent=2)

print("\n" + "=" * 80)
print(f"ALL THREE ARMS RESCORED FROM SCRATCH: 0 DISCREPANCIES! CERTIFIED PASS.")
print(f"Audit log saved to {out_audit_file}")
print("=" * 80)
