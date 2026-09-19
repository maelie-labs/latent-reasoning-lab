#!/usr/bin/env python3
"""
studies/tool_grounded_scratchpad/scripts/05_paired_bootstrap_scratchpad.py
Paired Hierarchical Bootstrap Analysis for Study 4 Tool Scratchpad.

Compares Study 4 Tool Scratchpad against:
1. Control 1: Headers-Only Control (streaming_study3_headers_only_qwen3_4b.jsonl)
Computes observed delta, 95% bootstrap CI, and two-sided p-value across N=1,000 queries.
"""

import os
import sys
import json
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

def load_streaming_results(jsonl_path):
    records = {}
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid = d["problem_id"]
            seed = d["seed"]
            key = (pid, seed)
            ds = d.get("dataset", "unknown")
            if ds == "unknown" or not ds:
                ds = "gsm8k" if pid.startswith("gsm8k") else "math500"
            records[key] = {
                "is_correct": bool(d.get("is_correct", False)),
                "is_truncated": bool(d.get("is_truncated", False)),
                "level": d.get("level", 0),
                "dataset": ds,
                "subject": d.get("subject", "unknown"),
                "tokens": d.get("tokens", {})
            }
    return records

def paired_bootstrap(arm_a, arm_b, filter_fn=None, B=10000, seed=42):
    keys = sorted(list(set(arm_a.keys()) & set(arm_b.keys())))
    if filter_fn:
        keys = [k for k in keys if filter_fn(arm_a[k])]

    n = len(keys)
    if n == 0:
        return {"observed_delta": 0.0, "ci_95": [0.0, 0.0], "p_value": 1.0, "n": 0}

    diffs = np.array([float(arm_a[k]["is_correct"]) - float(arm_b[k]["is_correct"]) for k in keys])
    obs_delta = float(np.mean(diffs)) * 100.0

    rng = np.random.RandomState(seed)
    boot_indices = rng.randint(0, n, size=(B, n))
    boot_means = np.mean(diffs[boot_indices], axis=1) * 100.0

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    p_gt = np.mean(boot_means >= 0.0)
    p_lt = np.mean(boot_means <= 0.0)
    p_value = min(1.0, 2.0 * min(p_gt, p_lt))

    return {
        "observed_delta": round(obs_delta, 2),
        "ci_95": [round(float(ci_low), 2), round(float(ci_high), 2)],
        "p_value": round(float(p_value), 4),
        "n": n
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, default="data/streaming_study4_tool_scratchpad_qwen3_4b.jsonl")
    parser.add_argument("--control", type=str, default="data/streaming_study3_headers_only_qwen3_4b.jsonl")
    parser.add_argument("--output_file", type=str, default="data/paired_bootstrap_study4_tool_scratchpad_qwen3_4b.json")
    parser.add_argument("--b_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    candidate_path = os.path.join(PROJECT_ROOT, args.candidate)
    control_path = os.path.join(PROJECT_ROOT, args.control)
    output_path = os.path.join(PROJECT_ROOT, args.output_file)

    print("=" * 80)
    print("PAIRED HIERARCHICAL BOOTSTRAP: STUDY 4 TOOL SCRATCHPAD")
    print(f"Candidate: {candidate_path}")
    print(f"Control:   {control_path}")
    print(f"B = {args.b_samples} resamples | Seed: {args.seed}")
    print("=" * 80)

    arm_cand = load_streaming_results(candidate_path)
    arm_ctrl = load_streaming_results(control_path)

    overall = paired_bootstrap(arm_cand, arm_ctrl, B=args.b_samples, seed=args.seed)
    gsm8k = paired_bootstrap(arm_cand, arm_ctrl, filter_fn=lambda x: x["dataset"] == "gsm8k", B=args.b_samples, seed=args.seed)
    math500 = paired_bootstrap(arm_cand, arm_ctrl, filter_fn=lambda x: x["dataset"] == "math500", B=args.b_samples, seed=args.seed)
    math_hard = paired_bootstrap(arm_cand, arm_ctrl, filter_fn=lambda x: x["dataset"] == "math500" and x["level"] in [3, 4, 5], B=args.b_samples, seed=args.seed)

    print("\n" + "=" * 80)
    print("BOOTSTRAP RESULTS SUMMARY:")
    print(f"Overall Delta (N={overall['n']}):       {overall['observed_delta']:+.2f}% (95% CI: [{overall['ci_95'][0]:+.2f}%, {overall['ci_95'][1]:+.2f}%], p={overall['p_value']:.4f})")
    print(f"GSM8K Arithmetic (N={gsm8k['n']}):     {gsm8k['observed_delta']:+.2f}% (95% CI: [{gsm8k['ci_95'][0]:+.2f}%, {gsm8k['ci_95'][1]:+.2f}%], p={gsm8k['p_value']:.4f})")
    print(f"MATH-500 Math (N={math500['n']}):       {math500['observed_delta']:+.2f}% (95% CI: [{math500['ci_95'][0]:+.2f}%, {math500['ci_95'][1]:+.2f}%], p={math500['p_value']:.4f})")
    print(f"MATH Hard L3-5 (N={math_hard['n']}):     {math_hard['observed_delta']:+.2f}% (95% CI: [{math_hard['ci_95'][0]:+.2f}%, {math_hard['ci_95'][1]:+.2f}%], p={math_hard['p_value']:.4f})")
    print("=" * 80)

    results = {
        "candidate": args.candidate,
        "control": args.control,
        "overall": overall,
        "gsm8k": gsm8k,
        "math500": math500,
        "math_hard": math_hard
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved bootstrap analysis to {output_path}")

if __name__ == "__main__":
    main()
