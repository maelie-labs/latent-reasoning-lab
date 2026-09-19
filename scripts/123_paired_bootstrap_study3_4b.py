#!/usr/bin/env python3
"""
scripts/123_paired_bootstrap_study3_4b.py
Paired Hierarchical Bootstrapping for Study 3 on 4B Models.

Protocol:
- B = 10,000 bootstrap iterations
- Paired by (problem_id, seed) query across N=1,000 samples
- Computes two-sided empirical p-values and 95% percentile confidence intervals
- Evaluates:
  1. Delta_dense: Telegraphic CoT vs. Headers-Only on Qwen3-4B
  2. Delta_hybrid: Telegraphic CoT vs. Headers-Only on Qwen3.5-4B
  3. Compression Penalty Attenuation vs. 1.7B baseline (-8.00%)
"""

import os
import sys
import json
import argparse
import numpy as np

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
    parser = argparse.ArgumentParser(description="Paired Hierarchical Bootstrap Analysis for Study 3 on 4B")
    parser.add_argument("--candidate", type=str, required=True, help="Streaming JSONL for candidate arm (e.g. Telegraphic)")
    parser.add_argument("--control", type=str, required=True, help="Streaming JSONL for control arm (e.g. Headers-Only)")
    parser.add_argument("--model_tag", type=str, required=True, choices=["qwen3_4b", "qwen3_5_4b"])
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--b_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"data/paired_bootstrap_study3_{args.model_tag}.json"

    arm_cand = load_streaming_results(args.candidate)
    arm_ctrl = load_streaming_results(args.control)

    print("=" * 80)
    print(f"PAIRED HIERARCHICAL BOOTSTRAP (B={args.b_samples}): {args.model_tag.upper()}")
    print(f"Candidate: {args.candidate}")
    print(f"Control:   {args.control}")
    print("=" * 80)

    # 1. Overall Delta
    overall = paired_bootstrap(arm_cand, arm_ctrl, B=args.b_samples, seed=args.seed)

    # 2. GSM8K Arithmetic Delta
    gsm8k = paired_bootstrap(
        arm_cand, arm_ctrl,
        filter_fn=lambda x: x.get("dataset") == "gsm8k",
        B=args.b_samples, seed=args.seed
    )

    # 3. MATH-500 Delta
    math500 = paired_bootstrap(
        arm_cand, arm_ctrl,
        filter_fn=lambda x: x.get("dataset") in ["math500", "math"],
        B=args.b_samples, seed=args.seed
    )

    # 4. MATH Hard (Levels 3-5) Delta
    math_hard = paired_bootstrap(
        arm_cand, arm_ctrl,
        filter_fn=lambda x: x.get("dataset") in ["math500", "math"] and x.get("level", 0) >= 3,
        B=args.b_samples, seed=args.seed
    )

    report = {
        "model_tag": args.model_tag,
        "candidate_file": args.candidate,
        "control_file": args.control,
        "overall": overall,
        "gsm8k": gsm8k,
        "math500": math500,
        "math_hard_l3_5": math_hard,
        "historical_1_7b_reference_delta": -8.00
    }

    with open(args.output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {args.output_file}")
    print(f"Overall Delta:     {overall['observed_delta']:+.2f}% (95% CI: {overall['ci_95']}, p={overall['p_value']})")
    print(f"GSM8K Delta:       {gsm8k['observed_delta']:+.2f}% (95% CI: {gsm8k['ci_95']}, p={gsm8k['p_value']})")
    print(f"MATH-500 Delta:    {math500['observed_delta']:+.2f}% (95% CI: {math500['ci_95']}, p={math500['p_value']})")
    print(f"MATH Hard Delta:   {math_hard['observed_delta']:+.2f}% (95% CI: {math_hard['ci_95']}, p={math_hard['p_value']})")

if __name__ == "__main__":
    main()
