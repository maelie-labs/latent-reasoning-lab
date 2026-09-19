#!/usr/bin/env python3
"""
studies/telegraphic_cot/scripts/04_paired_bootstrap_telegraphic.py
Phase 4: Paired Hierarchical Bootstrapping for Telegraphic CoT Study.

Computes:
- Delta1: Arm 1 (Telegraphic CoT) vs Control 3 (Matched Pause)
- Delta2: Arm 1 (Telegraphic CoT) vs Control 2 (Base Direct)
- Delta3: Arm 2 (Dual-Channel Latents) vs Arm 1 (Telegraphic CoT)
- Delta4: Arm 1 (Telegraphic CoT) vs Control 1 (Verbose CoT)

Protocol:
- B = 10,000 bootstrap resamples
- Paired by (problem_id, seed) query
- Two-sided empirical p-value and 95% percentile confidence intervals
"""

import os
import sys
import json
import argparse
import numpy as np

def load_results(jsonl_path):
    records = {}
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid = d.get("id") or d.get("problem_id")
            key = (pid, d["seed"])
            bench = d.get("benchmark", "")
            dset = d.get("dataset", "gsm8k" if bench == "GSM8K" else ("math" if bench == "MATH-500" else ""))
            records[key] = {
                "is_correct": bool(d.get("is_correct", False)),
                "level": d.get("level", 1),
                "stratum": d.get("stratum", ""),
                "benchmark": bench,
                "dataset": dset
            }
    return records

def bootstrap_paired_delta(arm_a, arm_b, filter_fn=None, B=10000, seed=42):
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

    # Two-sided empirical p-value under null hypothesis (mean = 0)
    p_gt = np.mean(boot_means >= 0.0)
    p_lt = np.mean(boot_means <= 0.0)
    p_value = min(1.0, 2.0 * min(p_gt, p_lt))

    return {
        "observed_delta": obs_delta,
        "ci_95": [float(ci_low), float(ci_high)],
        "p_value": float(p_value),
        "n": n
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streaming_a", type=str, required=True, help="Streaming JSONL for candidate arm")
    parser.add_argument("--streaming_b", type=str, required=True, help="Streaming JSONL for comparator arm")
    parser.add_argument("--name_a", type=str, default="Arm A")
    parser.add_argument("--name_b", type=str, default="Arm B")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arm_a = load_results(args.streaming_a)
    arm_b = load_results(args.streaming_b)

    # 1. Overall comparison
    overall = bootstrap_paired_delta(arm_a, arm_b, B=args.bootstrap_samples, seed=args.seed)

    # 2. Hard MATH stratum (Levels 3, 4, 5)
    math_hard = bootstrap_paired_delta(
        arm_a, arm_b,
        filter_fn=lambda x: (x.get("benchmark") == "MATH-500" or x.get("dataset") == "math") and (x.get("level") in [3, 4, 5] or x.get("stratum") in ["Level 3", "Level 4", "Level 5"]),
        B=args.bootstrap_samples,
        seed=args.seed
    )

    # 3. GSM8K arithmetic stratum
    gsm8k = bootstrap_paired_delta(
        arm_a, arm_b,
        filter_fn=lambda x: x.get("benchmark") == "GSM8K" or x.get("dataset") == "gsm8k",
        B=args.bootstrap_samples,
        seed=args.seed
    )

    res = {
        "comparison": f"{args.name_a} vs {args.name_b}",
        "total_pairs": overall["n"],
        "overall": overall,
        "math_hard_l3_5": math_hard,
        "gsm8k": gsm8k
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(res, f, indent=2)

    print("=" * 80)
    print(f"PAIRED BOOTSTRAP: {args.name_a} vs. {args.name_b}")
    print(f"Total Pairs:               {overall['n']}")
    print(f"Overall Delta:             {overall['observed_delta']:+.2f}% [95% CI: {overall['ci_95'][0]:+.2f}%, {overall['ci_95'][1]:+.2f}%], p = {overall['p_value']:.4f}")
    print(f"MATH Hard (L3-5):          {math_hard['observed_delta']:+.2f}% [95% CI: {math_hard['ci_95'][0]:+.2f}%, {math_hard['ci_95'][1]:+.2f}%], p = {math_hard['p_value']:.4f}")
    print(f"GSM8K Arithmetic:          {gsm8k['observed_delta']:+.2f}% [95% CI: {gsm8k['ci_95'][0]:+.2f}%, {gsm8k['ci_95'][1]:+.2f}%], p = {gsm8k['p_value']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
