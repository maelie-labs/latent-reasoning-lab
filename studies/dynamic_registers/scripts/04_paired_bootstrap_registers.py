#!/usr/bin/env python3
"""
studies/dynamic_registers/scripts/04_paired_bootstrap_registers.py
Paired Hierarchical Bootstrapping (B=10,000 iterations) for Staged Dynamic Registers:
1. Delta_1: Arm 1 (Dynamic Registers) - Control 1 (Arm 1b-R Headers Only)
2. Delta_2: Arm 1 (Dynamic Registers) - Arm 2 (Matched Filler Registers)
"""

import json
import argparse
import numpy as np

def load_streaming_results(path):
    records = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pid = r.get("id") or r.get("problem_id")
                key = (pid, r["seed"])
                records[key] = r
    return records

def paired_bootstrap(y_test, y_ctrl, b=10000, seed=42):
    np.random.seed(seed)
    n = len(y_test)
    deltas = []
    diffs = np.array(y_test, dtype=float) - np.array(y_ctrl, dtype=float)
    observed_delta = float(np.mean(diffs)) * 100

    indices = np.random.randint(0, n, size=(b, n))
    boot_diffs = np.mean(diffs[indices], axis=1) * 100
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    # Two-sided p-value
    p_val = float(np.mean(boot_diffs <= 0)) if observed_delta > 0 else float(np.mean(boot_diffs >= 0))
    p_val = min(1.0, 2.0 * p_val)

    return {
        "observed_delta": observed_delta,
        "ci_95": [ci_low, ci_high],
        "p_value": p_val,
        "n": n
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm1_file", type=str, required=True, help="Streaming JSONL for Arm 1 (Dynamic Registers)")
    parser.add_argument("--control_file", type=str, required=True, help="Streaming JSONL for Control (Arm 1b-R or Arm 2)")
    parser.add_argument("--comparator_name", type=str, default="Control")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    r_arm1 = load_streaming_results(args.arm1_file)
    r_ctrl = load_streaming_results(args.control_file)

    common_keys = sorted(list(set(r_arm1.keys()) & set(r_ctrl.keys())))
    print(f"Total paired evaluations: {len(common_keys)}")

    y1_all = [1.0 if r_arm1[k]["is_correct"] else 0.0 for k in common_keys]
    yc_all = [1.0 if r_ctrl[k]["is_correct"] else 0.0 for k in common_keys]
    overall_res = paired_bootstrap(y1_all, yc_all)

    # MATH Hard L3-5
    hard_keys = [k for k in common_keys if r_arm1[k].get("benchmark") == "MATH-500" and r_arm1[k].get("stratum") in ["Level 3", "Level 4", "Level 5"]]
    y1_hard = [1.0 if r_arm1[k]["is_correct"] else 0.0 for k in hard_keys]
    yc_hard = [1.0 if r_ctrl[k]["is_correct"] else 0.0 for k in hard_keys]
    hard_res = paired_bootstrap(y1_hard, yc_hard)

    print("\n" + "=" * 60)
    print(f"PAIRED BOOTSTRAP: Arm 1 vs. {args.comparator_name}")
    print(f"Overall Delta:   {overall_res['observed_delta']:+.2f}% [95% CI: {overall_res['ci_95'][0]:+.2f}%, {overall_res['ci_95'][1]:+.2f}%] (p={overall_res['p_value']:.4f})")
    print(f"MATH Hard Delta: {hard_res['observed_delta']:+.2f}% [95% CI: {hard_res['ci_95'][0]:+.2f}%, {hard_res['ci_95'][1]:+.2f}%] (p={hard_res['p_value']:.4f})")
    print("=" * 60)

    out_data = {
        "comparator": args.comparator_name,
        "total_pairs": len(common_keys),
        "overall": overall_res,
        "math_hard": hard_res
    }
    with open(args.output_file, "w") as f:
        json.dump(out_data, f, indent=2)

if __name__ == "__main__":
    main()
