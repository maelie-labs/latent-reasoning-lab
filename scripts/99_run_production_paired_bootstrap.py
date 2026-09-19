#!/usr/bin/env python3
"""
scripts/99_run_production_paired_bootstrap.py
Official Paired Hierarchical Bootstrapping Analysis for Register-Bundle Ladder (Arm 3-R)
vs. Direct SFT Baseline (Arm 1b) on Qwen/Qwen3-1.7B.

Runs B=10,000 paired bootstrap iterations across all N=1,000 benchmark queries
(250 problems x 4 seeds: 42, 123, 456, 789).
Computes:
- Overall Pass@1 and Delta_1 = Arm 3-R - Arm 1b
- True 95% Confidence Intervals [ci_low, ci_high]
- Empirical p-value
- Sub-benchmark breakdowns: GSM8K (600 queries), MATH-500 (400 queries), MATH Levels 1-5
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict

def run_bootstrap_analysis(
    arm3_file="data/streaming_arm3_register_ladder_production_best.jsonl",
    arm1b_file="data/streaming_arm1b_qwen_qwen3-1.7b.jsonl",
    output_file="data/paired_bootstrap_arm3_register_ladder_vs_arm1b.json",
    n_boot=10000,
    seed=42
):
    print("=" * 80)
    print("PAIRED HIERARCHICAL BOOTSTRAPPING: ARM 3-R VS. ARM 1b")
    print(f"Arm 3-R Stream : {arm3_file}")
    print(f"Arm 1b Stream  : {arm1b_file}")
    print(f"Bootstrap Iter : B={n_boot:,} | Seed: {seed}")
    print("=" * 80)

    with open(arm3_file) as f:
        arm3_records = [json.loads(line) for line in f]
    with open(arm1b_file) as f:
        arm1b_records = [json.loads(line) for line in f]

    print(f"Loaded {len(arm3_records)} Arm 3-R records, {len(arm1b_records)} Arm 1b records.")

    # Index by (id, seed)
    def get_id(r):
        return r.get("id") or r.get("problem_id")

    dict_1b = {(get_id(r), r["seed"]): r for r in arm1b_records}

    paired_data = []
    missing_in_1b = 0
    for r3 in arm3_records:
        r3_id = get_id(r3)
        key = (r3_id, r3["seed"])
        if key in dict_1b:
            r1b = dict_1b[key]
            paired_data.append({
                "id": r3_id,
                "seed": r3["seed"],
                "benchmark": r3["benchmark"].upper(),
                "stratum": r3.get("stratum", "unknown"),
                "arm3_correct": bool(r3["is_correct"]),
                "arm1b_correct": bool(r1b["is_correct"]),
                "arm3_tokens": r3.get("tokens", 0),
                "arm1b_tokens": r1b.get("tokens", 0),
                "arm3_truncated": bool(r3.get("is_truncated", False)),
                "arm1b_truncated": bool(r1b.get("is_truncated", False))
            })
        else:
            missing_in_1b += 1

    n_pairs = len(paired_data)
    print(f"Successfully paired {n_pairs} queries. (Missing in 1b: {missing_in_1b})")

    if n_pairs == 0:
        print("ERROR: No paired queries found!")
        return None

    # Compute Core Metrics
    def eval_subset(subset, name):
        n = len(subset)
        if n == 0:
            return None
        c3 = sum(1 for p in subset if p["arm3_correct"])
        c1b = sum(1 for p in subset if p["arm1b_correct"])
        acc3 = (c3 / n) * 100.0
        acc1b = (c1b / n) * 100.0
        delta = acc3 - acc1b

        # Paired Bootstrap
        np.random.seed(seed)
        scores3 = np.array([float(p["arm3_correct"]) for p in subset])
        scores1b = np.array([float(p["arm1b_correct"]) for p in subset])
        
        boot_deltas = []
        for _ in range(n_boot):
            idx = np.random.randint(0, n, size=n)
            diff = np.mean(scores3[idx]) - np.mean(scores1b[idx])
            boot_deltas.append(diff * 100.0)
        boot_deltas = np.array(boot_deltas)
        ci_low = float(np.percentile(boot_deltas, 2.5))
        ci_high = float(np.percentile(boot_deltas, 97.5))
        p_val = float(np.mean(boot_deltas <= 0.0)) if delta > 0 else float(np.mean(boot_deltas >= 0.0))

        return {
            "name": name,
            "n": n,
            "arm3_correct": c3,
            "arm1b_correct": c1b,
            "arm3_acc": round(acc3, 2),
            "arm1b_acc": round(acc1b, 2),
            "delta": round(delta, 2),
            "ci_95": [round(ci_low, 2), round(ci_high, 2)],
            "p_value": round(p_val, 4)
        }

    overall_res = eval_subset(paired_data, "Overall (N=1,000)")
    gsm_subset = [p for p in paired_data if "GSM" in p["benchmark"]]
    gsm_res = eval_subset(gsm_subset, "GSM8K (N=600)")
    math_subset = [p for p in paired_data if "MATH" in p["benchmark"]]
    math_res = eval_subset(math_subset, "MATH-500 (N=400)")

    level_results = {}
    for lvl in ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]:
        lvl_sub = [p for p in math_subset if p["stratum"] == lvl]
        level_results[lvl] = eval_subset(lvl_sub, f"MATH {lvl}")

    # Seed Breakdown
    seed_results = {}
    for s in [42, 123, 456, 789]:
        s_sub = [p for p in paired_data if p["seed"] == s]
        seed_results[s] = eval_subset(s_sub, f"Seed {s}")

    # Token Distribution for Arm 3-R
    tokens3 = [p["arm3_tokens"] for p in paired_data if p["arm3_tokens"] > 0]
    trunc3 = sum(1 for p in paired_data if p["arm3_truncated"])

    summary = {
        "model_id": "Qwen/Qwen3-1.7B",
        "arm": "Arm 3-R (Register-Bundle Ladder)",
        "comparator": "Arm 1b (Trained Direct No-CoT Control)",
        "total_pairs": n_pairs,
        "overall": overall_res,
        "gsm8k": gsm_res,
        "math500": math_res,
        "math_levels": level_results,
        "seeds": seed_results,
        "arm3_telemetry": {
            "truncation_count": trunc3,
            "truncation_rate_pct": round((trunc3 / n_pairs) * 100.0, 2) if n_pairs > 0 else 0.0,
            "tokens_median": int(np.median(tokens3)) if tokens3 else 0,
            "tokens_mean": round(float(np.mean(tokens3)), 1) if tokens3 else 0.0,
            "tokens_p90": round(float(np.percentile(tokens3, 90)), 1) if tokens3 else 0.0,
            "tokens_max": int(np.max(tokens3)) if tokens3 else 0
        }
    }

    # Print Formatted Report
    print("\n" + "=" * 80)
    print("PAIRED BOOTSTRAP TELEMETRY REPORT")
    print("=" * 80)
    print(f"{'Metric / Stratum':<28} | {'Arm 3-R':<9} | {'Arm 1b':<9} | {'Delta':<8} | {'95% CI':<16} | {'p-value':<8}")
    print("-" * 80)

    for item in [overall_res, gsm_res, math_res] + list(level_results.values()):
        if item:
            ci_str = f"[{item['ci_95'][0]:+5.2f}, {item['ci_95'][1]:+5.2f}]"
            print(f"{item['name']:<28} | {item['arm3_acc']:5.2f}%   | {item['arm1b_acc']:5.2f}%   | {item['delta']:+5.2f}%  | {ci_str:<16} | {item['p_value']:<8}")

    print("\n--- Per-Seed Breakdown ---")
    for s, item in seed_results.items():
        if item:
            ci_str = f"[{item['ci_95'][0]:+5.2f}, {item['ci_95'][1]:+5.2f}]"
            print(f"Seed {s:<23} | {item['arm3_acc']:5.2f}%   | {item['arm1b_acc']:5.2f}%   | {item['delta']:+5.2f}%  | {ci_str:<16} | {item['p_value']:<8}")

    print("\n--- Telemetry Sanity ---")
    t = summary["arm3_telemetry"]
    print(f"  * Truncation Rate : {t['truncation_rate_pct']}% ({t['truncation_count']}/{n_pairs})")
    print(f"  * Token Length    : Median {t['tokens_median']} | Mean {t['tokens_mean']} | P90 {t['tokens_p90']} | Max {t['tokens_max']}")

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull paired bootstrap analysis saved to {output_file}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm3_file", default="data/streaming_arm3_register_ladder_production_best.jsonl")
    parser.add_argument("--arm1b_file", default="data/streaming_arm1b_qwen_qwen3-1.7b.jsonl")
    parser.add_argument("--output_file", default="data/paired_bootstrap_arm3_register_ladder_vs_arm1b.json")
    parser.add_argument("--n_boot", type=int, default=10000)
    args = parser.parse_args()
    run_bootstrap_analysis(args.arm3_file, args.arm1b_file, args.output_file, args.n_boot)
