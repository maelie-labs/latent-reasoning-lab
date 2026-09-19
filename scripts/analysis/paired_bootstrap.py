#!/usr/bin/env python3
"""
scripts/analysis/paired_bootstrap.py

Paired Hierarchical Bootstrapping for Continuous Latent Recurrence Lab.
Computes true 95% confidence intervals and paired differences using B=10,000
bootstrap iterations clustered at the problem level (clustering the 4 seeds per problem).

Execution Environment: CPU only (CUDA_VISIBLE_DEVICES="")
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def load_eval_jsonl(filepath: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Loads an evaluation JSONL file and indexes entries by (problem_id, seed).
    Handles variations in schema across arms:
      - 'id' vs 'problem_id'
      - 'is_correct' as bool or int (1/0)
      - 'run' nested dictionary vs flat fields
    """
    records = {}
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception as e:
                print(f"Warning: Failed to parse line {line_idx} in {filepath}: {e}", file=sys.stderr)
                continue

            prob_id = str(data.get("problem_id", data.get("id", ""))).strip()
            if not prob_id:
                continue

            seed = int(data.get("seed", 42))

            # Normalize correctness: strict boolean
            is_correct_raw = data.get("is_correct")
            if is_correct_raw is None and "run" in data and isinstance(data["run"], dict):
                is_correct_raw = data["run"].get("is_correct")
            
            is_correct = bool(is_correct_raw == True or is_correct_raw == 1)

            # Truncation
            is_truncated_raw = data.get("is_truncated")
            if is_truncated_raw is None and "run" in data and isinstance(data["run"], dict):
                is_truncated_raw = data["run"].get("is_truncated")
            is_truncated = bool(is_truncated_raw == True or is_truncated_raw == 1)

            # If truncated, strict protocol marks as incorrect
            if is_truncated:
                is_correct = False

            # Latency / duration
            dur = None
            for key in ["dur", "total_time_s", "duration"]:
                if key in data and data[key] is not None:
                    dur = float(data[key])
                    break
            if dur is None and "run" in data and isinstance(data["run"], dict):
                for key in ["total_time_s", "dur", "duration"]:
                    if key in data["run"] and data["run"][key] is not None:
                        dur = float(data["run"][key])
                        break
                if dur is None and "total_time_ms" in data["run"]:
                    dur = float(data["run"]["total_time_ms"]) / 1000.0

            # Answer / total tokens
            tokens = data.get("total_tokens")
            if tokens is None and "run" in data and isinstance(data["run"], dict):
                tokens = data["run"].get("ans_tokens", data["run"].get("total_tokens"))

            benchmark = data.get("benchmark", "")
            if not benchmark:
                if prob_id.startswith("gsm8k"):
                    benchmark = "GSM8K"
                elif prob_id.startswith("math500"):
                    benchmark = "MATH-500"

            records[(prob_id, seed)] = {
                "problem_id": prob_id,
                "seed": seed,
                "benchmark": benchmark,
                "is_correct": is_correct,
                "is_truncated": is_truncated,
                "dur": dur,
                "tokens": tokens,
            }

    return records


def run_paired_bootstrap(
    records_a: Dict[Tuple[str, int], Dict[str, Any]],
    records_b: Dict[Tuple[str, int], Dict[str, Any]],
    n_bootstrap: int = 10000,
    seed: int = 42,
    subset_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Performs problem-level paired hierarchical bootstrapping.
    """
    rng = np.random.default_rng(seed)

    # Find common problem_ids
    problems_a = {p for (p, s) in records_a.keys()}
    problems_b = {p for (p, s) in records_b.keys()}
    common_problems = sorted(list(problems_a & problems_b))

    if subset_ids is not None:
        subset_set = set(subset_ids)
        common_problems = [p for p in common_problems if p in subset_set]

    if not common_problems:
        raise ValueError("Zero overlapping problem IDs found between comparator files.")

    # Find common seeds across these problems
    pairs = []
    all_seeds = [42, 123, 456, 789]
    for p in common_problems:
        p_seeds = [s for s in all_seeds if (p, s) in records_a and (p, s) in records_b]
        if not p_seeds:
            p_seeds = [s for (prob, s) in records_a.keys() if prob == p and (p, s) in records_b]
        pairs.append((p, p_seeds))

    n_problems = len(pairs)
    total_evals = sum(len(seeds) for _, seeds in pairs)

    # Pre-build 2D matrices: shape (n_problems, max_seeds)
    max_s = max(len(s) for _, s in pairs)
    y_a = np.zeros((n_problems, max_s), dtype=np.float32)
    y_b = np.zeros((n_problems, max_s), dtype=np.float32)
    mask = np.zeros((n_problems, max_s), dtype=np.bool_)

    for i, (p, s_list) in enumerate(pairs):
        for j, s in enumerate(s_list):
            y_a[i, j] = 1.0 if records_a[(p, s)]["is_correct"] else 0.0
            y_b[i, j] = 1.0 if records_b[(p, s)]["is_correct"] else 0.0
            mask[i, j] = True

    # Point estimates
    acc_a_point = float(np.sum(y_a[mask]) / total_evals * 100.0)
    acc_b_point = float(np.sum(y_b[mask]) / total_evals * 100.0)
    delta_point = acc_a_point - acc_b_point

    # Sum per problem for vectorization
    p_sum_a = np.sum(y_a * mask, axis=1)  # shape: (n_problems,)
    p_sum_b = np.sum(y_b * mask, axis=1)
    p_weights = np.sum(mask, axis=1)

    # Sample problem indices with replacement: shape (n_bootstrap, n_problems)
    boot_indices = rng.choice(n_problems, size=(n_bootstrap, n_problems), replace=True)

    # Vectorized compute across bootstrap resamples
    sampled_a = np.take(p_sum_a, boot_indices)  # (B, P)
    sampled_b = np.take(p_sum_b, boot_indices)  # (B, P)
    sampled_w = np.take(p_weights, boot_indices) # (B, P)

    boot_acc_a = (np.sum(sampled_a, axis=1) / np.sum(sampled_w, axis=1)) * 100.0
    boot_acc_b = (np.sum(sampled_b, axis=1) / np.sum(sampled_w, axis=1)) * 100.0
    boot_delta = boot_acc_a - boot_acc_b

    # Empirical 95% Confidence Intervals
    ci_a = [float(np.percentile(boot_acc_a, 2.5)), float(np.percentile(boot_acc_a, 97.5))]
    ci_b = [float(np.percentile(boot_acc_b, 2.5)), float(np.percentile(boot_acc_b, 97.5))]
    ci_delta = [float(np.percentile(boot_delta, 2.5)), float(np.percentile(boot_delta, 97.5))]

    # Two-tailed p-value testing H0: Delta = 0
    p_le_0 = np.mean(boot_delta <= 0.0)
    p_ge_0 = np.mean(boot_delta >= 0.0)
    p_value = float(min(1.0, 2.0 * min(p_le_0, p_ge_0)))

    # Truncation rates
    trunc_a = float(np.mean([records_a[(p, s)]["is_truncated"] for p, seeds in pairs for s in seeds]) * 100.0)
    trunc_b = float(np.mean([records_b[(p, s)]["is_truncated"] for p, seeds in pairs for s in seeds]) * 100.0)

    return {
        "n_problems": n_problems,
        "total_evaluations": total_evals,
        "n_bootstrap": n_bootstrap,
        "acc_a": acc_a_point,
        "ci_a": ci_a,
        "acc_b": acc_b_point,
        "ci_b": ci_b,
        "delta": delta_point,
        "ci_delta": ci_delta,
        "p_value": p_value,
        "truncation_a": trunc_a,
        "truncation_b": trunc_b,
    }


def main():
    parser = argparse.ArgumentParser(description="Paired Hierarchical Bootstrapping for Model Arms")
    parser.add_argument("--file_a", type=str, required=True, help="Evaluation JSONL for candidate arm (e.g. Arm 3)")
    parser.add_argument("--file_b", type=str, required=True, help="Evaluation JSONL for comparator arm (e.g. Arm 1b, 2, 4)")
    parser.add_argument("--name_a", type=str, default="Candidate (Arm A)", help="Display label for arm A")
    parser.add_argument("--name_b", type=str, default="Comparator (Arm B)", help="Display label for arm B")
    parser.add_argument("--suite_file", type=str, default="data/benchmark_suite_250.json", help="Suite JSON file")
    parser.add_argument("--n_bootstrap", type=int, default=10000, help="Number of bootstrap resamples (default 10,000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping")
    parser.add_argument("--output_json", type=str, default="data/analysis_paired_bootstrap_results.json", help="Output path for bootstrap results")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick smoke test with B=500")
    args = parser.parse_args()

    n_boot = 500 if args.smoke_test else args.n_bootstrap

    print(f"=== Paired Hierarchical Bootstrapping Analysis ===")
    print(f"Arm A: {args.name_a} ({args.file_a})")
    print(f"Arm B: {args.name_b} ({args.file_b})")
    print(f"Resamples B={n_boot:,} | Seed={args.seed}")

    records_a = load_eval_jsonl(args.file_a)
    records_b = load_eval_jsonl(args.file_b)

    # Overall analysis
    overall_res = run_paired_bootstrap(records_a, records_b, n_bootstrap=n_boot, seed=args.seed)

    # Subsets from suite_file if present
    gsm8k_res = None
    math500_res = None
    math_hard_res = None

    if os.path.exists(args.suite_file):
        with open(args.suite_file, "r") as f:
            suite = json.load(f)
        gsm8k_ids = [p["id"] for p in suite if p.get("benchmark") == "GSM8K"]
        math_ids = [p["id"] for p in suite if p.get("benchmark") == "MATH-500"]
        math_hard_ids = [p["id"] for p in suite if p.get("benchmark") == "MATH-500" and p.get("level", 0) >= 3]

        if gsm8k_ids:
            gsm8k_res = run_paired_bootstrap(records_a, records_b, n_bootstrap=n_boot, seed=args.seed, subset_ids=gsm8k_ids)
        if math_ids:
            math500_res = run_paired_bootstrap(records_a, records_b, n_bootstrap=n_boot, seed=args.seed, subset_ids=math_ids)
        if math_hard_ids:
            math_hard_res = run_paired_bootstrap(records_a, records_b, n_bootstrap=n_boot, seed=args.seed, subset_ids=math_hard_ids)

    # Console display
    def print_section(title: str, res: Dict[str, Any]):
        print(f"\n--- {title} (Problems: {res['n_problems']}, N={res['total_evaluations']}) ---")
        print(f"  {args.name_a:24s}: {res['acc_a']:6.2f}% (95% CI: [{res['ci_a'][0]:5.2f}%, {res['ci_a'][1]:5.2f}%], Trunc: {res['truncation_a']:.2f}%)")
        print(f"  {args.name_b:24s}: {res['acc_b']:6.2f}% (95% CI: [{res['ci_b'][0]:5.2f}%, {res['ci_b'][1]:5.2f}%], Trunc: {res['truncation_b']:.2f}%)")
        print(f"  Delta (A - B)           : {res['delta']:+6.2f}% (95% CI: [{res['ci_delta'][0]:+5.2f}%, {res['ci_delta'][1]:+5.2f}%])")
        sig_label = "STATISTICALLY SIGNIFICANT (p < 0.05)" if res['p_value'] < 0.05 else "NOT SIGNIFICANT (p >= 0.05)"
        print(f"  Two-Tailed p-value      : {res['p_value']:.4f} [{sig_label}]")

    print_section("Overall Benchmark Suite", overall_res)
    if gsm8k_res:
        print_section("GSM8K Arithmetic Subset", gsm8k_res)
    if math500_res:
        print_section("MATH-500 Math Subset", math500_res)
    if math_hard_res:
        print_section("MATH-500 Hard Subset (Levels 3-5)", math_hard_res)

    # Save to JSON
    output_data = {
        "arm_a": {"name": args.name_a, "file": args.file_a},
        "arm_b": {"name": args.name_b, "file": args.file_b},
        "n_bootstrap": n_boot,
        "seed": args.seed,
        "overall": overall_res,
        "gsm8k": gsm8k_res,
        "math500": math500_res,
        "math500_hard_l35": math_hard_res,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved bootstrap results to {args.output_json}")

if __name__ == "__main__":
    main()
