#!/usr/bin/env python3
"""
scripts/106_multivariate_bootstrap_analysis.py
Full Comparative Analysis Matrix for Continuous Latent Recurrence Lab:
1. Arm 1b   : Direct Baseline SFT (71.20% overall)
2. Arm 1b-R : Structural Headers Only (3 headers, 0 latents)
3. Arm 2b-R : Structural Headers + 24 Pause Tokens
4. Arm 3-R  : Continuous Latent Ladder (3 headers + 24 latents, 71.30% overall)

Computes paired hierarchical bootstrapping (B=10,000, seed=42) across all 1,000 queries:
- Overall (N=1,000)
- GSM8K (N=600)
- MATH-500 (N=400)
- MATH Levels 1 to 5 (N=80 each)
- MATH Hard L3-5 (N=240)
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict

def run_paired_bootstrap(scores_a, scores_b, n_boot=10000, seed=42):
    np.random.seed(seed)
    n = len(scores_a)
    deltas = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        deltas.append(np.mean(scores_a[idx]) - np.mean(scores_b[idx]))
    deltas = np.array(deltas) * 100.0
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    obs_delta = float(np.mean(scores_a) - np.mean(scores_b)) * 100.0
    p_val = float(np.mean(deltas <= 0.0)) if obs_delta > 0 else float(np.mean(deltas >= 0.0))
    return {
        "mean_a": float(np.mean(scores_a)) * 100.0,
        "mean_b": float(np.mean(scores_b)) * 100.0,
        "observed_delta": obs_delta,
        "ci_95": [ci_low, ci_high],
        "p_value": p_val * 2.0
    }

def get_id(r):
    return r.get("id") or r.get("problem_id")

def load_stream(filepath):
    with open(filepath) as f:
        records = [json.loads(line) for line in f]
    return {(get_id(r), r["seed"]): r for r in records}

def analyze_slice(name, dict_a, dict_b, keys):
    scores_a = []
    scores_b = []
    for k in keys:
        if k in dict_a and k in dict_b:
            scores_a.append(int(dict_a[k]["is_correct"]))
            scores_b.append(int(dict_b[k]["is_correct"]))
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    res = run_paired_bootstrap(scores_a, scores_b)
    res["slice"] = name
    res["n"] = len(scores_a)
    return res

def compare_arms(name_a, stream_a, name_b, stream_b, all_keys, key_metadata):
    slices = {
        "Overall": all_keys,
        "GSM8K": [k for k in all_keys if key_metadata[k]["benchmark"] == "GSM8K"],
        "MATH-500": [k for k in all_keys if key_metadata[k]["benchmark"] in ["MATH-500", "MATH500"]],
        "MATH L1": [k for k in all_keys if key_metadata[k]["stratum"] == "Level 1"],
        "MATH L2": [k for k in all_keys if key_metadata[k]["stratum"] == "Level 2"],
        "MATH L3": [k for k in all_keys if key_metadata[k]["stratum"] == "Level 3"],
        "MATH L4": [k for k in all_keys if key_metadata[k]["stratum"] == "Level 4"],
        "MATH L5": [k for k in all_keys if key_metadata[k]["stratum"] == "Level 5"],
        "MATH L3-5 (Hard)": [k for k in all_keys if key_metadata[k]["stratum"] in ["Level 3", "Level 4", "Level 5"]],
    }

    results = {}
    print(f"\n{'='*80}\nPAIRED BOOTSTRAP: {name_a} vs. {name_b}\n{'='*80}")
    print(f"{'Slice':<18} | {'N':<5} | {name_a:<10} | {name_b:<10} | {'Delta':<10} | {'95% CI':<18} | {'p-val':<8}")
    print("-" * 85)

    for s_name, s_keys in slices.items():
        if not s_keys:
            continue
        res = analyze_slice(s_name, stream_a, stream_b, s_keys)
        results[s_name] = res
        ci_str = f"[{res['ci_95'][0]:+5.2f}%, {res['ci_95'][1]:+5.2f}%]"
        p_str = f"{res['p_value']:.4f}" if res['p_value'] >= 0.0001 else "<0.0001"
        print(f"{s_name:<18} | {res['n']:<5} | {res['mean_a']:6.2f}%    | {res['mean_b']:6.2f}%    | {res['observed_delta']:+6.2f}%    | {ci_str:<18} | {p_str:<8}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm1b_file", default="data/streaming_arm1b_qwen_qwen3-1.7b.jsonl")
    parser.add_argument("--arm1b_r_file", default="data/streaming_arm1b_register_headers_only.jsonl")
    parser.add_argument("--arm2b_r_file", default="data/streaming_arm2b_register_headers_pause.jsonl")
    parser.add_argument("--arm3_r_file", default="data/streaming_arm3_register_ladder_production_best.jsonl")
    parser.add_argument("--output_file", default="data/multivariate_scaffolding_bootstrap_matrix.json")
    args = parser.parse_args()

    files = {
        "Arm 1b (Baseline SFT)": args.arm1b_file,
        "Arm 1b-R (Headers Only)": args.arm1b_r_file,
        "Arm 2b-R (Pause Tokens)": args.arm2b_r_file,
        "Arm 3-R (Latent Ladder)": args.arm3_r_file
    }

    streams = {}
    for name, fpath in files.items():
        if os.path.exists(fpath):
            streams[name] = load_stream(fpath)
            print(f"Loaded {name}: {len(streams[name])} records from {fpath}")
        else:
            print(f"Warning: {fpath} not found for {name}")

    # Build reference metadata from first available stream
    first_stream = list(streams.values())[0]
    all_keys = list(first_stream.keys())
    key_metadata = {k: {"benchmark": v["benchmark"], "stratum": v.get("stratum", "")} for k, v in first_stream.items()}

    all_comparisons = {}

    # Key Scientific Comparisons:
    pairs = [
        ("Arm 1b-R (Headers Only)", "Arm 1b (Baseline SFT)", "scaffolding_effect"),
        ("Arm 2b-R (Pause Tokens)", "Arm 1b-R (Headers Only)", "pause_token_effect"),
        ("Arm 3-R (Latent Ladder)", "Arm 1b-R (Headers Only)", "latent_vs_headers"),
        ("Arm 3-R (Latent Ladder)", "Arm 2b-R (Pause Tokens)", "latent_vs_pause_tokens"),
        ("Arm 3-R (Latent Ladder)", "Arm 1b (Baseline SFT)", "arm3_vs_baseline_sft"),
    ]

    for name_a, name_b, comp_id in pairs:
        if name_a in streams and name_b in streams:
            all_comparisons[comp_id] = compare_arms(name_a, streams[name_a], name_b, streams[name_b], all_keys, key_metadata)

    with open(args.output_file, "w") as f:
        json.dump(all_comparisons, f, indent=2)

    print(f"\nMatrix analysis saved to {args.output_file}")

if __name__ == "__main__":
    main()
