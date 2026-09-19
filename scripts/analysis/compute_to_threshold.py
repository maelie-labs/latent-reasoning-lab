#!/usr/bin/env python3
"""
scripts/analysis/compute_to_threshold.py

Computes reasoning FLOPs, KV cache footprint, latency per correct query,
and Pareto trade-offs across experimental arms for Qwen/Qwen3-1.7B.

Execution Environment: CPU only (CUDA_VISIBLE_DEVICES="")
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Any

# Qwen3-1.7B architectural constants
QWEN3_17B_PARAMS = 1.7e9
QWEN3_17B_LAYERS = 28
QWEN3_17B_HKV = 8
QWEN3_17B_DHEAD = 128
# KV cache bytes per token = 2 (k+v) * layers * h_kv * d_head * 2 (bf16 bytes)
BYTES_PER_KV_TOKEN = 2 * QWEN3_17B_LAYERS * QWEN3_17B_HKV * QWEN3_17B_DHEAD * 2 # 114,688 bytes = 112.0 KiB

def analyze_arm_compute(
    arm_name: str,
    filepath: str,
    k_latents: int = 0,
    is_latent_arm: bool = False,
    n_params: float = QWEN3_17B_PARAMS,
) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    evals = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            # Correctness
            c = data.get("is_correct")
            if c is None and "run" in data and isinstance(data["run"], dict):
                c = data["run"].get("is_correct")
            is_correct = bool(c == True or c == 1)

            # Truncation
            t = data.get("is_truncated")
            if t is None and "run" in data and isinstance(data["run"], dict):
                t = data["run"].get("is_truncated")
            is_trunc = bool(t == True or t == 1)
            if is_trunc:
                is_correct = False

            # Latency in ms
            dur_ms = None
            if "total_time_ms" in data and data["total_time_ms"] is not None:
                dur_ms = float(data["total_time_ms"])
            elif "run" in data and isinstance(data["run"], dict) and "total_time_ms" in data["run"]:
                dur_ms = float(data["run"]["total_time_ms"])
            elif "dur" in data and data["dur"] is not None:
                dur_ms = float(data["dur"]) * 1000.0
            elif "run" in data and isinstance(data["run"], dict) and "total_time_s" in data["run"]:
                dur_ms = float(data["run"]["total_time_s"]) * 1000.0
            elif "run" in data and isinstance(data["run"], dict) and "dur" in data["run"]:
                dur_ms = float(data["run"]["dur"]) * 1000.0

            # Tokens
            ans_tok = 0
            think_tok = 0
            if "ans_tokens" in data and data["ans_tokens"] is not None:
                ans_tok = int(data["ans_tokens"])
            elif "run" in data and isinstance(data["run"], dict) and "ans_tokens" in data["run"]:
                ans_tok = int(data["run"]["ans_tokens"])

            if "total_tokens" in data and data["total_tokens"] is not None:
                tot = int(data["total_tokens"])
                if ans_tok > 0:
                    think_tok = max(0, tot - ans_tok)
                else:
                    # Estimate based on thinking_text vs content
                    think_text = data.get("thinking_text", "")
                    content_text = data.get("content", "")
                    t_len = len(think_text)
                    c_len = len(content_text)
                    if t_len + c_len > 0:
                        think_tok = int(tot * (t_len / (t_len + c_len)))
                        ans_tok = tot - think_tok
                    else:
                        ans_tok = tot

            if k_latents > 0:
                think_tok = k_latents

            evals.append({
                "is_correct": is_correct,
                "is_truncated": is_trunc,
                "dur_ms": dur_ms,
                "think_tokens": think_tok,
                "ans_tokens": ans_tok,
            })

    n = len(evals)
    acc = (sum(1 for e in evals if e["is_correct"]) / n) * 100.0 if n > 0 else 0.0
    trunc_rate = (sum(1 for e in evals if e["is_truncated"]) / n) * 100.0 if n > 0 else 0.0

    valid_durations = [e["dur_ms"] for e in evals if e["dur_ms"] is not None]
    mean_lat_ms = float(np.mean(valid_durations)) if valid_durations else 0.0
    median_lat_ms = float(np.median(valid_durations)) if valid_durations else 0.0

    mean_think_tok = float(np.mean([e["think_tokens"] for e in evals]))
    med_think_tok = float(np.median([e["think_tokens"] for e in evals]))
    mean_ans_tok = float(np.mean([e["ans_tokens"] for e in evals]))
    med_ans_tok = float(np.median([e["ans_tokens"] for e in evals]))

    # Reasoning FLOPs = 2 * N_params * think_steps
    mean_reasoning_flops = 2.0 * n_params * mean_think_tok
    giga_flops = mean_reasoning_flops / 1e9
    tera_flops = mean_reasoning_flops / 1e12

    # Peak KV Cache for reasoning phase (KiB)
    # Assumes average prompt length of ~150 tokens
    avg_prompt_tokens = 150.0
    total_reasoning_tokens = avg_prompt_tokens + mean_think_tok
    reasoning_kv_kib = (total_reasoning_tokens * BYTES_PER_KV_TOKEN) / 1024.0
    total_kv_kib = ((total_reasoning_tokens + mean_ans_tok) * BYTES_PER_KV_TOKEN) / 1024.0

    # Latency per correct query (cost efficiency): Mean Latency (ms) / (Accuracy / 100)
    # i.e., ms of compute spent to produce 1 correct answer
    acc_frac = acc / 100.0
    lat_per_correct_ms = (mean_lat_ms / acc_frac) if acc_frac > 0 else float("inf")

    return {
        "arm_name": arm_name,
        "n_evaluations": n,
        "accuracy": acc,
        "truncation_rate": trunc_rate,
        "mean_latency_ms": mean_lat_ms,
        "median_latency_ms": median_lat_ms,
        "mean_think_tokens": mean_think_tok,
        "median_think_tokens": med_think_tok,
        "mean_ans_tokens": mean_ans_tok,
        "median_ans_tokens": med_ans_tok,
        "reasoning_gflops": giga_flops,
        "reasoning_tflops": tera_flops,
        "reasoning_kv_kib": reasoning_kv_kib,
        "total_kv_kib": total_kv_kib,
        "latency_per_correct_ms": lat_per_correct_ms,
    }

def main():
    parser = argparse.ArgumentParser(description="Compute reasoning FLOPs, KV cache footprint, and cost efficiency")
    parser.add_argument("--arms", type=str, nargs="+", default=[
        "Arm4_Unconstrained=data/streaming_arm4_thinking_qwen_qwen3-1.7b.jsonl:0:0",
        "Arm2_K32=data/streaming_arm2_minimal_discrete_qwen_qwen3-1.7b_k32.jsonl:32:0",
        "Arm2_K6=data/streaming_arm2_minimal_discrete_qwen_qwen3-1.7b_k6.jsonl:6:0",
    ], help="Format: Label=path:k_tokens:is_latent (0 or 1)")
    parser.add_argument("--output_md", type=str, default="data/compute_threshold_table.md")
    parser.add_argument("--output_json", type=str, default="data/compute_threshold_results.json")
    args = parser.parse_args()

    results = []
    for item in args.arms:
        name_path, *rest = item.split(":")
        label, path = name_path.split("=", 1)
        k_val = int(rest[0]) if len(rest) > 0 else 0
        is_latent = bool(int(rest[1])) if len(rest) > 1 else False
        print(f"Analyzing compute for {label} (k={k_val}, is_latent={is_latent})...")
        res = analyze_arm_compute(label, path, k_latents=k_val, is_latent_arm=is_latent)
        results.append(res)

    # Build Markdown table
    md_lines = []
    md_lines.append("# Compute-to-Threshold & Resource Trade-Off Table\n")
    md_lines.append("| Experimental Arm | Pass@1 Acc (%) | Mean Think Tokens / Latents | Reasoning TFLOPs | Peak Reasoning KV (KiB) | Mean Latency (ms) | Latency / Correct Query (ms) |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in results:
        flops_str = f"{r['reasoning_tflops']:.3f} TFLOPs" if r['reasoning_tflops'] >= 0.1 else f"{r['reasoning_gflops']:.1f} GFLOPs"
        md_lines.append(
            f"| **{r['arm_name']}** | {r['accuracy']:.2f}% | {r['mean_think_tokens']:.1f} | "
            f"{flops_str} | {r['reasoning_kv_kib']:,.0f} KiB | {r['mean_latency_ms']:,.1f} ms | "
            f"{r['latency_per_correct_ms']:,.1f} ms |"
        )

    md_content = "\n".join(md_lines)
    print("\n" + md_content + "\n")

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {args.output_md} and {args.output_json}")

if __name__ == "__main__":
    main()
