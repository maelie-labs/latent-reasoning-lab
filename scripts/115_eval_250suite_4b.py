#!/usr/bin/env python3
"""
scripts/115_eval_250suite_4b.py
Universal High-Throughput 250-Suite Evaluator for 4B Architectures (Base Arms).

Covers:
- Arm 1 (Control 2: Untrained Base Direct Floor, enable_thinking=False)
- Arm 4 (Control 1: Untrained Base Verbose CoT Ceiling, enable_thinking=True)

Model Configurations (Strict Gate 0 frozen specs):
- Qwen/Qwen3-4B:
    Arm 1 Floor:   temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5, max_tokens=8192, prompt with <think>\n\n</think>\n\n
    Arm 4 Ceiling: temp=0.6, top_p=0.95, top_k=20, presence_penalty=0.0, max_tokens=32768, prompt with <think>\n
    Concurrency:   C* = 16
- Qwen/Qwen3.5-4B:
    Arm 1 Floor:   temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5, max_tokens=8192, prompt with <think>\n\n</think>\n\n
    Arm 4 Ceiling: temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=81920, prompt with <think>\n
    Concurrency:   C* = 8

Evaluation Protocol:
- Multi-seed determinism: SEEDS = [42, 123, 456, 789] (N = 1,000 queries per arm).
- Grader: Canonical math_verify 0.9.0 CAS parser, 0 heuristic fallbacks.
- Strict Truncation Rule: Hit max_tokens without emission => is_correct = False.
- Auto-resume and streaming persistence to .jsonl.
- Mandatory Section 11 Telemetry & Post-Eval Sanity Audit.
"""

import os
import re
import json
import time
import asyncio
import argparse
import numpy as np
import httpx
from math_verify import parse, verify

SEEDS = [42, 123, 456, 789]
THINK_END_TOKEN = "</think>"

def extract_math_boxed_expression(text):
    if not text or "\\boxed{" not in text:
        return ""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    end = start
    while end < len(text) and depth > 0:
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1
    if depth == 0:
        res = text[start:end-1].strip()
        if "=" in res:
            res = res.split("=")[-1].strip()
        return res.rstrip(".")
    return ""

def verify_solution(pred_text, gt_target):
    if not pred_text or not gt_target:
        return False
    try:
        gold_p = parse(gt_target)
        pred_p = parse(pred_text)
        if gold_p and pred_p and verify(gold_p, pred_p):
            return True
    except Exception:
        pass
    return False

def get_arm_config(model_id, arm):
    is_qwen35 = "3.5" in model_id.lower() or "3_5" in model_id.lower()
    
    if arm == "arm1":
        # Untrained Base Direct Floor
        temp = 0.7
        top_p = 0.80
        top_k = 20
        pres_pen = 1.5
        max_tokens = 8192
        prompt_suffix = "<think>\n\n</think>\n\n"
        enable_thinking = False
    elif arm == "arm4":
        # Untrained Base Verbose CoT Ceiling
        if is_qwen35:
            temp = 1.0
            top_p = 0.95
            top_k = 20
            pres_pen = 1.5
            max_tokens = 81920
        else:
            temp = 0.6
            top_p = 0.95
            top_k = 20
            pres_pen = 0.0
            max_tokens = 32768
        prompt_suffix = "<think>\n"
        enable_thinking = True
    else:
        raise ValueError(f"Unknown arm: {arm}. Must be 'arm1' or 'arm4'.")
        
    concurrency = 8 if is_qwen35 else 16
    
    return {
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": pres_pen,
        "max_new_tokens": max_tokens,
        "prompt_suffix": prompt_suffix,
        "enable_thinking": enable_thinking,
        "concurrency": concurrency
    }

async def query_sglang(client, server_url, prompt_q, seed, config):
    raw_q = prompt_q.rstrip()
    if "\\boxed" not in raw_q:
        q = raw_q + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    else:
        q = raw_q
    prompt_text = (
        f"<|im_start|>user\n{q}<|im_end|>\n"
        f"<|im_start|>assistant\n{config['prompt_suffix']}"
    )
    
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
            "presence_penalty": config["presence_penalty"],
            "max_new_tokens": config["max_new_tokens"],
            "sampling_seed": seed
        }
    }
    
    t0 = time.time()
    try:
        resp = await client.post(f"{server_url}/generate", json=payload, timeout=1800.0)
        dur = time.time() - t0
        if resp.status_code != 200:
            return {
                "completion_tokens": 0,
                "dur": round(dur, 2),
                "is_truncated": True,
                "thinking_text": "",
                "answer_text": "",
                "raw_text": "",
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}"
            }
            
        data = resp.json()
        raw_text = data.get("text", "")
        meta = data.get("meta_info", {})
        finish_reason = meta.get("finish_reason", {})
        finish_type = finish_reason.get("type", "") if isinstance(finish_reason, dict) else str(finish_reason)
        is_truncated = (finish_type == "length")
        completion_tokens = meta.get("completion_tokens", 0)
        
        if config["enable_thinking"]:
            if THINK_END_TOKEN in raw_text:
                parts = raw_text.split(THINK_END_TOKEN, 1)
                thinking_text = parts[0].strip()
                answer_text = parts[1].strip()
            else:
                thinking_text = raw_text.strip()
                answer_text = ""
                is_truncated = True
        else:
            thinking_text = ""
            answer_text = raw_text.strip()
            
        return {
            "completion_tokens": completion_tokens,
            "dur": round(dur, 2),
            "is_truncated": is_truncated,
            "thinking_text": thinking_text,
            "answer_text": answer_text,
            "raw_text": raw_text,
            "error": None
        }
    except Exception as e:
        return {
            "completion_tokens": 0,
            "dur": round(time.time() - t0, 2),
            "is_truncated": True,
            "thinking_text": "",
            "answer_text": "",
            "raw_text": "",
            "error": str(e)
        }

async def run_evaluation(server_url, model_id, arm, suite_file, output_file, seeds, concurrency_override):
    config = get_arm_config(model_id, arm)
    concurrency = concurrency_override or config["concurrency"]
    
    with open(suite_file) as f:
        problems = json.load(f)
        
    total_queries = len(problems) * len(seeds)
    print(f"\n========================================================")
    print(f"=== 250-Suite Evaluation: {model_id} | Arm: {arm.upper()} ===")
    print(f"Problems: {len(problems)} | Seeds: {seeds} | Total Queries: {total_queries}")
    print(f"Sampling: temp={config['temperature']}, top_p={config['top_p']}, pp={config['presence_penalty']}, max_tokens={config['max_new_tokens']}")
    print(f"Concurrency: {concurrency} (calibrated C*) | Server: {server_url}")
    print(f"Streaming output to: {output_file}")
    print(f"========================================================\n")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    existing_runs = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        existing_runs[(rec["problem_id"], rec["seed"])] = rec
                    except Exception:
                        pass
        print(f"Resuming: found {len(existing_runs)} existing runs in {output_file}")
        
    tasks_to_run = []
    for prob in problems:
        for seed in seeds:
            key = (prob["id"], seed)
            if key not in existing_runs:
                tasks_to_run.append((prob, seed))
                
    print(f"Pending tasks to execute: {len(tasks_to_run)} / {total_queries}")
    
    results = list(existing_runs.values())
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done_count = len(results)
    correct_count = sum(r.get("is_correct", False) for r in results)
    trunc_count = sum(r.get("is_truncated", False) for r in results)
    t_start = time.time()
    
    limits = httpx.Limits(max_keepalive_connections=concurrency * 2, max_connections=concurrency * 4)
    async with httpx.AsyncClient(limits=limits, timeout=1800.0) as client:
        async def worker(prob, seed):
            nonlocal done_count, correct_count, trunc_count
            q = prob["question"]
            sol = prob["solution"]
            if "####" in sol:
                gt_boxed = sol.split("####")[-1].strip()
            else:
                gt_boxed = extract_math_boxed_expression(sol)
            gt_target = f"\\boxed{{{gt_boxed}}}" if gt_boxed else f"\\boxed{{{sol}}}"
            
            async with semaphore:
                res = await query_sglang(client, server_url, q, seed, config)
                
                is_correct = False
                if not res["is_truncated"]:
                    is_correct = verify_solution(res["answer_text"], gt_target)
                    
                rec = {
                    "problem_id": prob["id"],
                    "benchmark": prob["benchmark"],
                    "level": prob.get("level", 1),
                    "seed": seed,
                    "model_id": model_id,
                    "arm": arm,
                    "is_correct": is_correct,
                    "is_truncated": res["is_truncated"],
                    "completion_tokens": res["completion_tokens"],
                    "dur": res["dur"],
                    "error": res["error"],
                    "gt_target": gt_target,
                    "answer_text": res["answer_text"][:2000],
                    "thinking_snippet": res["thinking_text"][:500] if res["thinking_text"] else ""
                }
                
                async with lock:
                    with open(output_file, "a") as f_out:
                        f_out.write(json.dumps(rec) + "\n")
                    results.append(rec)
                    done_count += 1
                    if is_correct:
                        correct_count += 1
                    if res["is_truncated"]:
                        trunc_count += 1
                        
                    if done_count % 25 == 0 or done_count == total_queries:
                        acc = correct_count / done_count * 100.0
                        trunc = trunc_count / done_count * 100.0
                        elapsed = time.time() - t_start
                        qps = (done_count - len(existing_runs)) / (elapsed or 1e-5)
                        print(f"[{done_count:4d}/{total_queries}] Acc: {acc:6.2f}% | Trunc: {trunc:5.2f}% | QPS: {qps:.2f} ({elapsed:.1f}s)")
                        
        workers = [worker(prob, seed) for prob, seed in tasks_to_run]
        await asyncio.gather(*workers)
        
    final_acc = correct_count / total_queries * 100.0
    final_trunc = trunc_count / total_queries * 100.0
    
    # Section 11: Mandatory Post-Eval Telemetry & Sanity Audit
    print("\n" + "="*60)
    print(f"=== Section 11 Telemetry & Sanity Audit: {model_id} ({arm.upper()}) ===")
    token_lengths = [r["completion_tokens"] for r in results if r["completion_tokens"] > 0]
    if token_lengths:
        print(f"Token Lengths: min={int(np.min(token_lengths))}, median={int(np.median(token_lengths))}, p90={int(np.percentile(token_lengths, 90))}, max={int(np.max(token_lengths))}")
    print(f"Pass@1 Accuracy: {final_acc:.2f}% ({correct_count}/{total_queries})")
    print(f"Truncation Rate: {final_trunc:.2f}% ({trunc_count}/{total_queries})")
    
    if arm == "arm1" and final_trunc > 1.0:
        print(f"WARNING: Direct non-thinking arm truncation rate ({final_trunc:.2f}%) exceeds 1.0% threshold!")
        
    print("\n--- Spot-Checking 5 Output Samples ---")
    for i, r in enumerate(results[:5]):
        print(f"Sample {i+1} [{r['problem_id']}, seed {r['seed']}]: Correct={r['is_correct']} | Trunc={r['is_truncated']} | Ans: {r['answer_text'][:80]}...")
    print("="*60 + "\n")
    
    # Compute Stratified Breakdown
    by_bench = {}
    for r in results:
        b = r["benchmark"]
        if b not in by_bench:
            by_bench[b] = {"total": 0, "correct": 0}
        by_bench[b]["total"] += 1
        if r["is_correct"]:
            by_bench[b]["correct"] += 1
            
    print("Benchmark Breakdown:")
    for b, d in by_bench.items():
        print(f"  - {b}: {d['correct']}/{d['total']} ({d['correct']/d['total']*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="250-Suite Evaluator for 4B Baseline Arms")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--model_id", type=str, required=True, help="Qwen/Qwen3-4B or Qwen/Qwen3.5-4B")
    parser.add_argument("--arm", type=str, choices=["arm1", "arm4"], required=True, help="arm1 (floor) or arm4 (ceiling)")
    parser.add_argument("--suite_file", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()
    
    if args.output_file is None:
        safe_name = args.model_id.replace("/", "_").replace(".", "_").lower()
        args.output_file = f"data/eval_250suite_{safe_name}_{args.arm}.jsonl"
        
    asyncio.run(run_evaluation(
        server_url=args.server_url,
        model_id=args.model_id,
        arm=args.arm,
        suite_file=args.suite_file,
        output_file=args.output_file,
        seeds=args.seeds,
        concurrency_override=args.concurrency
    ))

if __name__ == "__main__":
    main()
