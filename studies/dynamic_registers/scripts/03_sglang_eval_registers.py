#!/usr/bin/env python3
"""
studies/dynamic_registers/scripts/03_sglang_eval_registers.py
High-throughput SGLang evaluation for Staged Dynamic Registers study on Qwen/Qwen3-1.7B.
Enforces:
- Gate 0 Frozen Spec: temp=0.7, top_p=0.80, top_k=20, pp=1.5, max_new_tokens=8192
- Stop tokens: ["<|im_end|>", "<|endoftext|>"]
- Strict truncation protocol: sequences hitting 8,192 without closing tags scored False
- Canonical math_verify 0.9.0 with symbolic extraction and verification
- Multi-seed determinism: SEEDS = [42, 123, 456, 789] (N=1,000 queries)
"""

import os
import sys
import json
import time
import asyncio
import argparse
import httpx
from tqdm.asyncio import tqdm
from math_verify import parse, verify

SEEDS = [42, 123, 456, 789]

def extract_math_boxed_expression(text: str) -> str:
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

def check_correctness(pred_text: str, solution: str) -> bool:
    if not pred_text or not solution:
        return False
    try:
        # GSM8K canonical answer extraction
        if "####" in solution:
            gold_ans = solution.split("####")[-1].strip()
            gold_target = f"\\boxed{{{gold_ans}}}"
        elif "\\boxed{" not in solution:
            gold_target = f"\\boxed{{{solution.strip()}}}"
        else:
            gold_target = solution

        gold_parsed = parse(gold_target, parsing_timeout=None)
        pred_parsed = parse(pred_text, parsing_timeout=None)
        if gold_parsed and pred_parsed:
            return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        pass
    return False

def format_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"

async def eval_single(client, server_url, sem, problem, seed, max_tokens=8192):
    prompt_text = format_prompt(problem["question"])
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.80,
            "top_k": 20,
            "presence_penalty": 1.5,
            "sampling_seed": seed,
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }
    }
    
    async with sem:
        t0 = time.time()
        try:
            resp = await client.post(f"{server_url}/generate", json=payload, timeout=300.0)
            latency = time.time() - t0
            if resp.status_code != 200:
                return {
                    "id": problem["id"],
                    "benchmark": problem["benchmark"],
                    "stratum": problem.get("stratum", ""),
                    "level": problem.get("level", None),
                    "seed": seed,
                    "is_correct": False,
                    "is_truncated": True,
                    "tokens": 0,
                    "content": f"ERROR: status {resp.status_code}",
                    "pred_boxed": "",
                    "latency_s": latency
                }
            data = resp.json()
        except Exception as e:
            latency = time.time() - t0
            return {
                "id": problem["id"],
                "benchmark": problem["benchmark"],
                "stratum": problem.get("stratum", ""),
                "level": problem.get("level", None),
                "seed": seed,
                "is_correct": False,
                "is_truncated": True,
                "tokens": 0,
                "content": f"ERROR: {str(e)}",
                "pred_boxed": "",
                "latency_s": latency
            }

    content = data.get("text", "").strip()
    meta = data.get("meta_info", {})
    finish_reason = meta.get("finish_reason", {})
    if isinstance(finish_reason, dict):
        finish_type = finish_reason.get("type", "stop")
    else:
        finish_type = str(finish_reason)
        
    tok_len = meta.get("completion_tokens", len(content.split()))
    is_truncated = (finish_type == "length") or (tok_len >= max_tokens)
    
    if is_truncated:
        is_correct = False
    else:
        is_correct = check_correctness(content, problem["solution"])

    pred_boxed = extract_math_boxed_expression(content)
    return {
        "id": problem["id"],
        "benchmark": problem["benchmark"],
        "stratum": problem.get("stratum", ""),
        "level": problem.get("level", None),
        "seed": seed,
        "is_correct": is_correct,
        "is_truncated": is_truncated,
        "tokens": tok_len,
        "content": content,
        "pred_boxed": pred_boxed,
        "latency_s": latency
    }

async def run_evaluation(args):
    print("=" * 80)
    print("HIGH-THROUGHPUT SGLANG EVALUATION: 250 SUITE (N=1,000)")
    print(f"Tag: {args.tag} | Server: {args.server_url} | Concurrency: {args.concurrency}")
    print("=" * 80)

    with open(args.benchmark_file) as f:
        bench_problems = json.load(f)
    print(f"Loaded {len(bench_problems)} benchmark problems across {len(SEEDS)} seeds.")

    os.makedirs(os.path.dirname(os.path.abspath(args.streaming_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_keepalive_connections=args.concurrency, max_connections=args.concurrency * 2)
    timeout = httpx.Timeout(300.0, connect=30.0)

    all_tasks = []
    streaming_f = open(args.streaming_file, "w")
    start_time = time.time()

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Check server health
        try:
            health = await client.get(f"{args.server_url}/health")
            if health.status_code != 200:
                print(f"WARNING: Server health check returned {health.status_code}")
        except Exception as e:
            print(f"ERROR: Cannot connect to SGLang server at {args.server_url}: {e}")
            sys.exit(1)

        tasks = []
        for seed in SEEDS:
            for p in bench_problems:
                tasks.append(eval_single(client, args.server_url, sem, p, seed, args.max_tokens))

        all_results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Eval {args.tag}"):
            res = await f
            all_results.append(res)
            streaming_f.write(json.dumps(res) + "\n")
            streaming_f.flush()

    streaming_f.close()
    elapsed = time.time() - start_time

    total_evals = len(all_results)
    total_corr = sum(1 for r in all_results if r["is_correct"])
    mean_acc = (total_corr / total_evals) * 100

    gsm_items = [r for r in all_results if r["benchmark"] == "GSM8K"]
    math_items = [r for r in all_results if r["benchmark"] == "MATH-500"]
    math_hard = [r for r in math_items if r.get("level") in [3, 4, 5] or r.get("stratum") in ["Level 3", "Level 4", "Level 5"]]

    gsm_pass = (sum(1 for r in gsm_items if r["is_correct"]) / len(gsm_items)) * 100 if gsm_items else 0.0
    math_pass = (sum(1 for r in math_items if r["is_correct"]) / len(math_items)) * 100 if math_items else 0.0
    math_hard_pass = (sum(1 for r in math_hard if r["is_correct"]) / len(math_hard)) * 100 if math_hard else 0.0

    trunc_count = sum(1 for r in all_results if r["is_truncated"])
    trunc_rate = (trunc_count / total_evals) * 100

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE: {args.tag}")
    print(f"Time Elapsed:     {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"Overall Pass@1:   {mean_acc:.2f}% ({total_corr}/{total_evals})")
    print(f"GSM8K Pass@1:     {gsm_pass:.2f}% ({sum(1 for r in gsm_items if r['is_correct'])}/{len(gsm_items)})")
    print(f"MATH-500 Pass@1:  {math_pass:.2f}% ({sum(1 for r in math_items if r['is_correct'])}/{len(math_items)})")
    print(f"MATH Hard (L3-5): {math_hard_pass:.2f}% ({sum(1 for r in math_hard if r['is_correct'])}/{len(math_hard)})")
    print(f"Truncation Rate:  {trunc_rate:.2f}% ({trunc_count}/{total_evals})")
    print("=" * 80)

    summary = {
        "tag": args.tag,
        "elapsed_seconds": elapsed,
        "total_queries": total_evals,
        "overall_pass1": mean_acc,
        "gsm8k_pass1": gsm_pass,
        "math500_pass1": math_pass,
        "math_hard_pass1": math_hard_pass,
        "truncation_rate": trunc_rate,
        "concurrency": args.concurrency,
        "seeds": SEEDS
    }

    with open(args.output_file, "w") as f:
        json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://localhost:30000")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--benchmark_file", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--streaming_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=8192)
    args = parser.parse_args()

    asyncio.run(run_evaluation(args))

if __name__ == "__main__":
    main()
