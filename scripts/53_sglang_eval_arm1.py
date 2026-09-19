#!/usr/bin/env python3
"""
scripts/53_sglang_eval_arm1.py
High-throughput Arm 1 (Base Direct) evaluator using SGLang.
Evaluates the 250 problems across 4 seeds with max_tokens=8192.
Runs with concurrency 48 to saturate the RTX PRO 4500 Blackwell.
"""

import os
import re
import json
import time
import asyncio
import argparse
import httpx
from math_verify import parse, verify

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



def check_match(pred_text, gt_target):
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

async def query_problem(client, server_url, prompt, seed, max_tokens=8192):
    prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.80,
            "top_k": 20,
            "presence_penalty": 1.5,
            "sampling_seed": seed
        }
    }
    t0 = time.time()
    try:
        resp = await client.post(f"{server_url}/generate", json=payload, timeout=300.0)
        dur = time.time() - t0
        
        if resp.status_code != 200:
            return {
                "think_tokens": 0,
                "ans_tokens": 0,
                "think_time_ms": 0.0,
                "total_time_s": round(dur, 3),
                "is_truncated": True,
                "content": "",
                "pred_val": None,
                "pred_raw": "",
                "error": resp.text
            }
    except Exception as e:
        dur = time.time() - t0
        return {
            "think_tokens": 0,
            "ans_tokens": 0,
            "think_time_ms": 0.0,
            "total_time_s": round(dur, 3),
            "is_truncated": True,
            "content": "",
            "pred_val": None,
            "pred_raw": "",
            "error": str(e)
        }
        
    data = resp.json()
    content = data.get("text", "").strip()
    meta = data.get("meta_info", {})
    finish_reason = meta.get("finish_reason", {})
    if isinstance(finish_reason, dict):
        finish_type = finish_reason.get("type", "stop")
    else:
        finish_type = str(finish_reason)
    is_truncated = (finish_type == "length")
    ans_tokens = meta.get("completion_tokens", len(content.split()))
    
    pred_boxed = extract_math_boxed_expression(content)
    return {
        "think_tokens": 0,
        "ans_tokens": ans_tokens,
        "think_time_ms": 0.0,
        "total_time_s": round(dur, 3),
        "is_truncated": is_truncated,
        "content": content,
        "pred_boxed": pred_boxed
    }


async def worker(queue, client, server_url, out_f, progress, lock):
    while True:
        try:
            item = await queue.get()
        except asyncio.CancelledError:
            break
        if item is None:
            queue.task_done()
            break
            
        prob, seed = item
        q = prob["question"]
        sol = prob["solution"]
        if "####" in sol:
            gt_boxed = sol.split("####")[-1].strip()
        else:
            gt_boxed = extract_math_boxed_expression(sol)
        gt_target = f"\\boxed{{{gt_boxed}}}" if gt_boxed else f"\\boxed{{{sol}}}"
        
        prompt = f"{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        run = await query_problem(client, server_url, prompt, seed)
        
        is_correct = False
        if not run["is_truncated"]:
            is_correct = check_match(run["content"], gt_target)
        
        rec = {
            "problem_id": prob["id"],
            "benchmark": prob["benchmark"],
            "seed": seed,
            "is_correct": 1 if is_correct else 0,
            "gt_target": gt_target,
            "run": run
        }
        
        async with lock:
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            progress["done"] += 1
            if is_correct:
                progress["correct"] += 1
            if run["is_truncated"]:
                progress["truncated"] += 1
                
            n = progress["done"]
            if n % 50 == 0 or n == progress["total"]:
                acc = progress["correct"] / n * 100
                trunc = progress["truncated"] / n * 100
                print(f"[Arm 1 Direct (SGLang) | {n}/{progress['total']}] Acc: {acc:.2f}% | Trunc: {trunc:.2f}%")
                
        queue.task_done()

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--suite_file", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--output_file", type=str, default="data/streaming_arm1_direct_qwen_qwen3-1.7b.jsonl")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789])
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()

    with open(args.suite_file) as f:
        problems = json.load(f)

    total_tasks = len(problems) * len(args.seeds)
    print(f"=== Fast SGLang Arm 1 Evaluation: {len(problems)} problems x {len(args.seeds)} seeds = {total_tasks} runs ===")
    print(f"Concurrency: {args.concurrency} | Endpoint: {args.server_url}")

    seen = set()
    records = []
    if os.path.exists(args.output_file):
        with open(args.output_file) as rf:
            for line in rf:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        seen.add((rec["problem_id"], rec["seed"]))
                        records.append(rec)
                    except Exception:
                        pass

    done_count = len(records)
    correct_count = sum(r.get("is_correct", 0) for r in records)
    trunc_count = sum(1 for r in records if r.get("run", {}).get("is_truncated", False))
    progress = {"done": done_count, "correct": correct_count, "truncated": trunc_count, "total": total_tasks}
    lock = asyncio.Lock()
    queue = asyncio.Queue()

    for prob in problems:
        for seed in args.seeds:
            if (prob["id"], seed) not in seen:
                queue.put_nowait((prob, seed))

    remaining = queue.qsize()
    print(f"Resuming: {done_count} already completed, {remaining} remaining.")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "a") as out_f:
        limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency)
        async with httpx.AsyncClient(limits=limits, timeout=300.0) as client:
            workers = [asyncio.create_task(worker(queue, client, args.server_url, out_f, progress, lock)) for _ in range(args.concurrency)]
            await queue.join()
            for w in workers:
                w.cancel()

    acc = progress["correct"] / total_tasks * 100
    trunc = progress["truncated"] / total_tasks * 100
    
    # Section 11: Mandatory Post-Eval Spot-Check & Telemetry Sanity Audit
    print("\n" + "="*50)
    print(f"=== Section 11 Telemetry & Sanity Audit: Arm 1 (Base Direct) ===")
    records = []
    with open(args.output_file) as rf:
        for line in rf:
            if line.strip():
                records.append(json.loads(line))
                
    ans_lens = [r["run"].get("ans_tokens", 0) for r in records if r["run"].get("ans_tokens", 0) > 0]
    if ans_lens:
        import numpy as np
        print(f"Token Length Distribution: min={int(np.min(ans_lens))}, median={int(np.median(ans_lens))}, p90={int(np.percentile(ans_lens, 90))}, max={int(np.max(ans_lens))}")
    print(f"Truncation Rate: {trunc:.2f}% (Count: {progress['truncated']}/{total_tasks})")
    if trunc > 1.0:
        print(f"WARNING: Truncation rate {trunc:.2f}% exceeds 1.0% threshold for direct arm!")
        
    print("\n--- Spot-Checking 5 Output Samples ---")
    for i, r in enumerate(records[:5]):
        prob_id = r["problem_id"]
        seed = r["seed"]
        corr = r["is_correct"]
        c_text = r["run"].get("content", "")[:120].replace("\n", " ")
        has_think = "<think>" in r["run"].get("content", "")
        print(f"Sample {i+1} [{prob_id}, seed {seed}]: Correct={bool(corr)} | Has <think>: {has_think} | Ans: {c_text}...")
        
    print("\n" + "="*50)
    print(f"\nCompleted all {total_tasks} runs! Final Acc: {acc:.2f}% | Trunc: {trunc:.2f}%")
    print(f"Saved to: {args.output_file}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main_async())
