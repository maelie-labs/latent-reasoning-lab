#!/usr/bin/env python3
"""
scripts/60_sglang_eval_arm2.py
High-throughput Arm 2 (Minimal Discrete Thinking Tokens) evaluator using SGLang.
Evaluates the 250 benchmark problems across 4 seeds with K discrete thinking tokens,
forced </think>\\n\\n transition, and max_tokens=8192 for the answer phase.
Runs with SGLang async concurrency (default 32) on GPU 1.
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

def clean_and_extract_candidate(cand):
    if not cand:
        return None
    cand = re.sub(r"\\(?:text|mathbf|mathrm)\{([^}]+)\}", r"\1", str(cand))
    cand = re.sub(r"[\$\\%!\s]", "", cand)
    cand = cand.replace(",", "").strip().rstrip(".")
    m_frac = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", cand)
    if m_frac:
        try:
            return float(m_frac.group(1)) / float(m_frac.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    m_div = re.fullmatch(r"(-?\d+)/(-?\d+)", cand)
    if m_div:
        try:
            return float(m_div.group(1)) / float(m_div.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(cand)
    except ValueError:
        return cand.strip()

def extract_answer(text):
    if not text:
        return None, ""
    b = extract_math_boxed_expression(text)
    if b:
        return clean_and_extract_candidate(b), b
    patterns = [
        r"(?:the\s+final\s+answer\s+is|the\s+answer\s+is|is\s+equal\s+to|final\s+answer:)\s*[:=]?\s*([^\n\.\,]+)",
        r"(?:answer|result):\s*([^\n\.\,]+)"
    ]
    for pat in patterns:
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            cand = m[-1].strip()
            val = clean_and_extract_candidate(cand)
            if val is not None:
                return val, cand
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if lines:
        last = lines[-1]
        nums = re.findall(r"-?\d+(?:\.\d+)?", last)
        if nums:
            return clean_and_extract_candidate(nums[-1]), nums[-1]
    return None, ""

def check_match(pred_val, pred_raw, gold_val, gold_boxed, full_pred_text="", full_gold_text=""):
    # 1. Symbolic math_verify check
    if full_pred_text and (gold_boxed or full_gold_text):
        target = gold_boxed if gold_boxed else full_gold_text
        try:
            gold_parsed = parse(f"\\boxed{{{target}}}")
            pred_parsed = parse(full_pred_text)
            if verify(gold_parsed, pred_parsed):
                return True
        except Exception:
            pass
    # 2. Extracted candidate check
    if pred_val is not None and gold_val is not None:
        if isinstance(pred_val, (int, float)) and isinstance(gold_val, (int, float)):
            if abs(pred_val - gold_val) < 1e-4:
                return True
        elif str(pred_val).strip().lower() == str(gold_val).strip().lower():
            return True
    return False

async def query_arm2_problem(client, server_url, raw_prompt, seed, k_tokens=6, max_ans_tokens=8192):
    t0 = time.time()
    # Format prompt for thinking mode
    prompt_text = f"<|im_start|>user\n{raw_prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    
    try:
        # Phase 1: Generate exactly K discrete thinking tokens
        req1 = {
            "text": prompt_text,
            "sampling_params": {
                "max_new_tokens": k_tokens,
                "temperature": 0.6,
                "top_p": 0.95,
                "sampling_seed": seed,
                "stop": ["</think>"]
            }
        }
        
        res1 = await client.post(f"{server_url}/generate", json=req1, timeout=120.0)
        if res1.status_code != 200:
            return {
                "k_tokens": k_tokens,
                "think_text": "",
                "think_time_ms": 0.0,
                "ans_tokens": 0,
                "total_time_s": round(time.time() - t0, 3),
                "is_truncated": True,
                "content": "",
                "pred_val": None,
                "pred_raw": ""
            }
        data1 = res1.json()
        think_text = data1.get("text", "")
        t_think_s = time.time() - t0
        
        # Phase 2: Force transition </think>\n\n and generate answer
        ans_prompt_text = prompt_text + think_text + "\n</think>\n\n"
        req2 = {
            "text": ans_prompt_text,
            "sampling_params": {
                "max_new_tokens": max_ans_tokens,
                "temperature": 0.6,
                "top_p": 0.95,
                "sampling_seed": seed,
                "stop": ["<|im_end|>", "<|endoftext|>"]
            }
        }
        
        res2 = await client.post(f"{server_url}/generate", json=req2, timeout=300.0)
        if res2.status_code != 200:
            return {
                "k_tokens": k_tokens,
                "think_text": think_text,
                "think_time_ms": round(t_think_s * 1000.0, 2),
                "ans_tokens": 0,
                "total_time_s": round(time.time() - t0, 3),
                "is_truncated": True,
                "content": "",
                "pred_val": None,
                "pred_raw": ""
            }
        data2 = res2.json()
        ans_text = data2.get("text", "")
        meta2 = data2.get("meta_info", {})
        finish_reason = meta2.get("finish_reason", {})
        finish_type = finish_reason.get("type", "") if isinstance(finish_reason, dict) else str(finish_reason)
        
        total_time_s = time.time() - t0
        ans_tokens = meta2.get("completion_tokens", len(ans_text.split()))
        is_truncated = (finish_type == "length")
        
        pred_val, pred_raw = extract_answer(ans_text)
        return {
            "k_tokens": k_tokens,
            "think_text": think_text,
            "think_time_ms": round(t_think_s * 1000.0, 2),
            "ans_tokens": ans_tokens,
            "total_time_s": round(total_time_s, 3),
            "is_truncated": is_truncated,
            "content": ans_text,
            "pred_val": pred_val,
            "pred_raw": pred_raw
        }
    except Exception as e:
        return {
            "k_tokens": k_tokens,
            "think_text": "",
            "think_time_ms": 0.0,
            "ans_tokens": 0,
            "total_time_s": round(time.time() - t0, 3),
            "is_truncated": True,
            "content": "",
            "pred_val": None,
            "pred_raw": ""
        }

async def worker(queue, client, server_url, out_f, progress, lock, k_tokens, max_ans_tokens):
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
        gt_boxed = extract_math_boxed_expression(sol)
        gt_val = clean_and_extract_candidate(gt_boxed or sol)
        
        run = await query_arm2_problem(client, server_url, q, seed, k_tokens=k_tokens, max_ans_tokens=max_ans_tokens)
        
        is_correct = check_match(
            run["pred_val"],
            run["pred_raw"],
            gt_val,
            gt_boxed or sol,
            full_pred_text=run["content"],
            full_gold_text=sol
        )
        # Strict truncation scoring protocol: sequences hitting limit are incorrect
        if run["is_truncated"]:
            is_correct = False
            
        rec = {
            "problem_id": prob["id"],
            "benchmark": prob["benchmark"],
            "seed": seed,
            "k_tokens": k_tokens,
            "is_correct": 1 if is_correct else 0,
            "gt_val": gt_val,
            "gt_boxed": gt_boxed,
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
                print(f"[Arm 2 (K={k_tokens}) | {n}/{progress['total']}] Acc: {acc:.2f}% | Trunc: {trunc:.2f}%")
                
        queue.task_done()

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--suite_file", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--k_tokens", type=int, default=6)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max_ans_tokens", type=int, default=8192)
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"data/streaming_arm2_minimal_discrete_qwen_qwen3-1.7b_k{args.k_tokens}.jsonl"

    with open(args.suite_file) as f:
        problems = json.load(f)

    total_tasks = len(problems) * len(args.seeds)
    print(f"=== Fast SGLang Arm 2 Evaluation: {len(problems)} problems x {len(args.seeds)} seeds = {total_tasks} runs ===")
    print(f"K Tokens: {args.k_tokens} | Concurrency: {args.concurrency} | Endpoint: {args.server_url}")

    progress = {"done": 0, "correct": 0, "truncated": 0, "total": total_tasks}
    lock = asyncio.Lock()
    queue = asyncio.Queue()

    for prob in problems:
        for seed in args.seeds:
            queue.put_nowait((prob, seed))

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as out_f:
        limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency)
        async with httpx.AsyncClient(limits=limits, timeout=300.0) as client:
            workers = [
                asyncio.create_task(
                    worker(queue, client, args.server_url, out_f, progress, lock, args.k_tokens, args.max_ans_tokens)
                ) for _ in range(args.concurrency)
            ]
            await queue.join()
            for w in workers:
                w.cancel()

    # Final summary calculations
    acc = progress["correct"] / total_tasks * 100
    trunc = progress["truncated"] / total_tasks * 100
    
    # Section 11: Mandatory Post-Eval Spot-Check & Telemetry Sanity Audit
    print("\n" + "="*50)
    print(f"=== Section 11 Telemetry & Sanity Audit: Arm 2 (K={args.k_tokens}) ===")
    records = []
    with open(args.output_file) as rf:
        for line in rf:
            if line.strip():
                records.append(json.loads(line))
                
    ans_lens = [r["run"].get("ans_tokens", 0) for r in records if r["run"].get("ans_tokens", 0) > 0]
    if ans_lens:
        import numpy as np
        print(f"Answer Length Distribution: min={int(np.min(ans_lens))}, median={int(np.median(ans_lens))}, p90={int(np.percentile(ans_lens, 90))}, max={int(np.max(ans_lens))}")
    print(f"Truncation Rate: {trunc:.2f}% (Count: {progress['truncated']}/{total_tasks})")
    
    print("\n--- Spot-Checking 5 Output Samples ---")
    for i, r in enumerate(records[:5]):
        prob_id = r["problem_id"]
        seed = r["seed"]
        corr = r["is_correct"]
        t_text = r["run"].get("think_text", "")
        c_text = r["run"].get("content", "")[:120].replace("\n", " ")
        print(f"Sample {i+1} [{prob_id}, seed {seed}]: Correct={bool(corr)} | Think: {len(t_text)} chars | Ans: {c_text}...")
        
    print("\n" + "="*50)
    print(f"Arm 2 (K={args.k_tokens}) Complete: {total_tasks} queries evaluated across {len(args.seeds)} seeds.")
    print(f"Pass@1 Accuracy: {acc:.2f}% | Truncation Rate: {trunc:.2f}%")
    print(f"Streaming results saved to: {args.output_file}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main_async())
