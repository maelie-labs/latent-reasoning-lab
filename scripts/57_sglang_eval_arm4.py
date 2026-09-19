#!/usr/bin/env python3
"""
scripts/57_sglang_eval_arm4.py
Fast SGLang Arm 4 Evaluator (Unconstrained Thinking Mode / CoT).

Evaluates the 250-problem suite (150 GSM8K + 100 MATH-500 Level 3-5)
across SEEDS = [42, 123, 456, 789] (1,000 total evaluations) on SGLang
under official sampling parameters (temperature=0.6, top_p=0.95, max_tokens=32768).

Adheres strictly to:
- Section 10: Model-Specific Thinking Mode Control Invariant (direct /generate endpoint with pre-rendered prompt)
- Section 11: Mandatory Post-Eval Spot-Check & Telemetry Sanity Invariant
- Streaming persistence to .jsonl with auto-resume
- Resilient exception handling (timeouts/errors flagged as is_truncated=True, scored as incorrect)
"""

import os
import re
import json
import time
import random
import asyncio
import argparse
import httpx
import numpy as np
from math_verify import parse, verify

SEEDS = [42, 123, 456, 789]
THINK_END_TOKEN = "</think>"

def extract_math_boxed_expression(text):
    if not text:
        return None
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{" :
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i-1].strip()
    return None

def clean_and_extract_candidate(raw_str):
    if not raw_str:
        return None
    s = str(raw_str).strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = s.replace(r"\text{", "").replace("}", "")
    s = s.replace(r"\mathbf{", "").replace(r"\mathrm{", "")
    s = s.replace(",", "")
    s = re.sub(r"\\(?:degree|circ)", "", s)
    s = s.replace("%", "").strip()
    try:
        return float(s)
    except Exception:
        pass
    m = re.match(r"^([-+]?\d+)/([-+]?\d+)$", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except Exception:
            pass
    return s

def extract_answer(content_text):
    if not content_text:
        return None, ""
    b = extract_math_boxed_expression(content_text)
    if b:
        val = clean_and_extract_candidate(b)
        if val is not None:
            return val, b
    ans_match = re.findall(r"(?:final answer is|answer is|the answer:)\s*[:\$]?\s*([^\n\.\$]+)", content_text, re.IGNORECASE)
    if ans_match:
        val = clean_and_extract_candidate(ans_match[-1])
        if val is not None:
            return val, ans_match[-1]
    nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", content_text)
    if nums:
        val = clean_and_extract_candidate(nums[-1])
        if val is not None:
            return val, nums[-1]
    return None, ""

async def query_sglang(client, server_url, prompt, seed, max_tokens=32768):
    prompt_text = (
        f"<|im_start|>user\n{prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "sampling_seed": seed
        }
    }
    t0 = time.time()
    try:
        resp = await client.post(f"{server_url}/generate", json=payload, timeout=600.0)
        dur = time.time() - t0
        if resp.status_code != 200:
            return {
                "error": resp.text,
                "dur": round(dur, 2),
                "is_truncated": True,
                "total_tokens": 0,
                "thinking_text": "",
                "content_text": ""
            }
            
        data = resp.json()
        raw_text = data.get("text", "")
        meta = data.get("meta_info", {})
        finish_reason = meta.get("finish_reason", {})
        if isinstance(finish_reason, dict):
            finish_type = finish_reason.get("type", "stop")
        else:
            finish_type = str(finish_reason)
            
        is_truncated = (finish_type == "length")
        total_tokens = meta.get("completion_tokens", len(raw_text.split()))
        
        if THINK_END_TOKEN in raw_text:
            parts = raw_text.split(THINK_END_TOKEN, 1)
            thinking_text = parts[0].strip()
            content_text = parts[1].strip()
        else:
            thinking_text = raw_text.strip()
            content_text = ""
            is_truncated = True  # Truncated before completing think block
            
        return {
            "dur": round(dur, 2),
            "total_tokens": total_tokens,
            "is_truncated": is_truncated,
            "thinking_text": thinking_text,
            "content_text": content_text
        }
    except Exception as e:
        dur = time.time() - t0
        return {
            "error": str(e),
            "dur": round(dur, 2),
            "is_truncated": True,
            "total_tokens": 0,
            "thinking_text": "",
            "content_text": ""
        }

async def main_async():
    parser = argparse.ArgumentParser(description="Fast SGLang Arm 4 Evaluator (Unconstrained CoT)")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--suite_file", type=str, default=None)
    parser.add_argument("--streaming_file", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789])
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    suite_path = args.suite_file if args.suite_file else os.path.join(data_dir, "benchmark_suite_250.json")
    with open(suite_path) as f:
        problems = json.load(f)
        
    model_tag = "qwen_qwen3-1.7b"
    streaming_path = args.streaming_file if args.streaming_file else os.path.join(data_dir, f"streaming_arm4_thinking_{model_tag}.jsonl")
    if args.output_file is None:
        args.output_file = os.path.join(data_dir, f"eval_arm4_{model_tag}.json")
        
    seeds_to_use = args.seeds
    print(f"=== Running Fast SGLang Arm 4 Evaluator ===")
    print(f"Server: {args.server_url} | Concurrency: {args.concurrency}")
    print(f"Problems: {len(problems)} | Seeds: {seeds_to_use} | Total Runs: {len(problems) * len(seeds_to_use)}")
    print(f"Streaming Output: {streaming_path}")
    
    # Detect model
    model_name = "Qwen/Qwen3-1.7B"
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{args.server_url}/v1/models", timeout=5.0)
            if r.status_code == 200:
                model_name = r.json()["data"][0]["id"]
    except Exception:
        pass
    print(f"Detected Model: {model_name}")

    # Load already evaluated items to support resuming
    completed_entries = []
    completed_keys = set()
    if os.path.exists(streaming_path):
        with open(streaming_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed_entries.append(rec)
                        completed_keys.add((rec["id"], rec["seed"]))
                    except Exception:
                        pass
        print(f"Resuming: Loaded {len(completed_entries)} already completed evaluations.")

    items_to_eval = []
    for prob in problems:
        for s in seeds_to_use:
            if (prob["id"], s) not in completed_keys:
                items_to_eval.append((prob, s))
                
    print(f"Pending evaluations to execute: {len(items_to_eval)}")

    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency + 10, max_keepalive_connections=args.concurrency + 10)
    all_results = list(completed_entries)
    file_lock = asyncio.Lock()
    t_start = time.time()
    
    if items_to_eval:
        streaming_file = open(streaming_path, "a")
        async with httpx.AsyncClient(limits=limits) as client:
            async def worker(idx, prob, seed):
                async with semaphore:
                    q = prob["question"]
                    gt_sol = prob["solution"]
                    gt_boxed = extract_math_boxed_expression(gt_sol)
                    
                    res = await query_sglang(client, args.server_url, q, seed, max_tokens=32768)
                    
                    # Check correctness
                    content = res.get("content_text", "")
                    is_correct = False
                    if not res.get("is_truncated", False) and content:
                        try:
                            gt_parsed = parse(f"\\boxed{{{gt_boxed or gt_sol}}}")
                            pred_parsed = parse(content)
                            is_correct = verify(gt_parsed, pred_parsed)
                        except Exception:
                            is_correct = False
                            
                    entry = {
                        "id": prob["id"],
                        "seed": seed,
                        "benchmark": prob.get("benchmark", ""),
                        "is_correct": is_correct,
                        "is_truncated": res.get("is_truncated", False),
                        "total_tokens": res.get("total_tokens", 0),
                        "dur": res.get("dur", 0.0),
                        "thinking_text": res.get("thinking_text", ""),
                        "content": content
                    }
                    
                    async with file_lock:
                        streaming_file.write(json.dumps(entry) + "\n")
                        streaming_file.flush()
                        all_results.append(entry)
                        n = len(all_results)
                        if n % 25 == 0 or n == len(problems) * len(seeds_to_use):
                            acc = np.mean([r["is_correct"] for r in all_results]) * 100.0
                            trunc = np.mean([r["is_truncated"] for r in all_results]) * 100.0
                            elapsed = time.time() - t_start
                            toks = sum(r["total_tokens"] for r in all_results)
                            th = toks / elapsed if elapsed > 0 else 0
                            print(f"  [{n}/{len(problems) * len(seeds_to_use)}] Acc: {acc:5.2f}% | Trunc: {trunc:4.1f}% | Tok/s: {th:6.1f} | Elapsed: {elapsed:5.1f}s")

            tasks = [worker(i, p, s) for i, (p, s) in enumerate(items_to_eval)]
            await asyncio.gather(*tasks)
        streaming_file.close()

    # Save summary
    acc = float(np.mean([r["is_correct"] for r in all_results]) * 100.0)
    trunc_rate = float(np.mean([r["is_truncated"] for r in all_results]) * 100.0)
    token_lengths = [r["total_tokens"] for r in all_results if r["total_tokens"] > 0]
    
    summary = {
        "arm": "arm4_unconstrained_cot",
        "model": model_name,
        "num_evaluations": len(all_results),
        "mean_accuracy_pct": round(acc, 2),
        "truncation_rate_pct": round(trunc_rate, 2),
        "total_wall_clock_s": round(time.time() - t_start, 2),
        "token_lengths": {
            "min": int(np.min(token_lengths)) if token_lengths else 0,
            "median": float(np.median(token_lengths)) if token_lengths else 0.0,
            "p90": float(np.percentile(token_lengths, 90)) if token_lengths else 0.0,
            "max": int(np.max(token_lengths)) if token_lengths else 0
        },
        "results": all_results
    }
    with open(args.output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nArm 4 Finished! Final Accuracy: {acc:.2f}% | Truncation: {trunc_rate:.2f}% | Total Time: {time.time()-t_start:.1f}s")
    print(f"Results saved to: {args.output_file}")

    # ========================================================
    # Mandatory Post-Eval Spot-Check & Telemetry Sanity Audit
    # ========================================================
    print("\n" + "=" * 60)
    print("=== MANDATORY POST-EVAL SPOT-CHECK & TELEMETRY SANITY AUDIT ===")
    print("=" * 60)
    print(f"Total Evaluations: {len(all_results)}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Truncation Rate: {trunc_rate:.2f}% ({sum(1 for r in all_results if r["is_truncated"])}/{len(all_results)})")
    if token_lengths:
        print(f"Token Lengths: Min={np.min(token_lengths)} | Median={np.median(token_lengths):.0f} | P90={np.percentile(token_lengths, 90):.0f} | Max={np.max(token_lengths)}")
    
    # 5 Sample Spot Checks
    print("\n--- Raw Generation Spot-Check (5 Random Samples) ---")
    random.seed(42)
    samples = random.sample(all_results, min(5, len(all_results)))
    for i, s in enumerate(samples):
        print(f"\n[Sample {i+1}] ID: {s["id"]} (Seed {s["seed"]}, {s["benchmark"]})")
        print(f"   Correct: {bool(s["is_correct"])} | Truncated: {bool(s["is_truncated"])} | Tokens: {s["total_tokens"]}")
        print(f"   Thinking Length (chars): {len(s.get("thinking_text", ""))}")
        print(f"   Content Length (chars): {len(s.get("content", ""))}")
        content_snippet = s.get("content", "")[-200:].replace("\n", " ") if s.get("content") else "[EMPTY]"
        print(f"   Content Tail: ...{content_snippet}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main_async())
