#!/usr/bin/env python3
"""
scripts/114_generate_training_traces_4b.py
Asynchronous Self-Distillation Trace Generation for 4B Models on Train Splits.

Zero Cross-Model Contamination:
- Each model generates its own curriculum from its own thinking mode traces.
- Sourced strictly from openai/gsm8k (train) and DigitalLearningGmbH/MATH-lighteval (train).
- Rigorous exclusion of all problem IDs in benchmark_suite_250.json, MATH-500, and GPQA Diamond.
"""

import os
import re
import json
import time
import asyncio
import argparse
import random
from collections import defaultdict
import httpx
from datasets import load_dataset
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

def load_contamination_set():
    test_questions = set()
    suite_path = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_suite_250.json")
    if os.path.exists(suite_path):
        with open(suite_path) as f:
            suite = json.load(f)
            for item in suite:
                test_questions.add(item["question"].strip().lower())
    math500_path = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_math500.json")
    if os.path.exists(math500_path):
        with open(math500_path) as f:
            m500 = json.load(f)
            for item in m500:
                test_questions.add(item.get("problem", item.get("question", "")).strip().lower())
    return test_questions

def load_gsm8k_candidates(n_candidates=1500, contamination_set=None):
    print(f"Loading GSM8K train split candidates (target: {n_candidates})...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    candidates = []
    for idx, ex in enumerate(ds):
        q = ex["question"].strip()
        if contamination_set and q.lower() in contamination_set:
            continue
        sol = ex["answer"].strip()
        parts = sol.split("####")
        gt_val = parts[1].strip() if len(parts) > 1 else ""
        candidates.append({
            "id": f"gsm8k_train_{idx}",
            "dataset": "gsm8k",
            "question": q,
            "raw_solution": sol,
            "ground_truth": gt_val,
            "level": 1,
            "subject": "arithmetic"
        })
        if len(candidates) >= n_candidates:
            break
    print(f"Loaded {len(candidates)} clean GSM8K train candidates.")
    return candidates

def load_math_candidates(n_candidates=1500, contamination_set=None):
    print(f"Loading MATH-lighteval train split candidates (target: {n_candidates})...")
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    by_bucket = defaultdict(list)
    for idx, ex in enumerate(ds):
        q = ex["problem"].strip()
        if contamination_set and q.lower() in contamination_set:
            continue
        sol = ex["solution"].strip()
        gt_boxed = extract_math_boxed_expression(sol)
        if not gt_boxed:
            continue
        level = ex.get("level", "Level 3")
        subject = ex.get("type", "Algebra")
        by_bucket[(level, subject)].append({
            "id": f"math_train_{idx}",
            "dataset": "math",
            "question": q,
            "raw_solution": sol,
            "ground_truth": gt_boxed,
            "level": level,
            "subject": subject
        })
    candidates = []
    buckets = list(by_bucket.keys())
    random.seed(42)
    random.shuffle(buckets)
    idx_in_bucket = defaultdict(int)
    while len(candidates) < n_candidates:
        added_any = False
        for b in buckets:
            items = by_bucket[b]
            if idx_in_bucket[b] < len(items):
                candidates.append(items[idx_in_bucket[b]])
                idx_in_bucket[b] += 1
                added_any = True
                if len(candidates) >= n_candidates:
                    break
        if not added_any:
            break
    print(f"Loaded {len(candidates)} clean MATH-lighteval train candidates.")
    return candidates

def format_prompt(cand):
    q = cand["question"].rstrip()
    if "\\boxed" not in q:
        prompt = q + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    else:
        prompt = q
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"

def get_sampling_params(model_id):
    is_qwen35 = "3.5" in model_id.lower() or "3_5" in model_id.lower()
    if is_qwen35:
        return {
            "max_new_tokens": 16384,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }
    else:
        return {
            "max_new_tokens": 16384,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }

async def generate_single_trace(client, server_url, prompt_text, sampling_params, timeout=900.0):
    t0 = time.time()
    payload = {
        "text": prompt_text,
        "sampling_params": sampling_params
    }
    try:
        resp = await client.post(f"{server_url}/generate", json=payload, timeout=timeout)
        dur = time.time() - t0
        if resp.status_code != 200:
            return None, dur, False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        content = data.get("text", "")
        meta = data.get("meta_info", {})
        finish_reason = meta.get("finish_reason", {})
        finish_type = finish_reason.get("type", "") if isinstance(finish_reason, dict) else str(finish_reason)
        is_truncated = (finish_type == "length")
        
        think_text = ""
        answer_text = ""
        if THINK_END_TOKEN in content:
            parts = content.split(THINK_END_TOKEN, 1)
            think_text = parts[0].replace("<think>", "").strip()
            answer_text = parts[1].strip()
        elif "<think>" in content:
            think_text = content.replace("<think>", "").strip()
            answer_text = ""
        else:
            answer_text = content.strip()
            
        return {
            "content": content,
            "think_text": think_text,
            "answer_text": answer_text,
            "is_truncated": is_truncated,
            "completion_tokens": meta.get("completion_tokens", 0),
            "dur": dur
        }, dur, is_truncated, None
    except Exception as e:
        return None, time.time() - t0, False, str(e)

def verify_trace_solution(cand, res_dict):
    if res_dict.get("is_truncated", False) or not res_dict.get("answer_text"):
        return False
    ans_text = res_dict["answer_text"]
    gt = cand["ground_truth"]
    gt_target = f"\\boxed{{{gt}}}" if "\\boxed" not in gt else gt
    try:
        parsed_gt = parse(gt_target)
        parsed_pred = parse(ans_text)
        if parsed_gt and parsed_pred:
            return bool(verify(parsed_gt, parsed_pred))
    except Exception:
        pass
    return False

async def run_trace_pipeline(server_url, model_id, concurrency, target_verified, output_file):
    print(f"\n========================================================")
    print(f"=== Self-Distillation Trace Generation: {model_id} ===")
    print(f"Target Verified Traces: {target_verified}")
    print(f"Concurrency: {concurrency}")
    print(f"Output File: {output_file}")
    print(f"========================================================\n")
    
    contamination_set = load_contamination_set()
    print(f"Loaded {len(contamination_set)} test set questions to exclude.")
    
    gsm8k_cands = load_gsm8k_candidates(n_candidates=target_verified, contamination_set=contamination_set)
    math_cands = load_math_candidates(n_candidates=target_verified, contamination_set=contamination_set)
    
    # Interleave candidates
    all_cands = []
    for g, m in zip(gsm8k_cands, math_cands):
        all_cands.append(g)
        all_cands.append(m)
        
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    existing_ids.add(d["id"])
                except Exception:
                    pass
        print(f"Resuming: found {len(existing_ids)} existing verified traces in {output_file}")
        
    sampling_params = get_sampling_params(model_id)
    semaphore = asyncio.Semaphore(concurrency)
    verified_count = len(existing_ids)
    processed_count = 0
    t_start = time.time()
    
    async with httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)) as client:
        async def worker(cand):
            nonlocal verified_count, processed_count
            if verified_count >= target_verified:
                return
            if cand["id"] in existing_ids:
                return
                
            async with semaphore:
                if verified_count >= target_verified:
                    return
                prompt_text = format_prompt(cand)
                res, dur, is_trunc, err = await generate_single_trace(client, server_url, prompt_text, sampling_params)
                processed_count += 1
                
                if res and not is_trunc:
                    is_correct = verify_trace_solution(cand, res)
                    if is_correct:
                        rec = {
                            "id": cand["id"],
                            "dataset": cand["dataset"],
                            "level": cand["level"],
                            "subject": cand["subject"],
                            "question": cand["question"],
                            "prompt": prompt_text,
                            "think": res["think_text"],
                            "answer": res["answer_text"],
                            "raw_content": res["content"],
                            "ground_truth": cand["ground_truth"],
                            "completion_tokens": res["completion_tokens"],
                            "duration_s": dur,
                            "model_id": model_id,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        }
                        with open(output_file, "a") as f_out:
                            f_out.write(json.dumps(rec) + "\n")
                        verified_count += 1
                        
                if processed_count % 10 == 0:
                    elapsed = time.time() - t_start
                    rate = verified_count / (processed_count or 1) * 100.0
                    print(f"Processed: {processed_count} | Verified: {verified_count}/{target_verified} ({rate:.1f}% yield) | Elapsed: {elapsed:.1f}s")
                    
        tasks = [worker(c) for c in all_cands]
        await asyncio.gather(*tasks)
        
    print(f"\nCompleted! Generated {verified_count} verified traces saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--target_verified", type=int, default=1500)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    
    if args.output_file is None:
        safe_name = args.model_id.replace("/", "_").replace(".", "_").lower()
        args.output_file = f"data/self_distill_train_traces_{safe_name}.jsonl"
        
    asyncio.run(run_trace_pipeline(
        server_url=args.server_url,
        model_id=args.model_id,
        concurrency=args.concurrency,
        target_verified=args.target_verified,
        output_file=args.output_file
    ))

if __name__ == "__main__":
    main()
