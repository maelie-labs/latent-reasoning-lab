#!/usr/bin/env python3
"""
scripts/66_generate_training_traces_qwen3.5_2b.py
High-Throughput Self-Distillation Trace Generation for Qwen/Qwen3.5-2B via SGLang on GPU 1.

Strictly Isolated to Model: Qwen/Qwen3.5-2B
Zero Cross-Model Distillation: Traces generated strictly by Qwen3.5-2B itself on train splits.

Objectives:
1. Load clean train-split candidates with zero contamination against test suites:
   - openai/gsm8k (train split, 1,500 candidates)
   - DigitalLearningGmbH/MATH-lighteval (train split, 1,500 candidates stratified by level & subject)
2. Format prompts using official Qwen3.5-2B thinking chat template:
   enable_thinking=True -> ends with `<|im_start|>assistant\n<think>\n`
3. Execute generation under official Qwen3.5-2B benchmark sampling params:
   temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
4. Verify with math_verify against ground-truth reference solution.
5. Save verified traces to data/self_distill_train_traces_qwen3.5_2b.jsonl.
6. Target: ~2,000 verified traces (1,000 GSM8K + 1,000 MATH).
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
from transformers import AutoTokenizer

SGLANG_URL = "http://127.0.0.1:30000"
MODEL_ID = "Qwen/Qwen3.5-2B"
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
        
        item = {
            "id": f"math_train_{idx}",
            "dataset": "math",
            "question": q,
            "raw_solution": sol,
            "ground_truth": gt_boxed,
            "level": level,
            "subject": subject
        }
        by_bucket[(level, subject)].append(item)
        
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
            
    print(f"Loaded {len(candidates)} clean MATH-lighteval train candidates across {len(buckets)} stratified buckets.")
    return candidates

async def generate_single_trace(client, server_url, prompt_text, seed=42, timeout=600.0):
    t0 = time.time()
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "max_new_tokens": 16384,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "sampling_seed": seed,
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }
    }
    try:
        resp = await client.post(f"{server_url}/generate", json=payload, timeout=timeout)
        dur = time.time() - t0
        if resp.status_code != 200:
            return None, dur, False, f"HTTP {resp.status_code}: {resp.text}"
            
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
            "completion_tokens": meta.get("completion_tokens", len(content.split())),
            "dur": dur
        }, dur, is_truncated, None
    except Exception as e:
        return None, time.time() - t0, False, str(e)

def verify_solution(pred_text, gold_target):
    if not pred_text or not gold_target:
        return False
    try:
        gold_parsed = parse(f"\\boxed{{{gold_target}}}")
        pred_parsed = parse(pred_text)
        if verify(gold_parsed, pred_parsed):
            return True
    except Exception:
        pass
        
    pred_boxed = extract_math_boxed_expression(pred_text)
    if pred_boxed:
        try:
            p_val = re.sub(r'[\$,]', '', pred_boxed).strip().lower()
            g_val = re.sub(r'[\$,]', '', str(gold_target)).strip().lower()
            if p_val == g_val:
                return True
            try:
                if abs(float(p_val) - float(g_val)) < 1e-4:
                    return True
            except ValueError:
                pass
        except Exception:
            pass
    return False

async def generate_and_filter_set(
    dataset_name,
    candidates,
    target_verified,
    concurrency,
    tokenizer,
    out_file,
    server_url=SGLANG_URL
):
    print(f"\n========================================================")
    print(f"Starting Generation for {dataset_name.upper()} (Target: {target_verified} verified traces)")
    print(f"Candidates: {len(candidates)} | Concurrency: {concurrency} | Model: {MODEL_ID}")
    print(f"========================================================")
    
    queue = asyncio.Queue()
    for cand in candidates:
        queue.put_nowait(cand)
        
    verified_records = []
    total_processed = 0
    t0_start = time.time()
    
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=600.0) as client:
        async def worker(worker_id):
            nonlocal total_processed
            while not queue.empty():
                if len(verified_records) >= target_verified:
                    break
                cand = await queue.get()
                prompt_messages = [{"role": "user", "content": cand["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
                
                prompt_str = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
                
                result, dur, is_trunc, err = await generate_single_trace(
                    client, server_url, prompt_str, seed=42
                )
                
                total_processed += 1
                if result and not is_trunc and result["answer_text"]:
                    is_correct = verify_solution(result["answer_text"], cand["ground_truth"])
                    has_boxed = "\\boxed{" in result["answer_text"]
                    
                    if is_correct and has_boxed:
                        rec = {
                            "id": cand["id"],
                            "dataset": cand["dataset"],
                            "level": cand["level"],
                            "subject": cand["subject"],
                            "question": cand["question"],
                            "ground_truth": cand["ground_truth"],
                            "prompt": prompt_str,
                            "think": result["think_text"],
                            "answer": result["answer_text"],
                            "completion_tokens": result["completion_tokens"],
                            "dur_seconds": round(dur, 2)
                        }
                        verified_records.append(rec)
                        with open(out_file, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                            
                if total_processed % 25 == 0 or len(verified_records) >= target_verified:
                    elapsed = time.time() - t0_start
                    tok_rate = total_processed / elapsed if elapsed > 0 else 0
                    print(f"[{dataset_name.upper()} Progress] Processed: {total_processed} | Verified: {len(verified_records)}/{target_verified} ({len(verified_records)/max(1, total_processed)*100:.1f}%) | Elapsed: {elapsed:.1f}s")
                    
                queue.task_done()
                
        workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
        await asyncio.gather(*workers)
        
    print(f"\n{dataset_name.upper()} Complete! Kept {len(verified_records)} verified traces.")
    return verified_records

async def main_async():
    parser = argparse.ArgumentParser(description="Self-Distillation Trace Generation for Qwen3.5-2B")
    parser.add_argument("--server_url", type=str, default=SGLANG_URL)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--target_gsm8k", type=int, default=1000)
    parser.add_argument("--target_math", type=int, default=1000)
    parser.add_argument("--out_file", type=str, default="data/self_distill_train_traces_qwen3.5_2b.jsonl")
    args = parser.parse_args()

    contamination = load_contamination_set()
    print(f"Loaded {len(contamination)} contamination test queries.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    gsm_cands = load_gsm8k_candidates(n_candidates=int(args.target_gsm8k * 1.3), contamination_set=contamination)
    math_cands = load_math_candidates(n_candidates=int(args.target_math * 1.5), contamination_set=contamination)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    if os.path.exists(args.out_file):
        os.remove(args.out_file)

    gsm_verified = await generate_and_filter_set(
        "gsm8k", gsm_cands, args.target_gsm8k, args.concurrency, tokenizer, args.out_file, args.server_url
    )

    math_verified = await generate_and_filter_set(
        "math", math_cands, args.target_math, args.concurrency, tokenizer, args.out_file, args.server_url
    )

    print("\n" + "="*70)
    print(f"SELF-DISTILLATION TRACE GENERATION COMPLETED ({MODEL_ID})")
    print(f"Total Verified GSM8K: {len(gsm_verified)}")
    print(f"Total Verified MATH:  {len(math_verified)}")
    print(f"Total Traces Saved:   {len(gsm_verified) + len(math_verified)}")
    print(f"Output File:          {args.out_file}")
    print("="*70)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
