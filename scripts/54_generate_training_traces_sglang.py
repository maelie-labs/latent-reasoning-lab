#!/usr/bin/env python3
"""
scripts/54_generate_training_traces_sglang.py
High-throughput self-distillation trace generation via SGLang on RTX PRO 4500 (GPU 1).

Generates verified reasoning traces from disjoint train splits:
- openai/gsm8k (train split, 1,500 candidates)
- DigitalLearningGmbH/MATH-lighteval (train split, 1,500 candidates stratified by level & subject)

Filtering:
- Evaluated with math_verify against ground-truth solutions.
- Only verified correct, non-truncated traces are saved.
- Target: 1,000 verified GSM8K traces + 1,000 verified MATH traces (~2,000 total).
- Generates stratified 500 and 1,000 trace manifests for the training-volume ablation.
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
MODEL_NAME = "Qwen/Qwen3-1.7B"
THINK_END_TOKEN = "</think>"

def clean_question(text):
    return text.strip()

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
    """Load all test questions to guarantee zero test-set contamination."""
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
    # Stratified round-robin selection across (level, subject) buckets
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
            
    print(f"Loaded {len(candidates)} clean stratified MATH train candidates.")
    return candidates

async def query_sglang_worker(client, server_url, prompt, max_tokens=32768):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": True},
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
            "top_k": 20
        }
    }
    
    t0 = time.time()
    try:
        resp = await client.post(f"{server_url}/v1/chat/completions", json=payload, timeout=600.0)
        dur = time.time() - t0
        if resp.status_code != 200:
            return None, dur, False, f"HTTP {resp.status_code}: {resp.text}"
            
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        is_truncated = (choice.get("finish_reason") == "length")
        
        # Deconstruct <think> and </think>
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
            "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            "dur": dur
        }, dur, is_truncated, None
    except Exception as e:
        return None, time.time() - t0, False, str(e)

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
    print(f"Candidates: {len(candidates)} | Concurrency: {concurrency}")
    print(f"========================================================")
    
    queue = asyncio.Queue()
    for cand in candidates:
        queue.put_nowait(cand)
        
    verified_records = []
    total_processed = 0
    total_verified = 0
    total_rejected = 0
    total_tokens_gen = 0
    t_start = time.time()
    lock = asyncio.Lock()
    stop_event = asyncio.Event()
    
    async with httpx.AsyncClient() as client:
        async def worker():
            nonlocal total_processed, total_verified, total_rejected, total_tokens_gen
            while not stop_event.is_set():
                try:
                    candidate = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
                prompt = f"{candidate['question']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                
                res, dur, is_truncated, err = await query_sglang_worker(client, server_url, prompt)
                
                if stop_event.is_set():
                    queue.task_done()
                    break
                    
                if err or res is None or is_truncated:
                    async with lock:
                        total_processed += 1
                        total_rejected += 1
                    queue.task_done()
                    continue
                    
                gt_str = f"\\boxed{{{candidate['ground_truth']}}}"
                pred_str = res["answer_text"] or res["content"]
                
                try:
                    gt_parsed = parse(gt_str)
                    pred_parsed = parse(pred_str)
                    is_correct = verify(gt_parsed, pred_parsed)
                except Exception:
                    is_correct = False
                    
                async with lock:
                    total_processed += 1
                    total_tokens_gen += res["completion_tokens"]
                    
                    if is_correct and not stop_event.is_set():
                        total_verified += 1
                        
                        messages = [{"role": "user", "content": candidate["question"]}]
                        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        prompt_with_think = formatted_prompt + "<think>\n"
                        
                        think_tokens = tokenizer.encode(res["think_text"], add_special_tokens=False)
                        answer_tokens = tokenizer.encode(f"</think>\n\n{res['answer_text']}", add_special_tokens=False)
                        
                        record = {
                            "id": candidate["id"],
                            "dataset": candidate["dataset"],
                            "level": candidate["level"],
                            "subject": candidate["subject"],
                            "question": candidate["question"],
                            "ground_truth": candidate["ground_truth"],
                            "formatted_prompt": prompt_with_think,
                            "think_text": res["think_text"],
                            "answer_text": res["answer_text"],
                            "full_completion": res["content"],
                            "num_think_tokens": len(think_tokens),
                            "num_answer_tokens": len(answer_tokens),
                            "dur": round(res["dur"], 2)
                        }
                        verified_records.append(record)
                        out_file.write(json.dumps(record) + "\n")
                        out_file.flush()
                        
                        elapsed = time.time() - t_start
                        tok_per_sec = total_tokens_gen / elapsed if elapsed > 0 else 0
                        print(f"[{dataset_name.upper()}] [{total_verified}/{target_verified}] "
                              f"(Processed: {total_processed}, Acc: {100*total_verified/total_processed:.1f}%, "
                              f"Throughput: {tok_per_sec:.0f} tok/s, Latency: {res['dur']:.1f}s)")
                              
                        if total_verified >= target_verified:
                            stop_event.set()
                            for w in workers:
                                if w is not asyncio.current_task():
                                    w.cancel()
                    else:
                        total_rejected += 1
                        if total_processed % 25 == 0:
                            elapsed = time.time() - t_start
                            tok_per_sec = total_tokens_gen / elapsed if elapsed > 0 else 0
                            print(f"[{dataset_name.upper()}] [Progress] "
                                  f"Verified: {total_verified}/{target_verified} | Processed: {total_processed} | "
                                  f"Throughput: {tok_per_sec:.0f} tok/s")
                                  
                queue.task_done()
                
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers, return_exceptions=True)
        
    print(f"\nFinished {dataset_name.upper()}: {len(verified_records)} verified out of {total_processed} candidates.")
    return verified_records

def generate_ablation_manifest(verified_records, manifest_path):
    """
    Creates stratified 500 and 1,000 trace manifests for the pre-sweep training volume ablation.
    Stratified by dataset (50% GSM8K, 50% MATH) and MATH level.
    """
    gsm_records = [r for r in verified_records if r["dataset"] == "gsm8k"]
    math_records = [r for r in verified_records if r["dataset"] == "math"]
    
    random.seed(42)
    random.shuffle(gsm_records)
    
    # Stratify MATH records by level
    math_by_level = defaultdict(list)
    for r in math_records:
        math_by_level[str(r["level"])].append(r)
        
    for lvl in math_by_level:
        random.shuffle(math_by_level[lvl])
        
    def sample_math(n_target):
        sampled = []
        levels = list(math_by_level.keys())
        idx_map = {lvl: 0 for lvl in levels}
        while len(sampled) < n_target:
            added = False
            for lvl in levels:
                if idx_map[lvl] < len(math_by_level[lvl]):
                    sampled.append(math_by_level[lvl][idx_map[lvl]])
                    idx_map[lvl] += 1
                    added = True
                    if len(sampled) >= n_target:
                        break
            if not added:
                break
        return sampled

    # 500 subset: 250 GSM8K, 250 MATH
    n_gsm_500 = min(250, len(gsm_records))
    n_math_500 = min(250, len(math_records))
    sub_500 = gsm_records[:n_gsm_500] + sample_math(n_math_500)
    
    # 1000 subset: 500 GSM8K, 500 MATH
    n_gsm_1000 = min(500, len(gsm_records))
    n_math_1000 = min(500, len(math_records))
    sub_1000 = gsm_records[:n_gsm_1000] + sample_math(n_math_1000)
    
    manifest = {
        "volume_500_ids": [r["id"] for r in sub_500],
        "volume_1000_ids": [r["id"] for r in sub_1000],
        "volume_full_count": len(verified_records),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Pre-sweep training volume ablation manifests: 500, 1000, and full verified set."
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved training volume ablation manifest to: {manifest_path}")

async def main():
    parser = argparse.ArgumentParser(description="Generate verified self-distillation training traces.")
    parser.add_argument("--concurrency", type=int, default=32, help="SGLang concurrency (streams)")
    parser.add_argument("--target_per_set", type=int, default=1500, help="Target verified traces per dataset (default: 1500 = full set)")
    parser.add_argument("--output", type=str, default=None, help="Output jsonl file")
    args = parser.parse_args()
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    if args.output is None:
        args.output = os.path.join(data_dir, "self_distill_train_traces_qwen3_1.7b.jsonl")
    manifest_path = os.path.join(data_dir, "ablation_subsets_manifest.json")
    
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    contamination_set = load_contamination_set()
    print(f"Loaded {len(contamination_set)} test set questions to exclude.")
    
    gsm_candidates = load_gsm8k_candidates(n_candidates=1500, contamination_set=contamination_set)
    math_candidates = load_math_candidates(n_candidates=1500, contamination_set=contamination_set)
    
    existing_records = []
    existing_ids = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        existing_records.append(rec)
                        existing_ids.add(rec["id"])
                    except Exception:
                        pass
        print(f"Resuming: found {len(existing_records)} existing verified traces in {args.output}")
        
    gsm_candidates = [c for c in gsm_candidates if c["id"] not in existing_ids]
    math_candidates = [c for c in math_candidates if c["id"] not in existing_ids]
    print(f"Remaining candidates to evaluate: {len(gsm_candidates)} GSM8K, {len(math_candidates)} MATH (Total: {len(gsm_candidates) + len(math_candidates)})")
    
    all_verified = list(existing_records)
    with open(args.output, "a", encoding="utf-8") as out_file:
        if gsm_candidates:
            gsm_target = max(1, args.target_per_set - sum(1 for r in existing_records if r.get("dataset") == "gsm8k"))
            gsm_verified = await generate_and_filter_set(
                "gsm8k", gsm_candidates, gsm_target, args.concurrency, tokenizer, out_file
            )
            all_verified.extend(gsm_verified)
            
        if math_candidates:
            math_target = max(1, args.target_per_set - sum(1 for r in existing_records if r.get("dataset") == "math"))
            math_verified = await generate_and_filter_set(
                "math", math_candidates, math_target, args.concurrency, tokenizer, out_file
            )
            all_verified.extend(math_verified)
            
    print(f"\nTotal verified training traces saved: {len(all_verified)}")
    generate_ablation_manifest(all_verified, manifest_path)

if __name__ == "__main__":
    asyncio.run(main())
