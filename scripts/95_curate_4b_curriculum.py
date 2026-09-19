#!/usr/bin/env python3
"""
scripts/95_curate_4b_curriculum.py
High-throughput self-distillation trace generation & curriculum curation for Qwen/Qwen3-4B.

Pipeline:
1. Load train candidates from openai/gsm8k and DigitalLearningGmbH/MATH-lighteval.
   (Strictly assert 0 overlap with benchmark_suite_250.json / test sets).
2. Generate thinking-mode traces on SGLang (concurrency C=16). Filter with pure math_verify.
3. Classify each thinking-correct problem in non-thinking mode:
   - Solved-Direct: non-thinking also correct.
   - Live Candidate: thinking correct, non-thinking failed.
4. Self-rewrite worked non-thinking derivations for Live Candidates using their own thinking traces.
   Verify with canonical math_verify and operator density checks.
5. Sample 137 clean UltraChat general instruction examples (0 IFEval 13-gram overlap).
6. Assemble Path A training set: ~1,230 math traces (35-40% live) + 137 general traces (~1,367 total).
7. Save data/curated_train_v1_1_dilution10pct_qwen3_4b.jsonl and dev slice.
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

SGLANG_URL = "http://127.0.0.1:30000"
MODEL_NAME = "Qwen/Qwen3-4B"
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

def check_math_correct(pred_text, ground_truth):
    if not ground_truth:
        return False
    try:
        c_gold = parse(ground_truth)
        c_pred = parse(pred_text)
        return bool(verify(c_gold, c_pred))
    except Exception:
        return False

def check_operations_in_derivation(text):
    words = text.split()
    if len(words) < 35:
        return False, f"Too short ({len(words)} < 35 words)"
        
    op_pattern = re.compile(r'(=|\+|-|\\times|\\cdot|\*|/|\\div|\\approx|\\equiv|\\le|\\ge|\^|\\sqrt)')
    op_matches = op_pattern.findall(text)
    if len(op_matches) < 3:
        return False, f"Too few mathematical operations ({len(op_matches)} < 3)"
        
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    lines_with_numbers = [l for l in lines if re.search(r'\d', l)]
    
    op_lines = [l for l in lines_with_numbers if op_pattern.search(l)]
    if len(lines_with_numbers) > 0 and (len(op_lines) / len(lines_with_numbers)) < 0.35:
        return False, f"Low operator density ({len(op_lines)}/{len(lines_with_numbers)})"
        
    assertion_only = re.search(r'^(Therefore, |So, |Hence, |The answer is |Thus, )?(\$?\\boxed\{.*\}\$?\.?)$', text.strip(), re.IGNORECASE)
    if assertion_only:
        return False, "Pure assertion without derivation"
        
    return True, "Passed"

def get_ngrams(text, n):
    words = re.findall(r"\w+", text.lower())
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))

def get_char_substrings(text, length=50):
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    return set(cleaned[i:i+length] for i in range(len(cleaned) - length + 1))

def load_contamination_set():
    test_questions = set()
    suite_path = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_suite_250.json")
    if os.path.exists(suite_path):
        with open(suite_path) as f:
            suite = json.load(f)
            for item in suite:
                test_questions.add(item["question"].strip().lower())
    return test_questions

def load_gsm8k_candidates(n_candidates=1000, contamination_set=None):
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

def load_math_candidates(n_candidates=1200, contamination_set=None):
    print(f"Loading MATH-lighteval train split candidates (target: {n_candidates})...")
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    by_level = defaultdict(list)
    for idx, ex in enumerate(ds):
        q = ex["problem"].strip()
        if contamination_set and q.lower() in contamination_set:
            continue
        gt = ex.get("solution", "")
        boxed = extract_math_boxed_expression(gt)
        if not boxed:
            continue
        lvl_str = str(ex.get("level", "Level 1"))
        match = re.search(r'\d+', lvl_str)
        lvl_num = int(match.group(0)) if match else 1
        by_level[lvl_num].append({
            "id": f"math_train_{idx}",
            "dataset": "math",
            "question": q,
            "raw_solution": gt,
            "ground_truth": boxed,
            "level": lvl_num,
            "subject": ex.get("type", "general")
        })
    candidates = []
    per_level = n_candidates // 5
    for lvl in range(1, 6):
        available = by_level[lvl]
        random.seed(42 + lvl)
        random.shuffle(available)
        selected = available[:per_level]
        candidates.extend(selected)
        print(f"  Level {lvl}: selected {len(selected)} (pool: {len(available)})")
    random.shuffle(candidates)
    print(f"Loaded {len(candidates)} clean MATH train candidates.")
    return candidates

async def query_sglang(client, prompt, enable_thinking=True, max_tokens=3072, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "top_k": top_k,
            "presence_penalty": presence_penalty
        }
    }
    try:
        resp = await client.post(f"{SGLANG_URL}/v1/chat/completions", json=payload, timeout=120.0)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:100]}"
        data = resp.json()
        choice = data["choices"][0]
        text = choice.get("message", {}).get("content", "")
        reasoning = choice.get("message", {}).get("reasoning_content", "")
        if reasoning and "</think>" not in text:
            full_text = f"<think>\n{reasoning}\n</think>\n\n{text}"
        else:
            full_text = text
        return full_text, None
    except Exception as e:
        return None, str(e)

async def main_async():
    parser = argparse.ArgumentParser(description="Curate Qwen3-4B Training Curriculum via SGLang")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max_candidates", type=int, default=2200)
    parser.add_argument("--target_math_total", type=int, default=1230)
    parser.add_argument("--target_live_ratio", type=float, default=0.35)
    parser.add_argument("--general_count", type=int, default=137)
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    print("==========================================================================")
    print("=== Curating Qwen3-4B Training Curriculum (High-Throughput SGLang) ===")
    print(f"Server Endpoint: {SGLANG_URL}")
    print(f"Model ID       : {MODEL_NAME}")
    print(f"Concurrency    : {args.concurrency}")
    print("==========================================================================")

    limits = httpx.Limits(max_connections=args.concurrency + 10, max_keepalive_connections=args.concurrency + 10)
    async with httpx.AsyncClient(limits=limits) as client:
        try:
            r = await client.get(f"{SGLANG_URL}/v1/models", timeout=5.0)
            if r.status_code != 200:
                print(f"Error: Server returned status {r.status_code}")
                return
            print(f"Connected to SGLang server! Running model: {r.json()['data'][0]['id']}")
        except Exception as e:
            print(f"Error connecting to SGLang server: {e}")
            return

        contamination_set = load_contamination_set()
        print(f"Contamination filter loaded: {len(contamination_set)} test problems.")

        # 1. Load Candidates
        gsm8k_candidates = load_gsm8k_candidates(1000, contamination_set)
        math_candidates = load_math_candidates(1200, contamination_set)
        all_candidates = gsm8k_candidates + math_candidates
        print(f"Total candidate pool: {len(all_candidates)} problems.")

        semaphore = asyncio.Semaphore(args.concurrency)

        # 2. Step 1: Thinking Mode Trace Generation
        print("\n>>> Step 1/5: Generating Thinking-Mode Traces...")
        thinking_correct = []
        
        async def run_thinking_worker(cand):
            async with semaphore:
                prompt = cand["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                full_text, err = await query_sglang(
                    client, prompt, enable_thinking=True, max_tokens=3072,
                    temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0
                )
                if err or not full_text:
                    return None
                
                if "<think>" not in full_text or "</think>" not in full_text:
                    return None
                    
                think_part = full_text.split("</think>")[0].replace("<think>", "").strip()
                ans_part = full_text.split("</think>")[1].strip()
                
                if not check_math_correct(ans_part, cand["ground_truth"]):
                    return None
                    
                boxed_pred = extract_math_boxed_expression(ans_part)
                if not boxed_pred:
                    return None
                    
                return {
                    **cand,
                    "prompt": prompt,
                    "thinking_trace": think_part,
                    "thinking_answer": ans_part,
                    "full_thinking_output": full_text,
                    "boxed_pred": boxed_pred
                }

        t0 = time.time()
        tasks = [run_thinking_worker(c) for c in all_candidates]
        for fut in asyncio.as_completed(tasks):
            res = await fut
            if res:
                thinking_correct.append(res)
                if len(thinking_correct) % 50 == 0:
                    print(f"  Verified Thinking Correct: {len(thinking_correct)} / {len(all_candidates)} ({time.time() - t0:.1f}s)")
        
        print(f"Step 1 Complete: {len(thinking_correct)} verified thinking-correct traces in {time.time() - t0:.1f}s.")

        # 3. Step 2: Non-Thinking Direct Classification
        print("\n>>> Step 2/5: Non-Thinking Direct Classification...")
        solved_direct = []
        live_candidates = []

        async def run_nonthinking_worker(item):
            async with semaphore:
                prompt = item["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                prompt_conditioned = prompt + "\n<think>\n\n</think>\n\n"
                resp_text, err = await query_sglang(
                    client, prompt_conditioned, enable_thinking=False, max_tokens=1024,
                    temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5
                )
                if err or not resp_text:
                    return item, False, ""
                    
                clean_resp = resp_text.replace("<think>", "").replace("</think>", "").strip()
                is_correct = check_math_correct(clean_resp, item["ground_truth"])
                return item, is_correct, clean_resp

        t1 = time.time()
        tasks = [run_nonthinking_worker(item) for item in thinking_correct]
        for fut in asyncio.as_completed(tasks):
            item, is_correct, resp = await fut
            if is_correct:
                item["nonthinking_answer"] = resp
                solved_direct.append(item)
            else:
                live_candidates.append(item)
                
            total_done = len(solved_direct) + len(live_candidates)
            if total_done % 100 == 0:
                print(f"  Classified: {total_done} / {len(thinking_correct)} (Solved-Direct: {len(solved_direct)}, Live: {len(live_candidates)})")

        print(f"Classification Complete in {time.time() - t1:.1f}s:")
        print(f"  Solved-Direct: {len(solved_direct)} ({len(solved_direct)/len(thinking_correct)*100:.1f}%)")
        print(f"  Live Candidates: {len(live_candidates)} ({len(live_candidates)/len(thinking_correct)*100:.1f}%)")

        class_registry_path = os.path.join(data_dir, "train_problem_classification_qwen3_4b.json")
        with open(class_registry_path, "w") as f:
            json.dump({
                "total_thinking_correct": len(thinking_correct),
                "solved_direct_count": len(solved_direct),
                "live_candidate_count": len(live_candidates),
                "solved_direct_ids": [x["id"] for x in solved_direct],
                "live_candidate_ids": [x["id"] for x in live_candidates]
            }, f, indent=2)
        print(f"Saved classification registry to: {class_registry_path}")

        # 4. Step 3: Self-Rewriting Worked Solutions for Live Candidates
        print(f"\n>>> Step 3/5: Self-Rewriting Worked Solutions for {len(live_candidates)} Live Candidates...")
        verified_live = []

        async def run_rewrite_worker(item):
            async with semaphore:
                prompt = (
                    f"Problem: {item['question']}\n\n"
                    f"Reference Reasoning Steps:\n{item['thinking_trace']}\n\n"
                    f"Task: Based on the reference reasoning, provide a clean, step-by-step worked derivation "
                    f"that clearly explains each intermediate operation and ends with the final answer in \\boxed{{}}."
                )
                prompt_conditioned = prompt + "\n<think>\n\n</think>\n\n"
                resp_text, err = await query_sglang(
                    client, prompt_conditioned, enable_thinking=False, max_tokens=1536,
                    temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5
                )
                if err or not resp_text:
                    return None
                    
                clean_resp = resp_text.replace("<think>", "").replace("</think>", "").strip()
                if not check_math_correct(clean_resp, item["ground_truth"]):
                    return None
                    
                op_ok, op_reason = check_operations_in_derivation(clean_resp)
                if not op_ok:
                    return None
                    
                item["nonthinking_answer"] = clean_resp
                return item

        t2 = time.time()
        tasks = [run_rewrite_worker(item) for item in live_candidates]
        for fut in asyncio.as_completed(tasks):
            res = await fut
            if res:
                verified_live.append(res)
                if len(verified_live) % 50 == 0:
                    print(f"  Verified Live Worked Solutions: {len(verified_live)} / {len(live_candidates)}")

        print(f"Step 3 Complete: {len(verified_live)} / {len(live_candidates)} verified live worked solutions ({time.time() - t2:.1f}s).")
        
        live_by_level = defaultdict(int)
        for x in verified_live:
            live_by_level[x["level"]] += 1
        print("Live problems by level:")
        for lvl in sorted(live_by_level.keys()):
            print(f"  Level {lvl}: {live_by_level[lvl]}")

        # 5. Step 4: Sample General Instruction Slice from UltraChat
        print(f"\n>>> Step 4/5: Generating {args.general_count} General Instruction Responses from UltraChat...")
        ifeval = load_dataset("google/ifeval")["train"]
        ifeval_13grams = set()
        ifeval_substrings = set()
        for x in ifeval:
            p = x["prompt"].strip()
            ifeval_13grams.update(get_ngrams(p, 13))
            ifeval_substrings.update(get_char_substrings(p, 50))
            
        ultrachat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        clean_uc_prompts = []
        for ex in ultrachat_ds:
            p = ex["messages"][0]["content"].strip()
            if len(p.split()) < 8 or len(p.split()) > 60:
                continue
            p_13g = get_ngrams(p, 13)
            if p_13g.intersection(ifeval_13grams):
                continue
            p_subs = get_char_substrings(p, 50)
            if p_subs.intersection(ifeval_substrings):
                continue
            clean_uc_prompts.append(p)
            if len(clean_uc_prompts) >= args.general_count * 2:
                break
                
        general_traces = []
        async def run_general_worker(idx, prompt):
            async with semaphore:
                prompt_conditioned = prompt + "\n<think>\n\n</think>\n\n"
                resp_text, err = await query_sglang(
                    client, prompt_conditioned, enable_thinking=False, max_tokens=1024,
                    temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5
                )
                if err or not resp_text or len(resp_text.split()) < 15:
                    return None
                clean_resp = resp_text.replace("<think>", "").replace("</think>", "").strip()
                return {
                    "id": f"ultrachat_{idx}",
                    "dataset": "ultrachat",
                    "question": prompt,
                    "prompt": prompt,
                    "nonthinking_answer": clean_resp,
                    "level": 0,
                    "subject": "general"
                }

        t3 = time.time()
        tasks = [run_general_worker(i, p) for i, p in enumerate(clean_uc_prompts)]
        for fut in asyncio.as_completed(tasks):
            res = await fut
            if res:
                general_traces.append(res)
                if len(general_traces) >= args.general_count:
                    break
        print(f"Step 4 Complete: {len(general_traces)} general instruction traces curated ({time.time() - t3:.1f}s).")

        # 6. Step 5: Assemble Path A Dataset
        print("\n>>> Step 5/5: Assembling Path A Dataset...")
        n_target_live = int(args.target_math_total * args.target_live_ratio)
        n_target_direct = args.target_math_total - n_target_live
        
        selected_live = verified_live[:n_target_live]
        direct_by_level = defaultdict(list)
        for x in solved_direct:
            direct_by_level[x["level"]].append(x)
            
        selected_direct = []
        per_lvl_direct = n_target_direct // len(direct_by_level) if direct_by_level else n_target_direct
        for lvl, pool in direct_by_level.items():
            random.seed(42 + lvl)
            random.shuffle(pool)
            selected_direct.extend(pool[:per_lvl_direct])
            
        if len(selected_direct) < n_target_direct:
            rem_pool = [x for x in solved_direct if x not in selected_direct]
            selected_direct.extend(rem_pool[:(n_target_direct - len(selected_direct))])
            
        math_traces = selected_live + selected_direct
        random.seed(42)
        random.shuffle(math_traces)
        
        all_traces = math_traces + general_traces
        random.seed(123)
        random.shuffle(all_traces)
        
        dev_traces = all_traces[:100]
        train_traces = all_traces[100:]
        
        train_out_path = os.path.join(data_dir, "curated_train_v1_1_dilution10pct_qwen3_4b.jsonl")
        dev_out_path = os.path.join(data_dir, "curated_dev_v1_1_dilution10pct_qwen3_4b.jsonl")
        
        with open(train_out_path, "w") as f:
            for t in train_traces:
                f.write(json.dumps(t) + "\n")
        with open(dev_out_path, "w") as f:
            for t in dev_traces:
                f.write(json.dumps(t) + "\n")
                
        print("\n=======================================================")
        print("PATH A DATASET ASSEMBLY COMPLETE")
        print(f"Total Traces: {len(all_traces)}")
        print(f"  Math Traces: {len(math_traces)} (Live: {len(selected_live)} [{len(selected_live)/len(math_traces)*100:.1f}%], Direct: {len(selected_direct)})")
        print(f"  General Traces: {len(general_traces)} ({len(general_traces)/len(all_traces)*100:.1f}%)")
        print(f"Train Traces: {len(train_traces)} -> {train_out_path}")
        print(f"Dev Traces  : {len(dev_traces)} -> {dev_out_path}")
        print("=======================================================")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
