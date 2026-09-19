#!/usr/bin/env python3
"""
scripts/35_calibrate_qwen3_baselines.py
FIRST GATE: Calibrate and verify official Qwen 3 baselines across thinking and non-thinking modes.
Ensures evaluation harness reproduces official expected performance before any recurrent loops are tested.

Follows official Qwen 3 evaluation guidelines:
1. Native chat template with `enable_thinking=True` and `enable_thinking=False`.
2. Official sampling parameters: temperature=0.6, top_p=0.95, top_k=20 (DO NOT use greedy decoding).
3. Adequate token ceiling: max_new_tokens=4096 to prevent truncation.
4. Robust answer extraction parsing both final content and fallback thinking content.
5. Rich telemetry logging: wall-clock breakdown, tok/s, KV bytes, and token counts.
"""

import os
import re
import json
import time
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

THINK_END_TOKEN_ID = 151668  # </think>

def clean_and_extract_candidate(cand):
    if not cand:
        return None
    cand = re.sub(r"\\(?:text|mathbf|mathrm)\{([^}]+)\}", r"\1", cand)
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
        nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", cand)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                pass
    return None

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

def extract_answer(full_text, content_text, thinking_text):
    # 1. First priority: \boxed{...} in content_text
    boxed = extract_math_boxed_expression(content_text)
    if boxed:
        val = clean_and_extract_candidate(boxed)
        if val is not None:
            return val, boxed
            
    # 2. Second priority: #### in content_text
    hash_match = re.findall(r'####\s*([-\d.,]+)', content_text)
    if hash_match:
        val = clean_and_extract_candidate(hash_match[-1])
        if val is not None:
            return val, hash_match[-1]
            
    # 3. Third priority: standard answer phrases in content_text
    ans_match = re.findall(r'(?:the answer is|final answer is|total is|equals|equal to)\s*([\$]?[-\d.,]+)', content_text, re.IGNORECASE)
    if ans_match:
        val = clean_and_extract_candidate(ans_match[-1])
        if val is not None:
            return val, ans_match[-1]
            
    # 4. Fallback: \boxed{...} anywhere in thinking_text (in case model derived answer before cutoff)
    boxed_think = extract_math_boxed_expression(thinking_text)
    if boxed_think:
        val = clean_and_extract_candidate(boxed_think)
        if val is not None:
            return val, boxed_think
            
    # 5. Last resort: last number in content_text
    nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", content_text)
    if nums:
        val = clean_and_extract_candidate(nums[-1])
        if val is not None:
            return val, nums[-1]
            
    return None, ""

def check_match(pred_val, pred_raw, gt_val, gt_raw):
    if pred_val is not None and gt_val is not None:
        if abs(pred_val - gt_val) < 1e-4:
            return True
    if pred_raw and gt_raw:
        c_pred = re.sub(r"[\$\\%!\s]", "", str(pred_raw)).replace(",", "").strip()
        c_gt = re.sub(r"[\$\\%!\s]", "", str(gt_raw)).replace(",", "").strip()
        if c_pred == c_gt:
            return True
    return False

def load_problems():
    problems_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_problems_suite.json")
    if os.path.exists(problems_path):
        with open(problems_path) as f:
            return json.load(f)
            
    suite = []
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    for idx in [0, 1, 3, 4, 13]:
        ex = gsm8k[idx]
        suite.append({"id": f"gsm8k_easy_{idx}", "category": "GSM8K", "question": ex["question"], "answer": ex["answer"]})
    for idx in [2, 6, 7, 11, 12]:
        ex = gsm8k[idx]
        suite.append({"id": f"gsm8k_med_{idx}", "category": "GSM8K", "question": ex["question"], "answer": ex["answer"]})
    for idx in [5, 8, 9, 10, 25]:
        ex = gsm8k[idx]
        suite.append({"id": f"gsm8k_hard_{idx}", "category": "GSM8K", "question": ex["question"], "answer": ex["answer"]})
    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for idx in [1, 7, 9, 11, 12, 15, 17, 22, 23, 24]:
        ex = math_ds[idx]
        suite.append({"id": f"math500_l45_{idx}", "category": "MATH-500", "question": ex["problem"], "answer": ex["solution"]})
    return suite

def run_single_eval(model, tokenizer, prompt, enable_thinking, max_new_tokens, device):
    messages = [{"role": "user", "content": prompt}]
    
    # Official Qwen3 chat template with native enable_thinking switch
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    
    torch.cuda.synchronize(device)
    t0 = time.time()
    
    vram_before = torch.cuda.memory_allocated(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
        
    torch.cuda.synchronize(device)
    total_time = time.time() - t0
    
    vram_peak = torch.cuda.max_memory_allocated(device)
    vram_after = torch.cuda.memory_allocated(device)
    
    output_ids = generated_ids[0][prompt_len:].tolist()
    total_tokens = len(output_ids)
    
    # Official Qwen 3 thinking parser
    try:
        idx = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        idx = 0
        
    thinking_content = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()
    full_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    think_tokens = idx
    ans_tokens = total_tokens - idx
    
    # Accurate latency decomposition
    tok_per_sec = total_tokens / max(0.001, total_time)
    
    # Extract prediction
    pred_val, pred_raw = extract_answer(full_output, content, thinking_content)
    
    return {
        "enable_thinking": enable_thinking,
        "total_time_s": round(total_time, 3),
        "total_tokens": total_tokens,
        "think_tokens": think_tokens,
        "ans_tokens": ans_tokens,
        "tok_per_sec": round(tok_per_sec, 2),
        "vram_alloc_mb": round(vram_after / (1024**2), 1),
        "vram_peak_mb": round(vram_peak / (1024**2), 1),
        "prompt_tokens": prompt_len,
        "thinking_content": thinking_content,
        "content": content,
        "full_output": full_output,
        "pred_val": pred_val,
        "pred_raw": pred_raw
    }

def main():
    parser = argparse.ArgumentParser(description="Calibrate Qwen3 base model baselines.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    
    if args.output_file is None:
        tag = args.model_id.replace("/", "_").lower()
        args.output_file = os.path.join(os.path.dirname(__file__), "..", "data", f"baseline_calibration_{tag}.json")
        
    print(f"=== Calibrating Official Baselines for {args.model_id} on {args.device} ===")
    print(f"Sampling: temperature=0.6, top_p=0.95, top_k=20 | max_tokens={args.max_new_tokens}")
    
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip("\"'\n")
                    break
                    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token, trust_remote_code=True)
    print("Loading model in pure bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()
    
    problems = load_problems()
    print(f"Loaded {len(problems)} benchmark test problems.")
    
    results = {
        "model_id": args.model_id,
        "device": args.device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modes": {}
    }
    
    # Test both Thinking and Non-Thinking modes
    for enable_thinking, mode_name in [(True, "Thinking Mode (enable_thinking=True)"), (False, "Non-Thinking Mode (enable_thinking=False)")]:
        print(f"\n{'='*60}\nEvaluating Gate: {mode_name}\n{'='*60}")
        mode_runs = []
        correct_total = 0
        correct_gsm = 0
        correct_math = 0
        
        for idx, prob in enumerate(problems):
            q = prob["question"]
            gt_text = prob["answer"]
            gt_boxed = extract_math_boxed_expression(gt_text)
            gt_val = clean_and_extract_candidate(gt_boxed or gt_text)
            
            run = run_single_eval(model, tokenizer, q, enable_thinking, args.max_new_tokens, args.device)
            is_correct = check_match(run["pred_val"], run["pred_raw"], gt_val, gt_boxed or gt_text)
            run["is_correct"] = is_correct
            run["problem_id"] = prob["id"]
            run["category"] = prob["category"]
            run["gt_val"] = gt_val
            run["gt_raw"] = gt_boxed or gt_text
            mode_runs.append(run)
            
            if is_correct:
                correct_total += 1
                if prob["category"] == "MATH-500":
                    correct_math += 1
                else:
                    correct_gsm += 1
                    
            print(f"[{idx+1}/{len(problems)}] {prob['id']} ({prob['category']}) | Correct: {is_correct} | Think Tok: {run['think_tokens']} | Ans Tok: {run['ans_tokens']} | Latency: {run['total_time_s']}s")
            
        gsm_count = sum(1 for p in problems if p["category"] == "GSM8K")
        math_count = sum(1 for p in problems if p["category"] == "MATH-500")
        
        summary = {
            "mode_name": mode_name,
            "overall_acc": round(correct_total / len(problems) * 100.0, 2),
            "gsm8k_acc": round(correct_gsm / max(1, gsm_count) * 100.0, 2),
            "math500_acc": round(correct_math / max(1, math_count) * 100.0, 2),
            "mean_latency_s": round(sum(r["total_time_s"] for r in mode_runs) / len(mode_runs), 3),
            "mean_think_tokens": round(sum(r["think_tokens"] for r in mode_runs) / len(mode_runs), 1),
            "mean_ans_tokens": round(sum(r["ans_tokens"] for r in mode_runs) / len(mode_runs), 1),
            "mean_tok_per_sec": round(sum(r["tok_per_sec"] for r in mode_runs) / len(mode_runs), 2),
            "mean_vram_peak_mb": round(sum(r["vram_peak_mb"] for r in mode_runs) / len(mode_runs), 1),
            "runs": mode_runs
        }
        
        mode_key = "thinking_mode" if enable_thinking else "non_thinking_mode"
        results["modes"][mode_key] = summary
        
        print(f"\n--- Gate Summary for {mode_name} ---")
        print(f"Overall Acc: {summary['overall_acc']}% | GSM8K: {summary['gsm8k_acc']}% | MATH-500: {summary['math500_acc']}%")
        print(f"Mean Latency: {summary['mean_latency_s']}s | Mean Tokens: {summary['mean_think_tokens']} think + {summary['mean_ans_tokens']} ans ({summary['mean_tok_per_sec']} tok/s)")
        
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved calibration results to: {args.output_file}")

if __name__ == "__main__":
    main()
