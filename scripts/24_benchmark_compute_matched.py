#!/usr/bin/env python3
"""
scripts/24_benchmark_compute_matched.py
Rigorous validation: Compute-Matched Discrete Baseline vs. Continuous Latent Recurrence.
Compares:
1. Base Model forced to generate EXACTLY <= 6 discrete thinking tokens (Compute-Matched to K=6)
2. Base Model forced to generate <= 16 discrete thinking tokens (Compute-Matched to K=16)
3. Base Model with No Thinking (0 thinking tokens)
4. Head-to-head comparison against Fixed K=6 Continuous Recurrent LoRA (64.0% accuracy).
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

# Disable aggressive warmup allocation
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

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

def extract_numeric_answer(text, is_answer_only=False):
    if not text:
        return None
    boxed_val = extract_math_boxed_expression(text)
    if boxed_val:
        val = clean_and_extract_candidate(boxed_val)
        if val is not None:
            return val
    m = re.findall(r'####\s*([\$]?[-\d.,]+)', text)
    if m:
        val = clean_and_extract_candidate(m[-1])
        if val is not None:
            return val
    ans_match = re.findall(r'(?:the answer is|final answer is|total is|equals|equal to)\s*([\$]?[-\d.,]+)', text, re.IGNORECASE)
    if ans_match:
        val = clean_and_extract_candidate(ans_match[-1])
        if val is not None:
            return val
    if is_answer_only:
        nums = re.findall(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?', text)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                pass
    return None

def math_equal(pred_str, gt_str):
    if not pred_str or not gt_str:
        return False
    pred_c = clean_and_extract_candidate(pred_str)
    gt_c = clean_and_extract_candidate(gt_str)
    if pred_c is not None and gt_c is not None and abs(pred_c - gt_c) < 1e-4:
        return True
    return False

def check_match(pred_num, pred_text, gt_num, gt_raw, is_math=False):
    if pred_num is not None and gt_num is not None:
        if abs(pred_num - gt_num) < 1e-4:
            return True
    if gt_raw:
        pred_box = extract_math_boxed_expression(pred_text)
        if pred_box and math_equal(pred_box, gt_raw):
            return True
        if math_equal(pred_text, gt_raw):
            return True
    return False

def get_benchmark_items():
    from datasets import load_dataset
    ds_gsm = load_dataset("openai/gsm8k", "main", split="test")
    easy, med, hard = [], [], []
    for i, item in enumerate(ds_gsm):
        ops = re.findall(r"<<.*?>>", item["answer"])
        op_count = len(ops)
        if op_count <= 2 and len(easy) < 5:
            easy.append({"id": f"gsm8k_easy_{i}", "category": "Easy", "question": item["question"], "answer": item["answer"]})
        elif 3 <= op_count <= 4 and len(med) < 5:
            med.append({"id": f"gsm8k_med_{i}", "category": "Medium", "question": item["question"], "answer": item["answer"]})
        elif op_count >= 5 and len(hard) < 5:
            hard.append({"id": f"gsm8k_hard_{i}", "category": "Hard", "question": item["question"], "answer": item["answer"]})
        if len(easy) >= 5 and len(med) >= 5 and len(hard) >= 5:
            break
            
    ds_math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    hard_math = []
    for i, item in enumerate(ds_math):
        if item["level"] in [4, 5]:
            hard_math.append({
                "id": f"math500_lvl{item['level']}_{i}",
                "category": f"MATH-L{item['level']}",
                "question": item["problem"],
                "answer": item["answer"]
            })
            if len(hard_math) >= 10:
                break
    return easy + med + hard + hard_math

def run_compute_matched_trial(model, tokenizer, question, device, max_think_tokens=6, max_new_tokens=2048):
    """
    Two-stage generation:
    1. Generate strictly up to `max_think_tokens` inside <think>.
    2. Force </think>\n\n token injection.
    3. Generate the answer up to max_new_tokens.
    Strictly matches the forward pass compute of K = max_think_tokens.
    """
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    torch.cuda.synchronize(device)
    t0 = time.time()
    
    think_text = ""
    actual_think_tokens = 0
    
    if max_think_tokens > 0:
        # Phase 1: Generate up to max_think_tokens
        with torch.no_grad():
            think_out = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_think_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        new_ids = think_out[0][inputs.input_ids.shape[1]:].tolist()
        actual_think_tokens = len(new_ids)
        think_text = tokenizer.decode(new_ids, skip_special_tokens=False)
        
    torch.cuda.synchronize(device)
    t_think_end = time.time()
    think_time_ms = (t_think_end - t0) * 1000.0
    
    # Phase 2: Inject </think>\n\n and decode final answer
    if "</think>" not in think_text:
        forced_injection = f"{prompt}{think_text}\n</think>\n\n"
    else:
        forced_injection = f"{prompt}{think_text}"
        
    ans_inputs = tokenizer(forced_injection, return_tensors="pt").to(device)
    with torch.no_grad():
        ans_out = model.generate(
            input_ids=ans_inputs.input_ids,
            attention_mask=ans_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    torch.cuda.synchronize(device)
    t_final = time.time()
    total_time_s = t_final - t0
    ans_time_s = max(1e-5, t_final - t_think_end)
    
    ans_ids = ans_out[0][ans_inputs.input_ids.shape[1]:].tolist()
    ans_tokens = len(ans_ids)
    raw_answer_text = tokenizer.decode(ans_ids, skip_special_tokens=False)
    
    pred_num = extract_numeric_answer(raw_answer_text, is_answer_only=True)
    dec_tok_s = ans_tokens / ans_time_s
    e2e_tok_s = ans_tokens / total_time_s
    
    return {
        "think_time_ms": round(think_time_ms, 2),
        "answer_time_s": round(ans_time_s, 3),
        "total_e2e_time_s": round(total_time_s, 3),
        "think_tokens": actual_think_tokens,
        "answer_tokens": ans_tokens,
        "decoding_tok_s": round(dec_tok_s, 2),
        "e2e_tok_s": round(e2e_tok_s, 2),
        "predicted_num": pred_num,
        "raw_text": raw_answer_text,
        "think_text": think_text
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--output", type=str, default="data/eval_results_compute_matched_1.5b.json")
    args = parser.parse_args()
    
    print("=" * 90)
    print(f"EXPERIMENT: COMPUTE-MATCHED DISCRETE BASELINES VS RECURRENCE ({args.model}) on {args.device}")
    print("=" * 90)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading Base Model in BF16 on {args.device}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    print("Model loaded successfully.")
    
    items = get_benchmark_items()
    print(f"Loaded {len(items)} benchmark problems.")
    
    conditions = [
        ("no_think_0tok", "Base (0 Thinking Tokens - Pure Direct Answer)", 0),
        ("matched_6tok", "Compute-Matched Base (<= 6 Thinking Tokens, Exactly Matched to K=6 Loops)", 6),
        ("matched_16tok", "Compute-Matched Base (<= 16 Thinking Tokens, Exactly Matched to K=16 Loops)", 16),
    ]
    
    results = {}
    
    for cond_key, cond_name, max_think in conditions:
        print("\n" + "=" * 90)
        print(f"RUNNING CONDITION: {cond_name}")
        print("=" * 90)
        cond_runs = []
        correct_count = 0
        gsm_correct = 0
        math_correct = 0
        
        for idx, item in enumerate(items):
            is_math = "math500" in item["id"]
            gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
            
            res = run_compute_matched_trial(
                model, tokenizer, item["question"], args.device,
                max_think_tokens=max_think, max_new_tokens=2048
            )
            is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
            res["correct"] = is_correct
            res["id"] = item["id"]
            res["category"] = item["category"]
            res["ground_truth_num"] = gt_num
            cond_runs.append(res)
            
            if is_correct:
                correct_count += 1
                if is_math:
                    math_correct += 1
                else:
                    gsm_correct += 1
                    
            print(f"[{idx+1}/{len(items)}] [{item['category']}] -> Pred: {res['predicted_num']} | Correct: {is_correct} | Think: {res['think_time_ms']:.1f}ms ({res['think_tokens']} tok) | E2E: {res['total_e2e_time_s']:.2f}s | Dec: {res['decoding_tok_s']} tok/s")
            
        overall_acc = round(correct_count / len(items) * 100, 1)
        gsm_acc = round(gsm_correct / 15 * 100, 1)
        math_acc = round(math_correct / 10 * 100, 1)
        mean_think_ms = round(sum(r["think_time_ms"] for r in cond_runs) / len(cond_runs), 1)
        mean_e2e_s = round(sum(r["total_e2e_time_s"] for r in cond_runs) / len(cond_runs), 2)
        mean_dec_tok_s = round(sum(r["decoding_tok_s"] for r in cond_runs) / len(cond_runs), 1)
        
        results[cond_key] = {
            "name": cond_name,
            "max_think_tokens": max_think,
            "overall_accuracy": overall_acc,
            "gsm8k_accuracy": gsm_acc,
            "math500_accuracy": math_acc,
            "mean_think_time_ms": mean_think_ms,
            "mean_e2e_latency_s": mean_e2e_s,
            "mean_decoding_tok_s": mean_dec_tok_s,
            "runs": cond_runs
        }
        
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 100)
    print("COMPUTE-MATCHED BENCHMARK SUMMARY (1.5B BF16)")
    print("=" * 100)
    print(f"{'Condition':<50} | {'Overall':<7} | {'GSM8K':<7} | {'MATH500':<7} | {'Think(ms)':<10} | {'E2E(s)':<8}")
    print("-" * 100)
    for k, v in results.items():
        print(f"{v['name']:<50} | {v['overall_accuracy']:>6.1f}% | {v['gsm8k_accuracy']:>6.1f}% | {v['math500_accuracy']:>6.1f}% | {v['mean_think_time_ms']:>9.1f} | {v['mean_e2e_latency_s']:>7.2f}")
    print("=" * 100)

if __name__ == "__main__":
    main()
