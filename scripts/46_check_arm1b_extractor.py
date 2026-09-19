#!/usr/bin/env python3
"""
scripts/46_check_arm1b_extractor.py
Eyeball verification tool for Arm 1b (Trained Direct) outputs and answer extractor.
Runs Arm 1b on 20 stratified problems from the benchmark suite to inspect:
1. Raw generated text format (e.g. '#### 72', '\\boxed{}', 'The final answer is: 72').
2. Extracted candidate value.
3. math_verify / symbolic grader match.
Guarantees the extractor catches human GSM8K/MATH answer formats without penalty.
"""

import os
import re
import json
import time
import argparse
import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from math_verify import parse, verify

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

def extract_answer(full_text, content_text):
    text = content_text if (content_text and content_text.strip()) else full_text
    
    # 1. First priority: \boxed{...} in text
    boxed = extract_math_boxed_expression(text)
    if boxed:
        val = clean_and_extract_candidate(boxed)
        if val is not None:
            return val, boxed
            
    # 2. Second priority: #### in text (GSM8K human format)
    hash_match = re.findall(r'####\s*([\$]?[-+]?[\d,]+(?:\.\d+)?)', text)
    if hash_match:
        val = clean_and_extract_candidate(hash_match[-1])
        if val is not None:
            return val, hash_match[-1]
            
    # 3. Third priority: standard answer phrases (with/without colon: "final answer is: 72", "answer: 72", etc.)
    ans_match = re.findall(
        r'(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|:)\s*([\$]?[-+]?[\d,]+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    if ans_match:
        val = clean_and_extract_candidate(ans_match[-1])
        if val is not None:
            return val, ans_match[-1]
            
    # 4. Fourth priority: bolded answer (e.g. **72**)
    bold_match = re.findall(r'\*\*([\$]?[-+]?[\d,]+(?:\.\d+)?)\*\*', text)
    if bold_match:
        val = clean_and_extract_candidate(bold_match[-1])
        if val is not None:
            return val, bold_match[-1]

    # 5. Fallback: last number in text
    nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", text)
    if nums:
        val = clean_and_extract_candidate(nums[-1])
        if val is not None:
            return val, nums[-1]
            
    return None, ""

def check_match(pred_val, pred_raw, gt_val, gt_raw, full_pred_text=None, full_gold_text=None):
    # 1. Primary: math_verify symbolic parser
    if full_pred_text and full_gold_text:
        try:
            gold_p = parse(full_gold_text, parsing_timeout=None)
            pred_p = parse(full_pred_text, parsing_timeout=None)
            if gold_p and pred_p and verify(gold_p, pred_p):
                return True
        except Exception:
            pass

    if pred_raw and gt_raw:
        try:
            gold_p = parse(str(gt_raw), parsing_timeout=None)
            pred_p = parse(str(pred_raw), parsing_timeout=None)
            if gold_p and pred_p and verify(gold_p, pred_p):
                return True
        except Exception:
            pass

    # 2. Secondary: numeric equivalence
    if pred_val is not None and gt_val is not None:
        if abs(pred_val - gt_val) < 1e-4:
            return True

    # 3. Tertiary: string normalization fallback
    if pred_raw and gt_raw:
        c_pred = re.sub(r"[\$\\%!\s]", "", str(pred_raw)).replace(",", "").strip().lower()
        c_gt = re.sub(r"[\$\\%!\s]", "", str(gt_raw)).replace(",", "").strip().lower()
        if c_pred == c_gt:
            return True
    return False

def load_sample_problems(num_problems=20):
    cache_path = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_suite_250.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            suite = json.load(f)
        return suite[:num_problems]
        
    print("Loading 20 sample problems from GSM8K and MATH-500...")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i in range(15):
        problems.append({
            "id": f"gsm8k_{i}",
            "benchmark": "GSM8K",
            "question": gsm[i]["question"],
            "solution": gsm[i]["answer"]
        })
    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for i in range(5):
        problems.append({
            "id": f"math500_{i}",
            "benchmark": "MATH-500",
            "question": math_ds[i]["problem"],
            "solution": math_ds[i]["solution"]
        })
    return problems

def main():
    parser = argparse.ArgumentParser(description="Eyeball check for Arm 1b extractor.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora_path", type=str, default="checkpoints/lora_arm1b_qwen_qwen3-1.7b_k0")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num_problems", type=int, default=20)
    args = parser.parse_args()

    print(f"=== Eyeball Diagnostic for Arm 1b Extractor ===")
    print(f"Model: {args.model_id} | LoRA: {args.lora_path} | Device: {args.device}")
    
    problems = load_sample_problems(args.num_problems)
    print(f"Loaded {len(problems)} diagnostic problems.")

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip("\"'\n")
                    break

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token, trust_remote_code=True)
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    base_model.eval()

    print(f"Attaching Arm 1b LoRA adapter from {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()

    print("\n" + "="*80)
    print(f"INSPECTING GENERATIONS & EXTRACTION (Total: {len(problems)})")
    print("="*80)

    stats = {"correct": 0, "extraction_misses": 0, "wrong": 0}

    for idx, prob in enumerate(problems):
        q = prob["question"]
        sol = prob["solution"]
        gt_boxed = extract_math_boxed_expression(sol)
        gt_val = clean_and_extract_candidate(gt_boxed or sol)

        messages = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer([text], return_tensors="pt").to(args.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_tokens = out[0][prompt_len:].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        pred_val, pred_raw = extract_answer(gen_text, gen_text)
        is_match = check_match(pred_val, pred_raw, gt_val, gt_boxed or sol, full_pred_text=gen_text, full_gold_text=sol)

        # Check if gold answer appears in the text even if is_match is False (extraction miss detector)
        gold_str = str(int(gt_val)) if (gt_val is not None and gt_val.is_integer()) else str(gt_val)
        potential_miss = (not is_match) and (gold_str in gen_text)

        status_tag = "MATCH (CORRECT)" if is_match else ("POTENTIAL EXTRACTION MISS" if potential_miss else "WRONG (MODEL ERROR)")
        if is_match:
            stats["correct"] += 1
        elif potential_miss:
            stats["extraction_misses"] += 1
        else:
            stats["wrong"] += 1

        print(f"\n--- [Case {idx+1}/{len(problems)}] {prob['id']} ({prob['benchmark']}) : {status_tag} ---")
        print(f"Q: {q[:120]}...")
        print(f"Gold: {sol.splitlines()[-1] if '####' in sol else (gt_boxed or sol)[:80]}")
        print(f"Model Output:\n{gen_text}")
        print(f"Extracted: pred_val={pred_val}, pred_raw={pred_raw!r} | GT: gt_val={gt_val}, gt_boxed={gt_boxed!r}")
        print(f"Check Match: {is_match}")

    print("\n" + "="*80)
    print("EYEBALL AUDIT SUMMARY:")
    print(f"Correct:           {stats['correct']} / {len(problems)} ({stats['correct']/len(problems)*100:.1f}%)")
    print(f"Extraction Misses: {stats['extraction_misses']} / {len(problems)}")
    print(f"Wrong Derivations: {stats['wrong']} / {len(problems)}")
    print("="*80)

if __name__ == "__main__":
    main()
