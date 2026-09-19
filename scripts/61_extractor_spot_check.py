#!/usr/bin/env python3
"""
scripts/61_extractor_spot_check.py
Extractor Spot-Check: 20 Arm 1b outputs inspected by hand against the grader.
Guards against scoring trained arms by a rule they do not emit due to format changes
in self-generated curriculum training.
"""

import os
import re
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from math_verify import parse, verify

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
    cand = str(cand)
    if "####" in cand:
        cand = cand.split("####")[-1].strip()
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
    if full_pred_text and (gold_boxed or full_gold_text):
        try:
            gold_parsed = parse(f"\\boxed{{{gold_boxed}}}") if gold_boxed else parse(full_gold_text)
            pred_parsed = parse(full_pred_text)
            if gold_parsed and pred_parsed and verify(gold_parsed, pred_parsed):
                return True
        except Exception:
            pass
    if pred_val is not None and gold_val is not None:
        if isinstance(pred_val, (int, float)) and isinstance(gold_val, (int, float)):
            if abs(pred_val - gold_val) < 1e-4:
                return True
        elif str(pred_val).strip().lower() == str(gold_val).strip().lower():
            return True
    return False

def spot_check(adapter_path, model_id="Qwen/Qwen3-1.7B", suite_file="data/benchmark_suite_250.json", num_samples=20, device="cuda:0"):
    print(f"=== Extractor Spot-Check: Inspecting {num_samples} Arm 1b Outputs ===")
    print(f"Model: {model_id} | Adapter: {adapter_path} | Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"Successfully loaded LoRA adapter from {adapter_path}")
    else:
        print(f"WARNING: Adapter path {adapter_path} not found. Running base model for dry-run check.")
        model = base_model
    model.eval()

    with open(suite_file) as f:
        problems = json.load(f)

    # Pick 10 GSM8K and 10 MATH problems
    gsm = [p for p in problems if "gsm" in p.get("benchmark", "").lower()][:num_samples//2]
    math = [p for p in problems if "math" in p.get("benchmark", "").lower()][:num_samples//2]
    sample_pool = gsm + math

    results = []
    for idx, prob in enumerate(sample_pool):
        q = prob["question"]
        sol = prob["solution"]
        gold_boxed = extract_math_boxed_expression(sol)
        gold_val = clean_and_extract_candidate(gold_boxed or sol)

        prompt_str = f"<|im_start|>user\n{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n\n</think>\n\n"
        inputs = tokenizer([prompt_str], return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.encode("<|im_end|>")
            )
        pred_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred_val, pred_raw = extract_answer(pred_text)
        is_correct = check_match(pred_val, pred_raw, gold_val, gold_boxed, pred_text, sol)

        results.append({
            "idx": idx + 1,
            "id": prob["id"],
            "benchmark": prob["benchmark"],
            "pred_val": pred_val,
            "gold_val": gold_val,
            "gold_boxed": gold_boxed,
            "is_correct": is_correct,
            "pred_snippet": pred_text[-200:].replace("\n", " ")
        })

        print(f"[{idx+1:2d}/{num_samples}] ID: {prob['id']} ({prob['benchmark']})")
        print(f"   Gold Value: {gold_val} (Boxed: '{gold_boxed}')")
        print(f"   Pred Value: {pred_val} (Raw: '{pred_raw}')")
        print(f"   Match: {'CORRECT [OK]' if is_correct else 'INCORRECT'}")
        print(f"   Output Tail: ...{pred_text[-120:]}\n")

    correct_count = sum(1 for r in results if r["is_correct"])
    print("="*60)
    print(f"Spot Check Complete: {correct_count}/{num_samples} ({correct_count/num_samples*100:.1f}%) verified.")
    print("Inspect output tails above to confirm \\boxed{} extraction matches model emit style.")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default="checkpoints/lora_arm1b_qwen_qwen3-1.7b_k0")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    spot_check(args.adapter_path, args.model_id, num_samples=args.num_samples, device=args.device)
