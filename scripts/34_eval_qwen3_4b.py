#!/usr/bin/env python3
"""
scripts/34_eval_qwen3_4b.py
Comprehensive evaluation of Qwen/Qwen3-4B across:
1. Direct Base Generation (0 thinking tokens / forced direct answer)
2. Compute-Matched Discrete Baseline (<= 6 discrete thinking tokens, FLOPs matched to K=6)
3. Fixed K=6 Continuous Latent Recurrence (LoRA adapter)
4. Unconstrained Discrete Baseline CoT (Native think stream)

Evaluated across the standardized 25 problems:
- 15 GSM8K problems (5 Easy, 5 Medium, 5 Hard)
- 10 MATH-500 Olympiad Level 4/5 problems
All in pure uncompressed bfloat16 on cuda:1 (RTX PRO 4500 Blackwell 32GB).
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
from peft import PeftModel

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

MODEL_ID = "Qwen/Qwen3-4B"
DEVICE = "cuda:1"

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
    boxed = extract_math_boxed_expression(text)
    if boxed:
        val = clean_and_extract_candidate(boxed)
        if val is not None:
            return val
    hash_match = re.findall(r'####\s*([-\d.,]+)', text)
    if hash_match:
        val = clean_and_extract_candidate(hash_match[-1])
        if val is not None:
            return val
    ans_match = re.findall(r'(?:the answer is|final answer is|total is|equals|equal to)\s*([\$]?[-\d.,]+)', text, re.IGNORECASE)
    if ans_match:
        val = clean_and_extract_candidate(ans_match[-1])
        if val is not None:
            return val
    patterns = [
        r'=\s*([\$]?[-\d.,]+)\s*(?:\\n|$|\.|\))',
        r'(?:is|\$)\s*([-\d.,]+)\s*(?:\\n|$|\.|\))',
        r'\b([-\d.,]+)\b'
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            for m in reversed(matches):
                val = clean_and_extract_candidate(m)
                if val is not None:
                    return val
    return None

def check_correctness(prediction_text, ground_truth, is_math=False):
    gt_boxed = extract_math_boxed_expression(str(ground_truth))
    gt_clean = gt_boxed if gt_boxed else str(ground_truth).strip()
    gt_val = clean_and_extract_candidate(gt_clean)
    pred_boxed = extract_math_boxed_expression(prediction_text)
    if pred_boxed and gt_boxed:
        c_pred = re.sub(r"[\$\\%!\s]", "", pred_boxed).replace(",", "").strip()
        c_gt = re.sub(r"[\$\\%!\s]", "", gt_boxed).replace(",", "").strip()
        if c_pred == c_gt:
            return True
    pred_val = extract_numeric_answer(prediction_text)
    if pred_val is not None and gt_val is not None:
        if abs(pred_val - gt_val) < 1e-4:
            return True
    return False

def load_eval_suite():
    suite_file = os.path.join(os.path.dirname(__file__), "..", "data", "eval_problems_suite.json")
    if os.path.exists(suite_file):
        with open(suite_file, "r") as f:
            return json.load(f)
    print("Generating standardized 25-problem test suite...")
    suite = []
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    easy_indices = [0, 1, 3, 4, 13]
    for idx in easy_indices:
        ex = gsm8k[idx]
        gt = ex["answer"].split("####")[-1].strip()
        suite.append({"id": f"gsm8k_easy_{idx}", "category": "Easy", "question": ex["question"], "ground_truth": gt, "is_math": False})
    med_indices = [2, 6, 7, 11, 12]
    for idx in med_indices:
        ex = gsm8k[idx]
        gt = ex["answer"].split("####")[-1].strip()
        suite.append({"id": f"gsm8k_med_{idx}", "category": "Medium", "question": ex["question"], "ground_truth": gt, "is_math": False})
    hard_indices = [5, 8, 9, 10, 25]
    for idx in hard_indices:
        ex = gsm8k[idx]
        gt = ex["answer"].split("####")[-1].strip()
        suite.append({"id": f"gsm8k_hard_{idx}", "category": "Hard", "question": ex["question"], "ground_truth": gt, "is_math": False})
    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    l45_indices = [1, 7, 9, 11, 12, 15, 17, 22, 23, 24]
    for idx in l45_indices:
        ex = math_ds[idx]
        suite.append({"id": f"math500_l45_{idx}", "category": "MATH-500", "question": ex["problem"], "ground_truth": ex["solution"], "is_math": True})
    os.makedirs(os.path.dirname(suite_file), exist_ok=True)
    with open(suite_file, "w") as f:
        json.dump(suite, f, indent=2)
    return suite

def run_base_direct(model, tokenizer, item, device):
    prompt = item["question"]
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    forced_input = formatted + "<think>\n</think>\n\n"
    inputs = tokenizer(forced_input, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            suppress_tokens=[tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]
        )
    dur = time.time() - t0
    gen_tokens = out[0][inputs.input_ids.shape[1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return {
        "think_time_ms": 0.0,
        "gen_time_s": dur,
        "think_tokens": 0,
        "ans_tokens": len(gen_tokens),
        "think_text": "",
        "ans_text": text.strip(),
        "is_correct": check_correctness(text, item["ground_truth"], item.get("is_math", False))
    }

def run_compute_matched_discrete(model, tokenizer, item, device, max_think_tokens=6):
    prompt = item["question"]
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_str = formatted + "<think>\n"
    inputs = tokenizer(input_str, return_tensors="pt").to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out_think = model.generate(
            **inputs,
            max_new_tokens=max_think_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize()
    t_think = (time.time() - t0) * 1000.0
    think_ids = out_think[0][inputs.input_ids.shape[1]:]
    think_text = tokenizer.decode(think_ids, skip_special_tokens=True)
    forced_answer_input = input_str + think_text + "\n</think>\n\n"
    ans_inputs = tokenizer(forced_answer_input, return_tensors="pt").to(device)
    t1 = time.time()
    with torch.no_grad():
        out_ans = model.generate(
            **ans_inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            suppress_tokens=[tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]]
        )
    t_ans = time.time() - t1
    ans_ids = out_ans[0][ans_inputs.input_ids.shape[1]:]
    ans_text = tokenizer.decode(ans_ids, skip_special_tokens=True)
    return {
        "think_time_ms": t_think,
        "gen_time_s": t_ans,
        "think_tokens": len(think_ids),
        "ans_tokens": len(ans_ids),
        "think_text": think_text.strip(),
        "ans_text": ans_text.strip(),
        "is_correct": check_correctness(ans_text, item["ground_truth"], item.get("is_math", False))
    }

def run_recurrent_k6(model, tokenizer, item, device, scale_factor, k_steps=6):
    prompt = item["question"]
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_str = formatted + "<think>\n"
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        for _ in range(k_steps):
            scaled_latent = curr_latent * scale_factor
            step_out = model(
                inputs_embeds=scaled_latent,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]
    torch.cuda.synchronize()
    t_think = (time.time() - t0) * 1000.0
    t1 = time.time()
    with torch.no_grad():
        trans_text = "</think>\n\n"
        trans_tokens = tokenizer.encode(trans_text, add_special_tokens=False)
        step_kv = past_kv
        curr_input_ids = torch.tensor([trans_tokens], dtype=torch.long, device=device)
        step_out = model(input_ids=curr_input_ids, past_key_values=step_kv, use_cache=True)
        step_kv = step_out.past_key_values
        next_tok = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        if next_tok.item() in [tokenizer.eos_token_id, im_end_id]:
            logits_clone = step_out.logits[:, -1, :].clone()
            logits_clone[0, tokenizer.eos_token_id] = -float('inf')
            logits_clone[0, im_end_id] = -float('inf')
            next_tok = torch.argmax(logits_clone, dim=-1, keepdim=True)
        gen_tokens = [next_tok.item()]
        curr_tok = next_tok
        for _ in range(2048):
            step_out = model(input_ids=curr_tok, past_key_values=step_kv, use_cache=True)
            step_kv = step_out.past_key_values
            curr_tok = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
            t_val = curr_tok.item()
            if t_val in [tokenizer.eos_token_id, im_end_id]:
                break
            gen_tokens.append(t_val)
    t_ans = time.time() - t1
    ans_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return {
        "think_time_ms": t_think,
        "gen_time_s": t_ans,
        "think_tokens": k_steps,
        "ans_tokens": len(gen_tokens),
        "think_text": f"[{k_steps} latent recurrent loops]",
        "ans_text": ans_text.strip(),
        "is_correct": check_correctness(ans_text, item["ground_truth"], item.get("is_math", False))
    }

def run_unconstrained_cot(model, tokenizer, item, device):
    prompt = item["question"]
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    dur = time.time() - t0
    gen_tokens = out[0][inputs.input_ids.shape[1]:]
    full_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    think_text = ""
    ans_text = full_text
    if "<think>" in full_text and "</think>" in full_text:
        parts = full_text.split("</think>")
        think_text = parts[0].replace("<think>", "").strip()
        ans_text = parts[1].strip()
    think_tokens = len(tokenizer.encode(think_text, add_special_tokens=False)) if think_text else 0
    ans_tokens = len(tokenizer.encode(ans_text, add_special_tokens=False))
    return {
        "think_time_ms": dur * 1000.0 * (think_tokens / max(1, think_tokens + ans_tokens)),
        "gen_time_s": dur,
        "think_tokens": think_tokens,
        "ans_tokens": ans_tokens,
        "think_text": think_text,
        "ans_text": ans_text,
        "is_correct": check_correctness(ans_text, item["ground_truth"], item.get("is_math", False))
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-4B across 4 conditions.")
    parser.add_argument("--lora_path", type=str, default=os.path.join(os.path.dirname(__file__), "..", "checkpoints", "lora_recurrent_qwen3_4b"))
    parser.add_argument("--output_file", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "eval_results_qwen3_4b.json"))
    args = parser.parse_args()
    
    # Load scale factor from profiling results if available
    prof_file = os.path.join(os.path.dirname(__file__), "..", "data", "qwen3_4b_profiling_results.json")
    if os.path.exists(prof_file):
        with open(prof_file) as f:
            prof = json.load(f)
            scale_factor = prof.get("alpha_factor", 0.021689)
    else:
        scale_factor = 0.021689
        
    print(f"=== Qwen/Qwen3-4B 4-Arm Benchmark Evaluation on {DEVICE} ===")
    print(f"Scale Factor alpha: {scale_factor:.6f}")
    
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip("\"'\n")
                    break
                    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token, trust_remote_code=True)
    test_suite = load_eval_suite()
    print(f"Loaded {len(test_suite)} evaluation problems.")
    
    results = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "scale_factor": scale_factor,
        "conditions": {}
    }
    
    print("\nLoading base model in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    base_model.eval()
    
    # Condition 1: Base Direct
    print("\n" + "="*60)
    print("Evaluating Condition: Base Direct (0 thinking tokens)")
    print("="*60)
    c1_runs = []
    for i, item in enumerate(test_suite):
        res = run_base_direct(base_model, tokenizer, item, DEVICE)
        res["id"] = item["id"]
        res["category"] = item["category"]
        c1_runs.append(res)
        print(f"[{i+1}/{len(test_suite)}] {item['id']} ({item['category']}) | Correct: {res['is_correct']} | Think: {res['think_time_ms']:.1f}ms | Ans Tok: {res['ans_tokens']}")
        
    c1_acc = sum(r["is_correct"] for r in c1_runs) / len(c1_runs) * 100.0
    c1_gsm8k = sum(r["is_correct"] for r in c1_runs if "gsm8k" in r["id"]) / 15 * 100.0
    c1_math = sum(r["is_correct"] for r in c1_runs if "math500" in r["id"]) / 10 * 100.0
    print(f"\n--- Summary for Base Direct ---")
    print(f"Overall Acc: {c1_acc:.1f}% | GSM8K: {c1_gsm8k:.2f}% | MATH-500: {c1_math:.1f}%")
    results["conditions"]["base_direct"] = {"overall_acc": c1_acc, "gsm8k_acc": c1_gsm8k, "math500_acc": c1_math, "runs": c1_runs}
    
    # Condition 2: Compute-Matched Discrete (<= 6 tokens)
    print("\n" + "="*60)
    print("Evaluating Condition: Compute-Matched Discrete (<= 6 tokens)")
    print("="*60)
    c2_runs = []
    for i, item in enumerate(test_suite):
        res = run_compute_matched_discrete(base_model, tokenizer, item, DEVICE, max_think_tokens=6)
        res["id"] = item["id"]
        res["category"] = item["category"]
        c2_runs.append(res)
        print(f"[{i+1}/{len(test_suite)}] {item['id']} ({item['category']}) | Correct: {res['is_correct']} | Think: {res['think_time_ms']:.1f}ms | Ans Tok: {res['ans_tokens']}")
        
    c2_acc = sum(r["is_correct"] for r in c2_runs) / len(c2_runs) * 100.0
    c2_gsm8k = sum(r["is_correct"] for r in c2_runs if "gsm8k" in r["id"]) / 15 * 100.0
    c2_math = sum(r["is_correct"] for r in c2_runs if "math500" in r["id"]) / 10 * 100.0
    c2_think = sum(r["think_time_ms"] for r in c2_runs) / len(c2_runs)
    print(f"\n--- Summary for Compute-Matched Discrete (<= 6 tokens) ---")
    print(f"Overall Acc: {c2_acc:.1f}% | GSM8K: {c2_gsm8k:.2f}% | MATH-500: {c2_math:.1f}% | Mean Think: {c2_think:.1f} ms")
    results["conditions"]["compute_matched_discrete"] = {"overall_acc": c2_acc, "gsm8k_acc": c2_gsm8k, "math500_acc": c2_math, "mean_think_ms": c2_think, "runs": c2_runs}
    
    # Condition 3: Fixed K=6 Latent Recurrence
    if os.path.exists(args.lora_path):
        print("\nLoading LoRA adapter for Condition 3...")
        recurrent_model = PeftModel.from_pretrained(base_model, args.lora_path)
        recurrent_model.eval()
        
        print("\n" + "="*60)
        print("Evaluating Condition: Fixed K=6 Continuous Latent Recurrence")
        print("="*60)
        c3_runs = []
        for i, item in enumerate(test_suite):
            res = run_recurrent_k6(recurrent_model, tokenizer, item, DEVICE, scale_factor=scale_factor, k_steps=6)
            res["id"] = item["id"]
            res["category"] = item["category"]
            c3_runs.append(res)
            print(f"[{i+1}/{len(test_suite)}] {item['id']} ({item['category']}) | Correct: {res['is_correct']} | Think: {res['think_time_ms']:.1f}ms | Ans Tok: {res['ans_tokens']}")
            
        c3_acc = sum(r["is_correct"] for r in c3_runs) / len(c3_runs) * 100.0
        c3_gsm8k = sum(r["is_correct"] for r in c3_runs if "gsm8k" in r["id"]) / 15 * 100.0
        c3_math = sum(r["is_correct"] for r in c3_runs if "math500" in r["id"]) / 10 * 100.0
        c3_think = sum(r["think_time_ms"] for r in c3_runs) / len(c3_runs)
        print(f"\n--- Summary for Fixed K=6 Continuous Latent Recurrence ---")
        print(f"Overall Acc: {c3_acc:.1f}% | GSM8K: {c3_gsm8k:.2f}% | MATH-500: {c3_math:.1f}% | Mean Think: {c3_think:.1f} ms")
        results["conditions"]["recurrent_k6"] = {"overall_acc": c3_acc, "gsm8k_acc": c3_gsm8k, "math500_acc": c3_math, "mean_think_ms": c3_think, "runs": c3_runs}
        
        # Unload LoRA to evaluate Condition 4 clean
        del recurrent_model
        torch.cuda.empty_cache()
    else:
        print(f"LoRA adapter not found at {args.lora_path}, skipping Condition 3.")

    # Condition 4: Unconstrained Discrete Teacher CoT
    print("\n" + "="*60)
    print("Evaluating Condition: Unconstrained Discrete Teacher CoT")
    print("="*60)
    c4_runs = []
    for i, item in enumerate(test_suite):
        res = run_unconstrained_cot(base_model, tokenizer, item, DEVICE)
        res["id"] = item["id"]
        res["category"] = item["category"]
        c4_runs.append(res)
        print(f"[{i+1}/{len(test_suite)}] {item['id']} ({item['category']}) | Correct: {res['is_correct']} | Think: {res['think_time_ms']:.1f}ms | Think Tok: {res['think_tokens']} | Ans Tok: {res['ans_tokens']}")
        
    c4_acc = sum(r["is_correct"] for r in c4_runs) / len(c4_runs) * 100.0
    c4_gsm8k = sum(r["is_correct"] for r in c4_runs if "gsm8k" in r["id"]) / 15 * 100.0
    c4_math = sum(r["is_correct"] for r in c4_runs if "math500" in r["id"]) / 10 * 100.0
    c4_think = sum(r["think_time_ms"] for r in c4_runs) / len(c4_runs)
    c4_think_tok = sum(r["think_tokens"] for r in c4_runs) / len(c4_runs)
    print(f"\n--- Summary for Unconstrained Discrete Teacher CoT ---")
    print(f"Overall Acc: {c4_acc:.1f}% | GSM8K: {c4_gsm8k:.2f}% | MATH-500: {c4_math:.1f}% | Mean Think: {c4_think:.1f} ms ({c4_think_tok:.1f} tok)")
    results["conditions"]["unconstrained_cot"] = {"overall_acc": c4_acc, "gsm8k_acc": c4_gsm8k, "math500_acc": c4_math, "mean_think_ms": c4_think, "mean_think_tok": c4_think_tok, "runs": c4_runs}
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all 4-arm evaluation results to: {args.output_file}")

if __name__ == "__main__":
    main()
