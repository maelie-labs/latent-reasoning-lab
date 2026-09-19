#!/usr/bin/env python3
"""
scripts/30_eval_qwen3_1.7b.py
Comprehensive evaluation of Qwen/Qwen3-1.7B across:
1. Direct Base Generation (0 thinking tokens / forced direct answer)
2. Compute-Matched Discrete Baseline (<= 6 discrete thinking tokens, FLOPs matched to K=6)
3. Fixed K=6 Continuous Latent Recurrence (LoRA adapter)
4. Unconstrained Discrete Baseline CoT (Native think stream)

Evaluated across the standardized 25 problems:
- 15 GSM8K problems (5 Easy, 5 Medium, 5 Hard)
- 10 MATH-500 Olympiad Level 4/5 problems
All in pure uncompressed bfloat16 on cuda:0 (RTX 4080 16GB).
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

MODEL_ID = "Qwen/Qwen3-1.7B"
DEVICE = "cuda:0"
SCALE_FACTOR = 0.034016  # ||W_E|| / sqrt(2048)

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
    if is_answer_only:
        nums = re.findall(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?', text)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                pass
    return None

def check_match(pred_num, pred_text, gt_num, gt_raw):
    if pred_num is not None and gt_num is not None:
        if abs(pred_num - gt_num) < 1e-4:
            return True
    if gt_raw:
        pred_box = extract_math_boxed_expression(pred_text)
        if pred_box and pred_box.strip() == gt_raw.strip():
            return True
        if gt_raw.strip() in pred_text:
            return True
    return False

def get_benchmarks():
    # 1. GSM8K Stratified
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
            
    # 2. MATH-500 Level 4/5
    ds_math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math_hard = []
    for i, item in enumerate(ds_math):
        lvl = item.get("level")
        if lvl in [4, 5] or "Level 4" in str(lvl) or "Level 5" in str(lvl):
            math_hard.append({"id": f"math500_l45_{i}", "category": "MATH-500", "question": item["problem"], "answer": item["answer"]})
        if len(math_hard) >= 10:
            break
            
    return easy + med + hard + math_hard

def run_discrete_baseline(model, tokenizer, question, mode="unconstrained", max_new_tokens=2048):
    """
    mode can be:
    - 'unconstrained': Full native discrete CoT (<think>...</think> + answer)
    - 'matched_6tok': Exactly <= 6 tokens generated inside <think>, then forced to answer
    - 'direct_0tok': 0 tokens in think block (<think>\n</think>\n\n), forced direct answer
    """
    messages = [{"role": "user", "content": question}]
    base_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if mode == "direct_0tok":
        # Force immediate answer generation
        prompt = base_prompt + "<think>\n</think>\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        torch.cuda.synchronize()
        t0 = time.time()
        
        # Prefill prompt
        out_p = model(input_ids=inputs.input_ids, use_cache=True)
        past_kv = out_p.past_key_values
        
        # Suppress <|im_end|> on first answer token
        logits = out_p.logits[:, -1, :].clone()
        logits[:, 151645] = -float('inf')  # <|im_end|>
        first_tok = logits.argmax(dim=-1, keepdim=True)
        
        ans_ids = [first_tok.item()]
        curr_id = first_tok
        
        for _ in range(max_new_tokens - 1):
            step = model(input_ids=curr_id, past_key_values=past_kv, use_cache=True)
            past_kv = step.past_key_values
            nxt = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if nxt.item() == 151645 or nxt.item() == tokenizer.eos_token_id:
                break
            ans_ids.append(nxt.item())
            curr_id = nxt
            
        torch.cuda.synchronize()
        total_time = time.time() - t0
        ans_text = tokenizer.decode(ans_ids, skip_special_tokens=True)
        
        return {
            "think_time_ms": 0.0,
            "think_tokens": 0,
            "answer_time_s": round(total_time, 3),
            "answer_tokens": len(ans_ids),
            "e2e_latency_s": round(total_time, 3),
            "decoding_tok_s": round(len(ans_ids) / max(0.001, total_time), 2),
            "full_output": ans_text
        }
        
    elif mode == "matched_6tok":
        # Generate exactly 6 tokens in <think>
        prompt = base_prompt + "<think>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        torch.cuda.synchronize()
        t0 = time.time()
        out_think = model.generate(**inputs, max_new_tokens=6, do_sample=False)
        torch.cuda.synchronize()
        think_time_ms = (time.time() - t0) * 1000
        
        think_toks = out_think[0][inputs.input_ids.shape[1]:]
        think_str = tokenizer.decode(think_toks, skip_special_tokens=True)
        
        forced_prompt = prompt + think_str + "\n</think>\n\n"
        inputs_ans = tokenizer(forced_prompt, return_tensors="pt").to(DEVICE)
        
        torch.cuda.synchronize()
        t_ans0 = time.time()
        out_p = model(input_ids=inputs_ans.input_ids, use_cache=True)
        past_kv = out_p.past_key_values
        
        logits = out_p.logits[:, -1, :].clone()
        logits[:, 151645] = -float('inf')  # Suppress <|im_end|> on transition
        first_tok = logits.argmax(dim=-1, keepdim=True)
        
        ans_ids = [first_tok.item()]
        curr_id = first_tok
        
        for _ in range(max_new_tokens - 1):
            step = model(input_ids=curr_id, past_key_values=past_kv, use_cache=True)
            past_kv = step.past_key_values
            nxt = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if nxt.item() == 151645 or nxt.item() == tokenizer.eos_token_id:
                break
            ans_ids.append(nxt.item())
            curr_id = nxt
            
        torch.cuda.synchronize()
        ans_time_s = time.time() - t_ans0
        total_time_s = (think_time_ms / 1000.0) + ans_time_s
        ans_text = tokenizer.decode(ans_ids, skip_special_tokens=True)
        
        return {
            "think_time_ms": round(think_time_ms, 2),
            "think_tokens": len(think_toks),
            "answer_time_s": round(ans_time_s, 3),
            "answer_tokens": len(ans_ids),
            "e2e_latency_s": round(total_time_s, 3),
            "decoding_tok_s": round(len(ans_ids) / max(0.001, ans_time_s), 2),
            "full_output": "</think>\n\n" + ans_text
        }
        
    else:
        # Unconstrained discrete baseline
        prompt = base_prompt + "<think>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        torch.cuda.synchronize()
        t0 = time.time()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        total_time = time.time() - t0
        
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        full_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        if "</think>" in full_text:
            parts = full_text.split("</think>", 1)
            think_str = parts[0]
            ans_str = parts[1]
            think_tok_cnt = len(tokenizer.encode(think_str, add_special_tokens=False))
            ans_tok_cnt = len(tokenizer.encode(ans_str, add_special_tokens=False))
            tok_s = (think_tok_cnt + ans_tok_cnt) / max(0.001, total_time)
            think_time_ms = (think_tok_cnt / max(1, (think_tok_cnt + ans_tok_cnt))) * total_time * 1000
            ans_time_s = total_time - (think_time_ms / 1000.0)
        else:
            think_time_ms = total_time * 1000
            think_tok_cnt = len(new_tokens)
            ans_tok_cnt = 0
            ans_time_s = 0.0
            tok_s = think_tok_cnt / max(0.001, total_time)
            
        return {
            "think_time_ms": round(think_time_ms, 2),
            "think_tokens": think_tok_cnt,
            "answer_time_s": round(ans_time_s, 3),
            "answer_tokens": ans_tok_cnt,
            "e2e_latency_s": round(total_time, 3),
            "decoding_tok_s": round(tok_s, 2),
            "full_output": full_text
        }

def run_recurrent_lora(model, tokenizer, question, k_steps=6, max_new_tokens=2048):
    """
    Runs fixed K-step continuous latent recurrence on last token position,
    then generates step-by-step mathematical proof and boxed answer.
    """
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    torch.cuda.synchronize()
    t_start = time.time()
    
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        
        torch.cuda.synchronize()
        t_think0 = time.time()
        
        for _ in range(k_steps):
            scaled = curr_latent * SCALE_FACTOR
            step_out = model(
                inputs_embeds=scaled,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]
            
        torch.cuda.synchronize()
        think_time_ms = (time.time() - t_think0) * 1000
        
        # Transition out of thinking mode with </think>\n\n prefix
        prefix_ids = tokenizer.encode("</think>\n\n", add_special_tokens=False, return_tensors="pt").to(DEVICE)
        out_prefix = model(input_ids=prefix_ids, past_key_values=past_kv, use_cache=True)
        past_kv = out_prefix.past_key_values
        
        # Suppress <|im_end|> on first answer token
        logits = out_prefix.logits[:, -1, :].clone()
        logits[:, 151645] = -float('inf')
        first_token_id = logits.argmax(dim=-1, keepdim=True)
        
        # Autoregressive decoding for remainder of answer
        t_ans0 = time.time()
        ans_ids = [first_token_id.item()]
        curr_id = first_token_id
        
        for _ in range(max_new_tokens - 1):
            step_dec = model(
                input_ids=curr_id,
                past_key_values=past_kv,
                use_cache=True
            )
            past_kv = step_dec.past_key_values
            next_id = step_dec.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_id.item() == 151645 or next_id.item() == tokenizer.eos_token_id:
                break
            ans_ids.append(next_id.item())
            curr_id = next_id
                
        torch.cuda.synchronize()
        ans_time_s = time.time() - t_ans0
        total_time_s = (think_time_ms / 1000.0) + ans_time_s
        
        full_text = tokenizer.decode(ans_ids, skip_special_tokens=True)
        return {
            "think_time_ms": round(think_time_ms, 2),
            "think_tokens": k_steps,
            "answer_time_s": round(ans_time_s, 3),
            "answer_tokens": len(ans_ids),
            "e2e_latency_s": round(total_time_s, 3),
            "decoding_tok_s": round(len(ans_ids) / max(0.001, ans_time_s), 2),
            "full_output": "</think>\n\n" + full_text
        }

def main():
    print(f"=== Rigorous Benchmark Suite: {MODEL_ID} on {DEVICE} ===")
    
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip("\"'\n")
                    break

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token, trust_remote_code=True)
    
    print("Loading base model in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    
    lora_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "lora_recurrent_qwen3_1.7b")
    has_lora = os.path.exists(lora_dir)
    if has_lora:
        print(f"Loading LoRA adapter from {lora_dir}...")
        recurrent_model = PeftModel.from_pretrained(base_model, lora_dir)
        recurrent_model.eval()
    else:
        print("LoRA adapter not found. Exiting.")
        return

    problems = get_benchmarks()
    print(f"Loaded {len(problems)} benchmark problems (15 GSM8K + 10 MATH-500).")
    
    results = {
        "metadata": {
            "model_id": MODEL_ID,
            "device": DEVICE,
            "num_problems": len(problems),
            "scale_factor": SCALE_FACTOR,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "summary": {},
        "detailed_runs": []
    }
    
    conditions = [
        ("base_direct_0tok", "Base Direct (0 thinking tokens)"),
        ("matched_discrete_6tok", "Compute-Matched Discrete (<= 6 tokens)"),
        ("fixed_k6_recurrent", "Fixed K=6 Continuous Recurrent"),
        ("unconstrained_cot", "Unconstrained Discrete Baseline CoT"),
    ]
        
    for cond_key, cond_name in conditions:
        print(f"\n{'='*60}\nEvaluating Condition: {cond_name}\n{'='*60}")
        correct_total = 0
        correct_gsm = 0
        correct_math = 0
        think_times = []
        ans_times = []
        ans_toks = []
        e2e_times = []
        
        for idx, prob in enumerate(problems):
            q = prob["question"]
            gt_text = prob["answer"]
            gt_num = clean_and_extract_candidate(extract_math_boxed_expression(gt_text) or gt_text)
            
            if cond_key == "base_direct_0tok":
                run_res = run_discrete_baseline(base_model, tokenizer, q, mode="direct_0tok")
            elif cond_key == "matched_discrete_6tok":
                run_res = run_discrete_baseline(base_model, tokenizer, q, mode="matched_6tok")
            elif cond_key == "unconstrained_cot":
                run_res = run_discrete_baseline(base_model, tokenizer, q, mode="unconstrained")
            elif cond_key == "fixed_k6_recurrent":
                run_res = run_recurrent_lora(recurrent_model, tokenizer, q, k_steps=6)
                
            pred_num = extract_numeric_answer(run_res["full_output"])
            is_correct = check_match(pred_num, run_res["full_output"], gt_num, gt_text)
            
            if is_correct:
                correct_total += 1
                if prob["category"] == "MATH-500":
                    correct_math += 1
                else:
                    correct_gsm += 1
                    
            think_times.append(run_res["think_time_ms"])
            ans_times.append(run_res["answer_time_s"])
            ans_toks.append(run_res["answer_tokens"])
            e2e_times.append(run_res["e2e_latency_s"])
            
            print(f"[{idx+1}/{len(problems)}] {prob['id']} ({prob['category']}) | Correct: {is_correct} | Think: {run_res['think_time_ms']:.1f}ms | Ans Tok: {run_res['answer_tokens']}")
            
        summary_stat = {
            "name": cond_name,
            "overall_accuracy": round(correct_total / len(problems) * 100, 2),
            "gsm8k_accuracy": round(correct_gsm / 15 * 100, 2),
            "math500_accuracy": round(correct_math / 10 * 100, 2),
            "mean_think_time_ms": round(sum(think_times) / len(think_times), 2),
            "mean_answer_time_s": round(sum(ans_times) / len(ans_times), 3),
            "mean_answer_tokens": round(sum(ans_toks) / len(ans_toks), 1),
            "mean_e2e_latency_s": round(sum(e2e_times) / len(e2e_times), 3),
        }
        results["summary"][cond_key] = summary_stat
        print(f"\n--- Summary for {cond_name} ---")
        print(f"Overall Acc: {summary_stat['overall_accuracy']}% | GSM8K: {summary_stat['gsm8k_accuracy']}% | MATH-500: {summary_stat['math500_accuracy']}% | Mean Think: {summary_stat['mean_think_time_ms']} ms")
        
    out_file = os.path.join(os.path.dirname(__file__), "..", "data", "eval_results_qwen3_1.7b.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll benchmark results saved to: {out_file}")

if __name__ == "__main__":
    main()
