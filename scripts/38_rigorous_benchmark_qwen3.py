#!/usr/bin/env python3
"""
scripts/38_rigorous_benchmark_qwen3.py
COMPREHENSIVE MULTI-SAMPLE BENCHMARK HARNESS (Revision 8)

Evaluates Qwen 3 models across the rigorous benchmark arms:
- Arm 1: Base Direct (enable_thinking=False)
- Arm 1b: Trained Direct (No-CoT LoRA adapter)
- Arm 2: Minimal Discrete / Near-FLOP Matched (K discrete thinking tokens)
- Arm 2b: Pause-Token Control (K learned <pause> tokens)
- Arm 3: Continuous Latent Autoregression (K latent passes with empirical alpha)
- Arm 4: Unconstrained Discrete CoT (up to 32,768 tokens)
- Arm 5a: Wall-Clock Matched Discrete CoT

Key Scientific Controls:
1. Multi-seed sampling (seeds = [42, 123, 456, 789]).
2. Hierarchical problem bootstrapping (B = 10,000) for true 95% CIs and paired differences.
3. Strict truncation accounting (truncations = 0; truncation rate reported).
4. Coconut-style sequential latent unrolling with KV caching and empirical alpha.
5. Identical stochastic sampler on recurrent answer decoding.
6. Exact post-RMSNorm terminal hidden state h_k extraction.
7. </think>\\n\\n transition token embedding bridge into answer generation.
"""

import os
import re
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from math_verify import parse, verify
import transformers.modeling_utils

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

THINK_END_TOKEN_ID = 151668  # </think>
PAUSE_TOKEN = "<pause>"

# Calibrated Empirical Alpha Factors (Accounting for RMSNorm gamma)
CALIBRATED_ALPHAS = {
    "qwen/qwen3-1.7b": 0.011440,
    "qwen/qwen3-4b": 0.006702,
    "qwen/qwen3-8b": 0.008500,   # To be confirmed at 8B gate
    "qwen/qwen3-14b": 0.005200   # To be confirmed at 14B gate
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sanitize_for_json(obj):
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj

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

def extract_answer(full_text, content_text, thinking_text=""):
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
            gold_p = parse(full_gold_text, parsing_timeout=2.0)
            pred_p = parse(full_pred_text, parsing_timeout=2.0)
            if gold_p and pred_p and verify(gold_p, pred_p):
                return True
        except Exception:
            pass

    if pred_raw and gt_raw:
        try:
            gold_p = parse(str(gt_raw), parsing_timeout=2.0)
            pred_p = parse(str(pred_raw), parsing_timeout=2.0)
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

def count_solution_steps(answer_text):
    steps = len(re.findall(r'<<.*?>>', answer_text))
    return max(1, steps)

def load_benchmark_suite(suite_size=250):
    cache_path = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_suite_250.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
            return data[:suite_size]

    suite = []
    print(f"Generating mechanically stratified 250-problem benchmark suite...")
    
    # GSM8K: 150 problems stratified by reference calculation step count
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    short_gsm, med_gsm, long_gsm = [], [], []
    for idx, ex in enumerate(gsm8k):
        steps = count_solution_steps(ex["answer"])
        item = {
            "id": f"gsm8k_{idx}",
            "benchmark": "GSM8K",
            "question": ex["question"],
            "solution": ex["answer"],
            "steps": steps
        }
        if steps <= 3:
            short_gsm.append(item)
        elif steps <= 5:
            med_gsm.append(item)
        else:
            long_gsm.append(item)
            
    random.Random(42).shuffle(short_gsm)
    random.Random(42).shuffle(med_gsm)
    random.Random(42).shuffle(long_gsm)
    
    selected_gsm = short_gsm[:50] + med_gsm[:50] + long_gsm[:50]
    for idx, item in enumerate(selected_gsm):
        item["stratum"] = "Short" if item["steps"] <= 3 else ("Medium" if item["steps"] <= 5 else "Long")
        suite.append(item)

    # MATH-500: 100 problems uniformly stratified across difficulty levels
    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    levels = {i: [] for i in range(1, 6)}
    for idx, ex in enumerate(math_ds):
        lvl = ex.get("level", 3)
        if isinstance(lvl, str) and "Level " in lvl:
            try:
                lvl = int(lvl.replace("Level ", ""))
            except ValueError:
                lvl = 3
        elif not isinstance(lvl, int):
            lvl = 3
        lvl = max(1, min(5, lvl))
        levels[lvl].append({
            "id": f"math500_{idx}",
            "benchmark": "MATH-500",
            "question": ex["problem"],
            "solution": ex["solution"],
            "subject": ex.get("subject", "Math"),
            "level": lvl,
            "stratum": f"Level {lvl}"
        })
        
    for lvl in range(1, 6):
        random.Random(42).shuffle(levels[lvl])
        suite.extend(levels[lvl][:20])  # 20 per level = 100 total
        
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(suite, f, indent=2)
    print(f"Saved {len(suite)} problems to {cache_path}")
    return suite[:suite_size]

def sample_token(logits, temperature=0.6, top_p=0.95, top_k=20):
    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("Inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def run_arm1_base_direct(model, tokenizer, prompt, device, max_tokens=8192):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    
    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize(device)
    dur = time.time() - t0
    gen_tokens = out[0][prompt_len:].tolist()
    is_truncated = bool(len(gen_tokens) >= max_tokens and out[0][-1].item() != tokenizer.eos_token_id)
    content = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    pred_val, pred_raw = extract_answer(content, content)
    return {
        "think_tokens": 0,
        "ans_tokens": len(gen_tokens),
        "think_time_ms": 0.0,
        "total_time_s": round(dur, 3),
        "is_truncated": is_truncated,
        "content": content,
        "pred_val": pred_val,
        "pred_raw": pred_raw
    }

def run_arm1b_trained_direct(model, tokenizer, prompt, device, max_tokens=8192):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    
    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize(device)
    dur = time.time() - t0
    gen_tokens = out[0][prompt_len:].tolist()
    is_truncated = bool(len(gen_tokens) >= max_tokens and out[0][-1].item() != tokenizer.eos_token_id)
    content = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    pred_val, pred_raw = extract_answer(content, content)
    return {
        "think_tokens": 0,
        "ans_tokens": len(gen_tokens),
        "think_time_ms": 0.0,
        "total_time_s": round(dur, 3),
        "is_truncated": is_truncated,
        "content": content,
        "pred_val": pred_val,
        "pred_raw": pred_raw
    }

def run_arm2b_pause_tokens(model, tokenizer, pause_token_id, prompt, device, k_tokens=6, max_ans_tokens=8192):
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    enc_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    pause_tokens = [pause_token_id] * k_tokens
    transition_text = "\n</think>\n\n"
    enc_transition = tokenizer.encode(transition_text, add_special_tokens=False)
    
    full_prefix = enc_prompt + pause_tokens + enc_transition
    full_prefix_ids = torch.tensor([full_prefix], dtype=torch.long, device=device)
    prefix_len = full_prefix_ids.shape[1]
    
    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=full_prefix_ids,
            max_new_tokens=max_ans_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize(device)
    dur = time.time() - t0
    gen_tokens = out[0][prefix_len:].tolist()
    is_truncated = bool(len(gen_tokens) >= max_ans_tokens and out[0][-1].item() != tokenizer.eos_token_id)
    content = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    pred_val, pred_raw = extract_answer(content, content)
    return {
        "think_tokens": k_tokens,
        "ans_tokens": len(gen_tokens),
        "think_time_ms": 0.0,
        "total_time_s": round(dur, 3),
        "is_truncated": is_truncated,
        "content": content,
        "pred_val": pred_val,
        "pred_raw": pred_raw
    }

def run_arm2_minimal_discrete(model, tokenizer, prompt, device, k_tokens=6, max_ans_tokens=8192):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    
    t0 = time.time()
    with torch.no_grad():
        # Step 1: Generate exactly K thinking tokens
        out_think = model.generate(
            **inputs,
            max_new_tokens=k_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
        t_think_ms = (time.time() - t0) * 1000.0
        
        # Step 2: Ingest </think>\n\n transition
        think_tokens = out_think[0][prompt_len:].tolist()
        think_str = tokenizer.decode(think_tokens, skip_special_tokens=True)
        transition_str = text + think_str + "\n</think>\n\n"
        ans_inputs = tokenizer([transition_str], return_tensors="pt").to(device)
        ans_prompt_len = ans_inputs.input_ids.shape[1]
        
        t1 = time.time()
        out_ans = model.generate(
            **ans_inputs,
            max_new_tokens=max_ans_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
        t_ans_s = time.time() - t1
        
    ans_tokens = out_ans[0][ans_prompt_len:].tolist()
    is_truncated = bool(len(ans_tokens) >= max_ans_tokens and out_ans[0][-1].item() != tokenizer.eos_token_id)
    content = tokenizer.decode(ans_tokens, skip_special_tokens=True).strip()
    pred_val, pred_raw = extract_answer(content, content)
    return {
        "think_tokens": len(think_tokens),
        "ans_tokens": len(ans_tokens),
        "think_time_ms": round(t_think_ms, 2),
        "total_time_s": round(time.time() - t0, 3),
        "is_truncated": is_truncated,
        "content": content,
        "pred_val": pred_val,
        "pred_raw": pred_raw
    }

def run_arm3_latent_autoregression(model, tokenizer, prompt, device, scale_factor, k_steps=6, max_ans_tokens=8192):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        # 1. Prefill Prompt
        out = model(input_ids=inputs.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        
        # 2. Sequential Latent Autoregression
        p_think_trajectory = []
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        
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
            
            # Log P(</think> | h_t) telemetry from already-computed logits (near-zero cost)
            step_probs = F.softmax(step_out.logits[:, -1, :], dim=-1)
            p_think = step_probs[0, think_end_id].item()
            p_think_trajectory.append(round(p_think, 6))
            
        torch.cuda.synchronize(device)
        t_think_ms = (time.time() - t0) * 1000.0
        
        # 3. Transition: Ingest </think>\n\n token embeddings
        t1 = time.time()
        trans_tokens = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
        trans_ids = torch.tensor([trans_tokens], dtype=torch.long, device=device)
        step_out = model(input_ids=trans_ids, past_key_values=past_kv, use_cache=True)
        past_kv = step_out.past_key_values
        
        # 4. Stochastic Answer Decoding (Exact Sampler Parity)
        curr_logits = step_out.logits[:, -1, :]
        next_tok = sample_token(curr_logits, temperature=0.6, top_p=0.95, top_k=20)
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        
        gen_tokens = [next_tok.item()]
        curr_tok = next_tok
        is_truncated = True
        
        for _ in range(max_ans_tokens):
            step_out = model(input_ids=curr_tok, past_key_values=past_kv, use_cache=True)
            past_kv = step_out.past_key_values
            curr_tok = sample_token(step_out.logits[:, -1, :], temperature=0.6, top_p=0.95, top_k=20)
            t_val = curr_tok.item()
            if t_val in [tokenizer.eos_token_id, im_end_id]:
                is_truncated = False
                break
            gen_tokens.append(t_val)
            
        torch.cuda.synchronize(device)
        total_time_s = time.time() - t0
        
    content = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    pred_val, pred_raw = extract_answer(content, content)
    return {
        "think_tokens": k_steps,
        "ans_tokens": len(gen_tokens),
        "think_time_ms": round(t_think_ms, 2),
        "total_time_s": round(total_time_s, 3),
        "is_truncated": is_truncated,
        "p_think_trajectory": p_think_trajectory,
        "content": content,
        "pred_val": pred_val,
        "pred_raw": pred_raw
    }

def run_arm4_unconstrained_cot(model, tokenizer, prompt, device, max_tokens=32768):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id
        )
    total_time = time.time() - t0
    output_ids = out[0][prompt_len:].tolist()
    total_tokens = len(output_ids)
    is_truncated = bool(total_tokens >= max_tokens and out[0][-1].item() != tokenizer.eos_token_id)
    
    try:
        idx = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        idx = 0
        
    thinking_content = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()
    full_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    pred_val, pred_raw = extract_answer(full_output, content, thinking_content)
    return {
        "think_tokens": idx,
        "ans_tokens": total_tokens - idx,
        "think_time_ms": round(total_time * 1000.0 * (idx / max(1, total_tokens)), 2),
        "total_time_s": round(total_time, 3),
        "is_truncated": is_truncated,
        "content": content,
        "pred_val": pred_val,
        "pred_raw": pred_raw
    }

def compute_hierarchical_bootstrap(problem_seed_matrix, n_boot=10000):
    # problem_seed_matrix shape: [num_problems, num_seeds] (binary 0/1)
    # 1. Compute per-problem mean across seeds
    prob_means = np.mean(problem_seed_matrix, axis=1)  # shape: [num_problems]
    n_probs = len(prob_means)
    
    # 2. Bootstrap over problems
    boot_means = []
    for _ in range(n_boot):
        idx = np.random.choice(n_probs, size=n_probs, replace=True)
        boot_means.append(np.mean(prob_means[idx]))
    boot_means = np.array(boot_means)
    
    mean_val = np.mean(prob_means) * 100.0
    ci_low = np.percentile(boot_means, 2.5) * 100.0
    ci_high = np.percentile(boot_means, 97.5) * 100.0
    return {
        "mean_pct": round(float(mean_val), 2),
        "ci_95_low": round(float(ci_low), 2),
        "ci_95_high": round(float(ci_high), 2)
    }

def compute_paired_arm_diff(matrix_a, matrix_b, n_boot=10000):
    # matrix shape: [num_problems, num_seeds]
    prob_means_a = np.mean(matrix_a, axis=1)
    prob_means_b = np.mean(matrix_b, axis=1)
    diffs = prob_means_a - prob_means_b
    n_probs = len(diffs)
    
    boot_diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(n_probs, size=n_probs, replace=True)
        boot_diffs.append(np.mean(diffs[idx]))
    boot_diffs = np.array(boot_diffs)
    
    mean_diff = np.mean(diffs) * 100.0
    ci_low = np.percentile(boot_diffs, 2.5) * 100.0
    ci_high = np.percentile(boot_diffs, 97.5) * 100.0
    p_val = np.mean(boot_diffs <= 0) if mean_diff > 0 else np.mean(boot_diffs >= 0)
    return {
        "mean_diff_pct": round(float(mean_diff), 2),
        "ci_95_low": round(float(ci_low), 2),
        "ci_95_high": round(float(ci_high), 2),
        "p_val": round(float(p_val), 4)
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-Sample Rigorous Benchmark Harness (Revision 8).")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora_path", type=str, default="checkpoints/lora_arm3_qwen_qwen3-1.7b_k6")
    parser.add_argument("--lora_arm1b_path", type=str, default="checkpoints/lora_arm1b_qwen_qwen3-1.7b_k0")
    parser.add_argument("--lora_arm2b_path", type=str, default="checkpoints/lora_arm2b_qwen_qwen3-1.7b_k6")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--suite_size", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789])
    parser.add_argument("--k_steps", type=int, default=6)
    parser.add_argument("--arms", type=str, nargs="+", default=["arm1", "arm1b", "arm2", "arm2b", "arm3"])
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    model_key = args.model_id.lower()
    scale_factor = CALIBRATED_ALPHAS.get(model_key, 0.011440)
    
    if args.output_file is None:
        tag = args.model_id.replace("/", "_").lower()
        args.output_file = os.path.join(os.path.dirname(__file__), "..", "data", f"rigorous_eval_{tag}_k{args.k_steps}.json")

    print(f"=== Multi-Sample Benchmark for {args.model_id} on {args.device} ===")
    print(f"Empirical Scale Factor: alpha={scale_factor:.6f}")
    print(f"Seeds ({len(args.seeds)}): {args.seeds} | Suite: {args.suite_size} problems")
    print(f"Selected Arms: {args.arms}")

    problems = load_benchmark_suite(args.suite_size)
    
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip("\"'\n")
                    break

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token, trust_remote_code=True)
    if PAUSE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})
    pause_token_id = tokenizer.convert_tokens_to_ids(PAUSE_TOKEN)

    print("Loading base model in pure bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.eval()

    arm_specs = {
        "arm1": ("arm1_direct", "Arm 1: Base Direct (No-Think)", "base", None, "arm1_base_direct"),
        "arm1b": ("arm1b_trained_direct", "Arm 1b: Trained Direct (No-CoT)", "adapter", args.lora_arm1b_path, "arm1b_trained_direct"),
        "arm2": ("arm2_minimal_discrete", f"Arm 2: Minimal Discrete (K={args.k_steps} tokens)", "base", None, "arm2_minimal_discrete"),
        "arm2b": ("arm2b_pause_tokens", f"Arm 2b: Pause-Token Control (K={args.k_steps} tokens)", "adapter", args.lora_arm2b_path, "arm2b_pause_tokens"),
        "arm3": ("arm3_latent_recurrent", f"Arm 3: Continuous Latent Recurrence (K={args.k_steps} loops)", "adapter", args.lora_path, "arm3_latent_recurrent"),
        "arm4": ("arm4_unconstrained_cot", "Arm 4: Unconstrained Discrete CoT (max 32k)", "base", None, "arm4_unconstrained_cot")
    }

    active_arms = []
    for a in args.arms:
        if a in arm_specs:
            active_arms.append(arm_specs[a])
        else:
            print(f"Warning: Unknown arm '{a}' ignored.")

    results_data = {
        "model_id": args.model_id,
        "device": args.device,
        "scale_factor": scale_factor,
        "suite_size": args.suite_size,
        "seeds": args.seeds,
        "k_steps": args.k_steps,
        "arms": {}
    }

    arm_matrices = {}

    for arm_id, arm_label, model_type, adapter_path, arm_func_type in active_arms:
        print(f"\n{'='*60}\nPreparing {arm_label}\n{'='*60}")
        if model_type == "adapter":
            if adapter_path and os.path.exists(adapter_path):
                print(f"Attaching adapter from {adapter_path}...")
                active_model = PeftModel.from_pretrained(base_model, adapter_path)
                active_model.eval()
            else:
                print(f"Warning: Adapter {adapter_path} not found! Skipping {arm_label}.")
                continue
        else:
            active_model = base_model

        perf_matrix = np.zeros((len(problems), len(args.seeds)), dtype=np.float32)
        trunc_matrix = np.zeros((len(problems), len(args.seeds)), dtype=np.float32)
        latencies = []
        think_tok_list = []
        ans_tok_list = []
        p_think_correct = []
        p_think_wrong = []
        detailed_runs = []

        tag = args.model_id.replace("/", "_").lower()
        streaming_log_path = os.path.join(
            os.path.dirname(args.output_file or "data/benchmark_results.json"),
            f"streaming_{arm_id}_{tag}.jsonl"
        )
        os.makedirs(os.path.dirname(streaming_log_path), exist_ok=True)
        with open(streaming_log_path, "w") as f_stream:
            pass

        for p_idx, prob in enumerate(problems):
            q = prob["question"]
            sol = prob["solution"]
            gt_boxed = extract_math_boxed_expression(sol)
            gt_val = clean_and_extract_candidate(gt_boxed or sol)

            for s_idx, seed in enumerate(args.seeds):
                set_seed(seed)
                if arm_func_type == "arm1_base_direct":
                    run = run_arm1_base_direct(active_model, tokenizer, q, args.device)
                elif arm_func_type == "arm1b_trained_direct":
                    run = run_arm1b_trained_direct(active_model, tokenizer, q, args.device)
                elif arm_func_type == "arm2_minimal_discrete":
                    run = run_arm2_minimal_discrete(active_model, tokenizer, q, args.device, k_tokens=args.k_steps)
                elif arm_func_type == "arm2b_pause_tokens":
                    run = run_arm2b_pause_tokens(active_model, tokenizer, pause_token_id, q, args.device, k_tokens=args.k_steps)
                elif arm_func_type == "arm3_latent_recurrent":
                    run = run_arm3_latent_autoregression(active_model, tokenizer, q, args.device, scale_factor, k_steps=args.k_steps)
                elif arm_func_type == "arm4_unconstrained_cot":
                    run = run_arm4_unconstrained_cot(active_model, tokenizer, q, args.device)
                    
                is_correct = check_match(
                    run["pred_val"],
                    run["pred_raw"],
                    gt_val,
                    gt_boxed or sol,
                    full_pred_text=run.get("content_text", ""),
                    full_gold_text=sol
                )
                if run["is_truncated"]:
                    is_correct = False
                    trunc_matrix[p_idx, s_idx] = 1.0

                if "p_think_trajectory" in run and len(run["p_think_trajectory"]) > 0:
                    if is_correct:
                        p_think_correct.append(run["p_think_trajectory"])
                    else:
                        p_think_wrong.append(run["p_think_trajectory"])

                perf_matrix[p_idx, s_idx] = 1.0 if is_correct else 0.0
                latencies.append(run["total_time_s"])
                think_tok_list.append(run["think_tokens"])
                ans_tok_list.append(run["ans_tokens"])

                record = {
                    "problem_id": prob["id"],
                    "benchmark": prob.get("benchmark", "Unknown"),
                    "seed": seed,
                    "is_correct": is_correct,
                    "gt_val": gt_val,
                    "gt_boxed": gt_boxed,
                    "run": run
                }
                detailed_runs.append(record)
                with open(streaming_log_path, "a") as f_stream:
                    f_stream.write(json.dumps(sanitize_for_json(record), default=str) + "\n")

            if (p_idx + 1) % 25 == 0 or (p_idx + 1) == len(problems):
                curr_acc = np.mean(perf_matrix[:p_idx+1]) * 100.0
                curr_trunc = np.mean(trunc_matrix[:p_idx+1]) * 100.0
                print(f"  [{arm_id} | {p_idx+1}/{len(problems)}] Acc: {curr_acc:.2f}% | Trunc: {curr_trunc:.2f}%")

        if model_type == "adapter":
            base_model = active_model.unload()

        stats = compute_hierarchical_bootstrap(perf_matrix, n_boot=10000)
        trunc_rate = float(np.mean(trunc_matrix) * 100.0)
        mean_lat = float(np.mean(latencies))
        p25, p50, p75, p90, p99 = np.percentile(think_tok_list, [25, 50, 75, 90, 99]).tolist()

        arm_summary = {
            "label": arm_label,
            "accuracy_bootstrap": stats,
            "truncation_rate_pct": round(trunc_rate, 2),
            "mean_latency_s": round(mean_lat, 3),
            "think_token_percentiles": {
                "p25": round(p25, 1),
                "p50": round(p50, 1),
                "p75": round(p75, 1),
                "p90": round(p90, 1),
                "p99": round(p99, 1)
            },
            "mean_ans_tokens": round(float(np.mean(ans_tok_list)), 1),
            "detailed_runs": detailed_runs
        }
        if p_think_correct or p_think_wrong:
            arm_summary["telemetry_p_think"] = {
                "mean_curve_correct": [round(float(x), 6) for x in np.mean(p_think_correct, axis=0)] if len(p_think_correct) > 0 else [],
                "mean_curve_wrong": [round(float(x), 6) for x in np.mean(p_think_wrong, axis=0)] if len(p_think_wrong) > 0 else [],
                "n_correct": len(p_think_correct),
                "n_wrong": len(p_think_wrong)
            }
        results_data["arms"][arm_id] = arm_summary
        arm_matrices[arm_id] = perf_matrix

        print(f"\n--- Summary for {arm_label} ---")
        print(f"Pass@1 (Bootstrap): {stats['mean_pct']}% (95% CI: [{stats['ci_95_low']}%, {stats['ci_95_high']}%])")
        print(f"Truncation Rate:    {trunc_rate:.2f}%")
        print(f"Mean Latency:       {mean_lat:.3f}s")
        print(f"Thinking Lengths:   P50={p50:.0f}, P90={p90:.0f}, P99={p99:.0f}")

    # Paired Statistical Differences
    print("\n" + "="*70 + "\nPAIRED ARM-VS-ARM STATISTICAL COMPARISONS\n" + "="*70)
    paired_comparisons = {}

    if "arm3_latent_recurrent" in arm_matrices:
        m3 = arm_matrices["arm3_latent_recurrent"]
        
        # Primary Hypothesis 1: Arm 3 vs Arm 1b (Latent vs. Trained Direct)
        if "arm1b_trained_direct" in arm_matrices:
            diff_3_vs_1b = compute_paired_arm_diff(m3, arm_matrices["arm1b_trained_direct"])
            paired_comparisons["arm3_vs_arm1b"] = diff_3_vs_1b
            p1_pass = (diff_3_vs_1b["mean_diff_pct"] > 0 and diff_3_vs_1b["p_val"] < 0.05)
            print(f"Arm 3 vs Arm 1b (Latent vs. Trained Direct):     {diff_3_vs_1b['mean_diff_pct']:+.2f}% (95% CI: [{diff_3_vs_1b['ci_95_low']}%, {diff_3_vs_1b['ci_95_high']}%], p={diff_3_vs_1b['p_val']}) -> {'PASSED' if p1_pass else 'NOT SIGNIFICANT'}")

        # Primary Hypothesis 2: Arm 3 vs Arm 2b (Latent vs. Pause Tokens)
        if "arm2b_pause_tokens" in arm_matrices:
            diff_3_vs_2b = compute_paired_arm_diff(m3, arm_matrices["arm2b_pause_tokens"])
            paired_comparisons["arm3_vs_arm2b"] = diff_3_vs_2b
            p2_pass = (diff_3_vs_2b["mean_diff_pct"] > 0 and diff_3_vs_2b["p_val"] < 0.05)
            print(f"Arm 3 vs Arm 2b (Latent vs. Pause-Token Control): {diff_3_vs_2b['mean_diff_pct']:+.2f}% (95% CI: [{diff_3_vs_2b['ci_95_low']}%, {diff_3_vs_2b['ci_95_high']}%], p={diff_3_vs_2b['p_val']}) -> {'PASSED' if p2_pass else 'NOT SIGNIFICANT'}")

        # Secondary: Arm 3 vs Arm 1 (Base Direct)
        if "arm1_direct" in arm_matrices:
            diff_3_vs_1 = compute_paired_arm_diff(m3, arm_matrices["arm1_direct"])
            paired_comparisons["arm3_vs_arm1"] = diff_3_vs_1
            print(f"Arm 3 vs Arm 1  (Latent vs. Base Direct):          {diff_3_vs_1['mean_diff_pct']:+.2f}% (95% CI: [{diff_3_vs_1['ci_95_low']}%, {diff_3_vs_1['ci_95_high']}%], p={diff_3_vs_1['p_val']})")

        # Secondary: Arm 3 vs Arm 2 (Minimal Discrete)
        if "arm2_minimal_discrete" in arm_matrices:
            diff_3_vs_2 = compute_paired_arm_diff(m3, arm_matrices["arm2_minimal_discrete"])
            paired_comparisons["arm3_vs_arm2"] = diff_3_vs_2
            print(f"Arm 3 vs Arm 2  (Latent vs. FLOP-Matched Discrete):{diff_3_vs_2['mean_diff_pct']:+.2f}% (95% CI: [{diff_3_vs_2['ci_95_low']}%, {diff_3_vs_2['ci_95_high']}%], p={diff_3_vs_2['p_val']})")

        # Ceiling: Arm 3 vs Arm 4 (Unconstrained CoT)
        if "arm4_unconstrained_cot" in arm_matrices:
            diff_3_vs_4 = compute_paired_arm_diff(m3, arm_matrices["arm4_unconstrained_cot"])
            paired_comparisons["arm3_vs_arm4"] = diff_3_vs_4
            print(f"Arm 3 vs Arm 4  (Latent vs. Unconstrained CoT):   {diff_3_vs_4['mean_diff_pct']:+.2f}% (95% CI: [{diff_3_vs_4['ci_95_low']}%, {diff_3_vs_4['ci_95_high']}%], p={diff_3_vs_4['p_val']})")

    results_data["paired_comparisons"] = paired_comparisons

    # Decision Gate 1 Evaluation
    gate_1_passed = False
    if "arm3_vs_arm1b" in paired_comparisons and "arm3_vs_arm2b" in paired_comparisons:
        g1_pass1 = (paired_comparisons["arm3_vs_arm1b"]["mean_diff_pct"] > 0 and paired_comparisons["arm3_vs_arm1b"]["p_val"] < 0.05)
        g1_pass2 = (paired_comparisons["arm3_vs_arm2b"]["mean_diff_pct"] > 0 and paired_comparisons["arm3_vs_arm2b"]["p_val"] < 0.05)
        gate_1_passed = g1_pass1 and g1_pass2
        print("\n" + "="*70)
        print("DECISION GATE 1 VERDICT")
        print("="*70)
        print(f"Criterion 1: Delta(Arm 3 - Arm 1b) > 0 (p < 0.05): {g1_pass1} ({paired_comparisons['arm3_vs_arm1b']['mean_diff_pct']:+.2f}%, p={paired_comparisons['arm3_vs_arm1b']['p_val']})")
        print(f"Criterion 2: Delta(Arm 3 - Arm 2b) > 0 (p < 0.05): {g1_pass2} ({paired_comparisons['arm3_vs_arm2b']['mean_diff_pct']:+.2f}%, p={paired_comparisons['arm3_vs_arm2b']['p_val']})")
        print(f"Decision Gate 1 Result: {'PASSED' if gate_1_passed else 'FAILED / INCONCLUSIVE'}")
        print("="*70)

    results_data["gate_1_passed"] = gate_1_passed

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(sanitize_for_json(results_data), f, indent=2, default=str)
    print(f"\nSaved complete benchmark results to: {args.output_file}")

if __name__ == "__main__":
    main()
