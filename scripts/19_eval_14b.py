#!/usr/bin/env python3
"""
19_eval_14b.py
Comprehensive evaluation of DeepSeek-R1-Distill-Qwen-14B (8-bit) across 25 benchmark problems:
- Discrete Baseline CoT
- Fixed K=6 Recurrent LoRA
- Trigger A: Native </think> probability (tau=0.95)
- Trigger B: Geometric convergence (cos >= 0.992 or rel_L2 <= 0.10)
- Trigger C: Dynamic Hybrid (P >= 0.90 AND convergence)
- Trigger D: Auxiliary Probe Entropy Stabilization (|Delta H| <= 0.20 AND P >= 0.90)
- Think Harder: Native </think> logit suppression (steps 1..6)

Saves to data/eval_results_14b.json.
Runs on cuda:1.
"""

import os
import re
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEVICE = "cuda:1"
LORA_PATH = "checkpoints/lora_recurrent_14b"
PROBE_PATH = "checkpoints/thought_probe_14b.pt"
OUTPUT_FILE = "data/eval_results_14b.json"

class ThoughtStreamProbe(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size, bias=False)
        )
    def forward(self, h):
        return self.net(h)

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

def normalize_math_str(s):
    if not s:
        return ""
    s = s.strip()
    boxed = extract_math_boxed_expression(s)
    if boxed:
        s = boxed
    if "=" in s:
        s = s.split("=")[-1].strip()
    s = re.sub(r"\\(?:text|mathbf|mathrm)\{([^}]+)\}", r"\1", s)
    s = re.sub(r"[\$\s]", "", s)
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"^{\circ}", "").replace(r"^\circ", "")
    return s.rstrip(".")

def try_eval_numeric(s):
    norm = normalize_math_str(s)
    if not norm:
        return None
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", norm)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", norm)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(norm)
    except ValueError:
        return None

def math_equal(pred_str, gt_str):
    if not pred_str or not gt_str:
        return False
    p_norm = normalize_math_str(pred_str)
    g_norm = normalize_math_str(gt_str)
    if p_norm == g_norm:
        return True
    p_num = try_eval_numeric(p_norm)
    g_num = try_eval_numeric(g_norm)
    if p_num is not None and g_num is not None:
        return abs(p_num - g_num) < 1e-4
    return False

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
    bold_matches = re.findall(r"\*\*(?:[A-Za-z\s:]*)?([\$]?[-\d.,]+)(?:[A-Za-z\s]*)\*\*", text)
    if bold_matches:
        val = clean_and_extract_candidate(bold_matches[-1])
        if val is not None:
            return val
    ans_match = re.findall(r'(?:the answer is|final answer is|total is|equals|equal to|in total|profit:)\s*([\$]?[-\d.,]+)', text, re.IGNORECASE)
    if ans_match:
        val = clean_and_extract_candidate(ans_match[-1])
        if val is not None:
            return val
    if is_answer_only:
        nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", text)
        if nums:
            val = clean_and_extract_candidate(nums[-1])
            if val is not None:
                return val
    return None

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

def get_gsm8k_stratified(num_per_tier=5):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    easy, med, hard = [], [], []
    for i, item in enumerate(ds):
        ops = re.findall(r"<<.*?>>", item["answer"])
        op_count = len(ops)
        if op_count <= 2 and len(easy) < num_per_tier:
            easy.append({"id": f"gsm8k_easy_{i}", "ds_idx": i, "category": "Easy", "op_count": op_count, "question": item["question"], "answer": item["answer"]})
        elif 3 <= op_count <= 4 and len(med) < num_per_tier:
            med.append({"id": f"gsm8k_med_{i}", "ds_idx": i, "category": "Medium", "op_count": op_count, "question": item["question"], "answer": item["answer"]})
        elif op_count >= 5 and len(hard) < num_per_tier:
            hard.append({"id": f"gsm8k_hard_{i}", "ds_idx": i, "category": "Hard", "op_count": op_count, "question": item["question"], "answer": item["answer"]})
        if len(easy) >= num_per_tier and len(med) >= num_per_tier and len(hard) >= num_per_tier:
            break
    return easy + med + hard

def get_math500_hard(num_samples=10):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    hard_samples = []
    for i, item in enumerate(ds):
        if item["level"] in [4, 5]:
            hard_samples.append({
                "id": f"math500_lvl{item['level']}_{i}",
                "ds_idx": i,
                "category": f"MATH-L{item['level']}",
                "level": item["level"],
                "subject": item["subject"],
                "question": item["problem"],
                "answer": item["answer"]
            })
            if len(hard_samples) >= num_samples:
                break
    return hard_samples

def run_discrete_baseline(model, tokenizer, question, max_baseline_tokens=2048):
    messages = [{"role": "user", "content": question}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    
    torch.cuda.synchronize()
    t0 = time.time()
    
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_baseline_tokens,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    torch.cuda.synchronize()
    total_time = time.time() - t0
    
    gen_ids = gen_out[0][inputs.input_ids.shape[1]:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    
    if "</think>" in gen_text:
        parts = gen_text.split("</think>")
        think_text = parts[0].strip()
        answer_text = parts[1].strip()
        think_tokens = len(tokenizer.encode(think_text, add_special_tokens=False))
        ans_tokens = len(tokenizer.encode(answer_text, add_special_tokens=False))
        total_tokens = max(1, think_tokens + ans_tokens)
        think_time_ms = (think_tokens / total_tokens) * total_time * 1000.0
        pred_num = extract_numeric_answer(answer_text, is_answer_only=True)
    else:
        think_tokens = len(gen_ids)
        ans_tokens = 0
        think_time_ms = total_time * 1000.0
        answer_text = ""
        pred_num = None
        
    return {
        "mode": "discrete_baseline",
        "think_time_ms": round(think_time_ms, 2),
        "total_time_s": round(total_time, 3),
        "think_tokens": think_tokens,
        "answer_tokens": ans_tokens,
        "predicted_num": pred_num,
        "raw_text": gen_text
    }

def run_recurrent_step(
    model,
    tokenizer,
    probe,
    question,
    scale_factor,
    think_end_id,
    k_mode="dynamic",
    k_fixed=6,
    k_min=2,
    k_max=16,
    trigger_type="hybrid",
    tau_think=0.90,
    tau_cos=0.992,
    tau_l2=0.10,
    tau_entropy=0.20,
    suppress_think_steps=0,
    max_new_tokens=512
):
    messages = [{"role": "user", "content": question}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    
    torch.cuda.synchronize()
    t0 = time.time()
    
    with torch.no_grad():
        out = model(inputs.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        prev_latent = curr_latent
        prev_entropy = None
        
        telemetry = []
        actual_steps = 0
        halt_reason = "max_steps"
        target_max = k_fixed if k_mode == "fixed" else k_max
        
        torch.cuda.synchronize()
        rec_t0 = time.time()
        
        for step in range(1, target_max + 1):
            actual_steps = step
            latent_in = curr_latent
            scaled_latent = (latent_in * scale_factor).to(torch.float16)
            step_out = model(
                inputs_embeds=scaled_latent,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = step_out.past_key_values
            next_latent = step_out.hidden_states[-1][:, -1:, :]
            
            cos_sim = min(1.0, F.cosine_similarity(prev_latent.squeeze(1).float(), next_latent.squeeze(1).float(), dim=-1).item())
            l2_diff = torch.norm(next_latent.float() - prev_latent.float(), p=2).item()
            norm_next = torch.norm(next_latent.float(), p=2).item()
            rel_l2 = l2_diff / (norm_next + 1e-9)
            
            logits = model.lm_head(next_latent)[0, -1]
            if step <= suppress_think_steps:
                logits = logits.clone()
                logits[think_end_id] -= 10.0
                
            probs = torch.softmax(logits, dim=-1)
            p_think = probs[think_end_id].item()
            top1_val, top1_idx = torch.topk(probs, 1)
            
            if probe is not None:
                p_logits = probe(next_latent.to(torch.bfloat16))
                p_probs = torch.softmax(p_logits[0, -1], dim=-1)
                entropy = -torch.sum(p_probs * torch.log(p_probs + 1e-9)).item()
                delta_entropy = abs(entropy - prev_entropy) if prev_entropy is not None else 999.0
                prev_entropy = entropy
            else:
                entropy = 0.0
                delta_entropy = 0.0
                
            telemetry.append({
                "step": step,
                "p_think": round(p_think, 4),
                "cos_sim": round(cos_sim, 4),
                "rel_l2": round(rel_l2, 4),
                "entropy": round(entropy, 2),
                "delta_entropy": round(delta_entropy, 2),
                "top1_token": tokenizer.decode([top1_idx[0].item()]).replace("\n", "\\n")
            })
            
            prev_latent = next_latent
            curr_latent = next_latent
            
            if k_mode == "dynamic" and step >= k_min:
                converged = (cos_sim >= tau_cos or rel_l2 <= tau_l2)
                think_ready = (p_think >= tau_think)
                
                if trigger_type == "trigger_a_logit" and think_ready:
                    halt_reason = "trigger_a_think_prob"
                    break
                elif trigger_type == "trigger_b_conv" and converged:
                    halt_reason = "trigger_b_convergence"
                    break
                elif trigger_type == "hybrid" and (think_ready and converged):
                    halt_reason = "hybrid_think_and_conv"
                    break
                elif trigger_type == "entropy_stabilized" and think_ready and delta_entropy <= tau_entropy:
                    halt_reason = "entropy_stabilized"
                    break
                    
        torch.cuda.synchronize()
        think_time_ms = (time.time() - rec_t0) * 1000.0
        
        next_token = torch.argmax(model.lm_head(curr_latent)[0, -1]).unsqueeze(0)
        generated_tokens = [next_token.item()]
        
        for _ in range(max_new_tokens):
            if next_token.item() == tokenizer.eos_token_id:
                break
            step_out = model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_kv,
                use_cache=True
            )
            past_kv = step_out.past_key_values
            next_token = torch.argmax(step_out.logits[0, -1]).unsqueeze(0)
            generated_tokens.append(next_token.item())
            
        torch.cuda.synchronize()
        total_time = time.time() - t0
        
    full_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    clean_text = full_text.replace("</think>", "").replace("<｜end of sentence｜>", "").strip()
    is_finished = (tokenizer.eos_token in full_text) or len(generated_tokens) < max_new_tokens
    pred_num = extract_numeric_answer(clean_text, is_answer_only=is_finished)
    
    return {
        "think_time_ms": round(think_time_ms, 2),
        "total_time_s": round(total_time, 3),
        "think_tokens": 0,
        "answer_tokens": len(generated_tokens),
        "predicted_num": pred_num,
        "actual_steps": actual_steps,
        "halt_reason": halt_reason,
        "telemetry": telemetry,
        "raw_text": full_text
    }

def compute_stats(runs, name):
    total = len(runs)
    correct = sum(1 for r in runs if r["correct"])
    gsm_runs = [r for r in runs if "gsm8k" in r["question_id"]]
    math_runs = [r for r in runs if "math500" in r["question_id"]]
    
    easy_runs = [r for r in runs if r["category"] == "Easy"]
    med_runs = [r for r in runs if r["category"] == "Medium"]
    hard_runs = [r for r in runs if r["category"] == "Hard"]
    
    return {
        "name": name,
        "overall_accuracy": round(correct / total * 100.0, 2) if total else 0.0,
        "gsm8k_accuracy": round(sum(1 for r in gsm_runs if r["correct"]) / len(gsm_runs) * 100.0, 2) if gsm_runs else 0.0,
        "gsm8k_easy_acc": round(sum(1 for r in easy_runs if r["correct"]) / len(easy_runs) * 100.0, 2) if easy_runs else 0.0,
        "gsm8k_med_acc": round(sum(1 for r in med_runs if r["correct"]) / len(med_runs) * 100.0, 2) if med_runs else 0.0,
        "gsm8k_hard_acc": round(sum(1 for r in hard_runs if r["correct"]) / len(hard_runs) * 100.0, 2) if hard_runs else 0.0,
        "math500_accuracy": round(sum(1 for r in math_runs if r["correct"]) / len(math_runs) * 100.0, 2) if math_runs else 0.0,
        "mean_think_time_ms": round(sum(r["think_time_ms"] for r in runs) / total, 2) if total else 0.0,
        "mean_total_time_s": round(sum(r["total_time_s"] for r in runs) / total, 3) if total else 0.0,
        "mean_steps": round(sum(r.get("actual_steps", 0) for r in runs) / total, 2) if total else 0.0,
        "steps_by_tier": {
            "easy": round(sum(r.get("actual_steps", 0) for r in easy_runs) / len(easy_runs), 2) if easy_runs else 0.0,
            "medium": round(sum(r.get("actual_steps", 0) for r in med_runs) / len(med_runs), 2) if med_runs else 0.0,
            "hard": round(sum(r.get("actual_steps", 0) for r in hard_runs) / len(hard_runs), 2) if hard_runs else 0.0,
            "math500": round(sum(r.get("actual_steps", 0) for r in math_runs) / len(math_runs), 2) if math_runs else 0.0
        }
    }

def main():
    print(f"=== Stage 4: Evaluating 14B Model (Discrete vs Recurrent) on {DEVICE} ===")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]
    
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    print(f"Loading base model {MODEL_ID} in 8-bit on {DEVICE}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=DEVICE,
        torch_dtype=torch.float16
    )
    base_model.eval()
    
    embed_weights = base_model.get_input_embeddings().weight.detach()
    avg_embed_norm = torch.norm(embed_weights.float(), dim=-1).mean().item()
    hidden_dim = base_model.config.hidden_size
    scale_factor = avg_embed_norm / (hidden_dim ** 0.5)
    print(f"Calculated scale factor alpha_14B: {scale_factor:.6f}")
    
    probe = None
    if os.path.exists(PROBE_PATH):
        print(f"Loading auxiliary thought probe from {PROBE_PATH}...")
        probe = ThoughtStreamProbe(hidden_dim, base_model.config.vocab_size).to(DEVICE, dtype=torch.bfloat16)
        probe.load_state_dict(torch.load(PROBE_PATH, map_location=DEVICE))
        probe.eval()
        
    gsm_samples = get_gsm8k_stratified(num_per_tier=5)
    math_samples = get_math500_hard(num_samples=10)
    eval_items = gsm_samples + math_samples
    print(f"Loaded {len(eval_items)} benchmark items (15 GSM8K + 10 MATH-500).")
    
    summary_dict = {}
    detailed_dict = {}
    
    # 1. DISCRETE BASELINE COT
    print("\n" + "="*90)
    print("CONDITION 1: DISCRETE BASELINE COT (DeepSeek-R1-Distill-Qwen-14B Teacher)")
    print("="*90)
    discrete_runs = []
    for idx, item in enumerate(eval_items):
        q = item["question"]
        is_math = "math500" in item["id"]
        gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
        print(f"[{idx+1}/{len(eval_items)}] [{item['category']}] Discrete CoT: {q[:55]}... (GT: {gt_num or item['answer'][:15]})")
        res = run_discrete_baseline(base_model, tokenizer, q, max_baseline_tokens=2048)
        is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
        res["question_id"] = item["id"]
        res["category"] = item["category"]
        res["ground_truth_num"] = gt_num
        res["ground_truth_raw"] = item["answer"]
        res["correct"] = is_correct
        discrete_runs.append(res)
        print(f"      -> Pred: {res['predicted_num']} | Correct: {is_correct} | ThinkTime: {res['think_time_ms']:.1f}ms | TotTime: {res['total_time_s']:.2f}s")
        
    detailed_dict["discrete_baseline"] = discrete_runs
    summary_dict["discrete_baseline"] = compute_stats(discrete_runs, "Discrete Baseline CoT")
    
    # 2. LOAD PEFT LORA ADAPTER
    print("\n" + "="*90)
    print("LOADING PEFT LORA ADAPTER...")
    print("="*90)
    lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    lora_model.eval()
    print(f"Loaded LoRA adapter from {LORA_PATH} successfully.")
    
    conditions = [
        ("fixed_k6_recurrent", "Fixed K=6 Recurrent", {"k_mode": "fixed", "k_fixed": 6}),
        ("trigger_a_logit", "Trigger A (Logit/P_think >= 0.95)", {"k_mode": "dynamic", "trigger_type": "trigger_a_logit", "tau_think": 0.95}),
        ("trigger_b_conv", "Trigger B (Geometric Conv)", {"k_mode": "dynamic", "trigger_type": "trigger_b_conv", "tau_cos": 0.992, "tau_l2": 0.10}),
        ("dynamic_hybrid", "Dynamic Hybrid (Trigger C)", {"k_mode": "dynamic", "trigger_type": "hybrid", "tau_think": 0.90, "tau_cos": 0.992, "tau_l2": 0.10}),
        ("trigger_d_entropy", "Trigger D (Entropy Stabilized)", {"k_mode": "dynamic", "trigger_type": "entropy_stabilized", "tau_think": 0.90, "tau_entropy": 0.20}),
        ("think_harder_suppress", "Think Harder (Logit Suppress steps 1..6)", {"k_mode": "dynamic", "trigger_type": "hybrid", "tau_think": 0.90, "suppress_think_steps": 6, "k_min": 8, "k_max": 16})
    ]
    
    for cond_key, cond_name, cond_kwargs in conditions:
        print("\n" + "="*90)
        print(f"EVALUATING CONDITION: {cond_name}")
        print("="*90)
        cond_runs = []
        for idx, item in enumerate(eval_items):
            q = item["question"]
            is_math = "math500" in item["id"]
            gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
            res = run_recurrent_step(
                lora_model, tokenizer, probe, q,
                scale_factor=scale_factor,
                think_end_id=think_end_id,
                **cond_kwargs
            )
            is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
            res["question_id"] = item["id"]
            res["category"] = item["category"]
            res["ground_truth_num"] = gt_num
            res["ground_truth_raw"] = item["answer"]
            res["correct"] = is_correct
            cond_runs.append(res)
            print(f"[{idx+1}/{len(eval_items)}] [{item['category']}] -> Pred: {res['predicted_num']} | Correct: {is_correct} | Steps: {res['actual_steps']} ({res['halt_reason']}) | ThinkTime: {res['think_time_ms']:.1f}ms")
            
        detailed_dict[cond_key] = cond_runs
        summary_dict[cond_key] = compute_stats(cond_runs, cond_name)
        
    disc_think_ms = summary_dict["discrete_baseline"]["mean_think_time_ms"]
    for k, s in summary_dict.items():
        if k == "discrete_baseline":
            s["speedup_vs_discrete"] = 1.0
        else:
            s["speedup_vs_discrete"] = round(disc_think_ms / max(1e-5, s["mean_think_time_ms"]), 2)
            
    output_data = {
        "metadata": {
            "model_id": MODEL_ID,
            "device": DEVICE,
            "scale_factor": scale_factor,
            "precision": "8-bit",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "summary": summary_dict,
        "detailed_runs": detailed_dict
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    print("\n" + "="*110)
    print("14B BENCHMARK EVALUATION SUMMARY TABLE")
    print("="*110)
    print(f"{'Condition':<35} | {'Overall':<7} | {'GSM8K':<7} | {'MATH500':<7} | {'Think(ms)':<10} | {'Speedup':<8} | {'Steps':<6}")
    print("-" * 110)
    for k, s in summary_dict.items():
        print(f"{s['name']:<35} | {s['overall_accuracy']:>6.1f}% | {s['gsm8k_accuracy']:>6.1f}% | {s['math500_accuracy']:>6.1f}% | {s['mean_think_time_ms']:>9.1f} | {s['speedup_vs_discrete']:>7.1f}x | {s['mean_steps']:>5.1f}")
    print("="*110)
    print(f"Full results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
