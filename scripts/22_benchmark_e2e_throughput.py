#!/usr/bin/env python3
"""
22_benchmark_e2e_throughput.py
Comprehensive Latent Recurrence Benchmark measuring:
1. Pure End-to-End Latency:
   Total wall-clock time from prompt input to the final EOS token
   (t_total = prefill + thinking + answer decoding) with strict torch.cuda.synchronize()
   at start and end.
2. Total Tokens/s:
   - Decoding throughput: num_answer_tokens / answer_generation_time (tok/s)
   - Effective E2E throughput: num_answer_tokens / total_e2e_time (tok/s)
3. Best version of "Think Harder":
   - Logit Suppression: penalizing </think> by -10.0 for steps 1..6, enforcing K_min = 8, K_max = 16
   - Coupled with Trigger D: Auxiliary Probe Entropy Stabilization (|Delta H| <= 0.20 AND P(</think>) >= 0.90)
   Evaluated alongside Fixed K=6 and Discrete Baseline across all models with max_new_tokens = 2048.
4. Models supported:
   - 1.5B: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (BF16)
   - 7B:   deepseek-ai/DeepSeek-R1-Distill-Qwen-7B (BF16)
   - 14B:  deepseek-ai/DeepSeek-R1-Distill-Qwen-14B (8-bit BNB)
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
import transformers.modeling_utils
# Bypass Transformers 5.17 aggressive caching_allocator_warmup which tries to allocate 30GB extra on device
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_CONFIGS = {
    "1.5b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "precision": "bfloat16",
        "default_device": "cuda:0",
        "lora_path": "checkpoints/lora_recurrent_1.5b",
        "probe_path": "checkpoints/thought_probe.pt",
        "output_file": "data/e2e_eval_results_1.5b_2048.json",
        "prev_results": "data/dynamic_halting_results.json"
    },
    "7b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "precision": "bfloat16",
        "default_device": "cuda:1",
        "lora_path": "checkpoints/lora_recurrent_7b",
        "probe_path": "checkpoints/thought_probe_7b.pt",
        "output_file": "data/e2e_eval_results_7b_2048.json",
        "prev_results": "data/eval_results_7b.json"
    },
    "14b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "precision": "bfloat16",
        "default_device": "cuda:1",
        "lora_path": "checkpoints/lora_recurrent_14b",
        "probe_path": "checkpoints/thought_probe_14b.pt",
        "output_file": "data/e2e_eval_results_14b_2048.json",
        "prev_results": "data/eval_results_14b.json"
    }
}

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

def run_discrete_baseline(model, tokenizer, question, device, max_baseline_tokens=2048):
    """
    Measures discrete baseline with strict torch.cuda.synchronize() at the start and end.
    Calculates pure E2E latency, answer decoding time, and decoding / effective E2E throughput.
    """
    messages = [{"role": "user", "content": question}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        gen_out = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_baseline_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize(device)
    total_time_s = time.time() - t0
    
    gen_ids = gen_out[0][inputs.input_ids.shape[1]:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    
    if "</think>" in gen_text:
        parts = gen_text.split("</think>")
        think_text = parts[0].strip()
        answer_text = parts[1].strip()
        think_tokens = len(tokenizer.encode(think_text, add_special_tokens=False))
        ans_tokens = len(tokenizer.encode(answer_text, add_special_tokens=False))
        total_tokens = max(1, think_tokens + ans_tokens)
        think_time_ms = (think_tokens / total_tokens) * total_time_s * 1000.0
        ans_time_s = max(1e-5, total_time_s - (think_time_ms / 1000.0))
        decoding_tok_s = ans_tokens / ans_time_s
        e2e_tok_s = ans_tokens / total_time_s
        pred_num = extract_numeric_answer(answer_text, is_answer_only=True)
    else:
        think_tokens = len(gen_ids)
        ans_tokens = 0
        think_time_ms = total_time_s * 1000.0
        ans_time_s = 0.0
        decoding_tok_s = 0.0
        e2e_tok_s = 0.0
        pred_num = None
        
    return {
        "mode": "discrete_baseline",
        "think_time_ms": round(think_time_ms, 2),
        "answer_time_s": round(ans_time_s, 3),
        "total_e2e_time_s": round(total_time_s, 3),
        "think_tokens": think_tokens,
        "answer_tokens": ans_tokens,
        "decoding_tok_s": round(decoding_tok_s, 2),
        "e2e_tok_s": round(e2e_tok_s, 2),
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
    device,
    k_mode="fixed",
    k_fixed=6,
    k_min=8,
    k_max=16,
    trigger_type="entropy_stabilized",
    tau_think=0.90,
    tau_cos=0.992,
    tau_l2=0.10,
    tau_entropy=0.20,
    prompt_prefix="",
    suppress_think_steps=0,
    noise_sigma=0.0,
    max_new_tokens=2048
):
    """
    Executes continuous latent recurrence with strict torch.cuda.synchronize() timing:
    - Pure E2E latency: start of prefill to final EOS token
    - Thinking latency: start to completion of recurrent loop
    - Answer decoding time: completion of thinking to final EOS token
    - Decoding throughput: num_answer_tokens / answer_time_s
    - Effective E2E throughput: num_answer_tokens / total_e2e_time_s
    """
    user_content = prompt_prefix + question if prompt_prefix else question
    messages = [{"role": "user", "content": user_content}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # Pure End-to-End Latency start
    torch.cuda.synchronize(device)
    t0 = time.time()
    
    with torch.no_grad():
        # 1. Prefill
        out = model(inputs.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        prev_latent = curr_latent
        prev_entropy = None
        
        telemetry = []
        actual_steps = 0
        halt_reason = "max_steps"
        
        target_max = k_fixed if k_mode == "fixed" else k_max
        
        # 2. Recurrent Thinking Phase
        torch.cuda.synchronize(device)
        rec_t0 = time.time()
        
        for step in range(1, target_max + 1):
            actual_steps = step
            latent_in = curr_latent
            
            if noise_sigma > 0.0 and step <= 3:
                latent_in = latent_in + torch.randn_like(latent_in) * noise_sigma
                
            scaled_latent = latent_in * scale_factor
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
                probe_device = next(probe.parameters()).device
                probe_dtype = next(probe.parameters()).dtype
                p_in = next_latent.to(device=probe_device, dtype=probe_dtype)
                p_logits = probe(p_in)
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
                    
        torch.cuda.synchronize(device)
        think_time_ms = (time.time() - rec_t0) * 1000.0
        
        # 3. Answer Generation Phase
        torch.cuda.synchronize(device)
        ans_t0 = time.time()
        
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
            
        torch.cuda.synchronize(device)
        t_end = time.time()
        
        ans_time_s = max(1e-5, t_end - ans_t0)
        total_time_s = t_end - t0
        
    full_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    clean_text = full_text.replace("</think>", "").replace("<｜end of sentence｜>", "").strip()
    is_finished = (tokenizer.eos_token in full_text) or len(generated_tokens) < max_new_tokens
    pred_num = extract_numeric_answer(clean_text, is_answer_only=is_finished)
    
    ans_tokens = len(generated_tokens)
    decoding_tok_s = ans_tokens / ans_time_s
    e2e_tok_s = ans_tokens / total_time_s
    
    return {
        "think_time_ms": round(think_time_ms, 2),
        "answer_time_s": round(ans_time_s, 3),
        "total_e2e_time_s": round(total_time_s, 3),
        "think_tokens": 0,
        "answer_tokens": ans_tokens,
        "decoding_tok_s": round(decoding_tok_s, 2),
        "e2e_tok_s": round(e2e_tok_s, 2),
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
        "mean_e2e_latency_s": round(sum(r["total_e2e_time_s"] for r in runs) / total, 3) if total else 0.0,
        "mean_think_time_ms": round(sum(r["think_time_ms"] for r in runs) / total, 2) if total else 0.0,
        "mean_answer_time_s": round(sum(r["answer_time_s"] for r in runs) / total, 3) if total else 0.0,
        "mean_answer_tokens": round(sum(r["answer_tokens"] for r in runs) / total, 1) if total else 0.0,
        "mean_decoding_tok_s": round(sum(r["decoding_tok_s"] for r in runs) / total, 2) if total else 0.0,
        "mean_e2e_tok_s": round(sum(r["e2e_tok_s"] for r in runs) / total, 2) if total else 0.0,
        "mean_steps": round(sum(r.get("actual_steps", 0) for r in runs) / total, 2) if total else 0.0,
        "steps_by_tier": {
            "easy": round(sum(r.get("actual_steps", 0) for r in easy_runs) / len(easy_runs), 2) if easy_runs else 0.0,
            "medium": round(sum(r.get("actual_steps", 0) for r in med_runs) / len(med_runs), 2) if med_runs else 0.0,
            "hard": round(sum(r.get("actual_steps", 0) for r in hard_runs) / len(hard_runs), 2) if hard_runs else 0.0,
            "math500": round(sum(r.get("actual_steps", 0) for r in math_runs) / len(math_runs), 2) if math_runs else 0.0
        }
    }

def save_benchmark_results(output_file, metadata, summary_dict, detailed_dict):
    out = {
        "metadata": metadata,
        "summary": summary_dict,
        "detailed_runs": detailed_dict
    }
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Latent Recurrence E2E Latency, Throughput, and Think Harder Benchmark.")
    parser.add_argument("--model", type=str, choices=["1.5b", "7b", "14b"], required=True, help="Target model size")
    parser.add_argument("--device", type=str, default=None, help="CUDA device (e.g., cuda:0 or cuda:1)")
    parser.add_argument("--precision", type=str, default=None, help="Override precision (e.g. bfloat16 or 8-bit)")
    parser.add_argument("--condition", type=str, default=None, choices=["all", "discrete", "fixed", "think_harder"], help="Evaluate specific condition only")
    parser.add_argument("--rerun_discrete", action="store_true", help="Re-run discrete baseline instead of loading previous 2048-token discrete runs")
    parser.add_argument("--resume", action="store_true", help="Resume from partially completed output file")
    parser.add_argument("--output_file", type=str, default=None, help="Custom output JSON path")
    args = parser.parse_args()
    
    cfg = dict(MODEL_CONFIGS[args.model])
    device = args.device if args.device is not None else cfg["default_device"]
    output_file = args.output_file if args.output_file is not None else cfg["output_file"]
    precision = args.precision if args.precision is not None else cfg["precision"]
    cfg["precision"] = precision
    
    print("=" * 100)
    print(f"BENCHMARK: DeepSeek-R1 Latent Recurrence ({args.model.upper()}) on {device}")
    print(f"Base Model: {cfg['model_id']} | Precision: {cfg['precision']}")
    print(f"LoRA: {cfg['lora_path']} | Probe: {cfg['probe_path']}")
    print(f"Max New Tokens: 2048 | Output File: {output_file}")
    print("=" * 100)
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    
    # 2. Load Base Model
    print(f"Loading Base Model {cfg['model_id']} in {cfg['precision']}...")
    if cfg["precision"] == "8-bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.float16
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            device_map=device,
            torch_dtype=torch.bfloat16
        )
    base_model.eval()
    
    embed_weights = base_model.get_input_embeddings().weight.detach()
    avg_embed_norm = torch.norm(embed_weights.float(), dim=-1).mean().item()
    hidden_dim = base_model.config.hidden_size
    scale_factor = avg_embed_norm / (hidden_dim ** 0.5)
    print(f"Calculated scale factor alpha: {scale_factor:.6f}")
    
    # 3. Load Auxiliary Probe
    probe = None
    if os.path.exists(cfg["probe_path"]):
        probe_device = "cpu" if args.model == "14b" else device
        probe_dtype = torch.float16 if (cfg["precision"] == "8-bit" and probe_device != "cpu") else torch.bfloat16
        print(f"Loading auxiliary thought probe from {cfg['probe_path']} on {probe_device} ({probe_dtype})...")
        probe = ThoughtStreamProbe(hidden_dim, base_model.config.vocab_size).to(probe_device, dtype=probe_dtype)
        probe.load_state_dict(torch.load(cfg["probe_path"], map_location=probe_device, weights_only=True))
        probe.eval()
        
    # 4. Load Benchmark Problems
    gsm_samples = get_gsm8k_stratified(num_per_tier=5)
    math_samples = get_math500_hard(num_samples=10)
    eval_items = gsm_samples + math_samples
    print(f"Loaded {len(eval_items)} benchmark items (15 GSM8K + 10 MATH-500).")
    
    summary_dict = {}
    detailed_dict = {}
    
    # Check if output file already exists for resuming
    if (args.resume or os.path.exists(output_file)) and os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            summary_dict = existing_data.get("summary", {})
            detailed_dict = existing_data.get("detailed_runs", {})
            print(f"Loaded existing results from {output_file} ({list(detailed_dict.keys())})")
        except Exception as e:
            print(f"Notice: Could not parse existing output file {output_file}: {e}")
            
    metadata = {
        "model_size": args.model,
        "model_id": cfg["model_id"],
        "precision": cfg["precision"],
        "device": device,
        "scale_factor": scale_factor,
        "max_new_tokens": 2048,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # ---------------------------------------------------------
    # CONDITION 1: DISCRETE BASELINE COT
    # ---------------------------------------------------------
    run_discrete = (args.condition is None or args.condition in ["all", "discrete"])
    if run_discrete:
        print("\n" + "=" * 90)
        print(f"CONDITION 1: DISCRETE BASELINE COT ({cfg['model_id']})")
        print("=" * 90)
        
        if "discrete_baseline" in detailed_dict and len(detailed_dict["discrete_baseline"]) == len(eval_items) and not args.rerun_discrete:
            print(f"Discrete baseline already complete in {output_file}.")
        elif not args.rerun_discrete and os.path.exists(cfg["prev_results"]):
            print(f"Reusing verified 2048-token discrete baseline results from {cfg['prev_results']}...")
            with open(cfg["prev_results"], "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            raw_discrete_runs = prev_data["detailed_runs"]["discrete_baseline"]
            discrete_runs = []
            for r in raw_discrete_runs:
                r_copy = dict(r)
                ans_time_s = max(1e-5, r_copy["total_time_s"] - (r_copy["think_time_ms"] / 1000.0))
                ans_tok = r_copy["answer_tokens"]
                tot_time = r_copy["total_time_s"]
                r_copy["answer_time_s"] = round(ans_time_s, 3)
                r_copy["total_e2e_time_s"] = round(tot_time, 3)
                r_copy["decoding_tok_s"] = round(ans_tok / ans_time_s, 2) if ans_time_s > 0 else 0.0
                r_copy["e2e_tok_s"] = round(ans_tok / tot_time, 2) if tot_time > 0 else 0.0
                discrete_runs.append(r_copy)
                
            detailed_dict["discrete_baseline"] = discrete_runs
            summary_dict["discrete_baseline"] = compute_stats(discrete_runs, "Discrete Baseline CoT")
            save_benchmark_results(output_file, metadata, summary_dict, detailed_dict)
            print(f"Discrete Baseline Loaded: Acc={summary_dict['discrete_baseline']['overall_accuracy']}%, ThinkTime={summary_dict['discrete_baseline']['mean_think_time_ms']:.1f}ms, E2E={summary_dict['discrete_baseline']['mean_e2e_latency_s']:.2f}s, E2E Tok/s={summary_dict['discrete_baseline']['mean_e2e_tok_s']:.1f}")
        else:
            discrete_runs = detailed_dict.get("discrete_baseline", [])
            completed_ids = {r["question_id"] for r in discrete_runs}
            for idx, item in enumerate(eval_items):
                if item["id"] in completed_ids:
                    continue
                q = item["question"]
                is_math = "math500" in item["id"]
                gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
                print(f"[{idx+1}/{len(eval_items)}] [{item['category']}] Discrete: {q[:50]}...")
                res = run_discrete_baseline(base_model, tokenizer, q, device=device, max_baseline_tokens=2048)
                is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
                res["question_id"] = item["id"]
                res["category"] = item["category"]
                res["ground_truth_num"] = gt_num
                res["ground_truth_raw"] = item["answer"]
                res["correct"] = is_correct
                discrete_runs.append(res)
                detailed_dict["discrete_baseline"] = discrete_runs
                summary_dict["discrete_baseline"] = compute_stats(discrete_runs, "Discrete Baseline CoT")
                save_benchmark_results(output_file, metadata, summary_dict, detailed_dict)
                print(f"      -> Pred: {res['predicted_num']} | Correct: {is_correct} | Think: {res['think_time_ms']:.1f}ms | E2E: {res['total_e2e_time_s']:.2f}s | Dec: {res['decoding_tok_s']} tok/s | E2E: {res['e2e_tok_s']} tok/s")
                
    # ---------------------------------------------------------
    # LOAD PEFT LORA ADAPTER
    # ---------------------------------------------------------
    run_recurrent = (args.condition is None or args.condition in ["all", "fixed", "think_harder"])
    if run_recurrent:
        print("\n" + "=" * 90)
        print(f"LOADING PEFT LORA ADAPTER FROM {cfg['lora_path']}...")
        print("=" * 90)
        lora_model = PeftModel.from_pretrained(base_model, cfg["lora_path"])
        lora_model.eval()
        print(f"Loaded LoRA adapter successfully.")
        
        # ---------------------------------------------------------
        # CONDITIONS 2 & 3: FIXED K=6 AND THINK HARDER (LOGIT SUPPRESS + TRIGGER D)
        # ---------------------------------------------------------
        all_eval_conditions = [
            (
                "fixed_k6_recurrent",
                "Fixed K=6 Recurrent",
                {
                    "k_mode": "fixed",
                    "k_fixed": 6,
                    "max_new_tokens": 2048
                }
            ),
            (
                "think_harder_suppress_entropy",
                "Think Harder (Logit Suppress + Trigger D)",
                {
                    "k_mode": "dynamic",
                    "trigger_type": "entropy_stabilized",
                    "suppress_think_steps": 6,
                    "k_min": 8,
                    "k_max": 16,
                    "tau_think": 0.90,
                    "tau_entropy": 0.20,
                    "max_new_tokens": 2048
                }
            )
        ]
        
        eval_conditions = []
        for ck, cn, ckw in all_eval_conditions:
            if args.condition == "fixed" and ck != "fixed_k6_recurrent":
                continue
            if args.condition == "think_harder" and ck != "think_harder_suppress_entropy":
                continue
            eval_conditions.append((ck, cn, ckw))
            
        for cond_key, cond_name, cond_kwargs in eval_conditions:
            print("\n" + "=" * 90)
            print(f"EVALUATING CONDITION: {cond_name} (max_new_tokens=2048)")
            print("=" * 90)
            cond_runs = detailed_dict.get(cond_key, [])
            completed_ids = {r["question_id"] for r in cond_runs}
            if len(completed_ids) == len(eval_items):
                print(f"Condition {cond_name} already 100% completed ({len(completed_ids)}/{len(eval_items)}).")
                summary_dict[cond_key] = compute_stats(cond_runs, cond_name)
                continue
                
            for idx, item in enumerate(eval_items):
                if item["id"] in completed_ids:
                    continue
                q = item["question"]
                is_math = "math500" in item["id"]
                gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
                res = run_recurrent_step(
                    lora_model, tokenizer, probe, q,
                    scale_factor=scale_factor,
                    think_end_id=think_end_id,
                    device=device,
                    **cond_kwargs
                )
                is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
                res["question_id"] = item["id"]
                res["category"] = item["category"]
                res["ground_truth_num"] = gt_num
                res["ground_truth_raw"] = item["answer"]
                res["correct"] = is_correct
                cond_runs.append(res)
                detailed_dict[cond_key] = cond_runs
                summary_dict[cond_key] = compute_stats(cond_runs, cond_name)
                save_benchmark_results(output_file, metadata, summary_dict, detailed_dict)
                print(f"[{idx+1}/{len(eval_items)}] [{item['category']}] -> Pred: {res['predicted_num']} | Correct: {is_correct} | Steps: {res['actual_steps']} ({res['halt_reason']}) | Think: {res['think_time_ms']:.1f}ms | E2E: {res['total_e2e_time_s']:.2f}s | Dec: {res['decoding_tok_s']:.1f} tok/s | E2E: {res['e2e_tok_s']:.1f} tok/s")
                
    # Compute relative speedups vs discrete baseline
    if "discrete_baseline" in summary_dict:
        disc_think_ms = summary_dict["discrete_baseline"]["mean_think_time_ms"]
        disc_e2e_s = summary_dict["discrete_baseline"]["mean_e2e_latency_s"]
        for k, s in summary_dict.items():
            if k == "discrete_baseline":
                s["speedup_vs_discrete"] = 1.0
                s["e2e_speedup_vs_discrete"] = 1.0
            else:
                s["speedup_vs_discrete"] = round(disc_think_ms / max(1e-5, s["mean_think_time_ms"]), 2)
                s["e2e_speedup_vs_discrete"] = round(disc_e2e_s / max(1e-5, s["mean_e2e_latency_s"]), 2)
                
    save_benchmark_results(output_file, metadata, summary_dict, detailed_dict)
    
    print("\n" + "=" * 120)
    print(f"BENCHMARK SUMMARY FOR {args.model.upper()} ({cfg['model_id']}) [max_new_tokens=2048]")
    print("=" * 120)
    print(f"{'Condition':<40} | {'Overall':<7} | {'GSM8K':<7} | {'MATH500':<7} | {'Think(ms)':<10} | {'E2E(s)':<8} | {'Dec tok/s':<10} | {'E2E tok/s':<10} | {'Steps':<5}")
    print("-" * 120)
    for k, s in summary_dict.items():
        print(f"{s['name']:<40} | {s['overall_accuracy']:>6.1f}% | {s['gsm8k_accuracy']:>6.1f}% | {s['math500_accuracy']:>6.1f}% | {s['mean_think_time_ms']:>9.1f} | {s['mean_e2e_latency_s']:>7.2f} | {s['mean_decoding_tok_s']:>9.1f} | {s['mean_e2e_tok_s']:>9.1f} | {s['mean_steps']:>5.1f}")
    print("=" * 120)
    print(f"Results successfully recorded to {output_file}")

if __name__ == "__main__":
    main()
