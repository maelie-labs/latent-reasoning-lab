#!/usr/bin/env python3
"""
08_dynamic_halting.py
Dynamic Early Halting & "Think Harder" Latent Recurrence Benchmark.

Investigates:
1. Dynamic early exit triggers:
   - Trigger A: Model's native </think> token logit / softmax probability crossing threshold.
   - Trigger B: Hidden state convergence (cosine similarity cos(h_t, h_{t-1}) and relative L2 distance ||h_t - h_{t-1}|| / ||h_t||).
   - Trigger C (Hybrid): Joint calibrated exit (P(</think>) >= tau_think AND convergence).
   - Trigger D: Auxiliary probe entropy stabilization (|Delta H| <= tau_entropy AND P(</think>) >= 0.90).
2. Expanded loop budget (K_min=2, K_max=16).
3. "Think Harder" exploration:
   - Option A: Extended loop floor (K_min=8, K_max=16) + Deep reasoning prompt steering
   - Option B: Native </think> logit suppression (subtract 10.0 for first 6 passes, forcing K >= 8)
   - Option C: Latent noise perturbation (sigma=0.01 for early exploration)
4. Comprehensive multi-tier evaluation:
   - GSM8K Stratified (Easy: <=2 ops, Medium: 3-4 ops, Hard: >=5 ops)
   - MATH-500 Hard Challenge (Level 4 and 5 Olympiad / Competition Math)
5. Comparison against:
   - Discrete Baseline CoT (DeepSeek-R1-Distill-Qwen-1.5B teacher)
   - Fixed K=6 Recurrent LoRA (Phase 3 baseline)
   - Dynamic Exit Triggers (A, B, C, D)
   - Think Harder mechanisms (Prompt, Logit Suppression, Latent Perturbation)
"""

import os
import re
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LORA_PATH = "checkpoints/lora_recurrent_1.5b"
PROBE_PATH = "checkpoints/thought_probe.pt"

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
    """Clean LaTeX macros and isolate float value from candidate string."""
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
    try:
        return float(cand)
    except ValueError:
        nums = re.findall(r"[-+]?\d*\.?\d+", cand)
        if nums:
            return float(nums[-1])
    return None

def extract_math_boxed_expression(text):
    """Extracts content inside the last \\boxed{...}, correctly handling nested braces."""
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
        return text[start:end-1].strip()
    return ""

def normalize_math_str(s):
    """Normalizes a LaTeX mathematical string for comparison."""
    if not s:
        return ""
    s = s.strip()
    boxed = extract_math_boxed_expression(s)
    if boxed:
        s = boxed
    s = re.sub(r"\\(?:text|mathbf|mathrm)\{([^}]+)\}", r"\1", s)
    s = re.sub(r"[\$\s]", "", s)
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"^{\circ}", "").replace(r"^\circ", "")
    if s.startswith("x="):
        s = s[2:]
    return s

def try_eval_numeric(s):
    """Attempts to evaluate a normalized math string as a float."""
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
    """Robust equivalence check for mathematical answers (LaTeX strings and numbers)."""
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
    """
    Extracts numerical answer from text, supporting GSM8K and MATH formats.
    Only falls back to trailing digits if is_answer_only is True (preventing false positives
    from unfinished thought monologues).
    """
    if not text:
        return None
        
    # 1. \\boxed{...}
    boxed = extract_math_boxed_expression(text)
    if boxed:
        val = clean_and_extract_candidate(boxed)
        if val is not None:
            return val
            
    # 2. #### <num>
    hash_match = re.findall(r'####\s*([-\d.,]+)', text)
    if hash_match:
        val = clean_and_extract_candidate(hash_match[-1])
        if val is not None:
            return val
            
    # 3. Explicit phrasing
    ans_match = re.findall(r'(?:the answer is|final answer is|total is|equals|equal to)\s*([\$]?[-\d.,]+)', text, re.IGNORECASE)
    if ans_match:
        val = clean_and_extract_candidate(ans_match[-1])
        if val is not None:
            return val
            
    # 4. Fallback to last number ONLY if generation finished and is dedicated answer text
    if is_answer_only:
        nums = re.findall(r'[-+]?\d*\.?\d+', text)
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                pass
            
    return None

def check_match(pred_num, pred_text, gt_num, gt_raw, is_math=False):
    """
    Checks if predicted answer matches ground truth.
    For MATH problems, strictly uses mathematical normalization or numerical equivalence.
    """
    # 1. Check numeric match
    if pred_num is not None and gt_num is not None:
        if abs(pred_num - gt_num) < 1e-4:
            return True
            
    # 2. For MATH problems or expressions with LaTeX
    if gt_raw:
        pred_box = extract_math_boxed_expression(pred_text)
        if pred_box and math_equal(pred_box, gt_raw):
            return True
        if math_equal(pred_text, gt_raw):
            return True
            
    return False

def run_discrete_baseline(model, tokenizer, question, max_baseline_tokens=2048):
    """
    Runs discrete baseline generation with full <think> tokens.
    Properly accounts for thinking latency and ensures no false positives on unfinished thoughts.
    """
    messages = [{"role": "user", "content": question}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    
    torch.cuda.synchronize()
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
        # Never finished thinking within token budget
        think_tokens = len(gen_ids)
        ans_tokens = 0
        think_time_ms = total_time * 1000.0
        answer_text = ""
        pred_num = None  # No valid answer generated
        
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
    tau_think=0.95,
    tau_cos=0.992,
    tau_l2=0.10,
    tau_entropy=0.20,
    prompt_prefix="",
    suppress_think_steps=0,
    noise_sigma=0.0,
    max_new_tokens=512
):
    """
    Executes continuous latent recurrence with dynamic early halting or think-harder controls.
    """
    user_content = prompt_prefix + question if prompt_prefix else question
    messages = [{"role": "user", "content": user_content}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    
    torch.cuda.synchronize()
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
        
        torch.cuda.synchronize()
        rec_t0 = time.time()
        
        for step in range(1, target_max + 1):
            actual_steps = step
            latent_in = curr_latent
            
            # Optional perturbation for exploration
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
            
            # Convergence metrics computed in float32 to prevent bfloat16 overflow/clamping
            cos_sim = min(1.0, F.cosine_similarity(prev_latent.squeeze(1).float(), next_latent.squeeze(1).float(), dim=-1).item())
            l2_diff = torch.norm(next_latent.float() - prev_latent.float(), p=2).item()
            norm_next = torch.norm(next_latent.float(), p=2).item()
            rel_l2 = l2_diff / (norm_next + 1e-9)
            
            # Model logits & </think> probability
            logits = model.lm_head(next_latent)[0, -1]
            if step <= suppress_think_steps:
                logits = logits.clone()
                logits[think_end_id] -= 10.0
                
            probs = torch.softmax(logits, dim=-1)
            p_think = probs[think_end_id].item()
            top1_val, top1_idx = torch.topk(probs, 1)
            
            # Probe metrics
            if probe is not None:
                p_logits = probe(next_latent)
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
            
            # Dynamic Halting checks (active only when k_mode == 'dynamic')
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
        
        # 3. First token predicted from final latent
        next_token = torch.argmax(model.lm_head(curr_latent)[0, -1]).unsqueeze(0)
        generated_tokens = [next_token.item()]
        
        # 4. Generate answer tokens autoregressively
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

def get_gsm8k_stratified(num_per_tier=5):
    """Selects stratified GSM8K test samples across Easy, Medium, and Hard tiers."""
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
    """Selects Level 4 and Level 5 challenging competition math problems from MATH-500."""
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

def save_intermediate_results(output_file, metadata, summary_dict, detailed_dict):
    """Safely saves results to JSON incrementally."""
    out = {
        "metadata": metadata,
        "summary": summary_dict,
        "detailed_runs": detailed_dict
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Evaluate dynamic halting and think-harder latent recurrence.")
    parser.add_argument("--output_file", type=str, default="data/dynamic_halting_results.json")
    parser.add_argument("--gsm8k_samples", type=int, default=15, help="Total GSM8K samples (5 easy, 5 med, 5 hard)")
    parser.add_argument("--math500_samples", type=int, default=10, help="Total MATH-500 Level 4/5 samples")
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=16)
    parser.add_argument("--tau_think", type=float, default=0.95)
    parser.add_argument("--tau_cos", type=float, default=0.992)
    parser.add_argument("--tau_l2", type=float, default=0.10)
    parser.add_argument("--tau_entropy", type=float, default=0.20)
    args = parser.parse_args()
    
    print("=" * 90)
    print("08_dynamic_halting.py: Comprehensive Dynamic Halting & 'Think Harder' Recurrence Benchmark")
    print(f"Device: {DEVICE} | K_min: {args.k_min} | K_max: {args.k_max}")
    print(f"Halting Thresholds: P(</think>) >= {args.tau_think}, CosSim >= {args.tau_cos}, RelL2 <= {args.tau_l2}")
    print("=" * 90)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    
    # 2. Load Base Model
    print(f"Loading Base Model {MODEL_ID} in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=DEVICE
    )
    base_model.eval()
    
    hidden_dim = base_model.config.hidden_size
    vocab_size = base_model.config.vocab_size
    scale_factor = 1.1641 / (hidden_dim ** 0.5)
    
    # 3. Load Auxiliary Probe
    probe = ThoughtStreamProbe(hidden_dim, vocab_size).to(DEVICE, dtype=torch.bfloat16)
    if os.path.exists(PROBE_PATH):
        probe.load_state_dict(torch.load(PROBE_PATH, weights_only=True))
        probe.eval()
        print(f"Loaded Auxiliary Thought Probe from {PROBE_PATH}")
    else:
        probe = None
        print("Auxiliary Probe not found, continuing without probe entropy.")
        
    # 4. Prepare Benchmark Samples
    gsm8k_items = get_gsm8k_stratified(num_per_tier=args.gsm8k_samples // 3)
    math_items = get_math500_hard(num_samples=args.math500_samples)
    all_eval_items = gsm8k_items + math_items
    
    print(f"\nBenchmark Suite: {len(all_eval_items)} questions total")
    print(f"  - GSM8K Stratified: {len(gsm8k_items)} (Easy: 5, Med: 5, Hard: 5)")
    print(f"  - MATH-500 Hard: {len(math_items)} (Level 4 & 5)")
    
    metadata = {
        "model": MODEL_ID,
        "lora_path": LORA_PATH,
        "device": DEVICE,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "tau_think": args.tau_think,
        "tau_cos": args.tau_cos,
        "tau_l2": args.tau_l2,
        "tau_entropy": args.tau_entropy,
        "total_questions": len(all_eval_items)
    }
    summary_dict = {}
    detailed_dict = {}
    
    # -------------------------------------------------------------
    # CONDITION 1: DISCRETE BASELINE COT (Base Model)
    # -------------------------------------------------------------
    print("\n" + "="*90)
    print("CONDITION 1: DISCRETE BASELINE COT (DeepSeek-R1-Distill-Qwen-1.5B Teacher)")
    print("="*90)
    discrete_runs = []
    for idx, item in enumerate(all_eval_items):
        q = item["question"]
        is_math = "math500" in item["id"]
        gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
        print(f"[{idx+1}/{len(all_eval_items)}] [{item['category']}] Discrete CoT: {q[:60]}... (GT: {gt_num or item['answer'][:15]})")
        res = run_discrete_baseline(base_model, tokenizer, q, max_baseline_tokens=2048)
        is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
        res["question_id"] = item["id"]
        res["category"] = item["category"]
        res["ground_truth_num"] = gt_num
        res["ground_truth_raw"] = item["answer"]
        res["correct"] = is_correct
        discrete_runs.append(res)
        print(f"      -> Pred: {res['predicted_num']} | Correct: {is_correct} | ThinkTime: {res['think_time_ms']:.1f}ms | TotTime: {res['total_time_s']:.2f}s | ClosedThink: {'</think>' in res['raw_text']}")
        
    detailed_dict["discrete_baseline"] = discrete_runs
    summary_dict["discrete_baseline"] = compute_stats(discrete_runs, "Discrete Baseline CoT")
    save_intermediate_results(args.output_file, metadata, summary_dict, detailed_dict)
    
    # -------------------------------------------------------------
    # LOAD PEFT LORA ADAPTER
    # -------------------------------------------------------------
    print("\n" + "="*90)
    print("LOADING PEFT LORA ADAPTER...")
    print("="*90)
    lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    lora_model.eval()
    print(f"Loaded LoRA adapter from {LORA_PATH} successfully.")
    
    # -------------------------------------------------------------
    # CONDITION 2: FIXED RECURRENT K=6 LORA
    # -------------------------------------------------------------
    print("\n" + "="*90)
    print("CONDITION 2: FIXED RECURRENT K=6 LORA (Phase 3 Champion)")
    print("="*90)
    fixed_k6_runs = []
    for idx, item in enumerate(all_eval_items):
        q = item["question"]
        is_math = "math500" in item["id"]
        gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
        print(f"[{idx+1}/{len(all_eval_items)}] [{item['category']}] Fixed K=6: {q[:60]}...")
        res = run_recurrent_step(
            lora_model, tokenizer, probe, q, scale_factor, think_end_id,
            k_mode="fixed", k_fixed=6
        )
        res["mode"] = "fixed_k6_recurrent"
        is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
        res["question_id"] = item["id"]
        res["category"] = item["category"]
        res["ground_truth_num"] = gt_num
        res["ground_truth_raw"] = item["answer"]
        res["correct"] = is_correct
        fixed_k6_runs.append(res)
        print(f"      -> Pred: {res['predicted_num']} | Correct: {is_correct} | ThinkTime: {res['think_time_ms']:.1f}ms | TotTime: {res['total_time_s']:.2f}s")
        
    detailed_dict["fixed_k6_recurrent"] = fixed_k6_runs
    summary_dict["fixed_k6_recurrent"] = compute_stats(fixed_k6_runs, "Fixed K=6 Recurrent LoRA")
    save_intermediate_results(args.output_file, metadata, summary_dict, detailed_dict)

    # -------------------------------------------------------------
    # CONDITION 3A: DYNAMIC TRIGGER A (Logit / P(</think>) >= tau)
    # -------------------------------------------------------------
    print("\n" + "="*90)
    print(f"CONDITION 3A: DYNAMIC TRIGGER A (Native </think> Probability >= {args.tau_think})")
    print("="*90)
    trigger_a_runs = []
    for idx, item in enumerate(all_eval_items):
        q = item["question"]
        is_math = "math500" in item["id"]
        gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
        print(f"[{idx+1}/{len(all_eval_items)}] [{item['category']}] Trigger A: {q[:60]}...")
        res = run_recurrent_step(
            lora_model, tokenizer, probe, q, scale_factor, think_end_id,
            k_mode="dynamic", k_min=args.k_min, k_max=args.k_max,
            trigger_type="trigger_a_logit", tau_think=args.tau_think
        )
        res["mode"] = "trigger_a_logit"
        is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
        res["question_id"] = item["id"]
        res["category"] = item["category"]
        res["ground_truth_num"] = gt_num
        res["ground_truth_raw"] = item["answer"]
        res["correct"] = is_correct
        trigger_a_runs.append(res)
        print(f"      -> Pred: {res['predicted_num']} | Correct: {is_correct} | Steps: {res['actual_steps']} ({res['halt_reason']}) | ThinkTime: {res['think_time_ms']:.1f}ms")

    detailed_dict["trigger_a_logit"] = trigger_a_runs
    summary_dict["trigger_a_logit"] = compute_stats(trigger_a_runs, "Trigger A (Logit/P_think)")
    save_intermediate_results(args.output_file, metadata, summary_dict, detailed_dict)

    # -------------------------------------------------------------
    # CONDITION 3B: DYNAMIC TRIGGER B (Hidden State Geometric Convergence)
    # -------------------------------------------------------------
    print("\n" + "="*90)
    print(f"CONDITION 3B: DYNAMIC TRIGGER B (Geometric Convergence: CosSim >= {args.tau_cos} or RelL2 <= {args.tau_l2})")
    print("="*90)
    trigger_b_runs = []
    for idx, item in enumerate(all_eval_items):
        q = item["question"]
        is_math = "math500" in item["id"]
        gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
        print(f"[{idx+1}/{len(all_eval_items)}] [{item['category']}] Trigger B: {q[:60]}...")
        res = run_recurrent_step(
            lora_model, tokenizer, probe, q, scale_factor, think_end_id,
            k_mode="dynamic", k_min=args.k_min, k_max=args.k_max,
            trigger_type="trigger_b_conv", tau_cos=args.tau_cos, tau_l2=args.tau_l2
        )
        res["mode"] = "trigger_b_conv"
        is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
        res["question_id"] = item["id"]
        res["category"] = item["category"]
        res["ground_truth_num"] = gt_num
        res["ground_truth_raw"] = item["answer"]
        res["correct"] = is_correct
        trigger_b_runs.append(res)
        print(f"      -> Pred: {res['predicted_num']} | Correct: {is_correct} | Steps: {res['actual_steps']} ({res['halt_reason']}) | ThinkTime: {res['think_time_ms']:.1f}ms")

    detailed_dict["trigger_b_conv"] = trigger_b_runs
    summary_dict["trigger_b_conv"] = compute_stats(trigger_b_runs, "Trigger B (Geometric Conv)")
    save_intermediate_results(args.output_file, metadata, summary_dict, detailed_dict)

    # -------------------------------------------------------------
    # CONDITION 3C: DYNAMIC TRIGGER C (Hybrid Calibrated: P >= 0.90 AND Conv)
    # -------------------------------------------------------------
    print("\n" + "="*90)
    print("CONDITION 3C: DYNAMIC TRIGGER C (Hybrid Calibrated: P >= 0.90 AND Geometric Convergence)")
    print("="*90)
    hybrid_runs = []
    for idx, item in enumerate(all_eval_items):
        q = item["question"]
        is_math = "math500" in item["id"]
        gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
        print(f"[{idx+1}/{len(all_eval_items)}] [{item['category']}] Hybrid Trigger: {q[:60]}...")
        res = run_recurrent_step(
            lora_model, tokenizer, probe, q, scale_factor, think_end_id,
            k_mode="dynamic", k_min=args.k_min, k_max=args.k_max,
            trigger_type="hybrid", tau_think=0.90, tau_cos=args.tau_cos, tau_l2=args.tau_l2
        )
        res["mode"] = "dynamic_hybrid"
        is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
        res["question_id"] = item["id"]
        res["category"] = item["category"]
        res["ground_truth_num"] = gt_num
        res["ground_truth_raw"] = item["answer"]
        res["correct"] = is_correct
        hybrid_runs.append(res)
        print(f"      -> Pred: {res['predicted_num']} | Correct: {is_correct} | Steps: {res['actual_steps']} ({res['halt_reason']}) | ThinkTime: {res['think_time_ms']:.1f}ms")

    detailed_dict["dynamic_hybrid"] = hybrid_runs
    summary_dict["dynamic_hybrid"] = compute_stats(hybrid_runs, "Dynamic Hybrid (Trigger C)")
    save_intermediate_results(args.output_file, metadata, summary_dict, detailed_dict)

    # -------------------------------------------------------------
    # CONDITION 3D: DYNAMIC TRIGGER D (Probe Entropy Stabilization)
    # -------------------------------------------------------------
    print("\n" + "="*90)
    print(f"CONDITION 3D: DYNAMIC TRIGGER D (Probe Entropy Delta <= {args.tau_entropy} AND P >= 0.90)")
    print("="*90)
    entropy_runs = []
    for idx, item in enumerate(all_eval_items):
        q = item["question"]
        is_math = "math500" in item["id"]
        gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
        print(f"[{idx+1}/{len(all_eval_items)}] [{item['category']}] Entropy Trigger: {q[:60]}...")
        res = run_recurrent_step(
            lora_model, tokenizer, probe, q, scale_factor, think_end_id,
            k_mode="dynamic", k_min=args.k_min, k_max=args.k_max,
            trigger_type="entropy_stabilized", tau_think=0.90, tau_entropy=args.tau_entropy
        )
        res["mode"] = "trigger_d_entropy"
        is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
        res["question_id"] = item["id"]
        res["category"] = item["category"]
        res["ground_truth_num"] = gt_num
        res["ground_truth_raw"] = item["answer"]
        res["correct"] = is_correct
        entropy_runs.append(res)
        print(f"      -> Pred: {res['predicted_num']} | Correct: {is_correct} | Steps: {res['actual_steps']} ({res['halt_reason']}) | ThinkTime: {res['think_time_ms']:.1f}ms")

    detailed_dict["trigger_d_entropy"] = entropy_runs
    summary_dict["trigger_d_entropy"] = compute_stats(entropy_runs, "Trigger D (Entropy Stabilized)")
    save_intermediate_results(args.output_file, metadata, summary_dict, detailed_dict)

    # -------------------------------------------------------------
    # CONDITION 4A: THINK HARDER (Prompt Guidance + Extended Floor K_min=8)
    # -------------------------------------------------------------
    print("\n" + "="*90)
    print("CONDITION 4A: THINK HARDER (Extended Floor K_min=8, K_max=16 + Prompt Guidance)")
    print("="*90)
    think_prompt_runs = []
    prompt_guidance = "Solve this difficult mathematics problem step by step. Reason deeply, verify all operations, and think through all constraints thoroughly before answering: "
    for idx, item in enumerate(all_eval_items):
        q = item["question"]
        is_math = "math500" in item["id"]
        gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
        print(f"[{idx+1}/{len(all_eval_items)}] [{item['category']}] Think Harder (Prompt): {q[:60]}...")
        res = run_recurrent_step(
            lora_model, tokenizer, probe, q, scale_factor, think_end_id,
            k_mode="dynamic", k_min=8, k_max=16,
            trigger_type="hybrid", tau_think=0.90, tau_cos=args.tau_cos, tau_l2=args.tau_l2,
            prompt_prefix=prompt_guidance
        )
        res["mode"] = "think_harder_prompt"
        is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
        res["question_id"] = item["id"]
        res["category"] = item["category"]
        res["ground_truth_num"] = gt_num
        res["ground_truth_raw"] = item["answer"]
        res["correct"] = is_correct
        think_prompt_runs.append(res)
        print(f"      -> Pred: {res['predicted_num']} | Correct: {is_correct} | Steps: {res['actual_steps']} ({res['halt_reason']}) | ThinkTime: {res['think_time_ms']:.1f}ms")

    detailed_dict["think_harder_prompt"] = think_prompt_runs
    summary_dict["think_harder_prompt"] = compute_stats(think_prompt_runs, "Think Harder (Prompt+K>=8)")
    save_intermediate_results(args.output_file, metadata, summary_dict, detailed_dict)

    # -------------------------------------------------------------
    # CONDITION 4B: THINK HARDER (Logit Suppression of </think>)
    # -------------------------------------------------------------
    print("\n" + "="*90)
    print("CONDITION 4B: THINK HARDER (Logit Suppression of </think> for passes 1-6, K_min=8)")
    print("="*90)
    think_suppress_runs = []
    for idx, item in enumerate(all_eval_items):
        q = item["question"]
        is_math = "math500" in item["id"]
        gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
        print(f"[{idx+1}/{len(all_eval_items)}] [{item['category']}] Think Harder (Suppress): {q[:60]}...")
        res = run_recurrent_step(
            lora_model, tokenizer, probe, q, scale_factor, think_end_id,
            k_mode="dynamic", k_min=8, k_max=16,
            trigger_type="hybrid", tau_think=0.90, tau_cos=args.tau_cos, tau_l2=args.tau_l2,
            suppress_think_steps=6
        )
        res["mode"] = "think_harder_suppress"
        is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
        res["question_id"] = item["id"]
        res["category"] = item["category"]
        res["ground_truth_num"] = gt_num
        res["ground_truth_raw"] = item["answer"]
        res["correct"] = is_correct
        think_suppress_runs.append(res)
        print(f"      -> Pred: {res['predicted_num']} | Correct: {is_correct} | Steps: {res['actual_steps']} ({res['halt_reason']}) | ThinkTime: {res['think_time_ms']:.1f}ms")

    detailed_dict["think_harder_suppress"] = think_suppress_runs
    summary_dict["think_harder_suppress"] = compute_stats(think_suppress_runs, "Think Harder (Logit Suppress)")
    save_intermediate_results(args.output_file, metadata, summary_dict, detailed_dict)

    # -------------------------------------------------------------
    # CONDITION 4C: THINK HARDER (Latent Noise Perturbation sigma=0.01)
    # -------------------------------------------------------------
    print("\n" + "="*90)
    print("CONDITION 4C: THINK HARDER (Latent Perturbation sigma=0.01, passes 1-3, K_min=6)")
    print("="*90)
    think_noise_runs = []
    for idx, item in enumerate(all_eval_items):
        q = item["question"]
        is_math = "math500" in item["id"]
        gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
        print(f"[{idx+1}/{len(all_eval_items)}] [{item['category']}] Think Harder (Perturb): {q[:60]}...")
        res = run_recurrent_step(
            lora_model, tokenizer, probe, q, scale_factor, think_end_id,
            k_mode="dynamic", k_min=6, k_max=16,
            trigger_type="hybrid", tau_think=0.90, tau_cos=args.tau_cos, tau_l2=args.tau_l2,
            noise_sigma=0.01
        )
        res["mode"] = "think_harder_perturb"
        is_correct = check_match(res["predicted_num"], res["raw_text"], gt_num, item["answer"], is_math=is_math)
        res["question_id"] = item["id"]
        res["category"] = item["category"]
        res["ground_truth_num"] = gt_num
        res["ground_truth_raw"] = item["answer"]
        res["correct"] = is_correct
        think_noise_runs.append(res)
        print(f"      -> Pred: {res['predicted_num']} | Correct: {is_correct} | Steps: {res['actual_steps']} ({res['halt_reason']}) | ThinkTime: {res['think_time_ms']:.1f}ms")

    detailed_dict["think_harder_perturb"] = think_noise_runs
    summary_dict["think_harder_perturb"] = compute_stats(think_noise_runs, "Think Harder (Latent Perturb)")
    
    # Speedup Factors vs Discrete Baseline
    disc_think_ms = summary_dict["discrete_baseline"]["mean_think_time_ms"]
    for k, s in summary_dict.items():
        if k != "discrete_baseline":
            s["speedup_vs_discrete"] = round(disc_think_ms / max(1e-5, s["mean_think_time_ms"]), 2)
            
    save_intermediate_results(args.output_file, metadata, summary_dict, detailed_dict)
    
    # -------------------------------------------------------------
    # FINAL COMPREHENSIVE SUMMARY REPORT TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 130)
    print("EXP-08: COMPREHENSIVE DYNAMIC HALTING & THINK-HARDER BENCHMARK SUMMARY")
    print("=" * 130)
    header = f"{'Condition':<28} | {'Overall':<8} | {'GSM8K':<8} | {'Easy':<6} | {'Med':<6} | {'Hard':<6} | {'MATH':<6} | {'Think (ms)':<10} | {'Tot (s)':<8} | {'Steps':<6} | {'Speedup':<8}"
    print(header)
    print("-" * 130)
    for k, s in summary_dict.items():
        spd = f"{s.get('speedup_vs_discrete', 1.0):.1f}x" if k != "discrete_baseline" else "1.0x"
        line = (
            f"{s['name']:<28} | "
            f"{s['overall_accuracy']:>6.1f}% | "
            f"{s['gsm8k_accuracy']:>6.1f}% | "
            f"{s['gsm8k_easy_acc']:>5.1f}% | "
            f"{s['gsm8k_med_acc']:>5.1f}% | "
            f"{s['gsm8k_hard_acc']:>5.1f}% | "
            f"{s['math500_accuracy']:>5.1f}% | "
            f"{s['mean_think_time_ms']:>10.1f} | "
            f"{s['mean_total_time_s']:>8.2f} | "
            f"{s['mean_steps']:>6.1f} | "
            f"{spd:>8}"
        )
        print(line)
    print("=" * 130)
    print(f"All empirical traces successfully saved to {args.output_file}")

if __name__ == "__main__":
    main()
