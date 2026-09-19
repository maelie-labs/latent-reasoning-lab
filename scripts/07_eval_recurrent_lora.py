#!/usr/bin/env python3
"""
07_eval_recurrent_lora.py
Comprehensive evaluation benchmark comparing:
1. Baseline Discrete CoT (Standard DeepSeek-R1-Distill-1.5B with full <think> token generation)
2. Zero-Shot Recurrent Model (Base model with 6 latent passes without LoRA)
3. Trained Recurrent LoRA Model (Base model + LoRA adapter trained via curriculum)

Measures:
- Thinking Latency (ms)
- Total Response Latency (s)
- Thinking Tokens (count)
- Generated Answer Tokens (count)
- Accuracy (%) on GSM8K Test Questions
- Speedup Factor
"""

import os
import re
import json
import time
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate continuous latent recurrence model vs baseline CoT.")
    parser.add_argument("--lora_path", type=str, default="checkpoints/lora_recurrent_1.5b", help="Path to trained LoRA adapter")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of test samples from GSM8K")
    parser.add_argument("--recurrent_steps", type=int, default=6, help="Recurrent steps for latent reasoning")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum discrete answer tokens for recurrent generation")
    parser.add_argument("--max_baseline_tokens", type=int, default=1024, help="Maximum total discrete tokens for baseline CoT")
    parser.add_argument("--output_file", type=str, default="data/eval_results_lora.json", help="Path to save evaluation results")
    return parser.parse_args()

def clean_and_extract_candidate(cand):
    """Clean LaTeX macros and isolate float value from a candidate string."""
    cand = re.sub(r"\\(text|mathbf)\{([^}]+)\}", r"\2", cand)
    cand = re.sub(r"[\$\\%!\s]", "", cand)
    cand = cand.replace(",", "").strip().rstrip(".")
    try:
        return float(cand)
    except ValueError:
        nums = re.findall(r"[-+]?\d*\.?\d+", cand)
        if nums:
            return float(nums[-1])
    return None

def extract_numeric_answer(text):
    """
    Extracts numerical ground truth or predicted answer from text.
    Handles \boxed{...}, #### <num>, and phrase matches with full LaTeX cleaning.
    """
    if not text:
        return None
        
    # Check for \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        val = clean_and_extract_candidate(boxed[-1])
        if val is not None:
            return val
            
    # Check for #### <num>
    hash_match = re.findall(r'####\s*([-\d.,]+)', text)
    if hash_match:
        val = clean_and_extract_candidate(hash_match[-1])
        if val is not None:
            return val
            
    # Check for explicit final answer phrasing
    ans_match = re.findall(r'(?:the answer is|final answer is|total is|equals|equal to)\s*([\$]?[-\d.,]+)', text, re.IGNORECASE)
    if ans_match:
        val = clean_and_extract_candidate(ans_match[-1])
        if val is not None:
            return val
            
    # Fallback to last number in text
    nums = re.findall(r'[-+]?\d*\.?\d+', text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
            
    return None

def run_discrete_baseline(model, tokenizer, question, max_baseline_tokens):
    """
    Baseline Discrete Generation: generates <think> tokens followed by </think> and answer.
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
            temperature=0.0,
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
    else:
        think_text = gen_text
        answer_text = ""
        think_tokens = len(gen_ids)
        ans_tokens = 0
        think_time_ms = total_time * 1000.0
        
    pred_num = extract_numeric_answer(answer_text if answer_text else gen_text)
    
    return {
        "mode": "discrete_baseline",
        "think_time_ms": round(think_time_ms, 2),
        "total_time_s": round(total_time, 3),
        "think_tokens": think_tokens,
        "answer_tokens": ans_tokens,
        "predicted_num": pred_num,
        "raw_text": gen_text
    }

def run_recurrent_generation(model, tokenizer, question, recurrent_steps, scale_factor, max_new_tokens, mode_name):
    """
    Continuous Latent Recurrence Generation:
    Replaces discrete thinking with recurrent_steps continuous forward passes, then streams answer.
    """
    messages = [{"role": "user", "content": question}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        # 1. Prefill prompt
        out = model(inputs.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        
        # 2. Recurrent continuous passes
        torch.cuda.synchronize()
        rec_t0 = time.time()
        for _ in range(recurrent_steps):
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
    pred_num = extract_numeric_answer(clean_text)
    
    return {
        "mode": mode_name,
        "think_time_ms": round(think_time_ms, 2),
        "total_time_s": round(total_time, 3),
        "think_tokens": 0,  # Zero discrete thinking tokens
        "answer_tokens": len(generated_tokens),
        "predicted_num": pred_num,
        "raw_text": full_text
    }

def main():
    args = parse_args()
    print("=== EXP-07: Evaluation Benchmark: Continuous Latent Recurrence vs Discrete CoT ===")
    print(f"Device: {DEVICE} | Samples: {args.num_samples} | Recurrent Steps: {args.recurrent_steps}")
    print(f"LoRA Adapter Path: {args.lora_path}")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Load Base Model
    print(f"Loading Base Model {MODEL_ID} in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=DEVICE
    )
    base_model.eval()
    
    # Calculate scale factor
    embed_weights = base_model.get_input_embeddings().weight.detach()
    avg_embed_norm = torch.norm(embed_weights, dim=-1).mean().item()
    hidden_dim = base_model.config.hidden_size
    theoretical_rmsnorm = hidden_dim ** 0.5
    scale_factor = avg_embed_norm / theoretical_rmsnorm
    print(f"Scale factor: {scale_factor:.6f}")
    
    # 3. Load GSM8K Test Dataset
    print("Loading GSM8K test split...")
    test_ds = load_dataset("openai/gsm8k", "main", split="test")
    test_samples = test_ds.select(range(min(args.num_samples, len(test_ds))))
    print(f"Selected {len(test_samples)} test samples.")
    
    results = []
    
    # Evaluate Baseline Discrete & Zero-shot Recurrent on Base Model
    print("\n--- Running Baseline & Zero-Shot Evaluations ---")
    base_discrete_runs = []
    zero_shot_runs = []
    
    for i, item in enumerate(test_samples):
        q = item["question"]
        gt_num = extract_numeric_answer(item["answer"])
        print(f"[{i+1}/{len(test_samples)}] Testing Base Model on: {q[:60]}... (GT: {gt_num})")
        
        # 1. Discrete Baseline
        disc_res = run_discrete_baseline(base_model, tokenizer, q, args.max_baseline_tokens)
        disc_res["question_id"] = i
        disc_res["ground_truth"] = gt_num
        disc_res["correct"] = (disc_res["predicted_num"] is not None and gt_num is not None and abs(disc_res["predicted_num"] - gt_num) < 1e-4)
        base_discrete_runs.append(disc_res)
        
        # 2. Zero-Shot Recurrent
        zs_res = run_recurrent_generation(base_model, tokenizer, q, args.recurrent_steps, scale_factor, args.max_new_tokens, "zero_shot_recurrent")
        zs_res["question_id"] = i
        zs_res["ground_truth"] = gt_num
        zs_res["correct"] = (zs_res["predicted_num"] is not None and gt_num is not None and abs(zs_res["predicted_num"] - gt_num) < 1e-4)
        zero_shot_runs.append(zs_res)
        
    # Free up memory before loading LoRA adapter
    print("\n--- Loading Trained LoRA Model ---")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_path)
    lora_model.eval()
    print(f"Loaded LoRA adapter from {args.lora_path} successfully.")
    
    lora_recurrent_runs = []
    for i, item in enumerate(test_samples):
        q = item["question"]
        gt_num = extract_numeric_answer(item["answer"])
        print(f"[{i+1}/{len(test_samples)}] Testing Trained LoRA on: {q[:60]}... (GT: {gt_num})")
        
        lora_res = run_recurrent_generation(lora_model, tokenizer, q, args.recurrent_steps, scale_factor, args.max_new_tokens, "trained_recurrent_lora")
        lora_res["question_id"] = i
        lora_res["ground_truth"] = gt_num
        lora_res["correct"] = (lora_res["predicted_num"] is not None and gt_num is not None and abs(lora_res["predicted_num"] - gt_num) < 1e-4)
        lora_recurrent_runs.append(lora_res)
        
    # Combine & Aggregate Metrics
    summary = {
        "num_samples": len(test_samples),
        "recurrent_steps": args.recurrent_steps,
        "scale_factor": scale_factor,
        "discrete_baseline": {
            "mean_think_time_ms": round(sum(r["think_time_ms"] for r in base_discrete_runs) / len(base_discrete_runs), 2),
            "mean_total_time_s": round(sum(r["total_time_s"] for r in base_discrete_runs) / len(base_discrete_runs), 3),
            "mean_think_tokens": round(sum(r["think_tokens"] for r in base_discrete_runs) / len(base_discrete_runs), 1),
            "accuracy": round(sum(1 for r in base_discrete_runs if r["correct"]) / len(base_discrete_runs) * 100.0, 2)
        },
        "zero_shot_recurrent": {
            "mean_think_time_ms": round(sum(r["think_time_ms"] for r in zero_shot_runs) / len(zero_shot_runs), 2),
            "mean_total_time_s": round(sum(r["total_time_s"] for r in zero_shot_runs) / len(zero_shot_runs), 3),
            "mean_think_tokens": 0,
            "accuracy": round(sum(1 for r in zero_shot_runs if r["correct"]) / len(zero_shot_runs) * 100.0, 2)
        },
        "trained_recurrent_lora": {
            "mean_think_time_ms": round(sum(r["think_time_ms"] for r in lora_recurrent_runs) / len(lora_recurrent_runs), 2),
            "mean_total_time_s": round(sum(r["total_time_s"] for r in lora_recurrent_runs) / len(lora_recurrent_runs), 3),
            "mean_think_tokens": 0,
            "accuracy": round(sum(1 for r in lora_recurrent_runs if r["correct"]) / len(lora_recurrent_runs) * 100.0, 2)
        },
        "speedup_thinking": round(
            sum(r["think_time_ms"] for r in base_discrete_runs) / max(1e-5, sum(r["think_time_ms"] for r in lora_recurrent_runs)),
            2
        ),
        "speedup_total": round(
            sum(r["total_time_s"] for r in base_discrete_runs) / max(1e-5, sum(r["total_time_s"] for r in lora_recurrent_runs)),
            2
        ),
        "detailed_runs": {
            "discrete_baseline": base_discrete_runs,
            "zero_shot_recurrent": zero_shot_runs,
            "trained_recurrent_lora": lora_recurrent_runs
        }
    }
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY RESULTS")
    print("="*80)
    print(f"{'Metric':<30} | {'Discrete CoT':<15} | {'Zero-Shot Rec':<15} | {'Trained Rec LoRA':<15}")
    print("-"*80)
    print(f"{'Thinking Latency (ms)':<30} | {summary['discrete_baseline']['mean_think_time_ms']:<15.1f} | {summary['zero_shot_recurrent']['mean_think_time_ms']:<15.1f} | {summary['trained_recurrent_lora']['mean_think_time_ms']:<15.1f}")
    print(f"{'Total Latency (s)':<30} | {summary['discrete_baseline']['mean_total_time_s']:<15.2f} | {summary['zero_shot_recurrent']['mean_total_time_s']:<15.2f} | {summary['trained_recurrent_lora']['mean_total_time_s']:<15.2f}")
    print(f"{'Thinking Tokens':<30} | {summary['discrete_baseline']['mean_think_tokens']:<15.1f} | {0:<15} | {0:<15}")
    print(f"{'Accuracy (%)':<30} | {summary['discrete_baseline']['accuracy']:<15.1f}% | {summary['zero_shot_recurrent']['accuracy']:<15.1f}% | {summary['trained_recurrent_lora']['accuracy']:<15.1f}%")
    print(f"{'Thinking Speedup':<30} | {'1.0x':<15} | {summary['discrete_baseline']['mean_think_time_ms']/summary['zero_shot_recurrent']['mean_think_time_ms']:<15.1f}x | {summary['speedup_thinking']:<15.1f}x")
    print("="*80)
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
