#!/usr/bin/env python3
"""
scripts/55_batched_eval_arm3.py
High-Throughput Batched Evaluator for Arm 3 (Continuous Latent Recurrence).

Eliminates the sequential batch-size-1 bottleneck by evaluating the 250-problem suite
across 4 seeds in batches of B=16 on the RTX PRO 4500 (32GB VRAM).
Reduces Arm 3 evaluation runtime from 4+ hours down to ~5-8 minutes.
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from math_verify import parse, verify

CALIBRATED_ALPHA = 0.011440
THINK_END_TOKEN = "</think>"
SEEDS = [42, 123, 456, 789]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def sample_tokens_batch(logits, temperature=0.6, top_p=0.95, top_k=20):
    """Batched stochastic sampling (B, V) -> (B, 1)."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    logits = logits / temperature
    # Top-k filter
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        min_v = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_v, torch.full_like(logits, -float("inf")), logits)
        
    # Top-p filter
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("inf"))
        
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def eval_batch_arm3(model, tokenizer, batch_probs, k_steps, scale_factor, max_ans_tokens, device):
    """
    Evaluates a batch of B problems under Arm 3 continuous latent recurrence.
    """
    B = len(batch_probs)
    prompts = [p["question"] for p in batch_probs]
    
    formatted_prompts = []
    for q in prompts:
        msg = [{"role": "user", "content": q}]
        formatted_prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=True))
        
    tokenizer.padding_side = "left"
    enc = tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(device)
    
    t0 = time.time()
    with torch.no_grad():
        out = model(**enc, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        
        # K latent steps across entire batch
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
            
        t_think_s = time.time() - t0
        
        # Transition token </think>\n\n
        trans_text = "\n</think>\n\n"
        trans_tokens = tokenizer.encode(trans_text, add_special_tokens=False)
        trans_ids = torch.tensor([trans_tokens] * B, dtype=torch.long, device=device)
        step_out = model(input_ids=trans_ids, past_key_values=past_kv, use_cache=True)
        past_kv = step_out.past_key_values
        
        # Batched autoregressive decoding
        curr_logits = step_out.logits[:, -1, :]
        next_tokens = sample_tokens_batch(curr_logits, temperature=0.6, top_p=0.95, top_k=20)
        
        gen_tokens = [[] for _ in range(B)]
        finished = [False] * B
        eos_id = tokenizer.eos_token_id
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        
        for b in range(B):
            tok = next_tokens[b].item()
            if tok in [eos_id, im_end_id]:
                finished[b] = True
            else:
                gen_tokens[b].append(tok)
                
        curr_ids = next_tokens
        
        for step in range(max_ans_tokens):
            if all(finished):
                break
            step_out = model(input_ids=curr_ids, past_key_values=past_kv, use_cache=True)
            past_kv = step_out.past_key_values
            curr_logits = step_out.logits[:, -1, :]
            next_tokens = sample_tokens_batch(curr_logits, temperature=0.6, top_p=0.95, top_k=20)
            
            for b in range(B):
                if not finished[b]:
                    tok = next_tokens[b].item()
                    if tok in [eos_id, im_end_id]:
                        finished[b] = True
                    else:
                        gen_tokens[b].append(tok)
                        
            curr_ids = next_tokens
            
    total_time_s = time.time() - t0
    
    results = []
    for b in range(B):
        prob = batch_probs[b]
        content = tokenizer.decode(gen_tokens[b], skip_special_tokens=True).strip()
        gt_sol = prob["solution"]
        gt_boxed = extract_math_boxed_expression(gt_sol)
        
        # Verify correctness with math_verify
        try:
            gt_parsed = parse(f"\\boxed{{{gt_boxed or gt_sol}}}")
            pred_parsed = parse(content)
            is_correct = verify(gt_parsed, pred_parsed)
        except Exception:
            is_correct = False
            
        results.append({
            "id": prob["id"],
            "benchmark": prob.get("benchmark", ""),
            "is_correct": is_correct,
            "gen_tokens": len(gen_tokens[b]),
            "is_truncated": not finished[b],
            "think_time_ms": round(t_think_s * 1000 / B, 1),
            "total_time_s": round(total_time_s / B, 2),
            "content": content
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Batched Arm 3 Evaluator")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--k_tokens", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    suite_path = os.path.join(data_dir, "benchmark_suite_250.json")
    with open(suite_path) as f:
        problems = json.load(f)
        
    tag = args.model_id.replace("/", "_").lower()
    if args.adapter_path is None:
        args.adapter_path = os.path.join(
            os.path.dirname(__file__), "..", "checkpoints", f"lora_arm3_{tag}_k{args.k_tokens}_selfdistill"
        )
    if args.output_file is None:
        args.output_file = os.path.join(data_dir, f"eval_arm3_{tag}_k{args.k_tokens}_batched.json")
        
    print(f"=== Running Batched Arm 3 Evaluation on {args.device} ===")
    print(f"Model: {args.model_id} | Adapter: {args.adapter_path}")
    print(f"Batch Size: {args.batch_size} | K: {args.k_tokens} | Seeds: {SEEDS}")
    print(f"Problems: {len(problems)} | Total Evaluations: {len(problems) * len(SEEDS)}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading base model in pure bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    
    if os.path.exists(args.adapter_path):
        print(f"Loading LoRA adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print(f"Warning: Adapter {args.adapter_path} not found. Evaluating base model with recurrent loop.")
        model = base_model
        
    model.eval()
    
    all_runs = []
    t_global_start = time.time()
    
    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        set_seed(seed)
        seed_results = []
        
        # Batch evaluation
        for i in range(0, len(problems), args.batch_size):
            batch = problems[i:i + args.batch_size]
            batch_res = eval_batch_arm3(
                model, tokenizer, batch,
                k_steps=args.k_tokens,
                scale_factor=CALIBRATED_ALPHA,
                max_ans_tokens=512,
                device=args.device
            )
            seed_results.extend(batch_res)
            correct_so_far = sum(1 for r in seed_results if r["is_correct"])
            print(f"Seed {seed} | Batch [{i + len(batch)}/{len(problems)}] | Acc: {100 * correct_so_far / len(seed_results):.1f}%")
            
        all_runs.append({"seed": seed, "results": seed_results})
        
    total_time = time.time() - t_global_start
    total_evals = len(problems) * len(SEEDS)
    total_correct = sum(sum(1 for r in run["results"] if r["is_correct"]) for run in all_runs)
    mean_acc = 100 * total_correct / total_evals
    
    print(f"\n========================================================")
    print(f"Arm 3 Batched Evaluation Complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total Evaluations: {total_evals} | Mean Accuracy: {mean_acc:.2f}%")
    print(f"Throughput: {total_evals / total_time:.2f} problems/sec")
    print(f"========================================================")
    
    output_data = {
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "k_tokens": args.k_tokens,
        "batch_size": args.batch_size,
        "mean_accuracy": round(mean_acc, 2),
        "total_time_s": round(total_time, 2),
        "all_runs": all_runs
    }
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved evaluation results to: {args.output_file}")

if __name__ == "__main__":
    main()
