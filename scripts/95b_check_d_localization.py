#!/usr/bin/env python3
"""
scripts/95b_check_d_localization.py
Quantitative Measurement of Header-Latent Binding & Localized Damage on cuda:0.

Operationalization:
1. Extract ordered intermediate and final numbers from baseline generation.
2. Partition numbers into 3 segments corresponding to the 3 rungs.
3. For noise injected into Bundle b (1, 2, 3), find the first changed numeric value.
4. Classify each trial into:
   - Localized Damage: First mismatch occurs in Segment b.
   - Premature Damage: First mismatch occurs in Segment < b (anomalous).
   - Delayed / Diffuse Damage: First mismatch occurs in Segment > b.
   - Immune / No Change: Exact numeric match preserved.
"""

import os
import re
import sys
import json
import time
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

CALIBRATED_ALPHA = 0.011440  # Qwen3-1.7B
K_PER_BUNDLE = 8
NUM_RUNGS = 3

def extract_ordered_numbers(text: str):
    """Extracts ordered list of numerical strings from text."""
    if not text:
        return []
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    cleaned = []
    for m in matches:
        if m in [".", "+", "-"]:
            continue
        try:
            val = float(m)
            cleaned.append(int(val) if val.is_integer() else val)
        except ValueError:
            pass
    return cleaned

def get_headers(sample):
    ans = sample.get("solution") or sample.get("nonthinking_answer") or ""
    steps = re.findall(r"###\s*(Step \d+:[^\n]+)", ans)
    default_headers = [
        "Identify givens, constraints, and target variable",
        "Compute intermediate operations and verify relations",
        "Execute final deduction and verify constraints"
    ]
    headers = []
    for r in range(NUM_RUNGS):
        if r < len(steps):
            clean_step = steps[r].strip().replace(":", " -")
            headers.append(f"[R{r+1}: {clean_step}]\n")
        else:
            headers.append(f"[R{r+1}: {default_headers[r]}]\n")
    return headers

def run_single_ladder_inference(
    model, tokenizer, question, headers, device,
    noise_bundle=None, max_ans_tokens=256
):
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    L_prompt = enc_prompt.input_ids.shape[1]
    
    with torch.inference_mode():
        out = model(input_ids=enc_prompt.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        curr_seq_len = L_prompt
        
        backbone = model.base_model.model.model
        
        for r, hdr_text in enumerate(headers):
            rung_idx = r + 1
            enc_hdr = tokenizer(hdr_text, return_tensors="pt", add_special_tokens=False).to(device)
            hdr_len = enc_hdr.input_ids.shape[1]
            hdr_pos = torch.arange(curr_seq_len, curr_seq_len + hdr_len, device=device).unsqueeze(0)
            
            hdr_out = model(input_ids=enc_hdr.input_ids, position_ids=hdr_pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
            past_kv = hdr_out.past_key_values
            curr_latent = hdr_out.hidden_states[-1][:, -1:, :]
            curr_seq_len += hdr_len
            
            for k in range(K_PER_BUNDLE):
                step_pos = torch.tensor([[curr_seq_len]], device=device, dtype=torch.long)
                if noise_bundle == rung_idx:
                    sigma = curr_latent.norm(dim=-1, keepdim=True) / np.sqrt(curr_latent.shape[-1])
                    noise = torch.randn_like(curr_latent) * sigma
                    curr_latent = curr_latent + noise
                    
                scaled = curr_latent * CALIBRATED_ALPHA
                step_out = backbone(inputs_embeds=scaled, position_ids=step_pos, past_key_values=past_kv, use_cache=True)
                past_kv = step_out.past_key_values
                curr_latent = step_out.last_hidden_state[:, -1:, :]
                curr_seq_len += 1
                
        trans_text = "\n</think>\n\n"
        enc_trans = tokenizer(trans_text, return_tensors="pt", add_special_tokens=False).to(device)
        trans_len = enc_trans.input_ids.shape[1]
        trans_pos = torch.arange(curr_seq_len, curr_seq_len + trans_len, device=device).unsqueeze(0)
        trans_out = model(input_ids=enc_trans.input_ids, position_ids=trans_pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
        past_kv = trans_out.past_key_values
        curr_seq_len += trans_len
        
        curr_token = torch.argmax(trans_out.logits[:, -1:, :], dim=-1)
        gen_tokens = [curr_token.item()]
        curr_past = past_kv
        curr_pos = curr_seq_len
        
        for g in range(max_ans_tokens):
            step_g = model(input_ids=curr_token, position_ids=torch.tensor([[curr_pos]], device=device), past_key_values=curr_past, use_cache=True)
            curr_past = step_g.past_key_values
            curr_pos += 1
            curr_token = torch.argmax(step_g.logits[:, -1:, :], dim=-1)
            gen_tokens.append(curr_token.item())
            if curr_token.item() == tokenizer.eos_token_id:
                break
                
        return tokenizer.decode(gen_tokens, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/test_register_ladder_quick")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_file", type=str, default="data/check_d_quantitative_localization.json")
    args = parser.parse_args()

    print("=" * 80)
    print("CHECK D: QUANTITATIVE LOCALIZATION SUITE")
    print(f"Device: {args.device} | Checkpoint: {args.checkpoint}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=args.device,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    with open("data/benchmark_suite_250.json") as f:
        suite = json.load(f)
    math_l35 = [p for p in suite if p.get("benchmark") == "MATH-500" and p.get("stratum") in ["Level 3", "Level 4", "Level 5"]]
    print(f"Loaded {len(math_l35)} MATH-500 L3-5 problems.")

    stats = {
        1: {"localized": 0, "premature": 0, "delayed": 0, "no_change": 0, "too_few_nums": 0, "total": 0},
        2: {"localized": 0, "premature": 0, "delayed": 0, "no_change": 0, "too_few_nums": 0, "total": 0},
        3: {"localized": 0, "premature": 0, "delayed": 0, "no_change": 0, "too_few_nums": 0, "total": 0}
    }

    per_problem_details = []

    for idx, p in enumerate(tqdm(math_l35, desc="Quantitative Localization")):
        q = p["question"]
        hdrs = get_headers(p)
        
        base_text = run_single_ladder_inference(model, tokenizer, q, hdrs, args.device, noise_bundle=None)
        base_nums = extract_ordered_numbers(base_text)
        M = len(base_nums)
        
        prob_record = {
            "problem_id": p.get("id") or f"prob_{idx}",
            "base_text": base_text,
            "base_nums": base_nums,
            "bundle_evals": {}
        }
        
        if M < 3:
            for b in [1, 2, 3]:
                stats[b]["too_few_nums"] += 1
                stats[b]["total"] += 1
            prob_record["note"] = f"Skipped segmentation: only {M} numbers in baseline"
            per_problem_details.append(prob_record)
            continue
            
        s1 = max(1, M // 3)
        s2 = max(s1 + 1, (2 * M) // 3)
        
        for b in [1, 2, 3]:
            corrupt_text = run_single_ladder_inference(model, tokenizer, q, hdrs, args.device, noise_bundle=b)
            corrupt_nums = extract_ordered_numbers(corrupt_text)
            
            diff_idx = None
            for i in range(max(len(base_nums), len(corrupt_nums))):
                b_val = base_nums[i] if i < len(base_nums) else None
                c_val = corrupt_nums[i] if i < len(corrupt_nums) else None
                if b_val != c_val:
                    diff_idx = i
                    break
                    
            stats[b]["total"] += 1
            
            if diff_idx is None:
                category = "no_change"
                stats[b]["no_change"] += 1
            else:
                if diff_idx < s1:
                    seg = 1
                elif diff_idx < s2:
                    seg = 2
                else:
                    seg = 3
                    
                if seg == b:
                    category = "localized"
                    stats[b]["localized"] += 1
                elif seg < b:
                    category = "premature"
                    stats[b]["premature"] += 1
                else:
                    category = "delayed"
                    stats[b]["delayed"] += 1
                    
            prob_record["bundle_evals"][b] = {
                "category": category,
                "first_diff_idx": diff_idx,
                "corrupt_text_snippet": corrupt_text[:100]
            }
            
        per_problem_details.append(prob_record)

    print("\n" + "=" * 80)
    print("CHECK D: QUANTITATIVE LOCALIZATION RESULTS")
    print("=" * 80)
    for b in [1, 2, 3]:
        tot = stats[b]["total"]
        valid_tot = tot - stats[b]["too_few_nums"]
        loc_pct = (stats[b]["localized"] / valid_tot) * 100.0 if valid_tot > 0 else 0
        pre_pct = (stats[b]["premature"] / valid_tot) * 100.0 if valid_tot > 0 else 0
        del_pct = (stats[b]["delayed"] / valid_tot) * 100.0 if valid_tot > 0 else 0
        no_pct = (stats[b]["no_change"] / valid_tot) * 100.0 if valid_tot > 0 else 0
        print(f"Bundle {b} (Rung {b}):")
        print(f"  * Localized in Segment {b} (Targeted) : {loc_pct:5.2f}% ({stats[b]['localized']}/{valid_tot})")
        print(f"  * Premature in Earlier Seg (Anomalous): {pre_pct:5.2f}% ({stats[b]['premature']}/{valid_tot})")
        print(f"  * Delayed in Later Seg                : {del_pct:5.2f}% ({stats[b]['delayed']}/{valid_tot})")
        print(f"  * Immune / No Change                  : {no_pct:5.2f}% ({stats[b]['no_change']}/{valid_tot})")
        print(f"  * Too Few Numbers (< 3)               : {stats[b]['too_few_nums']}/{tot}")

    with open(args.output_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": stats,
            "details": per_problem_details
        }, f, indent=2)
    print(f"\nSaved quantitative results to {args.output_file}")

if __name__ == "__main__":
    main()
