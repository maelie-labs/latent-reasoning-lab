#!/usr/bin/env python3
"""
scripts/99b_test_latent_depth_extrapolation.py
Test Zero-Shot Depth Extrapolation and Recurrence Scaling for the Register-Bundle Ladder.

Evaluates the trained production model (checkpoints/lora_arm3_register_ladder_production_qwen3_1.7b/best_checkpoint)
across varying latent unroll depths per bundle:
K \in [4, 6, 8, 12, 16] (Total Latents = 3 * K \in [12, 18, 24, 36, 48])
on the 60 MATH Level 3-5 problems.

Measures:
1. Pass@1 accuracy at each depth K
2. Latent hidden state norm ||h_k|| dynamics across rungs
3. Cosine similarity drift per step: cos(h_k, h_{k-1})
4. Identifies whether additional continuous compute improves reasoning on hard math.
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
from math_verify import parse, verify
from tqdm import tqdm

CALIBRATED_ALPHA = 0.011440
NUM_RUNGS = 3

CANONICAL_HEADERS = [
    "[R1: Identify givens, constraints, and target variable]\n",
    "[R2: Compute intermediate operations and verify relations]\n",
    "[R3: Execute final deduction and verify constraints]\n"
]

def check_correctness(prediction_text, ground_truth):
    try:
        cand = parse(prediction_text, parsing_timeout=None)
        gold = parse(f"\\boxed{{{ground_truth}}}", parsing_timeout=None)
        return float(verify(gold, cand))
    except Exception:
        nums = re.findall(r"[-+]?\d*\.?\d+", prediction_text)
        if nums and nums[-1] == str(ground_truth):
            return 1.0
        return 0.0

def run_depth_inference(model, tokenizer, question, k_per_bundle, device, max_ans_tokens=512):
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    L_prompt = enc_prompt.input_ids.shape[1]

    latent_norms = []
    cosine_drifts = []

    with torch.no_grad():
        out = model(input_ids=enc_prompt.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        curr_seq_len = L_prompt

        for r_idx in range(NUM_RUNGS):
            hdr_text = CANONICAL_HEADERS[r_idx]
            enc_hdr = tokenizer(hdr_text, return_tensors="pt", add_special_tokens=False).to(device)
            hdr_len = enc_hdr.input_ids.shape[1]
            hdr_pos = torch.arange(curr_seq_len, curr_seq_len + hdr_len, device=device).unsqueeze(0)

            hdr_out = model(input_ids=enc_hdr.input_ids, position_ids=hdr_pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
            past_kv = hdr_out.past_key_values
            curr_latent = hdr_out.hidden_states[-1][:, -1:, :]
            curr_seq_len += hdr_len

            prev_latent = None
            for k in range(k_per_bundle):
                step_pos = torch.tensor([[curr_seq_len]], device=device)
                norm_val = curr_latent.norm(dim=-1).item()
                latent_norms.append(norm_val)

                if prev_latent is not None:
                    cos_sim = torch.cosine_similarity(curr_latent.squeeze(1), prev_latent.squeeze(1), dim=-1).item()
                    cosine_drifts.append(cos_sim)
                prev_latent = curr_latent.clone()

                scaled = curr_latent * CALIBRATED_ALPHA
                step_out = model(inputs_embeds=scaled, position_ids=step_pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
                past_kv = step_out.past_key_values
                curr_latent = step_out.hidden_states[-1][:, -1:, :]
                curr_seq_len += 1

        # Transition: \n</think>\n\n
        trans_text = "\n</think>\n\n"
        enc_trans = tokenizer(trans_text, return_tensors="pt", add_special_tokens=False).to(device)
        trans_len = enc_trans.input_ids.shape[1]
        trans_pos = torch.arange(curr_seq_len, curr_seq_len + trans_len, device=device).unsqueeze(0)
        trans_out = model(input_ids=enc_trans.input_ids, position_ids=trans_pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
        past_kv = trans_out.past_key_values
        curr_seq_len += trans_len

        # Greedy decoding
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

        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return {
            "text": gen_text,
            "latent_norms": latent_norms,
            "cosine_drifts": cosine_drifts
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lora_arm3_register_ladder_production_qwen3_1.7b/best_checkpoint")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--output_file", type=str, default="data/depth_extrapolation_study_qwen3_1.7b.json")
    args = parser.parse_args()

    print("=" * 80)
    print("RECURRENCE SCALING & DEPTH EXTRAPOLATION STUDY")
    print(f"Model: {args.model_id} | Checkpoint: {args.checkpoint} | Device: {args.device}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
    print(f"Loaded {len(math_l35)} MATH-500 Level 3-5 problems.")

    depths = [4, 6, 8, 12, 16]
    depth_results = {}

    for k in depths:
        total_latents = NUM_RUNGS * k
        print(f"\n--- Evaluating Depth K={k} per bundle ({total_latents} total latents) ---")
        correct = 0
        all_norms = []
        all_cosines = []

        for p in tqdm(math_l35, desc=f"Depth K={k}"):
            gold = p.get("solution") or p.get("answer", "")
            res = run_depth_inference(model, tokenizer, p["question"], k, args.device, max_ans_tokens=512)
            c = check_correctness(res["text"], gold)
            correct += c
            all_norms.extend(res["latent_norms"])
            all_cosines.extend(res["cosine_drifts"])

        acc = (correct / len(math_l35)) * 100.0
        mean_norm = float(np.mean(all_norms))
        max_norm = float(np.max(all_norms))
        mean_cos = float(np.mean(all_cosines)) if all_cosines else 1.0

        print(f"Result for K={k} ({total_latents} latents):")
        print(f"  * Accuracy   : {acc:5.2f}% ({int(correct)}/{len(math_l35)})")
        print(f"  * Mean Norm  : {mean_norm:.2f} (Max: {max_norm:.2f})")
        print(f"  * Mean Cosine: {mean_cos:.4f}")

        depth_results[f"K={k}"] = {
            "k_per_bundle": k,
            "total_latents": total_latents,
            "accuracy": acc,
            "correct": int(correct),
            "total": len(math_l35),
            "mean_norm": mean_norm,
            "max_norm": max_norm,
            "mean_cosine_sim": mean_cos
        }

    # Save summary
    out_data = {
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
        "tested_depths": depths,
        "results": depth_results
    }
    with open(args.output_file, "w") as f:
        json.dump(out_data, f, indent=2)

    print("\n" + "=" * 80)
    print("DEPTH EXTRAPOLATION SUMMARY")
    print("=" * 80)
    print(f"{'Depth (K/bundle)':<18} | {'Total Latents':<14} | {'Accuracy':<10} | {'Mean Norm':<12} | {'Mean Cosine':<12}")
    print("-" * 80)
    for k in depths:
        r = depth_results[f"K={k}"]
        print(f"K={r['k_per_bundle']:<16} | {r['total_latents']:<14} | {r['accuracy']:5.2f}%    | {r['mean_norm']:<12.2f} | {r['mean_cosine_sim']:<12.4f}")

    print(f"\nFull results saved to {args.output_file}")

if __name__ == "__main__":
    main()
