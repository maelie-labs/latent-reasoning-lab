#!/usr/bin/env python3
"""
scripts/88_gate2_causal_patching_v1_1.py
Gate 2: Check 3 Causal Activation Patching & Steering Assertion for v1.1 Arm 3.

Vectorized, High-Throughput Batched Implementation (Batch Size B=16).
Fully saturates RTX PRO 4500 (32GB) VRAM and Tensor Cores.
Evaluates Control B Causal Donor Steering on N=100 mathematical problems.

Pre-Registered Graded Decision Rule:
- Delta Steer >= +40.0%: PASS (Strong semantic grounding confirmed -> Proceed to 250-suite).
- 10.0% <= Delta Steer < 40.0%: PARTIAL (Run 250-suite, report partial grounding).
- Delta Steer < +10.0%: HALT (Latent channel mechanically uncoupled / bypassed -> Abort).
"""

import os
import re
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

CALIBRATED_ALPHAS = {
    "qwen/qwen3-1.7b": 0.011440,
    "qwen/qwen3.5-2b": 0.009800,
    "qwen/qwen3-4b": 0.007670,
}

def extract_all_numbers(text: str) -> set:
    if not text:
        return set()
    raw = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", text)
    cleaned = set()
    for n in raw:
        try:
            val = float(n.replace(",", ""))
            cleaned.add(int(val) if val.is_integer() else val)
        except ValueError:
            pass
    return cleaned

def run_batched_recurrent_unroll(
    model, tokenizer, prompts, scale_factor, k_steps,
    patch_step=None, patch_vectors=None, device="cuda:0",
    max_ans_tokens=512, return_intermediate_step=None
):
    """
    Vectorized batched recurrent unroll and greedy answer generation.
    """
    B = len(prompts)
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        pos_prefill = enc.attention_mask.long().cumsum(-1) - 1
        pos_prefill.masked_fill_(enc.attention_mask == 0, 0)
        out = model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            position_ids=pos_prefill,
            use_cache=True,
            output_hidden_states=True
        )
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        seq_lens = enc.attention_mask.sum(dim=-1)
        curr_mask = enc.attention_mask

        saved_intermediate = None
        for k in range(1, k_steps + 1):
            step_pos = (seq_lens + k - 1).unsqueeze(1)
            curr_mask = torch.cat([curr_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1)
            scaled_latent = curr_latent * scale_factor
            step_out = model(
                inputs_embeds=scaled_latent,
                attention_mask=curr_mask,
                position_ids=step_pos,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]

            if patch_step == k and patch_vectors is not None:
                curr_latent = patch_vectors.to(device).to(curr_latent.dtype)

            if return_intermediate_step == k:
                saved_intermediate = curr_latent.clone()

        if max_ans_tokens == 0:
            return {
                "texts": [""] * B,
                "intermediate_state": saved_intermediate
            }

        # Transition tokens: \n</think>\n\n
        trans_tokens = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
        trans_len = len(trans_tokens)
        trans_ids = torch.tensor([trans_tokens] * B, dtype=torch.long, device=device)
        trans_mask = torch.cat([curr_mask, torch.ones((B, trans_len), dtype=torch.long, device=device)], dim=1)
        trans_pos = torch.stack([torch.arange(trans_len, device=device) + (seq_lens[b] + k_steps) for b in range(B)], dim=0)

        trans_out = model(
            input_ids=trans_ids,
            attention_mask=trans_mask,
            position_ids=trans_pos,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True
        )
        past_kv = trans_out.past_key_values
        curr_mask = trans_mask
        curr_token = torch.argmax(trans_out.logits[:, -1:, :], dim=-1)

        gen_tokens = [[] for _ in range(B)]
        finished = [False] * B
        curr_pos = seq_lens + k_steps + trans_len

        for step in range(max_ans_tokens):
            for b in range(B):
                if not finished[b]:
                    gen_tokens[b].append(curr_token[b, 0].item())
                    if curr_token[b, 0].item() == tokenizer.eos_token_id:
                        finished[b] = True
            if all(finished):
                break

            step_pos = (curr_pos + step).unsqueeze(1)
            curr_mask = torch.cat([curr_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1)
            step_out = model(
                input_ids=curr_token,
                attention_mask=curr_mask,
                position_ids=step_pos,
                past_key_values=past_kv,
                use_cache=True
            )
            past_kv = step_out.past_key_values
            curr_token = torch.argmax(step_out.logits[:, -1:, :], dim=-1)

        texts = [tokenizer.decode(toks, skip_special_tokens=True).strip() for toks in gen_tokens]

        return {
            "texts": texts,
            "intermediate_state": saved_intermediate
        }

def main():
    parser = argparse.ArgumentParser(description="Gate 2: Batched Causal Patching and Steering Assertion")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--n_problems", type=int, default=100)
    parser.add_argument("--k_steps", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for unrolling (default: 1 for bitwise determinism)")
    parser.add_argument("--patch_step", type=int, default=None, help="Recurrent step to patch (default: k_steps // 2)")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    scale_factor = CALIBRATED_ALPHAS.get(args.model_id.lower(), 0.011440)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    suite_path = os.path.join(data_dir, "benchmark_suite_250.json")

    print(f"=== GATE 2: Batched Causal Donor Steering Assertion for v1.1 Arm 3 ===")
    print(f"Checkpoint: {args.checkpoint} on {args.device} | N={args.n_problems} Problems | Batch Size: {args.batch_size}")

    with open(suite_path) as f:
        suite = json.load(f)
    problems = suite[:args.n_problems]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if os.path.exists(os.path.join(args.checkpoint, "adapter_config.json")):
        print(f"Loading LoRA adapter from: {args.checkpoint}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=args.device,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
    else:
        print(f"Loading full fine-tuned model directly from: {args.checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=args.device,
            trust_remote_code=True
        )
    model.eval()

    mid_t = args.patch_step if args.patch_step is not None else max(1, args.k_steps // 2)

    # Phase 1: Pre-unroll all N problems in batches to capture clean texts and h_mid vectors
    print(f"\n[Phase 1] Pre-unrolling {len(problems)} clean problems in batches of {args.batch_size}...")
    clean_prompts = [
        f"<|im_start|>user\n{p['question']}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        for p in problems
    ]
    clean_texts = []
    clean_h_mid_list = []

    for b_start in tqdm(range(0, len(clean_prompts), args.batch_size), desc="Clean Unrolls"):
        b_prompts = clean_prompts[b_start : b_start + args.batch_size]
        res = run_batched_recurrent_unroll(
            model, tokenizer, b_prompts, scale_factor, args.k_steps,
            return_intermediate_step=mid_t, device=args.device, max_ans_tokens=512
        )
        clean_texts.extend(res["texts"])
        clean_h_mid_list.append(res["intermediate_state"])

    clean_h_mid = torch.cat(clean_h_mid_list, dim=0)  # (N, 1, d_model)
    clean_nums = [extract_all_numbers(t) for t in clean_texts]

    # Phase 2: Create deterministic counterfactual donor pairs (seed 42)
    random.seed(42)
    indices = list(range(len(problems)))
    shuffled = list(indices)
    while any(i == j for i, j in zip(indices, shuffled)):
        random.shuffle(shuffled)

    donor_pairs = [(indices[idx], shuffled[idx]) for idx in range(len(indices))]

    # Phase 3: Filter valid trials and evaluate chance baseline
    valid_trials_data = []
    chance_hits = 0

    for r_idx, d_idx in donor_pairs:
        prob_R = problems[r_idx]
        prob_D = problems[d_idx]

        sol_D = prob_D.get("solution") or prob_D.get("reference_answer", "")
        nums_D = clean_nums[d_idx] | extract_all_numbers(sol_D)
        nums_R_prompt = extract_all_numbers(prob_R["question"])
        nums_D_filtered = {n for n in nums_D if n > 20 and n not in nums_R_prompt}

        if not nums_D_filtered:
            continue

        if nums_D_filtered.intersection(clean_nums[r_idx]):
            chance_hits += 1

        valid_trials_data.append({
            "r_idx": r_idx,
            "d_idx": d_idx,
            "nums_D_filtered": nums_D_filtered
        })

    valid_count = len(valid_trials_data)
    print(f"\n[Phase 2 & 3] Filtered {valid_count} valid trials out of {len(donor_pairs)} pairs.")

    # Phase 4: Control 0 (Null Patch Check) - Batched verification
    print(f"\n[Phase 4] Evaluating Control 0 (Null Patch Identity) on {valid_count} valid trials...")
    null_identical = 0
    null_prompts = [clean_prompts[item["r_idx"]] for item in valid_trials_data]
    null_vectors = clean_h_mid[[item["r_idx"] for item in valid_trials_data]]

    for b_start in tqdm(range(0, valid_count, args.batch_size), desc="Null Patch"):
        b_prompts = null_prompts[b_start : b_start + args.batch_size]
        b_vectors = null_vectors[b_start : b_start + args.batch_size]
        res = run_batched_recurrent_unroll(
            model, tokenizer, b_prompts, scale_factor, args.k_steps,
            patch_step=mid_t, patch_vectors=b_vectors, device=args.device, max_ans_tokens=512
        )
        for i, text in enumerate(res["texts"]):
            global_idx = b_start + i
            orig_r_idx = valid_trials_data[global_idx]["r_idx"]
            if text == clean_texts[orig_r_idx]:
                null_identical += 1

    # Phase 5: Finite-Difference Sensitivity Check (Batched, max_ans_tokens=0 for instant execution)
    print(f"\n[Phase 5] Evaluating Finite-Difference Sensitivity on {valid_count} valid trials...")
    amplifications = []
    torch.manual_seed(42)

    for b_start in tqdm(range(0, valid_count, args.batch_size), desc="Sensitivity"):
        b_prompts = null_prompts[b_start : b_start + args.batch_size]
        b_vectors = null_vectors[b_start : b_start + args.batch_size]
        
        eps_dir = torch.randn_like(b_vectors)
        norms = torch.norm(b_vectors, dim=-1, keepdim=True)
        eps = 1e-3 * norms * (eps_dir / torch.norm(eps_dir, dim=-1, keepdim=True))

        if mid_t < args.k_steps:
            # Unroll clean to step mid_t + 1
            clean_step_res = run_batched_recurrent_unroll(
                model, tokenizer, b_prompts, scale_factor, args.k_steps,
                return_intermediate_step=mid_t + 1, device=args.device, max_ans_tokens=0
            )
            # Unroll perturbed to step mid_t + 1
            pert_step_res = run_batched_recurrent_unroll(
                model, tokenizer, b_prompts, scale_factor, args.k_steps,
                patch_step=mid_t, patch_vectors=b_vectors + eps,
                return_intermediate_step=mid_t + 1, device=args.device, max_ans_tokens=0
            )
            h_pert = pert_step_res["intermediate_state"]
            h_clean = clean_step_res["intermediate_state"]
            diff_norm = torch.norm(h_pert - h_clean, dim=-1)
            eps_norm = torch.norm(eps, dim=-1)
            amps = (diff_norm / eps_norm).squeeze(-1).tolist()
            amplifications.extend(amps)
        else:
            # At terminal step k_steps, measure next-step projection sensitivity through lm_head
            logits_clean = model.lm_head(b_vectors)
            logits_pert = model.lm_head(b_vectors + eps)
            diff_norm = torch.norm(logits_pert - logits_clean, dim=-1)
            eps_norm = torch.norm(eps, dim=-1)
            amps = (diff_norm / eps_norm).squeeze(-1).tolist()
            amplifications.extend(amps)

    # Phase 6: Control B (Steered Recipient Runs) - Batched unroll with donor vectors
    print(f"\n[Phase 6] Evaluating Control B Donor Steering on {valid_count} valid trials...")
    steered_hits = 0
    steer_prompts = [clean_prompts[item["r_idx"]] for item in valid_trials_data]
    steer_vectors = clean_h_mid[[item["d_idx"] for item in valid_trials_data]]

    for b_start in tqdm(range(0, valid_count, args.batch_size), desc="Steered Unrolls"):
        b_prompts = steer_prompts[b_start : b_start + args.batch_size]
        b_vectors = steer_vectors[b_start : b_start + args.batch_size]
        res = run_batched_recurrent_unroll(
            model, tokenizer, b_prompts, scale_factor, args.k_steps,
            patch_step=mid_t, patch_vectors=b_vectors, device=args.device, max_ans_tokens=512
        )
        for i, text in enumerate(res["texts"]):
            global_idx = b_start + i
            nums_steered = extract_all_numbers(text)
            if valid_trials_data[global_idx]["nums_D_filtered"].intersection(nums_steered):
                steered_hits += 1

    # Summary & Decision Rule Calculation
    p_chance = (chance_hits / valid_count) * 100.0 if valid_count > 0 else 0.0
    p_steered = (steered_hits / valid_count) * 100.0 if valid_count > 0 else 0.0
    delta_steer = p_steered - p_chance
    mean_amp = float(np.mean(amplifications)) if amplifications else 0.0
    null_rate = (null_identical / valid_count) * 100.0 if valid_count > 0 else 0.0

    print(f"\n=======================================================")
    print(f"GATE 2 CAUSAL PATCHING RESULTS (Valid Trials: {valid_count})")
    print(f"Control 0 (Null Patch Identity): {null_identical}/{valid_count} ({null_rate:.1f}%)")
    print(f"Finite-Difference Amplification: {mean_amp:.2f}x (v1.0 baseline: 17.31x)")
    print(f"P_chance: {p_chance:.2f}% | P_steered: {p_steered:.2f}%")
    print(f"NET STEERING DELTA: Delta Steer = {delta_steer:+.2f}%")
    print(f"Pre-Registered Threshold: >= +40.0% (Pass), >= +10.0% (Partial)")
    print(f"=======================================================")

    report_path = args.output_file if args.output_file else os.path.join(data_dir, "gate2_causal_patching_report.json")
    report_data = {
        "checkpoint": args.checkpoint,
        "valid_trials": valid_count,
        "p_chance": p_chance,
        "p_steered": p_steered,
        "delta_steer": delta_steer,
        "mean_amplification": mean_amp,
        "null_patch_identity_rate": null_rate / 100.0,
        "verdict": "PASS" if delta_steer >= 40.0 else ("PARTIAL" if delta_steer >= 10.0 else "FAIL")
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Report saved to: {report_path}")

    if delta_steer >= 40.0:
        print(f"VERDICT: GATE 2 CERTIFIED [PASS] (Delta Steer {delta_steer:+.2f}% >= +40.0%)")
        print("Strong semantic grounding confirmed. Proceed to 250-suite benchmark.")
    elif delta_steer >= 10.0:
        print(f"VERDICT: GATE 2 PARTIAL GROUNDING ({delta_steer:+.2f}%)")
        print("Run 250-suite benchmark; document mechanism as partially grounded.")
    else:
        print(f"VERDICT: GATE 2 REJECTED [HALT] (Delta Steer {delta_steer:+.2f}% < +10.0%)")
        print("HALT. Latent channel is mechanically bypassed. Abort downstream benchmark.")
        sys.exit(1)

if __name__ == "__main__":
    main()
