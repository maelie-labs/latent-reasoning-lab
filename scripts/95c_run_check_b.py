#!/usr/bin/env python3
"""
scripts/95c_run_check_b.py
Dedicated runner for Check B (Control 0 + Donor Latent Patching) on the Register-Ladder prototype.
Imports core inference and helper routines from scripts/95_phase0_prototype_validation.py.
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import importlib.util
spec = importlib.util.spec_from_file_location("val", "scripts/95_phase0_prototype_validation.py")
val = importlib.util.module_from_spec(spec)
spec.loader.exec_module(val)

def main():
    parser = argparse.ArgumentParser(description="Check B Runner: Latent Patching & Control 0 Gate")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/test_register_ladder_quick")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--output_file", type=str, default="data/phase0_prototype_validation_results.json")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 0: CHECK B (CONTROL 0 IDENTITY & LATENT DONOR PATCHING)")
    print(f"Model: {args.model_id} | Checkpoint: {args.checkpoint} | Device: {args.device}")
    print("=" * 80)

    # 1. Load Model & Tokenizer with Eager Attention for Bitwise Determinism
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

    # Load 250 Suite
    with open("data/benchmark_suite_250.json") as f:
        suite = json.load(f)

    math_100 = [p for p in suite if p.get("benchmark") == "MATH-500"]
    print(f"Extracted {len(math_100)} MATH-500 problems for donor pairing.")

    # Phase B.1: Generate clean baselines and extract Bundle 2 mid-latents
    print("\nExtracting clean recipient answers and Bundle 2 latent states...")
    clean_texts_100 = []
    clean_latents_b2 = []
    clean_nums_100 = []

    for p in tqdm(math_100, desc="Clean Extraction"):
        hdrs = val.get_headers(p, mode="full")
        res = val.run_single_ladder_inference(
            model, tokenizer, p["question"], hdrs, args.device,
            condition="full", return_bundle_latent=2, max_ans_tokens=256
        )
        clean_texts_100.append(res["text"])
        clean_latents_b2.append(res["saved_latent"])
        sol = p.get("solution") or p.get("answer", "")
        nums = val.extract_all_numbers(res["text"]) | val.extract_all_numbers(sol)
        clean_nums_100.append(nums)

    # Phase B.2: Deterministic counterfactual pairs (seed 42)
    random.seed(42)
    indices = list(range(len(math_100)))
    shuffled = list(indices)
    while any(i == j for i, j in zip(indices, shuffled)):
        random.shuffle(shuffled)
    donor_pairs = [(indices[idx], shuffled[idx]) for idx in range(len(indices))]

    valid_trials = []
    chance_hits = 0
    for r_idx, d_idx in donor_pairs:
        prob_R = math_100[r_idx]
        nums_D = clean_nums_100[d_idx]
        nums_R_prompt = val.extract_all_numbers(prob_R["question"])
        nums_D_filtered = {n for n in nums_D if n > 20 and n not in nums_R_prompt}
        if not nums_D_filtered:
            continue
        if nums_D_filtered.intersection(val.extract_all_numbers(clean_texts_100[r_idx])):
            chance_hits += 1
        valid_trials.append({
            "r_idx": r_idx,
            "d_idx": d_idx,
            "nums_D_filtered": nums_D_filtered
        })

    n_valid = len(valid_trials)
    print(f"\nIdentified {n_valid} valid counterfactual trials out of {len(donor_pairs)} pairs.")

    # Phase B.3: Control 0 (Null Patch Self-Identity)
    print("\n[MANDATORY GATE] Running Control 0 (Null Patch Self-Identity)...")
    null_identical = 0
    for trial in tqdm(valid_trials, desc="Control 0"):
        r_idx = trial["r_idx"]
        prob_R = math_100[r_idx]
        hdrs_R = val.get_headers(prob_R, mode="full")
        self_vec = clean_latents_b2[r_idx]

        res_null = val.run_single_ladder_inference(
            model, tokenizer, prob_R["question"], hdrs_R, args.device,
            condition="full", patch_bundle=2, patch_vector=self_vec, max_ans_tokens=256
        )
        if res_null["text"] == clean_texts_100[r_idx]:
            null_identical += 1

    null_identity_pct = (null_identical / n_valid) * 100.0
    print(f"\nControl 0 Result: {null_identical}/{n_valid} ({null_identity_pct:.2f}% Identity)")

    check_b_data = {}
    if null_identity_pct < 95.0:
        print(f"\n[FATAL ERROR] Control 0 failed ({null_identity_pct:.2f}% < 95.0%).")
        print("Stopping Check B evaluation due to harness variance.")
        check_b_data = {
            "control_0_identity_pct": null_identity_pct,
            "verdict": "FAIL (Control 0 < 95%)"
        }
    else:
        print("--> Control 0 PASSED (>= 95.0%). Proceeding to Control B donor patching...")
        steered_hits = 0
        for trial in tqdm(valid_trials, desc="Control B Patching"):
            r_idx = trial["r_idx"]
            d_idx = trial["d_idx"]
            prob_R = math_100[r_idx]
            hdrs_R = val.get_headers(prob_R, mode="full")
            donor_vec = clean_latents_b2[d_idx]

            res_patched = val.run_single_ladder_inference(
                model, tokenizer, prob_R["question"], hdrs_R, args.device,
                condition="full", patch_bundle=2, patch_vector=donor_vec, max_ans_tokens=256
            )
            patched_nums = val.extract_all_numbers(res_patched["text"])
            if trial["nums_D_filtered"].intersection(patched_nums):
                steered_hits += 1

        p_chance = (chance_hits / n_valid) * 100.0
        p_steered = (steered_hits / n_valid) * 100.0
        delta_steer = p_steered - p_chance

        print(f"\n--- Check B Results ({n_valid} Valid Trials) ---")
        print(f"  * P(chance)           : {p_chance:5.2f}% ({chance_hits}/{n_valid})")
        print(f"  * P(steered)          : {p_steered:5.2f}% ({steered_hits}/{n_valid})")
        print(f"  * Net Delta Steer     : {delta_steer:+5.2f}%")

        if delta_steer >= 40.0:
            b_verdict = "PASS (>= +40%)"
        elif delta_steer >= 10.0:
            b_verdict = "PARTIAL (+10% to +40%)"
        else:
            b_verdict = "FAIL (< +10% - Latents Inert)"
        print(f"Check B Verdict: [{b_verdict}]")

        check_b_data = {
            "control_0_identity_pct": null_identity_pct,
            "n_valid_trials": n_valid,
            "p_chance": p_chance,
            "p_steered": p_steered,
            "delta_steer": delta_steer,
            "verdict": b_verdict
        }

    # Load existing results and update
    if os.path.exists(args.output_file):
        with open(args.output_file) as f:
            full_results = json.load(f)
    else:
        full_results = {}

    full_results["check_b"] = check_b_data
    with open(args.output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nCheck B complete! Results updated in {args.output_file}")

if __name__ == "__main__":
    main()
