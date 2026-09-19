#!/usr/bin/env python3
"""
scripts/98_acid_test_production_model.py
Production Acid Test Suite for the Register-Bundle Ladder Architecture.

Executes four rigorous scientific checks on the trained production model
(checkpoints/lora_arm3_register_ladder_production_qwen3_1.7b/best_checkpoint):
1. Check A: Attention Split Decomposition (Prompt vs. Headers vs. Latents vs. Transition)
   across all 28 layers on 60 MATH L3-5 problems.
   - Decision Rule: Latents >= 10.0% -> PASS
2. Check C: 4-Way Matched Inference Ablation on 60 MATH L3-5 problems:
   - Cond 1: Full Ladder (3 canonical headers + 24 latents)
   - Cond 2: Headers Only (3 canonical headers + 0 latents; tests no compute)
   - Cond 3: Latents Only (generic filler headers + 24 latents; tests no text subgoals)
   - Cond 4: Headers + 24 Pause Tokens (3 canonical headers + 24 pause tokens; near-FLOP matched discrete filler)
   - The Acid Test: Cond 1 vs Cond 4 (Continuous Latents vs Discrete Pause Tokens).
3. Check D: Multi-Bundle Latent Noise Sensitivity:
   - Evaluates output divergence rate under norm-matched Gaussian noise across Bundle 1, Bundle 2, and Bundle 3.
4. Check B: Latent-Only Donor Patching on N=94 valid counterfactual pairs:
   - Control 0 (Null Patch Self-Identity): Mandates >= 95.0% identity gate before reporting DeltaSteer.
   - Control B: Patches donor latent state while keeping recipient headers intact.
   - Calibrated Reference: Discrete CoT steering reference (+0.0% at K=6, +8.7% at Step 1, +21.7% at full chain).
"""

import os
import re
import sys
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
from tqdm import tqdm

CALIBRATED_ALPHA = 0.011440  # Qwen3-1.7B
K_PER_BUNDLE = 8
NUM_RUNGS = 3
TOTAL_LATENTS = NUM_RUNGS * K_PER_BUNDLE  # 24

CANONICAL_HEADERS = [
    "[R1: Identify givens, constraints, and target variable]\n",
    "[R2: Compute intermediate operations and verify relations]\n",
    "[R3: Execute final deduction and verify constraints]\n"
]

GENERIC_FILLER_HEADERS = [
    "[R1: Step 1]\n",
    "[R2: Step 2]\n",
    "[R3: Step 3]\n"
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

def get_headers(mode="full"):
    if mode == "generic_filler":
        return GENERIC_FILLER_HEADERS
    return CANONICAL_HEADERS

def run_single_ladder_inference(
    model, tokenizer, question, headers, device,
    condition="full",           # 'full', 'headers_only', 'latents_only', 'headers_pause'
    noise_bundle=None,          # 1, 2, or 3
    patch_bundle=None,          # bundle index to patch (e.g. 2)
    patch_vector=None,          # replacement latent tensor (1, 1, d_model)
    return_bundle_latent=None,  # bundle index whose mid-latent to return (e.g. 2)
    return_attention=False,
    max_ans_tokens=256
):
    """
    Unified batch-invariant (B=1) ladder inference using eager attention for bitwise determinism.
    """
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    L_prompt = enc_prompt.input_ids.shape[1]

    with torch.no_grad():
        out = model(input_ids=enc_prompt.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        curr_seq_len = L_prompt

        reg_token_indices = []
        latent_indices = []
        pause_indices = []
        saved_latent = None

        for r_idx in range(NUM_RUNGS):
            rung_idx = r_idx + 1  # 1-indexed (1, 2, 3)
            hdr_text = headers[r_idx]
            enc_hdr = tokenizer(hdr_text, return_tensors="pt", add_special_tokens=False).to(device)
            hdr_len = enc_hdr.input_ids.shape[1]
            hdr_pos = torch.arange(curr_seq_len, curr_seq_len + hdr_len, device=device).unsqueeze(0)

            hdr_out = model(input_ids=enc_hdr.input_ids, position_ids=hdr_pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
            past_kv = hdr_out.past_key_values
            curr_latent = hdr_out.hidden_states[-1][:, -1:, :]
            for idx in range(curr_seq_len, curr_seq_len + hdr_len):
                reg_token_indices.append(idx)
            curr_seq_len += hdr_len

            if condition in ["full", "latents_only"]:
                for k in range(K_PER_BUNDLE):
                    step_pos = torch.tensor([[curr_seq_len]], device=device)
                    
                    # Check D: Noise Injection
                    if noise_bundle == rung_idx:
                        sigma = curr_latent.norm(dim=-1, keepdim=True) / np.sqrt(curr_latent.shape[-1])
                        noise = torch.randn_like(curr_latent) * sigma
                        curr_latent = curr_latent + noise
                        
                    # Check B: Latent Patching
                    if patch_bundle == rung_idx and k == (K_PER_BUNDLE // 2) and patch_vector is not None:
                        curr_latent = patch_vector.to(device).to(curr_latent.dtype)
                        
                    if return_bundle_latent == rung_idx and k == (K_PER_BUNDLE // 2):
                        saved_latent = curr_latent.clone()
                        
                    scaled = curr_latent * CALIBRATED_ALPHA
                    step_out = model(inputs_embeds=scaled, position_ids=step_pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
                    past_kv = step_out.past_key_values
                    curr_latent = step_out.hidden_states[-1][:, -1:, :]
                    latent_indices.append(curr_seq_len)
                    curr_seq_len += 1
                    
            elif condition == "headers_pause":
                pause_id = tokenizer.encode(".", add_special_tokens=False)[0]
                pause_ids = torch.full((1, K_PER_BUNDLE), pause_id, dtype=torch.long, device=device)
                pause_pos = torch.arange(curr_seq_len, curr_seq_len + K_PER_BUNDLE, device=device).unsqueeze(0)
                p_out = model(input_ids=pause_ids, position_ids=pause_pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
                past_kv = p_out.past_key_values
                curr_latent = p_out.hidden_states[-1][:, -1:, :]
                for idx in range(curr_seq_len, curr_seq_len + K_PER_BUNDLE):
                    pause_indices.append(idx)
                curr_seq_len += K_PER_BUNDLE
                
            elif condition == "headers_only":
                pass

        # Transition: \n</think>\n\n
        trans_text = "\n</think>\n\n"
        enc_trans = tokenizer(trans_text, return_tensors="pt", add_special_tokens=False).to(device)
        trans_len = enc_trans.input_ids.shape[1]
        trans_pos = torch.arange(curr_seq_len, curr_seq_len + trans_len, device=device).unsqueeze(0)
        trans_out = model(input_ids=enc_trans.input_ids, position_ids=trans_pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
        past_kv = trans_out.past_key_values
        trans_indices = list(range(curr_seq_len, curr_seq_len + trans_len))
        curr_seq_len += trans_len
        
        # Greedy Answer Generation
        # First answer token predicted from trans_out logits (trained target)
        curr_token = torch.argmax(trans_out.logits[:, -1:, :], dim=-1)
        gen_tokens = [curr_token.item()]
        curr_past = past_kv
        curr_pos = curr_seq_len
        
        attn_masses = None
        if return_attention:
            step_eval = model(input_ids=curr_token, position_ids=torch.tensor([[curr_pos]], device=device), past_key_values=curr_past, use_cache=True, output_attentions=True)
            curr_past = step_eval.past_key_values
            curr_pos += 1
            curr_token = torch.argmax(step_eval.logits[:, -1:, :], dim=-1)
            gen_tokens.append(curr_token.item())
            
            num_layers = len(step_eval.attentions)
            prompt_m, hdr_m, latent_m, trans_m = 0.0, 0.0, 0.0, 0.0
            for layer in step_eval.attentions:
                attn = layer[0, :, 0, :].mean(dim=0)  # (total_kv_len,)
                prompt_m += attn[:L_prompt].sum().item()
                if reg_token_indices:
                    hdr_m += attn[reg_token_indices].sum().item()
                if latent_indices:
                    latent_m += attn[latent_indices].sum().item()
                if trans_indices:
                    trans_m += attn[trans_indices].sum().item()
                    
            attn_masses = {
                "prompt_mass": (prompt_m / num_layers) * 100.0,
                "header_mass": (hdr_m / num_layers) * 100.0,
                "latent_mass": (latent_m / num_layers) * 100.0,
                "trans_mass": (trans_m / num_layers) * 100.0
            }

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
            "attn_masses": attn_masses,
            "saved_latent": saved_latent
        }

def main():
    parser = argparse.ArgumentParser(description="Production Acid Test Suite")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lora_arm3_register_ladder_production_qwen3_1.7b/best_checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_file", type=str, default="data/acid_test_production_model_results.json")
    args = parser.parse_args()

    print("=" * 80)
    print("PRODUCTION ACID TEST SUITE: REGISTER-BUNDLE LADDER")
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

    if os.path.exists(args.checkpoint):
        print(f"Loading trained production adapter from {args.checkpoint}...")
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
    else:
        print(f"WARNING: Checkpoint {args.checkpoint} not found. Running base model.")
        model = base_model

    model.eval()

    # Load 250 Suite
    with open("data/benchmark_suite_250.json") as f:
        suite = json.load(f)

    math_l35 = [p for p in suite if p.get("benchmark") == "MATH-500" and p.get("stratum") in ["Level 3", "Level 4", "Level 5"]]
    print(f"Loaded {len(math_l35)} MATH-500 Level 3-5 problems.")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": args.model_id,
        "checkpoint": args.checkpoint
    }

    # =========================================================================
    # CHECK A: Layer-by-Layer Attention Split (Headers vs. Latents)
    # =========================================================================
    print("\n" + "=" * 80)
    print("RUNNING CHECK A: ATTENTION SPLIT DECOMPOSITION (60 MATH L3-5 PROBLEMS)")
    print("Pre-registered Decision Rule: Latents >= 10.0% -> PASS | Latents < 5.0% -> FAIL")
    print("=" * 80)

    prompt_masses, header_masses, latent_masses, trans_masses = [], [], [], []
    hdrs_full = get_headers(mode="full")
    for p in tqdm(math_l35, desc="Check A Attention"):
        res = run_single_ladder_inference(model, tokenizer, p["question"], hdrs_full, args.device, condition="full", return_attention=True, max_ans_tokens=1)
        m = res["attn_masses"]
        prompt_masses.append(m["prompt_mass"])
        header_masses.append(m["header_mass"])
        latent_masses.append(m["latent_mass"])
        trans_masses.append(m["trans_mass"])

    avg_prompt_a = float(np.mean(prompt_masses))
    avg_header_a = float(np.mean(header_masses))
    avg_latent_a = float(np.mean(latent_masses))
    avg_trans_a = float(np.mean(trans_masses))
    total_recurrent_a = avg_header_a + avg_latent_a

    print("\n--- Check A Results Across 60 MATH L3-5 Problems ---")
    print(f"  * Prompt Attention Mass   : {avg_prompt_a:5.2f}%")
    print(f"  * Header Attention Mass   : {avg_header_a:5.2f}%")
    print(f"  * Latent Attention Mass   : {avg_latent_a:5.2f}%")
    print(f"  * Transition Punctuation  : {avg_trans_a:5.2f}%")
    print(f"  -----------------------------------------------")
    print(f"  * Total Recurrent Zone    : {total_recurrent_a:5.2f}%")

    check_a_verdict = "PASS (Latents >= 10%)" if avg_latent_a >= 10.0 else ("PARTIAL (5-10%)" if avg_latent_a >= 5.0 else "FAIL (< 5% - Text Subgoal Only)")
    print(f"Check A Decision Rule Verdict: [{check_a_verdict}]")
    results["check_a"] = {
        "avg_prompt_mass": avg_prompt_a,
        "avg_header_mass": avg_header_a,
        "avg_latent_mass": avg_latent_a,
        "avg_trans_mass": avg_trans_a,
        "total_recurrent_mass": total_recurrent_a,
        "verdict": check_a_verdict
    }

    # =========================================================================
    # CHECK C: 4-Way Matched Inference Ablation on 60 MATH L3-5 Problems
    # =========================================================================
    print("\n" + "=" * 80)
    print("RUNNING CHECK C: 4-WAY MATCHED INFERENCE ABLATION (60 MATH L3-5 PROBLEMS)")
    print("Conditions: (1) Full Ladder | (2) Headers Only | (3) Latents Only | (4) Headers + 24 Pause Tokens")
    print("=" * 80)

    accs = {"full": 0, "headers_only": 0, "latents_only": 0, "headers_pause": 0}
    clean_generations = {"full": [], "headers_only": [], "latents_only": [], "headers_pause": []}
    hdrs_filler = get_headers(mode="generic_filler")

    for p in tqdm(math_l35, desc="Check C 4-Way Ablation"):
        gold = p.get("solution") or p.get("answer", "")
        q = p["question"]

        # 1. Full Ladder
        res1 = run_single_ladder_inference(model, tokenizer, q, hdrs_full, args.device, condition="full", max_ans_tokens=512)
        c1 = check_correctness(res1["text"], gold)
        accs["full"] += c1
        clean_generations["full"].append(res1["text"])

        # 2. Headers Only (0 latents)
        res2 = run_single_ladder_inference(model, tokenizer, q, hdrs_full, args.device, condition="headers_only", max_ans_tokens=512)
        c2 = check_correctness(res2["text"], gold)
        accs["headers_only"] += c2
        clean_generations["headers_only"].append(res2["text"])

        # 3. Latents Only (generic filler headers + 24 latents)
        res3 = run_single_ladder_inference(model, tokenizer, q, hdrs_filler, args.device, condition="latents_only", max_ans_tokens=512)
        c3 = check_correctness(res3["text"], gold)
        accs["latents_only"] += c3
        clean_generations["latents_only"].append(res3["text"])

        # 4. Headers + 24 Pause Tokens
        res4 = run_single_ladder_inference(model, tokenizer, q, hdrs_full, args.device, condition="headers_pause", max_ans_tokens=512)
        c4 = check_correctness(res4["text"], gold)
        accs["headers_pause"] += c4
        clean_generations["headers_pause"].append(res4["text"])

    n_math = len(math_l35)
    pct_full = (accs["full"] / n_math) * 100.0
    pct_headers = (accs["headers_only"] / n_math) * 100.0
    pct_latents = (accs["latents_only"] / n_math) * 100.0
    pct_pause = (accs["headers_pause"] / n_math) * 100.0
    delta_latent_vs_pause = pct_full - pct_pause

    print("\n--- Check C Accuracy Across 60 MATH L3-5 Problems ---")
    print(f"  1. Full Ladder (Headers + 24 Latents)        : {pct_full:5.2f}% ({int(accs['full'])}/{n_math})")
    print(f"  2. Headers Only (0 Latents)                 : {pct_headers:5.2f}% ({int(accs['headers_only'])}/{n_math})")
    print(f"  3. Latents Only (Generic Headers + 24 Lat.)  : {pct_latents:5.2f}% ({int(accs['latents_only'])}/{n_math})")
    print(f"  4. Headers + 24 Pause Tokens                 : {pct_pause:5.2f}% ({int(accs['headers_pause'])}/{n_math})")
    print(f"  -------------------------------------------------------------")
    print(f"  * Acid Test (Cond 1 - Cond 4)                : {delta_latent_vs_pause:+5.2f}%")

    if delta_latent_vs_pause >= 3.0:
        c_verdict = "PASS (Latents beat Pause Tokens by >= +3.0%)"
    elif delta_latent_vs_pause >= 0.0:
        c_verdict = "NEUTRAL (Latents tie Pause Tokens)"
    else:
        c_verdict = "FAIL (Latents underperform Pause Tokens)"
    print(f"Check C Acid Test Verdict: [{c_verdict}]")

    results["check_c"] = {
        "full_ladder_acc": pct_full,
        "headers_only_acc": pct_headers,
        "latents_only_acc": pct_latents,
        "headers_pause_acc": pct_pause,
        "delta_latent_vs_pause": delta_latent_vs_pause,
        "verdict": c_verdict
    }

    # =========================================================================
    # CHECK D: Multi-Bundle Latent Noise Sensitivity (Bundles 1, 2, 3)
    # =========================================================================
    print("\n" + "=" * 80)
    print("RUNNING CHECK D: MULTI-BUNDLE NOISE SENSITIVITY")
    print("Testing divergence rate under norm-matched Gaussian noise across Bundle 1, 2, 3")
    print("=" * 80)

    divergence_counts = {1: 0, 2: 0, 3: 0}
    for i, p in enumerate(tqdm(math_l35, desc="Check D Noise")):
        baseline_text = clean_generations["full"][i]

        for b in [1, 2, 3]:
            res_noise = run_single_ladder_inference(model, tokenizer, p["question"], hdrs_full, args.device, condition="full", noise_bundle=b, max_ans_tokens=256)
            if res_noise["text"] != baseline_text:
                divergence_counts[b] += 1

    div_rate_1 = (divergence_counts[1] / n_math) * 100.0
    div_rate_2 = (divergence_counts[2] / n_math) * 100.0
    div_rate_3 = (divergence_counts[3] / n_math) * 100.0

    print("\n--- Check D Divergence Rates (Fraction of Generations Changed) ---")
    print(f"  * Bundle 1 (k=1..8)   : {div_rate_1:5.2f}% ({divergence_counts[1]}/{n_math})")
    print(f"  * Bundle 2 (k=9..16)  : {div_rate_2:5.2f}% ({divergence_counts[2]}/{n_math})")
    print(f"  * Bundle 3 (k=17..24) : {div_rate_3:5.2f}% ({divergence_counts[3]}/{n_math})")

    results["check_d"] = {
        "bundle1_divergence_pct": div_rate_1,
        "bundle2_divergence_pct": div_rate_2,
        "bundle3_divergence_pct": div_rate_3
    }

    # =========================================================================
    # CHECK B: Latent-Only Donor Patching (N=94 Pairs, with Control 0 Gate)
    # =========================================================================
    print("\n" + "=" * 80)
    print("RUNNING CHECK B: LATENT-ONLY DONOR PATCHING WITH CONTROL 0 GATE")
    print("Target: N=94 valid counterfactual pairs | Control 0 requirement: >= 95.0% Identity")
    print("=" * 80)

    math_100 = [p for p in suite if p.get("benchmark") == "MATH-500"]
    print(f"Extracted {len(math_100)} MATH-500 problems for donor pairing.")

    print("\nExtracting clean recipient answers and Bundle 2 latent states...")
    clean_texts_100 = []
    clean_latents_b2 = []
    clean_nums_100 = []

    for p in tqdm(math_100, desc="Clean Extraction"):
        res = run_single_ladder_inference(model, tokenizer, p["question"], hdrs_full, args.device, condition="full", return_bundle_latent=2, max_ans_tokens=256)
        clean_texts_100.append(res["text"])
        clean_latents_b2.append(res["saved_latent"])
        sol = p.get("solution") or p.get("answer", "")
        nums = extract_all_numbers(res["text"]) | extract_all_numbers(sol)
        clean_nums_100.append(nums)

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
        nums_R_prompt = extract_all_numbers(prob_R["question"])
        nums_D_filtered = {n for n in nums_D if n > 20 and n not in nums_R_prompt}
        if not nums_D_filtered:
            continue
        if nums_D_filtered.intersection(extract_all_numbers(clean_texts_100[r_idx])):
            chance_hits += 1
        valid_trials.append({
            "r_idx": r_idx,
            "d_idx": d_idx,
            "nums_D_filtered": nums_D_filtered
        })

    n_valid = len(valid_trials)
    print(f"Identified {n_valid} valid counterfactual trials out of {len(donor_pairs)} pairs.")

    print("\n[MANDATORY GATE] Running Control 0 (Null Patch Self-Identity)...")
    null_identical = 0
    for trial in tqdm(valid_trials, desc="Control 0"):
        r_idx = trial["r_idx"]
        prob_R = math_100[r_idx]
        self_vec = clean_latents_b2[r_idx]

        res_null = run_single_ladder_inference(
            model, tokenizer, prob_R["question"], hdrs_full, args.device,
            condition="full", patch_bundle=2, patch_vector=self_vec, max_ans_tokens=256
        )
        if res_null["text"] == clean_texts_100[r_idx]:
            null_identical += 1

    null_identity_pct = (null_identical / n_valid) * 100.0
    print(f"\nControl 0 Result: {null_identical}/{n_valid} ({null_identity_pct:.2f}% Identity)")

    if null_identity_pct < 95.0:
        print(f"\n[FATAL ERROR] Control 0 failed ({null_identity_pct:.2f}% < 95.0%).")
        print("Stopping Check B evaluation due to harness variance.")
        results["check_b"] = {
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
            donor_vec = clean_latents_b2[d_idx]

            res_patched = run_single_ladder_inference(
                model, tokenizer, prob_R["question"], hdrs_full, args.device,
                condition="full", patch_bundle=2, patch_vector=donor_vec, max_ans_tokens=256
            )
            patched_nums = extract_all_numbers(res_patched["text"])
            if trial["nums_D_filtered"].intersection(patched_nums):
                steered_hits += 1

        p_chance = (chance_hits / n_valid) * 100.0
        p_steered = (steered_hits / n_valid) * 100.0
        delta_steer = p_steered - p_chance

        print(f"\n--- Check B Results ({n_valid} Valid Trials) ---")
        print(f"  * P(chance)           : {p_chance:5.2f}% ({chance_hits}/{n_valid})")
        print(f"  * P(steered)          : {p_steered:5.2f}% ({steered_hits}/{n_valid})")
        print(f"  * Net Delta Steer     : {delta_steer:+5.2f}%")
        print(f"  * Calibrated Reference: Discrete CoT steering reference is +0.00% (K=6), +8.70% (Step 1), +21.74% (Full Chain)")

        if delta_steer >= 21.74:
            b_verdict = "PASS (>= Discrete Full Chain Reference +21.74%)"
        elif delta_steer >= 8.70:
            b_verdict = "PARTIAL (Matches Discrete Step 1 Reference +8.70%)"
        elif delta_steer > 0.0:
            b_verdict = "WEAK POSITIVE (> 0%)"
        else:
            b_verdict = "INERT (<= 0% - Latents do not steer)"
        print(f"Check B Verdict: [{b_verdict}]")

        results["check_b"] = {
            "control_0_identity_pct": null_identity_pct,
            "n_valid_trials": n_valid,
            "p_chance": p_chance,
            "p_steered": p_steered,
            "delta_steer": delta_steer,
            "verdict": b_verdict
        }

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAcid Test Suite Complete! Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
