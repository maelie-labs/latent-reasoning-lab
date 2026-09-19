#!/usr/bin/env python3
"""
scripts/91_diagnose_causal_patching.py
Diagnostic for Gate 2 Causal Patching:
1. Inspects the 13 steered hits: are they identical to clean_text (chance hits)?
2. Measures Control A (norm-matched Gaussian noise at step 3 and step 6): does noise break generation?
3. Evaluates text divergence: how much does donor patching alter generated tokens?
4. Compares step 3 patch vs step 6 patch.
"""

import os
import re
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

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

def run_batched_unroll_with_noise_and_patch(
    model, tokenizer, prompts, scale_factor, k_steps,
    patch_step=None, patch_vectors=None, device="cuda:1",
    max_ans_tokens=512, return_intermediate_step=None
):
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
            return {"texts": [""] * B, "intermediate_state": saved_intermediate}

        trans_tokens = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
        trans_len = len(trans_tokens)
        trans_ids = torch.tensor([trans_tokens] * B, dtype=torch.long, device=device)
        trans_mask = torch.cat([curr_mask, torch.ones((B, trans_len), dtype=torch.long, device=device)], dim=1)
        pos = torch.stack([torch.arange(trans_len, device=device) + (seq_lens[b] + k_steps) for b in range(B)], dim=0)

        gen_out = model.generate(
            input_ids=trans_ids,
            attention_mask=trans_mask,
            position_ids=pos,
            past_key_values=past_kv,
            max_new_tokens=max_ans_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

        texts = []
        for b in range(B):
            gen_toks = gen_out[b][trans_len:].tolist()
            text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
            texts.append(text)

        return {"texts": texts, "intermediate_state": saved_intermediate}

def main():
    model_id = "Qwen/Qwen3-1.7B"
    checkpoint = "checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_lambda1.0"
    device = "cuda:1"
    batch_size = 16
    k_steps = 6
    scale_factor = 0.011440

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    suite_path = os.path.join(data_dir, "benchmark_suite_250.json")

    print(f"=== GATE 2 DEEP DIAGNOSTIC ===")
    print(f"Model: {model_id} | Checkpoint: {checkpoint} on {device}")

    with open(suite_path) as f:
        problems = json.load(f)[:100]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model.eval()

    clean_prompts = [
        f"<|im_start|>user\n{p['question']}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        for p in problems
    ]

    # Pre-unroll clean and get h_3 and h_6
    print("\n[Phase 1] Pre-unrolling clean problems (capturing h_3 and h_6)...")
    clean_texts = []
    clean_h3_list = []
    clean_h6_list = []

    for b in tqdm(range(0, len(clean_prompts), batch_size), desc="Clean Unroll h_3"):
        res = run_batched_unroll_with_noise_and_patch(
            model, tokenizer, clean_prompts[b : b + batch_size], scale_factor, k_steps,
            return_intermediate_step=3, device=device, max_ans_tokens=512
        )
        clean_texts.extend(res["texts"])
        clean_h3_list.append(res["intermediate_state"])

    for b in tqdm(range(0, len(clean_prompts), batch_size), desc="Clean Unroll h_6"):
        res = run_batched_unroll_with_noise_and_patch(
            model, tokenizer, clean_prompts[b : b + batch_size], scale_factor, k_steps,
            return_intermediate_step=6, device=device, max_ans_tokens=0
        )
        clean_h6_list.append(res["intermediate_state"])

    clean_h3 = torch.cat(clean_h3_list, dim=0)  # (100, 1, d_model)
    clean_h6 = torch.cat(clean_h6_list, dim=0)  # (100, 1, d_model)
    clean_nums = [extract_all_numbers(t) for t in clean_texts]

    # Filter donor pairs (same seed 42)
    random.seed(42)
    indices = list(range(len(problems)))
    shuffled = list(indices)
    while any(i == j for i, j in zip(indices, shuffled)):
        random.shuffle(shuffled)
    donor_pairs = [(indices[idx], shuffled[idx]) for idx in range(len(indices))]

    valid_trials = []
    for r_idx, d_idx in donor_pairs:
        sol_D = problems[d_idx].get("solution") or problems[d_idx].get("reference_answer", "")
        nums_D = clean_nums[d_idx] | extract_all_numbers(sol_D)
        nums_R_prompt = extract_all_numbers(problems[r_idx]["question"])
        nums_D_filtered = {n for n in nums_D if n > 20 and n not in nums_R_prompt}
        if nums_D_filtered:
            valid_trials.append({
                "r_idx": r_idx,
                "d_idx": d_idx,
                "nums_D_filtered": nums_D_filtered,
                "chance_hit": bool(nums_D_filtered.intersection(clean_nums[r_idx]))
            })

    print(f"Total valid trials: {len(valid_trials)}")
    chance_count = sum(1 for v in valid_trials if v["chance_hit"])
    print(f"Baseline Chance Hits: {chance_count}/{len(valid_trials)} ({chance_count/len(valid_trials)*100:.2f}%)")

    # 1. Steered Unrolls at Step 3
    print("\n[Phase 2] Evaluating Steered Unrolls (Patch at Step 3)...")
    steer_prompts = [clean_prompts[v["r_idx"]] for v in valid_trials]
    steer_h3 = clean_h3[[v["d_idx"] for v in valid_trials]]

    steered_texts_t3 = []
    for b in tqdm(range(0, len(valid_trials), batch_size), desc="Steer Step 3"):
        res = run_batched_unroll_with_noise_and_patch(
            model, tokenizer, steer_prompts[b : b + batch_size], scale_factor, k_steps,
            patch_step=3, patch_vectors=steer_h3[b : b + batch_size], device=device, max_ans_tokens=512
        )
        steered_texts_t3.extend(res["texts"])

    # 2. Control A: Norm-Matched Gaussian Noise at Step 3
    print("\n[Phase 3] Evaluating Control A (Norm-Matched Gaussian Noise at Step 3)...")
    torch.manual_seed(42)
    noise_h3 = []
    for i in range(len(valid_trials)):
        orig_v = clean_h3[valid_trials[i]["r_idx"]]
        norm = torch.norm(orig_v, dim=-1, keepdim=True)
        g_noise = torch.randn_like(orig_v)
        g_noise = norm * (g_noise / torch.norm(g_noise, dim=-1, keepdim=True))
        noise_h3.append(g_noise)
    noise_h3_tensor = torch.cat(noise_h3, dim=0).unsqueeze(1) if len(noise_h3[0].shape) == 2 else torch.stack(noise_h3, dim=0)

    noise_texts_t3 = []
    for b in tqdm(range(0, len(valid_trials), batch_size), desc="Control A Noise Step 3"):
        res = run_batched_unroll_with_noise_and_patch(
            model, tokenizer, steer_prompts[b : b + batch_size], scale_factor, k_steps,
            patch_step=3, patch_vectors=noise_h3_tensor[b : b + batch_size], device=device, max_ans_tokens=512
        )
        noise_texts_t3.extend(res["texts"])

    # 3. Steered Unrolls at Step 6 (Final Latent Hand-Off Position)
    print("\n[Phase 4] Evaluating Steered Unrolls (Patch at Step 6 - Final Latent)...")
    steer_h6 = clean_h6[[v["d_idx"] for v in valid_trials]]
    steered_texts_t6 = []
    for b in tqdm(range(0, len(valid_trials), batch_size), desc="Steer Step 6"):
        res = run_batched_unroll_with_noise_and_patch(
            model, tokenizer, steer_prompts[b : b + batch_size], scale_factor, k_steps,
            patch_step=6, patch_vectors=steer_h6[b : b + batch_size], device=device, max_ans_tokens=512
        )
        steered_texts_t6.extend(res["texts"])

    # 4. Analyze Results
    print("\n" + "=" * 70)
    print("GATE 2 DIAGNOSTIC AUDIT RESULTS")
    print("=" * 70)

    identical_to_clean_t3 = 0
    steered_hits_t3 = 0
    noise_divergence_count_t3 = 0
    steered_hits_t6 = 0

    hit_details_t3 = []

    for i, v in enumerate(valid_trials):
        r_idx = v["r_idx"]
        d_idx = v["d_idx"]
        clean_t = clean_texts[r_idx]
        steer_t3 = steered_texts_t3[i]
        noise_t3 = noise_texts_t3[i]
        steer_t6 = steered_texts_t6[i]

        nums_steer_t3 = extract_all_numbers(steer_t3)
        nums_steer_t6 = extract_all_numbers(steer_t6)

        overlap_t3 = v["nums_D_filtered"].intersection(nums_steer_t3)
        overlap_t6 = v["nums_D_filtered"].intersection(nums_steer_t6)

        if overlap_t3:
            steered_hits_t3 += 1
            hit_details_t3.append({
                "trial_idx": i,
                "r_idx": r_idx,
                "d_idx": d_idx,
                "chance_hit": v["chance_hit"],
                "identical_to_clean": (clean_t == steer_t3),
                "overlapping_numbers": list(overlap_t3),
                "clean_text_snippet": clean_t[:100],
                "steer_text_snippet": steer_t3[:100]
            })

        if overlap_t6:
            steered_hits_t6 += 1

        if steer_t3 == clean_t:
            identical_to_clean_t3 += 1

        if noise_t3 != clean_t:
            noise_divergence_count_t3 += 1

    print(f"\n1. IDENTICAL-13 ANOMALY ANALYSIS:")
    print(f"Total Steered Hits at Step 3: {steered_hits_t3} / {len(valid_trials)} ({steered_hits_t3/len(valid_trials)*100:.2f}%)")
    print(f"Of the {steered_hits_t3} steered hits:")
    chance_in_steered = sum(1 for h in hit_details_t3 if h["chance_hit"])
    identical_in_steered = sum(1 for h in hit_details_t3 if h["identical_to_clean"])
    print(f"  - Were also Chance Hits in unpatched generation: {chance_in_steered} / {steered_hits_t3} ({chance_in_steered/steered_hits_t3*100:.1f}%)")
    print(f"  - Produced 100% BIT-IDENTICAL text to clean generation: {identical_in_steered} / {steered_hits_t3} ({identical_in_steered/steered_hits_t3*100:.1f}%)")

    print(f"\nAcross all {len(valid_trials)} trials:")
    print(f"  - Steered text == Clean text (Unperturbed): {identical_to_clean_t3} / {len(valid_trials)} ({identical_to_clean_t3/len(valid_trials)*100:.1f}%)")

    print(f"\n2. CONTROL A (NORM-MATCHED GAUSSIAN NOISE AT STEP 3):")
    print(f"  - Divergence Rate (noise_text != clean_text): {noise_divergence_count_t3} / {len(valid_trials)} ({noise_divergence_count_t3/len(valid_trials)*100:.2f}%)")
    if noise_divergence_count_t3 == 0:
        print("  -> CRITICAL: Decoder completely ignores latent positions! 0% divergence under Gaussian noise.")
    elif noise_divergence_count_t3 == len(valid_trials):
        print("  -> Decoder is physically coupled to latent positions (100% divergence).")
    else:
        print(f"  -> Partial physical coupling ({noise_divergence_count_t3/len(valid_trials)*100:.1f}%).")

    print(f"\n3. STEP 6 PATCHING (HAND-OFF POSITION):")
    print(f"  - Steered Hits at Step 6: {steered_hits_t6} / {len(valid_trials)} ({steered_hits_t6/len(valid_trials)*100:.2f}%)")
    delta_step6 = (steered_hits_t6 - chance_count) / len(valid_trials) * 100.0
    print(f"  - Delta Steer at Step 6: {delta_step6:+.2f}%")

    diag_path = os.path.join(data_dir, "gate2_deep_diagnostic_lambda1.0.json")
    with open(diag_path, "w") as f:
        json.dump({
            "checkpoint": checkpoint,
            "valid_trials": len(valid_trials),
            "chance_hits": chance_count,
            "steered_hits_t3": steered_hits_t3,
            "steered_hits_t6": steered_hits_t6,
            "identical_to_clean_t3": identical_to_clean_t3,
            "control_a_noise_divergence_t3": noise_divergence_count_t3,
            "control_a_divergence_rate_pct": noise_divergence_count_t3 / len(valid_trials) * 100.0,
            "hit_details_t3": hit_details_t3
        }, f, indent=2)
    print(f"\nSaved diagnostic report to: {diag_path}")

if __name__ == "__main__":
    main()
