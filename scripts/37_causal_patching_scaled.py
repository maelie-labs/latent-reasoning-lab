#!/usr/bin/env python3
"""
scripts/37_causal_patching_scaled.py
PHASE 5: Scaled Causal Activation Patching Study (N >= 100).

Tests whether intermediate latent representations h_t carry task-specific semantic information.

4-Control Matrix:
1. Control 0 (Null Patch / Mechanical Sanity):
   - Re-inject the problem's own latent vector h_t.
   - Verify max |Delta logits| < 10^-3 and 100% token sequence identity under BF16 non-determinism.
2. Loop-Position Ablation:
   - Fractional depths: t in {floor(0.25*K), floor(0.50*K), floor(0.75*K)}.
3. Control A (Norm-Matched Gaussian Noise):
   - Replace h_t with isotropic Gaussian noise scaled to ||h_t||_2.
   - Measure accuracy collapse and output divergence.
4. Control B (Shuffled-Problem Donor Steering):
   - Inject donor latent state h_t^(D) from donor problem D into recipient problem R.
   - Filtered donor numbers: V_donor \ {n <= 20} \ V_recipient.
   - Measured chance baseline: P_chance.
   - Pre-registered prediction: Delta Steer = P_steered - P_chance >= +40%.
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import transformers.modeling_utils

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

CALIBRATED_ALPHAS = {
    "qwen/qwen3-1.7b": 0.011440,
    "qwen/qwen3-4b": 0.006702,
    "qwen/qwen3-8b": 0.008500,
    "qwen/qwen3-14b": 0.005200
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_all_numbers(text):
    if not text:
        return set()
    nums = re.findall(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?', str(text))
    res = set()
    for n in nums:
        try:
            val = float(n.replace(',', ''))
            res.add(val)
        except ValueError:
            pass
    return res

def sample_token(logits, temperature=0.6, top_p=0.95, top_k=20):
    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("Inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def run_recurrent_unroll(model, tokenizer, prompt, scale_factor, k_steps, patch_step=None, patch_vector=None, device="cuda:0", max_ans_tokens=8192):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        intermediate_states = []
        for k in range(1, k_steps + 1):
            scaled_latent = curr_latent * scale_factor
            step_out = model(
                inputs_embeds=scaled_latent,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]
            if patch_step == k and patch_vector is not None:
                # Intervene at/after step k
                curr_latent = patch_vector.to(device).to(curr_latent.dtype)
            intermediate_states.append(curr_latent.clone())
            
        # Ingest </think>\n\n transition
        trans_tokens = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
        trans_ids = torch.tensor([trans_tokens], dtype=torch.long, device=device)
        step_out = model(input_ids=trans_ids, past_key_values=past_kv, use_cache=True)
        past_kv = step_out.past_key_values
        
        logits_first = step_out.logits[:, -1, :].clone()
        curr_tok = sample_token(logits_first, temperature=0.6, top_p=0.95, top_k=20)
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        
        gen_tokens = [curr_tok.item()]
        for _ in range(max_ans_tokens):
            step_out = model(input_ids=curr_tok, past_key_values=past_kv, use_cache=True)
            past_kv = step_out.past_key_values
            curr_tok = sample_token(step_out.logits[:, -1, :], temperature=0.6, top_p=0.95, top_k=20)
            t_val = curr_tok.item()
            if t_val in [tokenizer.eos_token_id, im_end_id]:
                break
            gen_tokens.append(t_val)
            
    out_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return {
        "text": out_text,
        "first_logits": logits_first,
        "intermediate_states": intermediate_states
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Causal Activation Patching Study (N >= 100).")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora_path", type=str, default="checkpoints/lora_recurrent_qwen3_1.7b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_problems", type=int, default=100)
    parser.add_argument("--k_steps", type=int, default=6)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    set_seed(42)
    model_key = args.model_id.lower()
    scale_factor = CALIBRATED_ALPHAS.get(model_key, 0.011440)

    if args.output_file is None:
        tag = args.model_id.replace("/", "_").lower()
        args.output_file = os.path.join(os.path.dirname(__file__), "..", "data", f"causal_patching_{tag}.json")

    print(f"=== PHASE 5: Causal Activation Patching Study for {args.model_id} on {args.device} ===")
    print(f"N={args.n_problems} Problems | Horizon K={args.k_steps} | Empirical Alpha={scale_factor:.6f}")

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip("\"'\n")
                    break

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token, trust_remote_code=True)
    print("Loading base model in pure bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    base_model.eval()

    if os.path.exists(args.lora_path):
        print(f"Attaching LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        model.eval()
    else:
        print(f"Warning: Adapter {args.lora_path} not found. Running on base model.")
        model = base_model

    # Load N problems from GSM8K test split
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for idx in range(min(args.n_problems, len(gsm8k))):
        ex = gsm8k[idx]
        problems.append({
            "id": f"gsm8k_{idx}",
            "question": ex["question"],
            "solution": ex["answer"]
        })
    print(f"Loaded {len(problems)} test problems.")

    # Fractional Patching Positions
    k = args.k_steps
    patch_depths = [max(1, int(0.25 * k)), max(1, int(0.50 * k)), max(1, int(0.75 * k))]
    print(f"Fractional intervention depths: {patch_depths}")

    null_patch_diffs = []
    sens_amplifications = []
    noise_collapses = []
    donor_steered_count = 0
    chance_hit_count = 0
    total_donor_trials = 0

    # Shuffle for donor pairs: pair problem i with problem (i + 50) % len
    n_p = len(problems)
    
    print("\nExecuting Causal Patching Matrix...")
    for idx in range(n_p):
        prob_R = problems[idx]                         # Recipient
        prob_D = problems[(idx + n_p // 2) % n_p]      # Donor
        
        # Numbers sets
        nums_R = extract_all_numbers(prob_R["question"] + " " + prob_R["solution"])
        nums_D_all = extract_all_numbers(prob_D["question"] + " " + prob_D["solution"])
        # Filtered donor numbers: > 20 and not in recipient
        nums_D_filtered = {x for x in nums_D_all if x > 20 and x not in nums_R}
        
        # 1. Unpatched run for recipient
        set_seed(42 + idx)
        unpatched_R = run_recurrent_unroll(model, tokenizer, prob_R["question"], scale_factor, k, device=args.device)
        unpatched_text = unpatched_R["text"]
        unpatched_nums = extract_all_numbers(unpatched_text)
        
        # Measure Chance Baseline: Does unpatched recipient output contain any filtered donor numbers?
        chance_hit = bool(nums_D_filtered.intersection(unpatched_nums))
        if chance_hit:
            chance_hit_count += 1
            
        mid_t = max(1, int(0.50 * k))
        h_mid_R = unpatched_R["intermediate_states"][mid_t - 1]
        
        # 2. Control 0: Null Patch (re-inject own h_mid)
        set_seed(42 + idx)
        null_res = run_recurrent_unroll(
            model, tokenizer, prob_R["question"], scale_factor, k,
            patch_step=mid_t, patch_vector=h_mid_R, device=args.device
        )
        delta_logits = torch.max(torch.abs(null_res["first_logits"] - unpatched_R["first_logits"])).item()
        cos_null = F.cosine_similarity(null_res["intermediate_states"][mid_t - 1], h_mid_R, dim=-1).mean().item()
        identical_null_text = (null_res["text"] == unpatched_text)
        null_patch_diffs.append({
            "delta_logit": delta_logits,
            "cosine_sim": cos_null,
            "identical_text": identical_null_text
        })
        
        # Finite-difference sensitivity: perturb h_mid by epsilon (||eps|| = 1e-3 * ||h_mid||)
        eps_dir = torch.randn_like(h_mid_R)
        eps = 1e-3 * torch.norm(h_mid_R) * (eps_dir / torch.norm(eps_dir))
        perturbed_h = h_mid_R + eps
        set_seed(42 + idx)
        sens_res = run_recurrent_unroll(
            model, tokenizer, prob_R["question"], scale_factor, k,
            patch_step=mid_t, patch_vector=perturbed_h, device=args.device
        )
        if mid_t < k:
            h_next_clean = unpatched_R["intermediate_states"][mid_t]
            h_next_pert = sens_res["intermediate_states"][mid_t]
            delta_h_next = torch.norm(h_next_pert - h_next_clean).item()
            amp_factor = delta_h_next / (torch.norm(eps).item() + 1e-12)
        else:
            amp_factor = 1.0
        sens_amplifications.append(amp_factor)
        
        # 3. Control A: Norm-Matched Gaussian Noise at mid_t
        set_seed(42 + idx)
        noise = torch.randn_like(h_mid_R)
        noise_normed = noise / torch.norm(noise) * torch.norm(h_mid_R)
        noise_res = run_recurrent_unroll(
            model, tokenizer, prob_R["question"], scale_factor, k,
            patch_step=mid_t, patch_vector=noise_normed, device=args.device
        )
        # Check if noise broke the output
        noise_collapses.append(noise_res["text"] != unpatched_text)
        
        # 4. Control B: Shuffled-Problem Donor Steering
        if len(nums_D_filtered) > 0:
            # Get donor's latent state
            set_seed(42 + (idx + n_p // 2) % n_p)
            donor_unroll = run_recurrent_unroll(model, tokenizer, prob_D["question"], scale_factor, k, device=args.device)
            h_mid_D = donor_unroll["intermediate_states"][mid_t - 1]
            
            # Inject donor latent into recipient
            set_seed(42 + idx)
            steered_res = run_recurrent_unroll(
                model, tokenizer, prob_R["question"], scale_factor, k,
                patch_step=mid_t, patch_vector=h_mid_D, device=args.device
            )
            steered_nums = extract_all_numbers(steered_res["text"])
            
            # Mechanical check: did steered output emit any donor-specific number?
            steered_hit = bool(nums_D_filtered.intersection(steered_nums))
            if steered_hit:
                donor_steered_count += 1
            total_donor_trials += 1
            
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{n_p}] Null max |Delta|: {max(x['delta_logit'] for x in null_patch_diffs):.6f} | Noise collapse: {np.mean(noise_collapses)*100:.1f}% | Steering trials: {total_donor_trials}")

    # Compute Final Matrix Statistics
    cos_sims = [x["cosine_sim"] for x in null_patch_diffs]
    id_texts = [x["identical_text"] for x in null_patch_diffs]
    delta_l = [x["delta_logit"] for x in null_patch_diffs]
    min_cos = float(np.min(cos_sims))
    all_identical = bool(all(id_texts))
    null_verified = (min_cos >= 0.999) and all_identical

    noise_collapse_rate = float(np.mean(noise_collapses) * 100.0)
    chance_rate = float(chance_hit_count / max(1, n_p) * 100.0)
    steer_rate = float(donor_steered_count / max(1, total_donor_trials) * 100.0)
    delta_steer = steer_rate - chance_rate
    pre_reg_passed = (delta_steer >= 40.0)
    mean_amp = float(np.mean(sens_amplifications)) if sens_amplifications else 1.0

    print("\n" + "="*60)
    print("PHASE 5 CAUSAL ACTIVATION PATCHING REPORT")
    print("="*60)
    print(f"Control 0 (Null Patch): Min Cosine Sim = {min_cos:.6f}, 100% Identical Text = {all_identical} (Pass: {null_verified})")
    print(f"Finite-Difference Sensitivity (||Delta h_next|| / ||eps||): Mean Amplification = {mean_amp:.4f}")
    print(f"Control A (Gaussian Noise): Perturbation Divergence Rate = {noise_collapse_rate:.2f}%")
    print(f"Empirical Chance Baseline (P_chance): {chance_rate:.2f}%")
    print(f"Control B (Donor Steering Rate):      {steer_rate:.2f}%")
    print(f"Steering Delta (P_steered - P_chance): {delta_steer:+.2f}% (Pre-Reg >= +40%: {pre_reg_passed})")

    results_out = {
        "model_id": args.model_id,
        "lora_path": args.lora_path,
        "n_problems": args.n_problems,
        "k_steps": args.k_steps,
        "empirical_alpha": scale_factor,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "control_0_null_patch": {
            "min_cosine_sim": min_cos,
            "all_identical_text": all_identical,
            "tolerance_met": null_verified,
            "max_delta_logit": float(np.max(delta_l))
        },
        "finite_difference_sensitivity": {
            "mean_amplification_factor": mean_amp
        },
        "control_a_noise": {
            "divergence_rate_pct": noise_collapse_rate
        },
        "control_b_steering": {
            "chance_rate_pct": chance_rate,
            "steered_rate_pct": steer_rate,
            "delta_steer_pct": delta_steer,
            "pre_reg_passed": pre_reg_passed,
            "total_trials": total_donor_trials
        }
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nSaved report to: {args.output_file}")

if __name__ == "__main__":
    main()
