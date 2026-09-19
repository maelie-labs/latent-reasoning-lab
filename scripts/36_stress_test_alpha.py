#!/usr/bin/env python3
"""
scripts/36_stress_test_alpha.py
PHASE 4: Latent Scale Factor (alpha) and Horizon (K) Stress Testing.

Evaluates recurrent latent stability across:
- Horizons: K in {6, 16, 32, 64, 128}
- Scale factors: {0.5*alpha, 1.0*alpha (calibrated), 2.0*alpha, 3.0*alpha (legacy uncorrected), 1.0 (unscaled)}
- Models: Qwen3-1.7B, Qwen3-4B (and 8B/14B)

Tracks:
1. Per-step L2 norm trajectory: ||h_k||_2 for k in 0..K.
2. Cosine similarity drift: cos(h_k, h_0).
3. Numerical divergence detection (exploding/vanishing).
"""

import os
import json
import time
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

CALIBRATED_ALPHAS = {
    "qwen/qwen3-1.7b": 0.011440,
    "qwen/qwen3-4b": 0.006702,
    "qwen/qwen3-8b": 0.008500,
    "qwen/qwen3-14b": 0.005200
}

TEST_PROMPTS = [
    "Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. She sells the remainder for $2 each. How much does she make per day?",
    "A train travels 240 miles in 4 hours. How long will it take to travel 360 miles at the same speed?",
    "Find all real solutions to the equation x^2 - 5x + 6 = 0.",
    "The sum of three consecutive odd integers is 57. What are the integers?",
    "What is the probability of rolling a sum of 7 with two fair six-sided dice?"
]

def run_stress_test_prompt(model, tokenizer, prompt, scale_factor, max_k, device):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)
    
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_end_id is None or think_end_id == tokenizer.unk_token_id:
        think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]
        
    norms = []
    cosines = []
    stop_probs = []
    
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        h0 = out.hidden_states[-1][:, -1:, :].float()
        h0_norm = torch.norm(h0).item()
        norms.append(h0_norm)
        cosines.append(1.0)
        
        logits_0 = model.lm_head(out.hidden_states[-1][:, -1:, :])
        p0 = torch.softmax(logits_0.float(), dim=-1)[0, -1, think_end_id].item()
        stop_probs.append(round(p0, 6))
        
        curr_latent = out.hidden_states[-1][:, -1:, :]
        
        for k in range(1, max_k + 1):
            scaled_latent = curr_latent * scale_factor
            step_out = model(
                inputs_embeds=scaled_latent,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]
            
            hk_float = curr_latent.float()
            hk_norm = torch.norm(hk_float).item()
            norms.append(hk_norm)
            
            # Cosine similarity with h0
            cos = F.cosine_similarity(hk_float.view(1, -1), h0.view(1, -1)).item()
            cosines.append(cos)
            
            # P(</think>) from LM head
            logits_k = model.lm_head(curr_latent)
            pk = torch.softmax(logits_k.float(), dim=-1)[0, -1, think_end_id].item()
            stop_probs.append(round(pk, 6))
            
            if np.isnan(hk_norm) or np.isinf(hk_norm) or hk_norm > 1e6:
                print(f"    [DIVERGED] Step {k}: Norm exploded to {hk_norm}")
                break
                
    return {
        "max_k": max_k,
        "scale_factor": scale_factor,
        "steps_completed": len(norms) - 1,
        "norms": [round(n, 4) for n in norms],
        "cosines": [round(c, 4) for c in cosines],
        "stop_probs": stop_probs,
        "terminal_norm": round(norms[-1], 4),
        "terminal_cosine": round(cosines[-1], 4),
        "terminal_stop_prob": stop_probs[-1],
        "is_bounded": (norms[-1] < 500.0 and not np.isnan(norms[-1]))
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Scale Factor & Horizon Stress Testing.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_k", type=int, default=128)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    model_key = args.model_id.lower()
    base_alpha = CALIBRATED_ALPHAS.get(model_key, 0.011440)

    if args.output_file is None:
        tag = args.model_id.replace("/", "_").lower()
        adapter_tag = "_adapter" if args.adapter_path else ""
        args.output_file = os.path.join(os.path.dirname(__file__), "..", "data", f"stress_test_alpha_{tag}{adapter_tag}_k{args.max_k}.json")

    print(f"=== PHASE 4: Scale Factor & Horizon Stress Testing for {args.model_id} on {args.device} ===")
    print(f"Calibrated Base Alpha: {base_alpha:.6f} | Max Horizon K: {args.max_k}")
    if args.adapter_path:
        print(f"Loaded Adapter: {args.adapter_path}")

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip("\"'\n")
                    break

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token, trust_remote_code=True)
    print("Loading model in pure bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    if args.adapter_path:
        from peft import PeftModel
        print(f"Applying LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # The 5 Scale Factor Conditions in the Matrix
    scale_conditions = [
        ("0.5x_calibrated", 0.5 * base_alpha),
        ("1.0x_calibrated", 1.0 * base_alpha),
        ("2.0x_calibrated", 2.0 * base_alpha),
        ("3.0x_legacy_overshoot", 3.0 * base_alpha),  # The uncorrected legacy formula value
        ("1.0_unscaled_control", 1.0)                 # Raw unscaled control (proving explosion)
    ]

    results = {
        "model_id": args.model_id,
        "device": args.device,
        "base_alpha": base_alpha,
        "max_k": args.max_k,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conditions": {}
    }

    print("\n" + "="*70)
    print(f"{'Condition':<25} | {'Alpha':<10} | {'Term Norm':<12} | {'Term Cos':<10} | {'Bounded'}")
    print("="*70)

    for cond_name, alpha_val in scale_conditions:
        cond_prompt_runs = []
        for p_idx, prompt in enumerate(TEST_PROMPTS):
            run_res = run_stress_test_prompt(model, tokenizer, prompt, alpha_val, args.max_k, args.device)
            cond_prompt_runs.append(run_res)

        mean_term_norm = np.mean([r["terminal_norm"] for r in cond_prompt_runs])
        mean_term_cos = np.mean([r["terminal_cosine"] for r in cond_prompt_runs])
        all_bounded = all(r["is_bounded"] for r in cond_prompt_runs)

        results["conditions"][cond_name] = {
            "scale_factor": round(alpha_val, 6),
            "mean_terminal_norm": round(float(mean_term_norm), 2),
            "mean_terminal_cosine": round(float(mean_term_cos), 4),
            "is_bounded": all_bounded,
            "sample_runs": cond_prompt_runs
        }

        print(f"{cond_name:<25} | {alpha_val:<10.6f} | {mean_term_norm:<12.2f} | {mean_term_cos:<10.4f} | {all_bounded}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved stress test results to: {args.output_file}")

if __name__ == "__main__":
    main()
