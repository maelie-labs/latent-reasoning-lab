#!/usr/bin/env python3
"""
scripts/25_causal_patching.py
Causal Activation Patching & Intervention Experiment.
Proves whether continuous latent thought vectors h_k are causally load-bearing
or merely readable epiphenomena.

Three Conditions evaluated on test problems:
1. Control (Clean Recurrence K=6): Standard continuous latent recurrence.
2. Probe-Projected Patch: At loop k=3, decode h_3 -> top-1 token T_probe -> re-embed as W_E(T_probe) -> continue loops 4..6.
3. Counterfactual Thought Patch: At loop k=3, swap h_3 with h_3_other from a completely different problem -> continue loops 4..6.
   Measures whether injecting thought B into problem A causally steers the final answer.
"""

import os
import re
import json
import time
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import transformers.modeling_utils

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

class ThoughtStreamProbe(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size, bias=False)
        )
    def forward(self, h):
        return self.net(h)

def run_recurrent_with_intervention(
    model, tokenizer, prompt, scale_factor, device,
    intervention_type="none", # "none", "probe_reembed", "counterfactual"
    patch_vector=None,
    patch_step=3,
    probe=None,
    max_steps=6,
    max_new_tokens=512
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
            output_hidden_states=True
        )
    past_key_values = outputs.past_key_values
    h_curr = outputs.hidden_states[-1][:, -1:, :]
    
    embed_layer = model.get_input_embeddings()
    
    # Recurrent loop
    for step in range(1, max_steps + 1):
        # Apply intervention at target step
        if step == patch_step and intervention_type != "none":
            if intervention_type == "probe_reembed" and probe is not None:
                logits = probe(h_curr)
                top1_id = torch.argmax(logits, dim=-1)
                top1_tok = tokenizer.decode(top1_id[0, 0].item())
                re_embedded = embed_layer(top1_id) * scale_factor
                h_curr = re_embedded
            elif intervention_type == "counterfactual" and patch_vector is not None:
                h_curr = patch_vector.to(device)
                
        with torch.no_grad():
            out_step = model(
                inputs_embeds=h_curr * scale_factor,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
        past_key_values = out_step.past_key_values
        h_curr = out_step.hidden_states[-1][:, -1:, :]
        
    # 3. Generate final answer from post-recurrent KV cache
    next_token = torch.argmax(model.lm_head(h_curr)[0, -1]).unsqueeze(0)
    generated_tokens = [next_token.item()]
    
    for _ in range(max_new_tokens):
        if next_token.item() == tokenizer.eos_token_id:
            break
        step_out = model(
            input_ids=next_token.unsqueeze(0),
            past_key_values=past_key_values,
            use_cache=True
        )
        past_key_values = step_out.past_key_values
        next_token = torch.argmax(step_out.logits[0, -1]).unsqueeze(0)
        generated_tokens.append(next_token.item())
        
    gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return {
        "final_hidden": h_curr.detach().cpu(),
        "generated_text": gen_text
    }

def get_loop3_vector(model, tokenizer, prompt, scale_factor, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
            output_hidden_states=True
        )
    past_key_values = outputs.past_key_values
    h_curr = outputs.hidden_states[-1][:, -1:, :]
    for step in range(1, 4):
        with torch.no_grad():
            out_step = model(
                inputs_embeds=h_curr * scale_factor,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
        past_key_values = out_step.past_key_values
        h_curr = out_step.hidden_states[-1][:, -1:, :]
    return h_curr.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--lora", type=str, default="checkpoints/lora_recurrent_1.5b")
    parser.add_argument("--probe", type=str, default="checkpoints/thought_probe.pt")
    args = parser.parse_args()
    
    device = args.device
    model_id = args.model
    lora_path = args.lora
    probe_path = args.probe
    
    print("=" * 80)
    print(f"CAUSAL ACTIVATION PATCHING EXPERIMENT ({model_id}) on {device}")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    lora_model.eval()
    
    embed_weights = lora_model.get_input_embeddings().weight.detach()
    avg_embed_norm = torch.norm(embed_weights.float(), dim=-1).mean().item()
    hidden_dim = lora_model.config.hidden_size
    scale_factor = avg_embed_norm / (hidden_dim ** 0.5)
    
    probe = ThoughtStreamProbe(hidden_dim, lora_model.config.vocab_size).to(device, dtype=torch.bfloat16)
    probe.load_state_dict(torch.load(probe_path, map_location=device, weights_only=True))
    probe.eval()
    
    # Define two distinct problems with completely different answers and contexts
    prob_A = {
        "id": "prob_A",
        "question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers market?",
        "answer": 18.0
    }
    prob_B = {
        "id": "prob_B",
        "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run every week?",
        "answer": 540.0
    }
    
    prompt_A = tokenizer.apply_chat_template([{"role": "user", "content": prob_A["question"]}], tokenize=False, add_generation_prompt=True)
    prompt_B = tokenizer.apply_chat_template([{"role": "user", "content": prob_B["question"]}], tokenize=False, add_generation_prompt=True)
    
    print("\n--- STEP 1: Extracting Clean Loop 3 Vector from Problem B (Meters/Sprints) ---")
    h3_B = get_loop3_vector(lora_model, tokenizer, prompt_B, scale_factor, device)
    print(f"h3_B extracted with shape: {h3_B.shape}, norm: {torch.norm(h3_B.float()).item():.2f}")
    
    print("\n--- STEP 2: Running Problem A (Ducks/Eggs) Clean Control (K=6) ---")
    res_clean_A = run_recurrent_with_intervention(
        lora_model, tokenizer, prompt_A, scale_factor, device,
        intervention_type="none", max_steps=6
    )
    print("Clean Output A:")
    print(res_clean_A["generated_text"][:300])
    
    print("\n--- STEP 3: Running Problem A with Counterfactual Patch from Problem B at Loop 3 ---")
    res_patched_A = run_recurrent_with_intervention(
        lora_model, tokenizer, prompt_A, scale_factor, device,
        intervention_type="counterfactual", patch_vector=h3_B, patch_step=3, max_steps=6
    )
    print("Patched Output A (Injected B's thought into A):")
    print(res_patched_A["generated_text"][:300])
    
    print("\n--- STEP 4: Running Problem A with Probe-Decoded Token Re-embedded Patch at Loop 3 ---")
    res_probe_A = run_recurrent_with_intervention(
        lora_model, tokenizer, prompt_A, scale_factor, device,
        intervention_type="probe_reembed", patch_step=3, probe=probe, max_steps=6
    )
    print("Probe-Patched Output A:")
    print(res_probe_A["generated_text"][:300])
    
    causal_results = {
        "problem_A": prob_A,
        "problem_B": prob_B,
        "clean_output_A": res_clean_A["generated_text"],
        "counterfactual_patched_output_A": res_patched_A["generated_text"],
        "probe_patched_output_A": res_probe_A["generated_text"]
    }
    with open("data/causal_patching_results.json", "w") as f:
        json.dump(causal_results, f, indent=2)
    print("\nSaved causal patching results to data/causal_patching_results.json")

if __name__ == "__main__":
    main()
