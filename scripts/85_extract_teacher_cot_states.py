#!/usr/bin/env python3
"""
scripts/85_extract_teacher_cot_states.py
Extract and cache post-final-norm teacher CoT hidden states at step boundaries
for Step-Level State Distillation in Continuous Latent Recurrence v1.1.

1. Reads math problems and segmented teacher steps from:
   data/curated_train_v1_1_qwen3_1.7b.jsonl and data/curated_dev_v1_1_qwen3_1.7b.jsonl
2. Feeds teacher prompt + thinking steps into Qwen/Qwen3-1.7B on cuda:0:
   Identifies the token boundary index of each step s in {1 .. S}.
3. Extracts post-final-norm hidden state h_s in R^{d_model} at each step boundary.
4. Pre-computes and maps targets for horizons K in {6, 32}, plus the answer-position CODI state.
5. Saves indexed dictionary cache to data/teacher_cot_states_qwen3_1.7b.pt (~250 MB).
"""

import os
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Extract Teacher CoT States for v1.1 Distillation")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train_file", type=str, default="data/curated_train_v1_1_qwen3_1.7b.jsonl")
    parser.add_argument("--dev_file", type=str, default="data/curated_dev_v1_1_qwen3_1.7b.jsonl")
    parser.add_argument("--output_file", type=str, default="data/teacher_cot_states_qwen3_1.7b.pt")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train_path = os.path.join(data_dir, os.path.basename(args.train_file))
    dev_path = os.path.join(data_dir, os.path.basename(args.dev_file))
    out_path = os.path.join(data_dir, os.path.basename(args.output_file))

    print(f"=== Extracting Teacher CoT States for {args.model_id} on {args.device} ===")
    
    # Load traces
    traces = []
    for path in [train_path, dev_path]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("teacher_steps") and len(rec["teacher_steps"]) > 0:
                        traces.append(rec)
    print(f"Total traces with teacher steps: {len(traces)}")

    if not traces:
        print("No traces found to process. Run scripts/84_curate_v1_1_nonthinking_targets.py first.")
        return

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    teacher_state_cache = {}
    
    with torch.inference_mode():
        for rec in tqdm(traces, desc="Extracting Teacher States"):
            prob_id = rec["id"]
            prompt = rec["prompt"]  # Ends in <think>\n
            steps = rec["teacher_steps"]
            S = len(steps)
            if S == 0:
                continue

            # Tokenize prompt
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            
            # Tokenize each step and record boundary token indices
            step_token_ids = []
            boundary_indices = []
            curr_pos = len(prompt_ids)
            
            for s_idx, step_text in enumerate(steps):
                s_ids = tokenizer.encode(step_text + "\n\n", add_special_tokens=False)
                step_token_ids.extend(s_ids)
                curr_pos += len(s_ids)
                boundary_indices.append(curr_pos - 1)  # 0-indexed position of step terminal token
                
            full_input_ids = prompt_ids + step_token_ids
            input_tensor = torch.tensor([full_input_ids], dtype=torch.long, device=args.device)
            
            # Forward pass to extract hidden states
            out = model(input_tensor, output_hidden_states=True)
            last_hidden = out.hidden_states[-1][0]  # (seq_len, d_model) post-final-norm
            
            # Extract step boundary states
            step_states = []
            for b_idx in boundary_indices:
                step_states.append(last_hidden[b_idx].detach().cpu().to(torch.bfloat16))
            step_states_tensor = torch.stack(step_states)  # (S, d_model)
            
            # Map to K=6 and K=32 targets
            targets = {}
            for K in [6, 32]:
                k_targets = []
                for k in range(1, K + 1):
                    # Fractional step index
                    s_target = min(S - 1, max(0, int(round(k * S / K)) - 1))
                    k_targets.append(step_states_tensor[s_target])
                targets[f"K{K}"] = torch.stack(k_targets)  # (K, d_model)
                
            # True CODI fallback target: state at answer position (terminal step)
            targets["codi_answer_state"] = step_states_tensor[-1]  # (d_model,)
            
            teacher_state_cache[prob_id] = targets

    print(f"Extracted teacher states for {len(teacher_state_cache)} problems.")
    torch.save(teacher_state_cache, out_path)
    print(f"Saved teacher state cache to: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
