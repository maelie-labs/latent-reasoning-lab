#!/usr/bin/env python3
"""
studies/dynamic_registers/scripts/07_attention_decomposition.py
Phase 5: Attention Allocation Analysis for Staged Dynamic Registers.
Decomposes answer-phase attention mass across:
1. Prompt tokens
2. Header tokens ([R1: ...], [R2: ...], [R3: ...])
3. Register tokens (<|reg|>...<|/reg|>)
4. Thinking transition punctuation (</think>\n\n)
"""

import os
import re
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_token_spans(tokenizer, full_input_ids, target_substring):
    """Finds token indices in full_input_ids corresponding to target_substring."""
    full_text = tokenizer.decode(full_input_ids[0], skip_special_tokens=False)
    matches = list(re.finditer(re.escape(target_substring), full_text))
    if not matches:
        return []

    # Map character positions to token indices
    token_indices = []
    curr_char = 0
    tok_spans = []
    for tok_idx, tid in enumerate(full_input_ids[0]):
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        start = curr_char
        end = curr_char + len(tok_str)
        tok_spans.append((start, end, tok_idx))
        curr_char = end

    for m in matches:
        m_start, m_end = m.start(), m.end()
        for s, e, idx in tok_spans:
            if max(s, m_start) < min(e, m_end): # overlap
                token_indices.append(idx)

    return sorted(list(set(token_indices)))

def analyze_attention_mass(model, tokenizer, prompt_text, full_think_text, device="cuda:0", num_gen_tokens=30):
    full_prefix = prompt_text + full_think_text + "\n</think>\n\n"
    enc = tokenizer(full_prefix, return_tensors="pt").to(device)
    input_ids = enc.input_ids
    total_prefix_len = input_ids.shape[1]

    # Partition prefix indices
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    L_prompt = len(prompt_ids)
    prompt_indices = set(range(L_prompt))

    # Register matches
    reg_matches = re.findall(r"<\|reg\|>.*?<\|/reg\|>", full_think_text)
    reg_indices = set()
    for rm in reg_matches:
        reg_indices.update(find_token_spans(tokenizer, input_ids, rm))

    # Header matches
    header_indices = set()
    for h in [
        "[R1: Identify givens, constraints, and target variable]",
        "[R2: Compute intermediate operations and verify relations]",
        "[R3: Execute final deduction and verify constraints]"
    ]:
        header_indices.update(find_token_spans(tokenizer, input_ids, h))

    # Transition tokens
    trans_indices = set(find_token_spans(tokenizer, input_ids, "\n</think>\n\n"))

    # Remaining think tokens
    think_other_indices = set(range(L_prompt, total_prefix_len)) - reg_indices - header_indices - trans_indices

    # Run forward pass with output_attentions=True to decode answer tokens
    masses = {
        "prompt": 0.0,
        "headers": 0.0,
        "registers": 0.0,
        "transition": 0.0,
        "think_other": 0.0
    }

    curr_input = input_ids
    with torch.inference_mode():
        for step in range(num_gen_tokens):
            out = model(input_ids=curr_input, output_attentions=True)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # out.attentions is tuple of 28 tensors, each (1, num_heads, seq_len, seq_len)
            # We look at attention from the last token to all preceding prefix tokens
            step_attns = []
            for layer_attn in out.attentions:
                last_row = layer_attn[0, :, -1, :total_prefix_len].mean(dim=0) # avg over heads
                step_attns.append(last_row)
            
            mean_attn = torch.stack(step_attns, dim=0).mean(dim=0) # avg over layers
            
            p_mass = mean_attn[list(prompt_indices)].sum().item() if prompt_indices else 0.0
            h_mass = mean_attn[list(header_indices)].sum().item() if header_indices else 0.0
            r_mass = mean_attn[list(reg_indices)].sum().item() if reg_indices else 0.0
            t_mass = mean_attn[list(trans_indices)].sum().item() if trans_indices else 0.0
            o_mass = mean_attn[list(think_other_indices)].sum().item() if think_other_indices else 0.0
            
            total_mass = p_mass + h_mass + r_mass + t_mass + o_mass
            if total_mass > 0:
                masses["prompt"] += p_mass / total_mass
                masses["headers"] += h_mass / total_mass
                masses["registers"] += r_mass / total_mass
                masses["transition"] += t_mass / total_mass
                masses["think_other"] += o_mass / total_mass

            curr_input = torch.cat([curr_input, next_token], dim=1)

    for k in masses:
        masses[k] = (masses[k] / num_gen_tokens) * 100.0

    return masses

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "studies/dynamic_registers/checkpoints/merged_arm1_dynamic_registers_qwen3_1.7b"
    output_path = "studies/dynamic_registers/data/attention_decomposition_arm1.json"
    streaming_file = "studies/dynamic_registers/data/streaming_arm1_dynamic_registers.jsonl"

    print("=" * 80)
    print("PHASE 5: ATTENTION DECOMPOSITION ANALYSIS (Arm 1 Dynamic Registers)")
    print(f"Model: {model_path} | Device: {device}")
    print("=" * 80)

    print("Loading model with eager attention...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    # Load evaluated problems that contain authentic registers
    records = []
    with open(streaming_file) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "<|reg|>" in r.get("content", "") and "</think>" in r.get("content", ""):
                    records.append(r)
                if len(records) >= 30:
                    break

    print(f"Loaded {len(records)} authentic trajectory samples.")

    aggregated = {
        "prompt": [],
        "headers": [],
        "registers": [],
        "transition": [],
        "think_other": []
    }

    for r in tqdm(records, desc="Decomposing attention"):
        q = r.get("question") or r.get("id")
        content = r["content"]
        think_part = content.split("</think>")[0].strip()
        prompt_text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"

        masses = analyze_attention_mass(model, tokenizer, prompt_text, think_part, device=device)
        for k in aggregated:
            aggregated[k].append(masses[k])

    results = {
        "num_samples": len(records),
        "mean_attention_pct": {k: float(np.mean(v)) for k, v in aggregated.items()},
        "std_attention_pct": {k: float(np.std(v)) for k, v in aggregated.items()}
    }

    print("\n" + "=" * 80)
    print("ATTENTION ALLOCATION BREAKDOWN (Answer-Phase):")
    for k, v in results["mean_attention_pct"].items():
        std_v = results["std_attention_pct"][k]
        print(f"  * {k.capitalize():<12}: {v:6.2f}% (±{std_v:.2f}%)")
    print("=" * 80)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
