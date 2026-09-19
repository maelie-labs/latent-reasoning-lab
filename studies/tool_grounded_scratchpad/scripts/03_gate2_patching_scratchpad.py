#!/usr/bin/env python3
"""
studies/tool_grounded_scratchpad/scripts/03_gate2_patching_scratchpad.py
Gate 2 Mechanistic Counterfactual Steering Audit for Tool-Grounded Scratchpad.

Tests whether the model's downstream answer decoder is causally bound to the
working memory table entries returned in `<tool_response>`.

Protocol:
1. Form $N=100$ disjoint problem pairs (Donor $D$, Recipient $R$) with $X_D \neq X_R$.
2. Control 0: Unperturbed generation on Recipient $R$ with authentic scratchpad.
3. Chance Baseline: Frequency of $X_D$ in unperturbed generation.
4. Counterfactual Patch: Inject donor variables containing $X_D$ into Recipient $R$'s `<tool_response>`.
5. Measure $\Delta\text{Steer} = P(X_D \mid \text{corrupted}) - P(X_D \mid \text{chance})$.

Assertion:
- In custom registers (<|reg|>): $\Delta\text{Steer} = 0.00\%$ (attentional passivity).
- In native tool scratchpad: Assert $\Delta\text{Steer} \ge 25.0\%$ (active causal steering).
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from math_verify import parse, verify

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

SCRATCHPAD_TOOL = {
    "name": "update_scratchpad",
    "description": "Updates key-value entries in the working memory scratchpad.",
    "parameters": {
        "type": "object",
        "properties": {
            "variables": {
                "type": "object",
                "description": "Key-value variable map to update"
            }
        },
        "required": ["variables"]
    }
}

def check_math_match(pred_text: str, gold_target: str) -> bool:
    if not pred_text or not gold_target:
        return False
    try:
        if "####" in gold_target:
            gold_target = gold_target.split("####")[-1].strip()
        if "\\boxed{" not in gold_target:
            gold_target = f"\\boxed{{{gold_target}}}"
        gold_p = parse(gold_target, parsing_timeout=None)
        pred_p = parse(pred_text, parsing_timeout=None)
        if gold_p and pred_p:
            return bool(verify(gold_p, pred_p))
    except Exception:
        pass
    return False

def build_scratchpad_prompt(question: str, scratchpad_dict: dict, tokenizer: AutoTokenizer) -> str:
    call_args = json.dumps({"variables": scratchpad_dict}, ensure_ascii=False)
    response_content = json.dumps({"scratchpad": scratchpad_dict}, ensure_ascii=False)

    messages = [
        {"role": "system", "content": "You are a helpful assistant with an in-place working memory scratchpad."},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "update_scratchpad",
                    "arguments": call_args
                }
            }]
        },
        {
            "role": "tool",
            "name": "update_scratchpad",
            "content": response_content
        }
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[SCRATCHPAD_TOOL],
        tokenize=False,
        add_generation_prompt=True
    )
    # Ensure immediate answer phase
    if not rendered.endswith("<think>\n\n</think>\n\n"):
        if rendered.endswith("<|im_start|>assistant\n"):
            rendered += "<think>\n\n</think>\n\n"
    return rendered

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lora_tool_scratchpad_qwen3_4b/best_checkpoint")
    parser.add_argument("--output_file", type=str, default="studies/tool_grounded_scratchpad/data/gate2_steering_results.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_pairs", type=int, default=100)
    args = parser.parse_args()

    print("=" * 80)
    print("GATE 2 COUNTERFACTUAL STEERING AUDIT: TOOL SCRATCHPAD")
    print(f"Base Model: {args.model_id} | Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device} | Number of Pairs: {args.num_pairs}")
    print("=" * 80)

    # Load dev / test problems
    data_file = os.path.join(SCRIPT_DIR, "../data/dev_tool_scratchpad.jsonl")
    with open(data_file) as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Filter records with clear numeric answers
    valid_records = []
    for r in records:
        gt = r.get("ground_truth", "").strip()
        if gt and r.get("final_scratchpad"):
            valid_records.append(r)

    print(f"Found {len(valid_records)} valid records with scratchpads.")
    if len(valid_records) < args.num_pairs:
        # Fall back to train records if needed
        train_file = os.path.join(SCRIPT_DIR, "../data/train_tool_scratchpad.jsonl")
        with open(train_file) as f:
            for line in f:
                if len(valid_records) >= args.num_pairs * 2:
                    break
                r = json.loads(line)
                gt = r.get("ground_truth", "").strip()
                if gt and r.get("final_scratchpad"):
                    valid_records.append(r)

    rng = random.Random(42)
    rng.shuffle(valid_records)

    # Form disjoint pairs (Donor, Recipient) with gt_donor != gt_recip
    pairs = []
    for i in range(len(valid_records)):
        recip = valid_records[i]
        gt_r = recip["ground_truth"]
        for j in range(len(valid_records)):
            if i == j:
                continue
            donor = valid_records[j]
            gt_d = donor["ground_truth"]
            if gt_d != gt_r and len(gt_d) <= 10:
                pairs.append((donor, recip))
                break
        if len(pairs) >= args.num_pairs:
            break

    print(f"Constructed {len(pairs)} disjoint counterfactual pairs.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model onto {args.device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading LoRA adapter from {args.checkpoint}...")
        model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    steered_count = 0
    chance_count = 0
    null_match_count = 0
    pair_results = []

    for idx, (donor, recip) in enumerate(tqdm(pairs, desc="Gate 2 Counterfactual Patching")):
        gt_donor = donor["ground_truth"]
        gt_recip = recip["ground_truth"]

        # 1. Unperturbed Recipient prompt
        prompt_unpert = build_scratchpad_prompt(recip["question"], recip["final_scratchpad"], tokenizer)
        enc_u = tokenizer(prompt_unpert, return_tensors="pt").to(args.device)

        with torch.no_grad():
            out_u = model.generate(
                **enc_u,
                max_new_tokens=512,
                do_sample=False, # Greedy for exact counterfactual measurement
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        text_u = tokenizer.decode(out_u[0, enc_u.input_ids.shape[1]:], skip_special_tokens=True)

        # Check if authentic generation matched true recipient answer
        if check_math_match(text_u, gt_recip):
            null_match_count += 1
        # Check chance appearance of donor answer
        if check_math_match(text_u, gt_donor):
            chance_count += 1

        # 2. Corrupted Recipient prompt: swap final target variable with donor target
        corrupted_scratchpad = dict(recip["final_scratchpad"])
        # Inject donor target
        corrupted_scratchpad["final_target"] = gt_donor
        corrupted_scratchpad["ans"] = gt_donor
        prompt_corr = build_scratchpad_prompt(recip["question"], corrupted_scratchpad, tokenizer)
        enc_c = tokenizer(prompt_corr, return_tensors="pt").to(args.device)

        with torch.no_grad():
            out_c = model.generate(
                **enc_c,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        text_c = tokenizer.decode(out_c[0, enc_c.input_ids.shape[1]:], skip_special_tokens=True)

        is_steered = check_math_match(text_c, gt_donor)
        if is_steered:
            steered_count += 1

        pair_results.append({
            "pair_idx": idx,
            "donor_id": donor["id"],
            "recip_id": recip["id"],
            "donor_target": gt_donor,
            "recip_target": gt_recip,
            "unperturbed_output": text_u[:200],
            "corrupted_output": text_c[:200],
            "is_steered": is_steered
        })

    n = len(pairs)
    p_steered = (steered_count / n) * 100.0
    p_chance = (chance_count / n) * 100.0
    delta_steer = p_steered - p_chance
    p_null = (null_match_count / n) * 100.0

    print("\n" + "=" * 80)
    print("GATE 2 STEERING AUDIT RESULTS")
    print(f"Total Evaluated Pairs:         {n}")
    print(f"Authentic Output Accuracy:     {p_null:.2f}% ({null_match_count}/{n})")
    print(f"Chance Target Baseline:        {p_chance:.2f}% ({chance_count}/{n})")
    print(f"Corrupted Target Steering:     {p_steered:.2f}% ({steered_count}/{n})")
    print(f"NET CAUSAL STEERING DELTA:    {delta_steer:+.2f}%")
    print(f"Gate 2 Verdict (>= 25.0%):     {'PASS [CAUSALLY ACTIVE]' if delta_steer >= 25.0 else 'FAIL [PASSIVE ATTENTION]'}")
    print("=" * 80)

    summary = {
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
        "num_pairs": n,
        "authentic_accuracy_pct": round(p_null, 2),
        "chance_baseline_pct": round(p_chance, 2),
        "corrupted_steering_pct": round(p_steered, 2),
        "delta_steer_pct": round(delta_steer, 2),
        "verdict": "PASS" if delta_steer >= 25.0 else "FAIL"
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump({"summary": summary, "evaluations": pair_results}, f, indent=2)
    print(f"Results written to {args.output_file}")

if __name__ == "__main__":
    main()
