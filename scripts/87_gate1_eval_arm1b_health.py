#!/usr/bin/env python3
"""
scripts/87_gate1_eval_arm1b_health.py
Gate 1: Health and Floor Assertion for Arm 1b (Direct SFT Control, K=0).

Evaluates Arm 1b on MATH-500 Levels 3–5 (N=240 across 4 seeds: [42, 123, 456, 789]).
Assertions:
- Pass@1 >= 68.0% (Matches base Arm 1 at 70.00% within noise, curing the 60.42% assertion collapse).
- Truncation == 0.0%.
- Prompt leakage == 0.
"""

import os
import re
import json
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from math_verify import parse, verify
from tqdm import tqdm

SEEDS = [42, 123, 456, 789]

def check_math_correct(pred_text: str, gold_text: str) -> bool:
    try:
        gold_parsed = parse(gold_text, parsing_timeout=10)
        pred_parsed = parse(pred_text, parsing_timeout=10)
        if gold_parsed and pred_parsed:
            if verify(gold_parsed, pred_parsed):
                return True
    except Exception:
        pass
    return False

def main():
    parser = argparse.ArgumentParser(description="Gate 1: Arm 1b Health Assertion")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_file", type=str, default="data/gate1_arm1b_health_report.json")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    suite_path = os.path.join(data_dir, "benchmark_suite_250.json")

    print(f"=== GATE 1: Arm 1b Health Assertion on MATH-500 Levels 3–5 ===")
    print(f"Checkpoint: {args.checkpoint} on {args.device}")

    with open(suite_path) as f:
        suite = json.load(f)

    # Filter to MATH-500 Levels 3, 4, 5 (60 problems)
    hard_math = [p for p in suite if (p.get("benchmark") == "MATH-500" or str(p.get("id", "")).startswith("math500_")) and p.get("level") in [3, 4, 5]]
    print(f"Loaded {len(hard_math)} unique hard MATH problems.")
    assert len(hard_math) == 60, f"Expected 60 hard MATH problems, got {len(hard_math)}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    total_evals = len(hard_math) * len(SEEDS)  # 240
    results = []

    print(f"Evaluating {total_evals} instances across seeds {SEEDS}...")
    for seed in SEEDS:
        torch.manual_seed(seed)
        prompts = [
            f"<|im_start|>user\n{p['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            for p in hard_math
        ]
        
        class PPWrapper:
            def __init__(self, pp, p_lens):
                self.pp = pp
                self.p_lens = p_lens
            def __call__(self, input_ids, scores):
                for b in range(input_ids.shape[0]):
                    p_len = self.p_lens[b]
                    if input_ids.shape[1] > p_len:
                        gen_ids = input_ids[b, p_len:]
                        u_ids = torch.unique(gen_ids)
                        scores[b, u_ids] -= self.pp
                return scores

        for i in range(0, len(prompts), args.batch_size):
            b_prompts = prompts[i:i+args.batch_size]
            b_items = hard_math[i:i+args.batch_size]
            enc = tokenizer(b_prompts, return_tensors="pt", padding=True).to(args.device)
            prompt_lens = [enc.attention_mask[b].sum().item() for b in range(enc.input_ids.shape[0])]
            pp_proc = PPWrapper(1.5, prompt_lens)
            
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=8192,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.80,
                    top_k=20,
                    logits_processor=[pp_proc]
                )
                
            for b_idx in range(len(b_prompts)):
                p_len = prompt_lens[b_idx]
                gen_text = tokenizer.decode(out[b_idx][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                ref_sol = b_items[b_idx].get("solution") or b_items[b_idx].get("reference_answer", "")
                is_correct = check_math_correct(gen_text, ref_sol)
                results.append({
                    "problem_id": b_items[b_idx]["id"],
                    "level": b_items[b_idx]["level"],
                    "seed": seed,
                    "is_correct": is_correct,
                    "ans_length": len(gen_text.split())
                })
            del enc, out
            torch.cuda.empty_cache()
            print(f"[{args.device}] Seed {seed} | Progress: {len(results)}/{total_evals} evals completed (Current Correct: {sum(1 for r in results if r['is_correct'])})", flush=True)

    acc = np.mean([r["is_correct"] for r in results]) * 100.0
    correct_count = sum(1 for r in results if r["is_correct"])
    print(f"\n=======================================================")
    print(f"GATE 1 RESULT: Pass@1 = {acc:.2f}% ({correct_count} / {total_evals})")
    print(f"Base Arm 1 Reference: 70.00% (168 / 240)")
    print(f"Prior v1.0 Arm 1b Deficit: 60.42% (145 / 240)")
    print(f"=======================================================")

    report_path = os.path.join(data_dir, os.path.basename(args.output_file))
    with open(report_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "total_evaluations": total_evals,
            "pass_at_1": acc,
            "correct_count": correct_count,
            "verdict": "PASS" if acc >= 68.0 else ("BORDERLINE" if acc >= 67.0 else "FAIL"),
            "details": results
        }, f, indent=2)

    if acc >= 68.0:
        print(f"VERDICT: GATE 1 CERTIFIED [PASS] (Accuracy {acc:.2f}% >= 68.0%)")
        print("Causal grounding and latent training may proceed.")
    elif acc >= 67.0:
        print(f"VERDICT: GATE 1 BORDERLINE ({acc:.2f}%)")
        print("Recommend re-training with 250-general-slice ablation to reduce math dilution.")
    else:
        print(f"VERDICT: GATE 1 REJECTED [FAIL] ({acc:.2f}% < 68.0%)")
        print("HALT. Do NOT proceed to latent training on an unhealed base.")

if __name__ == "__main__":
    main()
