#!/usr/bin/env python3
"""
scripts/103_eval_trained_controls.py
Evaluate the Trained Matched Controls on the 60 MATH L3-5 Acid Test Problems on GPU 0:
1. Arm 1b-R (Trained Headers Only: 3 headers, 0 latents)
2. Arm 2b-R (Trained Pause Tokens: 3 headers + 24 <pause> tokens)

Enforces:
- Gate 0 Frozen Specification Table (temp=0.7, top_p=0.80, top_k=20, pp=1.5, ans_cap=8192)
- Canonical math_verify 0.9.0
- Device: cuda:0 (RTX 4080)
"""

import os
import json
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from peft import PeftModel
from math_verify import parse, verify

CANONICAL_HEADERS = [
    "[R1: Identify givens, constraints, and target variable]\n",
    "[R2: Compute intermediate operations and verify relations]\n",
    "[R3: Execute final deduction and verify constraints]\n"
]
PAUSE_TOKEN = "<pause>"

class PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float, prompt_len: int):
        self.penalty = penalty
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0 or input_ids.shape[1] <= self.prompt_len:
            return scores
        for b in range(input_ids.shape[0]):
            gen_ids = input_ids[b, self.prompt_len:]
            unique_ids = torch.unique(gen_ids)
            scores[b, unique_ids] -= self.penalty
        return scores

def check_math_correct(pred_text: str, gold_text: str) -> bool:
    try:
        if "####" in gold_text:
            ans_part = gold_text.split("####")[-1].strip()
            gold_target = f"\\boxed{{{ans_part}}}"
        elif "\\boxed{" not in gold_text:
            gold_target = f"\\boxed{{{gold_text.strip()}}}"
        else:
            gold_target = gold_text
        gold_parsed = parse(gold_target, parsing_timeout=None)
        pred_parsed = parse(pred_text, parsing_timeout=None)
        if gold_parsed and pred_parsed:
            return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        pass
    return False

def format_prompt(problem, mode):
    q = problem.get("question") or problem.get("problem", "")
    p = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    for r in range(3):
        p += CANONICAL_HEADERS[r]
        if mode == "headers_pause":
            p += (PAUSE_TOKEN * 8)
    p += "\n</think>\n\n"
    return p

def evaluate_control(mode, checkpoint_path, problems, device="cuda:0", max_ans=8192):
    print("=" * 80)
    print(f"EVALUATING TRAINED CONTROL: {mode.upper()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device} | Problems: {len(problems)}")
    print("=" * 80)

    model_id = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if mode == "headers_pause" and PAUSE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    if mode == "headers_pause":
        base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    correct_count = 0
    results = []

    for idx, prob in enumerate(problems):
        prompt_str = format_prompt(prob, mode)
        enc = tokenizer(prompt_str, return_tensors="pt").to(device)
        prompt_len = enc.input_ids.shape[1]

        lp_list = LogitsProcessorList([PresencePenaltyLogitsProcessor(1.5, prompt_len)])

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_ans,
                do_sample=True,
                temperature=0.7,
                top_p=0.80,
                top_k=20,
                logits_processor=lp_list,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_tokens = out_ids[0, prompt_len:]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        gold_text = prob.get("solution") or prob.get("answer", "")
        is_corr = check_math_correct(pred_text, gold_text)
        if is_corr:
            correct_count += 1

        results.append({
            "id": prob.get("id", idx),
            "level": prob.get("level", 0),
            "is_correct": is_corr,
            "pred_len": len(gen_tokens),
            "pred_sample": pred_text[:200]
        })

        if (idx + 1) % 10 == 0 or idx == len(problems) - 1:
            print(f"[{idx+1:2d}/{len(problems)}] Running Acc: {correct_count/(idx+1)*100:.2f}% ({correct_count}/{idx+1})")

    acc = (correct_count / len(problems)) * 100.0
    print(f"\nFinal Accuracy for {mode}: {acc:.2f}% ({correct_count}/{len(problems)})")
    return {
        "mode": mode,
        "checkpoint": checkpoint_path,
        "accuracy": acc,
        "correct": correct_count,
        "total": len(problems),
        "results": results
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_file", type=str, default="data/trained_controls_acid_test_results.json")
    args = parser.parse_args()

    # Load 60 hardest MATH problems (Levels 3, 4, 5)
    with open("data/benchmark_suite_250.json") as f:
        suite = json.load(f)

    math_l3_5 = [p for p in suite if p.get("benchmark") == "MATH-500" and p.get("stratum") in ["Level 3", "Level 4", "Level 5"]]
    print(f"Loaded {len(math_l3_5)} MATH L3-5 test problems.")

    # 1. Evaluate Arm 1b-R (Trained Headers Only)
    res_1b = evaluate_control(
        mode="headers_only",
        checkpoint_path="checkpoints/lora_arm1b_register_ladder_headers_only_qwen3_1.7b/best_checkpoint",
        problems=math_l3_5,
        device=args.device
    )

    # 2. Evaluate Arm 2b-R (Trained Pause Tokens)
    res_2b = evaluate_control(
        mode="headers_pause",
        checkpoint_path="checkpoints/lora_arm2b_register_ladder_headers_pause_qwen3_1.7b/best_checkpoint",
        problems=math_l3_5,
        device=args.device
    )

    all_res = {
        "arm1b_headers_only": res_1b,
        "arm2b_headers_pause": res_2b,
        "arm3_reference_accuracy": 45.00,  # 27/60 from Check C
        "delta_arm3_vs_trained_pause": 45.00 - res_2b["accuracy"],
        "delta_arm3_vs_trained_headers": 45.00 - res_1b["accuracy"]
    }

    with open(args.output_file, "w") as f:
        json.dump(all_res, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINED CONTROLS COMPARISON ON MATH L3-5 (N=60)")
    print("=" * 80)
    print(f"Arm 3 Register Ladder (24 latents) : 45.00% (27/60)")
    print(f"Arm 1b-R Trained Headers Only       : {res_1b['accuracy']:.2f}% ({res_1b['correct']}/{res_1b['total']})")
    print(f"Arm 2b-R Trained Pause Tokens       : {res_2b['accuracy']:.2f}% ({res_2b['correct']}/{res_2b['total']})")
    print(f"Delta (Arm 3 - Trained Pause)       : {all_res['delta_arm3_vs_trained_pause']:+.2f}%")
    print(f"Delta (Arm 3 - Trained Headers)     : {all_res['delta_arm3_vs_trained_headers']:+.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
