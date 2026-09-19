#!/usr/bin/env python3
"""
scripts/72_eval_arm1_calibrated.py
Calibrated Arm 1 (Base Direct Non-Thinking) Evaluator for Qwen/Qwen3-1.7B.

Uses official non-thinking sampling:
  temperature = 0.7, top_p = 0.80, top_k = 20, presence_penalty = 1.5
Prompt format:
  <|im_start|>user\n{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
Scoring:
  Canonical math_verify parse & verify with LaTeX extraction.
Runs on cuda:0 (RTX 4080 16GB) with batched inference (B=16).
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
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from math_verify import parse, verify

SEEDS = [42, 123, 456, 789]

class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """
    Applies additive presence penalty to tokens that have already appeared.
    Subtracts penalty from logits of any token present in the generated sequence so far.
    """
    def __init__(self, penalty: float, prompt_lens: list):
        self.penalty = penalty
        self.prompt_lens = prompt_lens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        for b in range(input_ids.shape[0]):
            # Only penalize tokens generated in the assistant response to avoid penalizing prompt constants
            p_len = self.prompt_lens[b]
            if input_ids.shape[1] > p_len:
                gen_ids = input_ids[b, p_len:]
                unique_ids = torch.unique(gen_ids)
                scores[b, unique_ids] -= self.penalty
        return scores

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_math_boxed_expression(text):
    if not text or "\\boxed{" not in text:
        return ""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    end = start
    while end < len(text) and depth > 0:
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1
    if depth == 0:
        res = text[start:end-1].strip()
        if "=" in res:
            res = res.split("=")[-1].strip()
        return res.rstrip(".")
    return ""

def clean_and_extract_candidate(cand):
    if not cand:
        return None
    cand = re.sub(r"\\(?:text|mathbf|mathrm)\{([^}]+)\}", r"\1", str(cand))
    cand = re.sub(r"[\$\\%!\s]", "", cand)
    cand = cand.replace(",", "").strip().rstrip(".")
    m_frac = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", cand)
    if m_frac:
        try:
            return float(m_frac.group(1)) / float(m_frac.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    m_div = re.fullmatch(r"(-?\d+)/(-?\d+)", cand)
    if m_div:
        try:
            return float(m_div.group(1)) / float(m_div.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(cand)
    except ValueError:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", cand)
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                pass
    return None

def check_match(pred_text, gold_sol):
    if "####" in gold_sol:
        gt_boxed = gold_sol.split("####")[-1].strip()
    else:
        gt_boxed = extract_math_boxed_expression(gold_sol)
    gt_target = f"\\boxed{{{gt_boxed}}}" if gt_boxed else f"\\boxed{{{gold_sol}}}"
    
    # 1. Primary: math_verify symbolic parser
    try:
        gt_parsed = parse(gt_target)
        pred_parsed = parse(pred_text)
        if gt_parsed and pred_parsed and verify(gt_parsed, pred_parsed):
            return True
    except Exception:
        pass

    # 2. Secondary: direct boxed extraction match
    pred_boxed = extract_math_boxed_expression(pred_text)
    if pred_boxed and gt_boxed:
        try:
            gt_p = parse(f"\\boxed{{{gt_boxed}}}")
            pr_p = parse(f"\\boxed{{{pred_boxed}}}")
            if gt_p and pr_p and verify(gt_p, pr_p):
                return True
        except Exception:
            pass

    return False

def run_evaluation(model, tokenizer, problems, seeds, batch_size=16, max_new_tokens=1024, device="cuda:0", presence_penalty=1.5):
    all_results = []
    total_evals = len(problems) * len(seeds)
    print(f"Starting Calibrated Arm 1 Evaluation: {len(problems)} problems x {len(seeds)} seeds = {total_evals} evaluations.")
    print(f"Device: {device} | Batch Size: {batch_size} | Temp: 0.7 | Top_p: 0.80 | Top_k: 20 | PP: {presence_penalty}")

    global_idx = 0
    t_start = time.time()

    for seed in seeds:
        set_seed(seed)
        print(f"\n--- Running Seed {seed} ({len(problems)} problems) ---")
        
        for i in range(0, len(problems), batch_size):
            batch_probs = problems[i:i+batch_size]
            B = len(batch_probs)

            prompts = []
            for prob in batch_probs:
                q = prob["question"]
                user_msg = f"{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
                raw_prompt = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                prompts.append(raw_prompt)

            tokenizer.padding_side = "left"
            enc = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]

            logits_processors = LogitsProcessorList()
            if presence_penalty > 0.0:
                logits_processors.append(PresencePenaltyLogitsProcessor(presence_penalty, prompt_lens))

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.80,
                    top_k=20,
                    logits_processor=logits_processors,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
            t_dur = time.time() - t0
            dur_per_item = t_dur / B

            for b in range(B):
                prob = batch_probs[b]
                p_len = enc.input_ids.shape[1]
                gen_ids = out[b][p_len:].tolist()
                
                # Strip EOS and trailing padding
                if tokenizer.eos_token_id in gen_ids:
                    eos_pos = gen_ids.index(tokenizer.eos_token_id)
                    gen_ids = gen_ids[:eos_pos]
                    is_truncated = False
                else:
                    is_truncated = (len(gen_ids) >= max_new_tokens)

                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                ans_tokens = len(gen_ids)

                is_correct = False
                if not is_truncated and gen_text:
                    is_correct = check_match(gen_text, prob["solution"])

                all_results.append({
                    "id": prob["id"],
                    "benchmark": prob.get("benchmark", ""),
                    "seed": seed,
                    "is_correct": is_correct,
                    "is_truncated": is_truncated,
                    "ans_tokens": ans_tokens,
                    "dur_s": round(dur_per_item, 3),
                    "content": gen_text
                })
                global_idx += 1

            correct_so_far = sum(r["is_correct"] for r in all_results)
            print(f"Progress: [{global_idx}/{total_evals}] | Current Pass@1: {correct_so_far/global_idx*100:.2f}% | Batch dur: {t_dur:.2f}s", end="\r")

    total_time = time.time() - t_start
    print(f"\nEvaluation Complete in {total_time:.1f}s.")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Calibrated Arm 1 Evaluator for Qwen3-1.7B")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--suite_path", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--presence_penalty", type=float, default=1.5)
    parser.add_argument("--output_file", type=str, default="data/eval_arm1_calibrated_qwen_qwen3-1.7b.json")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    with open(args.suite_path) as f:
        problems = json.load(f)

    if args.smoke_test:
        problems = problems[:4]
        seeds = [42]
    else:
        seeds = SEEDS

    print(f"Loading {args.model_id} on {args.device} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    results = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        seeds=seeds,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        presence_penalty=args.presence_penalty
    )

    total = len(results)
    correct = sum(r["is_correct"] for r in results)
    truncated = sum(r["is_truncated"] for r in results)
    token_counts = [r["ans_tokens"] for r in results]

    gsm_results = [r for r in results if r["benchmark"] == "GSM8K"]
    math_results = [r for r in results if r["benchmark"] == "MATH-500"]

    gsm_acc = (sum(r["is_correct"] for r in gsm_results) / len(gsm_results) * 100) if gsm_results else 0.0
    math_acc = (sum(r["is_correct"] for r in math_results) / len(math_results) * 100) if math_results else 0.0
    overall_acc = (correct / total * 100) if total else 0.0

    summary = {
        "model_id": args.model_id,
        "arm": "Arm 1 (Base Direct Calibrated)",
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.80,
            "top_k": 20,
            "presence_penalty": args.presence_penalty,
            "max_new_tokens": args.max_new_tokens
        },
        "total_evaluations": total,
        "correct_count": correct,
        "overall_accuracy_pct": round(overall_acc, 2),
        "gsm8k_accuracy_pct": round(gsm_acc, 2),
        "math500_accuracy_pct": round(math_acc, 2),
        "truncation_count": truncated,
        "truncation_rate_pct": round(truncated / total * 100, 2) if total else 0.0,
        "tokens": {
            "min": int(np.min(token_counts)),
            "median": float(np.median(token_counts)),
            "p90": float(np.percentile(token_counts, 90)),
            "max": int(np.max(token_counts))
        }
    }

    print("\n" + "="*80)
    print("CALIBRATED ARM 1 EVALUATION SUMMARY:")
    print(f"Overall Pass@1:     {overall_acc:.2f}% ({correct}/{total})")
    print(f"GSM8K Pass@1:       {gsm_acc:.2f}% ({sum(r['is_correct'] for r in gsm_results)}/{len(gsm_results)})")
    print(f"MATH-500 Pass@1:    {math_acc:.2f}% ({sum(r['is_correct'] for r in math_results)}/{len(math_results)})")
    print(f"Truncation Rate:    {summary['truncation_rate_pct']:.2f}% ({truncated}/{total})")
    print(f"Token Distribution: min={summary['tokens']['min']}, med={summary['tokens']['median']}, p90={summary['tokens']['p90']}, max={summary['tokens']['max']}")
    print("="*80)

    # Spot check 5 samples per Rule 11
    print("\n--- Raw Output Spot Check (Rule 11 Compliance) ---")
    sample_indices = random.sample(range(total), min(5, total))
    for idx in sample_indices:
        r = results[idx]
        print(f"\n[Problem: {r['id']} | Benchmark: {r['benchmark']} | Correct: {r['is_correct']} | Tokens: {r['ans_tokens']}]")
        print(f"Output:\n{r['content'][:300]}...")

    output_payload = {
        "summary": summary,
        "evaluations": results
    }
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nResults written to {args.output_file}")

if __name__ == "__main__":
    main()
