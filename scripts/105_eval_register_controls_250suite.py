#!/usr/bin/env python3
"""
scripts/105_eval_register_controls_250suite.py
Full Benchmark Evaluation for Structural Scaffolding Arms on Qwen/Qwen3-1.7B:
- Arm 1b-R: Trained Headers Only (3 headers, 0 latents)
- Arm 2b-R: Trained Pause Tokens (3 headers + 24 <pause> tokens)

Evaluates across the full 250-suite benchmark (data/benchmark_suite_250.json)
across 4 deterministic seeds: SEEDS = [42, 123, 456, 789] (N=1,000 total queries).

Enforces:
- Gate 0 Frozen Specification Table (temp=0.7, top_p=0.80, top_k=20, pp=1.5, ans_cap=8192)
- High-Throughput Batched PyTorch Execution (B=16) on dedicated compute GPU
- Grader Invariant: Pure canonical math_verify 0.9.0 with symbolic equivalence
- Paired Hierarchical Bootstrapping against Arm 1b reference and Arm 3-R
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from peft import PeftModel
from math_verify import parse, verify

SEEDS = [42, 123, 456, 789]
PAUSE_TOKEN = "<pause>"
CANONICAL_HEADERS = [
    "[R1: Identify givens, constraints, and target variable]\n",
    "[R2: Compute intermediate operations and verify relations]\n",
    "[R3: Execute final deduction and verify constraints]\n"
]

class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Additive presence penalty for batched token generation with variable prompt lengths."""
    def __init__(self, penalty: float, prompt_lens: list):
        self.penalty = penalty
        self.prompt_lens = prompt_lens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        for b in range(input_ids.shape[0]):
            p_len = self.prompt_lens[b]
            if input_ids.shape[1] > p_len:
                gen_ids = input_ids[b, p_len:]
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

def format_control_prompt(problem: dict, mode: str, k_per_bundle: int = 8) -> str:
    q = problem.get("question") or problem.get("problem", "")
    p = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    for r in range(3):
        p += CANONICAL_HEADERS[r]
        if mode == "headers_pause":
            p += (PAUSE_TOKEN * k_per_bundle)
    p += "\n</think>\n\n"
    return p

def run_paired_bootstrap(scores_a, scores_b, n_boot=10000, seed=42):
    np.random.seed(seed)
    n = len(scores_a)
    deltas = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        deltas.append(np.mean(scores_a[idx]) - np.mean(scores_b[idx]))
    deltas = np.array(deltas) * 100.0
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    obs_delta = float(np.mean(scores_a) - np.mean(scores_b)) * 100.0
    p_val = float(np.mean(deltas <= 0.0)) if obs_delta > 0 else float(np.mean(deltas >= 0.0))
    return {
        "observed_delta": obs_delta,
        "ci_95": [ci_low, ci_high],
        "p_value": p_val * 2.0
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Structural Scaffolding Controls across 250-suite x 4 seeds")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mode", type=str, required=True, choices=["headers_only", "headers_pause"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark_file", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_ans_tokens", type=int, default=8192)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_jsonl = os.path.join(args.output_dir, f"streaming_{args.tag}.jsonl")
    out_results = os.path.join(args.output_dir, f"eval_results_{args.tag}.json")
    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    print("=" * 80)
    print(f"EVALUATION: {args.tag.upper()} ({args.mode})")
    print(f"Model ID     : {args.model_id}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Device       : {args.device} | Batch Size: {args.batch_size}")
    print(f"Sampling     : Temp=0.7, TopP=0.80, TopK=20, PP=1.5, MaxAns={args.max_ans_tokens}")
    print(f"Streaming Log: {out_jsonl}")
    print(f"Final Results: {out_results}")
    print("=" * 80)

    with open(args.benchmark_file) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} benchmark problems.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.mode == "headers_pause" and PAUSE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    if args.mode == "headers_pause":
        base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    all_evaluations = []
    start_time = time.time()

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        torch.manual_seed(seed)

        for i in range(0, len(problems), args.batch_size):
            b_items = problems[i:i + args.batch_size]
            B = len(b_items)

            b_prompts = [format_control_prompt(p, args.mode) for p in b_items]
            enc = tokenizer(b_prompts, return_tensors="pt", padding=True).to(args.device)

            # Prompt length for each sample in batch is enc.input_ids.shape[1] due to left padding
            prompt_len = enc.input_ids.shape[1]
            pp_proc = PresencePenaltyLogitsProcessor(1.5, [prompt_len] * B)

            with torch.inference_mode():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=args.max_ans_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.80,
                    top_k=20,
                    logits_processor=LogitsProcessorList([pp_proc]),
                    pad_token_id=tokenizer.pad_token_id
                )

            for b_idx in range(B):
                gen_toks = out_ids[b_idx, prompt_len:].tolist()

                is_trunc = bool(len(gen_toks) >= args.max_ans_tokens and tokenizer.eos_token_id not in gen_toks)
                gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
                gold = b_items[b_idx].get("solution") or b_items[b_idx].get("answer", "")
                is_correct = False if is_trunc else check_math_correct(gen_text, gold)

                rec = {
                    "id": b_items[b_idx]["id"],
                    "benchmark": b_items[b_idx]["benchmark"],
                    "stratum": b_items[b_idx].get("stratum", ""),
                    "seed": seed,
                    "is_correct": is_correct,
                    "is_truncated": is_trunc,
                    "tokens": len(gen_toks),
                    "content": gen_text
                }
                all_evaluations.append(rec)
                with open(out_jsonl, "a") as f_out:
                    f_out.write(json.dumps(rec) + "\n")

            done_count = len(all_evaluations) % (len(problems))
            if done_count == 0:
                done_count = len(problems)
            running_acc = float(np.mean([r["is_correct"] for r in all_evaluations])) * 100.0
            if (i // args.batch_size + 1) % 4 == 0 or (i + args.batch_size >= len(problems)):
                print(f"[Seed {seed} | {done_count:3d}/{len(problems)}] Running Overall Acc: {running_acc:.2f}% ({sum(r['is_correct'] for r in all_evaluations)}/{len(all_evaluations)})")

    overall_acc = float(np.mean([r["is_correct"] for r in all_evaluations])) * 100.0
    gsm_acc = float(np.mean([r["is_correct"] for r in all_evaluations if r["benchmark"] == "GSM8K"])) * 100.0
    math_acc = float(np.mean([r["is_correct"] for r in all_evaluations if r["benchmark"] in ["MATH-500", "MATH500"]])) * 100.0
    math_hard = float(np.mean([
        r["is_correct"] for r in all_evaluations
        if r["benchmark"] in ["MATH-500", "MATH500"] and str(r.get("stratum", "")) in ["Level 3", "Level 4", "Level 5"]
    ])) * 100.0
    trunc_rate = float(np.mean([r["is_truncated"] for r in all_evaluations])) * 100.0

    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS: {args.tag.upper()} (N={len(all_evaluations)})")
    print("=" * 80)
    print(f"Overall Accuracy (Pass@1) : {overall_acc:5.2f}%")
    print(f"GSM8K Accuracy            : {gsm_acc:5.2f}%")
    print(f"MATH-500 Accuracy         : {math_acc:5.2f}%")
    print(f"MATH Hard (L3-5)          : {math_hard:5.2f}%")
    print(f"Truncation Rate           : {trunc_rate:5.2f}%")
    print(f"Elapsed Time              : {time.time()-start_time:.1f}s")
    print("=" * 80)

    # Paired Bootstrap against Arm 1b Baseline (streaming_arm1b_qwen_qwen3-1.7b.jsonl)
    arm1b_file = "data/streaming_arm1b_qwen_qwen3-1.7b.jsonl"
    bootstrap_arm1b = None
    if os.path.exists(arm1b_file):
        with open(arm1b_file) as f:
            arm1b_evals = [json.loads(line) for line in f]
        if len(arm1b_evals) == len(all_evaluations):
            scores_curr = np.array([int(r["is_correct"]) for r in all_evaluations])
            scores_1b = np.array([int(r["is_correct"]) for r in arm1b_evals])
            bootstrap_arm1b = run_paired_bootstrap(scores_curr, scores_1b)
            print(f"\n--- PAIRED BOOTSTRAP vs ARM 1b BASELINE ---")
            print(f"Observed Delta (Curr - Arm 1b) : {bootstrap_arm1b['observed_delta']:+5.2f}%")
            print(f"95% Confidence Interval       : [{bootstrap_arm1b['ci_95'][0]:+5.2f}%, {bootstrap_arm1b['ci_95'][1]:+5.2f}%]")
            print(f"p-value                       : {bootstrap_arm1b['p_value']:.4f}")

    # Paired Bootstrap against Arm 3-R Production (streaming_arm3_register_ladder_production_best.jsonl)
    arm3_file = "data/streaming_arm3_register_ladder_production_best.jsonl"
    bootstrap_arm3 = None
    if os.path.exists(arm3_file):
        with open(arm3_file) as f:
            arm3_evals = [json.loads(line) for line in f]
        if len(arm3_evals) == len(all_evaluations):
            scores_curr = np.array([int(r["is_correct"]) for r in all_evaluations])
            scores_3r = np.array([int(r["is_correct"]) for r in arm3_evals])
            bootstrap_arm3 = run_paired_bootstrap(scores_curr, scores_3r)
            print(f"\n--- PAIRED BOOTSTRAP vs ARM 3-R REGISTER LADDER ---")
            print(f"Observed Delta (Curr - Arm 3-R) : {bootstrap_arm3['observed_delta']:+5.2f}%")
            print(f"95% Confidence Interval        : [{bootstrap_arm3['ci_95'][0]:+5.2f}%, {bootstrap_arm3['ci_95'][1]:+5.2f}%]")
            print(f"p-value                        : {bootstrap_arm3['p_value']:.4f}")

    results_data = {
        "tag": args.tag,
        "mode": args.mode,
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
        "total_evaluations": len(all_evaluations),
        "overall_accuracy": overall_acc,
        "gsm8k_accuracy": gsm_acc,
        "math500_accuracy": math_acc,
        "math_hard_accuracy": math_hard,
        "truncation_rate": trunc_rate,
        "duration_seconds": time.time() - start_time,
        "paired_bootstrap_vs_arm1b": bootstrap_arm1b,
        "paired_bootstrap_vs_arm3": bootstrap_arm3
    }
    with open(out_results, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nEvaluation complete! Results saved to {out_results}")

if __name__ == "__main__":
    main()
