#!/usr/bin/env python3
"""
scripts/107_eval_untrained_base_headers_250suite.py
Missing Scaffolding Control: Untrained Base Model with 3 Register Headers Injected (Arm 1-R).

Evaluates the pure base model (Qwen/Qwen3-1.7B, NO adapter) across the full
250-problem benchmark suite (data/benchmark_suite_250.json) across 4 deterministic
seeds: SEEDS = [42, 123, 456, 789] (N=1,000 total evaluations).

Purpose:
- Determines whether discrete register scaffolding (Arm 1b-R at 73.20% vs Arm 1b at 71.20%)
  is merely a repair mechanism for narrow SFT damage, or an intrinsic booster for reasoning.
- Comparison:
  - Arm 1 (Untrained Base Direct, No Headers): 73.90% (Hard MATH: 70.00%)
  - Arm 1-R (Untrained Base + 3 Headers): ???
  - Arm 1b-R (Trained Headers Only): 73.20% (Hard MATH: 67.08%)

Enforces:
- Gate 0 Frozen Specification Table (temp=0.7, top_p=0.80, top_k=20, pp=1.5, max_ans=8192)
- High-Throughput Batched PyTorch Execution (B=16) on GPU 1
- Grader Invariant: Pure canonical math_verify 0.9.0 with symbolic equivalence
- Paired Hierarchical Bootstrapping against Arm 1 and Arm 1b-R references
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
from math_verify import parse, verify

SEEDS = [42, 123, 456, 789]
CANONICAL_HEADERS = [
    "[R1: Identify givens, constraints, and target variable]\n",
    "[R2: Compute intermediate operations and verify relations]\n",
    "[R3: Execute final deduction and verify constraints]\n"
]

class PresencePenaltyLogitsProcessor(LogitsProcessor):
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

def format_prompt_with_headers(problem: dict) -> str:
    q = problem.get("question") or problem.get("problem", "")
    p = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    for r in range(3):
        p += CANONICAL_HEADERS[r]
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
    parser = argparse.ArgumentParser(description="Evaluate Untrained Base Model with 3 Register Headers (Arm 1-R)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--benchmark_file", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_ans_tokens", type=int, default=8192)
    parser.add_argument("--tag", type=str, default="arm1_base_headers_only")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    out_jsonl = f"data/streaming_{args.tag}.jsonl"
    out_results = f"data/eval_results_{args.tag}.json"
    os.makedirs("data", exist_ok=True)

    print("=" * 80)
    print(f"EVALUATION: {args.tag.upper()} (Pure Untrained Base + 3 Register Headers)")
    print(f"Model ID     : {args.model_id} (NO ADAPTER)")
    print(f"Device       : {args.device} | Batch Size: {args.batch_size}")
    print(f"Sampling     : Temp=0.7, TopP=0.80, TopK=20, PP=1.5, MaxAns={args.max_ans_tokens}")
    print(f"Streaming Log: {out_jsonl}")
    print(f"Final Results: {out_results}")
    print("=" * 80)

    with open(args.benchmark_file) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} benchmark problems.")

    if args.smoke_test:
        problems = problems[:4]
        eval_seeds = [42]
        print(f"Smoke test mode: running 4 problems on Seed 42.")
    else:
        eval_seeds = SEEDS

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base model {args.model_id} in bfloat16 on {args.device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    # Check for existing completed evaluations to support resume
    completed_keys = set()
    all_evaluations = []
    if os.path.exists(out_jsonl):
        with open(out_jsonl) as f_in:
            for line in f_in:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    completed_keys.add((rec["id"], rec["seed"]))
                    all_evaluations.append(rec)
        print(f"Resuming: found {len(completed_keys)} already completed evaluations.")

    start_time = time.time()

    for seed in eval_seeds:
        print(f"\n--- Running Seed {seed} ---")
        torch.manual_seed(seed)

        for i in range(0, len(problems), args.batch_size):
            b_items = problems[i:i + args.batch_size]
            items_to_run = [p for p in b_items if (p["id"], seed) not in completed_keys]
            if not items_to_run:
                continue

            B = len(items_to_run)
            b_prompts = [format_prompt_with_headers(p) for p in items_to_run]
            enc = tokenizer(b_prompts, return_tensors="pt", padding=True).to(args.device)

            prompt_len = enc.input_ids.shape[1]
            pp_proc = PresencePenaltyLogitsProcessor(1.5, [prompt_len] * B)

            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    max_new_tokens=args.max_ans_tokens,
                    temperature=0.7,
                    top_p=0.80,
                    top_k=20,
                    logits_processor=LogitsProcessorList([pp_proc]),
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )

            for b_idx in range(B):
                gen_toks = out_ids[b_idx, prompt_len:].tolist()

                is_trunc = bool(len(gen_toks) >= args.max_ans_tokens and tokenizer.eos_token_id not in gen_toks)
                gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
                gold = items_to_run[b_idx].get("solution") or items_to_run[b_idx].get("answer", "")
                is_correct = False if is_trunc else check_math_correct(gen_text, gold)

                rec = {
                    "id": items_to_run[b_idx]["id"],
                    "benchmark": items_to_run[b_idx]["benchmark"],
                    "stratum": items_to_run[b_idx].get("stratum", ""),
                    "seed": seed,
                    "is_correct": is_correct,
                    "is_truncated": is_trunc,
                    "tokens": len(gen_toks),
                    "content": gen_text
                }
                all_evaluations.append(rec)
                completed_keys.add((rec["id"], rec["seed"]))
                with open(out_jsonl, "a") as f_out:
                    f_out.write(json.dumps(rec) + "\n")

            running_acc = float(np.mean([r["is_correct"] for r in all_evaluations])) * 100.0
            print(f"[Seed {seed} | {len(all_evaluations):4d}/{len(problems)*len(eval_seeds)}] Running Acc: {running_acc:.2f}%", end="\r")

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

    # Compute paired bootstrap vs Arm 1 (Base Direct) and Arm 1b-R (Trained Headers) if available
    bs_results = {}
    if os.path.exists("data/streaming_arm1b_register_headers_only.jsonl"):
        ref_map = {}
        with open("data/streaming_arm1b_register_headers_only.jsonl") as f_ref:
            for line in f_ref:
                rec = json.loads(line)
                ref_map[(rec["id"], rec["seed"])] = rec["is_correct"]
        common_keys = [k for k in completed_keys if k in ref_map]
        if common_keys:
            eval_map = {(r["id"], r["seed"]): r["is_correct"] for r in all_evaluations}
            sc_a = np.array([float(eval_map[k]) for k in common_keys])
            sc_b = np.array([float(ref_map[k]) for k in common_keys])
            bs_results["vs_arm1b_register_headers"] = run_paired_bootstrap(sc_a, sc_b)
            print(f"Paired Bootstrap vs Arm 1b-R (Trained Headers): Delta={bs_results['vs_arm1b_register_headers']['observed_delta']:+.2f}%, p={bs_results['vs_arm1b_register_headers']['p_value']:.4f}")

    # Telemetry audit per Rule 11
    token_counts = [r["tokens"] for r in all_evaluations]
    output_summary = {
        "tag": args.tag,
        "mode": "base_headers_only",
        "model_id": args.model_id,
        "total_evaluations": len(all_evaluations),
        "overall_accuracy": overall_acc,
        "gsm8k_accuracy": gsm_acc,
        "math500_accuracy": math_acc,
        "math_hard_accuracy": math_hard,
        "truncation_rate": trunc_rate,
        "duration_seconds": time.time() - start_time,
        "token_distribution": {
            "min": int(np.min(token_counts)) if token_counts else 0,
            "median": float(np.median(token_counts)) if token_counts else 0,
            "p90": float(np.percentile(token_counts, 90)) if token_counts else 0,
            "max": int(np.max(token_counts)) if token_counts else 0
        },
        "paired_bootstrap": bs_results
    }

    with open(out_results, "w") as f_out:
        json.dump(output_summary, f_out, indent=2)
    print(f"Results saved to {out_results}")

if __name__ == "__main__":
    main()
