#!/usr/bin/env python3
"""
scripts/122_eval_study3_4b.py
Vectorized Batched Evaluation Harness for Study 3 on 4B Models.

Invariants:
- Evaluates N=1,000 queries (250 suite x 4 seeds: [42, 123, 456, 789])
- Model: Qwen/Qwen3-4B or Qwen/Qwen3.5-4B
- Batched PyTorch generation on dedicated GPU 1 (RTX PRO 4500 Blackwell 32GB)
- Grader: Canonical math_verify 0.9.0 (symbolic CAS verification)
- Token tracking: thinking tokens (<think>...</think>) vs answer tokens
- Strict truncation protocol: sequence hitting max tokens scored False
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from peft import PeftModel
from math_verify import parse, verify

SEEDS = [42, 123, 456, 789]

class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Additive presence penalty for token generation."""
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

def check_correctness(pred_text: str, solution: str) -> bool:
    if not pred_text or not solution:
        return False
    try:
        if "####" in solution:
            gold_ans = solution.split("####")[-1].strip()
            gold_target = f"\\boxed{{{gold_ans}}}"
        elif "\\boxed{" not in solution:
            gold_target = f"\\boxed{{{solution.strip()}}}"
        else:
            gold_target = solution

        gold_parsed = parse(gold_target, parsing_timeout=None)
        pred_parsed = parse(pred_text, parsing_timeout=None)
        if gold_parsed and pred_parsed:
            return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        pass
    return False

def split_thought_and_answer(text: str):
    think_match = ""
    ans_text = text
    if "<think>" in text:
        after_think = text.split("<think>", 1)[-1]
        if "</think>" in after_think:
            parts = after_think.split("</think>", 1)
            think_match = parts[0].strip()
            ans_text = parts[1].strip()
        else:
            think_match = after_think.strip()
            ans_text = ""
    elif "</think>" in text:
        parts = text.split("</think>", 1)
        think_match = parts[0].strip()
        ans_text = parts[1].strip()
    return think_match, ans_text

def run_evaluation(args):
    tag = "qwen3_4b" if "Qwen3-4B" in args.model_id else "qwen3_5_4b"
    output_summary = f"data/eval_study3_{args.mode}_{tag}.json"
    output_streaming = f"data/streaming_study3_{args.mode}_{tag}.jsonl"

    print("=" * 80)
    print(f"EVALUATION HARNESS: STUDY 3 ({args.mode.upper()}) for {args.model_id}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device} | Batch Size: {args.batch_size} | Seeds: {SEEDS}")
    print(f"Sampling: temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5, max_tokens={args.max_tokens}")
    print(f"Output: {output_summary}")
    print("=" * 80)

    with open(args.benchmark_file) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} benchmark problems from {args.benchmark_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Loading {args.model_id} in pure bfloat16 onto {args.device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    total_queries = len(problems) * len(SEEDS)
    print(f"Total queries to evaluate: {total_queries} (250 problems x 4 seeds)")

    results = []
    correct_count = 0
    trunc_count = 0
    think_tokens_list = []
    ans_tokens_list = []
    total_tokens_list = []
    t0_eval = time.time()

    os.makedirs(os.path.dirname(os.path.abspath(output_streaming)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_summary)), exist_ok=True)
    stream_f = open(output_streaming, "w")

    pbar = tqdm(total=total_queries, desc=f"Eval {args.mode}")

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for b_start in range(0, len(problems), args.batch_size):
            b_problems = problems[b_start : b_start + args.batch_size]
            b_prompts = [
                f"<|im_start|>user\n{p['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n<think>\n"
                for p in b_problems
            ]
            enc = tokenizer(b_prompts, return_tensors="pt", padding=True).to(args.device)
            prompt_lens = [enc.input_ids.shape[1]] * len(b_problems)
            logits_proc = LogitsProcessorList([PresencePenaltyLogitsProcessor(1.5, prompt_lens)])

            with torch.inference_mode():
                outputs = model.generate(
                    **enc,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.80,
                    top_k=20,
                    logits_processor=logits_proc,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            gen_ids = outputs[:, enc.input_ids.shape[1]:]

            for i, p in enumerate(b_problems):
                gen_text = tokenizer.decode(gen_ids[i], skip_special_tokens=False)
                clean_text = gen_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
                num_gen_tokens = len(gen_ids[i])

                is_truncated = (num_gen_tokens >= args.max_tokens) and ("<|im_end|>" not in gen_text)
                thought, ans = split_thought_and_answer(clean_text)
                eval_target = ans if ans else clean_text

                if is_truncated:
                    trunc_count += 1
                    is_correct = False
                else:
                    is_correct = check_correctness(eval_target, p.get("solution") or p.get("reference_answer", ""))

                if is_correct:
                    correct_count += 1

                th_toks = len(tokenizer.encode(thought, add_special_tokens=False)) if thought else 0
                ans_toks = len(tokenizer.encode(ans, add_special_tokens=False)) if ans else 0

                think_tokens_list.append(th_toks)
                ans_tokens_list.append(ans_toks)
                total_tokens_list.append(num_gen_tokens)

                ds_name = "gsm8k" if (p.get("benchmark") == "GSM8K" or p["id"].startswith("gsm8k")) else "math500"
                prob_level = p.get("level", 1 if ds_name == "gsm8k" else 0)
                record = {
                    "problem_id": p["id"],
                    "dataset": ds_name,
                    "level": prob_level,
                    "subject": p.get("subject", "unknown"),
                    "seed": seed,
                    "is_correct": is_correct,
                    "is_truncated": is_truncated,
                    "tokens": {
                        "thought": th_toks,
                        "answer": ans_toks,
                        "total": num_gen_tokens
                    },
                    "output": clean_text
                }
                results.append(record)
                stream_f.write(json.dumps(record) + "\n")
                stream_f.flush()
                pbar.update(1)

    pbar.close()
    stream_f.close()
    elapsed = time.time() - t0_eval

    pass_at_1 = (correct_count / total_queries) * 100.0
    trunc_rate = (trunc_count / total_queries) * 100.0

    # Sub-dataset breakdowns
    gsm8k_records = [r for r in results if r["dataset"] == "gsm8k" or r["problem_id"].startswith("gsm8k")]
    math_records = [r for r in results if r["dataset"] in ["math500", "math"] or r["problem_id"].startswith("math")]
    math_hard_records = [r for r in math_records if r["level"] >= 3]

    gsm8k_acc = (sum(1 for r in gsm8k_records if r["is_correct"]) / len(gsm8k_records) * 100.0) if gsm8k_records else 0.0
    math_acc = (sum(1 for r in math_records if r["is_correct"]) / len(math_records) * 100.0) if math_records else 0.0
    math_hard_acc = (sum(1 for r in math_hard_records if r["is_correct"]) / len(math_hard_records) * 100.0) if math_hard_records else 0.0

    summary = {
        "model_id": args.model_id,
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "total_queries": total_queries,
        "correct_count": correct_count,
        "pass_at_1": round(pass_at_1, 2),
        "truncation_rate": round(trunc_rate, 2),
        "gsm8k_accuracy": round(gsm8k_acc, 2),
        "math_accuracy": round(math_acc, 2),
        "math_hard_accuracy": round(math_hard_acc, 2),
        "median_thought_tokens": float(np.median(think_tokens_list)),
        "median_answer_tokens": float(np.median(ans_tokens_list)),
        "elapsed_seconds": round(elapsed, 1),
        "queries_per_second": round(total_queries / elapsed, 2)
    }

    with open(output_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"STUDY 3 RESULTS: {args.mode.upper()} for {args.model_id}")
    print(f"Overall Pass@1:     {pass_at_1:.2f}% ({correct_count}/{total_queries})")
    print(f"GSM8K Pass@1:       {gsm8k_acc:.2f}%")
    print(f"MATH-500 Pass@1:    {math_acc:.2f}%")
    print(f"MATH Hard (L3-5):   {math_hard_acc:.2f}%")
    print(f"Truncation Rate:    {trunc_rate:.2f}%")
    print(f"Median Thought:     {summary['median_thought_tokens']:.1f} tokens")
    print(f"Median Answer:      {summary['median_answer_tokens']:.1f} tokens")
    print(f"Elapsed Time:       {elapsed:.1f}s ({summary['queries_per_second']} q/s)")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Study 3 Adapters for 4B Models")
    parser.add_argument("--model_id", type=str, required=True, choices=["Qwen/Qwen3-4B", "Qwen/Qwen3.5-4B"])
    parser.add_argument("--mode", type=str, required=True, choices=["telegraphic_cot", "headers_only"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--benchmark_file", type=str, default="data/benchmark_suite_250.json")
    args = parser.parse_args()

    run_evaluation(args)

if __name__ == "__main__":
    main()
