#!/usr/bin/env python3
"""
studies/dynamic_registers/scripts/03_eval_register_models.py
Evaluate Staged Dynamic Registers on the official 250 Benchmark Suite across 4 deterministic seeds.

Protocol Invariants:
- Suite: data/benchmark_suite_250.json (150 GSM8K + 100 MATH-500 L1-5) x 4 seeds = 1,000 evaluations.
- Sampler: Gate 0 Frozen (temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5, max_new_tokens=8192).
- Delimiters: Prompt ends with <think>\n; model emits registers + </think>\n\n + worked answer + <|im_end|>.
- Grader: Canonical math_verify 0.9.0 with pure symbolic equivalence.
- Truncation rule: Any sequence hitting generation ceiling without </think> or <|im_end|> scored as False.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from peft import PeftModel
from tqdm import tqdm
from math_verify import parse, verify

SEEDS = [42, 123, 456, 789]
MODEL_ID = "Qwen/Qwen3-1.7B"

class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Additive presence penalty for token generation."""
    def __init__(self, penalty: float, prompt_lens: List[int]):
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

def extract_math_boxed_expression(text: str) -> str:
    """Extract expression inside \\boxed{...} with brace balancing."""
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

def check_correctness(prediction_text: str, ground_truth_sol: str) -> bool:
    try:
        if "</think>" in prediction_text:
            ans_part = prediction_text.split("</think>")[-1].strip()
        else:
            ans_part = prediction_text.strip()

        if "####" in ground_truth_sol:
            gt_boxed = ground_truth_sol.split("####")[-1].strip()
        else:
            gt_boxed = extract_math_boxed_expression(ground_truth_sol)

        gt_target = f"\\boxed{{{gt_boxed}}}" if gt_boxed else f"\\boxed{{{ground_truth_sol}}}"
        cand = parse(ans_part, parsing_timeout=None)
        ref = parse(gt_target, parsing_timeout=None)
        if cand is not None and ref is not None:
            return bool(verify(cand, ref))
    except Exception:
        return False
    return False

def make_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if "<think>\n\n</think>\n\n" in p:
        p = p.replace("<think>\n\n</think>\n\n", "<think>\n")
    if not p.endswith("<think>\n"):
        if "<think>" in p:
            p = p.split("<think>")[0] + "<think>\n"
        else:
            p += "<think>\n"
    return p

def evaluate(args):
    print("=" * 80)
    print("EVALUATING STAGED DYNAMIC REGISTERS: 250 SUITE (N=1,000)")
    print(f"Tag: {args.tag} | Checkpoint: {args.checkpoint} | Device: {args.device}")
    print("=" * 80)

    with open(args.benchmark_file) as f:
        bench_problems = json.load(f)

    print(f"Loaded {len(bench_problems)} benchmark problems across {len(SEEDS)} seeds.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model {args.base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    print(f"Loading LoRA checkpoint from {args.checkpoint}...")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    endoftext_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
    eos_token_ids = [im_end_id, endoftext_id]

    os.makedirs(os.path.dirname(os.path.abspath(args.streaming_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    streaming_f = open(args.streaming_file, "w")
    all_eval_items = []
    seed_runs = []
    start_time = time.time()

    total_evals = len(bench_problems) * len(SEEDS)
    pbar = tqdm(total=total_evals, desc=f"Eval {args.tag}")

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        seed_results = []
        batch_size = args.batch_size

        for i in range(0, len(bench_problems), batch_size):
            batch = bench_problems[i : i + batch_size]
            prompts = [make_prompt(tokenizer, p["question"]) for p in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
            prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]

            logits_proc = LogitsProcessorList([PresencePenaltyLogitsProcessor(1.5, prompt_lens)])

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.80,
                    top_k=20,
                    logits_processor=logits_proc,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids
                )

            p_len = enc.input_ids.shape[1]
            for j, p in enumerate(batch):
                raw_gen = out[j][p_len:].tolist()

                # Find first EOS token if emitted
                first_eos = len(raw_gen)
                found_eos = False
                for idx, tok in enumerate(raw_gen):
                    if tok in eos_token_ids:
                        first_eos = idx
                        found_eos = True
                        break

                # Strip trailing padding tokens
                gen_toks = raw_gen[:first_eos]
                tok_len = len(gen_toks)
                is_truncated = (tok_len >= args.max_tokens) and not found_eos
                text = tokenizer.decode(gen_toks, skip_special_tokens=False)

                if is_truncated:
                    is_correct = False
                else:
                    is_correct = check_correctness(text, p["solution"])

                record = {
                    "id": p["id"],
                    "benchmark": p["benchmark"],
                    "stratum": p["stratum"],
                    "level": p.get("level", None),
                    "seed": seed,
                    "is_correct": is_correct,
                    "is_truncated": is_truncated,
                    "tokens": tok_len,
                    "content": text,
                    "pred_boxed": extract_math_boxed_expression(text)
                }
                streaming_f.write(json.dumps(record) + "\n")
                streaming_f.flush()
                seed_results.append(record)
                all_eval_items.append(record)

            pbar.update(len(batch))

        seed_corr = sum(1 for r in seed_results if r["is_correct"])
        seed_acc = (seed_corr / len(seed_results)) * 100
        print(f"\n[Seed {seed}] Accuracy: {seed_acc:.2f}% ({seed_corr}/{len(seed_results)})")
        seed_runs.append({"seed": seed, "accuracy": seed_acc})

    streaming_f.close()
    pbar.close()

    elapsed = time.time() - start_time
    total_corr = sum(1 for r in all_eval_items if r["is_correct"])
    mean_acc = (total_corr / total_evals) * 100

    gsm_items = [r for r in all_eval_items if r["benchmark"] == "GSM8K"]
    math_items = [r for r in all_eval_items if r["benchmark"] == "MATH-500"]
    math_hard = [r for r in math_items if r.get("level") in [3, 4, 5] or r.get("stratum") in ["Level 3", "Level 4", "Level 5"]]

    gsm_pass = (sum(1 for r in gsm_items if r["is_correct"]) / len(gsm_items)) * 100 if gsm_items else 0.0
    math_pass = (sum(1 for r in math_items if r["is_correct"]) / len(math_items)) * 100 if math_items else 0.0
    math_hard_pass = (sum(1 for r in math_hard if r["is_correct"]) / len(math_hard)) * 100 if math_hard else 0.0

    trunc_count = sum(1 for r in all_eval_items if r["is_truncated"])
    trunc_rate = (trunc_count / total_evals) * 100

    tok_lens = [r["tokens"] for r in all_eval_items]
    tok_dist = {
        "min": int(np.min(tok_lens)) if tok_lens else 0,
        "median": float(np.median(tok_lens)) if tok_lens else 0.0,
        "mean": round(float(np.mean(tok_lens)), 2) if tok_lens else 0.0,
        "p90": round(float(np.percentile(tok_lens, 90)), 2) if tok_lens else 0.0,
        "p95": round(float(np.percentile(tok_lens, 95)), 2) if tok_lens else 0.0,
        "max": int(np.max(tok_lens)) if tok_lens else 0
    }

    seed_breakdown = {str(s["seed"]): round(s["accuracy"], 2) for s in seed_runs}

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE ({elapsed:.1f}s)")
    print(f"Tag:            {args.tag}")
    print(f"Overall Pass@1: {mean_acc:.2f}% ({total_corr}/{total_evals})")
    print(f"GSM8K:          {gsm_pass:.2f}% ({sum(1 for r in gsm_items if r['is_correct'])}/{len(gsm_items)})")
    print(f"MATH-500:       {math_pass:.2f}% ({sum(1 for r in math_items if r['is_correct'])}/{len(math_items)})")
    print(f"MATH Hard L3-5: {math_hard_pass:.2f}% ({sum(1 for r in math_hard if r['is_correct'])}/{len(math_hard)})")
    print(f"Truncation:     {trunc_rate:.2f}% ({trunc_count}/{total_evals})")
    print(f"Token Lengths:  median={tok_dist['median']:.1f}, p90={tok_dist['p90']:.1f}, max={tok_dist['max']}")
    print("=" * 80)

    summary = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "model_id": args.base_model_id,
        "total_queries": total_evals,
        "pass_at_1": round(mean_acc, 2),
        "gsm8k_pass_at_1": round(gsm_pass, 2),
        "math500_pass_at_1": round(math_pass, 2),
        "math500_l3_5_pass_at_1": round(math_hard_pass, 2),
        "truncation_rate": round(trunc_rate, 2),
        "truncation_count": trunc_count,
        "token_distribution": tok_dist,
        "seeds": SEEDS,
        "seed_breakdown": seed_breakdown,
        "duration_seconds": round(elapsed, 1),
        "grader": "math_verify 0.9.0 (pure canonical)",
        "sampling_config": {
            "temperature": 0.7,
            "top_p": 0.80,
            "top_k": 20,
            "presence_penalty": 1.5,
            "max_new_tokens": args.max_tokens
        }
    }
    with open(args.output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--base_model_id", type=str, default=MODEL_ID)
    parser.add_argument("--benchmark_file", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--streaming_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=8192)
    args = parser.parse_args()
    evaluate(args)
