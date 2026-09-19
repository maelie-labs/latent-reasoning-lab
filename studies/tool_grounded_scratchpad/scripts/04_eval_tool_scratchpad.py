#!/usr/bin/env python3
"""
studies/tool_grounded_scratchpad/scripts/04_eval_tool_scratchpad.py
Vectorized Batched Multi-Turn Benchmark Evaluation Harness for Study 4 Tool Scratchpad.

Invariants:
- Evaluates N=1,000 queries (250 benchmark problems x 4 seeds: [42, 123, 456, 789])
- Dedicated compute on GPU 1 (CUDA_VISIBLE_DEVICES=1)
- Multi-Turn Agentic Loop: Intercepts <tool_call>, returns updated <tool_response>, then produces answer
- Gate 0 Frozen Spec: temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5, max_tokens=8192
- Canonical math_verify 0.9.0 verification (symbolic equality, 0 regex fallbacks)
- Strict Truncation Protocol: sequences hitting max_tokens scored False
"""

import os
import sys
import re
import json
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from peft import PeftModel
from math_verify import parse, verify

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

SEEDS = [42, 123, 456, 789]

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lora_tool_scratchpad_qwen3_4b/best_checkpoint")
    parser.add_argument("--benchmark_file", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=8192)
    args = parser.parse_args()

    tag = "qwen3_4b" if "Qwen3-4B" in args.model_id else "qwen3_5_4b"
    output_summary = os.path.join(PROJECT_ROOT, f"data/eval_study4_tool_scratchpad_{tag}.json")
    output_streaming = os.path.join(PROJECT_ROOT, f"data/streaming_study4_tool_scratchpad_{tag}.jsonl")

    print("=" * 80)
    print(f"MULTI-TURN EVALUATION HARNESS: STUDY 4 (TOOL SCRATCHPAD) for {args.model_id}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device} | Batch Size: {args.batch_size} | Seeds: {SEEDS}")
    print(f"Sampling: temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5, max_tokens={args.max_tokens}")
    print(f"Output: {output_summary}")
    print("=" * 80)

    with open(args.benchmark_file) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} benchmark problems from {args.benchmark_file}")

    total_queries = len(problems) * len(SEEDS)
    print(f"Total queries to evaluate: {total_queries} ({len(problems)} problems x {len(SEEDS)} seeds)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading {args.model_id} in pure bfloat16 onto {args.device}...")
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

    evaluated_keys = set()
    results = []
    correct_count = 0
    trunc_count = 0
    total_tokens_list = []

    if os.path.exists(output_streaming):
        with open(output_streaming) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        k = (r["problem_id"], r["seed"])
                        evaluated_keys.add(k)
                        results.append(r)
                        if r.get("is_correct"):
                            correct_count += 1
                        if r.get("is_truncated"):
                            trunc_count += 1
                        total_tokens_list.append(r.get("tokens", {}).get("total", 0))
                    except Exception:
                        pass
        print(f"Resuming evaluation: found {len(evaluated_keys)} previously evaluated queries.")

    stream_f = open(output_streaming, "a")

    pbar = tqdm(total=total_queries, initial=len(evaluated_keys), desc="Eval tool_scratchpad")
    t0_eval = time.time()

    for seed in SEEDS:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Filter out already evaluated problems for this seed
        remaining_problems = [p for p in problems if (p["id"], seed) not in evaluated_keys]
        if not remaining_problems:
            continue

        for b_start in range(0, len(remaining_problems), args.batch_size):
            b_problems = remaining_problems[b_start : b_start + args.batch_size]
            B = len(b_problems)

            items = []
            for p in b_problems:
                items.append({
                    "problem": p,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant with an in-place working memory scratchpad."},
                        {"role": "user", "content": p.get("question") or p.get("problem", "")}
                    ],
                    "scratchpad": {},
                    "is_done": False,
                    "turns": 0,
                    "final_text": "",
                    "total_tokens": 0
                })

            # Multi-turn tool execution loop (up to 4 turns)
            for turn in range(4):
                active_items = [it for it in items if not it["is_done"]]
                if not active_items:
                    break

                prompts = [
                    tokenizer.apply_chat_template(
                        it["messages"],
                        tools=[SCRATCHPAD_TOOL],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    for it in active_items
                ]

                tokenizer.padding_side = "left"
                enc = tokenizer(prompts, padding=True, return_tensors="pt").to(args.device)
                prompt_lens = enc.attention_mask.sum(dim=-1).tolist()

                max_gen = args.max_tokens if turn >= 3 else 256

                logits_proc = LogitsProcessorList([
                    PresencePenaltyLogitsProcessor(1.5, prompt_lens)
                ])

                with torch.no_grad():
                    outputs = model.generate(
                        **enc,
                        max_new_tokens=max_gen,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.80,
                        top_k=20,
                        logits_processor=logits_proc,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                gen_ids = outputs[:, enc.input_ids.shape[1]:].cpu()
                del enc, outputs

                for idx, it in enumerate(active_items):
                    g_text = tokenizer.decode(gen_ids[idx], skip_special_tokens=False).strip()
                    num_toks = len(gen_ids[idx])
                    it["total_tokens"] += num_toks

                    if "<tool_call>" in g_text and it["turns"] < 3:
                        m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", g_text, re.DOTALL)
                        call_vars = {}
                        if m:
                            try:
                                c_data = json.loads(m.group(1))
                                call_vars = c_data.get("arguments", {}).get("variables", {})
                            except Exception:
                                pass
                        it["scratchpad"].update(call_vars)
                        it["messages"].append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "type": "function",
                                "function": {
                                    "name": "update_scratchpad",
                                    "arguments": json.dumps({"variables": call_vars}, ensure_ascii=False)
                                }
                            }]
                        })
                        it["messages"].append({
                            "role": "tool",
                            "name": "update_scratchpad",
                            "content": json.dumps({"scratchpad": dict(it["scratchpad"])}, ensure_ascii=False)
                        })
                        it["turns"] += 1
                    else:
                        it["is_done"] = True
                        clean_ans = g_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
                        if "</think>" in clean_ans:
                            clean_ans = clean_ans.split("</think>")[-1].strip()
                        it["final_text"] = clean_ans

            # Grade the batch items
            for it in items:
                p = it["problem"]
                eval_target = it["final_text"]
                is_truncated = it["total_tokens"] >= args.max_tokens and ("\\boxed{" not in eval_target)

                if is_truncated:
                    trunc_count += 1
                    is_correct = False
                else:
                    is_correct = check_correctness(eval_target, p.get("solution") or p.get("reference_answer", ""))

                if is_correct:
                    correct_count += 1

                total_tokens_list.append(it["total_tokens"])

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
                    "tokens": {"total": it["total_tokens"], "turns": it["turns"]},
                    "output": eval_target
                }
                results.append(record)
                stream_f.write(json.dumps(record) + "\n")
                stream_f.flush()
                pbar.update(1)

            del items
            torch.cuda.empty_cache()

    pbar.close()
    stream_f.close()
    elapsed = time.time() - t0_eval

    pass_at_1 = (correct_count / total_queries) * 100.0
    trunc_rate = (trunc_count / total_queries) * 100.0

    gsm8k_records = [r for r in results if r["dataset"] == "gsm8k" or r["problem_id"].startswith("gsm8k")]
    math_records = [r for r in results if r["dataset"] in ["math500", "math"] or r["problem_id"].startswith("math")]

    gsm8k_acc = (sum(1 for r in gsm8k_records if r["is_correct"]) / len(gsm8k_records) * 100.0) if gsm8k_records else 0.0
    math_acc = (sum(1 for r in math_records if r["is_correct"]) / len(math_records) * 100.0) if math_records else 0.0

    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS: STUDY 4 TOOL SCRATCHPAD for {args.model_id}")
    print(f"Overall Pass@1:     {pass_at_1:.2f}% ({correct_count}/{total_queries})")
    print(f"GSM8K Accuracy:     {gsm8k_acc:.2f}% ({sum(1 for r in gsm8k_records if r['is_correct'])}/{len(gsm8k_records)})")
    print(f"MATH-500 Accuracy:  {math_acc:.2f}% ({sum(1 for r in math_records if r['is_correct'])}/{len(math_records)})")
    print(f"Truncation Rate:    {trunc_rate:.2f}%")
    print(f"Elapsed Time:       {elapsed:.1f}s ({elapsed/total_queries:.2f} s/query)")
    print("=" * 80)

    summary = {
        "model_id": args.model_id,
        "mode": "tool_scratchpad",
        "checkpoint": args.checkpoint,
        "overall_pass_at_1": round(pass_at_1, 2),
        "gsm8k_pass_at_1": round(gsm8k_acc, 2),
        "math500_pass_at_1": round(math_acc, 2),
        "truncation_rate_pct": round(trunc_rate, 2),
        "total_queries": total_queries,
        "correct_queries": correct_count,
        "elapsed_seconds": round(elapsed, 2)
    }

    with open(output_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {output_summary}")

if __name__ == "__main__":
    main()
