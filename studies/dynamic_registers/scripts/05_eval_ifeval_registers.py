#!/usr/bin/env python3
"""
studies/dynamic_registers/scripts/05_eval_ifeval_registers.py
Evaluates IFEval (541 prompts) on SGLang serving instance for Arm 0 Surgical Neutrality.
Computes strict and loose prompt-level and instruction-level accuracies.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import httpx
from tqdm.asyncio import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from ifeval.utils import (
    InputExample,
    test_instruction_following_strict,
    test_instruction_following_loose,
)

async def eval_ifeval_prompt(client, server_url, sem, doc, tokenizer, max_tokens=1280):
    p_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": doc["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    if "<think>\n\n</think>\n\n" in p_text:
        pass
    elif "<think>\n" in p_text:
        p_text = p_text.replace("<think>\n", "<think>\n\n</think>\n\n")

    payload = {
        "text": p_text,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.80,
            "top_k": 20,
            "presence_penalty": 1.5,
            "sampling_seed": 42,
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }
    }

    async with sem:
        t0 = time.time()
        try:
            resp = await client.post(f"{server_url}/generate", json=payload, timeout=120.0)
            dur = time.time() - t0
            if resp.status_code != 200:
                return doc, "", dur, True
            data = resp.json()
            content = data.get("text", "").strip()
            meta = data.get("meta_info", {})
            finish_reason = meta.get("finish_reason", {})
            is_trunc = (finish_reason == "length") if isinstance(finish_reason, str) else (finish_reason.get("type") == "length")
            return doc, content, dur, is_trunc
        except Exception as e:
            dur = time.time() - t0
            return doc, "", dur, True

async def run_ifeval(args):
    print("=" * 80)
    print(f"ARM 0 IFEVAL EVALUATION: {args.tag}")
    print(f"Server: {args.server_url} | Model Path: {args.model_path}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = load_dataset("google/IFEval", split="train")
    print(f"Loaded {len(dataset)} IFEval prompts.")

    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_keepalive_connections=args.concurrency, max_connections=args.concurrency * 2)
    timeout = httpx.Timeout(120.0, connect=30.0)

    start_time = time.time()
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tasks = [eval_ifeval_prompt(client, args.server_url, sem, doc, tokenizer) for doc in dataset]
        responses = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"IFEval {args.tag}"):
            res = await f
            responses.append(res)

    strict_prompt_correct = 0
    loose_prompt_correct = 0
    strict_inst_correct = 0
    loose_inst_correct = 0
    total_instructions = 0
    total_prompts = len(dataset)
    trunc_count = 0
    records = []

    for doc, content, dur, is_trunc in responses:
        if is_trunc:
            trunc_count += 1

        inp = InputExample(
            key=doc["key"],
            instruction_id_list=doc["instruction_id_list"],
            prompt=doc["prompt"],
            kwargs=doc["kwargs"]
        )

        out_strict = test_instruction_following_strict(inp, content)
        out_loose = test_instruction_following_loose(inp, content)

        n_inst = len(inp.instruction_id_list)
        total_instructions += n_inst
        if out_strict.follow_all_instructions:
            strict_prompt_correct += 1
        if out_loose.follow_all_instructions:
            loose_prompt_correct += 1
        strict_inst_correct += sum(out_strict.follow_instruction_list)
        loose_inst_correct += sum(out_loose.follow_instruction_list)

        records.append({
            "key": inp.key,
            "prompt": inp.prompt,
            "response": content,
            "strict_all": out_strict.follow_all_instructions,
            "loose_all": out_loose.follow_all_instructions,
            "strict_list": out_strict.follow_instruction_list,
            "loose_list": out_loose.follow_instruction_list
        })

    elapsed = time.time() - start_time
    strict_p_acc = (strict_prompt_correct / total_prompts) * 100
    loose_p_acc = (loose_prompt_correct / total_prompts) * 100
    strict_i_acc = (strict_inst_correct / total_instructions) * 100
    loose_i_acc = (loose_inst_correct / total_instructions) * 100

    print("\n" + "=" * 80)
    print(f"IFEVAL RESULTS: {args.tag}")
    print(f"Prompt Strict Accuracy:      {strict_p_acc:.2f}% ({strict_prompt_correct}/{total_prompts})")
    print(f"Prompt Loose Accuracy:       {loose_p_acc:.2f}% ({loose_prompt_correct}/{total_prompts})")
    print(f"Instruction Strict Accuracy: {strict_i_acc:.2f}% ({strict_inst_correct}/{total_instructions})")
    print(f"Instruction Loose Accuracy:  {loose_i_acc:.2f}% ({loose_inst_correct}/{total_instructions})")
    print(f"Truncation Rate:             {trunc_count/total_prompts*100:.2f}% ({trunc_count}/{total_prompts})")
    print(f"Time Elapsed:                {elapsed:.2f}s")
    print("=" * 80)

    summary = {
        "tag": args.tag,
        "elapsed_s": elapsed,
        "total_prompts": total_prompts,
        "total_instructions": total_instructions,
        "strict_prompt_accuracy_pct": round(strict_p_acc, 2),
        "loose_prompt_accuracy_pct": round(loose_p_acc, 2),
        "strict_inst_accuracy_pct": round(strict_i_acc, 2),
        "loose_inst_accuracy_pct": round(loose_i_acc, 2),
        "truncation_count": trunc_count,
        "truncation_rate_pct": round(trunc_count / total_prompts * 100, 2)
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump({"summary": summary, "evaluations": records}, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()

    asyncio.run(run_ifeval(args))

if __name__ == "__main__":
    main()
