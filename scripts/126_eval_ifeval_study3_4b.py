#!/usr/bin/env python3
"""
scripts/126_eval_ifeval_study3_4b.py
Batched PyTorch IFEval Evaluator for 4B Models and LoRA Checkpoints.

Evaluates base models and trained checkpoints on all 541 google/IFEval prompts
to verify Arm 0 Surgical Neutrality (confirming no degradation of instruction following).
Runs batched inference in pure bfloat16 on GPU 1 (CUDA_VISIBLE_DEVICES=1).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from peft import PeftModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ifeval.utils import (
    InputExample,
    test_instruction_following_strict,
    test_instruction_following_loose,
)

class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Additive presence penalty matching SGLang presence_penalty."""
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Base model ID or path")
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--tag", type=str, required=True, help="Tag for this evaluation")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_tokens", type=int, default=1280)
    args = parser.parse_args()

    print("=" * 80)
    print(f"ARM 0 IFEVAL EVALUATOR: {args.tag}")
    print(f"Base Model: {args.model_id}")
    print(f"Checkpoint: {args.checkpoint or 'None (Base Model)'}")
    print(f"Device: {args.device} | Batch Size: {args.batch_size} | Max New Tokens: {args.max_tokens}")
    print("=" * 80)

    # 1. Load Dataset
    dataset = load_dataset("google/IFEval", split="train")
    total_prompts = len(dataset)
    print(f"Loaded {total_prompts} IFEval prompts.")

    # 2. Load Tokenizer & Model
    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base model {args.model_id} in pure bfloat16 onto {args.device}...")
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

    # 3. Batched Evaluation Loop
    strict_prompt_correct = 0
    loose_prompt_correct = 0
    strict_inst_correct = 0
    loose_inst_correct = 0
    total_instructions = 0
    trunc_count = 0
    records = []

    pbar = tqdm(total=total_prompts, desc=f"IFEval {args.tag}")
    t0_start = time.time()

    for i in range(0, total_prompts, args.batch_size):
        batch = [dataset[j] for j in range(i, min(i + args.batch_size, total_prompts))]
        B = len(batch)

        prompts = []
        input_examples = []
        for doc in batch:
            inp = InputExample(
                key=doc["key"],
                instruction_id_list=doc["instruction_id_list"],
                prompt=doc["prompt"],
                kwargs=doc["kwargs"]
            )
            input_examples.append(inp)

            # Apply chat template with non-thinking / direct answering
            p_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": doc["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            # Ensure model immediately answers without verbose thinking loop
            if "<think>\n\n</think>\n\n" in p_text:
                pass
            elif "<think>\n" in p_text:
                p_text = p_text.replace("<think>\n", "<think>\n\n</think>\n\n")

            prompts.append(p_text)

        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, padding=True, return_tensors="pt").to(args.device)
        prompt_lens = enc.attention_mask.sum(dim=-1).tolist()

        logits_proc = LogitsProcessorList([
            PresencePenaltyLogitsProcessor(1.5, prompt_lens)
        ])

        with torch.no_grad():
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

        for b in range(B):
            inp = input_examples[b]
            response_text = tokenizer.decode(gen_ids[b], skip_special_tokens=False)
            is_trunc = (len(gen_ids[b]) >= args.max_tokens) and ("<|im_end|>" not in response_text)

            clean_response = response_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            if "</think>" in clean_response:
                clean_response = clean_response.split("</think>")[-1].strip()

            if is_trunc:
                trunc_count += 1

            out_strict = test_instruction_following_strict(inp, clean_response)
            out_loose = test_instruction_following_loose(inp, clean_response)

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
                "response": clean_response,
                "strict_all": out_strict.follow_all_instructions,
                "loose_all": out_loose.follow_all_instructions,
                "strict_list": out_strict.follow_instruction_list,
                "loose_list": out_loose.follow_instruction_list,
                "is_truncated": is_trunc
            })
            pbar.update(1)

    pbar.close()
    elapsed = time.time() - t0_start

    strict_p_acc = (strict_prompt_correct / total_prompts) * 100.0
    loose_p_acc = (loose_prompt_correct / total_prompts) * 100.0
    strict_i_acc = (strict_inst_correct / total_instructions) * 100.0
    loose_i_acc = (loose_inst_correct / total_instructions) * 100.0
    trunc_rate = (trunc_count / total_prompts) * 100.0

    print("\n" + "=" * 80)
    print(f"ARM 0 IFEVAL SUMMARY: {args.tag}")
    print(f"Prompt Strict Accuracy:      {strict_p_acc:.2f}% ({strict_prompt_correct}/{total_prompts})")
    print(f"Prompt Loose Accuracy:       {loose_p_acc:.2f}% ({loose_prompt_correct}/{total_prompts})")
    print(f"Instruction Strict Accuracy: {strict_i_acc:.2f}% ({strict_inst_correct}/{total_instructions})")
    print(f"Instruction Loose Accuracy:  {loose_i_acc:.2f}% ({loose_inst_correct}/{total_instructions})")
    print(f"Truncation Rate:             {trunc_rate:.2f}% ({trunc_count}/{total_prompts})")
    print(f"Elapsed Time:                {elapsed:.2f}s ({elapsed/total_prompts:.3f} s/query)")
    print("=" * 80)

    summary = {
        "tag": args.tag,
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
        "elapsed_s": round(elapsed, 2),
        "total_prompts": total_prompts,
        "total_instructions": total_instructions,
        "strict_prompt_accuracy_pct": round(strict_p_acc, 2),
        "loose_prompt_accuracy_pct": round(loose_p_acc, 2),
        "strict_inst_accuracy_pct": round(strict_i_acc, 2),
        "loose_inst_accuracy_pct": round(loose_i_acc, 2),
        "truncation_count": trunc_count,
        "truncation_rate_pct": round(trunc_rate, 2)
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump({"summary": summary, "evaluations": records}, f, indent=2)
    print(f"Saved evaluation results to {args.output_file}")

if __name__ == "__main__":
    main()
