#!/usr/bin/env python3
"""
scripts/73_eval_ifeval_arms.py
Arm 0 Instruction-Following Neutrality Evaluator on google/IFEval.

Evaluates Base Model and Trained Adapters (Arm 1b, Arm 2b, Arm 3) on the full
541-prompt google/IFEval benchmark to verify zero degradation of general
instruction-following, formatting, and constraint compliance.

Computes:
1. Strict Prompt-level Accuracy (%)
2. Loose Prompt-level Accuracy (%)
3. Strict Instruction-level Accuracy (%)
4. Loose Instruction-level Accuracy (%)
"""

import os
import sys
import re
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from peft import PeftModel

from ifeval.utils import (
    InputExample,
    test_instruction_following_strict,
    test_instruction_following_loose,
)

PAUSE_TOKEN = "<pause>"

def get_calibrated_alpha(model_id: str) -> float:
    mid = model_id.lower()
    if "1.7b" in mid or "1.5b" in mid:
        return 0.011440
    elif "2b" in mid:
        return 0.009800
    elif "4b" in mid:
        return 0.006702
    elif "7b" in mid:
        return 0.008500
    elif "14b" in mid:
        return 0.005200
    return 0.010000

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

def sample_tokens_batch(logits, temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5, gen_tokens_list=None):
    if presence_penalty > 0.0 and gen_tokens_list is not None:
        for b in range(logits.shape[0]):
            if len(gen_tokens_list[b]) > 0:
                unique_toks = set(gen_tokens_list[b])
                for tok in unique_toks:
                    logits[b, tok] -= presence_penalty
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        min_v = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_v, torch.full_like(logits, -float("inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = torch.where(indices_to_remove, torch.full_like(logits, -float("inf")), logits)
    probs = F.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_ifeval(model, tokenizer, dataset, arm_type="base", k_val=0, batch_size=8, max_new_tokens=1280, device="cuda:0"):
    set_seed(42)
    pause_token_id = tokenizer.convert_tokens_to_ids(PAUSE_TOKEN) if PAUSE_TOKEN in tokenizer.get_vocab() else None

    strict_prompt_correct = 0
    loose_prompt_correct = 0
    strict_inst_correct = 0
    loose_inst_correct = 0
    total_instructions = 0
    total_prompts = len(dataset)

    results = []
    t_start = time.time()

    print(f"\nEvaluating IFEval ({total_prompts} prompts) | Arm: {arm_type} (K={k_val}) | Device: {device} | Batch Size: {batch_size}")

    for i in range(0, total_prompts, batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, total_prompts))]
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
            prompts.append(p_text)

        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        prompt_len = enc.input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            if arm_type == "arm2b" and k_val > 0:
                # Arm 2b: append K pause tokens to prompt
                pause_tensor = torch.full((B, k_val), pause_token_id, dtype=torch.long, device=device)
                input_ids = torch.cat([enc.input_ids, pause_tensor], dim=1)
                attn_mask = torch.cat([enc.attention_mask, torch.ones((B, k_val), dtype=torch.long, device=device)], dim=1)
                cur_prompt_len = input_ids.shape[1]
                logits_proc = LogitsProcessorList([PresencePenaltyLogitsProcessor(1.5, [cur_prompt_len] * B)])
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.80,
                    top_k=20,
                    logits_processor=logits_proc,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
                gen_start = cur_prompt_len
            elif arm_type == "arm3" and k_val > 0:
                # Arm 3: Continuous Latent Recurrence unrolling
                alpha_scale = get_calibrated_alpha(args.model_id) if "args" in locals() and hasattr(args, "model_id") else 0.011440
                pos_prefill = enc.attention_mask.long().cumsum(-1) - 1
                pos_prefill.masked_fill_(enc.attention_mask == 0, 0)
                out_base = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, position_ids=pos_prefill, use_cache=True, output_hidden_states=True)
                past_key_values = out_base.past_key_values
                h = out_base.hidden_states[-1][:, -1:, :]
                
                seq_lens = enc.attention_mask.sum(dim=-1)
                attn_mask = enc.attention_mask
                for step in range(k_val):
                    step_pos = (seq_lens + step).unsqueeze(1)
                    attn_mask = torch.cat([attn_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1)
                    scaled_latent = h * alpha_scale
                    out_step = model(inputs_embeds=scaled_latent, attention_mask=attn_mask, position_ids=step_pos, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
                    past_key_values = out_step.past_key_values
                    h = out_step.hidden_states[-1][:, -1:, :]

                # Transition into direct answer phase
                trans_tokens = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
                trans_len = len(trans_tokens)
                trans_ids = torch.tensor([trans_tokens] * B, dtype=torch.long, device=device)
                attn_mask = torch.cat([attn_mask, torch.ones((B, trans_len), dtype=torch.long, device=device)], dim=1)
                step_pos_trans = torch.stack([torch.arange(trans_len, device=device) + (seq_lens[b] + k_val) for b in range(B)], dim=0)

                logits_proc = LogitsProcessorList([PresencePenaltyLogitsProcessor(1.5, [trans_len] * B)])
                pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                gen_out = model.generate(
                    input_ids=trans_ids,
                    attention_mask=attn_mask,
                    position_ids=step_pos_trans,
                    past_key_values=past_key_values,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.80,
                    top_k=20,
                    logits_processor=logits_proc,
                    pad_token_id=pad_id
                )
                out = gen_out
                gen_start = trans_len
            else:
                # Base or Arm 1b (Direct generation with official sampling config)
                logits_proc = LogitsProcessorList([PresencePenaltyLogitsProcessor(1.5, [prompt_len] * B)])
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.80,
                    top_k=20,
                    logits_processor=logits_proc,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
                gen_start = prompt_len

        dur_batch = time.time() - t0

        for b in range(B):
            inp = input_examples[b]
            gen_ids = out[b][gen_start:].tolist()

            if tokenizer.eos_token_id in gen_ids:
                eos_idx = gen_ids.index(tokenizer.eos_token_id)
                gen_ids = gen_ids[:eos_idx]

            response_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            out_strict = test_instruction_following_strict(inp, response_text)
            out_loose = test_instruction_following_loose(inp, response_text)

            n_inst = len(inp.instruction_id_list)
            total_instructions += n_inst
            if out_strict.follow_all_instructions:
                strict_prompt_correct += 1
            if out_loose.follow_all_instructions:
                loose_prompt_correct += 1
            strict_inst_correct += sum(out_strict.follow_instruction_list)
            loose_inst_correct += sum(out_loose.follow_instruction_list)

            results.append({
                "key": inp.key,
                "prompt": inp.prompt,
                "response": response_text,
                "strict_all": out_strict.follow_all_instructions,
                "loose_all": out_loose.follow_all_instructions,
                "strict_list": out_strict.follow_instruction_list,
                "loose_list": out_loose.follow_instruction_list
            })

        processed = min(i + batch_size, total_prompts)
        curr_strict_pct = (strict_prompt_correct / processed) * 100
        print(f"Progress: [{processed}/{total_prompts}] | Strict Prompt Acc: {curr_strict_pct:.2f}% | Batch dur: {dur_batch:.2f}s", end="\r")

    total_time = time.time() - t_start
    response_lens = [len(tokenizer.encode(r["response"])) for r in results]
    truncated_count = sum(1 for l in response_lens if l >= max_new_tokens)
    sorted_lens = sorted(response_lens) if response_lens else [0]
    
    summary = {
        "arm": arm_type,
        "k_val": k_val,
        "total_prompts": total_prompts,
        "total_instructions": total_instructions,
        "strict_prompt_accuracy_pct": round(strict_prompt_correct / total_prompts * 100, 2),
        "loose_prompt_accuracy_pct": round(loose_prompt_correct / total_prompts * 100, 2),
        "strict_inst_accuracy_pct": round(strict_inst_correct / total_instructions * 100, 2),
        "loose_inst_accuracy_pct": round(loose_inst_correct / total_instructions * 100, 2),
        "truncation_count": truncated_count,
        "truncation_rate_pct": round(truncated_count / total_prompts * 100, 2),
        "token_length_min": min(response_lens) if response_lens else 0,
        "token_length_median": sorted_lens[len(sorted_lens) // 2],
        "token_length_p90": sorted_lens[int(len(sorted_lens) * 0.90)],
        "token_length_max": max(response_lens) if response_lens else 0,
        "elapsed_s": round(total_time, 1)
    }

    print("\n" + "="*80)
    print(f"IFEVAL EVALUATION RESULTS [{arm_type.upper()} K={k_val}]:")
    print(f"Strict Prompt Accuracy:      {summary['strict_prompt_accuracy_pct']:.2f}% ({strict_prompt_correct}/{total_prompts})")
    print(f"Loose Prompt Accuracy:       {summary['loose_prompt_accuracy_pct']:.2f}% ({loose_prompt_correct}/{total_prompts})")
    print(f"Strict Instruction Accuracy: {summary['strict_inst_accuracy_pct']:.2f}% ({strict_inst_correct}/{total_instructions})")
    print(f"Loose Instruction Accuracy:  {summary['loose_inst_accuracy_pct']:.2f}% ({loose_inst_correct}/{total_instructions})")
    print(f"Truncation Rate (>= {max_new_tokens}):   {summary['truncation_rate_pct']:.2f}% ({truncated_count}/{total_prompts})")
    print(f"Token Lengths (min/med/p90/max): {summary['token_length_min']} / {summary['token_length_median']} / {summary['token_length_p90']} / {summary['token_length_max']}")
    print(f"Elapsed Time:                {summary['elapsed_s']}s")
    print("="*80)

    return summary, results

def main():
    parser = argparse.ArgumentParser(description="IFEval Evaluator for Surgical Neutrality")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--arm", type=str, default="base", choices=["base", "arm1b", "arm2b", "arm3"])
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--k_val", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    print(f"Loading IFEval dataset from Hugging Face...")
    ds = load_dataset("google/IFEval", split="train")
    if args.smoke_test:
        ds = ds.select(range(4))

    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if PAUSE_TOKEN not in tokenizer.get_vocab() and args.arm == "arm2b":
        tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})

    print(f"Loading base model {args.model_id} on {args.device} in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )

    if PAUSE_TOKEN in tokenizer.get_vocab():
        model.resize_token_embeddings(len(tokenizer))

    if args.lora_path:
        print(f"Attaching LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()

    summary, evaluations = evaluate_ifeval(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        arm_type=args.arm,
        k_val=args.k_val,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        payload = {"summary": summary, "evaluations": evaluations}
        with open(args.output_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    main()
