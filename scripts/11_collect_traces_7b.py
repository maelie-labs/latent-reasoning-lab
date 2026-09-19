#!/usr/bin/env python3
"""
11_collect_traces_7b.py
Self-distillation trace collector for deepseek-ai/DeepSeek-R1-Distill-Qwen-7B.
Generates 150 reasoning traces (<think> ... </think> + final answer) on GSM8K training set.
Saves to data/traces_gsm8k_7b.jsonl.
Runs on cuda:1 (RTX PRO 4500 32GB).
"""

import os
import json
import time
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEVICE = "cuda:1"

def parse_args():
    parser = argparse.ArgumentParser(description="Collect reasoning traces from DeepSeek-R1-Distill-7B")
    parser.add_argument("--num_samples", type=int, default=150, help="Number of traces to collect")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--output_file", type=str, default="data/traces_gsm8k_7b.jsonl", help="Output jsonl path")
    parser.add_argument("--max_new_tokens", type=int, default=768, help="Max tokens per generation")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"=== Self-Distillation Trace Collection (7B): {args.num_samples} Samples ===")
    print(f"Device: {DEVICE} | Batch Size: {args.batch_size} | Output: {args.output_file}")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading {MODEL_ID} in bfloat16 on {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=DEVICE
    )
    model.eval()
    
    ds = load_dataset("openai/gsm8k", "main", split="train")
    samples = ds.select(range(min(args.num_samples, len(ds))))
    print(f"Loaded {len(samples)} GSM8K questions.")
    
    total_generated_tokens = 0
    total_think_tokens = 0
    total_answer_tokens = 0
    valid_format_count = 0
    start_total_time = time.time()
    
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for b_start in range(0, len(samples), args.batch_size):
            b_end = min(b_start + args.batch_size, len(samples))
            batch_slice = [samples[i] for i in range(b_start, b_end)]
            
            formatted_prompts = []
            for item in batch_slice:
                msgs = [{"role": "user", "content": item["question"]}]
                p_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                formatted_prompts.append(p_text)
                
            enc = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=False).to(DEVICE)
            
            b_time_start = time.time()
            with torch.no_grad():
                gen_outputs = model.generate(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.6,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            b_time = time.time() - b_time_start
            
            batch_tokens = 0
            for i, item in enumerate(batch_slice):
                out_ids = gen_outputs[i][enc.input_ids.shape[1]:]
                gen_text = tokenizer.decode(out_ids, skip_special_tokens=False)
                
                if tokenizer.eos_token in gen_text:
                    gen_text = gen_text.split(tokenizer.eos_token)[0]
                    
                total_tokens = len(tokenizer.encode(gen_text))
                batch_tokens += total_tokens
                total_generated_tokens += total_tokens
                
                has_end_think = "</think>" in gen_text
                if has_end_think:
                    parts = gen_text.split("</think>")
                    think_text = parts[0].strip()
                    answer_text = parts[1].strip()
                    think_tok_len = len(tokenizer.encode(think_text))
                    ans_tok_len = len(tokenizer.encode(answer_text))
                    valid = True
                    valid_format_count += 1
                    total_think_tokens += think_tok_len
                    total_answer_tokens += ans_tok_len
                else:
                    think_text = gen_text
                    answer_text = ""
                    think_tok_len = total_tokens
                    ans_tok_len = 0
                    valid = False
                    
                record = {
                    "id": b_start + i,
                    "question": item["question"],
                    "ground_truth": item["answer"],
                    "formatted_prompt": formatted_prompts[i],
                    "raw_generation": gen_text,
                    "think_text": think_text,
                    "answer_text": answer_text,
                    "num_think_tokens": think_tok_len,
                    "num_answer_tokens": ans_tok_len,
                    "valid_format": valid
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                
            tps = batch_tokens / b_time if b_time > 0 else 0
            print(f"Processed {b_end}/{len(samples)} | Batch: {b_time:.1f}s ({tps:.1f} tok/s) | Valid Format: {valid_format_count}/{b_end}")
            
    total_time = time.time() - start_total_time
    print("\n=== Data Collection Completed ===")
    print(f"Total Time: {total_time:.1f}s ({total_generated_tokens / total_time:.1f} avg tok/s)")
    print(f"Total Generated Tokens: {total_generated_tokens}")
    if valid_format_count > 0:
        print(f"Avg Thinking Tokens: {total_think_tokens / valid_format_count:.1f}")
        print(f"Avg Answer Tokens: {total_answer_tokens / valid_format_count:.1f}")
    print(f"Saved to: {args.output_file}")

if __name__ == "__main__":
    main()
