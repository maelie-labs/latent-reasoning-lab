#!/usr/bin/env python3
"""
scripts/27_format_traces_qwen3_1.7b.py
Formats GSM8K self-distillation reasoning traces for Qwen/Qwen3-1.7B.
Uses the model's native chat template with <think> and </think> delimiters.
"""

import os
import json
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"

def main():
    print(f"=== Formatting GSM8K Traces for {MODEL_ID} ===")
    
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip("\"'\n")
                    break

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token, trust_remote_code=True)
    
    src_path = os.path.join(os.path.dirname(__file__), "..", "data", "traces_gsm8k_200.jsonl")
    dst_path = os.path.join(os.path.dirname(__file__), "..", "data", "traces_gsm8k_qwen3_1.7b.jsonl")
    
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source traces not found at {src_path}")
        
    count = 0
    with open(src_path, "r", encoding="utf-8") as f_in, open(dst_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            question = data["question"]
            think_text = data["think_text"]
            answer_text = data["answer_text"]
            ground_truth = data["ground_truth"]
            
            # Format using Qwen3 chat template
            messages = [{"role": "user", "content": question}]
            # Apply template with generation prompt
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Append <think>\n to match the reasoning format
            prompt_with_think = formatted_prompt + "<think>\n"
            
            # Check token counts with Qwen3 tokenizer
            think_tokens = tokenizer.encode(think_text, add_special_tokens=False)
            answer_tokens = tokenizer.encode(f"</think>\n\n{answer_text}", add_special_tokens=False)
            
            record = {
                "id": data["id"],
                "question": question,
                "ground_truth": ground_truth,
                "formatted_prompt": prompt_with_think,
                "think_text": think_text,
                "answer_text": answer_text,
                "num_think_tokens": len(think_tokens),
                "num_answer_tokens": len(answer_tokens),
                "valid_format": True
            }
            f_out.write(json.dumps(record) + "\n")
            count += 1
            
    print(f"Successfully formatted {count} traces to {dst_path}")

if __name__ == "__main__":
    main()
