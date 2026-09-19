#!/usr/bin/env python3
"""
Test prompt-based register extraction on Qwen/Qwen3-1.7B on GPU 1.
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
DEVICE = "cuda:1"

EXTRACTION_SYSTEM_PROMPT = """You are a mathematical register compiler. Your job is to extract structured mathematical registers from a problem and its solution.
Format your output EXACTLY into the following 3 rungs using `<|reg|>key = value<|/reg|>` tags:

[R1: Identify givens, constraints, and target variable]
<|reg|>var1 = val1<|/reg|>
<|reg|>var2 = val2<|/reg|>

[R2: Compute intermediate operations and verify relations]
<|reg|>op1 = val<|/reg|>
<|reg|>op2 = val<|/reg|>

[R3: Execute final deduction and verify constraints]
<|reg|>ans = final_val<|/reg|>

Rules:
1. Each key must be a valid identifier (letters, numbers, underscore).
2. Values should be concise numbers, expressions, or equations.
3. Do NOT include any commentary or extra text outside the rungs.
"""

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )

    # Read 2 samples from curated train
    train_file = "data/curated_train_v1_1_dilution10pct_qwen3_1.7b.jsonl"
    samples = []
    with open(train_file) as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                if s.get("dataset") in ["gsm8k", "MATH"]:
                    samples.append(s)
                    if len(samples) >= 2:
                        break

    for i, s in enumerate(samples):
        print("=" * 60)
        print(f"Problem {i+1} ({s['dataset']}):")
        print(s["question"][:150], "...")
        user_msg = f"Problem:\n{s['question']}\n\nWorked Solution:\n{s['nonthinking_answer']}\n\nExtract the structured registers now:"
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Condition immediate non-thinking response
        prompt += "<think>\n\n</think>\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
            )
        gen_tokens = out[0][inputs.input_ids.shape[1]:]
        resp = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        print("--- EXTRACTED REGISTERS ---")
        print(resp)

if __name__ == "__main__":
    main()
