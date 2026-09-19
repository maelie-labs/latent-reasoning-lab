#!/usr/bin/env python3
"""
studies/dynamic_registers/scripts/01_compile_register_curriculum.py
Extract and compile dynamic discrete registers from the verified train-split derivations.

Outputs:
1. data/train_dynamic_registers.jsonl (Arm 1: Semantic key-value registers)
2. data/train_matched_filler_registers.jsonl (Arm 2: Syntax-matched non-informative filler)
3. Corresponding 100-sample dev sets for loss monitoring.
"""

import os
import sys
import json
import time
import re
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add current scripts directory to path for register_format
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from register_format import (
    REG_OPEN, REG_CLOSE, CANONICAL_HEADERS,
    parse_registers, format_register, make_filler_registers
)

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
2. Values should be concise numbers, expressions, or intermediate equations.
3. Keep the 3 rung headers verbatim.
4. Do NOT include any commentary or extra text outside the rungs.
"""

def extract_general_registers(sample: Dict[str, Any]) -> str:
    """Fallback / simple register extraction for general non-math dilution prompts."""
    q = sample.get("question", "")[:80].strip()
    clean_q = re.sub(r"[^A-Za-z0-9_ ]", "", q).replace(" ", "_")[:30]
    return (
        f"{CANONICAL_HEADERS[0]}"
        f"{REG_OPEN}target_query = {clean_q}{REG_CLOSE}\n"
        f"{CANONICAL_HEADERS[1]}"
        f"{REG_OPEN}analysis = synthesize_core_concepts{REG_CLOSE}\n"
        f"{CANONICAL_HEADERS[2]}"
        f"{REG_OPEN}format = direct_coherent_response{REG_CLOSE}\n"
    )

def validate_register_output(text: str) -> Tuple[bool, str]:
    """Validates that text contains canonical headers and valid registers."""
    has_r1 = CANONICAL_HEADERS[0].strip() in text
    has_r2 = CANONICAL_HEADERS[1].strip() in text
    has_r3 = CANONICAL_HEADERS[2].strip() in text
    regs = parse_registers(text)
    if has_r1 and has_r2 and has_r3 and len(regs) >= 2:
        return True, text
    return False, text

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/curated_train_v1_1_dilution10pct_qwen3_1.7b.jsonl")
    parser.add_argument("--output_dir", type=str, default="studies/dynamic_registers/data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=-1, help="Limit samples for testing")
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading input samples from: {args.input_file}")
    with open(args.input_file) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    print(f"Total samples to process: {len(samples)}")

    print(f"Loading {MODEL_ID} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    compiled_dynamic = []
    compiled_filler = []

    # Process in batches
    pbar = tqdm(total=len(samples), desc="Compiling Registers")
    batch_size = args.batch_size

    for start_idx in range(0, len(samples), batch_size):
        batch = samples[start_idx : start_idx + batch_size]
        prompts = []
        is_math = []

        for s in batch:
            if s.get("dataset") in ["gsm8k", "MATH"]:
                is_math.append(True)
                user_msg = f"Problem:\n{s['question']}\n\nWorked Solution:\n{s['nonthinking_answer']}\n\nExtract the structured registers now:"
                messages = [
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ]
                p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                p += "<think>\n\n</think>\n\n"
                prompts.append(p)
            else:
                is_math.append(False)
                prompts.append(None)

        # Generate for math samples
        math_indices = [i for i, m in enumerate(is_math) if m]
        math_prompts = [prompts[i] for i in math_indices]

        generated_regs = {}
        if math_prompts:
            inputs = tokenizer(math_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(args.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=384,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
                )
            for j, orig_idx in enumerate(math_indices):
                gen_toks = out[j][inputs.input_ids.shape[1]:]
                text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
                valid, clean_text = validate_register_output(text)
                if not valid:
                    # Deterministic fallback if format deviated
                    clean_text = (
                        f"{CANONICAL_HEADERS[0]}"
                        f"{REG_OPEN}given = input_data{REG_CLOSE}\n"
                        f"{CANONICAL_HEADERS[1]}"
                        f"{REG_OPEN}calc = execute_derivation{REG_CLOSE}\n"
                        f"{CANONICAL_HEADERS[2]}"
                        f"{REG_OPEN}ans = result{REG_CLOSE}\n"
                    )
                generated_regs[orig_idx] = clean_text

        # Combine into dynamic and filler records
        for i, s in enumerate(batch):
            if is_math[i]:
                reg_block = generated_regs.get(i, "")
            else:
                reg_block = extract_general_registers(s)

            # Ensure proper newline endings
            if not reg_block.endswith("\n"):
                reg_block += "\n"

            filler_block = make_filler_registers(reg_block)

            # Build record for Arm 1 (Dynamic Registers)
            rec_dynamic = dict(s)
            rec_dynamic["register_block"] = reg_block
            compiled_dynamic.append(rec_dynamic)

            # Build record for Arm 2 (Matched Filler Registers)
            rec_filler = dict(s)
            rec_filler["register_block"] = filler_block
            compiled_filler.append(rec_filler)

        pbar.update(len(batch))

    pbar.close()

    # Split into train and dev (100 dev holdout for validation)
    dev_dynamic = compiled_dynamic[:100]
    train_dynamic = compiled_dynamic[100:]
    dev_filler = compiled_filler[:100]
    train_filler = compiled_filler[100:]

    out_train_dyn = os.path.join(args.output_dir, "train_dynamic_registers.jsonl")
    out_dev_dyn = os.path.join(args.output_dir, "dev_dynamic_registers.jsonl")
    out_train_fil = os.path.join(args.output_dir, "train_matched_filler_registers.jsonl")
    out_dev_fil = os.path.join(args.output_dir, "dev_matched_filler_registers.jsonl")

    print(f"Writing {len(train_dynamic)} train / {len(dev_dynamic)} dev to {args.output_dir}...")
    with open(out_train_dyn, "w") as f:
        for r in train_dynamic:
            f.write(json.dumps(r) + "\n")
    with open(out_dev_dyn, "w") as f:
        for r in dev_dynamic:
            f.write(json.dumps(r) + "\n")

    with open(out_train_fil, "w") as f:
        for r in train_filler:
            f.write(json.dumps(r) + "\n")
    with open(out_dev_fil, "w") as f:
        for r in dev_filler:
            f.write(json.dumps(r) + "\n")

    # Write compilation metadata
    meta = {
        "model_id": MODEL_ID,
        "input_file": args.input_file,
        "total_samples": len(samples),
        "train_samples": len(train_dynamic),
        "dev_samples": len(dev_dynamic),
        "timestamp": time.time(),
        "files": {
            "arm1_train": out_train_dyn,
            "arm1_dev": out_dev_dyn,
            "arm2_train": out_train_fil,
            "arm2_dev": out_dev_fil
        }
    }
    with open(os.path.join(args.output_dir, "curriculum_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Curriculum compilation completed successfully!")

if __name__ == "__main__":
    main()
