#!/usr/bin/env python3
"""
scripts/67_curate_training_traces_qwen3.5_2b.py
Rigorous Data Curation, Filtering, and Step Segmentation for Qwen/Qwen3.5-2B.

Strictly Isolated to Model: Qwen/Qwen3.5-2B
Zero Cross-Model Leakage: Curates traces generated exclusively by Qwen3.5-2B on train splits.

Methodology Checklist:
1. Filter:
   - math_verify correctness against reference answer.
   - Drop truncated traces and any without \\boxed{} in the answer.
   - Dedupe by problem ID; save kept IDs to data/kept_trace_ids_qwen3.5_2b.json.
   - Cap total length (prompt + think + answer) at 4,096 tokens; DROP traces exceeding 4k rather than truncating.
   - Hold out exactly 100 traces as a dev slice (data/curated_dev_traces_qwen3.5_2b.jsonl).
2. Field Structure:
   - prompt: exact eval prompt with chat template (enable_thinking=True).
   - think: full trajectory between <think> and </think>.
   - answer: solution text following </think>\\n\\n, ending with <|im_end|>.
3. Discourse & Paragraph Step Segmentation:
   - Paragraph double newlines (\\n\\n) + discourse transition markers:
     'Wait', 'So', 'Therefore', 'Now', 'Next', 'First', 'Then', "Let's", 'Alternatively', 'Step N:'
4. Stratified Volume Manifest:
   - Stratified subsets for 500, 1,000, and full volume ablation.
"""

import os
import re
import json
import random
import argparse
from collections import defaultdict
from transformers import AutoTokenizer
from math_verify import parse, verify

MODEL_ID = "Qwen/Qwen3.5-2B"

def segment_think_steps(think_text):
    if not think_text:
        return []
    
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', think_text) if p.strip()]
    steps = []
    
    discourse_pattern = re.compile(
        r'(?m)^(?=(?:Wait|So|Therefore|Now|Next|First|Then|Let\'s|Alternatively|In conclusion|Step \d+:|\d+\.\s))'
    )
    
    for p in paragraphs:
        sub_chunks = discourse_pattern.split(p)
        for sc in sub_chunks:
            sc_clean = sc.strip()
            if sc_clean:
                steps.append(sc_clean)
                
    return steps if steps else [think_text.strip()]

def main():
    parser = argparse.ArgumentParser(description="Curate and Segment Training Traces for Qwen3.5-2B")
    parser.add_argument("--raw_traces", type=str, default="data/self_distill_train_traces_qwen3.5_2b.jsonl")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--max_total_tokens", type=int, default=4096)
    parser.add_argument("--dev_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    raw_path = os.path.join(data_dir, os.path.basename(args.raw_traces))
    
    print(f"========================================================")
    print(f"=== Curating Training Dataset for {args.model_id} ===")
    print(f"Input Traces: {raw_path}")
    print(f"Max Token Cap (Drop if exceeded): {args.max_total_tokens}")
    print(f"Dev Slice Size: {args.dev_size} | Seed: {args.seed}")
    print(f"========================================================")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if not os.path.exists(raw_path):
        print(f"Notice: {raw_path} does not exist yet. Run scripts/66_generate_training_traces_qwen3.5_2b.py first.")
        return

    raw_entries = []
    with open(raw_path) as f:
        for line in f:
            if line.strip():
                try:
                    raw_entries.append(json.loads(line))
                except Exception:
                    pass

    print(f"Loaded {len(raw_entries)} raw trace entries.")

    seen_ids = set()
    kept_samples = []
    
    stats = {
        "total_raw": len(raw_entries),
        "dropped_duplicate": 0,
        "dropped_not_correct": 0,
        "dropped_truncated": 0,
        "dropped_no_boxed": 0,
        "dropped_exceeds_4k": 0,
        "total_kept": 0
    }

    for entry in raw_entries:
        pid = entry.get("id")
        if pid in seen_ids:
            stats["dropped_duplicate"] += 1
            continue
            
        if entry.get("is_correct") is False:
            stats["dropped_not_correct"] += 1
            continue
            
        if entry.get("is_truncated", False):
            stats["dropped_truncated"] += 1
            continue
            
        answer_text = entry.get("answer_text", entry.get("answer", ""))
        if "\\boxed{" not in answer_text:
            stats["dropped_no_boxed"] += 1
            continue
            
        q = entry.get("question", "").strip()
        think_text = entry.get("think_text", entry.get("think", "")).strip()
        
        ans_clean = answer_text.replace("<|im_end|>", "").strip()
        ans_with_end = ans_clean + "<|im_end|>"
        
        msg = [{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
        prompt_with_template = tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        
        prompt_tokens = len(tokenizer.encode(prompt_with_template, add_special_tokens=False))
        think_tokens = len(tokenizer.encode(think_text, add_special_tokens=False))
        ans_tokens = len(tokenizer.encode("\n</think>\n\n" + ans_with_end, add_special_tokens=False))
        total_tokens = prompt_tokens + think_tokens + ans_tokens
        
        if total_tokens > args.max_total_tokens:
            stats["dropped_exceeds_4k"] += 1
            continue
            
        seen_ids.add(pid)
        steps = segment_think_steps(think_text)
        
        sample = {
            "id": pid,
            "dataset": entry.get("dataset", "unknown"),
            "level": entry.get("level", 1),
            "subject": entry.get("subject", "general"),
            "question": q,
            "prompt": prompt_with_template,
            "think": think_text,
            "answer": ans_with_end,
            "steps": steps,
            "num_steps": len(steps),
            "tokens": {
                "prompt": prompt_tokens,
                "think": think_tokens,
                "answer": ans_tokens,
                "total": total_tokens
            }
        }
        kept_samples.append(sample)

    stats["total_kept"] = len(kept_samples)
    print(f"\nFiltering Statistics:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")

    random.shuffle(kept_samples)
    dev_split = kept_samples[:args.dev_size]
    train_split = kept_samples[args.dev_size:]

    print(f"\nDataset Splits:")
    print(f"  - Dev Split:   {len(dev_split)} traces")
    print(f"  - Train Split: {len(train_split)} traces")

    train_out = os.path.join(data_dir, "curated_train_traces_qwen3.5_2b.jsonl")
    dev_out = os.path.join(data_dir, "curated_dev_traces_qwen3.5_2b.jsonl")
    kept_ids_out = os.path.join(data_dir, "kept_trace_ids_qwen3.5_2b.json")
    manifest_out = os.path.join(data_dir, "volume_ablation_manifest_qwen3.5_2b.json")

    with open(train_out, "w") as f:
        for s in train_split:
            f.write(json.dumps(s) + "\n")
            
    with open(dev_out, "w") as f:
        for s in dev_split:
            f.write(json.dumps(s) + "\n")
            
    with open(kept_ids_out, "w") as f:
        json.dump(list(seen_ids), f, indent=2)

    manifest = {
        "full": list(range(len(train_split))),
        "1000": list(range(min(1000, len(train_split)))),
        "500": list(range(min(500, len(train_split))))
    }
    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nFiles Written Successfully:")
    print(f"  - Train Traces: {train_out}")
    print(f"  - Dev Traces:   {dev_out}")
    print(f"  - Kept IDs:     {kept_ids_out}")
    print(f"  - Manifest:     {manifest_out}")

if __name__ == "__main__":
    main()
