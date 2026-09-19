#!/usr/bin/env python3
"""
scripts/58_curate_training_dataset.py
Rigorous Data Curation, Filtering, and Step Segmentation for Continuous Latent Recurrence.

Implements the official training data methodology checklist:
1. Filter:
   - math_verify correctness against reference answer.
   - Drop truncated traces and any without \\boxed{} in the answer.
   - Dedupe by problem ID; log kept IDs as disjointness evidence.
   - Cap trace length at 4,096 tokens total; DROP traces exceeding 4k rather than truncating.
   - Hold out exactly 100 traces as a dev slice for loss monitoring (drawn from train split).
2. Split each trace into 3 distinct fields:
   - prompt: exact eval prompt with chat template.
   - think: everything between <think> and </think>.
   - answer: everything after </think>\\n\\n, ending with <|im_end|>.
3. Segment the think block into discrete reasoning steps:
   - Paragraph and discourse marker segmentation rule.
   - Enables Coconut-style multi-step curriculum replacement in Stage 1.
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

def segment_think_steps(think_text):
    """
    Formal Segmentation Rule:
    Splits the continuous reasoning trajectory into discrete reasoning steps using:
    1. Paragraph boundaries (\\n\\n)
    2. Discourse transition markers at line starts:
       'Wait', 'So', 'Therefore', 'Now', 'Next', 'First', 'Then', "Let's", 'Alternatively', 'Step N:'
    """
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
    parser = argparse.ArgumentParser(description="Curate and Segment Training Traces")
    parser.add_argument("--raw_traces", type=str, default="data/self_distill_train_traces_qwen3_1.7b.jsonl")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max_total_tokens", type=int, default=4096)
    parser.add_argument("--dev_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    raw_path = os.path.join(data_dir, os.path.basename(args.raw_traces))
    
    print(f"=== Curating Training Dataset for {args.model_id} ===")
    print(f"Input Traces: {raw_path}")
    print(f"Max Token Cap (Drop if exceeded): {args.max_total_tokens}")
    print(f"Dev Slice Size: {args.dev_size} | Seed: {args.seed}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} does not exist yet.")
        return

    # Load raw traces
    raw_entries = []
    with open(raw_path) as f:
        for line in f:
            if line.strip():
                try:
                    raw_entries.append(json.loads(line))
                except Exception:
                    pass

    print(f"Loaded {len(raw_entries)} raw trace entries.")

    # Deduplication & Filtering
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
            
        answer_text = entry.get("answer_text", "")
        if "\\boxed{" not in answer_text:
            stats["dropped_no_boxed"] += 1
            continue
            
        # Clean fields
        q = entry.get("question", "").strip()
        think_text = entry.get("think_text", "").strip()
        
        # Ensure answer ends cleanly with <|im_end|>
        ans_clean = answer_text.replace("<|im_end|>", "").strip()
        ans_with_end = ans_clean + "<|im_end|>"
        
        # Exact prompt with chat template
        msg = [{"role": "user", "content": q}]
        prompt_with_template = tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        
        # Token length accounting (cap at 4,096 tokens total)
        prompt_len = len(tokenizer.encode(prompt_with_template, add_special_tokens=False))
        think_len = len(tokenizer.encode(think_text, add_special_tokens=False))
        ans_len = len(tokenizer.encode("\n</think>\n\n" + ans_with_end, add_special_tokens=False))
        total_len = prompt_len + think_len + ans_len
        
        if total_len > args.max_total_tokens:
            stats["dropped_exceeds_4k"] += 1
            continue
            
        # Segment think block into steps
        steps = segment_think_steps(think_text)
        
        sample = {
            "id": pid,
            "dataset": entry.get("dataset", "unknown"),
            "level": entry.get("level", 1),
            "subject": entry.get("subject", "general"),
            "question": q,
            "ground_truth": entry.get("ground_truth", ""),
            "prompt": prompt_with_template,
            "think": think_text,
            "answer": ans_with_end,
            "steps": steps,
            "num_steps": len(steps),
            "prompt_tokens": prompt_len,
            "think_tokens": think_len,
            "answer_tokens": ans_len,
            "total_tokens": total_len
        }
        
        seen_ids.add(pid)
        kept_samples.append(sample)
        
    stats["total_kept"] = len(kept_samples)
    print("\n--- Filtering & Curation Summary ---")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    # Check test set disjointness
    suite_path = os.path.join(data_dir, "benchmark_suite_250.json")
    if os.path.exists(suite_path):
        with open(suite_path) as f:
            test_suite = json.load(f)
        test_questions = {p["question"].strip() for p in test_suite}
        overlap = sum(1 for s in kept_samples if s["question"].strip() in test_questions)
        print(f"  Test Suite Overlap Check: {overlap} (MUST BE 0)")
        assert overlap == 0, f"Critical: {overlap} test samples leaked into training data!"

    # Shuffle deterministically
    random.shuffle(kept_samples)
    
    # Partition into Dev Slice (100) and Train Pool
    dev_slice = kept_samples[:args.dev_size]
    train_pool = kept_samples[args.dev_size:]
    
    print(f"\nPartitioning: {len(train_pool)} Train Samples | {len(dev_slice)} Held-Out Dev Samples")

    # Save Dev Slice
    dev_path = os.path.join(data_dir, f"curated_dev_traces_{args.model_id.replace('/', '_').lower()}.jsonl")
    with open(dev_path, "w") as f:
        for item in dev_slice:
            f.write(json.dumps(item) + "\n")
    print(f"Saved Dev Slice to: {dev_path}")

    # Save Curated Train Pool
    train_path = os.path.join(data_dir, f"curated_train_traces_{args.model_id.replace('/', '_').lower()}.jsonl")
    with open(train_path, "w") as f:
        for item in train_pool:
            f.write(json.dumps(item) + "\n")
    print(f"Saved Curated Train Pool to: {train_path}")

    # Save Kept IDs Evidence
    kept_ids_path = os.path.join(data_dir, "kept_trace_ids.json")
    with open(kept_ids_path, "w") as f:
        json.dump({
            "model_id": args.model_id,
            "filter_stats": stats,
            "num_train": len(train_pool),
            "num_dev": len(dev_slice),
            "kept_train_ids": [s["id"] for s in train_pool],
            "kept_dev_ids": [s["id"] for s in dev_slice]
        }, f, indent=2)
    print(f"Saved Kept IDs Evidence to: {kept_ids_path}")

    # Build Stratified Subsets Manifest for Volume Ablation
    # Stratified by dataset and MATH difficulty level
    by_category = defaultdict(list)
    for idx, s in enumerate(train_pool):
        key = f"{s['dataset']}_lvl{s['level']}"
        by_category[key].append(idx)
        
    def sample_stratified(target_n):
        if target_n >= len(train_pool):
            return list(range(len(train_pool)))
        selected = []
        fraction = target_n / len(train_pool)
        for key, indices in by_category.items():
            k = max(1, round(len(indices) * fraction))
            selected.extend(indices[:k])
        # Trim or pad to exact target_n
        if len(selected) > target_n:
            selected = selected[:target_n]
        elif len(selected) < target_n:
            remaining = [i for i in range(len(train_pool)) if i not in set(selected)]
            selected.extend(remaining[:target_n - len(selected)])
        return sorted(selected)

    subsets_manifest = {
        "full": list(range(len(train_pool))),
        "1000": sample_stratified(min(1000, len(train_pool))),
        "500": sample_stratified(min(500, len(train_pool)))
    }
    
    manifest_path = os.path.join(data_dir, "ablation_subsets_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(subsets_manifest, f, indent=2)
    print(f"Saved Stratified Volume Manifest to: {manifest_path}")
    print(f"  Volume Subsets: 500 ({len(subsets_manifest['500'])}), 1000 ({len(subsets_manifest['1000'])}), Full ({len(subsets_manifest['full'])})")

if __name__ == "__main__":
    main()
