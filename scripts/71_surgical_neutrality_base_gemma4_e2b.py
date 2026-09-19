#!/usr/bin/env python3
"""
scripts/71_surgical_neutrality_base_gemma4_e2b.py
ARM 0: Surgical Neutrality Baseline Evaluation for google/gemma-4-E2B-it on GPU 0.

Strictly Isolated to Model: google/gemma-4-E2B-it
Device: cuda:0 (RTX 4080 16GB)

Establishes the ground-truth base model general knowledge / reasoning performance on MMLU
(1,000 stratified questions across STEM, Humanities, Social Sciences, Other) prior to any
adapter fine-tuning.

Future adapters for Gemma-4-E2B (Arm 1b, Arm 2b, Arm 3) will be evaluated against this exact
baseline to assert the pre-registered non-inferiority bound: Delta(Adapter - Base) >= -1.5% (95% CI_low).
"""

import os
import re
import json
import time
import argparse
import random
from collections import defaultdict
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
import transformers.modeling_utils

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

MODEL_ID = "google/gemma-4-E2B-it"
DEVICE = "cuda:0"

CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning"
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions"
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy"
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology"
    ]
}

SUBJECT_TO_CAT = {}
for cat, subjs in CATEGORIES.items():
    for s in subjs:
        SUBJECT_TO_CAT[s] = cat

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_prompt(example, dev_by_subject, choices, n_shot=5):
    subj = example.get("subject", "")
    dev_pool = dev_by_subject.get(subj, [])
    
    prompt = f"The following are multiple choice questions (with answers) about {subj.replace('_', ' ')}.\n\n"
    
    # 5 few-shot demonstrations
    shots = dev_pool[:n_shot]
    for shot in shots:
        prompt += f"{shot['question']}\n"
        for idx, opt in enumerate(shot["choices"]):
            prompt += f"{choices[idx]}. {opt}\n"
        prompt += f"Answer: {choices[shot['answer']]}\n\n"
        
    # Target question
    prompt += f"{example['question']}\n"
    for idx, opt in enumerate(example["choices"]):
        prompt += f"{choices[idx]}. {opt}\n"
    prompt += "Answer:"
    return prompt

def get_last_token_logits(model, enc):
    out = model(**enc, logits_to_keep=1)
    return out.logits[:, 0, :]

def main():
    parser = argparse.ArgumentParser(description="Arm 0 Base Surgical Neutrality for Gemma-4-E2B on GPU 0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    out_file = args.output_file or os.path.join(data_dir, "surgical_neutrality_base_gemma4_e2b.json")

    print(f"========================================================")
    print(f"=== ARM 0: Base Surgical Neutrality Baseline: {MODEL_ID} ===")
    print(f"Target Device: {args.device} (RTX 4080 16GB)")
    print(f"Benchmark: MMLU ({args.n_samples} stratified questions across 57 subjects)")
    print(f"Output File: {out_file}")
    print(f"========================================================")

    # 1. Load MMLU dataset
    print("\n>>> Loading MMLU dataset...")
    test_ds = load_dataset("cais/mmlu", "all", split="test")
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")

    dev_by_subject = defaultdict(list)
    for row in dev_ds:
        dev_by_subject[row["subject"]].append(row)

    # Stratified selection
    by_subj = defaultdict(list)
    for row in test_ds:
        by_subj[row["subject"]].append(row)

    selected_examples = []
    subjs = sorted(by_subj.keys())
    per_subj = args.n_samples // len(subjs)
    remainder = args.n_samples % len(subjs)

    rng = random.Random(args.seed)
    for i, s in enumerate(subjs):
        count = per_subj + (1 if i < remainder else 0)
        items = list(by_subj[s])
        rng.shuffle(items)
        selected_examples.extend(items[:count])

    rng.shuffle(selected_examples)
    selected_examples = selected_examples[:args.n_samples]
    print(f"Selected {len(selected_examples)} stratified MMLU test questions.")

    # 2. Load tokenizer and model
    print(f"\n>>> Loading Tokenizer & Model in pure BFloat16 on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()

    choices = ["A", "B", "C", "D"]
    choice_ids = [tokenizer.encode(" " + c, add_special_tokens=False)[-1] for c in choices]
    raw_choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]

    # 3. Evaluate Base Model
    print(f"\n>>> Evaluating {args.n_samples} MMLU questions on Base Model...")
    t0 = time.time()
    correct_base = 0
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    detailed_records = []

    for i in range(0, len(selected_examples), args.batch_size):
        batch = selected_examples[i:i + args.batch_size]
        prompts = [format_prompt(ex, dev_by_subject, choices, n_shot=5) for ex in batch]
        
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, padding=True, return_tensors="pt").to(args.device)

        with torch.no_grad():
            logits = get_last_token_logits(model, enc)

        for b, ex in enumerate(batch):
            b_logits = logits[b]
            scores = [max(b_logits[choice_ids[c]].item(), b_logits[raw_choice_ids[c]].item()) for c in range(4)]
            pred = int(np.argmax(scores))
            gold = ex["answer"]
            is_corr = (pred == gold)

            if is_corr:
                correct_base += 1
            subj = ex["subject"]
            cat = SUBJECT_TO_CAT.get(subj, "Other")
            cat_total[cat] += 1
            if is_corr:
                cat_correct[cat] += 1

            detailed_records.append({
                "index": i + b,
                "subject": subj,
                "category": cat,
                "pred": choices[pred],
                "gold": choices[gold],
                "is_correct": is_corr
            })

        if (i + len(batch)) % 100 == 0 or (i + len(batch)) == len(selected_examples):
            done = i + len(batch)
            cur_acc = 100.0 * correct_base / done
            elapsed = time.time() - t0
            print(f"[{done:4d}/{len(selected_examples)}] Base Accuracy: {cur_acc:.2f}% | Elapsed: {elapsed:.1f}s")

    dur = time.time() - t0
    base_acc = 100.0 * correct_base / len(selected_examples)
    print(f"\n========================================================")
    print(f"Base MMLU Evaluation Complete in {dur:.1f}s ({dur/len(selected_examples)*1000:.1f} ms/q)")
    print(f"Base Overall Accuracy: {base_acc:.2f}% ({correct_base}/{len(selected_examples)})")
    print(f"Category Breakdown:")
    for cat in sorted(CATEGORIES.keys()):
        tot = cat_total[cat]
        cor = cat_correct[cat]
        acc_c = 100.0 * cor / tot if tot > 0 else 0.0
        print(f"  {cat:15s}: {acc_c:.2f}% ({cor}/{tot})")
    print(f"========================================================")

    # 4. Save Base Ground Truth
    result_data = {
        "model_id": MODEL_ID,
        "device": args.device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mmlu_samples": len(selected_examples),
        "base_accuracy": round(base_acc, 2),
        "duration_s": round(dur, 2),
        "ms_per_query": round(dur / len(selected_examples) * 1000, 1),
        "base_categories": {
            cat: {
                "total": cat_total[cat],
                "correct": cat_correct[cat],
                "accuracy_pct": round(100.0 * cat_correct[cat] / cat_total[cat], 2) if cat_total[cat] > 0 else 0.0
            }
            for cat in sorted(CATEGORIES.keys())
        },
        "per_sample_results": detailed_records
    }

    with open(out_file, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Saved Base Ground-Truth Neutrality Baseline to: {out_file}")

if __name__ == "__main__":
    main()
