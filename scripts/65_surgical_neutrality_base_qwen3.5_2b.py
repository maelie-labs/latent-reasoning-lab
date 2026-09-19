#!/usr/bin/env python3
"""
scripts/65_surgical_neutrality_base_qwen3.5_2b.py
ARM 0: Surgical Neutrality Baseline Evaluation for Qwen/Qwen3.5-2B on GPU 0.

Strictly Isolated to Model: Qwen/Qwen3.5-2B
Device: cuda:0 (RTX 4080 16GB)

Establishes the ground-truth base model general knowledge / reasoning performance on MMLU
(1,000 stratified questions across STEM, Humanities, Social Sciences, Other) prior to any
adapter fine-tuning.

Future adapters for Qwen3.5-2B (Arm 1b, Arm 2b, Arm 3) will be evaluated against this exact
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
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

MODEL_ID = "Qwen/Qwen3.5-2B"
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
    # Slice last token hidden state BEFORE lm_head to avoid allocating [B, L, 248320]
    backbone = model.model
    lm_head = model.lm_head
    h = backbone(**enc).last_hidden_state[:, -1:, :]
    logits = lm_head(h)[:, 0, :]
    return logits

def run_mmlu_eval(model, tokenizer, dataset_slice, dev_by_subject, device, batch_size=8):
    choices = ["A", "B", "C", "D"]
    choice_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]
    
    results = []
    t0 = time.time()
    
    for i in range(0, len(dataset_slice), batch_size):
        batch = dataset_slice[i:i + batch_size]
        prompts = [format_prompt(ex, dev_by_subject, choices) for ex in batch]
        
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        enc = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = get_last_token_logits(model, enc)
            
        for b_idx, ex in enumerate(batch):
            b_logits = logits[b_idx]
            scores = [b_logits[t].item() for t in choice_tokens]
            pred_choice = choices[scores.index(max(scores))]
            gt_choice = choices[ex["answer"]]
            is_correct = (pred_choice == gt_choice)
            
            subj = ex.get("subject", "unknown")
            cat = SUBJECT_TO_CAT.get(subj, "Other")
            
            results.append({
                "idx": i + b_idx,
                "subject": subj,
                "category": cat,
                "pred": pred_choice,
                "gt": gt_choice,
                "is_correct": is_correct,
                "scores": {c: round(s, 3) for c, s in zip(choices, scores)}
            })
            
        if len(results) % 200 == 0 or len(results) == len(dataset_slice):
            acc = sum(r["is_correct"] for r in results) / len(results) * 100.0
            elapsed = time.time() - t0
            ms_per_q = elapsed / len(results) * 1000
            print(f"  [Progress {len(results)}/{len(dataset_slice)}] Accuracy: {acc:.2f}% | Speed: {ms_per_q:.1f} ms/q")
            
    return results

def summarize_categories(results):
    cats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        c = r["category"]
        cats[c]["total"] += 1
        if r["is_correct"]:
            cats[c]["correct"] += 1
    summary = {}
    for c, v in sorted(cats.items()):
        acc = (v["correct"] / v["total"] * 100.0) if v["total"] > 0 else 0.0
        summary[c] = {
            "total": v["total"],
            "correct": v["correct"],
            "accuracy_pct": round(acc, 2)
        }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Arm 0: Surgical Neutrality Base Run for Qwen3.5-2B")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--mmlu_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_file", type=str, default="data/surgical_neutrality_base_qwen3.5_2b.json")
    args = parser.parse_args()

    set_seed(42)
    print(f"========================================================")
    print(f"=== ARM 0: Base Surgical Neutrality for {args.model_id} ===")
    print(f"Device: {args.device} | MMLU Questions: {args.mmlu_samples} | Batch Size: {args.batch_size}")
    print(f"========================================================")

    print("\n>>> Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("\n>>> Loading MMLU dataset (cais/mmlu)...")
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")
    test_ds = load_dataset("cais/mmlu", "all", split="test")

    dev_by_subject = defaultdict(list)
    for ex in dev_ds:
        dev_by_subject[ex["subject"]].append(ex)

    indices = list(range(len(test_ds)))
    random.Random(42).shuffle(indices)
    selected_indices = indices[:args.mmlu_samples]
    mmlu_slice = [test_ds[i] for i in selected_indices]
    print(f"Sampled {len(mmlu_slice)} stratified MMLU test questions across {len(set(ex['subject'] for ex in mmlu_slice))} subjects.")

    print(f"\n>>> Loading {args.model_id} in Pure BFloat16 on {args.device}...")
    t0_load = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    base_model.eval()
    print(f"Model loaded in {time.time() - t0_load:.1f}s | VRAM Allocated: {torch.cuda.memory_allocated(args.device) / (1024**2):.1f} MiB")

    print("\n>>> Running MMLU Evaluation...")
    base_results = run_mmlu_eval(base_model, tokenizer, mmlu_slice, dev_by_subject, args.device, batch_size=args.batch_size)
    base_acc = sum(r["is_correct"] for r in base_results) / len(base_results) * 100.0
    base_cat_summary = summarize_categories(base_results)
    print(f"\n>> Base Model MMLU Accuracy: {base_acc:.2f}%")
    for cat, data in base_cat_summary.items():
        print(f"   - {cat:<16}: {data['accuracy_pct']:5.2f}% ({data['correct']}/{data['total']})")

    report = {
        "model_id": args.model_id,
        "device": args.device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mmlu_samples": args.mmlu_samples,
        "base_accuracy": round(base_acc, 2),
        "base_categories": base_cat_summary,
        "results": base_results
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*70)
    print("ARM 0 SURGICAL NEUTRALITY BASE REPORT (Qwen3.5-2B)")
    print("="*70)
    print(f"Model: {args.model_id}")
    print(f"Base MMLU Accuracy: {base_acc:.2f}%")
    for cat, data in base_cat_summary.items():
        print(f"  {cat:<16}: {data['accuracy_pct']:5.2f}% ({data['correct']}/{data['total']})")
    print("="*70)
    print(f"Saved to: {args.output_file}")

if __name__ == "__main__":
    main()
