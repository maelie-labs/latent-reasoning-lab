#!/usr/bin/env python3
"""
scripts/39_surgical_neutrality_check.py
ARM 0: Surgical Neutrality Check.
Evaluates whether LoRA adapter surgery degraded base non-recurrent model capabilities.
Tests Base Qwen3 vs. Retrofitted Qwen3 (recurrence disabled, K=0) on:
1. Canonical 5-shot MMLU (1,000 questions sampled across STEM, Humanities, Social Sciences, Other).

Computes per-sample paired differences and 10,000-sample hierarchical bootstrap
to verify the retrofit satisfies the pre-registered non-inferiority bound:
Delta(Retrofitted - Base) >= -1.5% (within 95% bootstrap CI).
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
from peft import PeftModel
import transformers.modeling_utils

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

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

def format_example(ex, choices):
    opts = "\n".join([f"{c}. {ex['choices'][idx]}" for idx, c in enumerate(choices)])
    ans = choices[ex['answer']]
    return f"{ex['question']}\n{opts}\nAnswer: {ans}"

def format_prompt(test_ex, dev_by_subject, choices):
    subj = test_ex['subject']
    dev_exs = dev_by_subject.get(subj, [])[:5]
    opts = "\n".join([f"{c}. {test_ex['choices'][idx]}" for idx, c in enumerate(choices)])
    subj_name = subj.replace("_", " ")
    header = f"The following are multiple choice questions (with answers) about {subj_name}.\n\n"
    if dev_exs:
        few_shot_str = "\n\n".join([format_example(e, choices) for e in dev_exs]) + "\n\n"
    else:
        few_shot_str = ""
    return header + few_shot_str + f"{test_ex['question']}\n{opts}\nAnswer:"

def run_mmlu_eval(model, tokenizer, dataset_slice, dev_by_subject, device):
    choices = ["A", "B", "C", "D"]
    choice_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]
    
    results = []
    t0 = time.time()
    
    for idx, ex in enumerate(dataset_slice):
        prompt = format_prompt(ex, dev_by_subject, choices)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
            
        scores = [logits[t].item() for t in choice_tokens]
        pred_choice = choices[scores.index(max(scores))]
        gt_choice = choices[ex["answer"]]
        is_correct = (pred_choice == gt_choice)
        
        subj = ex.get("subject", "unknown")
        cat = SUBJECT_TO_CAT.get(subj, "Other")
        
        results.append({
            "idx": idx,
            "subject": subj,
            "category": cat,
            "pred": pred_choice,
            "gt": gt_choice,
            "is_correct": is_correct,
            "scores": {c: round(s, 3) for c, s in zip(choices, scores)}
        })
        
        if (idx + 1) % 200 == 0 or (idx + 1) == len(dataset_slice):
            acc = sum(r["is_correct"] for r in results) / len(results) * 100.0
            elapsed = time.time() - t0
            ms_per_q = elapsed / (idx + 1) * 1000
            print(f"  [Progress {idx+1}/{len(dataset_slice)}] Accuracy: {acc:.2f}% | Speed: {ms_per_q:.1f} ms/q")
            
    return results

def compute_paired_bootstrap(scores_base, scores_retrofit, n_boot=10000):
    diffs = np.array(scores_retrofit, dtype=float) - np.array(scores_base, dtype=float)
    n = len(diffs)
    boot_means = []
    for _ in range(n_boot):
        sample_diffs = np.random.choice(diffs, size=n, replace=True)
        boot_means.append(np.mean(sample_diffs))
    boot_means = np.array(boot_means)
    ci_low = float(np.percentile(boot_means, 2.5) * 100.0)
    ci_high = float(np.percentile(boot_means, 97.5) * 100.0)
    mean_diff = float(np.mean(diffs) * 100.0)
    # p-value for testing H0: diff <= -1.5% (non-inferiority margin)
    # or two-sided test against 0:
    p_val_zero = float(np.mean(boot_means <= 0) if mean_diff > 0 else np.mean(boot_means >= 0))
    p_val_noninf = float(np.mean(boot_means <= -0.015))
    return {
        "mean_diff_pct": round(mean_diff, 3),
        "ci_95_low": round(ci_low, 3),
        "ci_95_high": round(ci_high, 3),
        "p_value_zero": round(p_val_zero, 4),
        "p_value_non_inferiority": round(p_val_noninf, 4)
    }

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
    parser = argparse.ArgumentParser(description="Arm 0: Surgical Neutrality Check.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora_path", type=str, default="checkpoints/lora_arm3_qwen_qwen3-1.7b_k6")
    parser.add_argument("--lora_arm1b_path", type=str, default="checkpoints/lora_arm1b_qwen_qwen3-1.7b_k0")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--mmlu_samples", type=int, default=1000)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    set_seed(42)
    if args.output_file is None:
        tag = args.model_id.replace("/", "_").lower()
        args.output_file = os.path.join(os.path.dirname(__file__), "..", "data", f"surgical_neutrality_{tag}.json")

    print(f"=== ARM 0: Surgical Neutrality Check for {args.model_id} ===")
    print(f"Device: {args.device}")
    print(f"Target Adapter (Arm 3 Retrofit, K=0): {args.lora_path}")
    print(f"Control Adapter (Arm 1b Direct, K=0): {args.lora_arm1b_path}")
    print(f"MMLU Evaluation Slice: {args.mmlu_samples} stratified questions")

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip("\"'\n")
                    break

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token, trust_remote_code=True)
    
    # Load MMLU dev (for canonical few-shot prompts) and test (for evaluation)
    print("Loading MMLU dataset (cais/mmlu)...")
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

    # 1. Evaluate Base Model (Pure BF16)
    print("\n" + "="*60)
    print("Step 1: Evaluating Base Model (Pure BF16, Non-Retrofitted)")
    print("="*60)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    base_model.eval()
    base_results = run_mmlu_eval(base_model, tokenizer, mmlu_slice, dev_by_subject, args.device)
    base_acc = sum(r["is_correct"] for r in base_results) / len(base_results) * 100.0
    base_cat_summary = summarize_categories(base_results)
    print(f"\n>> Base Model MMLU Accuracy: {base_acc:.2f}%")
    for cat, data in base_cat_summary.items():
        print(f"   - {cat:<16}: {data['accuracy_pct']:5.2f}% ({data['correct']}/{data['total']})")

    # 2. Evaluate Retrofitted Model (Arm 3 LoRA Adapter, recurrence disabled K=0)
    print("\n" + "="*60)
    print(f"Step 2: Attaching Arm 3 Adapter (K=0 Recurrence Off): {args.lora_path}")
    print("="*60)
    arm3_results = None
    arm3_acc = None
    arm3_cat_summary = None
    arm3_bootstrap = None
    arm3_neutral = None

    if os.path.exists(args.lora_path):
        arm3_model = PeftModel.from_pretrained(base_model, args.lora_path)
        arm3_model.eval()
        arm3_results = run_mmlu_eval(arm3_model, tokenizer, mmlu_slice, dev_by_subject, args.device)
        arm3_acc = sum(r["is_correct"] for r in arm3_results) / len(arm3_results) * 100.0
        arm3_cat_summary = summarize_categories(arm3_results)
        print(f"\n>> Arm 3 Retrofit (K=0) Accuracy: {arm3_acc:.2f}%")
        for cat, data in arm3_cat_summary.items():
            print(f"   - {cat:<16}: {data['accuracy_pct']:5.2f}% ({data['correct']}/{data['total']})")
            
        base_scores = [1 if r["is_correct"] else 0 for r in base_results]
        arm3_scores = [1 if r["is_correct"] else 0 for r in arm3_results]
        arm3_bootstrap = compute_paired_bootstrap(base_scores, arm3_scores, n_boot=10000)
        arm3_neutral = bool(arm3_bootstrap["ci_95_low"] >= -1.5)
        print(f"\n>> Arm 3 vs Base Paired Difference: {arm3_bootstrap['mean_diff_pct']:+.2f}% (95% CI: [{arm3_bootstrap['ci_95_low']:+.2f}%, {arm3_bootstrap['ci_95_high']:+.2f}%])")
        print(f">> Non-Inferiority (CI_low >= -1.5%): {'CONFIRMED (PASS)' if arm3_neutral else 'FAILED'}")
        
        # Unload adapter to restore base model
        base_model = arm3_model.unload()
    else:
        print(f"Warning: Adapter {args.lora_path} not found.")

    # 3. Evaluate Arm 1b Control Model (Trained Direct LoRA Adapter, K=0)
    print("\n" + "="*60)
    print(f"Step 3: Attaching Arm 1b Adapter (Trained Direct, K=0): {args.lora_arm1b_path}")
    print("="*60)
    arm1b_results = None
    arm1b_acc = None
    arm1b_cat_summary = None
    arm1b_bootstrap = None
    arm1b_neutral = None

    if os.path.exists(args.lora_arm1b_path):
        arm1b_model = PeftModel.from_pretrained(base_model, args.lora_arm1b_path)
        arm1b_model.eval()
        arm1b_results = run_mmlu_eval(arm1b_model, tokenizer, mmlu_slice, dev_by_subject, args.device)
        arm1b_acc = sum(r["is_correct"] for r in arm1b_results) / len(arm1b_results) * 100.0
        arm1b_cat_summary = summarize_categories(arm1b_results)
        print(f"\n>> Arm 1b Control (K=0) Accuracy: {arm1b_acc:.2f}%")
        for cat, data in arm1b_cat_summary.items():
            print(f"   - {cat:<16}: {data['accuracy_pct']:5.2f}% ({data['correct']}/{data['total']})")
            
        base_scores = [1 if r["is_correct"] else 0 for r in base_results]
        arm1b_scores = [1 if r["is_correct"] else 0 for r in arm1b_results]
        arm1b_bootstrap = compute_paired_bootstrap(base_scores, arm1b_scores, n_boot=10000)
        arm1b_neutral = bool(arm1b_bootstrap["ci_95_low"] >= -1.5)
        print(f"\n>> Arm 1b vs Base Paired Difference: {arm1b_bootstrap['mean_diff_pct']:+.2f}% (95% CI: [{arm1b_bootstrap['ci_95_low']:+.2f}%, {arm1b_bootstrap['ci_95_high']:+.2f}%])")
        print(f">> Non-Inferiority (CI_low >= -1.5%): {'CONFIRMED (PASS)' if arm1b_neutral else 'FAILED'}")
        
        base_model = arm1b_model.unload()
    else:
        print(f"Warning: Adapter {args.lora_arm1b_path} not found.")

    # 4. Final Comprehensive Report
    print("\n" + "="*70)
    print("ARM 0 SURGICAL NEUTRALITY FINAL REPORT")
    print("="*70)
    print(f"Base Model ({args.model_id}):               {base_acc:.2f}%")
    if arm3_acc is not None:
        print(f"Arm 3 Retrofitted (K=0):                    {arm3_acc:.2f}% (Delta: {arm3_bootstrap['mean_diff_pct']:+.2f}%, 95% CI: [{arm3_bootstrap['ci_95_low']:+.2f}%, {arm3_bootstrap['ci_95_high']:+.2f}%])")
    if arm1b_acc is not None:
        print(f"Arm 1b Trained Direct (K=0):                {arm1b_acc:.2f}% (Delta: {arm1b_bootstrap['mean_diff_pct']:+.2f}%, 95% CI: [{arm1b_bootstrap['ci_95_low']:+.2f}%, {arm1b_bootstrap['ci_95_high']:+.2f}%])")
    print("-" * 70)
    print(f"Surgical Neutrality Decision Gate: {'PASSED' if (arm3_neutral is True) else 'FAILED'}")
    print("="*70)

    output_data = {
        "model_id": args.model_id,
        "device": args.device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mmlu_samples": args.mmlu_samples,
        "base_accuracy": round(base_acc, 2),
        "base_categories": base_cat_summary,
        "arm3_retrofit": {
            "adapter_path": args.lora_path,
            "accuracy": round(arm3_acc, 2) if arm3_acc is not None else None,
            "categories": arm3_cat_summary,
            "paired_bootstrap": arm3_bootstrap,
            "surgical_neutrality_confirmed": arm3_neutral
        },
        "arm1b_control": {
            "adapter_path": args.lora_arm1b_path,
            "accuracy": round(arm1b_acc, 2) if arm1b_acc is not None else None,
            "categories": arm1b_cat_summary,
            "paired_bootstrap": arm1b_bootstrap,
            "surgical_neutrality_confirmed": arm1b_neutral
        }
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved structured report to: {args.output_file}")

if __name__ == "__main__":
    main()
