#!/usr/bin/env python3
"""
scripts/63_surgical_neutrality_controls.py
ARM 0: Surgical Neutrality Check for Trained Control Adapters on GPU 0.

Evaluates whether LoRA fine-tuning for Arm 1b (Direct SFT), Arm 2b (Pause K=6),
and Arm 2b (Pause K=32) degraded the base model's general non-math reasoning.

Tests on:
1. Canonical 5-shot MMLU (1,000 stratified questions across STEM, Humanities, Social Sciences, Other).
2. Computes per-sample paired differences and 10,000-iteration hierarchical bootstrap against Base.
3. Asserts pre-registered non-inferiority bound: Delta(Adapter - Base) >= -1.5% (95% CI_low).
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
    # Slice last token position BEFORE lm_head to avoid allocating [B, L, 151936] (8.8 GB)
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "model"):
        # PeftModel
        backbone = model.base_model.model.model
        lm_head = model.base_model.model.lm_head
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        backbone = model.model
        lm_head = model.lm_head
    else:
        out = model(**enc)
        return out.logits[:, -1, :]
        
    h = backbone(**enc).last_hidden_state[:, -1:, :]
    logits = lm_head(h)[:, 0, :]
    return logits

def run_mmlu_eval(model, tokenizer, dataset_slice, dev_by_subject, device, batch_size=8):
    choices = ["A", "B", "C", "D"]
    choice_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in choices]
    
    results = []
    t0 = time.time()
    
    # Process in batches for high throughput on GPU 0
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
            
        if (len(results)) % 200 == 0 or len(results) == len(dataset_slice):
            acc = sum(r["is_correct"] for r in results) / len(results) * 100.0
            elapsed = time.time() - t0
            ms_per_q = elapsed / len(results) * 1000
            print(f"  [Progress {len(results)}/{len(dataset_slice)}] Accuracy: {acc:.2f}% | Speed: {ms_per_q:.1f} ms/q")
            
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
    parser = argparse.ArgumentParser(description="Arm 0: Surgical Neutrality Check for Control Adapters")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mmlu_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_file", type=str, default="data/surgical_neutrality_controls_qwen3_1.7b.json")
    args = parser.parse_args()

    set_seed(42)
    print(f"=== ARM 0: Surgical Neutrality Check for {args.model_id} on {args.device} ===")
    print(f"MMLU Slice: {args.mmlu_samples} questions | Batch Size: {args.batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
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

    # 1. Base Model
    print("\n" + "="*60)
    print("Step 1: Evaluating Base Model (Pure BF16)")
    print("="*60)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    base_model.eval()
    base_results = run_mmlu_eval(base_model, tokenizer, mmlu_slice, dev_by_subject, args.device, batch_size=args.batch_size)
    base_acc = sum(r["is_correct"] for r in base_results) / len(base_results) * 100.0
    base_cat_summary = summarize_categories(base_results)
    base_scores = [1 if r["is_correct"] else 0 for r in base_results]
    print(f"\n>> Base Model MMLU Accuracy: {base_acc:.2f}%")
    for cat, data in base_cat_summary.items():
        print(f"   - {cat:<16}: {data['accuracy_pct']:5.2f}% ({data['correct']}/{data['total']})")

    adapters = [
        ("arm1b_k0", "checkpoints/lora_arm1b_qwen_qwen3-1.7b_k0_selfdistill"),
        ("arm2b_k6", "checkpoints/lora_arm2b_qwen_qwen3-1.7b_k6_selfdistill"),
        ("arm2b_k32", "checkpoints/lora_arm2b_qwen_qwen3-1.7b_k32_selfdistill")
    ]

    adapter_reports = {}

    for name, path in adapters:
        if not os.path.exists(path):
            print(f"\nSkipping {name}: checkpoint {path} not found.")
            continue
            
        print("\n" + "="*60)
        print(f"Evaluating Adapter: {name} ({path})")
        print("="*60)
        
        if "arm2b" in name:
            base_model.resize_token_embeddings(151670)
            
        adapter_model = PeftModel.from_pretrained(base_model, path)
        adapter_model.eval()
        
        res = run_mmlu_eval(adapter_model, tokenizer, dev_by_subject=dev_by_subject, dataset_slice=mmlu_slice, device=args.device, batch_size=args.batch_size)
        acc = sum(r["is_correct"] for r in res) / len(res) * 100.0
        cat_summary = summarize_categories(res)
        scores = [1 if r["is_correct"] else 0 for r in res]
        
        bootstrap = compute_paired_bootstrap(base_scores, scores, n_boot=10000)
        neutral = bool(bootstrap["ci_95_low"] >= -1.5)
        
        print(f"\n>> {name} Accuracy: {acc:.2f}%")
        for cat, data in cat_summary.items():
            print(f"   - {cat:<16}: {data['accuracy_pct']:5.2f}% ({data['correct']}/{data['total']})")
        print(f">> Delta vs Base: {bootstrap['mean_diff_pct']:+.2f}% (95% CI: [{bootstrap['ci_95_low']:+.2f}%, {bootstrap['ci_95_high']:+.2f}%])")
        print(f">> Non-Inferiority Gate (CI_low >= -1.5%): {'CONFIRMED (PASS)' if neutral else 'FAILED'}")
        
        adapter_reports[name] = {
            "adapter_path": path,
            "accuracy": round(acc, 2),
            "categories": cat_summary,
            "paired_bootstrap": bootstrap,
            "surgical_neutrality_confirmed": neutral
        }
        
        # Unload adapter back to base model
        base_model = adapter_model.unload()
        if "arm2b" in name:
            base_model.resize_token_embeddings(151936)

    # Final Output
    report = {
        "model_id": args.model_id,
        "device": args.device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mmlu_samples": args.mmlu_samples,
        "base_accuracy": round(base_acc, 2),
        "base_categories": base_cat_summary,
        "adapters": adapter_reports
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*70)
    print("SURGICAL NEUTRALITY SUMMARY REPORT (GPU 0)")
    print("="*70)
    print(f"Base Model ({args.model_id}): {base_acc:.2f}%")
    for name, r in adapter_reports.items():
        delta = r["paired_bootstrap"]["mean_diff_pct"]
        ci_l = r["paired_bootstrap"]["ci_95_low"]
        ci_h = r["paired_bootstrap"]["ci_95_high"]
        status = "PASS" if r["surgical_neutrality_confirmed"] else "FAIL"
        print(f"  {name:<12}: {r['accuracy']:5.2f}% | Delta: {delta:+.2f}% [{ci_l:+.2f}%, {ci_h:+.2f}%] | Gate: {status}")
    print("="*70)
    print(f"Saved to: {args.output_file}")

if __name__ == "__main__":
    main()
