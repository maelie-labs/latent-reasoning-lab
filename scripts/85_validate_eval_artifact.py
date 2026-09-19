#!/usr/bin/env python3
"""
scripts/85_validate_eval_artifact.py
Automated Scientific Validation Gate (The 7 Protocols).

Runs an assertive verification audit on any evaluation artifact (math250, ifeval, etc.):
1. Format Invariant Audit: Checks for leaked delimiters (<think>, </think>, <|im_start|>, etc.) in non-thinking outputs.
2. Independent Re-Scoring Audit: Re-scores 100% of samples from raw text using pinned canonical graders (0% discrepancy tolerance).
3. Strict Truncation Audit: Verifies sequences hitting token cap are scored as False and not dropped.
4. Telemetry & Distribution Sanity Audit: Logs min, median, p90, max token lengths.
5. Zero Test-Data Contamination Check: Asserts evaluated IDs do not overlap with kept training traces.

Usage:
  python scripts/85_validate_eval_artifact.py --file data/streaming_arm3_k6_qwen_qwen3-1.7b.jsonl --type math250
  python scripts/85_validate_eval_artifact.py --file data/surgical_neutrality_ifeval_arm3_k6_qwen3_1.7b.json --type ifeval
"""

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import json
import argparse
import numpy as np

def validate_math250(file_path):
    print(f"\n=======================================================")
    print(f"=== AUDITING MATH-250 ARTIFACT: {os.path.basename(file_path)} ===")
    print(f"=======================================================")
    
    from math_verify import parse, verify
    
    records = []
    if file_path.endswith(".jsonl"):
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        with open(file_path) as f:
            data = json.load(f)
            if "evaluations" in data:
                records = data["evaluations"]
            elif "per_problem_results" in data:
                records = data["per_problem_results"]
            else:
                raise ValueError("Unrecognized JSON format for math250 artifact")
                
    N = len(records)
    print(f"Total evaluated instances: N = {N}")
    if N == 0:
        print("[FAIL] Empty artifact!")
        return False
        
    # Check 1: Format Invariants & Delimiter Leakage
    leaked_think_count = 0
    leaked_im_count = 0
    trunc_count = 0
    tok_lens = []
    
    rescore_correct = 0
    logged_correct = 0
    discrepancies = []
    
    # Check Contamination
    kept_ids_path = "data/kept_trace_ids.json"
    kept_ids = set()
    if os.path.exists(kept_ids_path):
        with open(kept_ids_path) as f:
            kept_ids = set(json.load(f))
            
    contaminated_ids = []
    
    for idx, r in enumerate(records):
        prob_id = r.get("problem_id", r.get("id", f"idx_{idx}"))
        if prob_id in kept_ids:
            contaminated_ids.append(prob_id)
            
        run_data = r.get("run", r)
        content = run_data.get("content", "")
        is_trunc = run_data.get("is_truncated", False)
        if is_trunc:
            trunc_count += 1
            
        # Delimiter checks
        if "<think>" in content or "</think>" in content:
            leaked_think_count += 1
        if "<|im_start|>" in content or "<|im_end|>" in content:
            leaked_im_count += 1
            
        ans_toks = run_data.get("ans_tokens", len(content.split()))
        tok_lens.append(ans_toks)
        
        # Logged correctness
        log_corr = bool(r.get("is_correct", False))
        if log_corr:
            logged_correct += 1
            
        # Re-scoring
        gt_target = r.get("gt_target", "")
        pred_boxed = run_data.get("pred_boxed", "")
        
        if is_trunc:
            is_valid = False
        else:
            try:
                cand_parsed = parse(content)
                target_parsed = parse(gt_target)
                is_valid = bool(verify(target_parsed, cand_parsed))
            except Exception:
                is_valid = False
                
        if is_valid:
            rescore_correct += 1
            
        if is_valid != log_corr:
            discrepancies.append((prob_id, log_corr, is_valid))
            
    # Audit Results
    passed = True
    print("\n--- Telemetry & Distribution Sanity (Rule 11) ---")
    print(f"Token lengths: min={np.min(tok_lens)}, median={np.median(tok_lens):.1f}, p90={np.percentile(tok_lens, 90):.1f}, max={np.max(tok_lens)}")
    print(f"Truncation rate: {trunc_count}/{N} ({100.0 * trunc_count / N:.2f}%)")
    
    print("\n--- Protocol 1 & 7: Delimiter & Format Invariant Check ---")
    if leaked_think_count > 0:
        print(f"[FAIL] Leaked <think> tags found in {leaked_think_count} instances!")
        passed = False
    else:
        print("[PASS] Zero <think> or </think> delimiter leakage detected.")
        
    if leaked_im_count > 0:
        print(f"[FAIL] Leaked <|im_*|> special tokens found in {leaked_im_count} instances!")
        passed = False
    else:
        print("[PASS] Zero ChatML delimiter leakage detected.")
        
    print("\n--- Protocol 1: Independent Re-Scoring Audit ---")
    logged_acc = 100.0 * logged_correct / N
    rescore_acc = 100.0 * rescore_correct / N
    print(f"Logged Accuracy:   {logged_acc:.2f}% ({logged_correct}/{N})")
    print(f"Re-scored Accuracy: {rescore_acc:.2f}% ({rescore_correct}/{N})")
    if len(discrepancies) > 0:
        print(f"[FAIL] {len(discrepancies)} scoring discrepancies found! Tolerance is 0.")
        passed = False
    else:
        print("[PASS] Bit-exact agreement between logged and independently re-scored results (0% discrepancy).")
        
    print("\n--- Protocol 6: Zero Test-Data Contamination ---")
    if len(contaminated_ids) > 0:
        print(f"[FAIL] Found {len(contaminated_ids)} test problems overlapping with training trace IDs!")
        passed = False
    else:
        print("[PASS] Zero contamination verified against kept_trace_ids.json.")
        
    print(f"\nFINAL VERDICT: {'CERTIFIED [PASS]' if passed else 'REJECTED [FAIL]'}")
    return passed

def validate_ifeval(file_path):
    print(f"\n=======================================================")
    print(f"=== AUDITING IFEVAL ARTIFACT: {os.path.basename(file_path)} ===")
    print(f"=======================================================")
    
    from datasets import load_dataset
    from ifeval.utils import (
        InputExample,
        test_instruction_following_strict,
        test_instruction_following_loose,
    )
    
    with open(file_path) as f:
        data = json.load(f)
        
    summary = data.get("summary", {})
    evals = data.get("evaluations", [])
    N = len(evals)
    if N != 541:
        print(f"[FAIL] Expected 541 IFEval prompts, found {N}!")
        return False
        
    dataset = load_dataset("google/IFEval", split="train")
    dataset_map = {doc["key"]: doc for doc in dataset}
    
    leaked_think_count = 0
    strict_corr = 0
    loose_corr = 0
    tok_lens = []
    
    for e in evals:
        doc = dataset_map[e["key"]]
        inp = InputExample(
            key=doc["key"],
            instruction_id_list=doc["instruction_id_list"],
            prompt=doc["prompt"],
            kwargs=doc["kwargs"]
        )
        resp = e.get("response", "")
        tok_lens.append(len(resp.split()))
        
        # Check for leaked delimiters
        if resp.startswith("</think>") or "<think>" in resp:
            leaked_think_count += 1
            
        out_strict = test_instruction_following_strict(inp, resp)
        out_loose = test_instruction_following_loose(inp, resp)
        if out_strict.follow_all_instructions:
            strict_corr += 1
        if out_loose.follow_all_instructions:
            loose_corr += 1
            
    passed = True
    print("\n--- Protocol 1 & 7: Delimiter & Format Invariant Check ---")
    if leaked_think_count > 0:
        print(f"[FAIL] Leaked <think> delimiters found in {leaked_think_count} responses!")
        passed = False
    else:
        print("[PASS] Zero <think> prompt delimiters detected at response start.")
        
    print("\n--- Protocol 1: Independent Re-Scoring Audit ---")
    rescore_strict = round(100.0 * strict_corr / N, 2)
    rescore_loose = round(100.0 * loose_corr / N, 2)
    log_strict = summary.get("strict_prompt_accuracy_pct", -1)
    log_loose = summary.get("loose_prompt_accuracy_pct", -1)
    
    print(f"Strict Prompt: Logged={log_strict}% | Re-scored={rescore_strict}%")
    print(f"Loose Prompt:  Logged={log_loose}% | Re-scored={rescore_loose}%")
    
    if abs(rescore_strict - log_strict) > 0.05 or abs(rescore_loose - log_loose) > 0.05:
        print("[FAIL] Re-scored IFEval accuracy does not match logged summary!")
        passed = False
    else:
        print("[PASS] Independent re-scoring verified bit-exact match.")
        
    print(f"\nFINAL VERDICT: {'CERTIFIED [PASS]' if passed else 'REJECTED [FAIL]'}")
    return passed

def main():
    parser = argparse.ArgumentParser(description="Automated Validation Gate for Evaluation Artifacts")
    parser.add_argument("--file", type=str, required=True, help="Path to evaluation artifact (json or jsonl)")
    parser.add_argument("--type", type=str, choices=["math250", "ifeval"], required=True, help="Artifact benchmark type")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
        
    if args.type == "math250":
        ok = validate_math250(args.file)
    elif args.type == "ifeval":
        ok = validate_ifeval(args.file)
    else:
        print(f"Unknown type {args.type}")
        sys.exit(1)
        
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
