#!/usr/bin/env python3
"""
studies/dynamic_registers/scripts/06_register_corruption_test.py
Phase 5: Mechanistic Intervention Audit & Controlled Register Corruption Test.
Investigates whether the model's downstream answer is causally bound to the symbolic
content in the dynamic registers (<|reg|>...<|/reg|>).

Protocols:
1. Control 0 (Null Patch Self-Identity):
   Feed recipient problem + authentic unperturbed registers -> verify output answer is identical (DeltaSteer_null = 0.0%).
2. Control B (Counterfactual Donor Register Corruption):
   Swap register values in R with values from a donor problem D (with gold_D != gold_R).
   Measure if the model's final answer shifts to gold_D.
3. Chance Baseline:
   Measure how often gold_D appears by chance in R's unperturbed generation.
4. Net Steering Delta:
   DeltaSteer = P(donor_target | corrupted) - P(donor_target | chance).
5. Calibration Reference:
   Compare against measured discrete-CoT steering:
   - Full-chain discrete replacement: +21.74%
   - Step-1 replacement: +8.70%
   - Latent vectors (K=6): +0.00%
"""

import os
import sys
import re
import json
import time
import asyncio
import argparse
import httpx
import numpy as np
from tqdm.asyncio import tqdm
from math_verify import parse, verify

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dynamic_registers.scripts.register_format import extract_registers, CANONICAL_HEADERS

def check_math_match(pred_text: str, gold_target: str) -> bool:
    if not pred_text or not gold_target:
        return False
    try:
        if "####" in gold_target:
            gold_target = gold_target.split("####")[-1].strip()
        if "\\boxed{" not in gold_target:
            gold_target = f"\\boxed{{{gold_target}}}"
        gold_p = parse(gold_target, parsing_timeout=None)
        pred_p = parse(pred_text, parsing_timeout=None)
        if gold_p and pred_p:
            return bool(verify(gold_p, pred_p))
    except Exception:
        pass
    return False

def extract_math_boxed(text: str) -> str:
    if not text or "\\boxed{" not in text:
        return ""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    end = start
    while end < len(text) and depth > 0:
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1
    if depth == 0:
        res = text[start:end-1].strip()
        if "=" in res:
            res = res.split("=")[-1].strip()
        return res.rstrip(".")
    return ""

async def query_sglang(client, server_url, prompt_text, max_tokens=1024, seed=42):
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.0, # Greedy for deterministic counterfactual isolation
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }
    }
    resp = await client.post(f"{server_url}/generate", json=payload, timeout=60.0)
    data = resp.json()
    return data.get("text", "").strip()

async def run_corruption_test(args):
    print("=" * 80)
    print("PHASE 5: CONTROLLED REGISTER CORRUPTION TEST")
    print(f"Model: {args.tag} | Server: {args.server_url}")
    print("=" * 80)

    with open(args.benchmark_file) as f:
        problems = json.load(f)

    # Filter to GSM8K and straightforward MATH problems with clear ground truth answers
    clean_problems = []
    for p in problems:
        sol = p.get("solution", "")
        if "####" in sol:
            ans = sol.split("####")[-1].strip()
            clean_problems.append({**p, "clean_ans": ans})
        elif "\\boxed{" in sol:
            ans = extract_math_boxed(sol)
            if ans and len(ans) < 20:
                clean_problems.append({**p, "clean_ans": ans})

    print(f"Selected {len(clean_problems)} valid benchmark problems.")
    eval_subset = clean_problems[:args.num_pairs]

    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    sem = asyncio.Semaphore(16)

    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(60.0)) as client:
        # Step 1: Generate unperturbed trajectories concurrently
        print("\nStep 1: Generating unperturbed register trajectories...")
        
        async def gen_unperturbed(p):
            async with sem:
                prompt = f"<|im_start|>user\n{p['question']}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                gen_text = await query_sglang(client, args.server_url, prompt, max_tokens=1500)
                
                # Split think and answer
                if "</think>" in gen_text:
                    think_part, ans_part = gen_text.split("</think>", 1)
                else:
                    think_part, ans_part = gen_text, ""
                
                pred_boxed = extract_math_boxed(ans_part) if ans_part else extract_math_boxed(gen_text)
                regs = extract_registers(think_part)
                return {
                    "problem": p,
                    "full_gen": gen_text,
                    "think_part": think_part,
                    "ans_part": ans_part,
                    "pred_boxed": pred_boxed,
                    "registers": regs
                }

        unperturbed_runs = await tqdm.gather(*[gen_unperturbed(p) for p in eval_subset], desc="Unperturbed generation")

        # Step 2: Form valid counterfactual donor pairs
        valid_pairs = []
        n = len(unperturbed_runs)
        for i in range(n):
            rec = unperturbed_runs[i]
            if not rec["registers"]:
                continue
            donor = None
            for offset in range(1, n):
                cand = unperturbed_runs[(i + offset) % n]
                if cand["registers"] and cand["problem"]["clean_ans"] != rec["problem"]["clean_ans"]:
                    donor = cand
                    break
            if donor:
                valid_pairs.append((rec, donor))

        print(f"Formed {len(valid_pairs)} valid counterfactual recipient-donor pairs.")

        # Step 3: Run Null Patch Control & Donor Corruption concurrently
        print("\nStep 2: Evaluating Null Patch Control and Donor Corruption...")

        async def run_trial(rec, donor):
            async with sem:
                rec_q = rec["problem"]["question"]
                gold_rec = rec["problem"]["clean_ans"]
                gold_donor = donor["problem"]["clean_ans"]
                base_prompt = f"<|im_start|>user\n{rec_q}<|im_end|>\n<|im_start|>assistant\n<think>\n"

                rec_unperturbed_pred = rec["pred_boxed"]
                is_chance = check_math_match(rec_unperturbed_pred, gold_donor)

                # 1. Null Patch: Feed exact unperturbed think part
                null_prompt = f"{base_prompt}{rec['think_part'].rstrip()}\n</think>\n\n"
                null_gen = await query_sglang(client, args.server_url, null_prompt, max_tokens=1024)
                null_pred = extract_math_boxed(null_gen)
                null_match = (null_pred == rec_unperturbed_pred or check_math_match(null_pred, rec_unperturbed_pred))

                # 2. Corrupt Registers: Inject donor's registers into recipient's think structure
                donor_final_val = donor["registers"][-1]["value"]
                donor_final_key = donor["registers"][-1]["key"]
                
                corrupted_think = rec["think_part"]
                last_reg_match = list(re.finditer(r"<\|reg\|>(.*?)<\|/reg\|>", corrupted_think))
                if last_reg_match:
                    m = last_reg_match[-1]
                    corrupted_think = corrupted_think[:m.start()] + f"<|reg|>{donor_final_key} = {donor_final_val}<|/reg|>" + corrupted_think[m.end():]

                corrupted_prompt = f"{base_prompt}{corrupted_think.rstrip()}\n</think>\n\n"
                corrupted_gen = await query_sglang(client, args.server_url, corrupted_prompt, max_tokens=1024)
                corrupted_pred = extract_math_boxed(corrupted_gen)

                is_steered = check_math_match(corrupted_pred, gold_donor)

                return {
                    "recipient_id": rec["problem"]["id"],
                    "donor_id": donor["problem"]["id"],
                    "gold_recipient": gold_rec,
                    "gold_donor": gold_donor,
                    "rec_unperturbed_pred": rec_unperturbed_pred,
                    "null_pred": null_pred,
                    "corrupted_pred": corrupted_pred,
                    "null_match": null_match,
                    "is_chance": is_chance,
                    "is_steered": is_steered
                }

        detailed_trials = await tqdm.gather(*[run_trial(rec, donor) for rec, donor in valid_pairs], desc="Intervention trials")

    null_identity_matches = sum(1 for t in detailed_trials if t["null_match"])
    chance_matches = sum(1 for t in detailed_trials if t["is_chance"])
    donor_steered_matches = sum(1 for t in detailed_trials if t["is_steered"])

    total_trials = len(detailed_trials)
    p_null = (null_identity_matches / total_trials) * 100 if total_trials else 0.0
    p_chance = (chance_matches / total_trials) * 100 if total_trials else 0.0
    p_steered = (donor_steered_matches / total_trials) * 100 if total_trials else 0.0
    delta_steer = p_steered - p_chance

    print("\n" + "=" * 80)
    print("PHASE 5 RESULTS: CONTROLLED REGISTER CORRUPTION TEST")
    print(f"Total Valid Trials:             {total_trials}")
    print(f"Control 0 (Null Patch Identity): {p_null:.2f}% ({null_identity_matches}/{total_trials}) [{'PASS' if p_null >= 90.0 else 'FAIL'}]")
    print(f"Chance Baseline P(chance):      {p_chance:.2f}% ({chance_matches}/{total_trials})")
    print(f"Donor Steered P(steered):       {p_steered:.2f}% ({donor_steered_matches}/{total_trials})")
    print(f"Net Steering Delta (DeltaSteer): {delta_steer:+.2f}%")
    print("-" * 80)
    print("CALIBRATION BENCHMARK COMPARISON:")
    print(f"  * Full-Chain Discrete CoT:     +21.74% (measured upper benchmark)")
    print(f"  * Step-1 Discrete CoT:         +8.70%")
    print(f"  * Latent Vectors (K=6):        +0.00% (inert floor)")
    print(f"  * Dynamic Discrete Registers:  {delta_steer:+.2f}%")
    print("=" * 80)

    results = {
        "tag": args.tag,
        "total_trials": total_trials,
        "null_patch_identity_pct": p_null,
        "p_chance_pct": p_chance,
        "p_steered_pct": p_steered,
        "delta_steer_pct": delta_steer,
        "calibration_benchmarks": {
            "full_chain_discrete_cot": 21.74,
            "step1_discrete_cot": 8.70,
            "latent_vectors_k6": 0.00
        },
        "trials": detailed_trials
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--tag", type=str, default="arm1_dynamic_registers")
    parser.add_argument("--benchmark_file", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--num_pairs", type=int, default=100)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    asyncio.run(run_corruption_test(args))

if __name__ == "__main__":
    main()
