#!/usr/bin/env python3
"""
scripts/97_eval_arm3_register_ladder.py
Unified Benchmark Evaluation for Arm 3 Register-Bundle Ladder Architecture.

Evaluates trained adapter across the 250-suite benchmark (data/benchmark_suite_250.json)
across 4 deterministic seeds: SEEDS = [42, 123, 456, 789] (N=1,000 total queries).

Enforces:
- Gate 0 Frozen Specification Table (temp=0.7, top_p=0.80, top_k=20, pp=1.5, ans_cap=8192)
- High-Throughput Batched PyTorch Execution (B=8/16)
- Grader Invariant: Canonical math_verify with symbolic equivalence
- Paired Hierarchical Bootstrapping against Arm 1b reference (69.17%)
"""

import os
import re
import json
import time
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from peft import PeftModel
from math_verify import parse, verify

SEEDS = [42, 123, 456, 789]
CALIBRATED_ALPHAS = {
    "qwen/qwen3-1.7b": 0.011440,
    "qwen/qwen3.5-2b": 0.009800,
    "qwen/qwen3-4b": 0.007670,
}

CANONICAL_HEADERS = [
    "[R1: Identify givens, constraints, and target variable]\n",
    "[R2: Compute intermediate operations and verify relations]\n",
    "[R3: Execute final deduction and verify constraints]\n"
]

class PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float, prompt_len: int):
        self.penalty = penalty
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0 or input_ids.shape[1] <= self.prompt_len:
            return scores
        for b in range(input_ids.shape[0]):
            gen_ids = input_ids[b, self.prompt_len:]
            unique_ids = torch.unique(gen_ids)
            scores[b, unique_ids] -= self.penalty
        return scores

def check_math_correct(pred_text: str, gold_text: str) -> bool:
    try:
        if "####" in gold_text:
            ans_part = gold_text.split("####")[-1].strip()
            gold_target = f"\\boxed{{{ans_part}}}"
        elif "\\boxed{" not in gold_text:
            gold_target = f"\\boxed{{{gold_text.strip()}}}"
        else:
            gold_target = gold_text
        gold_parsed = parse(gold_target, parsing_timeout=None)
        pred_parsed = parse(pred_text, parsing_timeout=None)
        if gold_parsed and pred_parsed:
            return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        pass
    return False

def run_paired_bootstrap(scores_a, scores_b, n_boot=10000, seed=42):
    np.random.seed(seed)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(scores_a)
    deltas = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        deltas.append(np.mean(scores_a[idx]) - np.mean(scores_b[idx]))
    deltas = np.array(deltas) * 100.0
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    obs_delta = float(np.mean(scores_a) - np.mean(scores_b)) * 100.0
    p_val = float(np.mean(deltas <= 0.0)) if obs_delta > 0 else float(np.mean(deltas >= 0.0))
    return {
        "observed_delta": obs_delta,
        "ci_95": [ci_low, ci_high],
        "p_value": p_val * 2.0
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Arm 3 Register-Bundle Ladder")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark_file", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_rungs", type=int, default=3)
    parser.add_argument("--k_per_bundle", type=int, default=8)
    parser.add_argument("--scale_factor", type=float, default=None)
    parser.add_argument("--max_ans_tokens", type=int, default=8192)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--tag", type=str, default="register_ladder_prod")
    args = parser.parse_args()

    scale_factor = args.scale_factor or CALIBRATED_ALPHAS.get(args.model_id.lower(), 0.011440)
    os.makedirs(args.output_dir, exist_ok=True)

    out_jsonl = os.path.join(args.output_dir, f"streaming_{args.tag}.jsonl")
    out_results = os.path.join(args.output_dir, f"eval_results_{args.tag}.json")

    print("=" * 80)
    print("EVALUATION: ARM 3 REGISTER-BUNDLE LADDER")
    print(f"Model ID     : {args.model_id}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Device       : {args.device} | Batch Size: {args.batch_size}")
    print(f"Architecture : {args.num_rungs} rungs x {args.k_per_bundle} latents = {args.num_rungs * args.k_per_bundle} total latents")
    print(f"Sampling     : Temp=0.7, TopP=0.80, TopK=20, PP=1.5, MaxAns={args.max_ans_tokens}")
    print("=" * 80)

    with open(args.benchmark_file) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} benchmark problems.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    completed_keys = set()
    all_evaluations = []
    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r") as f_in:
            for line in f_in:
                if line.strip():
                    r = json.loads(line)
                    all_evaluations.append(r)
                    completed_keys.add((r["id"], r["seed"]))
        print(f"Resuming evaluation: found {len(all_evaluations)} pre-existing evaluations in {out_jsonl}.")

    start_time = time.time()

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        torch.manual_seed(seed)

        for i in range(0, len(problems), args.batch_size):
            chunk = problems[i:i + args.batch_size]
            b_items = [p for p in chunk if (p["id"], seed) not in completed_keys]
            if not b_items:
                continue
            B = len(b_items)

            # Prompts ending with <think>\n
            b_prompts = [
                f"<|im_start|>user\n{p['question']}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                for p in b_items
            ]
            enc = tokenizer(b_prompts, return_tensors="pt", padding=True).to(args.device)

            with torch.inference_mode():
                pos_prefill = enc.attention_mask.long().cumsum(-1) - 1
                pos_prefill.masked_fill_(enc.attention_mask == 0, 0)
                out = model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    position_ids=pos_prefill,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_kv = out.past_key_values
                curr_latent = out.hidden_states[-1][:, -1:, :]
                curr_mask = enc.attention_mask
                seq_lens = enc.attention_mask.sum(dim=-1)
                curr_pos_end = seq_lens.clone()

                # Interleaved Register-Bundle Loop
                for r in range(args.num_rungs):
                    hdr_text = CANONICAL_HEADERS[r]
                    enc_hdr = tokenizer(hdr_text, return_tensors="pt", add_special_tokens=False).to(args.device)
                    hdr_len = enc_hdr.input_ids.shape[1]
                    hdr_ids = enc_hdr.input_ids.expand(B, -1)
                    hdr_mask = torch.ones((B, hdr_len), dtype=torch.long, device=args.device)
                    curr_mask = torch.cat([curr_mask, hdr_mask], dim=1)

                    pos_hdr = torch.stack([
                        torch.arange(hdr_len, device=args.device) + curr_pos_end[b]
                        for b in range(B)
                    ], dim=0)
                    curr_pos_end += hdr_len

                    hdr_out = model(
                        input_ids=hdr_ids,
                        attention_mask=curr_mask,
                        position_ids=pos_hdr,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_hidden_states=True
                    )
                    past_kv = hdr_out.past_key_values
                    curr_latent = hdr_out.hidden_states[-1][:, -1:, :]

                    # Unroll K latents for this bundle
                    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                        for k in range(args.k_per_bundle):
                            step_pos = curr_pos_end.unsqueeze(1)
                            curr_mask = torch.cat([curr_mask, torch.ones((B, 1), dtype=torch.long, device=args.device)], dim=1)
                            scaled_latent = curr_latent * scale_factor
                            step_out = model(
                                inputs_embeds=scaled_latent,
                                attention_mask=curr_mask,
                                position_ids=step_pos,
                                past_key_values=past_kv,
                                use_cache=True,
                                output_hidden_states=True
                            )
                            past_kv = step_out.past_key_values
                            curr_latent = step_out.hidden_states[-1][:, -1:, :]
                            curr_pos_end += 1

                # Transition Delimiter \n</think>\n\n
                trans_tokens = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
                trans_len = len(trans_tokens)
                trans_ids = torch.tensor([trans_tokens] * B, dtype=torch.long, device=args.device)
                curr_mask = torch.cat([curr_mask, torch.ones((B, trans_len), dtype=torch.long, device=args.device)], dim=1)
                pos_trans = torch.stack([
                    torch.arange(trans_len, device=args.device) + curr_pos_end[b]
                    for b in range(B)
                ], dim=0)

                pp_proc = PresencePenaltyLogitsProcessor(1.5, trans_len)
                gen_out = model.generate(
                    input_ids=trans_ids,
                    attention_mask=curr_mask,
                    position_ids=pos_trans,
                    past_key_values=past_kv,
                    max_new_tokens=args.max_ans_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.80,
                    top_k=20,
                    logits_processor=LogitsProcessorList([pp_proc]),
                    pad_token_id=tokenizer.pad_token_id
                )

            for b_idx in range(B):
                gen_toks = gen_out[b_idx][trans_len:].tolist()
                is_trunc = bool(len(gen_toks) >= args.max_ans_tokens and tokenizer.eos_token_id not in gen_toks)
                gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
                gold = b_items[b_idx].get("solution") or b_items[b_idx].get("answer", "")
                is_correct = False if is_trunc else check_math_correct(gen_text, gold)
                rec = {
                    "id": b_items[b_idx]["id"],
                    "benchmark": b_items[b_idx]["benchmark"],
                    "stratum": b_items[b_idx].get("stratum", ""),
                    "seed": seed,
                    "is_correct": is_correct,
                    "is_truncated": is_trunc,
                    "tokens": len(gen_toks),
                    "content": gen_text
                }
                completed_keys.add((rec["id"], rec["seed"]))
                all_evaluations.append(rec)
                with open(out_jsonl, "a") as f_out:
                    f_out.write(json.dumps(rec) + "\n")

            running_acc = float(np.mean([r["is_correct"] for r in all_evaluations])) * 100.0
            print(f"[Seed {seed} | {len(all_evaluations):4d}/{len(problems)*len(SEEDS)}] Running Acc: {running_acc:.2f}%", flush=True)

    overall_acc = float(np.mean([r["is_correct"] for r in all_evaluations])) * 100.0
    gsm_acc = float(np.mean([r["is_correct"] for r in all_evaluations if r["benchmark"] == "GSM8K"])) * 100.0
    math_acc = float(np.mean([r["is_correct"] for r in all_evaluations if r["benchmark"] in ["MATH-500", "MATH500"]])) * 100.0
    math_hard = float(np.mean([
        r["is_correct"] for r in all_evaluations
        if r["benchmark"] in ["MATH-500", "MATH500"] and str(r.get("stratum", "")) in ["Level 3", "Level 4", "Level 5"]
    ])) * 100.0
    trunc_rate = float(np.mean([r["is_truncated"] for r in all_evaluations])) * 100.0

    print("\n" + "=" * 80)
    print(f"ARM 3 REGISTER-BUNDLE LADDER RESULTS: {args.tag} (N={len(all_evaluations)})")
    print("=" * 80)
    print(f"Overall Accuracy (Pass@1) : {overall_acc:5.2f}%")
    print(f"GSM8K Accuracy            : {gsm_acc:5.2f}%")
    print(f"MATH-500 Accuracy         : {math_acc:5.2f}%")
    print(f"MATH Hard (L3-5)          : {math_hard:5.2f}%")
    print(f"Truncation Rate           : {trunc_rate:5.2f}%")
    print(f"Elapsed Time              : {time.time()-start_time:.1f}s")
    print("=" * 80)

    # Paired Bootstrap against Arm 1b Direct SFT (71.20%)
    arm1b_file = "data/streaming_arm1b_qwen_qwen3-1.7b.jsonl"
    bootstrap_arm1b = None
    if os.path.exists(arm1b_file):
        with open(arm1b_file) as f:
            arm1b_evals = [json.loads(line) for line in f]
        if len(arm1b_evals) == len(all_evaluations):
            scores_arm3 = [int(r["is_correct"]) for r in all_evaluations]
            scores_arm1b = [int(r["is_correct"]) for r in arm1b_evals]
            bootstrap_arm1b = run_paired_bootstrap(scores_arm3, scores_arm1b)
            print(f"\n--- PAIRED BOOTSTRAP vs ARM 1b DIRECT CONTROL ---")
            print(f"Observed Delta (Arm 3 - Arm 1b) : {bootstrap_arm1b['observed_delta']:+5.2f}%")
            print(f"95% Confidence Interval         : [{bootstrap_arm1b['ci_95'][0]:+5.2f}%, {bootstrap_arm1b['ci_95'][1]:+5.2f}%]")
            print(f"p-value                         : {bootstrap_arm1b['p_value']:.4f}")

    # Paired Bootstrap against Arm 1b-R Trained Headers (73.20%)
    arm1br_file = "data/streaming_arm1b_register_headers_only.jsonl"
    bootstrap_arm1br = None
    if os.path.exists(arm1br_file):
        with open(arm1br_file) as f:
            arm1br_evals = [json.loads(line) for line in f]
        if len(arm1br_evals) == len(all_evaluations):
            scores_arm3 = [int(r["is_correct"]) for r in all_evaluations]
            scores_arm1br = [int(r["is_correct"]) for r in arm1br_evals]
            bootstrap_arm1br = run_paired_bootstrap(scores_arm3, scores_arm1br)
            print(f"\n--- PAIRED BOOTSTRAP vs ARM 1b-R TRAINED HEADERS ---")
            print(f"Observed Delta (Arm 3 - Arm 1b-R) : {bootstrap_arm1br['observed_delta']:+5.2f}%")
            print(f"95% Confidence Interval           : [{bootstrap_arm1br['ci_95'][0]:+5.2f}%, {bootstrap_arm1br['ci_95'][1]:+5.2f}%]")
            print(f"p-value                           : {bootstrap_arm1br['p_value']:.4f}")

    # Paired Bootstrap against Arm 3-R Production Best (71.30%)
    arm3r_file = "data/streaming_arm3_register_ladder_production_best.jsonl"
    bootstrap_arm3r = None
    if os.path.exists(arm3r_file):
        with open(arm3r_file) as f:
            arm3r_evals = [json.loads(line) for line in f]
        if len(arm3r_evals) == len(all_evaluations):
            scores_arm3 = [int(r["is_correct"]) for r in all_evaluations]
            scores_arm3r = [int(r["is_correct"]) for r in arm3r_evals]
            bootstrap_arm3r = run_paired_bootstrap(scores_arm3, scores_arm3r)
            print(f"\n--- PAIRED BOOTSTRAP vs ARM 3-R STANDARD RECURRENCE ---")
            print(f"Observed Delta (LIB - Arm 3-R)   : {bootstrap_arm3r['observed_delta']:+5.2f}%")
            print(f"95% Confidence Interval          : [{bootstrap_arm3r['ci_95'][0]:+5.2f}%, {bootstrap_arm3r['ci_95'][1]:+5.2f}%]")
            print(f"p-value                          : {bootstrap_arm3r['p_value']:.4f}")

    results_data = {
        "tag": args.tag,
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
        "total_evaluations": len(all_evaluations),
        "overall_accuracy": overall_acc,
        "gsm8k_accuracy": gsm_acc,
        "math500_accuracy": math_acc,
        "math_hard_accuracy": math_hard,
        "truncation_rate": trunc_rate,
        "duration_seconds": time.time() - start_time,
        "paired_bootstrap_vs_arm1b": bootstrap_arm1b,
        "paired_bootstrap_vs_arm1b_r": bootstrap_arm1br,
        "paired_bootstrap_vs_arm3_r": bootstrap_arm3r
    }
    with open(out_results, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nEvaluation complete! Results saved to {out_results}")

if __name__ == "__main__":
    main()
