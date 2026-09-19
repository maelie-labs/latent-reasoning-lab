#!/usr/bin/env python3
"""
scripts/89_eval_decision_gate_v1_1.py
Unified Decision Gate 1b Benchmark Harness for Continuous Latent Recurrence v1.1.

Evaluates trained adapters across the 250-suite benchmark (data/benchmark_suite_250.json)
across 4 deterministic seeds: SEEDS = [42, 123, 456, 789] (N=1,000 total queries).

High-throughput batched PyTorch execution (B=16) strictly adhering to:
- Rule 2 (High-Throughput Execution Only, Zero Batch-Size-1 Loops)
- Rule 5 (Multi-Seed Sampling, Pure BF16, Zero Test Contamination)
- Rule 9 ("One Suite, One Harness, Every Model")
- Gate 0 Frozen Specification Table

Outputs:
- streaming_v1_1_{arm}_k{k}.jsonl
- eval_results_v1_1_{arm}_k{k}.json
- paired_bootstrap_v1_1_arm3_k{k}_vs_arm1b_k0.json (when arm3 finishes)
- paired_bootstrap_v1_1_arm3_k{k}_vs_arm2b_k{k}.json (when arm3 finishes)
"""

import os
import re
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from peft import PeftModel
from math_verify import parse, verify

SEEDS = [42, 123, 456, 789]
PAUSE_TOKEN = "<pause>"
CALIBRATED_ALPHAS = {
    "qwen/qwen3-1.7b": 0.011440,
    "qwen/qwen3.5-2b": 0.009800,
}

class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Additive presence penalty applied strictly to generated tokens."""
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
        "p_value": p_val * 2.0  # Two-tailed
    }

def try_paired_bootstrap(data_dir, k_tokens):
    """Automatically run paired bootstrap if comparison arms are present."""
    arm3_file = os.path.join(data_dir, f"streaming_v1_1_arm3_k{k_tokens}.jsonl")
    arm1b_file = os.path.join(data_dir, "streaming_v1_1_arm1b_k0.jsonl")
    arm2b_file = os.path.join(data_dir, f"streaming_v1_1_arm2b_k{k_tokens}.jsonl")
    
    if not os.path.exists(arm3_file):
        return

    def load_stream(path):
        data = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    data[(rec["id"], rec["seed"])] = 1.0 if rec["is_correct"] else 0.0
        return data

    arm3_scores_map = load_stream(arm3_file)

    # Compare Arm 3 vs Arm 1b
    if os.path.exists(arm1b_file):
        arm1b_scores_map = load_stream(arm1b_file)
        common_keys = sorted(list(set(arm3_scores_map.keys()).intersection(set(arm1b_scores_map.keys()))))
        if len(common_keys) == 1000:
            s_a = np.array([arm3_scores_map[k] for k in common_keys])
            s_b = np.array([arm1b_scores_map[k] for k in common_keys])
            boot_res = run_paired_bootstrap(s_a, s_b)
            boot_res["n_pairs"] = len(common_keys)
            boot_res["arm_a"] = f"arm3_k{k_tokens}"
            boot_res["arm_b"] = "arm1b_k0"
            out_path = os.path.join(data_dir, f"paired_bootstrap_v1_1_arm3_k{k_tokens}_vs_arm1b_k0.json")
            with open(out_path, "w") as f:
                json.dump(boot_res, f, indent=2)
            print(f"\n[PAIRED BOOTSTRAP] Arm 3 (K={k_tokens}) vs Arm 1b: Delta = {boot_res['observed_delta']:+.2f}% (95% CI: [{boot_res['ci_95'][0]:+.2f}%, {boot_res['ci_95'][1]:+.2f}%], p = {boot_res['p_value']:.4f})")

    # Compare Arm 3 vs Arm 2b
    if os.path.exists(arm2b_file):
        arm2b_scores_map = load_stream(arm2b_file)
        common_keys = sorted(list(set(arm3_scores_map.keys()).intersection(set(arm2b_scores_map.keys()))))
        if len(common_keys) == 1000:
            s_a = np.array([arm3_scores_map[k] for k in common_keys])
            s_b = np.array([arm2b_scores_map[k] for k in common_keys])
            boot_res = run_paired_bootstrap(s_a, s_b)
            boot_res["n_pairs"] = len(common_keys)
            boot_res["arm_a"] = f"arm3_k{k_tokens}"
            boot_res["arm_b"] = f"arm2b_k{k_tokens}"
            out_path = os.path.join(data_dir, f"paired_bootstrap_v1_1_arm3_k{k_tokens}_vs_arm2b_k{k_tokens}.json")
            with open(out_path, "w") as f:
                json.dump(boot_res, f, indent=2)
            print(f"[PAIRED BOOTSTRAP] Arm 3 (K={k_tokens}) vs Arm 2b (K={k_tokens}): Delta = {boot_res['observed_delta']:+.2f}% (95% CI: [{boot_res['ci_95'][0]:+.2f}%, {boot_res['ci_95'][1]:+.2f}%], p = {boot_res['p_value']:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Decision Gate 1b Benchmark Harness")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--arm", type=str, required=True, choices=["arm1b", "arm2b", "arm3"])
    parser.add_argument("--k_tokens", type=int, default=6)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_ans_tokens", type=int, default=8192)
    args = parser.parse_args()

    scale_factor = CALIBRATED_ALPHAS.get(args.model_id.lower(), 0.011440)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    suite_path = os.path.join(data_dir, "benchmark_suite_250.json")

    tag = f"{args.arm}_k{args.k_tokens}"
    out_json = os.path.join(data_dir, f"eval_results_v1_1_{tag}.json")
    out_jsonl = os.path.join(data_dir, f"streaming_v1_1_{tag}.jsonl")

    # Clear previous partial streaming file if starting fresh
    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    print(f"======================================================================")
    print(f"DECISION GATE 1b BENCHMARK EVALUATOR: {args.arm.upper()} (K={args.k_tokens})")
    print(f"Model: {args.model_id} | Device: {args.device} | Batch Size: {args.batch_size}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output JSON: {out_json}")
    print(f"Streaming JSONL: {out_jsonl}")
    print(f"======================================================================")

    with open(suite_path) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} benchmark suite problems.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.arm == "arm2b":
        if PAUSE_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    if args.arm == "arm2b":
        base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    all_evaluations = []
    start_time = time.time()

    # Load Live-Subset problem IDs
    live_ids_path = os.path.join(data_dir, "benchmark_250_live_subset_ids.json")
    live_ids = set()
    if os.path.exists(live_ids_path):
        with open(live_ids_path) as f:
            live_data = json.load(f)
            live_ids = set(p["id"] for p in live_data.get("problems", []))
    print(f"Loaded {len(live_ids)} pre-registered live-subset problem IDs.")

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        torch.manual_seed(seed)
        
        for i in range(0, len(problems), args.batch_size):
            b_items = problems[i:i + args.batch_size]
            B = len(b_items)
            
            if args.arm == "arm1b":
                b_prompts = [
                    f"<|im_start|>user\n{p['question']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    for p in b_items
                ]
                enc = tokenizer(b_prompts, return_tensors="pt", padding=True).to(args.device)
                prompt_len = enc.input_ids.shape[1]
                pp_proc = PresencePenaltyLogitsProcessor(1.5, prompt_len)
                with torch.inference_mode():
                    out = model.generate(
                        **enc,
                        max_new_tokens=args.max_ans_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.80,
                        top_k=20,
                        logits_processor=LogitsProcessorList([pp_proc]),
                        pad_token_id=tokenizer.pad_token_id
                    )
                for b_idx in range(B):
                    gen_toks = out[b_idx][prompt_len:].tolist()
                    is_trunc = bool(len(gen_toks) >= args.max_ans_tokens and tokenizer.eos_token_id not in gen_toks)
                    gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
                    is_correct = False if is_trunc else check_math_correct(gen_text, b_items[b_idx]["solution"])
                    rec = {
                        "id": b_items[b_idx]["id"],
                        "benchmark": b_items[b_idx]["benchmark"],
                        "stratum": b_items[b_idx].get("stratum", ""),
                        "seed": seed,
                        "is_correct": is_correct,
                        "is_truncated": is_trunc,
                        "is_live": b_items[b_idx]["id"] in live_ids,
                        "tokens": len(gen_toks),
                        "content": gen_text
                    }
                    all_evaluations.append(rec)
                    with open(out_jsonl, "a") as f_out:
                        f_out.write(json.dumps(rec) + "\n")

            elif args.arm == "arm2b":
                pause_seq = PAUSE_TOKEN * args.k_tokens
                b_prompts = [
                    f"<|im_start|>user\n{p['question']}<|im_end|>\n<|im_start|>assistant\n<think>\n{pause_seq}\n</think>\n\n"
                    for p in b_items
                ]
                enc = tokenizer(b_prompts, return_tensors="pt", padding=True).to(args.device)
                prompt_len = enc.input_ids.shape[1]
                pp_proc = PresencePenaltyLogitsProcessor(1.5, prompt_len)
                with torch.inference_mode():
                    out = model.generate(
                        **enc,
                        max_new_tokens=args.max_ans_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.80,
                        top_k=20,
                        logits_processor=LogitsProcessorList([pp_proc]),
                        pad_token_id=tokenizer.pad_token_id
                    )
                for b_idx in range(B):
                    gen_toks = out[b_idx][prompt_len:].tolist()
                    is_trunc = bool(len(gen_toks) >= args.max_ans_tokens and tokenizer.eos_token_id not in gen_toks)
                    gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
                    is_correct = False if is_trunc else check_math_correct(gen_text, b_items[b_idx]["solution"])
                    rec = {
                        "id": b_items[b_idx]["id"],
                        "benchmark": b_items[b_idx]["benchmark"],
                        "stratum": b_items[b_idx].get("stratum", ""),
                        "seed": seed,
                        "is_correct": is_correct,
                        "is_truncated": is_trunc,
                        "is_live": b_items[b_idx]["id"] in live_ids,
                        "tokens": len(gen_toks),
                        "content": gen_text
                    }
                    all_evaluations.append(rec)
                    with open(out_jsonl, "a") as f_out:
                        f_out.write(json.dumps(rec) + "\n")

            elif args.arm == "arm3":
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

                    seq_lens = enc.attention_mask.sum(dim=-1)
                    curr_mask = enc.attention_mask
                    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                        for k in range(args.k_tokens):
                            step_pos = (seq_lens + k).unsqueeze(1)
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

                    # Ingest \n</think>\n\n transition
                    trans_tokens = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
                    trans_len = len(trans_tokens)
                    trans_ids = torch.tensor([trans_tokens] * B, dtype=torch.long, device=args.device)
                    trans_mask = torch.cat([curr_mask, torch.ones((B, trans_len), dtype=torch.long, device=args.device)], dim=1)
                    pos = torch.stack([torch.arange(trans_len, device=args.device) + (seq_lens[b] + args.k_tokens) for b in range(B)], dim=0)

                    pp_proc = PresencePenaltyLogitsProcessor(1.5, trans_len)
                    gen_out = model.generate(
                        input_ids=trans_ids,
                        attention_mask=trans_mask,
                        position_ids=pos,
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
                    is_correct = False if is_trunc else check_math_correct(gen_text, b_items[b_idx]["solution"])
                    rec = {
                        "id": b_items[b_idx]["id"],
                        "benchmark": b_items[b_idx]["benchmark"],
                        "stratum": b_items[b_idx].get("stratum", ""),
                        "seed": seed,
                        "is_correct": is_correct,
                        "is_truncated": is_trunc,
                        "is_live": b_items[b_idx]["id"] in live_ids,
                        "tokens": len(gen_toks),
                        "content": gen_text
                    }
                    all_evaluations.append(rec)
                    with open(out_jsonl, "a") as f_out:
                        f_out.write(json.dumps(rec) + "\n")

    overall_acc = np.mean([r["is_correct"] for r in all_evaluations]) * 100.0
    gsm_acc = np.mean([r["is_correct"] for r in all_evaluations if r["benchmark"] == "GSM8K"]) * 100.0
    math_acc = np.mean([r["is_correct"] for r in all_evaluations if r["benchmark"] in ["MATH-500", "MATH500"]]) * 100.0
    math_hard_acc = np.mean([
        r["is_correct"] for r in all_evaluations
        if r["benchmark"] in ["MATH-500", "MATH500"] and str(r.get("stratum", "")) in ["Level 3", "Level 4", "Level 5"]
    ]) * 100.0
    
    live_evals = [r for r in all_evaluations if r["is_live"]]
    live_acc = np.mean([r["is_correct"] for r in live_evals]) * 100.0 if live_evals else 0.0
    trunc_rate = np.mean([r["is_truncated"] for r in all_evaluations]) * 100.0

    print(f"\n=======================================================")
    print(f"DECISION GATE 1b BENCHMARK RESULTS ({tag})")
    print(f"Overall Pass@1:       {overall_acc:.2f}% (N={len(all_evaluations)})")
    print(f"GSM8K Pass@1:         {gsm_acc:.2f}% (N=600)")
    print(f"MATH-500:             {math_acc:.2f}% (N=400)")
    print(f"MATH Hard L3-5:       {math_hard_acc:.2f}% (N=240)")
    print(f"Live-Subset:          {live_acc:.2f}% (N={len(live_evals)})")
    print(f"Truncation Rate:      {trunc_rate:.2f}%")
    print(f"Total Time:           {time.time()-start_time:.1f}s")
    print(f"=======================================================")

    with open(out_json, "w") as f:
        json.dump({
            "arm": args.arm,
            "k_tokens": args.k_tokens,
            "checkpoint": args.checkpoint,
            "overall_pass1": overall_acc,
            "gsm8k_pass1": gsm_acc,
            "math500_pass1": math_acc,
            "math_hard_l35_pass1": math_hard_acc,
            "live_subset_pass1": live_acc,
            "live_subset_count": len(live_evals),
            "truncation_rate": trunc_rate,
            "total_evaluations": len(all_evaluations),
            "eval_duration_seconds": time.time() - start_time
        }, f, indent=2)
    print(f"Saved evaluation summary to: {out_json}")

    # Run paired bootstrap if this was arm3
    if args.arm == "arm3":
        try_paired_bootstrap(data_dir, args.k_tokens)

if __name__ == "__main__":
    main()
