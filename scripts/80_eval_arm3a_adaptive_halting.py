#!/usr/bin/env python3
"""
scripts/80_eval_arm3a_adaptive_halting.py
Arm 3a: Adaptive Latent Halting Recurrence Benchmark for Qwen/Qwen3-1.7B.

Evaluates dynamic early halting:
- Recurrent steps K in [K_min, K_max]
- Halting criterion: P(</think> | h_k) >= tau
- Transitions to direct answer phase upon halting
- Evaluates across deterministic seeds [42, 123, 456, 789] on benchmark_suite_250.json
- Logs exit step distribution, accuracy, and latency Pareto frontier
- Grader: Canonical math_verify with symbolic equivalence
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from peft import PeftModel
from math_verify import parse, verify

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from utils_qwen3 import PresencePenaltyLogitsProcessor, extract_math_boxed_expression

CALIBRATED_ALPHA = 0.011440

def run_adaptive_eval(model, tokenizer, problems, k_min, k_max, tau, scale_factor, max_ans_tokens, device):
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    trans_tokens = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
    trans_len = len(trans_tokens)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    
    results = []
    exit_steps = []
    
    for idx, prob in enumerate(problems):
        q_text = prob["question"]
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        enc = tokenizer(prompt_text, return_tensors="pt").to(device)
        L_prompt = enc.input_ids.shape[1]
        
        t0_item = time.time()
        with torch.no_grad():
            out = model(**enc, use_cache=True, output_hidden_states=True)
            past_kv = out.past_key_values
            curr_latent = out.hidden_states[-1][:, -1:, :]
            curr_mask = enc.attention_mask
            
            halt_step = k_max
            p_stop_final = 0.0
            
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                for k in range(k_max):
                    pos = torch.tensor([[L_prompt + k]], device=device, dtype=torch.long)
                    curr_mask = torch.cat([curr_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
                    scaled_latent = curr_latent * scale_factor
                    
                    step_out = model(
                        inputs_embeds=scaled_latent,
                        attention_mask=curr_mask,
                        position_ids=pos,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_hidden_states=True
                    )
                    past_kv = step_out.past_key_values
                    curr_latent = step_out.hidden_states[-1][:, -1:, :]
                    
                    if (k + 1) >= k_min:
                        logits_k = model.lm_head(curr_latent)
                        p_stop = F.softmax(logits_k, dim=-1)[0, 0, think_end_id].item()
                        if p_stop >= tau:
                            halt_step = k + 1
                            p_stop_final = p_stop
                            break
                            
            exit_steps.append(halt_step)
            t_think_ms = (time.time() - t0_item) * 1000.0
            
            trans_ids = torch.tensor([trans_tokens], dtype=torch.long, device=device)
            trans_mask = torch.cat([curr_mask, torch.ones((1, trans_len), dtype=torch.long, device=device)], dim=1)
            trans_pos = torch.tensor([[L_prompt + halt_step + i for i in range(trans_len)]], device=device, dtype=torch.long)
            
            logits_proc = LogitsProcessorList([PresencePenaltyLogitsProcessor(1.5, [trans_len])])
            gen_out = model.generate(
                input_ids=trans_ids,
                attention_mask=trans_mask,
                position_ids=trans_pos,
                past_key_values=past_kv,
                max_new_tokens=max_ans_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.80,
                top_k=20,
                logits_processor=logits_proc,
                pad_token_id=pad_id
            )
            
            total_time_s = time.time() - t0_item
            gen_toks = gen_out[0][trans_len:].tolist()
            trunc = bool(len(gen_toks) >= max_ans_tokens and eos_id not in gen_toks)
            content = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
            
            gt_sol = prob["solution"]
            gt_boxed = gt_sol.split("####")[-1].strip() if "####" in gt_sol else extract_math_boxed_expression(gt_sol)
            is_correct = False
            gt_target = f"\\boxed{{{gt_boxed}}}" if gt_boxed else f"\\boxed{{{gt_sol}}}"
            if not trunc and content:
                try:
                    gt_p = parse(gt_target)
                    pred_p = parse(content)
                    is_correct = verify(gt_p, pred_p)
                except Exception:
                    is_correct = False
                    
            results.append({
                "id": prob["id"],
                "benchmark": prob.get("benchmark", ""),
                "stratum": prob.get("stratum", ""),
                "level": prob.get("level", None),
                "halt_step": halt_step,
                "p_stop": round(p_stop_final, 4),
                "is_correct": is_correct,
                "is_truncated": trunc,
                "think_time_ms": round(t_think_ms, 1),
                "total_time_s": round(total_time_s, 2),
                "gt_target": gt_target,
                "content": content
            })
            
    return results, exit_steps

def main():
    parser = argparse.ArgumentParser(description="Arm 3a Adaptive Latent Halting Evaluation")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora_path", type=str, default="checkpoints/lora_arm3_qwen_qwen3-1.7b_k32_selfdistill")
    parser.add_argument("--suite", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789])
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=32)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    parser.add_argument("--max_ans_tokens", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_file", type=str, default="data/eval_arm3a_adaptive_qwen3_1.7b.json")
    args = parser.parse_args()

    print("=" * 70)
    print("=== Arm 3a: Adaptive Latent Halting Evaluation ===")
    print(f"Model: {args.model_id} | Adapter: {args.lora_path}")
    print(f"Loop Horizon: K in [{args.k_min}, {args.k_max}] | Thresholds: {args.thresholds}")
    print(f"Seeds: {args.seeds} | Device: {args.device}")
    print("=" * 70)

    suite_path = os.path.join(ROOT_DIR, args.suite) if not os.path.isabs(args.suite) else args.suite
    with open(suite_path) as f:
        problems = json.load(f)
    print(f"Loaded {len(problems)} problems from {suite_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()

    all_threshold_results = {}

    for tau in args.thresholds:
        print(f"\n--- Evaluating Adaptive Halting with tau = {tau} ---")
        tau_records = []
        tau_exit_steps = []
        
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            records, exits = run_adaptive_eval(
                model=model,
                tokenizer=tokenizer,
                problems=problems,
                k_min=args.k_min,
                k_max=args.k_max,
                tau=tau,
                scale_factor=CALIBRATED_ALPHA,
                max_ans_tokens=args.max_ans_tokens,
                device=args.device
            )
            for r in records:
                r["seed"] = seed
            tau_records.extend(records)
            tau_exit_steps.extend(exits)
            
            correct = sum(1 for r in records if r["is_correct"])
            print(f"  Seed {seed:3d}: Pass@1 = {100.0 * correct / len(records):5.2f}% | Mean Steps K = {np.mean(exits):4.1f}")

        total_evals = len(tau_records)
        total_correct = sum(1 for r in tau_records if r["is_correct"])
        total_trunc = sum(1 for r in tau_records if r["is_truncated"])
        pass1_pct = round(100.0 * total_correct / total_evals, 2)
        trunc_pct = round(100.0 * total_trunc / total_evals, 2)
        mean_k = round(float(np.mean(tau_exit_steps)), 2)
        med_k = round(float(np.median(tau_exit_steps)), 1)
        mean_think_ms = round(float(np.mean([r["think_time_ms"] for r in tau_records])), 1)

        print(f"--> Tau {tau:.1f} Overall: Pass@1 = {pass1_pct}% | Mean K = {mean_k} (med {med_k}) | Think Latency = {mean_think_ms} ms | Trunc = {trunc_pct}%")

        all_threshold_results[str(tau)] = {
            "tau": tau,
            "pass1_pct": pass1_pct,
            "truncation_pct": trunc_pct,
            "mean_halt_steps": mean_k,
            "median_halt_steps": med_k,
            "mean_think_time_ms": mean_think_ms,
            "evaluations_count": total_evals,
            "per_problem_results": tau_records
        }

    out_path = os.path.join(ROOT_DIR, args.output_file) if not os.path.isabs(args.output_file) else args.output_file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_threshold_results, f, indent=2)
    print(f"\nAll Arm 3a evaluations complete. Saved results to: {out_path}")

if __name__ == "__main__":
    main()
