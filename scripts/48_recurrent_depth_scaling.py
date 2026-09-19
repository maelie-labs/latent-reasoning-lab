#!/usr/bin/env python3
"""
scripts/48_recurrent_depth_scaling.py
RECURRENT DEPTH SCALING & EXTRAPOLATION BENCHMARK

Evaluates how Continuous Latent Recurrence (Arm 3) scales with deeper recurrence budgets
beyond the training horizon (K=6) up to K=64:
K in [0, 6, 12, 16, 24, 32, 48, 64]

Telemetry captured:
1. Accuracy curve Pass@1(K)
2. Mean thinking latency (ms) per K
3. Hidden state norm evolution: ||h_t||_2 and step delta ||h_t - h_{t-1}||_2
4. P(</think> | h_t) termination logit dynamics across recurrence depth
"""

import os
import re
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from math_verify import parse, verify
import transformers.modeling_utils

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

THINK_END_TOKEN_ID = 151668
CALIBRATED_ALPHAS = {
    "qwen/qwen3-1.7b": 0.011440,
    "qwen/qwen3-4b": 0.006702,
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_and_extract_candidate(cand):
    if not cand:
        return None
    cand = re.sub(r"\\(?:text|mathbf|mathrm)\{([^}]+)\}", r"\1", str(cand))
    cand = re.sub(r"[\$\\%!\s]", "", cand)
    cand = cand.replace(",", "").strip().rstrip(".")
    m_frac = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", cand)
    if m_frac:
        try:
            return float(m_frac.group(1)) / float(m_frac.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    m_div = re.fullmatch(r"(-?\d+)/(-?\d+)", cand)
    if m_div:
        try:
            return float(m_div.group(1)) / float(m_div.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(cand)
    except ValueError:
        nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", cand)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                pass
    return None

def extract_math_boxed_expression(text):
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

def extract_answer(full_text, content_text):
    text = content_text if (content_text and content_text.strip()) else full_text
    boxed = extract_math_boxed_expression(text)
    if boxed:
        val = clean_and_extract_candidate(boxed)
        if val is not None:
            return val, boxed
            
    hash_match = re.findall(r'####\s*([\$]?[-+]?[\d,]+(?:\.\d+)?)', text)
    if hash_match:
        val = clean_and_extract_candidate(hash_match[-1])
        if val is not None:
            return val, hash_match[-1]
            
    ans_match = re.findall(
        r'(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|:)\s*([\$]?[-+]?[\d,]+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    if ans_match:
        val = clean_and_extract_candidate(ans_match[-1])
        if val is not None:
            return val, ans_match[-1]
            
    bold_match = re.findall(r'\*\*([\$]?[-+]?[\d,]+(?:\.\d+)?)\*\*', text)
    if bold_match:
        val = clean_and_extract_candidate(bold_match[-1])
        if val is not None:
            return val, bold_match[-1]

    nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", text)
    if nums:
        val = clean_and_extract_candidate(nums[-1])
        if val is not None:
            return val, nums[-1]
    return None, ""

def check_match(pred_val, pred_raw, gt_val, gt_raw, full_pred_text=None, full_gold_text=None):
    if full_pred_text and full_gold_text:
        try:
            gold_p = parse(full_gold_text, parsing_timeout=2.0)
            pred_p = parse(full_pred_text, parsing_timeout=2.0)
            if gold_p and pred_p and verify(gold_p, pred_p):
                return True
        except Exception:
            pass

    if pred_raw and gt_raw:
        try:
            gold_p = parse(str(gt_raw), parsing_timeout=2.0)
            pred_p = parse(str(pred_raw), parsing_timeout=2.0)
            if gold_p and pred_p and verify(gold_p, pred_p):
                return True
        except Exception:
            pass

    if pred_val is not None and gt_val is not None:
        if abs(pred_val - gt_val) < 1e-4:
            return True

    if pred_raw and gt_raw:
        c_pred = re.sub(r"[\$\\%!\s]", "", str(pred_raw)).replace(",", "").strip().lower()
        c_gt = re.sub(r"[\$\\%!\s]", "", str(gt_raw)).replace(",", "").strip().lower()
        if c_pred == c_gt:
            return True
    return False

def sample_token(logits, temperature=0.6, top_p=0.95, top_k=20):
    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("Inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def run_recurrent_step(model, tokenizer, prompt, device, scale_factor, k_steps=6, max_ans_tokens=8192):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]

        norms = []
        delta_norms = []
        p_think_trajectory = []
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")

        for step in range(k_steps):
            norm_val = float(torch.norm(curr_latent, p=2).item())
            norms.append(round(norm_val, 4))

            scaled_latent = curr_latent * scale_factor
            step_out = model(
                inputs_embeds=scaled_latent,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = step_out.past_key_values
            next_latent = step_out.hidden_states[-1][:, -1:, :]

            d_norm = float(torch.norm(next_latent - curr_latent, p=2).item())
            delta_norms.append(round(d_norm, 4))
            curr_latent = next_latent

            step_probs = F.softmax(step_out.logits[:, -1, :], dim=-1)
            p_think = step_probs[0, think_end_id].item()
            p_think_trajectory.append(round(p_think, 6))

        torch.cuda.synchronize(device)
        t_think_ms = (time.time() - t0) * 1000.0

        trans_tokens = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
        trans_ids = torch.tensor([trans_tokens], dtype=torch.long, device=device)
        step_out = model(input_ids=trans_ids, past_key_values=past_kv, use_cache=True)
        past_kv = step_out.past_key_values

        curr_logits = step_out.logits[:, -1, :]
        next_tok = sample_token(curr_logits, temperature=0.6, top_p=0.95, top_k=20)
        im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

        gen_tokens = [next_tok.item()]
        curr_tok = next_tok
        is_truncated = True

        for _ in range(max_ans_tokens):
            step_out = model(input_ids=curr_tok, past_key_values=past_kv, use_cache=True)
            past_kv = step_out.past_key_values
            curr_tok = sample_token(step_out.logits[:, -1, :], temperature=0.6, top_p=0.95, top_k=20)
            t_val = curr_tok.item()
            if t_val in [tokenizer.eos_token_id, im_end_id]:
                is_truncated = False
                break
            gen_tokens.append(t_val)

        torch.cuda.synchronize(device)
        total_time_s = time.time() - t0

    content = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    pred_val, pred_raw = extract_answer(content, content)
    return {
        "k_steps": k_steps,
        "think_time_ms": round(t_think_ms, 2),
        "total_time_s": round(total_time_s, 3),
        "ans_tokens": len(gen_tokens),
        "is_truncated": is_truncated,
        "latent_norms": norms,
        "delta_norms": delta_norms,
        "p_think_trajectory": p_think_trajectory,
        "content": content,
        "pred_val": pred_val,
        "pred_raw": pred_raw
    }

def main():
    parser = argparse.ArgumentParser(description="Recurrent Depth Scaling Benchmark.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora_path", type=str, default="checkpoints/lora_arm3_qwen_qwen3-1.7b_k6")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--suite_size", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--k_list", type=int, nargs="+", default=[0, 6, 12, 16, 24, 32, 48, 64])
    parser.add_argument("--output_file", type=str, default="data/recurrent_depth_scaling_qwen3_1.7b.json")
    args = parser.parse_args()

    scale_factor = CALIBRATED_ALPHAS.get(args.model_id.lower(), 0.011440)

    print(f"=== Recurrent Depth Scaling for {args.model_id} ===")
    print(f"Device: {args.device} | Adapter: {args.lora_path}")
    print(f"K Evaluation Frontier: {args.k_list}")
    print(f"Suite: {args.suite_size} problems | Seeds: {args.seeds}")

    cache_path = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_suite_250.json")
    with open(cache_path) as f:
        all_problems = json.load(f)
    problems = all_problems[:args.suite_size]

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip("\"'\n")
                    break

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    base_model.eval()

    print(f"Loading LoRA adapter from {args.lora_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()

    scaling_results = {}

    for k in args.k_list:
        print(f"\n{'='*60}\nEvaluating Recurrence Depth K = {k}\n{'='*60}")
        correct_count = 0
        total_runs = len(problems) * len(args.seeds)
        think_times = []
        total_times = []
        avg_norms = []
        avg_deltas = []
        avg_p_thinks = []

        for p_idx, prob in enumerate(problems):
            q = prob["question"]
            sol = prob["solution"]
            gt_boxed = extract_math_boxed_expression(sol)
            gt_val = clean_and_extract_candidate(gt_boxed or sol)

            for seed in args.seeds:
                set_seed(seed)
                run = run_recurrent_step(model, tokenizer, q, args.device, scale_factor, k_steps=k)
                is_correct = check_match(
                    run["pred_val"],
                    run["pred_raw"],
                    gt_val,
                    gt_boxed or sol,
                    full_pred_text=run["content"],
                    full_gold_text=sol
                )
                if is_correct:
                    correct_count += 1

                think_times.append(run["think_time_ms"])
                total_times.append(run["total_time_s"])
                if run["latent_norms"]:
                    avg_norms.append(run["latent_norms"])
                if run["delta_norms"]:
                    avg_deltas.append(run["delta_norms"])
                if run["p_think_trajectory"]:
                    avg_p_thinks.append(run["p_think_trajectory"])

            if (p_idx + 1) % 25 == 0 or (p_idx + 1) == len(problems):
                acc = correct_count / ((p_idx + 1) * len(args.seeds)) * 100.0
                print(f"  [K={k} | {p_idx+1}/{len(problems)}] Accuracy: {acc:.2f}%")

        pass_rate = correct_count / total_runs * 100.0
        mean_think = float(np.mean(think_times))
        mean_total = float(np.mean(total_times))

        k_summary = {
            "k_steps": k,
            "pass_rate_pct": round(pass_rate, 2),
            "correct_count": correct_count,
            "total_runs": total_runs,
            "mean_think_ms": round(mean_think, 2),
            "mean_total_s": round(mean_total, 3),
            "mean_norm_curve": [round(float(x), 4) for x in np.mean(avg_norms, axis=0)] if avg_norms else [],
            "mean_delta_curve": [round(float(x), 4) for x in np.mean(avg_deltas, axis=0)] if avg_deltas else [],
            "mean_p_think_curve": [round(float(x), 6) for x in np.mean(avg_p_thinks, axis=0)] if avg_p_thinks else []
        }
        scaling_results[f"k_{k}"] = k_summary

        print(f"\n--- Result for K={k}: Pass@1 = {pass_rate:.2f}% | Think Time: {mean_think:.1f}ms | Total: {mean_total:.2f}s ---")

    output_payload = {
        "model_id": args.model_id,
        "adapter_path": args.lora_path,
        "suite_size": args.suite_size,
        "seeds": args.seeds,
        "k_frontier": args.k_list,
        "results": scaling_results
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output_payload, f, indent=2, default=str)
    print(f"\nSaved scaling results to: {args.output_file}")

if __name__ == "__main__":
    main()
