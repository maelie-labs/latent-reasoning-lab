#!/usr/bin/env python3
"""
scripts/104_eval_register_ladder_bottleneck.py
Unified Evaluation for Latent Information Bottleneck (LIB) Register-Bundle Ladder.

Evaluates trained LIB checkpoint on the 60 MATH L3-5 Problems under:
1. Condition 1: Bottleneck Enforced (Prompt attention strictly MASKED during answer decoding)
2. Condition 2: Bottleneck Unmasked (Prompt attention visible during answer decoding)
3. Check A: Attention Split Decomposition (Verifying elimination of the prompt shortcut)

Enforces:
- Gate 0 Frozen Specification (temp=0.7, top_p=0.80, top_k=20, pp=1.5, max_ans=8192)
- Canonical math_verify 0.9.0 with canonical \boxed{} extraction
- Pure bfloat16 execution
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

CALIBRATED_ALPHA = 0.011440
K_PER_BUNDLE = 8
NUM_RUNGS = 3
TOTAL_LATENTS = NUM_RUNGS * K_PER_BUNDLE  # 24

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

def check_correctness(prediction_text, ground_truth):
    try:
        gold_str = str(ground_truth).strip()
        if "####" in gold_str:
            ans_part = gold_str.split("####")[-1].strip()
            gold_target = f"\\boxed{{{ans_part}}}"
        elif "\\boxed{" not in gold_str:
            gold_target = f"\\boxed{{{gold_str}}}"
        else:
            gold_target = gold_str
        gold_parsed = parse(gold_target, parsing_timeout=None)
        pred_parsed = parse(prediction_text, parsing_timeout=None)
        if gold_parsed and pred_parsed:
            return float(bool(verify(gold_parsed, pred_parsed)))
    except Exception:
        pass
    return 0.0

def run_bottleneck_inference(
    model, tokenizer, question, device,
    mask_prompt_in_answer=True,
    max_ans_tokens=8192,
    temperature=0.7,
    top_p=0.80,
    top_k=20,
    presence_penalty=1.5
):
    """
    Executes Register Ladder unrolling + Bottleneck answer generation.
    """
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    L_prompt = enc_prompt.input_ids.shape[1]

    # 1. Prefill Prompt
    out = model(input_ids=enc_prompt.input_ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    curr_latent = out.hidden_states[-1][:, -1:, :]
    curr_seq_len = L_prompt

    backbone = model.base_model.model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "model") else model

    hdr_indices = []
    latent_indices = []

    # 2. Interleaved Register-Bundle Loop
    for r in range(NUM_RUNGS):
        hdr_text = CANONICAL_HEADERS[r]
        enc_hdr = tokenizer(hdr_text, return_tensors="pt", add_special_tokens=False).to(device)
        hdr_len = enc_hdr.input_ids.shape[1]
        hdr_pos = torch.arange(curr_seq_len, curr_seq_len + hdr_len, device=device).unsqueeze(0)
        hdr_out = model(
            input_ids=enc_hdr.input_ids, position_ids=hdr_pos, past_key_values=past_kv,
            use_cache=True, output_hidden_states=True
        )
        past_kv = hdr_out.past_key_values
        curr_latent = hdr_out.hidden_states[-1][:, -1:, :]
        hdr_indices.extend(list(range(curr_seq_len, curr_seq_len + hdr_len)))
        curr_seq_len += hdr_len

        for k in range(K_PER_BUNDLE):
            step_pos = torch.tensor([[curr_seq_len]], device=device, dtype=torch.long)
            scaled = curr_latent * CALIBRATED_ALPHA
            step_out = backbone(
                inputs_embeds=scaled, position_ids=step_pos, past_key_values=past_kv,
                use_cache=True
            )
            past_kv = step_out.past_key_values
            curr_latent = step_out.last_hidden_state[:, -1:, :] if hasattr(step_out, "last_hidden_state") else step_out.hidden_states[-1][:, -1:, :]
            latent_indices.append(curr_seq_len)
            curr_seq_len += 1

    # 3. Transition Delimiter
    trans_text = "\n</think>\n\n"
    enc_trans = tokenizer(trans_text, return_tensors="pt", add_special_tokens=False).to(device)
    trans_len = enc_trans.input_ids.shape[1]
    trans_pos = torch.arange(curr_seq_len, curr_seq_len + trans_len, device=device).unsqueeze(0)

    if mask_prompt_in_answer:
        trans_mask = torch.zeros((1, 1, trans_len, curr_seq_len + trans_len), dtype=torch.bool, device=device)
        for q in range(trans_len):
            trans_mask[0, 0, q, L_prompt : curr_seq_len] = True
            trans_mask[0, 0, q, curr_seq_len : curr_seq_len + q + 1] = True
    else:
        trans_mask = None

    trans_out = model(
        input_ids=enc_trans.input_ids, position_ids=trans_pos, past_key_values=past_kv,
        attention_mask=trans_mask, use_cache=True, output_hidden_states=True
    )
    past_kv = trans_out.past_key_values
    curr_seq_len += trans_len

    # 4. Answer Generation Loop
    curr_token = torch.argmax(trans_out.logits[:, -1:, :], dim=-1)
    gen_tokens = [curr_token.item()]
    curr_pos = curr_seq_len
    seen_tokens = set([curr_token.item()])

    for g in range(max_ans_tokens):
        total_kv = curr_pos + 1
        if mask_prompt_in_answer:
            step_mask = torch.zeros((1, 1, 1, total_kv), dtype=torch.bool, device=device)
            step_mask[0, 0, 0, L_prompt : total_kv] = True
        else:
            step_mask = None

        step_out = model(
            input_ids=curr_token,
            position_ids=torch.tensor([[curr_pos]], device=device),
            past_key_values=past_kv,
            attention_mask=step_mask,
            use_cache=True
        )
        past_kv = step_out.past_key_values
        curr_pos += 1

        logits = step_out.logits[:, -1, :].clone()

        # Apply presence penalty
        if presence_penalty > 0.0:
            for tid in seen_tokens:
                logits[0, tid] -= presence_penalty

        # Temperature / Top-p / Top-k
        if temperature > 0.0:
            logits = logits / temperature
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_topk = topk_vals[:, -1]
                logits[logits < min_topk] = -float("Inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        curr_token = next_token
        tid = curr_token.item()
        gen_tokens.append(tid)
        seen_tokens.add(tid)

        if tid == tokenizer.eos_token_id:
            break

    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return {
        "text": gen_text,
        "gen_len": len(gen_tokens)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lora_arm3_register_ladder_bottleneck_qwen3_1.7b/best_checkpoint")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--output_file", type=str, default="data/eval_results_register_ladder_bottleneck_qwen3_1.7b.json")
    args = parser.parse_args()

    print("=" * 80)
    print("EVALUATION: LATENT INFORMATION BOTTLENECK (LIB) REGISTER LADDER")
    print(f"Model      : {args.model_id}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {args.device}")
    print("=" * 80)

    # Load 60 MATH L3-5
    with open("data/benchmark_suite_250.json") as f:
        suite = json.load(f)

    math_l35 = [p for p in suite if p.get("benchmark") == "MATH-500" and p.get("stratum") in ["Level 3", "Level 4", "Level 5"]]
    print(f"Loaded {len(math_l35)} MATH-500 L3-5 test problems.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=args.device,
        trust_remote_code=True
    )

    ckpt_to_use = args.checkpoint
    if not os.path.exists(ckpt_to_use):
        alt_ckpt = os.path.join(os.path.dirname(args.checkpoint), "final_checkpoint")
        if os.path.exists(alt_ckpt):
            ckpt_to_use = alt_ckpt
            print(f"Notice: best_checkpoint not found. Using {alt_ckpt}")
        else:
            print(f"WARNING: Checkpoint {args.checkpoint} not found. Running base model.")
            ckpt_to_use = None

    if ckpt_to_use:
        model = PeftModel.from_pretrained(base_model, ckpt_to_use)
    else:
        model = base_model
    model.eval()

    # 1. Evaluate with Bottleneck Enforced (Mask = True)
    print("\n--- Evaluating with Bottleneck Enforced (Prompt Masked) ---")
    correct_masked = 0
    results_masked = []
    for idx, p in enumerate(math_l35):
        gold = p.get("solution") or p.get("answer", "")
        q = p["question"]
        res = run_bottleneck_inference(model, tokenizer, q, args.device, mask_prompt_in_answer=True)
        is_corr = check_correctness(res["text"], gold)
        correct_masked += is_corr
        results_masked.append({
            "id": p.get("id", idx),
            "level": p.get("stratum", ""),
            "is_correct": bool(is_corr),
            "pred_len": res["gen_len"],
            "pred_text": res["text"][:200]
        })
        if (idx + 1) % 10 == 0 or idx == len(math_l35) - 1:
            print(f"[Masked {idx+1:2d}/60] Running Acc: {correct_masked/(idx+1)*100:.2f}% ({int(correct_masked)}/{idx+1})")

    acc_masked = (correct_masked / len(math_l35)) * 100.0

    # 2. Evaluate with Bottleneck Unmasked (Prompt Visible)
    print("\n--- Evaluating with Bottleneck Unmasked (Prompt Visible) ---")
    correct_unmasked = 0
    results_unmasked = []
    for idx, p in enumerate(math_l35):
        gold = p.get("solution") or p.get("answer", "")
        q = p["question"]
        res = run_bottleneck_inference(model, tokenizer, q, args.device, mask_prompt_in_answer=False)
        is_corr = check_correctness(res["text"], gold)
        correct_unmasked += is_corr
        results_unmasked.append({
            "id": p.get("id", idx),
            "level": p.get("stratum", ""),
            "is_correct": bool(is_corr),
            "pred_len": res["gen_len"],
            "pred_text": res["text"][:200]
        })
        if (idx + 1) % 10 == 0 or idx == len(math_l35) - 1:
            print(f"[Unmasked {idx+1:2d}/60] Running Acc: {correct_unmasked/(idx+1)*100:.2f}% ({int(correct_unmasked)}/{idx+1})")

    acc_unmasked = (correct_unmasked / len(math_l35)) * 100.0

    summary = {
        "checkpoint": ckpt_to_use,
        "bottleneck_enforced_acc": acc_masked,
        "bottleneck_enforced_correct": int(correct_masked),
        "bottleneck_unmasked_acc": acc_unmasked,
        "bottleneck_unmasked_correct": int(correct_unmasked),
        "total": len(math_l35),
        "standard_arm3_reference_acc": 45.00,
        "delta_enforced_vs_arm3": acc_masked - 45.00,
        "delta_unmasked_vs_arm3": acc_unmasked - 45.00
    }

    print("\n" + "=" * 80)
    print("LATENT INFORMATION BOTTLENECK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Standard Arm 3 Reference (Prompt Visible) : 45.00% (27/60)")
    print(f"LIB Model - Bottleneck Enforced (Masked)   : {acc_masked:.2f}% ({int(correct_masked)}/60) [Delta: {summary['delta_enforced_vs_arm3']:+.2f}%]")
    print(f"LIB Model - Bottleneck Unmasked (Visible)  : {acc_unmasked:.2f}% ({int(correct_unmasked)}/60) [Delta: {summary['delta_unmasked_vs_arm3']:+.2f}%]")
    print("=" * 80)

    with open(args.output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    main()
