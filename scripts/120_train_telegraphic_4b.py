#!/usr/bin/env python3
"""
scripts/120_train_telegraphic_4b.py
Study 3 Training Harness: Telegraphic Propositional CoT & Headers-Only Controls for 4B Models.

Supports:
1. Qwen/Qwen3-4B (Pure Dense Transformer)
2. Qwen/Qwen3.5-4B (Hybrid Gated DeltaNet)

Modes:
- telegraphic_cot: Loss on concise propositional thought + transition + answer
- headers_only: Loss on canonical structural headers + transition + answer
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

def load_records(path: str):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {path}")
    random.seed(42)
    random.shuffle(records)
    return records

def format_sample(sample, model_id):
    prompt = sample["prompt"]
    thought = sample["thought"]
    ans_text = sample["answer"]
    if not ans_text.endswith("<|im_end|>"):
        ans_text += "<|im_end|>"

    # Ensure prompt ends with <think>\n
    if not prompt.endswith("<think>\n"):
        prompt = prompt.replace("<think>\n\n</think>\n\n", "<think>\n")
        if not prompt.endswith("<think>\n"):
            prompt += "<think>\n"

    target = thought.rstrip("\n") + "\n</think>\n\n" + ans_text
    return prompt, target

def compute_loss(model, tokenizer, sample, model_id, max_seq_len, device):
    prompt, target = format_sample(sample, model_id)
    enc_p = tokenizer.encode(prompt, add_special_tokens=False)
    enc_t = tokenizer.encode(target, add_special_tokens=False)

    if len(enc_p) + len(enc_t) > max_seq_len:
        enc_t = enc_t[: max_seq_len - len(enc_p)]

    input_ids = torch.tensor([enc_p + enc_t], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, :len(enc_p)] = -100  # Zero loss on prompt tokens

    out = model(input_ids=input_ids, labels=labels)
    return out.loss

def evaluate_dev(model, tokenizer, dev_records, model_id, max_seq_len, device, max_eval=50):
    model.eval()
    losses = []
    with torch.no_grad():
        for s in dev_records[:max_eval]:
            l = compute_loss(model, tokenizer, s, model_id, max_seq_len, device)
            if l is not None and not torch.isnan(l):
                losses.append(l.item())
    model.train()
    return float(np.mean(losses)) if losses else 999.0

def main():
    parser = argparse.ArgumentParser(description="Train Study 3 Telegraphic CoT & Headers-Only Adapters for 4B Models")
    parser.add_argument("--model_id", type=str, required=True, choices=["Qwen/Qwen3-4B", "Qwen/Qwen3.5-4B"])
    parser.add_argument("--mode", type=str, required=True, choices=["telegraphic_cot", "headers_only"])
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tag = "qwen3_4b" if "Qwen3-4B" in args.model_id else "qwen3_5_4b"
    if args.output_dir is None:
        args.output_dir = f"checkpoints/lora_{args.mode}_{tag}"

    train_path = f"data/train_{args.mode}_{tag}.jsonl"
    dev_path = f"data/dev_{args.mode}_{tag}.jsonl"

    print("=" * 80)
    print(f"=== STUDY 3 TRAINING: {args.mode.upper()} for {args.model_id} ===")
    print(f"Device: {args.device} | Steps: {args.steps} | LR: {args.lr} | LoRA r={args.lora_r}, a={args.lora_alpha}")
    print(f"Train Dataset: {train_path}")
    print(f"Dev Dataset:   {dev_path}")
    print(f"Output Checkpoint: {args.output_dir}")
    print("=" * 80)

    train_records = load_records(train_path)
    dev_records = load_records(dev_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"Loading {args.model_id} in bfloat16 onto {args.device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )

    if "3.5" in args.model_id:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"
        ]
    else:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * args.steps),
        num_training_steps=args.steps
    )

    global_step = 0
    accum_loss = 0.0
    accum_count = 0
    best_dev_loss = float("inf")
    train_log = []
    t0_train = time.time()
    model.train()

    sample_idx = 0
    n_train = len(train_records)

    while global_step < args.steps:
        sample = train_records[sample_idx % n_train]
        sample_idx += 1

        try:
            loss = compute_loss(model, tokenizer, sample, args.model_id, args.max_seq_len, args.device)
            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()
            accum_loss += loss.item()
            accum_count += 1
        except Exception as e:
            print(f"Warning: Step failed on sample {sample.get('id', 'unknown')}: {e}")
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_count = 0
            continue

        if accum_count == args.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            avg_loss = accum_loss / args.grad_accum_steps
            accum_loss = 0.0
            accum_count = 0

            if global_step % 15 == 0 or global_step == args.steps:
                dev_loss = evaluate_dev(model, tokenizer, dev_records, args.model_id, args.max_seq_len, args.device)
                elapsed = time.time() - t0_train
                cur_lr = lr_scheduler.get_last_lr()[0]
                print(f"[{args.mode} | Step {global_step:3d}/{args.steps}] Train Loss: {avg_loss:.4f} | Dev Loss: {dev_loss:.4f} | LR: {cur_lr:.2e} | Elapsed: {elapsed:.1f}s")
                train_log.append({
                    "step": global_step,
                    "train_loss": round(avg_loss, 4),
                    "dev_loss": round(dev_loss, 4),
                    "lr": cur_lr,
                    "elapsed_sec": round(elapsed, 1)
                })

                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_dir = os.path.join(args.output_dir, "best_checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)

    print(f"\nTraining completed in {time.time() - t0_train:.1f}s! Best dev loss: {best_dev_loss:.4f}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "model_id": args.model_id,
        "mode": args.mode,
        "steps": args.steps,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
        "best_dev_loss": round(best_dev_loss, 4),
        "final_train_loss": train_log[-1]["train_loss"] if train_log else None,
        "final_dev_loss": train_log[-1]["dev_loss"] if train_log else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_log": train_log
    }
    meta_path = os.path.join(args.output_dir, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved LoRA adapter checkpoint to: {args.output_dir}")
    print(f"Metadata saved to: {meta_path}")

if __name__ == "__main__":
    main()
