#!/usr/bin/env python3
"""
studies/tool_grounded_scratchpad/scripts/02_train_tool_scratchpad.py
Training Harness for Study 4: Tool-Grounded In-Place Working Memory Scratchpad.

Trains LoRA adapter on Qwen/Qwen3-4B to invoke `update_scratchpad` tool calls
and condition on cumulative working memory tables to produce final answers.

Invariants:
- Dedicated GPU 1 (CUDA_VISIBLE_DEVICES=1)
- Pure bfloat16
- Loss masked strictly on assistant spans (zero loss on system, question, tool responses)
- Evaluates dev loss every 25 steps, saving best checkpoint
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

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

def create_loss_masked_inputs(full_text: str, tokenizer, max_seq_len: int, device: str):
    enc = tokenizer.encode(full_text, add_special_tokens=False)
    if len(enc) > max_seq_len:
        enc = enc[:max_seq_len]

    labels = [-100] * len(enc)
    asst_start_str = "<|im_start|>assistant\n"
    im_end_str = "<|im_end|>"

    cur_pos = 0
    while True:
        start_idx = full_text.find(asst_start_str, cur_pos)
        if start_idx == -1:
            break
        content_start = start_idx + len(asst_start_str)
        end_idx = full_text.find(im_end_str, content_start)
        if end_idx == -1:
            end_idx = len(full_text)

        prefix_tokens = len(tokenizer.encode(full_text[:content_start], add_special_tokens=False))
        span_tokens = len(tokenizer.encode(full_text[:end_idx + len(im_end_str)], add_special_tokens=False))

        for t in range(prefix_tokens, min(span_tokens, len(labels))):
            labels[t] = enc[t]

        cur_pos = end_idx + len(im_end_str)

    input_ids = torch.tensor([enc], dtype=torch.long, device=device)
    labels_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    return input_ids, labels_tensor

def evaluate_dev(model, tokenizer, dev_records, max_seq_len, device, max_eval=30):
    model.eval()
    losses = []
    with torch.no_grad():
        for s in dev_records[:max_eval]:
            input_ids, labels = create_loss_masked_inputs(s["full_text"], tokenizer, max_seq_len, device)
            out = model(input_ids=input_ids, labels=labels)
            if out.loss is not None and not torch.isnan(out.loss):
                losses.append(out.loss.item())
    model.train()
    return float(np.mean(losses)) if losses else 999.0

def main():
    parser = argparse.ArgumentParser(description="Train Study 4 Tool Scratchpad Adapter")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--device", type=str, default="cuda:0")
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
        args.output_dir = os.path.join(PROJECT_ROOT, f"checkpoints/lora_tool_scratchpad_{tag}")

    train_path = os.path.join(SCRIPT_DIR, "../data/train_tool_scratchpad.jsonl")
    dev_path = os.path.join(SCRIPT_DIR, "../data/dev_tool_scratchpad.jsonl")

    print("=" * 80)
    print(f"=== STUDY 4 TRAINING: TOOL SCRATCHPAD for {args.model_id} ===")
    print(f"Device: {args.device} | Steps: {args.steps} | LR: {args.lr} | LoRA r={args.lora_r}, a={args.lora_alpha}")
    print(f"Train Dataset: {train_path}")
    print(f"Dev Dataset:   {dev_path}")
    print(f"Output Checkpoint: {args.output_dir}")
    print("=" * 80)

    train_records = load_records(train_path)
    dev_records = load_records(dev_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base model {args.model_id} in pure bfloat16 onto {args.device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.steps * 0.05),
        num_training_steps=args.steps
    )

    best_dev_loss = float("inf")
    best_ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Evaluating pre-training dev loss...")
    initial_dev = evaluate_dev(model, tokenizer, dev_records, args.max_seq_len, args.device)
    print(f"Step 0 Initial Dev Loss: {initial_dev:.4f}")

    model.train()
    step = 0
    accum_loss = 0.0
    t0 = time.time()
    sample_idx = 0
    num_samples = len(train_records)

    optimizer.zero_grad()

    while step < args.steps:
        sample = train_records[sample_idx % num_samples]
        sample_idx += 1

        input_ids, labels = create_loss_masked_inputs(sample["full_text"], tokenizer, args.max_seq_len, args.device)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss / args.grad_accum_steps
        loss.backward()
        accum_loss += loss.item() * args.grad_accum_steps

        if sample_idx % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 5 == 0 or step == 1:
                lr_cur = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                print(f"Step {step}/{args.steps} | Train Loss: {accum_loss / 5:.4f} | LR: {lr_cur:.2e} | {steps_per_sec:.2f} step/s")
                accum_loss = 0.0

            if step % 25 == 0 or step == args.steps:
                dev_loss = evaluate_dev(model, tokenizer, dev_records, args.max_seq_len, args.device)
                print(f">>> Step {step} Validation Dev Loss: {dev_loss:.4f} (Best: {best_dev_loss:.4f})")
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    print(f"*** Saving new best checkpoint to {best_ckpt_dir} (loss: {dev_loss:.4f}) ***")
                    model.save_pretrained(best_ckpt_dir)
                    tokenizer.save_pretrained(best_ckpt_dir)

    print("=" * 80)
    print(f"TRAINING COMPLETE! Best Dev Loss: {best_dev_loss:.4f}")
    print(f"Best Checkpoint: {best_ckpt_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
