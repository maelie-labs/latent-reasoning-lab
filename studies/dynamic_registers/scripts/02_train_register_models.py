#!/usr/bin/env python3
"""
studies/dynamic_registers/scripts/02_train_register_models.py
Train LoRA models for the Staged Dynamic Registers study on Qwen/Qwen3-1.7B:
1. Arm 1: Dynamic Discrete Registers (model learns to emit structured key-value registers + answer)
2. Arm 2: Matched Structural Filler (model learns to emit length-matched filler registers + answer)

Parity Invariant:
- Same learning rate (1e-4), optimizer, schedule, random seed (42), and effective batch size (16).
- Strict loss on registers + transition + answer.
"""

import os
import sys
import json
import time
import random
import argparse
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

def format_training_sequence(sample: Dict[str, Any]) -> Tuple[str, str]:
    """
    Constructs:
    prompt (conditioning) and target (registers + transition + answer).
    """
    prompt = sample.get("prompt", "")
    if not prompt.endswith("<think>\n"):
        prompt = prompt.replace("<think>\n\n</think>\n\n", "<think>\n")
        if not prompt.endswith("<think>\n"):
            prompt += "<think>\n"

    reg_block = sample.get("register_block", "").rstrip("\n")
    ans_text = sample.get("nonthinking_answer", "").strip()
    if not ans_text.endswith("<|im_end|>"):
        ans_text += "<|im_end|>"

    # Target starts with register block, then transition, then worked answer
    target = reg_block + "\n</think>\n\n" + ans_text
    return prompt, target

def compute_sample_loss(model, tokenizer, sample, max_seq_len, device):
    prompt, target = format_training_sequence(sample)
    enc_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    enc_target = tokenizer.encode(target, add_special_tokens=False)

    if len(enc_prompt) + len(enc_target) > max_seq_len:
        enc_target = enc_target[: max_seq_len - len(enc_prompt)]

    input_ids = torch.tensor([enc_prompt + enc_target], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, :len(enc_prompt)] = -100  # Loss strictly on registers + transition + answer

    out = model(input_ids=input_ids, labels=labels)
    return out.loss

def evaluate_dev_loss(model, tokenizer, dev_records, max_seq_len, device, max_eval=100):
    model.eval()
    losses = []
    with torch.no_grad():
        for s in dev_records[:max_eval]:
            loss = compute_sample_loss(model, tokenizer, s, max_seq_len, device)
            if loss is not None and not torch.isnan(loss):
                losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else 999.0

def train(args):
    print("=" * 80)
    print(f"TRAINING STAGED DYNAMIC REGISTERS: MODE = {args.mode.upper()}")
    print(f"Device: {args.device} | Output Dir: {args.output_dir}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # Seed all RNGs for deterministic LoRA initialization and training
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Determine input files based on mode
    if args.mode == "dynamic_registers":
        train_path = os.path.join(args.data_dir, "train_dynamic_registers.jsonl")
        dev_path = os.path.join(args.data_dir, "dev_dynamic_registers.jsonl")
    elif args.mode == "matched_filler":
        train_path = os.path.join(args.data_dir, "train_matched_filler_registers.jsonl")
        dev_path = os.path.join(args.data_dir, "dev_matched_filler_registers.jsonl")
    elif args.mode == "headers_only":
        train_path = os.path.join(args.data_dir, "train_headers_only.jsonl")
        dev_path = os.path.join(args.data_dir, "dev_headers_only.jsonl")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"Reading train: {train_path}")
    print(f"Reading dev: {dev_path}")

    train_records = []
    with open(train_path) as f:
        for line in f:
            if line.strip():
                train_records.append(json.loads(line))

    dev_records = []
    with open(dev_path) as f:
        for line in f:
            if line.strip():
                dev_records.append(json.loads(line))

    print(f"Loaded {len(train_records)} train records and {len(dev_records)} dev records.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {args.model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base_model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    total_steps = (len(train_records) * args.epochs) // args.accum_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )

    # Initial dev evaluation
    init_dev_loss = evaluate_dev_loss(model, tokenizer, dev_records, args.max_seq_len, args.device)
    print(f"Step 0 Initial Dev Loss: {init_dev_loss:.4f}")

    best_dev_loss = init_dev_loss
    best_checkpoint_dir = os.path.join(args.output_dir, "best_checkpoint")

    global_step = 0
    accum_loss = 0.0
    accum_count = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        # Shuffle deterministically per epoch
        random.seed(args.seed + epoch)
        indices = list(range(len(train_records)))
        random.shuffle(indices)

        for idx in indices:
            sample = train_records[idx]
            loss = compute_sample_loss(model, tokenizer, sample, args.max_seq_len, args.device)
            if loss is None:
                continue

            loss_scaled = loss / args.accum_steps
            loss_scaled.backward()
            accum_loss += loss.item()
            accum_count += 1

            if accum_count % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_train_loss = accum_loss / args.accum_steps
                accum_loss = 0.0

                if global_step % args.eval_every == 0 or global_step == total_steps:
                    dev_loss = evaluate_dev_loss(model, tokenizer, dev_records, args.max_seq_len, args.device)
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step}/{total_steps} | Train Loss: {avg_train_loss:.4f} | Dev Loss: {dev_loss:.4f} | LR: {lr:.2e} | Elapsed: {elapsed:.1f}s")

                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        print(f"--> New best dev loss: {best_dev_loss:.4f}. Saving checkpoint to {best_checkpoint_dir}")
                        model.save_pretrained(best_checkpoint_dir)
                        tokenizer.save_pretrained(best_checkpoint_dir)

    # Save final
    final_dir = os.path.join(args.output_dir, "final_checkpoint")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Ensure best_checkpoint exists even if dev loss never beat initial
    if not os.path.exists(best_checkpoint_dir):
        print(f"Notice: saving final checkpoint to {best_checkpoint_dir}")
        model.save_pretrained(best_checkpoint_dir)
        tokenizer.save_pretrained(best_checkpoint_dir)

    train_meta = {
        "mode": args.mode,
        "model_id": args.model_id,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "epochs": args.epochs,
        "effective_batch_size": args.accum_steps,
        "init_dev_loss": init_dev_loss,
        "best_dev_loss": best_dev_loss,
        "total_steps": global_step,
        "duration_seconds": time.time() - start_time
    }
    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump(train_meta, f, indent=2)

    print(f"\nTraining completed! Best dev loss: {best_dev_loss:.4f}. Checkpoint at {best_checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["dynamic_registers", "matched_filler", "headers_only"])
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data_dir", type=str, default="studies/dynamic_registers/data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--max_seq_len", type=int, default=2560)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    args = parser.parse_args()
    train(args)
