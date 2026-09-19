#!/usr/bin/env python3
"""
scripts/99_train_register_ladder_controls.py
Train the Matched Discrete Controls for the Register-Bundle Ladder on Qwen/Qwen3-1.7B:
1. Arm 1b-R: Headers Only (3 headers, 0 latents / 0 pause tokens)
2. Arm 2b-R: Headers + 24 Pause Tokens (3 headers + 8 <pause> tokens per bundle)

Trained with identical:
- 1,267 train / 100 dev traces from curated_train_v1_1_dilution10pct_qwen3_1.7b.jsonl
- Stationary canonical semantic headers
- LoRA rank 32, alpha 64, lr 1e-4, 2 epochs, B_eff=16
- Target loss strictly on answer phase
"""

import os
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

PAUSE_TOKEN = "<pause>"
CANONICAL_HEADERS = [
    "[R1: Identify givens, constraints, and target variable]\n",
    "[R2: Compute intermediate operations and verify relations]\n",
    "[R3: Execute final deduction and verify constraints]\n"
]

def format_control_sequence(sample, mode="headers_only", k_per_bundle=8):
    """
    Constructs the exact text sequence:
    Prompt + R1 + (optional pauses) + R2 + (optional pauses) + R3 + (optional pauses) + \n</think>\n\n
    """
    prompt = sample.get("prompt", "")
    if not prompt.endswith("<think>\n"):
        prompt = prompt.replace("<think>\n\n</think>\n\n", "<think>\n")
        if not prompt.endswith("<think>\n"):
            prompt += "<think>\n"

    conditioning = prompt
    for r in range(3):
        conditioning += CANONICAL_HEADERS[r]
        if mode == "headers_pause":
            conditioning += (PAUSE_TOKEN * k_per_bundle)

    conditioning += "\n</think>\n\n"

    ans_text = sample.get("nonthinking_answer", "")
    if not ans_text.endswith("<|im_end|>"):
        ans_text += "<|im_end|>"

    return conditioning, ans_text

def compute_loss(model, tokenizer, sample, mode, k_per_bundle, max_target_len, device):
    cond_text, ans_text = format_control_sequence(sample, mode=mode, k_per_bundle=k_per_bundle)
    enc_cond = tokenizer.encode(cond_text, add_special_tokens=False)
    enc_ans = tokenizer.encode(ans_text, add_special_tokens=False)
    if len(enc_ans) > max_target_len:
        enc_ans = enc_ans[:max_target_len]

    if len(enc_cond) + len(enc_ans) > 2560:
        return None

    input_ids = torch.tensor([enc_cond + enc_ans], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, :len(enc_cond)] = -100  # Loss strictly on answer

    out = model(input_ids=input_ids, labels=labels)
    return out.loss

def train_control(args):
    print("=" * 80)
    print(f"TRAINING REGISTER LADDER CONTROL: {args.mode.upper()}")
    print(f"Device: {args.device} | Output: {args.output_dir}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    all_records = []
    with open(args.train_file) as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    random.seed(42)
    shuffled_all = list(all_records)
    random.shuffle(shuffled_all)
    dev_records = shuffled_all[:100]
    train_records = shuffled_all[100:]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.mode == "headers_pause" and PAUSE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    if args.mode == "headers_pause":
        base_model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base_model.enable_input_require_grads()

    modules_to_save = ["embed_tokens"] if args.mode == "headers_pause" else None
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=modules_to_save,
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_records) * args.epochs) // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(5, int(0.1 * total_steps)),
        num_training_steps=total_steps
    )

    step_count = 0
    accum_loss = 0.0
    start_time = time.time()
    best_dev_loss = float("inf")

    for epoch in range(args.epochs):
        random.seed(42 + epoch)
        shuffled = list(train_records)
        random.shuffle(shuffled)

        for idx, sample in enumerate(shuffled):
            try:
                loss = compute_loss(model, tokenizer, sample, args.mode, args.k_per_bundle, args.max_target_len, args.device)
                if loss is None:
                    continue
                (loss / args.grad_accum_steps).backward()
                accum_loss += loss.item()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                accum_loss = 0.0
                continue

            if (idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1
                avg_loss = accum_loss / args.grad_accum_steps
                accum_loss = 0.0

                if step_count % 25 == 0 or step_count == total_steps:
                    # Dev Loss
                    model.eval()
                    dev_losses = []
                    with torch.no_grad():
                        for d_sample in dev_records:
                            try:
                                d_l = compute_loss(model, tokenizer, d_sample, args.mode, args.k_per_bundle, args.max_target_len, args.device)
                                if d_l is not None:
                                    dev_losses.append(d_l.item())
                            except torch.cuda.OutOfMemoryError:
                                torch.cuda.empty_cache()
                                continue
                    model.train()
                    cur_dev = float(np.mean(dev_losses))
                    elapsed = time.time() - start_time
                    vram = torch.cuda.max_memory_allocated(device=args.device) / (1024**3)
                    print(f"Step {step_count:3d}/{total_steps} | Loss: {avg_loss:.4f} | Dev Loss: {cur_dev:.4f} | VRAM: {vram:.1f}GB | {elapsed:.0f}s")
                    if cur_dev < best_dev_loss:
                        best_dev_loss = cur_dev
                        model.save_pretrained(os.path.join(args.output_dir, "best_checkpoint"))
                        tokenizer.save_pretrained(os.path.join(args.output_dir, "best_checkpoint"))

    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_checkpoint"))
    meta = {
        "mode": args.mode,
        "best_dev_loss": best_dev_loss,
        "duration_seconds": time.time() - start_time
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Completed {args.mode}! Saved to {args.output_dir} in {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["headers_only", "headers_pause"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_file", type=str, default="data/curated_train_v1_1_dilution10pct_qwen3_1.7b.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k_per_bundle", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_target_len", type=int, default=1024)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    args = parser.parse_args()
    train_control(args)
