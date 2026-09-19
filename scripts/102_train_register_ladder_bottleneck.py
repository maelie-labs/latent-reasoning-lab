#!/usr/bin/env python3
"""
scripts/102_train_register_ladder_bottleneck.py
Latent Information Bottleneck (LIB) Training for Register-Bundle Ladder on Qwen/Qwen3-1.7B.

Forces the answer decoder to route reasoning through the continuous latent states by
blocking attention to prompt tokens during the answer phase.

Key Architecture:
- 3 Canonical Register Headers: [R1: Givens], [R2: Intermediate Ops], [R3: Deduction]
- K=8 latents per bundle (24 total continuous unrolled vector passes)
- Prompt is fully visible during ladder reasoning (Bundle 1, 2, 3)
- Attention to prompt tokens (0:L_prompt) is strictly masked during transition + answer generation
- Full Backpropagation Through Time (BPTT) across unrolling passes
- Loss strictly on transition + answer tokens
- Evaluates dev loss every 25 steps; saves best_checkpoint
"""

import os
import re
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

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

def get_calibrated_alpha(model_id):
    return CALIBRATED_ALPHAS.get(model_id.lower(), 0.011440)

def compute_loss_register_ladder_bottleneck(
    model, tokenizer, sample, k_per_bundle, num_rungs, scale_factor,
    max_target_len, device
):
    """
    Computes loss under the Latent Information Bottleneck:
    Prompt is visible during ladder unrolling, but completely masked during answer generation.
    """
    prompt = sample.get("prompt", "")
    if not prompt.endswith("<think>\n"):
        prompt = prompt.replace("<think>\n\n</think>\n\n", "<think>\n")
        if not prompt.endswith("<think>\n"):
            prompt += "<think>\n"

    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    L_prompt = enc_prompt.input_ids.shape[1]

    # Guard: drop extreme outlier sequences to respect 4,096 cap
    if L_prompt > 3000:
        return None, 0.0

    # 1. Prefill Prompt
    out = model(input_ids=enc_prompt.input_ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    curr_latent = out.hidden_states[-1][:, -1:, :]
    curr_seq_len = L_prompt

    backbone = model.base_model.model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "model") else model

    # 2. Interleaved Register-Bundle Loop (Headers + Latents)
    for r in range(num_rungs):
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
        curr_seq_len += hdr_len

        # Unroll K latent steps
        for k in range(k_per_bundle):
            step_pos = torch.tensor([[curr_seq_len]], device=device, dtype=torch.long)
            scaled = curr_latent * scale_factor
            step_out = backbone(
                inputs_embeds=scaled, position_ids=step_pos, past_key_values=past_kv,
                use_cache=True
            )
            past_kv = step_out.past_key_values
            curr_latent = step_out.last_hidden_state[:, -1:, :] if hasattr(step_out, "last_hidden_state") else step_out.hidden_states[-1][:, -1:, :]
            curr_seq_len += 1

    # 3. Transition Delimiter: \n</think>\n\n with Prompt Attention Masked
    trans_text = "\n</think>\n\n"
    enc_trans = tokenizer(trans_text, return_tensors="pt", add_special_tokens=False).to(device)
    trans_len = enc_trans.input_ids.shape[1]
    trans_pos = torch.arange(curr_seq_len, curr_seq_len + trans_len, device=device).unsqueeze(0)

    # 4D boolean mask for transition: attend to ladder, block prompt (0:L_prompt)
    trans_mask = torch.zeros((1, 1, trans_len, curr_seq_len + trans_len), dtype=torch.bool, device=device)
    for q in range(trans_len):
        trans_mask[0, 0, q, L_prompt : curr_seq_len] = True
        trans_mask[0, 0, q, curr_seq_len : curr_seq_len + q + 1] = True

    trans_out = model(
        input_ids=enc_trans.input_ids, position_ids=trans_pos, past_key_values=past_kv,
        attention_mask=trans_mask, use_cache=True, output_hidden_states=True
    )
    past_kv = trans_out.past_key_values
    curr_seq_len += trans_len

    # 4. Answer Phase with Prompt Attention Masked
    ans_text = sample.get("nonthinking_answer", "")
    if not ans_text.endswith("<|im_end|>"):
        ans_text += "<|im_end|>"
    enc_ans = tokenizer(ans_text, return_tensors="pt", add_special_tokens=False).to(device)
    if enc_ans.input_ids.shape[1] > max_target_len:
        enc_ans.input_ids = enc_ans.input_ids[:, :max_target_len]

    ans_len = enc_ans.input_ids.shape[1]
    ans_pos = torch.arange(curr_seq_len, curr_seq_len + ans_len, device=device).unsqueeze(0)

    # 4D boolean mask for answer: attend to ladder + transition + causal answer, block prompt (0:L_prompt)
    total_ans_kv = curr_seq_len + ans_len
    ans_mask = torch.zeros((1, 1, ans_len, total_ans_kv), dtype=torch.bool, device=device)
    for q in range(ans_len):
        ans_mask[0, 0, q, L_prompt : curr_seq_len] = True
        ans_mask[0, 0, q, curr_seq_len : curr_seq_len + q + 1] = True

    ans_out = model(
        input_ids=enc_ans.input_ids, position_ids=ans_pos, past_key_values=past_kv,
        attention_mask=ans_mask, use_cache=True
    )

    # 5. Full Answer Logits: Combine trans_out last logit (predicting ans[0]) with ans_out logits
    full_logits = torch.cat([trans_out.logits[:, -1:, :], ans_out.logits[:, :-1, :]], dim=1)
    vocab_size = full_logits.size(-1)
    ans_loss = F.cross_entropy(full_logits.reshape(-1, vocab_size), enc_ans.input_ids.reshape(-1))

    return ans_loss, ans_loss.item()

def evaluate_dev_loss(model, tokenizer, dev_records, k_per_bundle, num_rungs, scale_factor, max_target_len, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for sample in dev_records:
            try:
                loss, loss_val = compute_loss_register_ladder_bottleneck(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    k_per_bundle=k_per_bundle,
                    num_rungs=num_rungs,
                    scale_factor=scale_factor,
                    max_target_len=max_target_len,
                    device=device
                )
                if loss is not None:
                    losses.append(loss_val)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
    model.train()
    return float(np.mean(losses)) if losses else float("inf")

def main():
    parser = argparse.ArgumentParser(description="Train Arm 3 Register-Bundle Ladder with Latent Information Bottleneck")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_arm3_register_ladder_bottleneck_qwen3_1.7b")
    parser.add_argument("--train_file", type=str, default="data/curated_train_v1_1_dilution10pct_qwen3_1.7b.jsonl")
    parser.add_argument("--k_per_bundle", type=int, default=8)
    parser.add_argument("--num_rungs", type=int, default=3)
    parser.add_argument("--scale_factor", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_target_len", type=int, default=1024)
    args = parser.parse_args()

    scale_factor = args.scale_factor or get_calibrated_alpha(args.model_id)

    print("=" * 80)
    print("STARTING LATENT INFORMATION BOTTLENECK (LIB) LADDER TRAINING")
    print(f"Model ID        : {args.model_id}")
    print(f"Device          : {args.device} (Dedicated Compute GPU)")
    print(f"Output Dir      : {args.output_dir}")
    print(f"Architecture    : {args.num_rungs} rungs x {args.k_per_bundle} latents = {args.num_rungs * args.k_per_bundle} total latents")
    print(f"Scale Factor    : alpha = {scale_factor:.6f}")
    print(f"Bottleneck Mask : Prompt attention BLOCKED during answer phase")
    print(f"Hyperparameters : lr={args.lr}, r={args.lora_r}, alpha={args.lora_alpha}, epochs={args.epochs}, accum={args.grad_accum_steps}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    all_records = []
    with open(args.train_file, "r") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    random.seed(42)
    shuffled_all = list(all_records)
    random.shuffle(shuffled_all)
    dev_records = shuffled_all[:100]
    train_records = shuffled_all[100:]

    print(f"Dataset: {len(train_records)} train traces, {len(dev_records)} dev traces.")

    # 2. Tokenizer & Base Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.train()
    model.print_trainable_parameters()

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_records) * args.epochs) // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(5, int(0.1 * total_steps)),
        num_training_steps=total_steps
    )

    # Initial Dev Loss
    print("Evaluating initial dev loss...")
    init_dev_loss = evaluate_dev_loss(
        model=model, tokenizer=tokenizer, dev_records=dev_records,
        k_per_bundle=args.k_per_bundle, num_rungs=args.num_rungs,
        scale_factor=scale_factor, max_target_len=args.max_target_len,
        device=args.device
    )
    print(f"Initial Dev Loss: {init_dev_loss:.4f}")

    best_dev_loss = init_dev_loss
    step_count = 0
    accum_loss = 0.0
    start_time = time.time()

    meta = {
        "model_id": args.model_id,
        "k_per_bundle": args.k_per_bundle,
        "num_rungs": args.num_rungs,
        "scale_factor": scale_factor,
        "bottleneck": True,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "init_dev_loss": init_dev_loss,
        "history": []
    }

    print("\nBeginning Bottleneck Ladder Training Loop...")
    for epoch in range(args.epochs):
        random.seed(42 + epoch)
        shuffled = list(train_records)
        random.shuffle(shuffled)

        for idx, sample in enumerate(shuffled):
            try:
                loss, loss_val = compute_loss_register_ladder_bottleneck(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    k_per_bundle=args.k_per_bundle,
                    num_rungs=args.num_rungs,
                    scale_factor=scale_factor,
                    max_target_len=args.max_target_len,
                    device=args.device
                )
                if loss is None:
                    continue

                loss_scaled = loss / args.grad_accum_steps
                loss_scaled.backward()
                accum_loss += loss_val

            except torch.cuda.OutOfMemoryError:
                print(f"[WARN] OOM on sample {idx}, clearing cache...")
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
                    dev_loss = evaluate_dev_loss(
                        model=model, tokenizer=tokenizer, dev_records=dev_records,
                        k_per_bundle=args.k_per_bundle, num_rungs=args.num_rungs,
                        scale_factor=scale_factor, max_target_len=args.max_target_len,
                        device=args.device
                    )
                    elapsed = time.time() - start_time
                    vram = torch.cuda.max_memory_allocated(device=args.device) / (1024**3)
                    lr_curr = scheduler.get_last_lr()[0]

                    print(f"Step {step_count:3d}/{total_steps} | Loss: {avg_loss:.4f} | Dev Loss: {dev_loss:.4f} | LR: {lr_curr:.2e} | VRAM: {vram:.1f}GB | Elapsed: {elapsed:.0f}s")

                    meta["history"].append({
                        "step": step_count,
                        "epoch": epoch,
                        "loss": avg_loss,
                        "dev_loss": dev_loss,
                        "vram_gb": vram,
                        "elapsed_s": elapsed
                    })

                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        best_path = os.path.join(args.output_dir, "best_checkpoint")
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        print(f"  --> Saved new best checkpoint (Dev Loss: {best_dev_loss:.4f}) to {best_path}")

    # Save final
    final_path = os.path.join(args.output_dir, "final_checkpoint")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    meta["final_dev_loss"] = dev_loss
    meta["best_dev_loss"] = best_dev_loss
    meta["total_duration_seconds"] = time.time() - start_time

    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 80)
    print(f"TRAINING COMPLETE! Best Dev Loss: {best_dev_loss:.4f}")
    print(f"Saved best model to {os.path.join(args.output_dir, 'best_checkpoint')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
