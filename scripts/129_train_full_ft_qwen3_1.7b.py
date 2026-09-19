#!/usr/bin/env python3
"""
scripts/129_train_full_ft_qwen3_1.7b.py
Full Fine-Tuning Training Script for Qwen3-1.7B Continuous Latent Recurrence (Gate 2 Retrofit Test).

Replicates the exact v1.1 lambda=1.0 step-distillation arm, but updating ALL 1.7B parameters
(no LoRA, no frozen blocks) to determine whether Gate 2 donor steering moves when adapter
capacity constraints are removed.

Specifications:
- Model: Qwen/Qwen3-1.7B (pure bfloat16)
- Parameters: 1,720,574,976 (100% trainable)
- Optimizer: bitsandbytes.optim.AdamW8bit (lr=1e-5, weight_decay=0.01)
- Schedule: 5% warmup, cosine decay over 341 steps (2 epochs, effective batch 8)
- Memory: Non-reentrant gradient checkpointing, peak VRAM ~13GB on RTX PRO 4500 (32GB)
- Attention: Pinned torch.nn.attention.SDPBackend.MATH for batch-invariance
- Objective: L_total = L_CE(nonthinking_answer) + lambda_align * L_distill(student_states, teacher_states)
  with lambda_align=1.0, K=6, alpha=0.011440
"""

import os
import re
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

SCALE_FACTOR = 0.011440
K_TOKENS = 6

def load_dataset(file_path):
    records = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def compute_loss(model, tokenizer, sample, teacher_cache, k_tokens, scale_factor, lambda_align, max_target_len, device):
    prompt = sample.get("prompt", "")
    if not prompt.endswith("<think>\n"):
        prompt = prompt.replace("<think>\n\n</think>\n\n", "<think>\n")
        if not prompt.endswith("<think>\n"):
            prompt += "<think>\n"
            
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    ans_text = "\n</think>\n\n" + sample.get("nonthinking_answer", "")
    enc_ans = tokenizer.encode(ans_text, add_special_tokens=False)
    if len(enc_ans) > max_target_len:
        enc_ans = enc_ans[:max_target_len]
    target_ids = torch.tensor([enc_ans], dtype=torch.long, device=device)
    
    L_prompt = prompt_ids.shape[1]
    backbone = getattr(model, "model", model)
    
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        # Prefill prompt
        out = model(input_ids=prompt_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        
        student_states = []
        for k in range(k_tokens):
            pos = torch.tensor([[L_prompt + k]], device=device, dtype=torch.long)
            scaled_latent = curr_latent * scale_factor
            step_out = backbone(
                inputs_embeds=scaled_latent,
                position_ids=pos,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]
            student_states.append(curr_latent)
            
        logits_0 = model.lm_head(curr_latent)
        if target_ids.shape[1] > 1:
            out_target = model(input_ids=target_ids[:, :-1], past_key_values=past_kv, use_cache=True)
            all_logits = torch.cat([logits_0, out_target.logits], dim=1)
        else:
            all_logits = logits_0
            
    vocab_size = all_logits.size(-1)
    ce_loss = F.cross_entropy(all_logits.view(-1, vocab_size), target_ids.view(-1))
    
    distill_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
    prob_id = sample.get("id")
    if teacher_cache and prob_id in teacher_cache and lambda_align > 0.0:
        targets_dict = teacher_cache[prob_id]
        k_key = f"K{k_tokens}"
        if k_key in targets_dict:
            teacher_targets = targets_dict[k_key].to(device)
            cos_sims = [1.0 - F.cosine_similarity(student_states[k].squeeze(), teacher_targets[k], dim=-1) for k in range(k_tokens)]
            distill_loss = torch.stack(cos_sims).mean()
            
    total_loss = ce_loss + lambda_align * distill_loss
    return total_loss, ce_loss.item(), distill_loss.item()

def evaluate_dev_loss(model, tokenizer, dev_dataset, teacher_cache, k_tokens, scale_factor, lambda_align, max_target_len, device):
    model.eval()
    losses = []
    ce_losses = []
    distill_losses = []
    with torch.no_grad():
        for sample in dev_dataset:
            total_l, ce_l, dist_l = compute_loss(
                model, tokenizer, sample, teacher_cache, k_tokens, scale_factor,
                lambda_align, max_target_len, device
            )
            losses.append(total_l.item())
            ce_losses.append(ce_l)
            distill_losses.append(dist_l)
    model.train()
    mean_total = float(np.mean(losses))
    mean_ce = float(np.mean(ce_losses))
    mean_distill = float(np.mean(distill_losses)) if distill_losses else 0.0
    mean_cos_sim = float(1.0 - mean_distill) if distill_losses else 0.0
    return {
        "total_loss": mean_total,
        "ce_loss": mean_ce,
        "distill_loss": mean_distill,
        "alignment_cosine_sim": mean_cos_sim
    }

def main():
    parser = argparse.ArgumentParser(description="Qwen3-1.7B Full Fine-Tuning Harness")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--k_tokens", type=int, default=6)
    parser.add_argument("--output_dir", type=str, default="checkpoints/full_ft_arm3_v1_1_qwen3_1.7b_k6_lambda1.0")
    parser.add_argument("--train_file", type=str, default="data/curated_train_v1_1_dilution10pct_qwen3_1.7b.jsonl")
    parser.add_argument("--dev_file", type=str, default="data/curated_dev_v1_1_final_qwen3_1.7b.jsonl")
    parser.add_argument("--teacher_cache_file", type=str, default="data/teacher_cot_states_qwen3_1.7b.pt")
    parser.add_argument("--max_target_len", type=int, default=2048)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(project_root, args.train_file)
    dev_path = os.path.join(project_root, args.dev_file)
    teacher_cache_path = os.path.join(project_root, args.teacher_cache_file)
    output_dir = os.path.join(project_root, args.output_dir)

    print("=" * 80)
    print(f"FULL FINE-TUNING RETROFIT TEST: Qwen3-1.7B Latent Channel (Gate 2 Only)")
    print(f"Output Directory:   {output_dir}")
    print(f"Model ID:           {args.model_id}")
    print(f"Device:             {args.device}")
    print(f"Learning Rate:      {args.lr}")
    print(f"Lambda Align:       {args.lambda_align}")
    print(f"K Tokens:           {args.k_tokens}")
    print(f"Alpha Scale Factor: {SCALE_FACTOR:.6f}")
    print("=" * 80)

    train_dataset = load_dataset(train_path)
    dev_dataset = load_dataset(dev_path)
    print(f"Loaded {len(train_dataset)} train records and {len(dev_dataset)} dev records.")

    print(f"Loading teacher CoT state cache from: {teacher_cache_path}")
    teacher_cache = torch.load(teacher_cache_path, map_location="cpu")
    print(f"Loaded {len(teacher_cache)} teacher state records.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base model in pure bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )

    # Make 100% of parameters trainable
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        p.requires_grad = True
        total_params += p.numel()
        trainable_params += p.numel()
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} (100.0% FULL FINE-TUNING)")
    assert trainable_params == total_params

    # Gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # 8-bit AdamW optimizer
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_dataset) * args.epochs) // args.grad_accum_steps
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"\nBeginning Full-FT Training ({args.epochs} epochs, {total_steps} gradient steps, warmup {warmup_steps} steps)...")
    step_count = 0
    accum_loss = 0.0
    accum_ce = 0.0
    accum_distill = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        random.seed(42 + epoch)
        shuffled = list(train_dataset)
        random.shuffle(shuffled)

        for idx, sample in enumerate(shuffled):
            try:
                loss, ce_val, dist_val = compute_loss(
                    model, tokenizer, sample, teacher_cache, args.k_tokens, SCALE_FACTOR,
                    args.lambda_align, args.max_target_len, args.device
                )
                loss_scaled = loss / args.grad_accum_steps
                loss_scaled.backward()

                accum_loss += loss.item()
                accum_ce += ce_val
                accum_distill += dist_val
            except torch.cuda.OutOfMemoryError:
                print(f"Warning: CUDA OOM on sample {idx}, clearing cache and skipping to protect run.", flush=True)
                torch.cuda.empty_cache()
                continue

            if (idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 10 == 0:
                    torch.cuda.empty_cache()

                if step_count % 50 == 0 or step_count == total_steps:
                    avg_loss = accum_loss / args.grad_accum_steps
                    avg_ce = accum_ce / args.grad_accum_steps
                    avg_dist = accum_distill / args.grad_accum_steps
                    accum_loss, accum_ce, accum_distill = 0.0, 0.0, 0.0
                    elapsed = time.time() - start_time
                    est_total = (elapsed / step_count) * total_steps
                    remaining = est_total - elapsed
                    print(
                        f"Step [{step_count}/{total_steps}] Epoch {epoch+1} | "
                        f"Total Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Distill: {avg_dist:.4f}) | "
                        f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s",
                        flush=True
                    )

    # Final dev loss evaluation
    print("\n--- Evaluating Final Dev Loss on Held-Out Split ---")
    final_dev_loss = evaluate_dev_loss(
        model, tokenizer, dev_dataset, teacher_cache, args.k_tokens, SCALE_FACTOR,
        args.lambda_align, args.max_target_len, args.device
    )
    print(
        f"Final Dev Loss: {final_dev_loss['total_loss']:.4f} | "
        f"CE Loss: {final_dev_loss['ce_loss']:.4f} | "
        f"Distill Loss: {final_dev_loss['distill_loss']:.4f} | "
        f"Alignment Cosine Sim: {final_dev_loss['alignment_cosine_sim']*100:.2f}%"
    )

    # Save full checkpoint & metadata
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving full-parameter model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "arm": "arm3_full_ft",
        "k_tokens": args.k_tokens,
        "model_id": args.model_id,
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_align": args.lambda_align,
        "scale_factor": SCALE_FACTOR,
        "total_steps": total_steps,
        "final_dev_loss": final_dev_loss["total_loss"],
        "final_ce_loss": final_dev_loss["ce_loss"],
        "final_distill_loss": final_dev_loss["distill_loss"],
        "final_alignment_cosine_sim": final_dev_loss["alignment_cosine_sim"],
        "duration_seconds": time.time() - start_time,
        "is_full_ft": True
    }
    with open(os.path.join(output_dir, "arm_training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Checkpoint and metadata saved successfully to: {output_dir}")
    print("=== Full Fine-Tuning Complete [SUCCESS] ===")

if __name__ == "__main__":
    main()
