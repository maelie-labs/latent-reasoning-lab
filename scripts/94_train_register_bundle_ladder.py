#!/usr/bin/env python3
"""
scripts/94_train_register_bundle_ladder.py
Train the Register-Bundle Ladder Architecture on Qwen/Qwen3-1.7B.

Replaces verbose discrete CoT with an interleaved ladder of:
- Discrete Register Anchors (subgoals/checkpoints)
- Latent Compute Bundles (K=8 continuous unroll steps per bundle)
- Final Answer Decoding (with optional prompt bottleneck masking)

Optimized for high-throughput single or dual GPU execution (cuda:1 RTX PRO 4500 32GB).
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
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

CALIBRATED_ALPHAS = {
    "qwen/qwen3-1.7b": 0.011440,
    "qwen/qwen3.5-2b": 0.009800,
    "qwen/qwen3-4b": 0.007670,
}

def get_calibrated_alpha(model_id):
    return CALIBRATED_ALPHAS.get(model_id.lower(), 0.011440)

def extract_register_headers(sample, max_rungs=3):
    """
    Extracts up to max_rungs subgoal titles from the sample's nonthinking_answer or teacher_steps.
    Falls back to canonical semantic registers if steps are absent.
    """
    ans = sample.get("nonthinking_answer", "")
    steps = re.findall(r"###\s*(Step \d+:[^\n]+)", ans)
    
    default_headers = [
        "Identify givens, constraints, and target variable",
        "Compute intermediate operations and verify relations",
        "Execute final deduction and verify constraints"
    ]
    
    headers = []
    for r in range(max_rungs):
        if r < len(steps):
            clean_step = steps[r].strip().replace(":", " -")
            headers.append(f"[R{r+1}: {clean_step}]\n")
        else:
            headers.append(f"[R{r+1}: {default_headers[r]}]\n")
    return headers

def compute_loss_register_ladder(model, tokenizer, sample, k_per_bundle, num_rungs, scale_factor, 
                                reg_loss_weight, max_target_len, device, mask_prompt=False):
    """
    Computes joint loss for the Register-Bundle Ladder:
    1. Prefill Prompt (up to <think>\n)
    2. For r in 1..num_rungs:
       a. Ingest Register r text -> Compute CE loss on Register tokens
       b. Unroll K latent steps
    3. Ingest \n</think>\n\n
    4. Ingest Answer -> Compute CE loss on Answer tokens (with optional prompt masking)
    """
    prompt = sample.get("prompt", "")
    if not prompt.endswith("<think>\n"):
        prompt = prompt.replace("<think>\n\n</think>\n\n", "<think>\n")
        if not prompt.endswith("<think>\n"):
            prompt += "<think>\n"
            
    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    L_prompt = enc_prompt.input_ids.shape[1]
    
    # 1. Prefill Prompt
    out = model(input_ids=enc_prompt.input_ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    curr_latent = out.hidden_states[-1][:, -1:, :]
    curr_seq_len = L_prompt
    
    backbone = model.base_model.model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "model") else model
    
    register_headers = extract_register_headers(sample, max_rungs=num_rungs)
    reg_losses = []
    
    # 2. Interleaved Register-Bundle Loop
    for r, header_text in enumerate(register_headers):
        enc_hdr = tokenizer(header_text, return_tensors="pt", add_special_tokens=False).to(device)
        hdr_len = enc_hdr.input_ids.shape[1]
        
        # Ingest Register header tokens
        hdr_pos = torch.arange(curr_seq_len, curr_seq_len + hdr_len, device=device).unsqueeze(0)
        hdr_out = model(input_ids=enc_hdr.input_ids, position_ids=hdr_pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
        past_kv = hdr_out.past_key_values
        curr_latent = hdr_out.hidden_states[-1][:, -1:, :]
        
        # Loss on register tokens
        if hdr_len > 1:
            hdr_logits = hdr_out.logits[:, :-1, :]
            hdr_targets = enc_hdr.input_ids[:, 1:]
            r_loss = F.cross_entropy(hdr_logits.reshape(-1, hdr_logits.size(-1)), hdr_targets.reshape(-1))
            reg_losses.append(r_loss)
            
        curr_seq_len += hdr_len
        
        # Unroll K latent steps
        for k in range(k_per_bundle):
            step_pos = torch.tensor([[curr_seq_len]], device=device, dtype=torch.long)
            scaled = curr_latent * scale_factor
            step_out = backbone(inputs_embeds=scaled, position_ids=step_pos, past_key_values=past_kv, use_cache=True)
            past_kv = step_out.past_key_values
            curr_latent = step_out.last_hidden_state[:, -1:, :] if hasattr(step_out, "last_hidden_state") else step_out.hidden_states[-1][:, -1:, :]
            curr_seq_len += 1

    # 3. Transition: \n</think>\n\n
    trans_text = "\n</think>\n\n"
    enc_trans = tokenizer(trans_text, return_tensors="pt", add_special_tokens=False).to(device)
    trans_len = enc_trans.input_ids.shape[1]
    trans_pos = torch.arange(curr_seq_len, curr_seq_len + trans_len, device=device).unsqueeze(0)
    trans_out = model(input_ids=enc_trans.input_ids, position_ids=trans_pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
    past_kv = trans_out.past_key_values
    curr_latent = trans_out.hidden_states[-1][:, -1:, :]
    curr_seq_len += trans_len
    
    # 4. Answer Phase
    ans_text = sample.get("nonthinking_answer", "")
    if not ans_text.endswith("<|im_end|>"):
        ans_text += "<|im_end|>"
    enc_ans = tokenizer(ans_text, return_tensors="pt", add_special_tokens=False).to(device)
    if enc_ans.input_ids.shape[1] > max_target_len:
        enc_ans.input_ids = enc_ans.input_ids[:, :max_target_len]
        
    ans_len = enc_ans.input_ids.shape[1]
    ans_pos = torch.arange(curr_seq_len, curr_seq_len + ans_len, device=device).unsqueeze(0)
    
    if mask_prompt:
        total_len = curr_seq_len + ans_len
        # 4D attention mask: (batch=1, num_heads=1, q_len=ans_len, total_kv_len=total_len)
        mask_4d = torch.zeros((1, 1, ans_len, total_len), dtype=torch.bfloat16, device=device)
        # Mask out prompt keys
        mask_4d[:, :, :, :L_prompt] = -10000.0
        # Causal mask for answer tokens
        for i in range(ans_len):
            mask_4d[:, :, i, (curr_seq_len + i + 1):] = -10000.0
        ans_out = model(input_ids=enc_ans.input_ids, attention_mask=mask_4d, position_ids=ans_pos, past_key_values=past_kv, use_cache=True)
    else:
        ans_out = model(input_ids=enc_ans.input_ids, position_ids=ans_pos, past_key_values=past_kv, use_cache=True)
        
    ans_logits = ans_out.logits[:, :-1, :]
    ans_targets = enc_ans.input_ids[:, 1:]
    ans_loss = F.cross_entropy(ans_logits.reshape(-1, ans_logits.size(-1)), ans_targets.reshape(-1))
    
    # Aggregate Loss
    avg_reg_loss = torch.stack(reg_losses).mean() if reg_losses else torch.tensor(0.0, device=device)
    total_loss = ans_loss + reg_loss_weight * avg_reg_loss
    
    return total_loss, ans_loss.item(), avg_reg_loss.item()

def main():
    parser = argparse.ArgumentParser(description="Train Register-Bundle Ladder on Qwen/Qwen3-1.7B")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_file", type=str, default="data/curated_train_v1_1_dilution10pct_qwen3_1.7b.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_register_ladder_qwen3_1.7b")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num_rungs", type=int, default=3)
    parser.add_argument("--k_per_bundle", type=int, default=8)
    parser.add_argument("--scale_factor", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_target_len", type=int, default=1024)
    parser.add_argument("--reg_loss_weight", type=float, default=0.5)
    parser.add_argument("--mask_prompt_prob", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    args = parser.parse_args()

    scale_factor = args.scale_factor or get_calibrated_alpha(args.model_id)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("REGISTER-BUNDLE LADDER TRAINING INITIALIZATION")
    print("=" * 75)
    print(f"Model ID            : {args.model_id}")
    print(f"Compute Device      : {args.device}")
    print(f"Dataset             : {args.train_file}")
    print(f"Output Checkpoint   : {args.output_dir}")
    print(f"Ladder Architecture : {args.num_rungs} rungs x {args.k_per_bundle} latent steps = {args.num_rungs * args.k_per_bundle} total latents")
    print(f"Empirical Scale (a) : {scale_factor:.6f}")
    print(f"LoRA Configuration  : Rank {args.lora_r}, Alpha {args.lora_alpha}, lr={args.lr}")
    print(f"Prompt Mask Prob    : {args.mask_prompt_prob:.2f}")
    print("=" * 75)

    # Load dataset
    records = []
    with open(args.train_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.max_samples:
        records = records[:args.max_samples]
    print(f"Loaded {len(records)} training records.")

    # Initialize Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(records) * args.epochs) // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(5, int(0.1 * total_steps)), num_training_steps=total_steps)

    step_count = 0
    accum_total_loss = 0.0
    accum_ans_loss = 0.0
    accum_reg_loss = 0.0
    start_time = time.time()

    meta_log = {
        "model_id": args.model_id,
        "num_rungs": args.num_rungs,
        "k_per_bundle": args.k_per_bundle,
        "scale_factor": scale_factor,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "mask_prompt_prob": args.mask_prompt_prob,
        "training_records": len(records),
        "history": []
    }

    print("\nBeginning Training Loop...")
    for epoch in range(args.epochs):
        random.seed(42 + epoch)
        shuffled = list(records)
        random.shuffle(shuffled)

        for idx, sample in enumerate(shuffled):
            mask_prompt = (random.random() < args.mask_prompt_prob)
            try:
                loss, ans_l, reg_l = compute_loss_register_ladder(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    k_per_bundle=args.k_per_bundle,
                    num_rungs=args.num_rungs,
                    scale_factor=scale_factor,
                    reg_loss_weight=args.reg_loss_weight,
                    max_target_len=args.max_target_len,
                    device=args.device,
                    mask_prompt=mask_prompt
                )
                
                loss_scaled = loss / args.grad_accum_steps
                loss_scaled.backward()
                
                accum_total_loss += loss.item()
                accum_ans_loss += ans_l
                accum_reg_loss += reg_l
            except torch.cuda.OutOfMemoryError:
                print(f"[WARN] OOM on sample {idx}, clearing cache...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                accum_total_loss = 0.0
                accum_ans_loss = 0.0
                accum_reg_loss = 0.0
                continue

            if (idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                avg_total = accum_total_loss / args.grad_accum_steps
                avg_ans = accum_ans_loss / args.grad_accum_steps
                avg_reg = accum_reg_loss / args.grad_accum_steps
                accum_total_loss = 0.0
                accum_ans_loss = 0.0
                accum_reg_loss = 0.0

                elapsed = time.time() - start_time
                vram_used = torch.cuda.max_memory_allocated(device=args.device) / (1024**3)
                curr_lr = scheduler.get_last_lr()[0]

                print(f"Step {step_count:4d}/{total_steps} | Epoch {epoch+1} | "
                      f"Total Loss: {avg_total:.4f} | Ans Loss: {avg_ans:.4f} | Reg Loss: {avg_reg:.4f} | "
                      f"VRAM: {vram_used:.2f} GB | LR: {curr_lr:.2e} | Elapsed: {elapsed:.1f}s")

                meta_log["history"].append({
                    "step": step_count,
                    "epoch": epoch + 1,
                    "total_loss": avg_total,
                    "ans_loss": avg_ans,
                    "reg_loss": avg_reg,
                    "vram_gb": vram_used,
                    "lr": curr_lr
                })

                if step_count % args.save_steps == 0 or step_count == total_steps:
                    save_path = os.path.join(args.output_dir, f"checkpoint-step-{step_count}")
                    model.save_pretrained(save_path)
                    print(f"--> Saved checkpoint to {save_path}")

    # Final Save
    model.save_pretrained(args.output_dir)
    meta_log["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(args.output_dir, "arm_training_meta.json"), "w") as f:
        json.dump(meta_log, f, indent=2)
    print(f"\nTraining Complete! Final model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
