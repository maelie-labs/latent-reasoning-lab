#!/usr/bin/env python3
"""
scripts/96_train_register_ladder_production.py
Production Training for Continuous Latent Recurrence with Register-Bundle Ladder.

Key Invariants Enforced:
1. Intra-Model Isolation (Qwen/Qwen3-1.7B, pure bfloat16, alpha=0.011440).
2. Stationary Canonical Semantic Headers across all train & test instances.
3. Full Answer Phase Loss (including Token 0 from transition logits).
4. Full Backpropagation Through Time (BPTT) across unrolled latent states.
5. Zero Test Contamination (trains strictly on curated train split).
6. High-Throughput Dedicated Compute on GPU 1 (RTX PRO 4500 32GB).
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

def compute_loss_register_ladder_production(
    model, tokenizer, sample, k_per_bundle, num_rungs, scale_factor,
    max_target_len, device, teacher_cache=None, lambda_align=0.0
):
    """
    Production forward + loss computation for Register-Bundle Ladder.
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

    # 2. Interleaved Register-Bundle Loop (Canonical Semantic Headers)
    final_latent_bundle3 = None
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

        if r == num_rungs - 1:
            final_latent_bundle3 = curr_latent

    # 3. Transition Delimiter: \n</think>\n\n
    trans_text = "\n</think>\n\n"
    enc_trans = tokenizer(trans_text, return_tensors="pt", add_special_tokens=False).to(device)
    trans_len = enc_trans.input_ids.shape[1]
    trans_pos = torch.arange(curr_seq_len, curr_seq_len + trans_len, device=device).unsqueeze(0)
    trans_out = model(
        input_ids=enc_trans.input_ids, position_ids=trans_pos, past_key_values=past_kv,
        use_cache=True, output_hidden_states=True
    )
    past_kv = trans_out.past_key_values
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
    ans_out = model(input_ids=enc_ans.input_ids, position_ids=ans_pos, past_key_values=past_kv, use_cache=True)

    # 5. Full Answer Logits: Combine trans_out last logit (predicting ans[0]) with ans_out logits
    # trans_out.logits[:, -1:, :] predicts enc_ans.input_ids[:, 0:1]
    # ans_out.logits[:, :-1, :] predicts enc_ans.input_ids[:, 1:]
    full_logits = torch.cat([trans_out.logits[:, -1:, :], ans_out.logits[:, :-1, :]], dim=1)
    vocab_size = full_logits.size(-1)
    ans_loss = F.cross_entropy(full_logits.reshape(-1, vocab_size), enc_ans.input_ids.reshape(-1))

    # 6. Optional Distillation Loss to Teacher Thinking State
    distill_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
    prob_id = sample.get("id")
    if teacher_cache and prob_id in teacher_cache and lambda_align > 0.0:
        targets_dict = teacher_cache[prob_id]
        if "codi_answer_state" in targets_dict:
            teacher_target = targets_dict["codi_answer_state"].to(device).unsqueeze(0).unsqueeze(0)
            distill_loss = 1.0 - F.cosine_similarity(final_latent_bundle3, teacher_target, dim=-1).mean()

    total_loss = ans_loss + lambda_align * distill_loss
    return total_loss, ans_loss.item(), distill_loss.item()

def evaluate_dev_loss(model, tokenizer, dev_records, k_per_bundle, num_rungs, scale_factor, max_target_len, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for sample in dev_records:
            try:
                loss, ans_l, _ = compute_loss_register_ladder_production(
                    model=model, tokenizer=tokenizer, sample=sample,
                    k_per_bundle=k_per_bundle, num_rungs=num_rungs,
                    scale_factor=scale_factor, max_target_len=max_target_len,
                    device=device, lambda_align=0.0
                )
                losses.append(ans_l)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
    model.train()
    return float(np.mean(losses)) if losses else 0.0

def main():
    parser = argparse.ArgumentParser(description="Production Training for Register-Bundle Ladder on Qwen/Qwen3-1.7B")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_file", type=str, default="data/curated_train_v1_1_dilution10pct_qwen3_1.7b.jsonl")
    parser.add_argument("--teacher_cache", type=str, default="data/teacher_cot_states_qwen3_1.7b.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_arm3_register_ladder_production_qwen3_1.7b")
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
    parser.add_argument("--lambda_align", type=float, default=0.0, help="Weight for cosine distillation to teacher answer state")
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=25)
    args = parser.parse_args()

    scale_factor = args.scale_factor or get_calibrated_alpha(args.model_id)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("REGISTER-BUNDLE LADDER: PRODUCTION TRAINING")
    print(f"Model ID            : {args.model_id}")
    print(f"Compute Device      : {args.device}")
    print(f"Dataset             : {args.train_file}")
    print(f"Output Checkpoint   : {args.output_dir}")
    print(f"Architecture        : {args.num_rungs} rungs x {args.k_per_bundle} latents = {args.num_rungs * args.k_per_bundle} total latents")
    print(f"Empirical Scale (a) : {scale_factor:.6f}")
    print(f"LoRA Configuration  : Rank {args.lora_r}, Alpha {args.lora_alpha}, lr={args.lr}")
    print(f"Effective Batch Size: {args.grad_accum_steps} (grad_accum_steps={args.grad_accum_steps})")
    print(f"Distillation Weight : {args.lambda_align}")
    print("=" * 80)

    # Load records
    all_records = []
    with open(args.train_file) as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    # Hold out 100 dev records
    random.seed(42)
    shuffled_all = list(all_records)
    random.shuffle(shuffled_all)
    dev_records = shuffled_all[:100]
    train_records = shuffled_all[100:]
    print(f"Loaded {len(all_records)} total records -> {len(train_records)} Train / {len(dev_records)} Dev.")

    # Load teacher cache if lambda_align > 0
    teacher_cache = None
    if args.lambda_align > 0.0 and os.path.exists(args.teacher_cache):
        print(f"Loading teacher state cache from {args.teacher_cache}...")
        teacher_cache = torch.load(args.teacher_cache, map_location="cpu", weights_only=False)
        print(f"Loaded {len(teacher_cache)} teacher cache entries.")

    # Tokenizer & Model
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
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_records) * args.epochs) // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(5, int(0.1 * total_steps)),
        num_training_steps=total_steps
    )

    # Initial Dev Loss
    init_dev_loss = evaluate_dev_loss(
        model, tokenizer, dev_records, args.k_per_bundle, args.num_rungs,
        scale_factor, args.max_target_len, args.device
    )
    print(f"\nInitial Pre-Training Dev Loss: {init_dev_loss:.4f}\n")

    step_count = 0
    accum_total_loss = 0.0
    accum_ans_loss = 0.0
    accum_dist_loss = 0.0
    start_time = time.time()

    meta_log = {
        "model_id": args.model_id,
        "num_rungs": args.num_rungs,
        "k_per_bundle": args.k_per_bundle,
        "scale_factor": scale_factor,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "lambda_align": args.lambda_align,
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "init_dev_loss": init_dev_loss,
        "history": []
    }

    best_dev_loss = init_dev_loss

    print("Beginning Production Training Loop...")
    for epoch in range(args.epochs):
        random.seed(42 + epoch)
        shuffled = list(train_records)
        random.shuffle(shuffled)

        for idx, sample in enumerate(shuffled):
            try:
                loss, ans_l, dist_l = compute_loss_register_ladder_production(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    k_per_bundle=args.k_per_bundle,
                    num_rungs=args.num_rungs,
                    scale_factor=scale_factor,
                    max_target_len=args.max_target_len,
                    device=args.device,
                    teacher_cache=teacher_cache,
                    lambda_align=args.lambda_align
                )
                loss_scaled = loss / args.grad_accum_steps
                loss_scaled.backward()

                accum_total_loss += loss.item()
                accum_ans_loss += ans_l
                accum_dist_loss += dist_l
            except torch.cuda.OutOfMemoryError:
                print(f"[WARN] OOM on sample {idx}, clearing cache...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                accum_total_loss = 0.0
                accum_ans_loss = 0.0
                accum_dist_loss = 0.0
                continue

            if (idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                avg_total = accum_total_loss / args.grad_accum_steps
                avg_ans = accum_ans_loss / args.grad_accum_steps
                avg_dist = accum_dist_loss / args.grad_accum_steps
                accum_total_loss = 0.0
                accum_ans_loss = 0.0
                accum_dist_loss = 0.0

                elapsed = time.time() - start_time
                vram_used = torch.cuda.max_memory_allocated(device=args.device) / (1024**3)
                curr_lr = scheduler.get_last_lr()[0]

                print(f"Step {step_count:3d}/{total_steps} | Ep {epoch+1} | "
                      f"Loss: {avg_total:.4f} (Ans: {avg_ans:.4f}, Dist: {avg_dist:.4f}) | "
                      f"VRAM: {vram_used:.1f}GB | LR: {curr_lr:.2e} | {elapsed:.0f}s")

                log_entry = {
                    "step": step_count,
                    "epoch": epoch + 1,
                    "total_loss": avg_total,
                    "ans_loss": avg_ans,
                    "dist_loss": avg_dist,
                    "vram_gb": vram_used,
                    "lr": curr_lr
                }

                if step_count % args.eval_steps == 0 or step_count == total_steps:
                    dev_l = evaluate_dev_loss(
                        model, tokenizer, dev_records, args.k_per_bundle, args.num_rungs,
                        scale_factor, args.max_target_len, args.device
                    )
                    print(f"--> [Step {step_count}] Dev Loss: {dev_l:.4f} (Best: {best_dev_loss:.4f})")
                    log_entry["dev_loss"] = dev_l
                    if dev_l < best_dev_loss:
                        best_dev_loss = dev_l
                        best_path = os.path.join(args.output_dir, "best_checkpoint")
                        model.save_pretrained(best_path)
                        print(f"--> [*] Saved new best model to {best_path}")

                meta_log["history"].append(log_entry)

                if step_count % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-step-{step_count}")
                    model.save_pretrained(save_path)

    # Final Save
    final_path = os.path.join(args.output_dir, "final_checkpoint")
    model.save_pretrained(final_path)
    meta_log["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta_log["final_dev_loss"] = best_dev_loss
    meta_log["duration_seconds"] = time.time() - start_time
    with open(os.path.join(args.output_dir, "arm_training_meta.json"), "w") as f:
        json.dump(meta_log, f, indent=2)

    print(f"\nProduction Training Complete! Final model saved to {final_path}")
    print(f"Total Duration: {meta_log['duration_seconds']/60:.1f} minutes | Best Dev Loss: {best_dev_loss:.4f}")

if __name__ == "__main__":
    main()
