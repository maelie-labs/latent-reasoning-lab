#!/usr/bin/env python3
"""
scripts/86_train_qwen3_arms_v1_1.py
Continuous Latent Recurrence v1.1 Training Harness.

Implements the unified training recipe across:
- Arm 1b (Direct SFT Control, K=0): Non-thinking worked solutions + general instruction slice.
- Arm 2b (Pause-Token Control, K=6, 32): K pause tokens + non-thinking worked solutions.
- Arm 3 (Continuous Latent Recurrence v1.1, K=6, 32):
  * Loss: L_total = L_CE(answer) + lambda_align * L_distill(student_states, teacher_states).
  * L_distill: Step-level state distillation against post-final-norm teacher CoT states.
  * Removal Smoothing: Deng et al. continuous schedule (lambda_smooth=4).
  * Lambda_align tuning: Dev loss sweep over {0.1, 0.5, 1.0}.
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

PAUSE_TOKEN = "<pause>"
CALIBRATED_ALPHAS = {
    "qwen/qwen3-1.7b": 0.011440,
    "qwen/qwen3.5-2b": 0.009800,
    "qwen/qwen3-4b": 0.007670,
}

def get_calibrated_alpha(model_id):
    return CALIBRATED_ALPHAS.get(model_id.lower(), 0.011440)

def load_v1_1_dataset(file_path):
    records = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def compute_loss_arm1b(model, tokenizer, sample, max_target_len, device):
    """Arm 1b: Direct SFT on non-thinking worked solutions."""
    prompt = sample.get("prompt", "")
    # Conditioning for non-thinking mode
    if "<think>\n\n</think>\n\n" not in prompt:
        prompt = prompt.replace("<think>\n", "<think>\n\n</think>\n\n")
        
    ans_text = sample.get("nonthinking_answer", "")
    enc_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    enc_ans = tokenizer.encode(ans_text, add_special_tokens=False)
    if len(enc_ans) > max_target_len:
        enc_ans = enc_ans[:max_target_len]
        
    input_ids = torch.tensor([enc_prompt + enc_ans], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, :len(enc_prompt)] = -100
    
    out = model(input_ids=input_ids, labels=labels)
    return out.loss

def compute_loss_arm2b(model, tokenizer, sample, k_tokens, pause_token_id, max_target_len, device):
    """
    Arm 2b: Pause token control on non-thinking worked solutions.
    prompt (ending with <think>\n) + K <pause> tokens + \n</think>\n\n + answer.
    Loss computed strictly on \n</think>\n\n + answer.
    """
    prompt = sample.get("prompt", "")
    if not prompt.endswith("<think>\n"):
        prompt = prompt.replace("<think>\n\n</think>\n\n", "<think>\n")
        if not prompt.endswith("<think>\n"):
            prompt += "<think>\n"
        
    pause_seq = [pause_token_id] * k_tokens
    ans_text = "\n</think>\n\n" + sample.get("nonthinking_answer", "")
    if not ans_text.endswith("<|im_end|>"):
        ans_text += "<|im_end|>"
    
    enc_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    enc_ans = tokenizer.encode(ans_text, add_special_tokens=False)
    if len(enc_ans) > max_target_len:
        enc_ans = enc_ans[:max_target_len]
        
    full_tokens = enc_prompt + pause_seq + enc_ans
    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, :(len(enc_prompt) + k_tokens)] = -100
    
    out = model(input_ids=input_ids, labels=labels)
    return out.loss

def compute_loss_arm3_v1_1(model, tokenizer, sample, teacher_cache, k_tokens, scale_factor, lambda_align, max_target_len, device, use_codi_fallback=False, mask_prompt_in_answer=False):
    """
    Arm 3 v1.1: Grounded Latent Recurrence.
    L_total = L_CE(nonthinking_answer) + lambda_align * L_distill(student_states, teacher_states)
    """
    prompt = sample.get("prompt", "")
    # In Option (a), latents execute under <think>\n, then transition via \n</think>\n\n to non-thinking answer
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
    
    # Prefill prompt
    out = model(input_ids=prompt_ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    curr_latent = out.hidden_states[-1][:, -1:, :]  # post-final-norm
    
    L_prompt = prompt_ids.shape[1]
    backbone = model.base_model.model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "model") else model
    
    student_states = []
    for k in range(k_tokens):
        pos = torch.tensor([[L_prompt + k]], device=device, dtype=torch.long)
        scaled_latent = curr_latent * scale_factor
        step_out = backbone(inputs_embeds=scaled_latent, position_ids=pos, past_key_values=past_kv, use_cache=True)
        past_kv = step_out.past_key_values
        curr_latent = step_out.last_hidden_state[:, -1:, :] if hasattr(step_out, "last_hidden_state") else step_out.hidden_states[-1][:, -1:, :]
        student_states.append(curr_latent)
        
    logits_0 = model.lm_head(curr_latent)
    if target_ids.shape[1] > 1:
        if mask_prompt_in_answer:
            total_seq_len = L_prompt + k_tokens + (target_ids.shape[1] - 1)
            ans_mask = torch.ones((1, total_seq_len), dtype=torch.long, device=device)
            ans_mask[:, :L_prompt] = 0
            out_target = model(input_ids=target_ids[:, :-1], attention_mask=ans_mask, past_key_values=past_kv, use_cache=True)
        else:
            out_target = model(input_ids=target_ids[:, :-1], past_key_values=past_kv, use_cache=True)
        all_logits = torch.cat([logits_0, out_target.logits], dim=1)
    else:
        all_logits = logits_0
        
    vocab_size = all_logits.size(-1)
    ce_loss = F.cross_entropy(all_logits.view(-1, vocab_size), target_ids.view(-1))
    
    # Step-Level State Distillation Loss
    distill_loss = torch.tensor(0.0, device=device, dtype=torch.bfloat16)
    prob_id = sample.get("id")
    if teacher_cache and prob_id in teacher_cache and lambda_align > 0.0:
        targets_dict = teacher_cache[prob_id]
        if use_codi_fallback:
            # True CODI fallback: align student final latent to answer position teacher state
            teacher_target = targets_dict["codi_answer_state"].to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
            final_student = student_states[-1]
            distill_loss = 1.0 - F.cosine_similarity(final_student, teacher_target, dim=-1).mean()
        else:
            # Primary: Step-Level State Distillation across all K positions
            k_key = f"K{k_tokens}"
            if k_key in targets_dict:
                teacher_targets = targets_dict[k_key].to(device)  # (K, d_model)
                cos_sims = []
                for k in range(k_tokens):
                    s_state = student_states[k].squeeze()  # (d_model,)
                    t_state = teacher_targets[k]           # (d_model,)
                    cos_sims.append(1.0 - F.cosine_similarity(s_state, t_state, dim=-1))
                distill_loss = torch.stack(cos_sims).mean()
                
    total_loss = ce_loss + lambda_align * distill_loss
    return total_loss, ce_loss.item(), distill_loss.item()

def evaluate_dev_loss(model, tokenizer, dev_dataset, teacher_cache, arm, k_tokens, scale_factor, lambda_align, max_target_len, device, use_codi_fallback=False):
    model.eval()
    losses = []
    ce_losses = []
    distill_losses = []
    with torch.no_grad():
        for sample in dev_dataset:
            if arm == "arm1b":
                loss = compute_loss_arm1b(model, tokenizer, sample, max_target_len, device)
                losses.append(loss.item())
                ce_losses.append(loss.item())
            elif arm == "arm2b":
                pause_id = tokenizer.convert_tokens_to_ids(PAUSE_TOKEN)
                loss = compute_loss_arm2b(model, tokenizer, sample, k_tokens, pause_id, max_target_len, device)
                losses.append(loss.item())
                ce_losses.append(loss.item())
            elif arm == "arm3":
                total_l, ce_l, dist_l = compute_loss_arm3_v1_1(
                    model, tokenizer, sample, teacher_cache, k_tokens, scale_factor,
                    lambda_align, max_target_len, device, use_codi_fallback=use_codi_fallback
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
    parser = argparse.ArgumentParser(description="Continuous Latent Recurrence v1.1 Training Harness")
    parser.add_argument("--arm", type=str, required=True, choices=["arm1b", "arm2b", "arm3"])
    parser.add_argument("--k_tokens", type=int, default=6, choices=[0, 6, 32])
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--use_codi_fallback", action="store_true", help="Use answer-position CODI fallback")
    parser.add_argument("--use_bottleneck_mask", action="store_true", help="Mask question tokens from answer phase attention during training, relaxed over last 100 steps")
    parser.add_argument("--bottleneck_relax_steps", type=int, default=100, help="Number of final steps over which to linearly relax the bottleneck mask")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--dev_file", type=str, default=None)
    parser.add_argument("--teacher_cache_file", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing for VRAM efficiency (~6.6GB)")
    parser.add_argument("--max_target_len", type=int, default=2048, help="Maximum target token length")
    args = parser.parse_args()

    scale_factor = get_calibrated_alpha(args.model_id)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    model_tag = args.model_id.replace("/", "_").lower()
    
    # Resolve train and dev paths (prefer final stratified if available)
    if args.train_file:
        train_path = os.path.join(data_dir, os.path.basename(args.train_file))
    elif os.path.exists(os.path.join(data_dir, "curated_train_v1_1_final_qwen3_1.7b.jsonl")):
        train_path = os.path.join(data_dir, "curated_train_v1_1_final_qwen3_1.7b.jsonl")
    else:
        train_path = os.path.join(data_dir, "curated_train_v1_1_qwen3_1.7b.jsonl")

    if args.dev_file:
        dev_path = os.path.join(data_dir, os.path.basename(args.dev_file))
    elif os.path.exists(os.path.join(data_dir, "curated_dev_v1_1_final_qwen3_1.7b.jsonl")):
        dev_path = os.path.join(data_dir, "curated_dev_v1_1_final_qwen3_1.7b.jsonl")
    else:
        dev_path = os.path.join(data_dir, "curated_dev_v1_1_qwen3_1.7b.jsonl")

    if args.teacher_cache_file:
        teacher_cache_path = os.path.join(data_dir, os.path.basename(args.teacher_cache_file))
    else:
        teacher_cache_path = os.path.join(data_dir, f"teacher_cot_states_{model_tag}.pt")

    if args.output_dir is None:
        if args.arm == "arm3":
            if args.use_codi_fallback:
                tag = f"lora_{args.arm}_v1_1_{model_tag}_k{args.k_tokens}_codi"
            else:
                tag = f"lora_{args.arm}_v1_1_{model_tag}_k{args.k_tokens}_lambda{args.lambda_align}"
        else:
            tag = f"lora_{args.arm}_v1_1_{model_tag}_k{args.k_tokens}"
        args.output_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", tag)

    print(f"=== Training Harness v1.1: {args.arm.upper()} (K={args.k_tokens}) on {args.model_id} ===")
    print(f"Output Directory: {args.output_dir}")
    print(f"Calibrated Scale Factor: alpha={scale_factor:.6f}")
    if args.arm == "arm3":
        print(f"Distillation Weight: lambda_align={args.lambda_align}")
    if args.gradient_checkpointing:
        print("Gradient Checkpointing: ENABLED (peak VRAM ~6.6GB)")
    
    # Load dataset
    train_dataset = load_v1_1_dataset(train_path)
    dev_dataset = load_v1_1_dataset(dev_path)
    print(f"Loaded {len(train_dataset)} train traces and {len(dev_dataset)} dev traces.")

    teacher_cache = None
    if os.path.exists(teacher_cache_path) and args.arm == "arm3":
        print(f"Loading teacher CoT state cache from: {teacher_cache_path}")
        teacher_cache = torch.load(teacher_cache_path, map_location="cpu")
        print(f"Loaded {len(teacher_cache)} teacher state records.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if args.arm == "arm2b":
        if PAUSE_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    if args.arm == "arm2b":
        base_model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base_model.enable_input_require_grads()

    # LoRA config (rank 32, alpha 64)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_dataset) * args.epochs) // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    print(f"\nBeginning Training ({args.epochs} epochs, {total_steps} gradient steps)...")
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
                if args.arm == "arm1b":
                    loss = compute_loss_arm1b(model, tokenizer, sample, args.max_target_len, args.device)
                    ce_val = loss.item()
                    dist_val = 0.0
                elif args.arm == "arm2b":
                    pause_id = tokenizer.convert_tokens_to_ids(PAUSE_TOKEN)
                    loss = compute_loss_arm2b(model, tokenizer, sample, args.k_tokens, pause_id, args.max_target_len, args.device)
                    ce_val = loss.item()
                    dist_val = 0.0
                elif args.arm == "arm3":
                    mask_prompt = False
                    if args.use_bottleneck_mask:
                        if step_count < (total_steps - args.bottleneck_relax_steps):
                            mask_prompt = True
                        else:
                            p_mask = max(0.0, (total_steps - step_count) / float(args.bottleneck_relax_steps))
                            mask_prompt = (random.random() < p_mask)

                    loss, ce_val, dist_val = compute_loss_arm3_v1_1(
                        model, tokenizer, sample, teacher_cache, args.k_tokens, scale_factor,
                        args.lambda_align, args.max_target_len, args.device,
                        use_codi_fallback=args.use_codi_fallback,
                        mask_prompt_in_answer=mask_prompt
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
                    print(f"Step [{step_count}/{total_steps}] Epoch {epoch+1} | Total Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Distill: {avg_dist:.4f}) | Elapsed: {time.time()-start_time:.1f}s", flush=True)
                    
    # Final dev loss evaluation
    print("\n--- Evaluating Final Dev Loss on Held-Out Split ---")
    final_dev_loss = evaluate_dev_loss(model, tokenizer, dev_dataset, teacher_cache, args.arm, args.k_tokens, scale_factor, args.lambda_align, args.max_target_len, args.device, use_codi_fallback=args.use_codi_fallback)
    print(f"Final Dev Loss: {final_dev_loss['total_loss']:.4f} (CE: {final_dev_loss['ce_loss']:.4f}, Distill: {final_dev_loss['distill_loss']:.4f}, Alignment Cosine Sim: {final_dev_loss['alignment_cosine_sim']:.4f})")

    # Save checkpoint & metadata
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    meta = {
        "arm": args.arm,
        "k_tokens": args.k_tokens,
        "model_id": args.model_id,
        "epochs": args.epochs,
        "lr": args.lr,
        "lambda_align": args.lambda_align,
        "scale_factor": scale_factor,
        "total_steps": total_steps,
        "final_dev_loss": final_dev_loss["total_loss"],
        "final_ce_loss": final_dev_loss["ce_loss"],
        "final_distill_loss": final_dev_loss["distill_loss"],
        "final_alignment_cosine_sim": final_dev_loss["alignment_cosine_sim"],
        "duration_seconds": time.time() - start_time,
        "use_codi_fallback": args.use_codi_fallback,
        "use_bottleneck_mask": args.use_bottleneck_mask,
        "bottleneck_relax_steps": args.bottleneck_relax_steps
    }
    with open(os.path.join(args.output_dir, "arm_training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Checkpoint saved to: {args.output_dir}")
    print("=== Training Complete [SUCCESS] ===")

if __name__ == "__main__":
    main()
