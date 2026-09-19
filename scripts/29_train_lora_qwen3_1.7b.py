#!/usr/bin/env python3
"""
scripts/29_train_lora_qwen3_1.7b.py
Curriculum LoRA fine-tuning for continuous latent recurrence on Qwen/Qwen3-1.7B:
Stage 1: Partial latent replacement (3 latent steps replacing initial think tokens).
Stage 2: Full latent replacement (6 latent steps replacing the entire <think> block).

Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
LoRA rank: 32, alpha: 64
Scale factor: alpha_Q3 = 0.034016
Device: cuda:0 (RTX 4080 16GB) in pure bfloat16.
"""

import os
import json
import time
import math
import random
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import transformers.modeling_utils
from peft import LoraConfig, get_peft_model, TaskType

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

MODEL_ID = "Qwen/Qwen3-1.7B"
DEVICE = "cuda:0"
SCALE_FACTOR = 0.034016

def parse_args():
    default_traces = os.path.join(os.path.dirname(__file__), "..", "data", "traces_gsm8k_qwen3_1.7b.jsonl")
    default_output = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "lora_recurrent_qwen3_1.7b")
    parser = argparse.ArgumentParser(description="Train LoRA curriculum for Qwen3-1.7B latent recurrence.")
    parser.add_argument("--traces_file", type=str, default=default_traces)
    parser.add_argument("--output_dir", type=str, default=default_output)
    parser.add_argument("--stage1_epochs", type=int, default=1)
    parser.add_argument("--stage2_epochs", type=int, default=2)
    parser.add_argument("--stage1_steps", type=int, default=3)
    parser.add_argument("--stage2_steps", type=int, default=6)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_target_len", type=int, default=512)
    parser.add_argument("--train_split", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_traces(jsonl_path):
    valid_records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get("valid_format", False) and len(data.get("think_text", "")) > 10:
                valid_records.append(data)
    return valid_records

def compute_sample_loss(model, tokenizer, sample, stage, num_latent_steps, scale_factor, max_target_len, device):
    prompt_ids = tokenizer(sample["formatted_prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    
    if stage == 1:
        think_tokens = tokenizer.encode(sample["think_text"], add_special_tokens=False)
        split_idx = max(1, len(think_tokens) // 2)
        rem_think_tokens = think_tokens[split_idx:]
        suffix_text = "\n</think>\n\n" + sample["answer_text"] + tokenizer.eos_token
        suffix_tokens = tokenizer.encode(suffix_text, add_special_tokens=False)
        target_token_list = rem_think_tokens + suffix_tokens
    else:
        target_text = "</think>\n\n" + sample["answer_text"] + tokenizer.eos_token
        target_token_list = tokenizer.encode(target_text, add_special_tokens=False)

    if len(target_token_list) > max_target_len:
        target_token_list = target_token_list[:max_target_len]
        
    target_ids = torch.tensor([target_token_list], dtype=torch.long, device=device)
    
    # 1. Prefill prompt
    out = model(input_ids=prompt_ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    curr_latent = out.hidden_states[-1][:, -1:, :]
    
    # 2. Continuous Latent Recurrent Loop
    for _ in range(num_latent_steps):
        scaled_latent = curr_latent * scale_factor
        step_out = model(
            inputs_embeds=scaled_latent,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True
        )
        past_kv = step_out.past_key_values
        curr_latent = step_out.hidden_states[-1][:, -1:, :]
        
    # 3. Predict first target token directly from final recurrent latent state
    logits_0 = model.lm_head(curr_latent)
    
    # 4. Teacher-force remaining target tokens
    if target_ids.shape[1] > 1:
        out_target = model(
            input_ids=target_ids[:, :-1],
            past_key_values=past_kv,
            use_cache=True
        )
        all_logits = torch.cat([logits_0, out_target.logits], dim=1)
    else:
        all_logits = logits_0
        
    vocab_size = all_logits.size(-1)
    loss = F.cross_entropy(all_logits.view(-1, vocab_size), target_ids.view(-1))
    return loss

def evaluate(model, tokenizer, val_samples, stage, num_latent_steps, scale_factor, max_target_len, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for sample in val_samples:
            try:
                loss = compute_sample_loss(
                    model, tokenizer, sample,
                    stage=stage,
                    num_latent_steps=num_latent_steps,
                    scale_factor=scale_factor,
                    max_target_len=max_target_len,
                    device=device
                )
                total_loss += loss.item()
                count += 1
            except Exception:
                continue
    return total_loss / max(1, count)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    print(f"=== Training Curriculum LoRA on {MODEL_ID} ({DEVICE}) ===")
    print(f"  Target Rank: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Stage 1: {args.stage1_epochs} epochs ({args.stage1_steps} latent steps)")
    print(f"  Stage 2: {args.stage2_epochs} epochs ({args.stage2_steps} latent steps)")
    print(f"  Scale Factor: {SCALE_FACTOR}")
    
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip("\"'\n")
                    break

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token, trust_remote_code=True)
    
    # Load traces
    traces = load_traces(args.traces_file)
    print(f"Loaded {len(traces)} valid traces.")
    random.shuffle(traces)
    split_idx = int(len(traces) * args.train_split)
    train_samples = traces[:split_idx]
    val_samples = traces[split_idx:]
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
    
    # Load base model
    print("\nLoading base model in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    
    # Target all linear layers
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    # Optimizer & Scheduler
    total_epochs = args.stage1_epochs + args.stage2_epochs
    steps_per_epoch = math.ceil(len(train_samples) / args.grad_accum_steps)
    total_training_steps = total_epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_training_steps)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    
    history = []
    global_step = 0
    t_start = time.time()
    
    # Curriculum Execution Loop
    curriculum = [
        (1, args.stage1_epochs, args.stage1_steps, "Stage 1: Partial Replacement (3 steps)"),
        (2, args.stage2_epochs, args.stage2_steps, "Stage 2: Full Replacement (6 steps)")
    ]
    
    for stage_num, num_epochs, num_steps, stage_desc in curriculum:
        print(f"\n{'='*60}\nStarting {stage_desc}\n{'='*60}")
        for epoch in range(num_epochs):
            model.train()
            random.shuffle(train_samples)
            epoch_loss = 0.0
            accum_loss = 0.0
            optimizer.zero_grad()
            
            for idx, sample in enumerate(train_samples):
                try:
                    loss = compute_sample_loss(
                        model, tokenizer, sample,
                        stage=stage_num,
                        num_latent_steps=num_steps,
                        scale_factor=SCALE_FACTOR,
                        max_target_len=args.max_target_len,
                        device=DEVICE
                    )
                    loss = loss / args.grad_accum_steps
                    loss.backward()
                    accum_loss += loss.item()
                except Exception as e:
                    print(f"Warning: sample {idx} failed with {e}")
                    continue
                    
                if (idx + 1) % args.grad_accum_steps == 0 or (idx + 1) == len(train_samples):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    epoch_loss += accum_loss * args.grad_accum_steps
                    accum_loss = 0.0
                    
            mean_train_loss = epoch_loss / len(train_samples)
            val_loss = evaluate(model, tokenizer, val_samples, stage_num, num_steps, SCALE_FACTOR, args.max_target_len, DEVICE)
            lr_curr = scheduler.get_last_lr()[0]
            
            print(f"Stage {stage_num} Epoch [{epoch+1}/{num_epochs}] | Train Loss: {mean_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr_curr:.2e}")
            history.append({
                "stage": stage_num,
                "epoch": epoch + 1,
                "train_loss": round(mean_train_loss, 4),
                "val_loss": round(val_loss, 4),
                "lr": lr_curr
            })
            
    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time:.2f}s ({total_time/60:.1f} mins).")
    
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter checkpoint to: {args.output_dir}")
    
    hist_file = os.path.join(args.output_dir, "training_history.json")
    with open(hist_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {hist_file}")

if __name__ == "__main__":
    main()
