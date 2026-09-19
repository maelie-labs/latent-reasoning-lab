#!/usr/bin/env python3
"""
18_train_lora_14b.py
Curriculum LoRA fine-tuning for continuous latent recurrence on DeepSeek-R1-Distill-Qwen-14B in 8-bit.
Stage 1: Partial latent replacement (3 latent steps replacing initial think tokens).
Stage 2: Full latent replacement (6 latent steps replacing the entire <think> block).

Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B in 8-bit on cuda:1 (RTX PRO 4500 32GB)
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
LoRA rank: 32, alpha: 128
Optimizer: AdamW + Cosine schedule, grad clipping 1.0, float16.
Output: checkpoints/lora_recurrent_14b
"""

import os
import json
import time
import math
import random
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEVICE = "cuda:1"

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA curriculum for 14B continuous latent recurrence.")
    parser.add_argument("--traces_file", type=str, default="data/traces_gsm8k_14b.jsonl", help="Self-distilled traces path")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_recurrent_14b", help="Output checkpoint directory")
    parser.add_argument("--stage1_epochs", type=int, default=1, help="Epochs for Stage 1 (Partial replacement)")
    parser.add_argument("--stage2_epochs", type=int, default=2, help="Epochs for Stage 2 (Full replacement)")
    parser.add_argument("--stage1_steps", type=int, default=3, help="Recurrent steps for Stage 1")
    parser.add_argument("--stage2_steps", type=int, default=6, help="Recurrent steps for Stage 2")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--max_target_len", type=int, default=512, help="Maximum target token sequence length")
    parser.add_argument("--train_split", type=float, default=0.85, help="Ratio of examples for training")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of trace samples to load")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_traces(jsonl_path, max_samples=None):
    valid_records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get("valid_format", False) and len(data.get("think_text", "")) > 10:
                valid_records.append(data)
            if max_samples and len(valid_records) >= max_samples:
                break
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
        scaled_latent = (curr_latent * scale_factor).to(torch.float16)
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
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    count += 1
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"Warning in eval: {e}")
                torch.cuda.empty_cache()
                continue
    model.train()
    return total_loss / max(1, count)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    print(f"=== Stage 4: Continuous Latent Recurrence LoRA Training (14B 8-bit) ===")
    print(f"Device: {DEVICE} | LoRA r={args.lora_r}, alpha={args.lora_alpha} | LR={args.lr}")
    print(f"Curriculum: Stage 1 ({args.stage1_epochs} eps, {args.stage1_steps} steps) -> Stage 2 ({args.stage2_epochs} eps, {args.stage2_steps} steps)")
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    
    print(f"Loading base model {MODEL_ID} in 8-bit on {DEVICE}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=DEVICE,
        torch_dtype=torch.float16
    )
    base_model = prepare_model_for_kbit_training(base_model)
    
    embed_weights = base_model.get_input_embeddings().weight.detach()
    avg_embed_norm = torch.norm(embed_weights.float(), dim=-1).mean().item()
    hidden_dim = base_model.config.hidden_size
    theoretical_rmsnorm = hidden_dim ** 0.5
    scale_factor = avg_embed_norm / theoretical_rmsnorm
    print(f"Calculated scale factor: {scale_factor:.6f} (norm={avg_embed_norm:.4f}, sqrt_dim={theoretical_rmsnorm:.4f})")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model.train()
    
    all_traces = load_traces(args.traces_file, max_samples=args.max_samples)
    random.shuffle(all_traces)
    n_train = int(len(all_traces) * args.train_split)
    train_samples = all_traces[:n_train]
    val_samples = all_traces[n_train:]
    print(f"Dataset split: {len(train_samples)} training samples, {len(val_samples)} validation samples")
    
    total_training_epochs = args.stage1_epochs + args.stage2_epochs
    total_steps = (len(train_samples) * total_training_epochs) // args.grad_accum_steps
    warmup_steps = max(4, int(0.08 * total_steps))
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"Total optimizer steps: {total_steps} (warmup: {warmup_steps})")
    
    history = {
        "scale_factor": scale_factor,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "stage1": [],
        "stage2": []
    }
    
    start_total_time = time.time()
    global_step = 0
    overall_best_stage2_val_loss = float("inf")
    
    def train_stage(stage_num, num_epochs, recurrent_steps):
        nonlocal overall_best_stage2_val_loss, global_step
        print(f"\n{'='*60}")
        print(f"STARTING CURRICULUM STAGE {stage_num}: {recurrent_steps} Recurrent Steps | {num_epochs} Epochs")
        print(f"{'='*60}")
        
        stage_best_val_loss = float("inf")
        
        initial_val_loss = evaluate(
            model, tokenizer, val_samples,
            stage=stage_num,
            num_latent_steps=recurrent_steps,
            scale_factor=scale_factor,
            max_target_len=args.max_target_len,
            device=DEVICE
        )
        print(f"Stage {stage_num} Initial Validation Loss: {initial_val_loss:.4f}")
        
        for epoch in range(num_epochs):
            model.train()
            random.shuffle(train_samples)
            
            epoch_loss = 0.0
            accum_loss = 0.0
            accum_steps = 0
            valid_samples_count = 0
            epoch_start = time.time()
            
            optimizer.zero_grad()
            
            for idx, sample in enumerate(train_samples):
                try:
                    loss = compute_sample_loss(
                        model, tokenizer, sample,
                        stage=stage_num,
                        num_latent_steps=recurrent_steps,
                        scale_factor=scale_factor,
                        max_target_len=args.max_target_len,
                        device=DEVICE
                    )
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  [Warning] NaN/Inf loss at sample {idx}. Skipping.")
                        continue
                        
                    scaled_loss = loss / args.grad_accum_steps
                    scaled_loss.backward()
                    
                    accum_loss += loss.item()
                    epoch_loss += loss.item()
                    accum_steps += 1
                    valid_samples_count += 1
                    
                    if accum_steps == args.grad_accum_steps or (idx + 1) == len(train_samples):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, model.parameters()),
                            args.max_grad_norm
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        accum_steps = 0
                        global_step += 1
                        
                        if global_step % 5 == 0:
                            cur_lr = scheduler.get_last_lr()[0]
                            print(f"  Step {global_step:03d} | Sample {idx+1:03d}/{len(train_samples)} | Loss: {accum_loss/max(1, args.grad_accum_steps):.4f} | Grad Norm: {grad_norm:.3f} | LR: {cur_lr:.2e}")
                            accum_loss = 0.0
                            
                except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                    print(f"  [Warning] OOM/RuntimeError on sample {idx}: {err}. Clearing cache.")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    accum_steps = 0
                    continue
                    
            avg_epoch_loss = epoch_loss / max(1, valid_samples_count)
            val_loss = evaluate(
                model, tokenizer, val_samples,
                stage=stage_num,
                num_latent_steps=recurrent_steps,
                scale_factor=scale_factor,
                max_target_len=args.max_target_len,
                device=DEVICE
            )
            epoch_time = time.time() - epoch_start
            
            print(f"\n--- Stage {stage_num} Epoch {epoch+1}/{num_epochs} Finished ---")
            print(f"Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
            
            stage_key = f"stage{stage_num}"
            history[stage_key].append({
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "val_loss": val_loss,
                "time_sec": epoch_time
            })
            
            if val_loss < stage_best_val_loss:
                stage_best_val_loss = val_loss
                stage_ckpt_dir = os.path.join(args.output_dir, f"stage{stage_num}_best")
                os.makedirs(stage_ckpt_dir, exist_ok=True)
                model.save_pretrained(stage_ckpt_dir)
                tokenizer.save_pretrained(stage_ckpt_dir)
                print(f"  [Stage {stage_num} Best] New best val loss: {stage_best_val_loss:.4f} saved to {stage_ckpt_dir}")
                
            if stage_num == 2 and val_loss < overall_best_stage2_val_loss:
                overall_best_stage2_val_loss = val_loss
                best_ckpt_dir = os.path.join(args.output_dir, "best")
                os.makedirs(best_ckpt_dir, exist_ok=True)
                model.save_pretrained(best_ckpt_dir)
                tokenizer.save_pretrained(best_ckpt_dir)
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                print(f"  [Global Best Checkpoint] New best Stage 2 val loss: {overall_best_stage2_val_loss:.4f} saved to {best_ckpt_dir} and {args.output_dir}")
                
    if args.stage1_epochs > 0:
        train_stage(stage_num=1, num_epochs=args.stage1_epochs, recurrent_steps=args.stage1_steps)
        
    if args.stage2_epochs > 0:
        train_stage(stage_num=2, num_epochs=args.stage2_epochs, recurrent_steps=args.stage2_steps)
        
    total_time = time.time() - start_total_time
    history["total_time_sec"] = total_time
    history["best_val_loss"] = overall_best_stage2_val_loss
    history_file = os.path.join(args.output_dir, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    print(f"\n=== 14B LoRA Training Finished Successfully ===")
    print(f"Total Time: {total_time/60:.2f} minutes | Best Stage 2 Val Loss: {overall_best_stage2_val_loss:.4f}")
    print(f"Saved artifacts to {args.output_dir} and {history_file}")

if __name__ == "__main__":
    main()
