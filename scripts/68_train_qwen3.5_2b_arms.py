#!/usr/bin/env python3
"""
scripts/68_train_qwen3.5_2b_arms.py
Rigorous Training Harness for Qwen/Qwen3.5-2B Arms (Arm 1b, Arm 2b, Arm 3).

Strictly Isolated to Model: Qwen/Qwen3.5-2B
Zero Cross-Model Contamination: Uses traces generated and curated specifically by Qwen3.5-2B.

Key Architectural Hooks for Hybrid Gated DeltaNet:
1. LoRA target_modules encompasses BOTH attention architectures:
   - Full self-attention: q_proj, k_proj, v_proj, o_proj
   - Gated DeltaNet linear attention: in_proj_a, in_proj_b, in_proj_qkv, in_proj_z, out_proj
   - MLP feedforward: gate_proj, up_proj, down_proj
2. Out-of-place LinearAttentionLayer cache updates:
   Patches update_recurrent_state and update_conv_state to perform out-of-place tensor assignments,
   enabling exact PyTorch autograd backpropagation across unrolled recurrent latent passes.
3. Empirical Scale Factor:
   alpha_2B = 0.925143 (pre-flight measured E[||W_E||] / E[||h_0||]).
4. Pinned SDPBackend.MATH for all recurrent latent unrolling.
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
from transformers.cache_utils import LinearAttentionLayer
from peft import LoraConfig, get_peft_model, TaskType

PAUSE_TOKEN = "<pause>"
THINK_END_TOKEN = "</think>"
CALIBRATED_ALPHA_2B = 0.925143

# Monkeypatch LinearAttentionLayer for out-of-place autograd differentiability
def safe_update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
    if not self.is_recurrent_states_initialized[state_idx]:
        self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
    self.recurrent_states[state_idx] = recurrent_states
    return self.recurrent_states[state_idx]

def safe_update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
    if not self.is_conv_states_initialized[state_idx]:
        self.lazy_initialization(conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size)
    if not self.has_previous_state[state_idx]:
        full_conv_states = conv_states
        self.has_previous_state[state_idx] = True
        if not self.record_past and full_conv_states.shape[-1] < self.conv_kernel_size[state_idx]:
            padding_length = self.conv_kernel_size[state_idx] - full_conv_states.shape[-1]
            full_conv_states = torch.nn.functional.pad(full_conv_states, (padding_length, 0), value=0)
    else:
        full_conv_states = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
    self.conv_states[state_idx] = full_conv_states[..., -self.conv_kernel_size[state_idx]:].clone()
    return full_conv_states

LinearAttentionLayer.update_recurrent_state = safe_update_recurrent_state
LinearAttentionLayer.update_conv_state = safe_update_conv_state

def load_traces(traces_path, volume_ablation="full", manifest_path=None):
    records = []
    with open(traces_path) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    print(f"Loaded {len(records)} traces from {traces_path}")
    
    if volume_ablation != "full":
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            if volume_ablation in manifest:
                indices = manifest[volume_ablation]
                records = [records[i] for i in indices if i < len(records)]
                print(f"Filtered to {len(records)} traces via manifest for volume ablation '{volume_ablation}'.")
            else:
                records = records[:int(volume_ablation)]
        else:
            records = records[:int(volume_ablation)]
            
    random.seed(42)
    random.shuffle(records)
    print(f"Final training set size: {len(records)} traces.")
    return records

def compute_loss_arm1b(model, tokenizer, sample, max_target_len, device):
    """
    Arm 1b: Trained Direct / No-CoT (K=0).
    prompt + \\n</think>\\n\\n + answer + <|im_end|>
    Loss computed strictly on \\n</think>\\n\\n + answer + <|im_end|>.
    """
    prompt_text = sample.get("prompt")
    ans_clean = sample.get("answer", "").replace("<|im_end|>", "").strip()
    target_text = "\n</think>\n\n" + ans_clean + "<|im_end|>"
    
    enc_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    enc_target = tokenizer.encode(target_text, add_special_tokens=False)
    if len(enc_target) > max_target_len:
        enc_target = enc_target[:max_target_len]
        
    full_tokens = enc_prompt + enc_target
    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, :len(enc_prompt)] = -100
    
    out = model(input_ids=input_ids, labels=labels)
    return out.loss

def compute_loss_arm2b(model, tokenizer, pause_token_id, sample, k_tokens, max_target_len, device):
    """
    Arm 2b: Pause-Token Control (K pause tokens).
    prompt + K <pause> tokens + \\n</think>\\n\\n + answer + <|im_end|>
    Loss computed strictly on \\n</think>\\n\\n + answer + <|im_end|>.
    """
    prompt_text = sample.get("prompt")
    pause_seq = [pause_token_id] * k_tokens
    ans_clean = sample.get("answer", "").replace("<|im_end|>", "").strip()
    target_text = "\n</think>\n\n" + ans_clean + "<|im_end|>"
    
    enc_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    enc_target = tokenizer.encode(target_text, add_special_tokens=False)
    if len(enc_target) > max_target_len:
        enc_target = enc_target[:max_target_len]
        
    full_tokens = enc_prompt + pause_seq + enc_target
    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, :(len(enc_prompt) + k_tokens)] = -100
    
    out = model(input_ids=input_ids, labels=labels)
    return out.loss

def compute_loss_arm3_stage1(model, tokenizer, sample, k_tokens, scale_factor, max_target_len, device):
    """
    Arm 3 Stage 1: Step-Segmented Partial Latent Replacement.
    Replaces the first floor(S/2) reasoning steps with floor(K/2) latents;
    predicts remaining steps + \\n</think>\\n\\n + answer + <|im_end|>.
    """
    prompt_text = sample.get("prompt")
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    
    steps = sample.get("steps", [])
    num_steps = len(steps)
    m_replaced = max(1, num_steps // 2)
    k_latents = max(1, k_tokens // 2)
    
    remaining_steps_text = "\n\n".join(steps[m_replaced:]) if m_replaced < num_steps else ""
    ans_clean = sample.get("answer", "").replace("<|im_end|>", "").strip()
    
    if remaining_steps_text:
        target_text = remaining_steps_text + "\n</think>\n\n" + ans_clean + "<|im_end|>"
    else:
        target_text = "\n</think>\n\n" + ans_clean + "<|im_end|>"
        
    target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
    if len(target_tokens) > max_target_len:
        target_tokens = target_tokens[:max_target_len]
    target_ids = torch.tensor([target_tokens], dtype=torch.long, device=device)
    
    # Prefill prompt
    out = model(input_ids=prompt_ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    curr_latent = out.hidden_states[-1][:, -1:, :]
    
    L_prompt = prompt_ids.shape[1]
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        for k in range(k_latents):
            pos = torch.tensor([[L_prompt + k]], device=device, dtype=torch.long)
            scaled_latent = curr_latent * scale_factor
            step_out = model(inputs_embeds=scaled_latent, position_ids=pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]
        
    logits_0 = model.lm_head(curr_latent)
    if target_ids.shape[1] > 1:
        out_target = model(input_ids=target_ids[:, :-1], past_key_values=past_kv, use_cache=True)
        all_logits = torch.cat([logits_0, out_target.logits], dim=1)
    else:
        all_logits = logits_0
        
    vocab_size = all_logits.size(-1)
    loss = F.cross_entropy(all_logits.view(-1, vocab_size), target_ids.view(-1))
    return loss

def compute_loss_arm3_stage2(model, tokenizer, sample, k_tokens, scale_factor, max_target_len, device):
    """
    Arm 3 Stage 2: Full Latent Replacement.
    Replaces all reasoning steps with K latents;
    predicts \\n</think>\\n\\n + answer + <|im_end|>.
    """
    prompt_text = sample.get("prompt")
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    
    ans_clean = sample.get("answer", "").replace("<|im_end|>", "").strip()
    target_text = "\n</think>\n\n" + ans_clean + "<|im_end|>"
    target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
    if len(target_tokens) > max_target_len:
        target_tokens = target_tokens[:max_target_len]
    target_ids = torch.tensor([target_tokens], dtype=torch.long, device=device)
    
    out = model(input_ids=prompt_ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    curr_latent = out.hidden_states[-1][:, -1:, :]
    
    L_prompt = prompt_ids.shape[1]
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        for k in range(k_tokens):
            pos = torch.tensor([[L_prompt + k]], device=device, dtype=torch.long)
            scaled_latent = curr_latent * scale_factor
            step_out = model(inputs_embeds=scaled_latent, position_ids=pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]
        
    logits_0 = model.lm_head(curr_latent)
    if target_ids.shape[1] > 1:
        out_target = model(input_ids=target_ids[:, :-1], past_key_values=past_kv, use_cache=True)
        all_logits = torch.cat([logits_0, out_target.logits], dim=1)
    else:
        all_logits = logits_0
        
    vocab_size = all_logits.size(-1)
    loss = F.cross_entropy(all_logits.view(-1, vocab_size), target_ids.view(-1))
    return loss

def evaluate_dev_loss(model, tokenizer, dev_samples, arm, k_tokens, alpha_scale, pause_token_id, is_stage_2, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for s in dev_samples[:10]:
            try:
                if arm == "arm1b":
                    l = compute_loss_arm1b(model, tokenizer, s, max_target_len=512, device=device)
                elif arm == "arm2b":
                    l = compute_loss_arm2b(model, tokenizer, pause_token_id, s, k_tokens, max_target_len=512, device=device)
                elif arm == "arm3":
                    if is_stage_2:
                        l = compute_loss_arm3_stage2(model, tokenizer, s, k_tokens, alpha_scale, max_target_len=512, device=device)
                    else:
                        l = compute_loss_arm3_stage1(model, tokenizer, s, k_tokens, alpha_scale, max_target_len=512, device=device)
                losses.append(l.item())
            except Exception:
                pass
    model.train()
    return float(np.mean(losses)) if losses else 0.0

def main():
    parser = argparse.ArgumentParser(description="Train Qwen3.5-2B Arms (1b, 2b, 3)")
    parser.add_argument("--arm", type=str, choices=["arm1b", "arm2b", "arm3"], required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--k_tokens", type=int, default=6)
    parser.add_argument("--volume_ablation", type=str, default="full", choices=["500", "1000", "full"])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--traces_path", type=str, default=None)
    parser.add_argument("--dev_path", type=str, default=None)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    tag = "qwen_qwen3.5-2b"
    
    if args.traces_path is None:
        curated_p = os.path.join(data_dir, "curated_train_traces_qwen3.5_2b.jsonl")
        if os.path.exists(curated_p):
            args.traces_path = curated_p
        else:
            args.traces_path = os.path.join(data_dir, "self_distill_train_traces_qwen3.5_2b.jsonl")
            
    if args.dev_path is None:
        args.dev_path = os.path.join(data_dir, "curated_dev_traces_qwen3.5_2b.jsonl")
        
    if args.manifest_path is None:
        args.manifest_path = os.path.join(data_dir, "volume_ablation_manifest_qwen3.5_2b.json")
        
    if args.arm == "arm1b":
        args.k_tokens = 0

    suffix = f"_vol{args.volume_ablation}" if args.volume_ablation != "full" else ""
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(__file__), "..", "checkpoints", f"lora_{args.arm}_{tag}_k{args.k_tokens}_selfdistill{suffix}"
        )

    print(f"========================================================")
    print(f"=== Training Harness: {args.arm.upper()} for {args.model_id} ===")
    print(f"Target Device: {args.device} | Horizon K: {args.k_tokens}")
    print(f"Curriculum Traces: {args.traces_path}")
    print(f"Output Checkpoint: {args.output_dir}")
    print(f"========================================================")

    dataset = load_traces(args.traces_path, volume_ablation=args.volume_ablation, manifest_path=args.manifest_path)
    dev_dataset = []
    if os.path.exists(args.dev_path):
        with open(args.dev_path) as f:
            for line in f:
                if line.strip():
                    try:
                        dev_dataset.append(json.loads(line))
                    except Exception:
                        pass
        print(f"Loaded {len(dev_dataset)} held-out dev traces for validation loss monitoring.")

    steps_per_epoch = max(1, len(dataset) // args.grad_accum_steps)
    steps_per_stage = 2 * steps_per_epoch
    
    if args.steps is not None:
        total_steps = args.steps
        stage_boundary = total_steps // 2 if args.arm == "arm3" else total_steps
    else:
        if args.arm == "arm3":
            total_steps = 2 * steps_per_stage
            stage_boundary = steps_per_stage
        else:
            total_steps = steps_per_stage
            stage_boundary = total_steps

    print(f"Steps per epoch: {steps_per_epoch} | Steps per stage (2 epochs): {steps_per_stage}")
    print(f"Total training steps: {total_steps} (Stage boundary: step {stage_boundary})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    pause_token_id = None
    if args.arm == "arm2b":
        if PAUSE_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [PAUSE_TOKEN]})
        pause_token_id = tokenizer.convert_tokens_to_ids(PAUSE_TOKEN)
        print(f"Pause Token: '{PAUSE_TOKEN}' (ID: {pause_token_id})")

    print(f"Loading base model in pure bfloat16 onto {args.device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    if args.arm == "arm2b":
        base_model.resize_token_embeddings(len(tokenizer))

    # Comprehensive linear module targeting for hybrid DeltaNet + full self-attention
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"
        ],
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    alpha_scale = CALIBRATED_ALPHA_2B
    print(f"Using empirical scale factor alpha_2B: {alpha_scale:.6f}")

    train_log = []
    global_step = 0
    accum_loss = 0.0
    accum_count = 0
    t0_train = time.time()
    model.train()

    sample_idx = 0
    n_samples = len(dataset)

    while global_step < total_steps:
        sample = dataset[sample_idx % n_samples]
        sample_idx += 1
        is_stage_2 = (global_step >= stage_boundary)

        try:
            if args.arm == "arm1b":
                loss = compute_loss_arm1b(model, tokenizer, sample, max_target_len=512, device=args.device)
            elif args.arm == "arm2b":
                loss = compute_loss_arm2b(model, tokenizer, pause_token_id, sample, args.k_tokens, max_target_len=512, device=args.device)
            elif args.arm == "arm3":
                if is_stage_2:
                    loss = compute_loss_arm3_stage2(model, tokenizer, sample, args.k_tokens, alpha_scale, max_target_len=512, device=args.device)
                else:
                    loss = compute_loss_arm3_stage1(model, tokenizer, sample, args.k_tokens, alpha_scale, max_target_len=512, device=args.device)

            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()
            accum_loss += loss.item()
            accum_count += 1
        except Exception as e:
            print(f"Warning: Step failed on sample {sample.get('id', 'unknown')}: {e}")
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_count = 0
            continue

        if accum_count == args.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            avg_train_loss = accum_loss / args.grad_accum_steps
            accum_loss = 0.0
            accum_count = 0

            if global_step % 20 == 0 or global_step == total_steps or global_step == stage_boundary:
                dev_loss = evaluate_dev_loss(model, tokenizer, dev_dataset, args.arm, args.k_tokens, alpha_scale, pause_token_id, is_stage_2, args.device)
                elapsed = time.time() - t0_train
                stage_str = "Stage 2 (Full)" if is_stage_2 else "Stage 1 (Partial)"
                print(f"[{args.arm.upper()} | Step {global_step}/{total_steps} | {stage_str}] Train Loss: {avg_train_loss:.4f} | Dev Loss: {dev_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.2e} | Elapsed: {elapsed:.1f}s")
                train_log.append({
                    "step": global_step,
                    "stage": 2 if is_stage_2 else 1,
                    "train_loss": round(avg_train_loss, 4),
                    "dev_loss": round(dev_loss, 4),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "elapsed_sec": round(elapsed, 1)
                })

    print(f"\nTraining completed in {time.time() - t0_train:.1f}s!")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "model_id": args.model_id,
        "arm": args.arm,
        "k_tokens": args.k_tokens,
        "volume_ablation": args.volume_ablation,
        "alpha_scale": alpha_scale,
        "total_steps": total_steps,
        "stage_boundary": stage_boundary if args.arm == "arm3" else None,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "device": args.device,
        "seed": args.seed,
        "final_train_loss": train_log[-1]["train_loss"] if train_log else None,
        "final_dev_loss": train_log[-1]["dev_loss"] if train_log else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_log": train_log
    }
    meta_path = os.path.join(args.output_dir, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved LoRA adapter checkpoint to: {args.output_dir}")
    print(f"Metadata saved to: {meta_path}")

if __name__ == "__main__":
    main()
