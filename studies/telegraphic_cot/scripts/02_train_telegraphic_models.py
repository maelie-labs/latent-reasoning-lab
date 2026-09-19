#!/usr/bin/env python3
"""
studies/telegraphic_cot/scripts/02_train_telegraphic_models.py
Phase 2: Parity Training for Telegraphic CoT, Dual-Channel Latents, and Pause Controls.

Modes:
- arm1_telegraphic_cot: Pure discrete dense propositional CoT (loss on thought + answer)
- control3_matched_pause: Length-matched pause token delay (loss on answer only)
- arm2_dual_channel_latents: Dense propositions + K=12 continuous unrolled latents
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ALPHA_1_7B = 0.011440

def format_discrete_sequence(sample, mode):
    prompt = sample.get("prompt", "")
    if not prompt.endswith("<think>\n"):
        prompt = prompt.replace("<think>\n\n</think>\n\n", "<think>\n")
        if not prompt.endswith("<think>\n"):
            prompt += "<think>\n"

    ans_text = sample.get("nonthinking_answer", "").strip()
    if not ans_text.endswith("<|im_end|>"):
        ans_text += "<|im_end|>"

    if mode == "arm1_telegraphic_cot":
        thought = sample.get("telegraphic_think", "").rstrip("\n")
        target = thought + "\n</think>\n\n" + ans_text
        loss_on_thought = True
    elif mode == "control3_matched_pause":
        thought = sample.get("pause_think", "").rstrip("\n")
        target = thought + "\n</think>\n\n" + ans_text
        loss_on_thought = False
    else:
        raise ValueError(f"Unknown discrete mode: {mode}")

    return prompt, target, thought, ans_text, loss_on_thought

def compute_sample_loss_discrete(model, tokenizer, sample, mode, max_seq_len, device):
    prompt, target, thought, ans_text, loss_on_thought = format_discrete_sequence(sample, mode)
    enc_p = tokenizer.encode(prompt, add_special_tokens=False)
    enc_t = tokenizer.encode(target, add_special_tokens=False)

    if len(enc_p) + len(enc_t) > max_seq_len:
        enc_t = enc_t[: max_seq_len - len(enc_p)]

    input_ids = torch.tensor([enc_p + enc_t], dtype=torch.long, device=device)
    labels = input_ids.clone()

    if loss_on_thought:
        labels[:, :len(enc_p)] = -100  # Loss on thought + transition + answer
    else:
        # Pause control: loss only on transition + answer
        enc_thought = tokenizer.encode(thought, add_special_tokens=False)
        labels[:, :len(enc_p) + len(enc_thought)] = -100

    out = model(input_ids=input_ids, labels=labels)
    return out.loss

def compute_sample_loss_dual_channel(model, tokenizer, sample, k_latents, alpha, max_seq_len, device):
    """
    Dual-Channel Forward Pass:
    1. Prefill prompt + discrete telegraphic thought
    2. Extract last hidden state and unroll K continuous recurrent latents
    3. Feed transition (\n</think>\n\n) + answer and compute cross-entropy loss
    """
    prompt = sample.get("prompt", "")
    if not prompt.endswith("<think>\n"):
        prompt = prompt.replace("<think>\n\n</think>\n\n", "<think>\n")
        if not prompt.endswith("<think>\n"):
            prompt += "<think>\n"

    thought = sample.get("telegraphic_think", "").rstrip("\n")
    prefix_text = prompt + thought + "\n"
    enc_prefix = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).to(device)
    L_prefix = enc_prefix.input_ids.shape[1]

    # 1. Prefill prefix
    out = model(input_ids=enc_prefix.input_ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    curr_latent = out.hidden_states[-1][:, -1:, :]
    curr_seq_len = L_prefix

    backbone = model.base_model.model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "model") else model

    # 2. Unroll K continuous latents
    for k in range(k_latents):
        step_pos = torch.tensor([[curr_seq_len]], device=device, dtype=torch.long)
        scaled_latent = curr_latent * alpha
        step_out = backbone(
            inputs_embeds=scaled_latent,
            position_ids=step_pos,
            past_key_values=past_kv,
            use_cache=True
        )
        past_kv = step_out.past_key_values
        curr_latent = step_out.last_hidden_state[:, -1:, :] if hasattr(step_out, "last_hidden_state") else step_out.hidden_states[-1][:, -1:, :]
        curr_seq_len += 1

    # 3. Transition delimiter: \n</think>\n\n
    trans_text = "\n</think>\n\n"
    enc_trans = tokenizer(trans_text, return_tensors="pt", add_special_tokens=False).to(device)
    trans_len = enc_trans.input_ids.shape[1]
    trans_pos = torch.arange(curr_seq_len, curr_seq_len + trans_len, device=device).unsqueeze(0)
    trans_out = model(
        input_ids=enc_trans.input_ids,
        position_ids=trans_pos,
        past_key_values=past_kv,
        use_cache=True,
        output_hidden_states=True
    )
    past_kv = trans_out.past_key_values
    curr_seq_len += trans_len

    # 4. Answer Phase
    ans_text = sample.get("nonthinking_answer", "").strip()
    if not ans_text.endswith("<|im_end|>"):
        ans_text += "<|im_end|>"
    enc_ans = tokenizer(ans_text, return_tensors="pt", add_special_tokens=False).to(device)
    ans_len = enc_ans.input_ids.shape[1]
    ans_pos = torch.arange(curr_seq_len, curr_seq_len + ans_len, device=device).unsqueeze(0)
    ans_out = model(input_ids=enc_ans.input_ids, position_ids=ans_pos, past_key_values=past_kv, use_cache=True)

    # Compute cross-entropy on answer tokens
    logits = ans_out.logits[:, :-1, :].contiguous()
    target_ids = enc_ans.input_ids[:, 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
    return loss

def evaluate_dev(model, tokenizer, dev_records, mode, k_latents, alpha, max_seq_len, device, max_eval=100):
    model.eval()
    losses = []
    with torch.no_grad():
        for s in dev_records[:max_eval]:
            if mode == "arm2_dual_channel_latents":
                loss = compute_sample_loss_dual_channel(model, tokenizer, s, k_latents, alpha, max_seq_len, device)
            else:
                loss = compute_sample_loss_discrete(model, tokenizer, s, mode, max_seq_len, device)
            if loss is not None and not torch.isnan(loss):
                losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else 999.0

def train(args):
    print("=" * 80)
    print(f"TRAINING TELEGRAPHIC COT: MODE = {args.mode.upper()}")
    print(f"Device: {args.device} | Output Dir: {args.output_dir}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load datasets
    if args.mode == "control3_matched_pause":
        train_path = os.path.join(PROJECT_ROOT, "studies/telegraphic_cot/data/train_matched_pause_control.jsonl")
        dev_path = os.path.join(PROJECT_ROOT, "studies/telegraphic_cot/data/dev_matched_pause_control.jsonl")
    else:
        train_path = os.path.join(PROJECT_ROOT, "studies/telegraphic_cot/data/train_telegraphic_cot.jsonl")
        dev_path = os.path.join(PROJECT_ROOT, "studies/telegraphic_cot/data/dev_telegraphic_cot.jsonl")

    with open(train_path) as f:
        train_records = [json.loads(l) for l in f if l.strip()]
    with open(dev_path) as f:
        dev_records = [json.loads(l) for l in f if l.strip()]

    print(f"Loaded {len(train_records)} train and {len(dev_records)} dev records.")

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

    if args.gradient_checkpointing and args.mode != "arm2_dual_channel_latents":
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

    init_dev_loss = evaluate_dev(model, tokenizer, dev_records, args.mode, args.k_latents, args.alpha, args.max_seq_len, args.device)
    print(f"Step 0 Initial Dev Loss: {init_dev_loss:.4f}")

    best_dev_loss = init_dev_loss
    best_checkpoint_dir = os.path.join(args.output_dir, "best_checkpoint")

    global_step = 0
    accum_loss = 0.0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        rng = random.Random(args.seed + epoch)
        shuffled = list(train_records)
        rng.shuffle(shuffled)

        pbar = tqdm(shuffled, desc=f"Epoch {epoch + 1}")
        for i, sample in enumerate(pbar):
            if args.mode == "arm2_dual_channel_latents":
                loss = compute_sample_loss_dual_channel(model, tokenizer, sample, args.k_latents, args.alpha, args.max_seq_len, args.device)
            else:
                loss = compute_sample_loss_discrete(model, tokenizer, sample, args.mode, args.max_seq_len, args.device)

            loss = loss / args.accum_steps
            loss.backward()
            accum_loss += loss.item()

            if (i + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix({"train_loss": f"{accum_loss:.4f}", "step": global_step})
                accum_loss = 0.0

                if global_step % args.eval_interval == 0:
                    dev_loss = evaluate_dev(model, tokenizer, dev_records, args.mode, args.k_latents, args.alpha, args.max_seq_len, args.device)
                    print(f"\n[Step {global_step}] Dev Loss: {dev_loss:.4f} (Best: {best_dev_loss:.4f})")
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        print(f"--> New best dev loss! Saving checkpoint to {best_checkpoint_dir}")
                        model.save_pretrained(best_checkpoint_dir)
                        tokenizer.save_pretrained(best_checkpoint_dir)

    # Final dev eval and checkpoint save
    final_dev_loss = evaluate_dev(model, tokenizer, dev_records, args.mode, args.k_latents, args.alpha, args.max_seq_len, args.device)
    print(f"\nTraining Complete. Final Dev Loss: {final_dev_loss:.4f} | Best Dev Loss: {best_dev_loss:.4f}")

    if not os.path.exists(best_checkpoint_dir):
        print(f"Saving final model as best checkpoint to {best_checkpoint_dir}...")
        model.save_pretrained(best_checkpoint_dir)
        tokenizer.save_pretrained(best_checkpoint_dir)

    meta = {
        "mode": args.mode,
        "model_id": args.model_id,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "k_latents": args.k_latents,
        "alpha_scale": args.alpha,
        "best_dev_loss": best_dev_loss,
        "final_dev_loss": final_dev_loss,
        "train_samples": len(train_records),
        "dev_samples": len(dev_records)
    }
    with open(os.path.join(args.output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["arm1_telegraphic_cot", "control3_matched_pause", "arm2_dual_channel_latents"])
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--k_latents", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=ALPHA_1_7B)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
