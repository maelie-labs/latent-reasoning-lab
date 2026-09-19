#!/usr/bin/env python3
"""
scripts/32_train_probe_qwen3_4b.py
Trains the auxiliary Thought-Stream Projection Probe for Qwen/Qwen3-4B on GPU 1.
Maps intermediate continuous latent states h_t (in R^2560) to vocabulary space (151936).
Saves checkpoint to checkpoints/thought_probe_qwen3_4b.pt.

Memory Optimized: Pre-extracts latent states in a single inference pass on cuda:1, frees base model
from VRAM, and then trains the probe network with minimal footprint (<1.5 GB).
"""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

MODEL_ID = "Qwen/Qwen3-4B"
DEVICE = "cuda:1"

class ThoughtStreamProbe(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size, bias=False)
        )
        
    def forward(self, h):
        return self.net(h)

class TraceDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=150):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data.get("valid_format", False) and len(data.get("think_text", "")) > 10:
                    self.examples.append(data)
                if len(self.examples) >= max_samples:
                    break
        print(f"Loaded {len(self.examples)} valid reasoning traces from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def parse_args():
    default_traces = os.path.join(os.path.dirname(__file__), "..", "data", "traces_gsm8k_qwen3_1.7b.jsonl")
    default_save = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "thought_probe_qwen3_4b.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_file", type=str, default=default_traces)
    parser.add_argument("--save_path", type=str, default=default_save)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--recurrent_steps", type=int, default=6)
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"=== Training Thought-Stream Probe for {MODEL_ID} on {DEVICE} ===")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Load scale factor from profiling results if available
    prof_file = os.path.join(os.path.dirname(__file__), "..", "data", "qwen3_4b_profiling_results.json")
    if os.path.exists(prof_file):
        with open(prof_file) as f:
            prof = json.load(f)
            scale_factor = prof.get("alpha_factor", 0.030)
    else:
        scale_factor = 0.030
    print(f"Using scale factor alpha: {scale_factor:.6f}")
    
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    hf_token = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip("\"'\n")
                    break
                    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = TraceDataset(args.traces_file, max_samples=150)
    
    print("\nPhase 1: Loading base model to extract recurrent latent trajectories...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    base_model.eval()
    hidden_dim = base_model.config.hidden_size
    vocab_size = base_model.config.vocab_size
    
    latent_samples = []  # tuples of (h_t, target_token_id)
    print("Extracting latent representations across reasoning steps...")
    t0 = time.time()
    
    with torch.no_grad():
        for i, item in enumerate(dataset):
            prompt = item["formatted_prompt"]
            think_tokens = tokenizer.encode(item["think_text"], add_special_tokens=False)
            if len(think_tokens) < args.recurrent_steps:
                continue
                
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            out = base_model(input_ids=inputs.input_ids, use_cache=True, output_hidden_states=True)
            past_kv = out.past_key_values
            curr_latent = out.hidden_states[-1][:, -1:, :]
            
            step_stride = max(1, len(think_tokens) // args.recurrent_steps)
            
            for s in range(args.recurrent_steps):
                target_token_idx = min((s + 1) * step_stride - 1, len(think_tokens) - 1)
                target_token_id = think_tokens[target_token_idx]
                
                # Save latent vector on CPU to conserve VRAM
                h_vec = curr_latent.squeeze().detach().cpu()
                latent_samples.append((h_vec, target_token_id))
                
                scaled_latent = curr_latent * scale_factor
                step_out = base_model(
                    inputs_embeds=scaled_latent,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_kv = step_out.past_key_values
                curr_latent = step_out.hidden_states[-1][:, -1:, :]
                
            if (i + 1) % 25 == 0:
                print(f"  Processed {i+1}/{len(dataset)} traces | Total pairs: {len(latent_samples)}")
                
    extract_time = time.time() - t0
    print(f"Extracted {len(latent_samples)} (latent, token) pairs in {extract_time:.2f}s")
    
    print("\nPhase 2: Freeing base model from VRAM...")
    del base_model
    torch.cuda.empty_cache()
    
    # Probe training on GPU 1
    probe = ThoughtStreamProbe(hidden_dim, vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Create tensor dataset
    H_tensor = torch.stack([pair[0] for pair in latent_samples]).float()
    Y_tensor = torch.tensor([pair[1] for pair in latent_samples], dtype=torch.long)
    
    num_samples = len(H_tensor)
    indices = torch.randperm(num_samples)
    split = int(0.85 * num_samples)
    train_idx, val_idx = indices[:split], indices[split:]
    
    print(f"\nPhase 3: Training probe on {len(train_idx)} train / {len(val_idx)} val samples...")
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        probe.train()
        epoch_loss = 0.0
        num_batches = 0
        
        train_perm = train_idx[torch.randperm(len(train_idx))]
        for b_start in range(0, len(train_perm), args.batch_size):
            b_end = min(b_start + args.batch_size, len(train_perm))
            batch_indices = train_perm[b_start:b_end]
            
            h_b = H_tensor[batch_indices].to(DEVICE)
            y_b = Y_tensor[batch_indices].to(DEVICE)
            
            optimizer.zero_grad()
            logits = probe(h_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_train = epoch_loss / num_batches
        
        # Validation
        probe.eval()
        with torch.no_grad():
            h_val = H_tensor[val_idx].to(DEVICE)
            y_val = Y_tensor[val_idx].to(DEVICE)
            val_logits = probe(h_val)
            val_loss = criterion(val_logits, y_val).item()
            
        print(f"  Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": probe.state_dict(),
                "hidden_dim": hidden_dim,
                "vocab_size": vocab_size,
                "val_loss": val_loss,
                "scale_factor": scale_factor,
                "model_id": MODEL_ID
            }, args.save_path)
            
    print(f"\nTraining Complete! Best Val Loss: {best_val_loss:.4f}")
    print(f"Probe saved to: {args.save_path}")

if __name__ == "__main__":
    main()
