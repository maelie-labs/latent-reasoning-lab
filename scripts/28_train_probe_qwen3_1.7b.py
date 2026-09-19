#!/usr/bin/env python3
"""
scripts/28_train_probe_qwen3_1.7b.py
Trains the auxiliary Thought-Stream Projection Probe for Qwen/Qwen3-1.7B.
Maps intermediate continuous latent states h_t (in R^2048) to vocabulary space (151936).
Saves checkpoint to checkpoints/thought_probe_qwen3_1.7b.pt.

Memory Optimized: Pre-extracts latent states in a single inference pass, frees base model
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

MODEL_ID = "Qwen/Qwen3-1.7B"
DEVICE = "cuda:0"
SCALE_FACTOR = 0.034016  # ||W_E|| / sqrt(2048)

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
    default_save = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "thought_probe_qwen3_1.7b.pt")
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
    
    extracted_records = []
    t_extract0 = time.time()
    
    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            prompt = sample["formatted_prompt"]
            think_text = sample["think_text"]
            
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            out = base_model(input_ids=inputs.input_ids, use_cache=True, output_hidden_states=True)
            past_kv = out.past_key_values
            curr_h = out.hidden_states[-1][:, -1:, :]
            
            step_latents = []
            for _ in range(args.recurrent_steps):
                scaled_h = curr_h * SCALE_FACTOR
                step_out = base_model(
                    inputs_embeds=scaled_h,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_kv = step_out.past_key_values
                curr_h = step_out.hidden_states[-1][:, -1:, :]
                step_latents.append(curr_h.squeeze(0).squeeze(0).cpu())  # [hidden_dim]
                
            toks = tokenizer.encode(think_text, add_special_tokens=False)
            if len(toks) < args.recurrent_steps:
                toks = toks + [tokenizer.eos_token_id] * (args.recurrent_steps - len(toks))
            target_ids = toks[:args.recurrent_steps]
            
            extracted_records.append({
                "latents": torch.stack(step_latents),  # [recurrent_steps, hidden_dim]
                "targets": torch.tensor(target_ids, dtype=torch.long)  # [recurrent_steps]
            })
            
            if (idx + 1) % 50 == 0:
                print(f"  Extracted {idx + 1}/{len(dataset)} samples...")
                
    extract_time = time.time() - t_extract0
    print(f"Extracted {len(extracted_records)} trajectories in {extract_time:.2f}s.")
    
    # Free base model from VRAM completely
    print("Freeing base model from VRAM...")
    del base_model
    del past_kv
    torch.cuda.empty_cache()
    
    # Phase 2: Probe Training on Extracted Latents
    print("\nPhase 2: Training ThoughtStreamProbe on GPU 0...")
    probe = ThoughtStreamProbe(hidden_dim, vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Flatten across steps: each step is an independent (latent, target) training pair
    all_latents = torch.cat([r["latents"] for r in extracted_records], dim=0)  # [N * K, hidden_dim]
    all_targets = torch.cat([r["targets"] for r in extracted_records], dim=0)  # [N * K]
    
    train_dataset = torch.utils.data.TensorDataset(all_latents, all_targets)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"Training on {len(train_dataset)} latent-token projection pairs for {args.epochs} epochs...")
    t_train0 = time.time()
    
    for epoch in range(args.epochs):
        probe.train()
        total_loss = 0.0
        
        for batch_latents, batch_targets in train_loader:
            batch_latents = batch_latents.to(DEVICE).float()
            batch_targets = batch_targets.to(DEVICE)
            
            optimizer.zero_grad()
            logits = probe(batch_latents)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Cross-Entropy Loss: {avg_loss:.4f}")
        
    train_time = time.time() - t_train0
    print(f"Probe training completed in {train_time:.2f}s.")
    
    torch.save(probe.state_dict(), args.save_path)
    print(f"Saved probe checkpoint to: {args.save_path}")

if __name__ == "__main__":
    main()
