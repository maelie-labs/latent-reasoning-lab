#!/usr/bin/env python3
"""
12_train_probe_7b.py
Trains the auxiliary Thought-Stream Projection Probe for DeepSeek-R1-Distill-Qwen-7B.
Maps intermediate continuous latent states h_t (in R^3584) to vocabulary space (152064).
Saves checkpoint to checkpoints/thought_probe_7b.pt.
Runs on cuda:1.
"""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
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
    def __init__(self, jsonl_path, max_samples=130):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_file", type=str, default="data/traces_gsm8k_7b.jsonl")
    parser.add_argument("--save_path", type=str, default="checkpoints/thought_probe_7b.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--recurrent_steps", type=int, default=6)
    return parser.parse_args()

def main():
    args = parse_args()
    print("=== Stage 3: Training Auxiliary Thought-Stream Probe (7B) ===")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading base model {MODEL_ID} on {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=DEVICE
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    hidden_dim = model.config.hidden_size
    vocab_size = model.config.vocab_size
    embed_weights = model.get_input_embeddings().weight.detach()
    avg_embed_norm = torch.norm(embed_weights, dim=-1).mean().item()
    scale_factor = avg_embed_norm / (hidden_dim ** 0.5)
    print(f"Hidden Dim: {hidden_dim}, Vocab: {vocab_size}, Scale Factor: {scale_factor:.6f}")
    
    probe = ThoughtStreamProbe(hidden_dim, vocab_size).to(DEVICE, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    dataset = TraceDataset(args.traces_file)
    
    def collate_fn(batch):
        prompts = [item["formatted_prompt"] for item in batch]
        thinks = [item["think_text"] for item in batch]
        return prompts, thinks
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    print(f"Training probe on {len(dataset)} traces for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        steps = 0
        ep_t0 = time.time()
        
        for b_idx, (prompts, think_texts) in enumerate(dataloader):
            enc_prompts = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
            enc_targets = tokenizer(
                think_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.recurrent_steps
            ).to(DEVICE)
            target_ids = enc_targets.input_ids
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                out = model(
                    enc_prompts.input_ids,
                    attention_mask=enc_prompts.attention_mask,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_kv = out.past_key_values
                curr_latent = out.hidden_states[-1][:, -1:, :]
                
                latents_per_step = []
                for step in range(args.recurrent_steps):
                    step_out = model(
                        inputs_embeds=curr_latent * scale_factor,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_hidden_states=True
                    )
                    past_kv = step_out.past_key_values
                    curr_latent = step_out.hidden_states[-1][:, -1:, :]
                    latents_per_step.append(curr_latent)
                    
            all_latents = torch.cat(latents_per_step, dim=1)
            logits = probe(all_latents)
            
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            if (b_idx + 1) % 20 == 0 or (b_idx + 1) == len(dataloader):
                print(f"  Epoch {epoch+1} [{b_idx+1}/{len(dataloader)}] Batch Loss: {loss.item():.4f}")
            
        avg_loss = epoch_loss / max(1, steps)
        ep_time = time.time() - ep_t0
        print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f} | Time: {ep_time:.1f}s")
        
    torch.save(probe.state_dict(), args.save_path)
    total_time = time.time() - start_time
    print(f"7B Probe successfully saved to {args.save_path} in {total_time:.1f}s")

if __name__ == "__main__":
    main()
