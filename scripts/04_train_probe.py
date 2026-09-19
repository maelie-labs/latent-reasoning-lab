#!/usr/bin/env python3
"""
04_train_probe.py
Trains the auxiliary Thought-Stream Projection Probe.
Takes intermediate continuous states h_t (in R^1536) and projects them into vocabulary space,
supervised by the teacher model's initial reasoning tokens.
"""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_file", type=str, default="data/traces_gsm8k_200.jsonl")
    parser.add_argument("--save_path", type=str, default="checkpoints/thought_probe.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--recurrent_steps", type=int, default=6)
    return parser.parse_args()

def main():
    args = parse_args()
    print("=== EXP-05: Training Auxiliary Thought-Stream Probe ===")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
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
    scale_factor = 1.1641 / (hidden_dim ** 0.5)
    
    probe = ThoughtStreamProbe(hidden_dim, vocab_size).to(DEVICE, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    dataset = TraceDataset(args.traces_file)
    
    def collate_fn(batch):
        prompts = [item["formatted_prompt"] for item in batch]
        thinks = [item["think_text"] for item in batch]
        return prompts, thinks
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    print(f"Training probe across {len(dataset)} examples for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        steps = 0
        
        for prompts, think_texts in dataloader:
            enc_prompts = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
            
            # Supervise each recurrent step with the teacher's rationale tokens
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
                    
            # [batch_size, recurrent_steps, hidden_dim]
            all_latents = torch.cat(latents_per_step, dim=1)
            
            logits = probe(all_latents)  # [batch_size, recurrent_steps, vocab_size]
            
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            steps += 1
            
        avg_loss = epoch_loss / max(1, steps)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")
        
    torch.save(probe.state_dict(), args.save_path)
    total_time = time.time() - start_time
    print(f"Probe saved to {args.save_path} in {total_time:.1f}s")

if __name__ == "__main__":
    main()
