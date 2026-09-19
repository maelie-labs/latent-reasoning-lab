#!/usr/bin/env python3
"""
scripts/75_profile_latent_step.py
Profile GPU kernel vs idle time, memory footprint, and launch overhead
in the continuous latent recurrence loop (K=6 and K=32) for Qwen/Qwen3-1.7B.
"""

import os
import time
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def profile_latent_loop(device="cuda:0", k_tokens=6, micro_batch=1, use_profiler=True):
    print(f"\n========================================================")
    print(f"Profiling Latent Loop: Device={device}, K={k_tokens}, Micro-Batch={micro_batch}")
    print(f"========================================================")
    
    model_id = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.train()
    
    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    embed_weight = model.get_input_embeddings().weight
    d_model = embed_weight.shape[-1]
    scale_factor = 0.011440
    
    # Synthetic test batch
    seq_len = 128
    dummy_input_ids = torch.randint(0, 1000, (micro_batch, seq_len), device=device)
    dummy_target_ids = torch.randint(0, 1000, (micro_batch, 64), device=device)
    
    # Warmup
    print("Running 3 warmup steps...")
    for _ in range(3):
        out = model(input_ids=dummy_input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        for k in range(k_tokens):
            pos = torch.tensor([[seq_len + k]], device=device, dtype=torch.long).expand(micro_batch, 1)
            scaled = curr_latent * scale_factor
            step_out = model(inputs_embeds=scaled, position_ids=pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]
        logits = model.lm_head(curr_latent)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), dummy_target_ids[:, :1].view(-1))
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize(device)
    
    # Timed benchmark
    print(f"Timing 5 iterations...")
    t0 = time.time()
    for _ in range(5):
        out = model(input_ids=dummy_input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        for k in range(k_tokens):
            pos = torch.tensor([[seq_len + k]], device=device, dtype=torch.long).expand(micro_batch, 1)
            scaled = curr_latent * scale_factor
            step_out = model(inputs_embeds=scaled, position_ids=pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]
        logits = model.lm_head(curr_latent)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), dummy_target_ids[:, :1].view(-1))
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize(device)
    avg_dur = (time.time() - t0) / 5.0
    print(f"Average wall-clock per step: {avg_dur*1000:.1f} ms | Per-sequence: {(avg_dur/micro_batch)*1000:.1f} ms")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated(device)/1024**2:.1f} MiB")

if __name__ == "__main__":
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    profile_latent_loop(device=dev, k_tokens=6, micro_batch=1)
