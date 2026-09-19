#!/usr/bin/env python3
"""
scripts/26_verify_qwen3_1.7b.py
Architectural verification and norm-scaling factor derivation for Qwen/Qwen3-1.7B on GPU 0.
Tests pure uncompressed bfloat16 loading, GQA attention layout (16 heads, 8 kv heads = 2:1),
and embedding space dynamics.
"""

import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils

# Disable caching_allocator_warmup to prevent excessive VRAM pre-allocation
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

# Load HF token from local .env if available
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
hf_token = None
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                hf_token = line.split("=", 1)[1].strip("\"'\n")
                break

MODEL_ID = "Qwen/Qwen3-1.7B"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

def main():
    print(f"=== Profiling & Verifying {MODEL_ID} on {DEVICE} ===")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")
    
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    vram_before = torch.cuda.memory_allocated() / (1024**3)
    print(f"Initial VRAM Allocated: {vram_before:.2f} GB")
    
    print("\n1. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token, trust_remote_code=True)
    
    print("2. Loading Model in pure bfloat16...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True
    )
    load_time = time.time() - t0
    vram_model = torch.cuda.memory_allocated() / (1024**3)
    print(f"Model loaded in {load_time:.2f}s | VRAM: {vram_model:.2f} GB")
    
    # 3. Model Architecture Details
    config = model.config
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    gqa_ratio = num_heads // num_kv_heads
    vocab_size = config.vocab_size
    
    print("\n3. Architecture Specification:")
    print(f"  Model Type:            {getattr(config, 'model_type', 'unknown')}")
    print(f"  Hidden Size (d_model): {hidden_size}")
    print(f"  Hidden Layers:         {num_layers}")
    print(f"  Attention Heads (Q):   {num_heads}")
    print(f"  KV Heads (K, V):       {num_kv_heads}")
    print(f"  GQA Ratio:             {gqa_ratio}:1")
    print(f"  Vocab Size:            {vocab_size}")
    
    # 4. Measure Embedding Norms and Compute Alpha
    embed_tokens = model.model.embed_tokens.weight.data
    norm_fro = torch.norm(embed_tokens.float(), p="fro").item()
    norm_spectral = torch.linalg.matrix_norm(embed_tokens.float(), ord=2).item()
    mean_row_norm = torch.norm(embed_tokens.float(), dim=-1).mean().item()
    std_row_norm = torch.norm(embed_tokens.float(), dim=-1).std().item()
    
    # Scale factor derivation: alpha = ||W_E|| / sqrt(d_model)
    alpha = mean_row_norm / (hidden_size ** 0.5)
    
    print("\n4. Embedding Space & Scale Factor Derivation:")
    print(f"  Embedding Matrix Shape: {list(embed_tokens.shape)}")
    print(f"  Mean Row Norm (||W_E||): {mean_row_norm:.6f} +/- {std_row_norm:.6f}")
    print(f"  Frobenius Norm:          {norm_fro:.2f}")
    print(f"  Spectral Norm:           {norm_spectral:.4f}")
    print(f"  Calculated Alpha Factor: {alpha:.6f}  (Formula: {mean_row_norm:.6f} / sqrt({hidden_size}))")
    
    # 5. Baseline Greedy Generation Test
    test_prompt = "Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. She sells the remainder for $2 each. How much does she make per day?"
    messages = [{"role": "user", "content": test_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    torch.cuda.synchronize()
    t_gen_start = time.time()
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )
    torch.cuda.synchronize()
    gen_time = time.time() - t_gen_start
    
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    num_tokens = len(new_tokens)
    tok_per_sec = num_tokens / gen_time if gen_time > 0 else 0
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    
    print("\n5. Baseline Generation Test:")
    print(f"  Tokens Generated: {num_tokens} tokens in {gen_time:.2f}s ({tok_per_sec:.1f} tok/s)")
    print(f"  Peak VRAM:        {peak_vram:.2f} GB")
    print(f"  Generated Response:\n{'-'*40}\n{response_text}\n{'-'*40}")
    
    # 6. Save Profiling Results
    results = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "vram_model_gb": round(vram_model, 2),
        "peak_vram_gb": round(peak_vram, 2),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "gqa_ratio": gqa_ratio,
        "vocab_size": vocab_size,
        "mean_row_norm": round(mean_row_norm, 6),
        "std_row_norm": round(std_row_norm, 6),
        "alpha_factor": round(alpha, 6),
        "baseline_tok_per_sec": round(tok_per_sec, 2),
        "test_prompt": test_prompt,
        "test_response": response_text
    }
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "qwen3_1.7b_profiling_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved profiling results to: {out_file}")

if __name__ == "__main__":
    main()
