#!/usr/bin/env python3
"""
scripts/112_preflight_profiling_4b.py
Architectural Profiling & Empirical Scale Factor Calibration for 4B Models.

Evaluates on GPU 0 (RTX 4080 16GB):
1. Qwen/Qwen3.5-4B (Hybrid Gated DeltaNet)
2. Qwen/Qwen3-4B (Pure Dense Transformer)

Measures:
- Layer topology (linear attention vs full attention)
- Token embedding norm E[||W_E||]
- Layer 0 hidden state norm E[||h_0||]
- Empirical scale factor alpha = E[||W_E||] / E[||h_0||]
- KV cache and recurrent state memory accounting
- Chat template delimiter inspection
"""

import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

DEVICE = "cuda:0"

SAMPLE_PROMPTS = [
    "What is the square root of 144?",
    "Solve for x: 2x + 5 = 15.",
    "A farmer has 15 sheep. All but 8 die. How many sheep are left?",
    "Compute the integral of x^2 from 0 to 3.",
    "Explain the difference between a mutex and a semaphore.",
    "If f(x) = x^2 - 4x + 4, find f'(2)."
]

def profile_model(model_id):
    print(f"\n========================================================")
    print(f"=== Profiling Model: {model_id} on {DEVICE} ===")
    print(f"========================================================")
    
    # 1. Tokenizer and Templates
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    vocab_size = len(tok)
    
    test_msg = [{"role": "user", "content": "Solve 2+2."}]
    p_think = tok.apply_chat_template(test_msg, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    p_nothink = tok.apply_chat_template(test_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    
    # 2. Config & Architecture
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    t_cfg = getattr(config, "text_config", config)
    hidden_size = getattr(t_cfg, "hidden_size", getattr(t_cfg, "d_model", None))
    num_layers = getattr(t_cfg, "num_hidden_layers", getattr(t_cfg, "n_layer", None))
    layer_types = getattr(t_cfg, "layer_types", ["full_attention"] * (num_layers or 0))
    
    num_full = sum(1 for lt in layer_types if lt == "full_attention")
    num_linear = sum(1 for lt in layer_types if lt == "linear_attention")
    
    # 3. Load Model in BF16
    print(f"Loading {model_id} in pure bfloat16 on {DEVICE}...")
    torch.cuda.empty_cache()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    load_time = time.time() - t0
    alloc_mib = torch.cuda.memory_allocated(DEVICE) / (1024**2)
    print(f"Loaded in {load_time:.2f}s | VRAM: {alloc_mib:.1f} MiB")
    
    # 4. Measure Embedding Norm
    embed_tokens = model.get_input_embeddings()
    w_e_norm = torch.norm(embed_tokens.weight.float(), dim=-1).mean().item()
    print(f"Embedding Norm E[||W_E||]: {w_e_norm:.6f}")
    
    # 5. Measure Layer-0 Activation Norm
    model.eval()
    h0_norms = []
    with torch.no_grad():
        for p in SAMPLE_PROMPTS:
            enc = tok(p, return_tensors="pt").to(DEVICE)
            out = model(**enc, output_hidden_states=True)
            h0 = out.hidden_states[0][0].float()
            norm = torch.norm(h0, dim=-1).mean().item()
            h0_norms.append(norm)
            
    mean_h0 = float(sum(h0_norms) / len(h0_norms))
    alpha = float(w_e_norm / mean_h0)
    print(f"Layer-0 Norm E[||h_0||]: {mean_h0:.6f}")
    print(f"Calibrated Scale Factor alpha: {alpha:.6f}")
    
    # Clean up model from GPU 0
    del model
    del embed_tokens
    torch.cuda.empty_cache()
    
    return {
        "model_id": model_id,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_full_attention": num_full,
        "num_linear_attention": num_linear,
        "layer_types": layer_types,
        "w_e_norm": round(w_e_norm, 6),
        "h0_norm": round(mean_h0, 6),
        "alpha": round(alpha, 6),
        "prompt_thinking_prefix": p_think,
        "prompt_nothinking_prefix": p_nothink
    }

def main():
    models = ["Qwen/Qwen3.5-4B", "Qwen/Qwen3-4B"]
    results = {}
    for m in models:
        res = profile_model(m)
        results[m] = res
        
    out_path = "data/preflight_profiling_4b.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved profiling artifact to {out_path}")

if __name__ == "__main__":
    main()
