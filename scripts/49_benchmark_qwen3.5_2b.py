#!/usr/bin/env python3
"""
scripts/49_benchmark_qwen3.5_2b.py
Qwen3.5-2B Hybrid Gated DeltaNet Architecture Benchmark & Dual-State Memory Accounting

Analyzes and evaluates the architectural efficiency of Qwen3.5-2B:
1. Dual-State Memory Profiling:
   - Dynamic KV Cache (Full Attention layers): 12 KiB/token (vs 224 KiB/token for Qwen3-1.7B, 18.7x reduction)
   - Fixed Recurrent State (Gated DeltaNet linear attention): Constant ~18.3 MiB regardless of sequence length
2. Execution of GPQA Diamond Anchor (198 problems x 8 samples = 1,584 evaluations)
   - Replicates / benchmarks against official 51.6% anchor.
   - SGLang server integration with fallback to batched HuggingFace generation.
"""

import os
import re
import sys
import json
import time
import argparse
import subprocess
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

def compute_memory_accounting(config):
    """
    Computes theoretical and empirical state memory for Qwen3.5-2B hybrid architecture.
    """
    t_cfg = getattr(config, "text_config", config)
    num_layers = getattr(t_cfg, "num_hidden_layers", 24)
    layer_types = getattr(t_cfg, "layer_types", [])
    
    num_full_attn = sum(1 for lt in layer_types if lt == "full_attention")
    num_linear_attn = sum(1 for lt in layer_types if lt == "linear_attention")
    
    # Full attention KV cache per token (bfloat16 = 2 bytes)
    kv_heads = getattr(t_cfg, "num_key_value_heads", 2)
    head_dim = getattr(t_cfg, "head_dim", 256)
    bytes_per_tok_layer = 2 * (kv_heads * head_dim * 2)  # K and V
    dynamic_bytes_per_token = num_full_attn * bytes_per_tok_layer
    dynamic_kib_per_token = dynamic_bytes_per_token / 1024.0
    
    # Linear attention fixed recurrent state (FP32 = 4 bytes)
    lin_k_heads = getattr(t_cfg, "linear_num_key_heads", 16)
    lin_k_dim = getattr(t_cfg, "linear_key_head_dim", 128)
    lin_v_dim = getattr(t_cfg, "linear_value_head_dim", 128)
    state_per_layer = lin_k_heads * lin_k_dim * lin_v_dim * 4  # SSM recurrent matrix
    conv_dim = getattr(t_cfg, "linear_conv_kernel_dim", 4)
    conv_per_layer = lin_k_heads * lin_k_dim * conv_dim * 2   # 1D conv state (bf16)
    
    fixed_state_bytes = num_linear_attn * (state_per_layer + conv_per_layer)
    fixed_state_mib = fixed_state_bytes / (1024.0 * 1024.0)
    
    # Comparison model: Qwen3-1.7B standard transformer (28 full attention layers)
    qwen3_17b_layers = 28
    qwen3_17b_kv_heads = 16
    qwen3_17b_head_dim = 128
    qwen3_17b_bytes_per_tok = qwen3_17b_layers * 2 * (qwen3_17b_kv_heads * qwen3_17b_head_dim * 2)
    qwen3_17b_kib_per_tok = qwen3_17b_bytes_per_tok / 1024.0
    
    reduction_factor = qwen3_17b_kib_per_tok / dynamic_kib_per_token
    
    accounting = {
        "model_type": getattr(config, "model_type", "qwen3_5"),
        "total_layers": num_layers,
        "full_attention_layers": num_full_attn,
        "linear_attention_layers": num_linear_attn,
        "dynamic_kv_cache_bytes_per_token": dynamic_bytes_per_token,
        "dynamic_kv_cache_kib_per_token": round(dynamic_kib_per_token, 2),
        "fixed_recurrent_state_bytes": fixed_state_bytes,
        "fixed_recurrent_state_mib": round(fixed_state_mib, 2),
        "baseline_qwen3_1.7b_kib_per_token": round(qwen3_17b_kib_per_tok, 2),
        "kv_cache_growth_reduction_ratio": round(reduction_factor, 2)
    }
    return accounting

def run_gpqa_benchmark(server_url, model_id, output_file, num_samples=8, concurrency=32):
    """
    Invokes scripts/45_eval_gpqa.py against the specified model.
    """
    script_path = os.path.join(os.path.dirname(__file__), "45_eval_gpqa.py")
    cmd = [
        sys.executable, script_path,
        "--server_url", server_url,
        "--model_id", model_id,
        "--mode", "both",
        "--num_samples", str(num_samples),
        "--concurrency", str(concurrency),
        "--output_file", output_file
    ]
    print(f"Executing GPQA benchmark: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-2B Benchmark & Dual-State Profiler")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--skip_eval", action="store_true", help="Only run architecture memory accounting")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--output_file", type=str, default="data/eval_gpqa_diamond_2b.json")
    parser.add_argument("--accounting_output", type=str, default="data/qwen3.5_2b_memory_accounting.json")
    args = parser.parse_args()

    print("=" * 70)
    print(f"QWEN3.5-2B GATED DELTANET ARCHITECTURE BENCHMARK")
    print("=" * 70)

    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    accounting = compute_memory_accounting(cfg)
    
    print("\n--- Dual-State Memory Accounting ---")
    print(f"Architecture:              {accounting['total_layers']} Layers ({accounting['full_attention_layers']} Full Attn + {accounting['linear_attention_layers']} Gated DeltaNet)")
    print(f"Dynamic KV Cache Growth:   {accounting['dynamic_kv_cache_kib_per_token']} KiB/token")
    print(f"Fixed Recurrent State:     {accounting['fixed_recurrent_state_mib']} MiB (Constant across context length)")
    print(f"Qwen3-1.7B KV Cache Growth:{accounting['baseline_qwen3_1.7b_kib_per_token']} KiB/token")
    print(f"Dynamic Growth Reduction:  {accounting['kv_cache_growth_reduction_ratio']}x Less VRAM per token")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.accounting_output), exist_ok=True)
    with open(args.accounting_output, "w") as f:
        json.dump(accounting, f, indent=2)
    print(f"Saved memory accounting to: {args.accounting_output}")

    if not args.skip_eval:
        print("\n--- Executing GPQA Diamond Evaluation (1,584 Samples) ---")
        run_gpqa_benchmark(
            server_url=args.server_url,
            model_id=args.model_id,
            output_file=args.output_file,
            num_samples=args.num_samples
        )

if __name__ == "__main__":
    main()
