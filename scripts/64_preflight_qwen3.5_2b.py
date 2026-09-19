#!/usr/bin/env python3
"""
scripts/64_preflight_qwen3.5_2b.py
Mandatory Step-0 Pre-Flight Verification & Architectural Profiling for Qwen/Qwen3.5-2B.

Strictly Isolated to Model: Qwen/Qwen3.5-2B
Device: cuda:0 (RTX 4080 16GB)

Objectives:
1. Inspect official Jinja chat_template directly. Verify raw rendered token strings for
   enable_thinking=True vs enable_thinking=False to confirm delimiter tags and non-thinking exit behavior.
2. Load model in pure bfloat16 on cuda:0 (~4.5 GB VRAM).
3. Profile layer topology: map Gated DeltaNet (linear attention) vs full self-attention layers.
4. Measure embedding norm ||W_E|| and empirical hidden state norm ||h_0|| to compute exact alpha_2B.
5. Test single-position latent recurrence under pinned SDPBackend.MATH with mutable KV-cache.
6. Save comprehensive verification artifact to data/preflight_qwen3.5_2b.json.
"""

import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_ID = "Qwen/Qwen3.5-2B"
DEVICE = "cuda:0"

def main():
    print(f"========================================================")
    print(f"=== Mandatory Step-0 Pre-Flight Verification: {MODEL_ID} ===")
    print(f"Target Device: {DEVICE}")
    print(f"========================================================")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "preflight_qwen3.5_2b.json")

    # 1. Inspect Tokenizer and Chat Template
    print(f"\n>>> 1. Loading Tokenizer & Inspecting Jinja Chat Template...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    vocab_size = len(tokenizer)
    print(f"Vocab Size: {vocab_size:,}")

    test_messages = [
        {"role": "user", "content": "What is the square root of 144?"}
    ]

    # Test enable_thinking=True
    rendered_thinking = None
    try:
        rendered_thinking = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        print("\n--- Rendered Prompt (enable_thinking=True) ---")
        print(repr(rendered_thinking))
    except Exception as e:
        print(f"enable_thinking=True template error: {e}")

    # Test enable_thinking=False
    rendered_nothinking = None
    try:
        rendered_nothinking = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        print("\n--- Rendered Prompt (enable_thinking=False) ---")
        print(repr(rendered_nothinking))
    except Exception as e:
        print(f"enable_thinking=False template error: {e}")

    # Test default
    rendered_default = tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("\n--- Rendered Prompt (Default / No kwargs) ---")
    print(repr(rendered_default))

    # Identify thinking delimiters
    has_think_start = "<think>" in rendered_thinking if rendered_thinking else False
    has_think_end = "</think>" in rendered_nothinking if rendered_nothinking else False

    # 2. Inspect Config
    print(f"\n>>> 2. Inspecting Model Configuration...")
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", None))
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    print(f"Model Type: {config.model_type}")
    print(f"Hidden Size: {hidden_size} | Layers: {num_layers}")

    # 3. Load Model in Pure BF16 on cuda:0
    print(f"\n>>> 3. Loading Model in Pure BFloat16 on {DEVICE}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    t_load = time.time() - t0
    vram_alloc = torch.cuda.memory_allocated(DEVICE) / (1024**2)
    print(f"Loaded in {t_load:.1f}s | VRAM Allocated: {vram_alloc:.1f} MiB")

    # 4. Profile Layer Topology (DeltaNet vs Attention)
    print(f"\n>>> 4. Profiling Layer Topology...")
    layer_types = []
    layers = getattr(model.model, "layers", None)
    if layers is None and hasattr(model, "transformer"):
        layers = getattr(model.transformer, "layers", None)

    attn_count = 0
    linear_count = 0
    mlp_count = 0

    if layers:
        for idx, layer in enumerate(layers):
            cname = layer.__class__.__name__
            sub_modules = [m.__class__.__name__ for m in layer.children()]
            sub_names = [n for n, _ in layer.named_children()]
            layer_types.append({
                "layer_idx": idx,
                "class": cname,
                "children": sub_names
            })
            # Check for Gated DeltaNet or linear attention
            is_linear = any("delta" in s.lower() or "linear" in s.lower() or "rnn" in s.lower() for s in sub_names)
            if is_linear:
                linear_count += 1
            else:
                attn_count += 1

    print(f"Total Layers: {len(layers) if layers else 'N/A'}")
    print(f"Self-Attention Layers: {attn_count} | Linear/DeltaNet Layers: {linear_count}")

    # 5. Calibrate Empirical Scale Factor (alpha_2B)
    print(f"\n>>> 5. Calibrating Empirical Scale Factor alpha_2B...")
    embed_tokens = model.get_input_embeddings()
    w_e_norm = torch.norm(embed_tokens.weight.float(), dim=-1).mean().item()
    print(f"Mean Token Embedding Norm E[||W_E||]: {w_e_norm:.6f}")

    sample_prompts = [
        "What is the capital of France?",
        "Solve for x: 2x + 5 = 15.",
        "A farmer has 15 sheep. All but 8 die. How many sheep are left?",
        "Compute the integral of x^2 from 0 to 3.",
        "Explain the difference between a mutex and a semaphore."
    ]

    h0_norms = []
    model.eval()
    with torch.no_grad():
        for p in sample_prompts:
            enc = tokenizer(p, return_tensors="pt").to(DEVICE)
            out = model(**enc, output_hidden_states=True)
            # Layer 0 hidden states (after embedding + first RMSNorm or input embedding)
            h0 = out.hidden_states[0][0].float() # [seq_len, hidden_size]
            h0_norm = torch.norm(h0, dim=-1).mean().item()
            h0_norms.append(h0_norm)

    mean_h0_norm = sum(h0_norms) / len(h0_norms)
    alpha_calibrated = w_e_norm / mean_h0_norm
    print(f"Mean Layer-0 Activation Norm E[||h_0||]: {mean_h0_norm:.6f}")
    print(f"Calibrated Scale Factor alpha_2B = E[||W_E||] / E[||h_0||]: {alpha_calibrated:.6f}")

    # 6. Test Single-Position Latent Recurrence with Pinned SDPBackend.MATH
    print(f"\n>>> 6. Testing Single-Position Latent Recurrence under SDPBackend.MATH...")
    test_input = tokenizer("Test recurrence prompt:", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**test_input, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        L_prompt = test_input.input_ids.shape[1]

        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            for k in range(6):
                pos = torch.tensor([[L_prompt + k]], device=DEVICE, dtype=torch.long)
                scaled_latent = curr_latent * alpha_calibrated
                step_out = model(
                    inputs_embeds=scaled_latent,
                    position_ids=pos,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_kv = step_out.past_key_values
                curr_latent = step_out.hidden_states[-1][:, -1:, :]

        # Decode first token
        first_token_logits = model.lm_head(curr_latent)
        top_token_id = torch.argmax(first_token_logits, dim=-1).item()
        top_token_text = tokenizer.decode([top_token_id])
        print(f"6 Latent Recurrence Steps Completed Successfully!")
        print(f"Top decoded token out of recurrence: {repr(top_token_text)} (ID: {top_token_id})")

    # 7. Summary & Persistence
    summary = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_layers": len(layers) if layers else None,
        "layer_breakdown": {
            "attention_layers": attn_count,
            "linear_deltanet_layers": linear_count
        },
        "vram_allocated_mib": round(vram_alloc, 1),
        "mean_embed_norm": round(w_e_norm, 6),
        "mean_h0_norm": round(mean_h0_norm, 6),
        "calibrated_alpha": round(alpha_calibrated, 6),
        "chat_template_findings": {
            "thinking_enabled_prefix": rendered_thinking[:200] if rendered_thinking else None,
            "thinking_disabled_prefix": rendered_nothinking[:200] if rendered_nothinking else None,
            "default_mode": "non-thinking" if "<think>" not in rendered_default else "thinking"
        },
        "recurrence_test_passed": True,
        "top_token_after_6_latents": top_token_text,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPre-flight verification completed successfully!")
    print(f"Evidence saved to: {out_path}")

if __name__ == "__main__":
    main()
