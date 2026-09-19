#!/usr/bin/env python3
"""
scripts/70_preflight_gemma4_e2b.py
Mandatory Step-0 Pre-Flight Verification & Architectural Profiling for google/gemma-4-E2B-it.

Strictly Isolated to Model: google/gemma-4-E2B-it
Device: cuda:0 (RTX 4080 16GB)

Objectives:
1. Inspect official Jinja chat_template directly. Verify raw rendered token strings for
   thinking mode vs non-thinking mode to confirm delimiter tags (<|turn>model\n, <|channel>thought\n, <channel|>).
2. Load model in pure bfloat16 on cuda:0 (~9.5 GB VRAM).
3. Profile layer topology: inspect heterogeneous per_layer_config across all 35 layers.
4. Measure embedding norm E[||W_E||] and empirical hidden state norm E[||h_0||] to compute exact alpha_gemma4_e2b.
5. Test single-position latent recurrence under pinned SDPBackend.MATH with mutable KV-cache.
6. Save comprehensive verification artifact to data/preflight_gemma4_e2b.json.
"""

import os
import sys
import json
import time
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

MODEL_ID = "google/gemma-4-E2B-it"
DEVICE = "cuda:0"

def main():
    print(f"========================================================")
    print(f"=== Mandatory Step-0 Pre-Flight Verification: {MODEL_ID} ===")
    print(f"Target Device: {DEVICE}")
    print(f"========================================================")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "preflight_gemma4_e2b.json")

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
    has_thought_channel = "<|channel>thought" in rendered_thinking if rendered_thinking else False

    # 2. Inspect Config
    print(f"\n>>> 2. Inspecting Model Configuration...")
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    hidden_size = getattr(text_config, "hidden_size", 1536)
    num_layers = getattr(text_config, "num_hidden_layers", 35)
    num_heads = getattr(text_config, "num_attention_heads", 8)
    num_kv_heads = getattr(text_config, "num_key_value_heads", 1)
    print(f"Model Type: {config.model_type}")
    print(f"Hidden Size: {hidden_size} | Layers: {num_layers} | Heads: {num_heads} | KV Heads: {num_kv_heads}")

    # 3. Load Model in Pure BF16 on cuda:0
    print(f"\n>>> 3. Loading Model in Pure BFloat16 on {DEVICE}...")
    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=DEVICE
    )
    t_load = time.time() - t0
    vram_alloc = torch.cuda.memory_allocated(DEVICE) / (1024**2)
    print(f"Loaded in {t_load:.1f}s | VRAM Allocated: {vram_alloc:.1f} MiB")

    # 4. Profile Layer Topology
    print(f"\n>>> 4. Profiling Layer Topology...")
    layer_info = []
    text_model = getattr(model, "model", model)
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    layers = getattr(text_model, "layers", None)

    if layers:
        for idx, layer in enumerate(layers):
            cname = layer.__class__.__name__
            sub_names = [n for n, _ in layer.named_children()]
            layer_info.append({
                "layer_idx": idx,
                "class": cname,
                "children": sub_names
            })

    print(f"Total Text Layers: {len(layers) if layers else 'N/A'}")

    # 5. Calibrate Empirical Scale Factor (alpha_gemma4_e2b)
    print(f"\n>>> 5. Calibrating Empirical Scale Factor alpha_gemma4_e2b...")
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
    print(f"Calibrated Scale Factor alpha_gemma4_e2b = E[||W_E||] / E[||h_0||]: {alpha_calibrated:.6f}")

    # 6. Test Single-Position Latent Recurrence with Pinned SDPBackend.MATH
    print(f"\n>>> 6. Testing Single-Position Latent Recurrence under SDPBackend.MATH...")
    test_input = tokenizer("Test recurrence prompt:", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**test_input, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        L_prompt = test_input.input_ids.shape[1]

        # Gemma 4 uses Positional / Per-Layer Embeddings (PLE)
        # Pre-registered Option (b) default: Vocabulary-mean PLE vector
        # E_{v in V}[PLE[v]] in R^{1 x 1 x num_layers x 256}
        ple_weights = model.language_model.embed_tokens_per_layer.weight
        mean_ple = ple_weights.float().mean(dim=0).view(1, 1, num_layers, 256).to(dtype=torch.bfloat16, device=DEVICE)

        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            for k in range(6):
                pos = torch.tensor([[L_prompt + k]], device=DEVICE, dtype=torch.long)
                scaled_latent = curr_latent * alpha_calibrated
                step_out = model(
                    inputs_embeds=scaled_latent,
                    per_layer_inputs=mean_ple,
                    position_ids=pos,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_kv = step_out.past_key_values
                curr_latent = step_out.hidden_states[-1][:, -1:, :]

        first_token_logits = model.lm_head(curr_latent)
        top_token_id = torch.argmax(first_token_logits, dim=-1).item()
        top_token_text = tokenizer.decode([top_token_id])
        print(f"6 Latent Recurrence Steps Completed Successfully!")
        print(f"Top decoded token out of recurrence: {repr(top_token_text)} (ID: {top_token_id})")

    # 7. Compile Artifact
    artifact = {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "vram_allocated_mib": round(vram_alloc, 1),
        "chat_template_analysis": {
            "rendered_thinking": rendered_thinking,
            "rendered_nothinking": rendered_nothinking,
            "rendered_default": rendered_default,
            "has_thought_channel": has_thought_channel
        },
        "empirical_calibration": {
            "mean_token_embedding_norm": round(w_e_norm, 6),
            "mean_h0_norm": round(mean_h0_norm, 6),
            "calibrated_alpha": round(alpha_calibrated, 6)
        },
        "recurrence_verification": {
            "k_steps": 6,
            "sdp_backend": "SDPBackend.MATH",
            "status": "PASSED",
            "top_decoded_token": top_token_text,
            "top_decoded_token_id": top_token_id
        }
    }

    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\n========================================================")
    print(f"Pre-flight verification complete! Saved to: {out_path}")
    print(f"Calibrated Alpha (Gemma4-E2B): {alpha_calibrated:.6f}")
    print(f"========================================================")

if __name__ == "__main__":
    main()
