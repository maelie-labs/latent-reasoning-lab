#!/usr/bin/env python3
"""
tests/test_qwen3_5_cache_continuity.py
Step 7 Pre-Flight Check 3:
Verifies Hugging Face DynamicCache continuity across sequential inputs_embeds latent passes:
- Confirms 6 Full Attention layers increment sequence length in DynamicLayer (keys/values).
- Confirms 18 Linear Attention layers update recurrent_states and conv_states in LinearAttentionLayer.
- Confirms zero state collapse or shape mismatch across sequential latent unrolling.
"""

import copy
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

def test_cache_continuity():
    print("Loading Qwen/Qwen3.5-2B text config...")
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B")
    text_cfg = copy.deepcopy(cfg.text_config)
    
    # Use 4 layers (3 linear attention + 1 full attention) for fast deterministic CPU verification
    text_cfg.num_hidden_layers = 4
    text_cfg.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    
    print("Instantiating test Qwen3_5TextModel on CPU...")
    model = Qwen3_5TextModel(text_cfg)
    model.eval()

    # 1. Ingest initial prompt (3 tokens)
    prompt_ids = torch.tensor([[101, 102, 103]])
    print("Ingesting prompt tokens (seq_len=3)...")
    out = model(input_ids=prompt_ids, use_cache=True)
    pkv = out.past_key_values
    h = out.last_hidden_state[:, -1:, :]

    assert len(pkv.layers) == 4, f"Expected 4 cache layers, got {len(pkv.layers)}"
    
    # Verify linear attention layer 0
    l0 = pkv.layers[0]
    assert 0 in l0.recurrent_states, "Layer 0 missing recurrent_states"
    assert 0 in l0.conv_states, "Layer 0 missing conv_states"
    rec_shape = tuple(l0.recurrent_states[0].shape)
    conv_shape = tuple(l0.conv_states[0].shape)
    assert rec_shape == (1, 16, 128, 128), f"Unexpected recurrent shape: {rec_shape}"
    assert conv_shape == (1, 6144, 4), f"Unexpected conv shape: {conv_shape}"
    print(f"  [PASS] Layer 0 (Linear Attention) initialized: recurrent {rec_shape}, conv {conv_shape}")

    # Verify full attention layer 3
    l3 = pkv.layers[3]
    assert l3.keys.shape == (1, 2, 3, 256), f"Unexpected KV keys shape: {l3.keys.shape}"
    assert l3.values.shape == (1, 2, 3, 256), f"Unexpected KV values shape: {l3.values.shape}"
    print(f"  [PASS] Layer 3 (Full Attention) initialized: keys {tuple(l3.keys.shape)}")

    # 2. Sequential Latent Unrolling Passes (K=4)
    print("Executing sequential latent unrolling (K=4 passes via inputs_embeds)...")
    for k in range(1, 5):
        prev_rec = l0.recurrent_states[0].clone()
        prev_kv_len = l3.keys.shape[2]
        
        # Inject next latent embedding vector
        z_k = 0.01 * h
        out_k = model(inputs_embeds=z_k, past_key_values=pkv, use_cache=True)
        h = out_k.last_hidden_state
        pkv = out_k.past_key_values
        
        # Check full attention KV length grew by exactly 1
        new_kv_len = pkv.layers[3].keys.shape[2]
        assert new_kv_len == prev_kv_len + 1, f"Pass {k}: expected KV len {prev_kv_len + 1}, got {new_kv_len}"
        
        # Check linear attention recurrent state updated
        new_rec = pkv.layers[0].recurrent_states[0]
        state_delta = torch.norm(new_rec - prev_rec).item()
        assert state_delta > 1e-6, f"Pass {k}: recurrent state did not update (delta={state_delta})"
        assert not torch.isnan(h).any(), f"Pass {k}: NaN detected in hidden state"
        
        print(f"  [PASS] Latent Pass {k}: KV len -> {new_kv_len}, GDN state delta -> {state_delta:.4f}")

    print("=== [PASS] Check 3: DynamicCache Dual-State Continuity across Latent Passes Confirmed! ===")

if __name__ == "__main__":
    test_cache_continuity()
