#!/usr/bin/env python3
"""
15_verify_14b.py
Loads deepseek-ai/DeepSeek-R1-Distill-Qwen-14B in 8-bit on cuda:1 (RTX PRO 4500 32GB).
Profiles architecture, measures embedding norm, computes alpha_14B,
and verifies both discrete generation and continuous inputs_embeds forward pass.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DEVICE = "cuda:1"

def main():
    print(f"=== Stage 4: Profiling & Verifying {MODEL_ID} on {DEVICE} in 8-bit ===")
    
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16
    )
    load_time = time.time() - start_time
    
    config = model.config
    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    
    mem_allocated = torch.cuda.memory_allocated(DEVICE) / (1024 ** 3)
    mem_reserved = torch.cuda.memory_reserved(DEVICE) / (1024 ** 3)
    
    print(f"Model loaded in: {load_time:.2f}s")
    print(f"Hidden Dimension (d_model): {hidden_dim}")
    print(f"Number of Layers: {num_layers}")
    print(f"Vocab Size: {vocab_size}")
    print(f"VRAM Allocated: {mem_allocated:.2f} GB | Reserved: {mem_reserved:.2f} GB")
    
    embed_weights = model.get_input_embeddings().weight.detach()
    avg_embed_norm = torch.norm(embed_weights.float(), dim=-1).mean().item()
    rms_scale = hidden_dim ** 0.5
    scale_factor = avg_embed_norm / rms_scale
    
    print(f"Average Token Embedding L2 Norm ||W_E||: {avg_embed_norm:.6f}")
    print(f"Theoretical RMSNorm scale (sqrt(d_model)): {rms_scale:.6f}")
    print(f"Computed Scale Factor alpha_14B: {scale_factor:.6f}")
    
    # 1. Test Baseline Reasoning Generation
    test_prompt = "A farmer has 15 sheep. All but 8 die. How many sheep are left?"
    messages = [{"role": "user", "content": test_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    
    print("\n--- Running Baseline Generation Test ---")
    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_time = time.time() - gen_start
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    full_response = tokenizer.decode(new_tokens, skip_special_tokens=False)
    print(f"Gen Time: {gen_time:.2f}s | Tokens: {len(new_tokens)} | Speed: {len(new_tokens)/gen_time:.1f} tok/s")
    print(f"Response:\n{full_response}")
    print(f"Contains </think>: {'</think>' in full_response}")
    
    # 2. Test Recurrent Step (inputs_embeds forward pass)
    print("\n--- Testing Latent Recurrent Step (inputs_embeds) ---")
    with torch.no_grad():
        out = model(inputs.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        
        # Step with inputs_embeds
        scaled_latent = (curr_latent * scale_factor).to(torch.bfloat16)
        step_out = model(
            inputs_embeds=scaled_latent,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True
        )
        print("Recurrent step succeeded! Next latent shape:", step_out.hidden_states[-1].shape)

if __name__ == "__main__":
    main()
