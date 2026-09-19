#!/usr/bin/env python3
"""
01_verify_model.py
Verifies baseline loading, memory footprint, and native <think> generation 
for deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B on GPU 0.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    print(f"=== EXP-02: Verifying Model {MODEL_ID} on {DEVICE} ===")
    
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )
    load_time = time.time() - start_time
    
    # 1. Inspect Architecture
    config = model.config
    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    
    # Measure memory
    mem_allocated = torch.cuda.memory_allocated(DEVICE) / (1024 ** 3)
    mem_reserved = torch.cuda.memory_reserved(DEVICE) / (1024 ** 3)
    
    print(f"Model loaded in: {load_time:.2f}s")
    print(f"Hidden Dimension (d_model): {hidden_dim}")
    print(f"Number of Layers: {num_layers}")
    print(f"Vocab Size: {vocab_size}")
    print(f"VRAM Allocated: {mem_allocated:.2f} GB | Reserved: {mem_reserved:.2f} GB")
    
    # Check embedding norm vs expected RMSNorm
    embed_weights = model.get_input_embeddings().weight.detach()
    avg_embed_norm = torch.norm(embed_weights, dim=-1).mean().item()
    print(f"Average Token Embedding L2 Norm: {avg_embed_norm:.4f}")
    print(f"Theoretical RMSNorm scale (sqrt(d_model)): {hidden_dim ** 0.5:.4f}")
    
    # 2. Test Baseline Reasoning Generation
    test_prompt = (
        "A farmer has 15 sheep. All but 8 die. How many sheep are left? "
        "Think step by step and give your final answer."
    )
    
    messages = [
        {"role": "user", "content": test_prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    
    print("\n--- Running Baseline Generation ---")
    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_time = time.time() - gen_start
    
    # Decode full generation
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    full_response = tokenizer.decode(new_tokens, skip_special_tokens=False)
    
    tokens_generated = len(new_tokens)
    tps = tokens_generated / gen_time if gen_time > 0 else 0
    
    print(f"\nResponse Time: {gen_time:.2f}s ({tps:.1f} tokens/sec)")
    print(f"Total Tokens Generated: {tokens_generated}")
    print("\nGenerated Output:")
    print(full_response)
    
    # Verify presence of <think> and </think>
    has_think_start = "<think>" in full_response
    has_think_end = "</think>" in full_response
    print(f"\n[Verification] Contains <think>: {has_think_start} | Contains </think>: {has_think_end}")
    
    if has_think_start and has_think_end:
        think_segment = full_response.split("<think>")[1].split("</think>")[0]
        answer_segment = full_response.split("</think>")[1]
        think_tokens = len(tokenizer.encode(think_segment))
        answer_tokens = len(tokenizer.encode(answer_segment))
        print(f"Thinking Tokens: {think_tokens} ({(think_tokens/tokens_generated)*100:.1f}%)")
        print(f"Answer Tokens: {answer_tokens} ({(answer_tokens/tokens_generated)*100:.1f}%)")

if __name__ == "__main__":
    main()
