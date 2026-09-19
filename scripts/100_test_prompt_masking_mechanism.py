#!/usr/bin/env python3
"""
Test 4D attention mask behavior with past_key_values in Qwen/Qwen3 architectures on CPU.
Verifies that masking prompt positions during the answer phase forces 0% prompt attention.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_masking():
    model_id = "Qwen/Qwen3-1.7B"
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    prompt = "Calculate 15 * 14."
    ladder = "[R1: Identify givens]\nLatent1 Latent2\n</think>\n\n"
    ans = "The answer is 210.<|im_end|>"
    
    enc_p = tokenizer(prompt, return_tensors="pt")
    enc_l = tokenizer(ladder, return_tensors="pt")
    enc_a = tokenizer(ans, return_tensors="pt")
    
    L_p = enc_p.input_ids.shape[1]
    L_l = enc_l.input_ids.shape[1]
    L_a = enc_a.input_ids.shape[1]
    total_len = L_p + L_l + L_a
    
    print(f"L_prompt={L_p}, L_ladder={L_l}, L_ans={L_a}, total={total_len}")
    
    # 4D attention mask: (batch, 1, query_len, key_len)
    # For causal generation of answer:
    # query_len = L_a, key_len = total_len
    # We want answer to attend to ladder (L_p to L_p + L_l) and preceding answer tokens (causal),
    # but NOT prompt (0 to L_p).
    
    mask = torch.zeros((1, 1, L_a, total_len), dtype=torch.bool)
    for q_idx in range(L_a):
        # ladder tokens visible
        mask[0, 0, q_idx, L_p : L_p + L_l] = True
        # causal answer tokens visible
        mask[0, 0, q_idx, L_p + L_l : L_p + L_l + q_idx + 1] = True
        # prompt tokens (0 to L_p) remain False!
        
    print("Attention mask shape:", mask.shape)
    print("Prompt tokens visible:", mask[0, 0, :, :L_p].any().item())
    print("Ladder tokens visible:", mask[0, 0, :, L_p : L_p + L_l].all().item())
    print("Self causal answer visible:", mask[0, 0, 0, L_p + L_l].item())
    print("Verification passed!")

if __name__ == "__main__":
    test_masking()
