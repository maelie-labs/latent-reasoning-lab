#!/usr/bin/env python3
"""
scripts/101_test_bottleneck_forward.py
Verify that 4D attention mask with past_key_values works seamlessly in Qwen3 forward + backward passes.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def test():
    device = "cpu"
    model_id = "Qwen/Qwen3-1.7B"
    print("Loading model and tokenizer on CPU (eval mode)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.train()
    
    prompt = "Question: Find 12 * 15.<|im_start|>assistant\n<think>\n"
    ladder = "[R1: Identify givens]\n[R2: Compute 12*15]\n[R3: Final answer]\n\n</think>\n\n"
    ans = "12 * 15 = 180.<|im_end|>"
    
    enc_p = tokenizer(prompt, return_tensors="pt")
    enc_l = tokenizer(ladder, return_tensors="pt")
    enc_a = tokenizer(ans, return_tensors="pt")
    
    L_p = enc_p.input_ids.shape[1]
    L_l = enc_l.input_ids.shape[1]
    L_a = enc_a.input_ids.shape[1]
    total_len = L_p + L_l + L_a
    
    # 1. Prefill prompt
    out_p = model(input_ids=enc_p.input_ids, use_cache=True)
    past_kv = out_p.past_key_values
    
    # 2. Prefill ladder
    pos_l = torch.arange(L_p, L_p + L_l).unsqueeze(0)
    out_l = model(input_ids=enc_l.input_ids, position_ids=pos_l, past_key_values=past_kv, use_cache=True)
    past_kv = out_l.past_key_values
    
    # 3. Answer phase with Bottleneck Mask: (1, 1, L_a, total_len)
    # 0 for prompt, 1 for ladder and causal answer
    # In HuggingFace, if attention_mask is float/bfloat16: 0.0 means attend, min_dtype means do not attend.
    # If boolean: True means attend, False means do not attend.
    mask = torch.zeros((1, 1, L_a, total_len), dtype=torch.bool)
    for q in range(L_a):
        mask[0, 0, q, L_p : L_p + L_l] = True
        mask[0, 0, q, L_p + L_l : L_p + L_l + q + 1] = True
        # prompt 0:L_p remains False
        
    pos_a = torch.arange(L_p + L_l, total_len).unsqueeze(0)
    
    # Check forward pass
    try:
        out_a = model(
            input_ids=enc_a.input_ids,
            position_ids=pos_a,
            past_key_values=past_kv,
            attention_mask=mask,
            use_cache=True
        )
        print("Forward pass successful! Logits shape:", out_a.logits.shape)
        
        # Check backward pass
        loss = out_a.logits.sum()
        loss.backward()
        print("Backward pass successful! Gradient norm computed successfully.")
    except Exception as e:
        print("Encountered error:", e)

if __name__ == "__main__":
    test()
