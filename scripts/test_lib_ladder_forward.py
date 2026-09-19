#!/usr/bin/env python3
"""
Test Latent Information Bottleneck (LIB) Ladder Forward + Backward Pass on GPU 1
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

def test():
    device = "cuda:1"
    model_id = "Qwen/Qwen3-1.7B"
    print(f"Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    model = get_peft_model(base_model, peft_config)
    model.train()
    
    prompt = "Question: If a box has 5 apples and you add 7, how many apples are there in total?<|im_start|>assistant\n<think>\n"
    CANONICAL_HEADERS = [
        "[R1: Identify givens, constraints, and target variable]\n",
        "[R2: Compute intermediate operations and verify relations]\n",
        "[R3: Execute final deduction and verify constraints]\n"
    ]
    ans_text = "5 + 7 = 12 apples. The total is \\boxed{12}.<|im_end|>"
    
    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    L_prompt = enc_prompt.input_ids.shape[1]
    
    # 1. Prefill Prompt
    out = model(input_ids=enc_prompt.input_ids, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    curr_latent = out.hidden_states[-1][:, -1:, :]
    curr_seq_len = L_prompt
    
    backbone = model.base_model.model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "model") else model
    scale_factor = 0.011440
    k_per_bundle = 4
    
    # 2. Interleaved Register-Bundle Loop
    for r in range(3):
        hdr_text = CANONICAL_HEADERS[r]
        enc_hdr = tokenizer(hdr_text, return_tensors="pt", add_special_tokens=False).to(device)
        hdr_len = enc_hdr.input_ids.shape[1]
        hdr_pos = torch.arange(curr_seq_len, curr_seq_len + hdr_len, device=device).unsqueeze(0)
        hdr_out = model(
            input_ids=enc_hdr.input_ids, position_ids=hdr_pos, past_key_values=past_kv,
            use_cache=True, output_hidden_states=True
        )
        past_kv = hdr_out.past_key_values
        curr_latent = hdr_out.hidden_states[-1][:, -1:, :]
        curr_seq_len += hdr_len
        
        for k in range(k_per_bundle):
            step_pos = torch.tensor([[curr_seq_len]], device=device, dtype=torch.long)
            scaled = curr_latent * scale_factor
            step_out = backbone(
                inputs_embeds=scaled, position_ids=step_pos, past_key_values=past_kv,
                use_cache=True
            )
            past_kv = step_out.past_key_values
            curr_latent = step_out.last_hidden_state[:, -1:, :] if hasattr(step_out, "last_hidden_state") else step_out.hidden_states[-1][:, -1:, :]
            curr_seq_len += 1
            
    # 3. Transition with Bottleneck Mask (Masking out Prompt)
    trans_text = "\n</think>\n\n"
    enc_trans = tokenizer(trans_text, return_tensors="pt", add_special_tokens=False).to(device)
    trans_len = enc_trans.input_ids.shape[1]
    trans_pos = torch.arange(curr_seq_len, curr_seq_len + trans_len, device=device).unsqueeze(0)
    
    trans_mask = torch.zeros((1, 1, trans_len, curr_seq_len + trans_len), dtype=torch.bool, device=device)
    for q in range(trans_len):
        trans_mask[0, 0, q, L_prompt : curr_seq_len] = True
        trans_mask[0, 0, q, curr_seq_len : curr_seq_len + q + 1] = True
        # 0 : L_prompt remains False!
        
    trans_out = model(
        input_ids=enc_trans.input_ids, position_ids=trans_pos, past_key_values=past_kv,
        attention_mask=trans_mask, use_cache=True, output_hidden_states=True
    )
    past_kv = trans_out.past_key_values
    curr_seq_len += trans_len
    
    # 4. Answer Phase with Bottleneck Mask
    enc_ans = tokenizer(ans_text, return_tensors="pt", add_special_tokens=False).to(device)
    ans_len = enc_ans.input_ids.shape[1]
    ans_pos = torch.arange(curr_seq_len, curr_seq_len + ans_len, device=device).unsqueeze(0)
    
    ans_mask = torch.zeros((1, 1, ans_len, curr_seq_len + ans_len), dtype=torch.bool, device=device)
    for q in range(ans_len):
        ans_mask[0, 0, q, L_prompt : curr_seq_len] = True
        ans_mask[0, 0, q, curr_seq_len : curr_seq_len + q + 1] = True
        # 0 : L_prompt remains False!
        
    ans_out = model(
        input_ids=enc_ans.input_ids, position_ids=ans_pos, past_key_values=past_kv,
        attention_mask=ans_mask, use_cache=True
    )
    
    full_logits = torch.cat([trans_out.logits[:, -1:, :], ans_out.logits[:, :-1, :]], dim=1)
    vocab_size = full_logits.size(-1)
    ans_loss = F.cross_entropy(full_logits.reshape(-1, vocab_size), enc_ans.input_ids.reshape(-1))
    
    print(f"Calculated ans_loss = {ans_loss.item():.4f}")
    ans_loss.backward()
    print("Backward pass SUCCESS! Gradient computed through latent unrolling + bottleneck!")
    
    # Check max VRAM
    vram = torch.cuda.max_memory_allocated(device=device) / (1024**3)
    print(f"Peak VRAM: {vram:.2f} GB")

if __name__ == "__main__":
    test()
