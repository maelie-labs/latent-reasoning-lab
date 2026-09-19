#!/usr/bin/env python3
"""
03b_recurrent_scaled.py
Tests the norm-rescaling hypothesis:
Adjusts the recurrent feedback scale: h_feed = h_t * (avg_embed_norm / sqrt(d_model))
to prevent representation drift in zero-shot recurrence before fine-tuning.
"""

import time
import queue
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def async_thought_observer(lm_head, tokenizer, thought_queue, stop_event, device):
    obs_stream = torch.cuda.Stream(device=device)
    while not stop_event.is_set() or not thought_queue.empty():
        try:
            step_idx, latent_snapshot = thought_queue.get(timeout=0.05)
        except queue.Empty:
            continue
            
        with torch.cuda.stream(obs_stream):
            with torch.no_grad():
                logits = lm_head(latent_snapshot)
                probs = torch.softmax(logits[0, -1], dim=-1)
                topk = torch.topk(probs, k=3)
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
                tokens = [tokenizer.decode([idx.item()]).replace("\n", "\\n") for idx in topk.indices]
                confidences = [round(p.item(), 3) for p in topk.values]
                streams = list(zip(tokens, confidences))
                
        print(f"  [Async Telemetry] Round {step_idx:02d} | Entropy: {entropy:.2f} | Top Streams: {streams}")
        thought_queue.task_done()

def main():
    print(f"=== EXP-04b: Testing Normalized Latent Recurrence ===")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )
    model.eval()
    
    test_prompt = "A farmer has 15 sheep. All but 8 die. How many sheep are left?"
    messages = [{"role": "user", "content": test_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids
    
    # Calculate exact scale factor
    embed_weights = model.get_input_embeddings().weight.detach()
    avg_embed_norm = torch.norm(embed_weights, dim=-1).mean().item()
    hidden_dim = model.config.hidden_size
    theoretical_rmsnorm = hidden_dim ** 0.5
    scale_factor = avg_embed_norm / theoretical_rmsnorm
    print(f"Embedding Norm: {avg_embed_norm:.4f} | RMSNorm: {theoretical_rmsnorm:.4f} | Scale Factor: {scale_factor:.5f}")
    
    thought_queue = queue.Queue()
    stop_event = threading.Event()
    observer_thread = threading.Thread(
        target=async_thought_observer,
        args=(model.lm_head, tokenizer, thought_queue, stop_event, DEVICE)
    )
    observer_thread.start()
    
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        
        K_ROUNDS = 6
        rec_start = time.time()
        for step in range(K_ROUNDS):
            thought_queue.put((step + 1, curr_latent.detach().clone()))
            
            # Scaled recurrent feedback
            scaled_latent = curr_latent * scale_factor
            step_out = model(
                inputs_embeds=scaled_latent,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]
            
        rec_time = time.time() - rec_start
        print(f"\n{K_ROUNDS} Scaled Passes in {rec_time*1000:.2f} ms")
        
        # Generation
        next_token = torch.argmax(model.lm_head(curr_latent)[0, -1]).unsqueeze(0)
        answer_tokens = [next_token.item()]
        
        for _ in range(64):
            step_out = model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_kv,
                use_cache=True
            )
            past_kv = step_out.past_key_values
            next_token = torch.argmax(step_out.logits[0, -1]).unsqueeze(0)
            if next_token.item() == tokenizer.eos_token_id:
                break
            answer_tokens.append(next_token.item())
            
    stop_event.set()
    observer_thread.join()
    
    print("\nGenerated Output with Scaled Recurrence:")
    print(tokenizer.decode(answer_tokens))

if __name__ == "__main__":
    main()
