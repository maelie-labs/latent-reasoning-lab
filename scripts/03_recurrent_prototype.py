#!/usr/bin/env python3
"""
03_recurrent_prototype.py
Implements and profiles the core Recurrent Forward Pass:
1. Prefill prompt (stops right after <think>\n)
2. Runs K continuous latent passes without generating tokens
3. Taps intermediate latents asynchronously to a background queue
4. Observes entropy & top-3 decoded tokens in real time
5. Transitions to discrete token generation for final answer
"""

import time
import queue
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def async_thought_observer(lm_head, tokenizer, thought_queue, stop_event, device):
    """Auxiliary observer: decodes latent telemetry snapshots without blocking the main model."""
    obs_stream = torch.cuda.Stream(device=device)
    while not stop_event.is_set() or not thought_queue.empty():
        try:
            step_idx, latent_snapshot = thought_queue.get(timeout=0.05)
        except queue.Empty:
            continue
            
        with torch.cuda.stream(obs_stream):
            with torch.no_grad():
                # Project latent snapshot through unembedding head
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
    print(f"=== EXP-04: Testing Continuous Latent Recurrence & Async Telemetry ===")
    
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
    # The formatted prompt already ends with <｜Assistant｜><think>\n
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids
    print(f"Prompt: {test_prompt}")
    print(f"Prompt Tokens: {input_ids.shape[1]}")
    
    # ------------------------------------------------------------------
    # 1. Start Async Observer Worker
    # ------------------------------------------------------------------
    thought_queue = queue.Queue()
    stop_event = threading.Event()
    observer_thread = threading.Thread(
        target=async_thought_observer,
        args=(model.lm_head, tokenizer, thought_queue, stop_event, DEVICE)
    )
    observer_thread.start()
    
    # ------------------------------------------------------------------
    # 2. Main Fast Recurrent Engine
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    total_start = time.time()
    
    with torch.no_grad():
        # Prefill prompt
        prefill_start = time.time()
        out = model(input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start
        print(f"Prefill Time: {prefill_time*1000:.1f} ms")
        
        # Recurrent Thinking Loop (K = 8 rounds)
        K_ROUNDS = 8
        recurrent_start = time.time()
        
        hidden_dim = model.config.hidden_size
        scale_factor = 1.0  # In Phase 3 fine-tuning, this will be learned or calibrated
        
        for step in range(K_ROUNDS):
            # Dispatch non-blocking snapshot to async telemetry
            thought_queue.put((step + 1, curr_latent.detach().clone()))
            
            # Recurrent step: feedback latent state as inputs_embeds
            step_out = model(
                inputs_embeds=curr_latent * scale_factor,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]
            
        torch.cuda.synchronize()
        recurrent_time = time.time() - recurrent_start
        print(f"\nCompleted {K_ROUNDS} Latent Recurrent Passes in: {recurrent_time*1000:.2f} ms ({recurrent_time/K_ROUNDS*1000:.2f} ms/pass)")
        
        # --------------------------------------------------------------
        # 3. Transition to Answer Generation
        # --------------------------------------------------------------
        # Project last latent to first answer token
        next_token = torch.argmax(model.lm_head(curr_latent)[0, -1]).unsqueeze(0)
        answer_tokens = [next_token.item()]
        
        decode_start = time.time()
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
            
        torch.cuda.synchronize()
        decode_time = time.time() - decode_start
        total_time = time.time() - total_start
        
    stop_event.set()
    observer_thread.join()
    
    print("\n--- Summary Performance ---")
    print(f"Total Wall-Clock Time: {total_time:.3f}s (vs 6.78s for discrete CoT)")
    print(f"Recurrent Thinking Time: {recurrent_time*1000:.1f}ms (vs ~2,300ms for discrete CoT)")
    print(f"Answer Decoding Time: {decode_time:.3f}s")
    print("\nGenerated Output:")
    print(tokenizer.decode(answer_tokens))

if __name__ == "__main__":
    main()
