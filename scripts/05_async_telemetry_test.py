#!/usr/bin/env python3
"""
05_async_telemetry_test.py
Connects the newly trained Auxiliary Thought-Stream Probe to the live
asynchronous background observer and runs inference on novel questions.
Demonstrates:
1. Fast continuous latent recurrence (<250ms).
2. Non-blocking out-of-band telemetry push.
3. Auxiliary probe decoding multi-hypothesis thought streams with probabilities and entropy.
4. Clean generation of the final answer.
"""

import time
import queue
import threading
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROBE_PATH = "checkpoints/thought_probe.pt"

class ThoughtStreamProbe(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size, bias=False)
        )
        
    def forward(self, h):
        return self.net(h)

def async_thought_observer(probe, tokenizer, thought_queue, stop_event, device):
    obs_stream = torch.cuda.Stream(device=device)
    while not stop_event.is_set() or not thought_queue.empty():
        try:
            step_idx, latent_snapshot = thought_queue.get(timeout=0.05)
        except queue.Empty:
            continue
            
        with torch.cuda.stream(obs_stream):
            with torch.no_grad():
                logits = probe(latent_snapshot)
                probs = torch.softmax(logits[0, -1], dim=-1)
                topk = torch.topk(probs, k=3)
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
                tokens = [tokenizer.decode([idx.item()]).replace("\n", "\\n") for idx in topk.indices]
                confidences = [round(p.item(), 3) for p in topk.values]
                streams = list(zip(tokens, confidences))
                
        print(f"  [Async Telemetry] Round {step_idx:02d} | Entropy: {entropy:.2f} | Hypotheses: {streams}")
        thought_queue.task_done()

def run_test(prompt, model, tokenizer, probe):
    print(f"\n=======================================================")
    print(f"Question: {prompt}")
    print(f"=======================================================")
    
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    
    thought_queue = queue.Queue()
    stop_event = threading.Event()
    observer_thread = threading.Thread(
        target=async_thought_observer,
        args=(probe, tokenizer, thought_queue, stop_event, DEVICE)
    )
    observer_thread.start()
    
    hidden_dim = model.config.hidden_size
    scale_factor = 1.1641 / (hidden_dim ** 0.5)
    
    with torch.no_grad():
        # 1. Prefill
        t0 = time.time()
        out = model(inputs.input_ids, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        curr_latent = out.hidden_states[-1][:, -1:, :]
        
        # 2. Continuous Latent Recurrence Loop (K = 6 rounds)
        K_ROUNDS = 6
        rec_start = time.time()
        for step in range(K_ROUNDS):
            # Non-blocking push of latent state snapshot
            thought_queue.put((step + 1, curr_latent.detach().clone()))
            
            step_out = model(
                inputs_embeds=curr_latent * scale_factor,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = step_out.past_key_values
            curr_latent = step_out.hidden_states[-1][:, -1:, :]
            
        rec_time = time.time() - rec_start
        
        # 3. Answer Generation
        next_token = torch.argmax(model.lm_head(curr_latent)[0, -1]).unsqueeze(0)
        answer_tokens = [next_token.item()]
        
        for _ in range(80):
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
            
        total_time = time.time() - t0
        
    stop_event.set()
    observer_thread.join()
    
    print(f"\nThinking Time: {rec_time*1000:.1f} ms | Total Time: {total_time:.2f}s")
    print("\n--- Final Answer ---")
    print(tokenizer.decode(answer_tokens))

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=DEVICE
    )
    model.eval()
    
    hidden_dim = model.config.hidden_size
    vocab_size = model.config.vocab_size
    
    probe = ThoughtStreamProbe(hidden_dim, vocab_size).to(DEVICE, dtype=torch.bfloat16)
    probe.load_state_dict(torch.load(PROBE_PATH, weights_only=True))
    probe.eval()
    print("Model and Trained Probe Loaded Successfully.")
    
    test_questions = [
        "A baker makes 24 cupcakes. He sells half in the morning and 6 in the afternoon. How many cupcakes are left?",
        "Natalia has 5 packs of stickers with 10 stickers in each pack. She gives 15 stickers to her sister. How many stickers does Natalia have now?"
    ]
    
    for q in test_questions:
        run_test(q, model, tokenizer, probe)

if __name__ == "__main__":
    main()
