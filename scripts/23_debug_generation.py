import sys
import faulthandler
faulthandler.enable()

import torch
import transformers.modeling_utils
# Disable aggressive warmup allocation that tries to allocate 30GB on top of loaded model
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:1"
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
model.eval()

q = "What is 2 + 2?"
messages = [{"role": "user", "content": q}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("Starting generation...", flush=True)
try:
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    print("Generation successful!", flush=True)
    print("Output:", tokenizer.decode(out[0]), flush=True)
except Exception as e:
    import traceback
    print("Exception during generation:", e, flush=True)
    traceback.print_exc()
