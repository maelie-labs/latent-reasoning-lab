#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

messages = [{"role": "user", "content": "Hello, how many sheep are left?"}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Formatted Prompt:\n", repr(formatted))
print("\nContains <think>? ", "<think>" in formatted)
