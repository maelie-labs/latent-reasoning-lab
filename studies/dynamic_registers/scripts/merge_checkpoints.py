#!/usr/bin/env python3
"""
studies/dynamic_registers/scripts/merge_checkpoints.py
Merges LoRA adapter weights with base Qwen/Qwen3-1.7B into standalone models.
This enables high-throughput SGLang serving with asynchronous continuous batching.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-1.7B"

MODELS_TO_MERGE = [
    {
        "name": "control1_headers_only",
        "lora_path": "studies/dynamic_registers/checkpoints/lora_control1_headers_only_qwen3_1.7b/best_checkpoint",
        "output_path": "studies/dynamic_registers/checkpoints/merged_control1_headers_only_qwen3_1.7b"
    },
    {
        "name": "arm1_dynamic_registers",
        "lora_path": "studies/dynamic_registers/checkpoints/lora_arm1_dynamic_registers_qwen3_1.7b/best_checkpoint",
        "output_path": "studies/dynamic_registers/checkpoints/merged_arm1_dynamic_registers_qwen3_1.7b"
    },
    {
        "name": "arm2_matched_filler",
        "lora_path": "studies/dynamic_registers/checkpoints/lora_arm2_matched_filler_qwen3_1.7b/best_checkpoint",
        "output_path": "studies/dynamic_registers/checkpoints/merged_arm2_matched_filler_qwen3_1.7b"
    }
]

def merge_model(lora_path: str, output_path: str):
    if os.path.exists(output_path) and (os.path.exists(os.path.join(output_path, "model.safetensors")) or os.path.exists(os.path.join(output_path, "model.safetensors.index.json"))):
        print(f"Merged model already exists at {output_path}. Skipping.")
        return

    print(f"Loading base model {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:1",
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter from {lora_path}...")
    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging weights into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print(f"Successfully saved merged model to {output_path}!\n")

def main():
    for item in MODELS_TO_MERGE:
        print("=" * 60)
        print(f"Merging {item['name']}...")
        print("=" * 60)
        merge_model(item["lora_path"], item["output_path"])

if __name__ == "__main__":
    main()
