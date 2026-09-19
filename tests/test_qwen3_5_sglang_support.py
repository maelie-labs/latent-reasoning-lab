#!/usr/bin/env python3
"""
tests/test_qwen3_5_sglang_support.py
Step 7 Pre-Flight Checks 1 & 2:
1. Verifies Qwen/Qwen3.5-2B chat template native enable_thinking toggle:
   - enable_thinking=True: leaves <think> open for reasoning trace generation (<|im_start|>assistant\n<think>\n)
   - enable_thinking=False: injects empty think block (<|im_start|>assistant\n<think>\n\n</think>\n\n) bypassing reasoning
2. Verifies SGLang 0.5.19 native model registry and hybrid attention overrides for Qwen3_5ForConditionalGeneration.
"""

import sys
import subprocess
from transformers import AutoTokenizer

def test_chat_template_thinking():
    print("Checking Qwen/Qwen3.5-2B chat template...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    messages = [{"role": "user", "content": "What is 2+2?"}]
    
    # 1. Test Thinking Enabled
    prompt_think = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    assert prompt_think.endswith("<|im_start|>assistant\n<think>\n"), f"Expected open <think> tag, got: {prompt_think[-50:]}"
    print("  [PASS] Check 1a: Thinking enabled leaves open <think> block for reasoning.")
    
    # 2. Test Thinking Disabled
    prompt_nothink = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert prompt_nothink.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), f"Expected closed empty <think> block, got: {prompt_nothink[-50:]}"
    print("  [PASS] Check 1b: Thinking disabled pre-closes <think> block (<think>\\n\\n</think>\\n\\n) forcing direct answer mode.")

def test_sglang_qwen3_5_registry():
    print("Checking SGLang 0.5.19 model registry for Qwen3_5ForConditionalGeneration...")
    cmd = [
        "python", "-c",
        "from sglang.srt.models.registry import ModelRegistry; "
        "from sglang.srt.arg_groups.model_overrides.qwen3_5 import _qwen3_5_hybrid_overrides; "
        "from sglang.srt.configs.model_config import is_qwen3_5; "
        "assert 'Qwen3_5ForConditionalGeneration' in ModelRegistry.models, 'Qwen3_5ForConditionalGeneration not in registry'; "
        "assert 'Qwen3_5ForCausalLM' in ModelRegistry.models, 'Qwen3_5ForCausalLM not in registry'; "
        "print('  [PASS] Check 2: SGLang 0.5.19 registry confirmed: Qwen3_5ForConditionalGeneration & _qwen3_5_hybrid_overrides present.')"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise AssertionError("SGLang qwen3_5 registry check failed.")

if __name__ == "__main__":
    test_chat_template_thinking()
    test_sglang_qwen3_5_registry()
    print("=== [PASS] All Step 7 Pre-Flight Checks 1 & 2 Succeeded! ===")
