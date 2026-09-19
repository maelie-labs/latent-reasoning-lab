#!/usr/bin/env bash
set -euo pipefail


echo "================================================================="
echo "=== Phase 3a: Training Arm 3 (K=32) Recurrent Adapter ==="
echo "Target Model: Qwen/Qwen3-1.7B"
echo "Target Device: GPU 1 (NVIDIA RTX PRO 4500 32GB via CUDA_VISIBLE_DEVICES=1)"
echo "Optimizations: Backbone unroll (zero lm_head overhead), fused=True AdamW"
echo "================================================================="

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/41_train_qwen3_arms.py \
  --arm arm3 \
  --model_id Qwen/Qwen3-1.7B \
  --k_tokens 32 \
  --device cuda:0

echo ""
echo "================================================================="
echo "=== Arm 3 (K=32) Training COMPLETE! ==="
echo "Checkpoint saved to: checkpoints/lora_arm3_qwen_qwen3-1.7b_k32_selfdistill"
echo "================================================================="
