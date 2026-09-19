#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "================================================================="
echo "=== Arm 0: IFEval Surgical Neutrality Suite for Trained Adapters ==="
echo "Target Model: Qwen/Qwen3-1.7B"
echo "Device: cuda:0 (NVIDIA GeForce RTX 4080 16GB)"
echo "Benchmark: google/IFEval (541 prompts, strict & loose scoring)"
echo "================================================================="

echo ""
echo ">>> [1/3] Evaluating Arm 1b (Direct SFT, No-CoT Control)..."
.venv/bin/python scripts/73_eval_ifeval_arms.py \
  --arm arm1b \
  --lora_path checkpoints/lora_arm1b_qwen_qwen3-1.7b_k0_selfdistill \
  --device cuda:0 \
  --batch_size 8 \
  --output_file data/surgical_neutrality_ifeval_arm1b_qwen3_1.7b.json

echo ""
echo ">>> [2/3] Evaluating Arm 2b (Trained Pause Tokens, K=6)..."
.venv/bin/python scripts/73_eval_ifeval_arms.py \
  --arm arm2b \
  --k_val 6 \
  --lora_path checkpoints/lora_arm2b_qwen_qwen3-1.7b_k6_selfdistill \
  --device cuda:0 \
  --batch_size 8 \
  --output_file data/surgical_neutrality_ifeval_arm2b_k6_qwen3_1.7b.json

echo ""
echo ">>> [3/4] Evaluating Arm 2b (Trained Pause Tokens, K=32)..."
.venv/bin/python scripts/73_eval_ifeval_arms.py \
  --arm arm2b \
  --k_val 32 \
  --lora_path checkpoints/lora_arm2b_qwen_qwen3-1.7b_k32_selfdistill \
  --device cuda:0 \
  --batch_size 8 \
  --output_file data/surgical_neutrality_ifeval_arm2b_k32_qwen3_1.7b.json

echo ""
echo ">>> [4/4] Evaluating Arm 3 (Continuous Latent Recurrence, K=6)..."
.venv/bin/python scripts/73_eval_ifeval_arms.py \
  --arm arm3 \
  --k_val 6 \
  --lora_path checkpoints/lora_arm3_qwen_qwen3-1.7b_k6_selfdistill \
  --device cuda:0 \
  --batch_size 8 \
  --output_file data/surgical_neutrality_ifeval_arm3_k6_qwen3_1.7b.json

echo ""
echo "================================================================="
echo "=== Adapter IFEval Evaluations Complete! ==="
echo "Results saved in data/surgical_neutrality_ifeval_*.json"
echo "================================================================="
