#!/bin/bash
# scripts/121_run_study3_training_4b.sh
# Master Training Pipeline for Study 3 (Telegraphic CoT vs Headers-Only on 4B Models)
# Executes sequentially on dedicated GPU 1 (RTX PRO 4500 32GB)

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=".venv/bin/python3"
DEVICE="cuda:1"

echo "========================================================"
echo "STARTING STUDY 3 TRAINING PIPELINE: 4B MODELS"
echo "Device: $DEVICE | Pure bfloat16"
echo "========================================================"

# 1. Qwen3-4B (Dense) - Headers-Only Control
echo ">>> [1/4] Training Qwen3-4B Headers-Only Control..."
$PYTHON scripts/120_train_telegraphic_4b.py \
    --model_id "Qwen/Qwen3-4B" \
    --mode "headers_only" \
    --device "$DEVICE" \
    --steps 150 \
    --lr 2e-4 \
    --grad_accum_steps 8

# 2. Qwen3-4B (Dense) - Telegraphic CoT
echo ">>> [2/4] Training Qwen3-4B Telegraphic CoT..."
$PYTHON scripts/120_train_telegraphic_4b.py \
    --model_id "Qwen/Qwen3-4B" \
    --mode "telegraphic_cot" \
    --device "$DEVICE" \
    --steps 150 \
    --lr 2e-4 \
    --grad_accum_steps 8

# 3. Qwen3.5-4B (Hybrid GDN) - Headers-Only Control
echo ">>> [3/4] Training Qwen3.5-4B Headers-Only Control..."
$PYTHON scripts/120_train_telegraphic_4b.py \
    --model_id "Qwen/Qwen3.5-4B" \
    --mode "headers_only" \
    --device "$DEVICE" \
    --steps 150 \
    --lr 2e-4 \
    --grad_accum_steps 8

# 4. Qwen3.5-4B (Hybrid GDN) - Telegraphic CoT
echo ">>> [4/4] Training Qwen3.5-4B Telegraphic CoT..."
$PYTHON scripts/120_train_telegraphic_4b.py \
    --model_id "Qwen/Qwen3.5-4B" \
    --mode "telegraphic_cot" \
    --device "$DEVICE" \
    --steps 150 \
    --lr 2e-4 \
    --grad_accum_steps 8

echo "========================================================"
echo "STUDY 3 TRAINING COMPLETE! All 4 checkpoints saved."
echo "========================================================"
