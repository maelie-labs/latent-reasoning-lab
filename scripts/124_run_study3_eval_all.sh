#!/bin/bash
# scripts/124_run_study3_eval_all.sh
# Master Evaluation & Bootstrap Pipeline for Study 3 on 4B Models
# Evaluates N=1,000 queries across 4 seeds for each of the 4 checkpoints

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=".venv/bin/python3"
export CUDA_VISIBLE_DEVICES=1
DEVICE="cuda:0"
BATCH_SIZE=16

echo "========================================================"
echo "STARTING STUDY 3 BENCHMARK EVALUATION (4B REGIME)"
echo "Device: $DEVICE | Batch Size: $BATCH_SIZE | Seeds: 42, 123, 456, 789"
echo "========================================================"

# 1. Evaluate Qwen3-4B Headers-Only Control
echo ">>> [1/4] Evaluating Qwen3-4B Headers-Only Control..."
$PYTHON scripts/122_eval_study3_4b.py \
    --model_id "Qwen/Qwen3-4B" \
    --mode "headers_only" \
    --checkpoint "checkpoints/lora_headers_only_qwen3_4b/best_checkpoint" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"

# 2. Evaluate Qwen3-4B Telegraphic CoT
echo ">>> [2/4] Evaluating Qwen3-4B Telegraphic CoT..."
$PYTHON scripts/122_eval_study3_4b.py \
    --model_id "Qwen/Qwen3-4B" \
    --mode "telegraphic_cot" \
    --checkpoint "checkpoints/lora_telegraphic_cot_qwen3_4b/best_checkpoint" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"

# Run Paired Bootstrap for Qwen3-4B
echo ">>> Running Paired Bootstrap Analysis for Qwen3-4B..."
$PYTHON scripts/123_paired_bootstrap_study3_4b.py \
    --candidate "data/streaming_study3_telegraphic_cot_qwen3_4b.jsonl" \
    --control "data/streaming_study3_headers_only_qwen3_4b.jsonl" \
    --model_tag "qwen3_4b"

# 3. Evaluate Qwen3.5-4B Headers-Only Control
echo ">>> [3/4] Evaluating Qwen3.5-4B Headers-Only Control..."
$PYTHON scripts/122_eval_study3_4b.py \
    --model_id "Qwen/Qwen3.5-4B" \
    --mode "headers_only" \
    --checkpoint "checkpoints/lora_headers_only_qwen3_5_4b/best_checkpoint" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"

# 4. Evaluate Qwen3.5-4B Telegraphic CoT
echo ">>> [4/4] Evaluating Qwen3.5-4B Telegraphic CoT..."
$PYTHON scripts/122_eval_study3_4b.py \
    --model_id "Qwen/Qwen3.5-4B" \
    --mode "telegraphic_cot" \
    --checkpoint "checkpoints/lora_telegraphic_cot_qwen3_5_4b/best_checkpoint" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"

# Run Paired Bootstrap for Qwen3.5-4B
echo ">>> Running Paired Bootstrap Analysis for Qwen3.5-4B..."
$PYTHON scripts/123_paired_bootstrap_study3_4b.py \
    --candidate "data/streaming_study3_telegraphic_cot_qwen3_5_4b.jsonl" \
    --control "data/streaming_study3_headers_only_qwen3_5_4b.jsonl" \
    --model_tag "qwen3_5_4b"

echo "========================================================"
echo "STUDY 3 EVALUATION & BOOTSTRAP PIPELINE COMPLETE!"
echo "========================================================"
