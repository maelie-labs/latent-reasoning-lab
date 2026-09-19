#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CKPT="checkpoints/full_ft_arm3_v1_1_qwen3_1.7b_k6_lambda1.0"
REPORT_T3="data/gate2_causal_patching_report_full_ft_t3.json"
REPORT_T6="data/gate2_causal_patching_report_full_ft_t6.json"

echo "======================================================================"
echo "EVALUATING GATE 2 CAUSAL STEERING ON FULL-FT QWEN3-1.7B"
echo "Checkpoint: $CKPT"
echo "======================================================================"

echo "[1/2] Running Gate 2 at t=3 (mid-thought)..."
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/88_gate2_causal_patching_v1_1.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --checkpoint "$CKPT" \
    --device "cuda:0" \
    --n_problems 100 \
    --k_steps 6 \
    --batch_size 16 \
    --patch_step 3 \
    --output_file "$REPORT_T3" || true

echo "[2/2] Running Gate 2 at t=6 (hand-off)..."
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/88_gate2_causal_patching_v1_1.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --checkpoint "$CKPT" \
    --device "cuda:0" \
    --n_problems 100 \
    --k_steps 6 \
    --batch_size 16 \
    --patch_step 6 \
    --output_file "$REPORT_T6" || true

echo "======================================================================"
echo "GATE 2 EVALUATION FINISHED."
echo "======================================================================"
