#!/bin/bash
set -e
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1


echo "=========================================================="
echo "PHASE 1 TRAINING: Qwen3-1.7B ARMS ON RTX PRO 4500 (cuda:1)"
echo "=========================================================="

echo ""
echo "--- [1/3] Training Arm 1b (Trained Direct / No-CoT, K=0) ---"
.venv/bin/python scripts/41_train_qwen3_arms.py --model_id Qwen/Qwen3-1.7B --arm arm1b --k_tokens 0 --device cuda:1 --steps 500

echo ""
echo "--- [2/3] Training Arm 2b (Pause-Token Control, K=6) ---"
.venv/bin/python scripts/41_train_qwen3_arms.py --model_id Qwen/Qwen3-1.7B --arm arm2b --k_tokens 6 --device cuda:1 --steps 500

echo ""
echo "--- [3/3] Training Arm 3 (Continuous Latent Recurrence, K=6) ---"
.venv/bin/python scripts/41_train_qwen3_arms.py --model_id Qwen/Qwen3-1.7B --arm arm3 --k_tokens 6 --device cuda:1 --steps 500

echo ""
echo "=========================================================="
echo "ALL THREE ARMS SUCCESSFULLY TRAINED AND CHECKPOINTED!"
echo "=========================================================="
