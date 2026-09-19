#!/bin/bash
set -eo pipefail

PROJECT_ROOT="."
cd "${PROJECT_ROOT}"

echo "================================================================================"
echo "LAUNCHING PHASE 2: TELEGRAPHIC COT TRAINING PIPELINE (GPU 1)"
echo "================================================================================"

# 1. Train Arm 1: Pure Telegraphic Discrete CoT
echo -e "\n[1/3] Training Arm 1: Pure Telegraphic Discrete CoT..."
.venv/bin/python studies/telegraphic_cot/scripts/02_train_telegraphic_models.py \
    --mode arm1_telegraphic_cot \
    --device cuda:1 \
    --output_dir studies/telegraphic_cot/checkpoints/lora_arm1_telegraphic_cot_qwen3_1.7b \
    --epochs 3 \
    --lr 2e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --accum_steps 2

# 2. Train Control 3: Matched Token Pause Control
echo -e "\n[2/3] Training Control 3: Matched Token Pause Control..."
.venv/bin/python studies/telegraphic_cot/scripts/02_train_telegraphic_models.py \
    --mode control3_matched_pause \
    --device cuda:1 \
    --output_dir studies/telegraphic_cot/checkpoints/lora_control3_matched_pause_qwen3_1.7b \
    --epochs 3 \
    --lr 2e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --accum_steps 2

# 3. Train Arm 2: Dual-Channel Telegraphic + Continuous Latents
echo -e "\n[3/3] Training Arm 2: Dual-Channel Telegraphic + Continuous Latents..."
.venv/bin/python studies/telegraphic_cot/scripts/02_train_telegraphic_models.py \
    --mode arm2_dual_channel_latents \
    --device cuda:1 \
    --output_dir studies/telegraphic_cot/checkpoints/lora_arm2_dual_channel_latents_qwen3_1.7b \
    --epochs 3 \
    --lr 2e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --accum_steps 2 \
    --k_latents 12 \
    --alpha 0.011440

# 4. Merge Checkpoints for High-Throughput SGLang Serving
echo -e "\n[4/4] Merging Checkpoints into Standalone Safetensors for SGLang Serving..."
.venv/bin/python studies/telegraphic_cot/scripts/merge_checkpoints.py

echo "================================================================================"
echo "PHASE 2 TRAINING PIPELINE COMPLETED SUCCESSFULLY!"
echo "================================================================================"

# 5. Launch High-Throughput Evaluation Pipeline
echo -e "\n[5/5] Launching High-Throughput Evaluation and Paired Bootstrapping Pipeline..."
bash studies/telegraphic_cot/scripts/run_eval_pipeline.sh
