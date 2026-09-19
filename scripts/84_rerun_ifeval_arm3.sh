#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "=========================================================================="
echo "RERUNNING ARM 3 SURGICAL NEUTRALITY (IFEVAL) WITH ROPE PREFILL FIX"
echo "Target Device: GPU 0 (NVIDIA GeForce RTX 4080 16GB)"
echo "Prompts: 541 | Batch Size: 8 | Max New Tokens: 2048 | Sampling: temp 0.7, pp 1.5"
echo "Start Time: $(date -u)"
echo "=========================================================================="

echo ""
echo ">>> Step 1/2: Rerunning Arm 3 (K=6) IFEval on GPU 0..."
python scripts/73_eval_ifeval_arms.py \
  --arm arm3 \
  --k_val 6 \
  --lora_path checkpoints/lora_arm3_qwen_qwen3-1.7b_k6_selfdistill \
  --device cuda:0 \
  --batch_size 8 \
  --output_file data/surgical_neutrality_ifeval_arm3_k6_qwen3_1.7b.json

echo ""
echo ">>> Step 2/2: Rerunning Arm 3 (K=32) IFEval on GPU 0..."
python scripts/73_eval_ifeval_arms.py \
  --arm arm3 \
  --k_val 32 \
  --lora_path checkpoints/lora_arm3_qwen_qwen3-1.7b_k32_selfdistill \
  --device cuda:0 \
  --batch_size 8 \
  --output_file data/surgical_neutrality_ifeval_arm3_k32_qwen3_1.7b.json

echo ""
echo "=========================================================================="
echo "ARM 3 IFEVAL RERUNS COMPLETE WITH ROPE PREFILL FIX!"
echo "End Time: $(date -u)"
echo "=========================================================================="
