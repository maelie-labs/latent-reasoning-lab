#!/usr/bin/env bash
set -euo pipefail


echo "================================================================="
echo "=== Phase 3a: Training Arm 3 Recurrent Adapters (Qwen/Qwen3-1.7B) ==="
echo "Target Device: cuda:1 (NVIDIA RTX PRO 4500 32GB)"
echo "Curriculum: Stage 1 (Partial Latents) -> Stage 2 (Full Latents)"
echo "Traces: 2,070 curated train traces, 100 held-out dev traces"
echo "================================================================="

echo ""
echo ">>> [1/2] Launching Arm 3 (K=6) Continuous Latent Recurrence Training..."
.venv/bin/python scripts/41_train_qwen3_arms.py \
  --arm arm3 \
  --model_id Qwen/Qwen3-1.7B \
  --k_tokens 6 \
  --device cuda:1

echo ""
echo ">>> [1/2] Arm 3 (K=6) Training COMPLETE!"
echo ""

# echo ">>> [2/2] Launching Arm 3 (K=32) Continuous Latent Recurrence Training..."
# .venv/bin/python scripts/41_train_qwen3_arms.py \
#   --arm arm3 \
#   --model_id Qwen/Qwen3-1.7B \
#   --k_tokens 32 \
#   --device cuda:1
exit 0

echo ""
echo "================================================================="
echo "=== Phase 3a Complete: All Arm 3 Adapters Trained (K=6, K=32) ==="
echo "Checkpoints saved in: checkpoints/lora_arm3_qwen_qwen3-1.7b_*"
echo "================================================================="
