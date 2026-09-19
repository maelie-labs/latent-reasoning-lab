#!/usr/bin/env bash
set -euo pipefail

# =================================================================
# scripts/76_run_all_trained_arms.sh
# Evaluates all pre-trained arms on GPU 1 (RTX PRO 4500 32GB)
# Arms:
#   1. Arm 1b (Direct SFT Control, K=0)
#   2. Arm 2b (Pause Tokens Control, K=6)
#   3. Arm 3  (Continuous Latent Recurrence, K=6)
#   4. Arm 2b (Pause Tokens Control, K=32)
# =================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
source .venv/bin/activate

if [ -f "${ROOT_DIR}/.env" ]; then
    set -a
    source "${ROOT_DIR}/.env"
    set +a
fi

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

echo "================================================================="
echo "=== Phase 3: Evaluating All Trained Qwen3-1.7B Arms on GPU 1 ==="
echo "Target Device: NVIDIA RTX PRO 4500 32GB (CUDA_VISIBLE_DEVICES=1)"
echo "Sampling Spec: temp 0.7 / top_p 0.80 / top_k 20 / pp 1.5 / cap 8192"
echo "Seeds: 42, 123, 456, 789 (250 problems x 4 = 1,000 queries/arm)"
echo "================================================================="

# Arm 1b: Direct SFT Control (K=0)
echo ""
echo ">>> [1/4] Running Arm 1b (Direct SFT Control, K=0)..."
python scripts/55_batched_eval_arms.py \
    --arm arm1b \
    --batch_size 16 \
    --device cuda:0 \
    --max_ans_tokens 8192

# Arm 2b: Pause Tokens Control (K=6)
echo ""
echo ">>> [2/4] Running Arm 2b (Pause Tokens Control, K=6)..."
python scripts/55_batched_eval_arms.py \
    --arm arm2b \
    --k_tokens 6 \
    --batch_size 16 \
    --device cuda:0 \
    --max_ans_tokens 8192

# Arm 3: Continuous Latent Recurrence (K=6)
echo ""
echo ">>> [3/4] Running Arm 3 (Continuous Latent Recurrence, K=6)..."
python scripts/55_batched_eval_arms.py \
    --arm arm3 \
    --k_tokens 6 \
    --batch_size 16 \
    --device cuda:0 \
    --max_ans_tokens 8192

# Arm 2b: Pause Tokens Control (K=32)
echo ""
echo ">>> [4/4] Running Arm 2b (Pause Tokens Control, K=32)..."
python scripts/55_batched_eval_arms.py \
    --arm arm2b \
    --k_tokens 32 \
    --batch_size 16 \
    --device cuda:0 \
    --max_ans_tokens 8192

echo ""
echo "================================================================="
echo "=== All 4 Trained Arms Completed Successfully ==="
echo "Generating Paired Bootstrap Analyses (B=10,000)..."
echo "================================================================="

# Comparison 1: Delta_1 = Arm 3 (K=6) - Arm 1b (Direct SFT)
python scripts/analysis/paired_bootstrap.py \
    --file_a data/streaming_arm3_k6_qwen_qwen3-1.7b.jsonl \
    --file_b data/streaming_arm1b_qwen_qwen3-1.7b.jsonl \
    --name_a "Arm 3 (Latents K=6)" \
    --name_b "Arm 1b (Direct SFT)" \
    --output_json data/paired_bootstrap_arm3_vs_arm1b.json

# Comparison 2: Delta_2 = Arm 3 (K=6) - Arm 2b (Pause K=6)
python scripts/analysis/paired_bootstrap.py \
    --file_a data/streaming_arm3_k6_qwen_qwen3-1.7b.jsonl \
    --file_b data/streaming_arm2b_k6_qwen_qwen3-1.7b.jsonl \
    --name_a "Arm 3 (Latents K=6)" \
    --name_b "Arm 2b (Pause K=6)" \
    --output_json data/paired_bootstrap_arm3_vs_arm2b_k6.json

# Comparison 3: Control Check = Arm 1b vs Arm 1 (Base Direct)
python scripts/analysis/paired_bootstrap.py \
    --file_a data/streaming_arm1b_qwen_qwen3-1.7b.jsonl \
    --file_b data/streaming_arm1_direct_qwen_qwen3-1.7b.jsonl \
    --name_a "Arm 1b (Direct SFT)" \
    --name_b "Arm 1 (Base Direct)" \
    --output_json data/paired_bootstrap_arm1b_vs_arm1.json

echo "Evaluation pipeline finished successfully."
