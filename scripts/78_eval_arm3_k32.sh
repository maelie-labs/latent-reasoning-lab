#!/usr/bin/env bash
set -euo pipefail

# =================================================================
# scripts/78_eval_arm3_k32.sh
# Evaluates Arm 3 (Continuous Latent Recurrence, K=32) on GPU 1
# Across all 4 deterministic seeds (N=1,000 queries total)
# Sampling Spec: temp 0.7 / top_p 0.80 / top_k 20 / pp 1.5 / cap 8192
# Followed by updated paired hierarchical bootstrapping (B=10,000).
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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "================================================================="
echo "=== Step 1/2: Re-evaluating Arm 3 (Latents K=6) with RoPE Fix ==="
echo "Target Device: NVIDIA RTX PRO 4500 32GB (CUDA_VISIBLE_DEVICES=1)"
echo "Sampling Spec: temp 0.7 / top_p 0.80 / top_k 20 / pp 1.5 / cap 8192"
echo "Seeds: 42, 123, 456, 789 (250 problems x 4 = 1,000 queries)"
echo "================================================================="

python scripts/55_batched_eval_arms.py \
    --arm arm3 \
    --k_tokens 6 \
    --batch_size 16 \
    --device cuda:0 \
    --max_ans_tokens 8192

echo ""
echo "================================================================="
echo "=== Step 2/2: Evaluating Arm 3 (Latents K=32) ==="
echo "Target Device: NVIDIA RTX PRO 4500 32GB (CUDA_VISIBLE_DEVICES=1)"
echo "Sampling Spec: temp 0.7 / top_p 0.80 / top_k 20 / pp 1.5 / cap 8192"
echo "Seeds: 42, 123, 456, 789 (250 problems x 4 = 1,000 queries)"
echo "================================================================="

python scripts/55_batched_eval_arms.py \
    --arm arm3 \
    --k_tokens 32 \
    --batch_size 16 \
    --device cuda:0 \
    --max_ans_tokens 8192

echo ""
echo "================================================================="
echo "=== Arm 3 Evaluations Complete! ==="
echo "Running Paired Hierarchical Bootstrapping Analyses (B=10,000)..."
echo "================================================================="

# Comparison 1: Delta_1(K=6) = Arm 3 (K=6) - Arm 1b (Direct SFT)
python scripts/analysis/paired_bootstrap.py \
    --file_a data/streaming_arm3_k6_qwen_qwen3-1.7b.jsonl \
    --file_b data/streaming_arm1b_qwen_qwen3-1.7b.jsonl \
    --name_a "Arm 3 (Latents K=6, Fixed)" \
    --name_b "Arm 1b (Direct SFT)" \
    --output_json data/paired_bootstrap_arm3_vs_arm1b.json

# Comparison 2: Delta_2(K=6) = Arm 3 (K=6) - Arm 2b (Pause K=6)
python scripts/analysis/paired_bootstrap.py \
    --file_a data/streaming_arm3_k6_qwen_qwen3-1.7b.jsonl \
    --file_b data/streaming_arm2b_k6_qwen_qwen3-1.7b.jsonl \
    --name_a "Arm 3 (Latents K=6, Fixed)" \
    --name_b "Arm 2b (Pause K=6)" \
    --output_json data/paired_bootstrap_arm3_vs_arm2b_k6.json

# Comparison 3: Delta_1(K=32) = Arm 3 (K=32) - Arm 1b (Direct SFT)
python scripts/analysis/paired_bootstrap.py \
    --file_a data/streaming_arm3_k32_qwen_qwen3-1.7b.jsonl \
    --file_b data/streaming_arm1b_qwen_qwen3-1.7b.jsonl \
    --name_a "Arm 3 (Latents K=32)" \
    --name_b "Arm 1b (Direct SFT)" \
    --output_json data/paired_bootstrap_arm3_k32_vs_arm1b.json

# Comparison 4: Delta_2(K=32) = Arm 3 (K=32) - Arm 2b (Pause K=32)
python scripts/analysis/paired_bootstrap.py \
    --file_a data/streaming_arm3_k32_qwen_qwen3-1.7b.jsonl \
    --file_b data/streaming_arm2b_k32_qwen_qwen3-1.7b.jsonl \
    --name_a "Arm 3 (Latents K=32)" \
    --name_b "Arm 2b (Pause K=32)" \
    --output_json data/paired_bootstrap_arm3_k32_vs_arm2b_k32.json

# Comparison 5: Recurrence Scaling = Arm 3 (K=32) - Arm 3 (K=6)
python scripts/analysis/paired_bootstrap.py \
    --file_a data/streaming_arm3_k32_qwen_qwen3-1.7b.jsonl \
    --file_b data/streaming_arm3_k6_qwen_qwen3-1.7b.jsonl \
    --name_a "Arm 3 (Latents K=32)" \
    --name_b "Arm 3 (Latents K=6, Fixed)" \
    --output_json data/paired_bootstrap_arm3_k32_vs_arm3_k6.json

echo "All Arm 3 evaluations and bootstrap analyses complete."
