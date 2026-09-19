#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "================================================================================"
echo "STUDY 4 POST-EVALUATION PIPELINE: BOOTSTRAP + IFEVAL"
echo "================================================================================"

echo ">>> [1/2] Running Paired Hierarchical Bootstrapping (B=10,000)..."
.venv/bin/python3 studies/tool_grounded_scratchpad/scripts/05_paired_bootstrap_scratchpad.py \
    --candidate "data/streaming_study4_tool_scratchpad_qwen3_4b.jsonl" \
    --control "data/streaming_study3_headers_only_qwen3_4b.jsonl" \
    --output_file "data/paired_bootstrap_study4_tool_scratchpad_qwen3_4b.json" \
    --b_samples 10000

echo ">>> [2/2] Running Arm 0 IFEval Surgical Neutrality (541 prompts)..."
.venv/bin/python3 scripts/126_eval_ifeval_study3_4b.py \
    --model_id "Qwen/Qwen3-4B" \
    --checkpoint "checkpoints/lora_tool_scratchpad_qwen3_4b/best_checkpoint" \
    --tag "tool_scratchpad_qwen3_4b" \
    --output_file "data/ifeval_tool_scratchpad_qwen3_4b.json" \
    --batch_size 16 \
    --device "cuda:0"

echo "================================================================================"
echo "STUDY 4 POST-EVALUATION COMPLETE"
echo "================================================================================"
