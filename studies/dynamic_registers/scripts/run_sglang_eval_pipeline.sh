#!/usr/bin/env bash
set -euo pipefail

# Project root
PROJECT_ROOT="."
cd "${PROJECT_ROOT}"

# Activate virtual environment
source .venv/bin/activate

MODELS=(
  "control1_headers_only:${PROJECT_ROOT}/studies/dynamic_registers/checkpoints/merged_control1_headers_only_qwen3_1.7b"
  "arm1_dynamic_registers:${PROJECT_ROOT}/studies/dynamic_registers/checkpoints/merged_arm1_dynamic_registers_qwen3_1.7b"
  "arm2_matched_filler:${PROJECT_ROOT}/studies/dynamic_registers/checkpoints/merged_arm2_matched_filler_qwen3_1.7b"
)

cleanup_server() {
  echo "Terminating any running servers on port 30000..."
  fuser -k 30000/tcp 2>/dev/null || true
  pkill -9 -f "sglang_env/bin/sglang" 2>/dev/null || true
  sleep 3
}

trap cleanup_server EXIT

for ITEM in "${MODELS[@]}"; do
  TAG="${ITEM%%:*}"
  MODEL_PATH="${ITEM##*:}"

  echo "================================================================================"
  echo "STARTING HIGH-THROUGHPUT SGLANG EVALUATION FOR: ${TAG}"
  echo "Model Path: ${MODEL_PATH}"
  echo "================================================================================"

  cleanup_server

  # Launch SGLang server on dedicated GPU 1 (RTX PRO 4500 32GB)
  # mem-fraction-static 0.90 ensures hardware saturation
  echo "Launching SGLang server on GPU 1 (RTX PRO 4500 32GB)..."
  CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID sglang serve \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port 30000 \
    --dtype bfloat16 \
    --mem-fraction-static 0.90 \
    --disable-radix-cache \
    --disable-decode-cuda-graph \
    --disable-prefill-cuda-graph \
    --trust-remote-code > "${PROJECT_ROOT}/logs/sglang_${TAG}.log" 2>&1 &
  SERVER_PID=$!

  # Wait for server health
  echo "Waiting for SGLang server to initialize..."
  for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; then
      echo "SGLang server is healthy and ready!"
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "ERROR: SGLang server process exited unexpectedly! Check logs/sglang_${TAG}.log"
      cat "${PROJECT_ROOT}/logs/sglang_${TAG}.log" | tail -n 20
      exit 1
    fi
    sleep 2
  done

  echo "Running 250-suite benchmark evaluation across 4 seeds (N=1,000 queries, Concurrency=32)..."
  "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/studies/dynamic_registers/scripts/03_sglang_eval_registers.py" \
    --server_url "http://127.0.0.1:30000" \
    --tag "${TAG}" \
    --streaming_file "${PROJECT_ROOT}/studies/dynamic_registers/data/streaming_${TAG}.jsonl" \
    --output_file "${PROJECT_ROOT}/studies/dynamic_registers/data/eval_results_${TAG}.json" \
    --concurrency 32

  echo "Evaluation of ${TAG} complete! Shutting down server..."
  cleanup_server
  sleep 2
done

echo "================================================================================"
echo "ALL THREE ARMS EVALUATED! COMPUTING PAIRED HIERARCHICAL BOOTSTRAPPING (B=10,000)"
echo "================================================================================"

"${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/studies/dynamic_registers/scripts/04_paired_bootstrap_registers.py" \
  --arm1_file "${PROJECT_ROOT}/studies/dynamic_registers/data/streaming_arm1_dynamic_registers.jsonl" \
  --control_file "${PROJECT_ROOT}/studies/dynamic_registers/data/streaming_control1_headers_only.jsonl" \
  --comparator_name "Control 1 (Headers Only)" \
  --output_file "${PROJECT_ROOT}/studies/dynamic_registers/data/bootstrap_arm1_vs_control1.json"

"${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/studies/dynamic_registers/scripts/04_paired_bootstrap_registers.py" \
  --arm1_file "${PROJECT_ROOT}/studies/dynamic_registers/data/streaming_arm1_dynamic_registers.jsonl" \
  --control_file "${PROJECT_ROOT}/studies/dynamic_registers/data/streaming_arm2_matched_filler.jsonl" \
  --comparator_name "Arm 2 (Matched Value Filler)" \
  --output_file "${PROJECT_ROOT}/studies/dynamic_registers/data/bootstrap_arm1_vs_arm2.json"

if [ -f "${PROJECT_ROOT}/data/streaming_arm1_qwen_qwen3-1.7b.jsonl" ]; then
  "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/studies/dynamic_registers/scripts/04_paired_bootstrap_registers.py" \
    --arm1_file "${PROJECT_ROOT}/studies/dynamic_registers/data/streaming_arm1_dynamic_registers.jsonl" \
    --control_file "${PROJECT_ROOT}/data/streaming_arm1_qwen_qwen3-1.7b.jsonl" \
    --comparator_name "Untrained Base Direct" \
    --output_file "${PROJECT_ROOT}/studies/dynamic_registers/data/bootstrap_arm1_vs_base.json"
fi

echo "Pipeline execution finished successfully!"
