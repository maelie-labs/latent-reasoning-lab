#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="."
cd "${PROJECT_ROOT}"

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
  echo "LAUNCHING IFEVAL EVALUATION FOR: ${TAG}"
  echo "Model Path: ${MODEL_PATH}"
  echo "================================================================================"

  cleanup_server

  CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID sglang serve \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port 30000 \
    --dtype bfloat16 \
    --mem-fraction-static 0.90 \
    --disable-radix-cache \
    --disable-decode-cuda-graph \
    --disable-prefill-cuda-graph \
    --trust-remote-code > "${PROJECT_ROOT}/logs/sglang_ifeval_${TAG}.log" 2>&1 &
  SERVER_PID=$!

  echo "Waiting for SGLang server to initialize..."
  for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; then
      echo "SGLang server is healthy and ready!"
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "ERROR: SGLang server exited unexpectedly! Check logs/sglang_ifeval_${TAG}.log"
      cat "${PROJECT_ROOT}/logs/sglang_ifeval_${TAG}.log" | tail -n 20
      exit 1
    fi
    sleep 2
  done

  "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/studies/dynamic_registers/scripts/05_eval_ifeval_registers.py" \
    --server_url "http://127.0.0.1:30000" \
    --tag "${TAG}" \
    --model_path "${MODEL_PATH}" \
    --output_file "${PROJECT_ROOT}/studies/dynamic_registers/data/ifeval_${TAG}.json" \
    --concurrency 32

  cleanup_server
  sleep 2
done

echo "IFEval pipeline completed successfully for all 3 models!"
