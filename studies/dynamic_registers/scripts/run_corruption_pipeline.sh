#!/bin/bash
set -eo pipefail

PROJECT_ROOT="."
cd "${PROJECT_ROOT}"

MODEL_PATH="${PROJECT_ROOT}/studies/dynamic_registers/checkpoints/merged_arm1_dynamic_registers_qwen3_1.7b"
LOG_PATH="${PROJECT_ROOT}/logs/sglang_corruption_arm1.log"
OUTPUT_FILE="${PROJECT_ROOT}/studies/dynamic_registers/data/register_corruption_results.json"

cleanup() {
    echo "Cleaning up SGLang server..."
    fuser -k 30000/tcp 2>/dev/null || true
    pkill -9 -f "sglang_env/bin/sglang" 2>/dev/null || true
}
trap cleanup EXIT

echo "Ensuring port 30000 is free..."
fuser -k 30000/tcp 2>/dev/null || true
pkill -9 -f "sglang_env/bin/sglang" 2>/dev/null || true
sleep 3

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/studies/dynamic_registers/data"

echo "Starting SGLang server for Arm 1 on GPU 1 (port 30000)..."
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID sglang serve \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port 30000 \
    --dtype bfloat16 \
    --mem-fraction-static 0.90 \
    --disable-radix-cache \
    --disable-decode-cuda-graph \
    --disable-prefill-cuda-graph \
    --trust-remote-code > "${LOG_PATH}" 2>&1 &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}. Waiting for health check..."

MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; then
        echo "SGLang server is healthy after ${WAITED}s!"
        break
    fi
    sleep 3
    WAITED=$((WAITED + 3))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Server failed to become healthy within ${MAX_WAIT}s. Log tail:"
    tail -n 30 "${LOG_PATH}"
    exit 1
fi

echo "Running Controlled Register Corruption Test..."
.venv/bin/python studies/dynamic_registers/scripts/06_register_corruption_test.py \
    --server_url http://127.0.0.1:30000 \
    --tag arm1_dynamic_registers \
    --benchmark_file data/benchmark_suite_250.json \
    --num_pairs 100 \
    --output_file "${OUTPUT_FILE}"

echo "Corruption test completed successfully!"
