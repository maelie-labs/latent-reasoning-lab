#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="."
cd "${PROJECT_ROOT}"

MODELS=(
  "arm1_telegraphic_cot:${PROJECT_ROOT}/studies/telegraphic_cot/checkpoints/merged_arm1_telegraphic_cot_qwen3_1.7b"
  "control3_matched_pause:${PROJECT_ROOT}/studies/telegraphic_cot/checkpoints/merged_control3_matched_pause_qwen3_1.7b"
  "arm2_dual_channel_latents:${PROJECT_ROOT}/studies/telegraphic_cot/checkpoints/merged_arm2_dual_channel_latents_qwen3_1.7b"
)

cleanup_server() {
  echo "Terminating any running servers on port 30000..."
  fuser -k 30000/tcp 2>/dev/null || true
  pkill -9 -f "sglang_env/bin/sglang" 2>/dev/null || true
  sleep 3
}

trap cleanup_server EXIT

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/studies/telegraphic_cot/data"

for entry in "${MODELS[@]}"; do
  TAG="${entry%%:*}"
  MODEL_PATH="${entry##*:}"

  echo "================================================================================"
  echo "PROCESSING MODEL: ${TAG}"
  echo "PATH: ${MODEL_PATH}"
  echo "================================================================================"

  if [ ! -d "${MODEL_PATH}" ]; then
    echo "Directory ${MODEL_PATH} does not exist. Skipping..."
    continue
  fi

  cleanup_server

  echo "Starting SGLang server for ${TAG} on GPU 1 (port 30000)..."
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
  echo "Launched server (PID: ${SERVER_PID}). Waiting for health check..."

  MAX_WAIT=120
  WAITED=0
  until curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; do
    sleep 3
    WAITED=$((WAITED + 3))
    if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
      echo "Server failed to respond within ${MAX_WAIT} seconds. Exiting."
      cat "${PROJECT_ROOT}/logs/sglang_${TAG}.log" | tail -n 30
      exit 1
    fi
  done
  echo "SGLang server is healthy after ${WAITED}s."

  echo "Starting high-throughput evaluation for ${TAG}..."
  .venv/bin/python studies/telegraphic_cot/scripts/03_eval_telegraphic_models.py \
    --tag "${TAG}" \
    --server_url http://127.0.0.1:30000 \
    --base_model_id Qwen/Qwen3-1.7B \
    --benchmark_file data/benchmark_suite_250.json \
    --streaming_file "studies/telegraphic_cot/data/streaming_${TAG}.jsonl" \
    --output_file "studies/telegraphic_cot/data/eval_results_${TAG}.json" \
    --concurrency 32

  echo "Starting Arm 0 IFEval evaluation for ${TAG}..."
  .venv/bin/python studies/telegraphic_cot/scripts/05_eval_ifeval_telegraphic.py \
    --tag "${TAG}" \
    --server_url http://127.0.0.1:30000 \
    --model_path "${MODEL_PATH}" \
    --output_file "studies/telegraphic_cot/data/ifeval_${TAG}.json" \
    --concurrency 32

  echo "Evaluation of ${TAG} completed successfully!"
  cleanup_server
done

echo "================================================================================"
echo "RUNNING PAIRED HIERARCHICAL BOOTSTRAPPING (B=10,000)"
echo "================================================================================"

# 1. Arm 1 (Telegraphic CoT) vs Control 3 (Matched Pause)
if [ -f "studies/telegraphic_cot/data/streaming_arm1_telegraphic_cot.jsonl" ] && [ -f "studies/telegraphic_cot/data/streaming_control3_matched_pause.jsonl" ]; then
  .venv/bin/python studies/telegraphic_cot/scripts/04_paired_bootstrap_telegraphic.py \
    --streaming_a "studies/telegraphic_cot/data/streaming_arm1_telegraphic_cot.jsonl" \
    --streaming_b "studies/telegraphic_cot/data/streaming_control3_matched_pause.jsonl" \
    --name_a "Arm 1 (Telegraphic CoT)" \
    --name_b "Control 3 (Matched Pause)" \
    --output_file "studies/telegraphic_cot/data/bootstrap_arm1_vs_control3.json"
fi

# 2. Arm 1 (Telegraphic CoT) vs Control 2 (Base Direct)
if [ -f "studies/telegraphic_cot/data/streaming_arm1_telegraphic_cot.jsonl" ] && [ -f "data/streaming_arm1_direct_qwen_qwen3-1.7b.jsonl" ]; then
  .venv/bin/python studies/telegraphic_cot/scripts/04_paired_bootstrap_telegraphic.py \
    --streaming_a "studies/telegraphic_cot/data/streaming_arm1_telegraphic_cot.jsonl" \
    --streaming_b "data/streaming_arm1_direct_qwen_qwen3-1.7b.jsonl" \
    --name_a "Arm 1 (Telegraphic CoT)" \
    --name_b "Control 2 (Base Direct Floor)" \
    --output_file "studies/telegraphic_cot/data/bootstrap_arm1_vs_control2_base.json"
fi

# 3. Arm 1 (Telegraphic CoT) vs Control 1 (Verbose CoT)
if [ -f "studies/telegraphic_cot/data/streaming_arm1_telegraphic_cot.jsonl" ] && [ -f "data/streaming_arm4_thinking_qwen_qwen3-1.7b.jsonl" ]; then
  .venv/bin/python studies/telegraphic_cot/scripts/04_paired_bootstrap_telegraphic.py \
    --streaming_a "studies/telegraphic_cot/data/streaming_arm1_telegraphic_cot.jsonl" \
    --streaming_b "data/streaming_arm4_thinking_qwen_qwen3-1.7b.jsonl" \
    --name_a "Arm 1 (Telegraphic CoT)" \
    --name_b "Control 1 (Verbose CoT Ceiling)" \
    --output_file "studies/telegraphic_cot/data/bootstrap_arm1_vs_control1_verbose.json"
fi

# 4. Arm 2 (Dual-Channel Latents) vs Arm 1 (Telegraphic CoT)
if [ -f "studies/telegraphic_cot/data/streaming_arm2_dual_channel_latents.jsonl" ] && [ -f "studies/telegraphic_cot/data/streaming_arm1_telegraphic_cot.jsonl" ]; then
  .venv/bin/python studies/telegraphic_cot/scripts/04_paired_bootstrap_telegraphic.py \
    --streaming_a "studies/telegraphic_cot/data/streaming_arm2_dual_channel_latents.jsonl" \
    --streaming_b "studies/telegraphic_cot/data/streaming_arm1_telegraphic_cot.jsonl" \
    --name_a "Arm 2 (Dual-Channel Latents)" \
    --name_b "Arm 1 (Telegraphic CoT)" \
    --output_file "studies/telegraphic_cot/data/bootstrap_arm2_latents_vs_arm1.json"
fi

echo "================================================================================"
echo "TELEGRAPHIC COT EVALUATION AND BOOTSTRAPPING PIPELINE COMPLETED!"
echo "================================================================================"
