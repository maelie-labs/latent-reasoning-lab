#!/bin/bash
set -e
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1


echo "================================================================="
echo "PHASE 0 CLEAN BASELINE RERUN & GPQA SHARED ANCHOR ON RTX PRO 4500"
echo "================================================================="

SERVER_PORT=30000
DEVICE="cuda:1"
MODEL_ID="Qwen/Qwen3-1.7B"

echo "Launching SGLang server on RTX PRO 4500 (CUDA_VISIBLE_DEVICES=1, port $SERVER_PORT)..."
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID sglang serve \
  --model-path "$MODEL_ID" \
  --host 127.0.0.1 \
  --port $SERVER_PORT \
  --dtype bfloat16 \
  --mem-fraction-static 0.80 \
  --disable-radix-cache \
  --disable-decode-cuda-graph \
  --disable-prefill-cuda-graph \
  --trust-remote-code > server_sglang_eval.log 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID. Waiting for server to become healthy..."
while ! curl -s "http://127.0.0.1:$SERVER_PORT/health" > /dev/null; do
    sleep 3
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Error: Server failed to start. Showing last 20 lines of log:"
        tail -n 20 server_sglang_eval.log
        exit 1
    fi
done
echo "SGLang server is online and healthy!"

echo ""
echo "--- [1/3] MATH-500 Baseline Rerun (Both Modes at 32k, boxed prompt, math_verify) ---"
.venv/bin/python scripts/43_eval_math500.py \
  --server_url "http://127.0.0.1:$SERVER_PORT" \
  --model_id "$MODEL_ID" \
  --mode both \
  --max_tokens 32768 \
  --concurrency 32 \
  --output_file "data/rerun_baseline_math500.json"

echo ""
echo "--- [2/3] GSM8K Baseline Rerun (Both Modes at 32k, boxed prompt, math_verify) ---"
.venv/bin/python scripts/42_eval_gsm8k.py \
  --server_url "http://127.0.0.1:$SERVER_PORT" \
  --model_id "$MODEL_ID" \
  --mode both \
  --max_tokens 32768 \
  --concurrency 32 \
  --output_file "data/rerun_baseline_gsm8k.json"

echo ""
echo "--- [3/3] GPQA Diamond Shared Anchor Run (Thinking + Non-Thinking) ---"
.venv/bin/python scripts/45_eval_gpqa.py \
  --server_url "http://127.0.0.1:$SERVER_PORT" \
  --model_id "$MODEL_ID" \
  --mode both \
  --max_tokens 32768 \
  --concurrency 32 \
  --output_file "data/eval_gpqa_diamond_1.7b.json"

echo ""
echo "--- Running Automated Failure Diagnostic on GSM8K Thinking Outputs ---"
.venv/bin/python scripts/44_diagnose_failures.py \
  --file "data/streaming_gsm8k_qwen_qwen3-1.7b_thinking.jsonl" \
  --num_cases 20

echo ""
echo "Shutting down SGLang server (PID: $SERVER_PID)..."
kill -9 $SERVER_PID 2>/dev/null || true
echo "Server terminated. RTX PRO 4500 (cuda:1) is clean."

echo ""
echo "================================================================="
echo "PHASE 0 CALIBRATION SUITE & GPQA DIAMOND COMPLETED SUCCESSFULLY!"
echo "================================================================="
