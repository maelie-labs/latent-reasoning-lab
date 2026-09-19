#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="."
source .venv/bin/activate

echo "=========================================================================="
echo "=== Phase 0: Qwen3.5-2B Hybrid Gated DeltaNet Baselines & Anchor ==="
echo "Target Device: NVIDIA RTX PRO 4500 32GB (CUDA_VISIBLE_DEVICES=1)"
echo "Static Memory Fraction: --mem-fraction-static 0.90"
echo "Start Time: $(date -u)"
echo "=========================================================================="

# 1. Dual-State Architecture Accounting
echo ">>> Step 1/6: Dual-State Memory Accounting for Qwen/Qwen3.5-2B..."
python scripts/49_benchmark_qwen3.5_2b.py --skip_eval

# 2. Launch SGLang Server on GPU 1
SERVER_PORT=30000
echo ">>> Step 2/6: Launching SGLang Server on GPU 1 (Port $SERVER_PORT)..."
CUDA_VISIBLE_DEVICES=1 sglang serve \
  --model-path "Qwen/Qwen3.5-2B" \
  --host 127.0.0.1 \
  --port $SERVER_PORT \
  --dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --disable-radix-cache > server_sglang_qwen35.log 2>&1 &
SERVER_PID=$!
echo "SGLang Server PID: $SERVER_PID. Waiting for healthy status..."

while ! curl -s "http://127.0.0.1:$SERVER_PORT/health" > /dev/null; do
    sleep 3
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Error: SGLang server failed to start. Tail log:"
        tail -n 30 server_sglang_qwen35.log
        exit 1
    fi
done
echo "SGLang Server is healthy and responding!"

# 3. Concurrency Smoke Test
echo ">>> Step 3/6: Concurrency Smoke Test for Qwen/Qwen3.5-2B..."
python scripts/56_concurrency_smoke_test.py \
  --server_url "http://127.0.0.1:$SERVER_PORT" \
  --model_id "Qwen/Qwen3.5-2B" \
  --register_suite "qwen3.5_2b"

# Read locked concurrency from registry (calibrated knee C*=8)
CONCURRENCY=8

# 4. Fundamental Baselines: MATH-500
echo ">>> Step 4/6: MATH-500 Baselines (Thinking & Non-Thinking)..."
python scripts/43_eval_math500.py \
  --server_url "http://127.0.0.1:$SERVER_PORT" \
  --model_id "Qwen/Qwen3.5-2B" \
  --mode both \
  --concurrency $CONCURRENCY \
  --output_file "data/eval_math500_qwen3.5_2b_both.json"

# 5. Fundamental Baselines: GSM8K
echo ">>> Step 5/6: GSM8K Baselines (Thinking & Non-Thinking)..."
python scripts/42_eval_gsm8k.py \
  --server_url "http://127.0.0.1:$SERVER_PORT" \
  --model_id "Qwen/Qwen3.5-2B" \
  --mode both \
  --concurrency $CONCURRENCY \
  --output_file "data/eval_gsm8k_qwen3.5_2b_both.json"

# 6. GPQA Diamond Calibration Anchor (1,584 evaluations)
echo ">>> Step 6/6: GPQA Diamond Calibration Anchor (Target: 51.6%)..."
python scripts/45_eval_gpqa.py \
  --server_url "http://127.0.0.1:$SERVER_PORT" \
  --model_id "Qwen/Qwen3.5-2B" \
  --mode thinking \
  --num_samples 8 \
  --concurrency $CONCURRENCY \
  --output_file "data/eval_gpqa_diamond_2b.json"

# Shutdown SGLang Server
echo "Shutting down SGLang server (PID: $SERVER_PID)..."
kill -9 $SERVER_PID 2>/dev/null || true
echo "Server terminated cleanly."

echo "=========================================================================="
echo "=== Phase 0 Baselines for Qwen3.5-2B Complete! ==="
echo "=========================================================================="
