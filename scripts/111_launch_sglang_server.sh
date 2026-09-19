#!/usr/bin/env bash
# scripts/111_launch_sglang_server.sh
# Parametric SGLang server launcher on dedicated GPU 1 (RTX PRO 4500 32GB).
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen3.5-4B}"
PORT="${2:-30000}"
CONTEXT_LEN="${3:-90000}"

echo "=========================================================================="
echo "=== Launching SGLang Server on GPU 1 (RTX PRO 4500 32GB) ==="
echo "Model: $MODEL_ID"
echo "Port: $PORT"
echo "Context Length: $CONTEXT_LEN"
echo "Memory Fraction: 0.90"
echo "Start Time: $(date -u)"
echo "=========================================================================="

LOG_FILE="sglang_${PORT}.log"

# Clean up any existing process on this port
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

nohup env CUDA_VISIBLE_DEVICES=1 sglang serve \
  --model-path "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --context-length "$CONTEXT_LEN" \
  --disable-radix-cache > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
disown "$SERVER_PID"
echo "SGLang server started with PID $SERVER_PID. Monitoring $LOG_FILE..."

# Wait for healthy endpoint (HTTP 200)
MAX_WAIT=240
WAITED=0
while ! curl -s -f "http://127.0.0.1:${PORT}/health" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: SGLang server died during startup. Log output:"
        tail -n 40 "$LOG_FILE"
        exit 1
    fi
    sleep 3
    WAITED=$((WAITED + 3))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "Timeout waiting for SGLang server to become healthy after ${MAX_WAIT}s."
        tail -n 40 "$LOG_FILE"
        kill -9 "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
done

echo "SGLang Server is healthy and ready on http://127.0.0.1:${PORT} (PID: $SERVER_PID)"
