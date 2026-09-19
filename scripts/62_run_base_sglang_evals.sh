#!/bin/bash
set -e

LAB_DIR="."

echo "=== High-Throughput Base SGLang Evaluation Pipeline (GPU 1) ==="
echo "Endpoint: http://127.0.0.1:30000"

# 1. Arm 4: Unconstrained Discrete CoT
echo -e "\n>>> Launching Arm 4: Unconstrained Thinking Mode (Concurrency 32)..."
.venv/bin/python scripts/57_sglang_eval_arm4.py \
  --server_url http://127.0.0.1:30000 \
  --concurrency 32

# 2. Arm 2: Minimal Discrete Thinking Tokens (K=6)
echo -e "\n>>> Launching Arm 2: Minimal Discrete Tokens (K=6, Concurrency 32)..."
.venv/bin/python scripts/60_sglang_eval_arm2.py \
  --server_url http://127.0.0.1:30000 \
  --k_tokens 6 \
  --concurrency 32

# 3. Arm 2: Minimal Discrete Thinking Tokens (K=32)
echo -e "\n>>> Launching Arm 2: Minimal Discrete Tokens (K=32, Concurrency 32)..."
.venv/bin/python scripts/60_sglang_eval_arm2.py \
  --server_url http://127.0.0.1:30000 \
  --k_tokens 32 \
  --concurrency 32

# 4. Arm 1: Clean Direct Non-Thinking Baseline (Explicit Prompt <think>\n\n</think>\n\n)
echo -e "\n>>> Rerunning Arm 1: Clean Direct Non-Thinking Baseline (Concurrency 32)..."
.venv/bin/python scripts/53_sglang_eval_arm1.py \
  --server_url http://127.0.0.1:30000 \
  --concurrency 32

echo -e "\n========================================================"
echo "SUCCESS: All Base Model SGLang evaluations complete!"
echo "========================================================"

