#!/usr/bin/env bash
set -e

REPO_DIR="."

mkdir -p logs/overnight
MASTER_LOG="logs/overnight/lambda_retrain_and_gate2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "======================================================================"
echo "STARTING CONCURRENT LAMBDA RETRAINING & GATE 2 VERIFICATION"
echo "Start Time: $(date)"
echo "Master Log: $MASTER_LOG"
echo "======================================================================"

TRAIN_DATA="data/curated_train_v1_1_dilution10pct_qwen3_1.7b.jsonl"
CKPT_L1="checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_lambda1.0"
CKPT_L3="checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_lambda3.0"

echo "[Step 1] Launching concurrent training on GPU 1 (lambda=1.0) and GPU 0 (lambda=3.0)..."

# Job 1: Lambda = 1.0 on GPU 1
.venv/bin/python scripts/86_train_qwen3_arms_v1_1.py \
    --arm arm3 \
    --k_tokens 6 \
    --model_id "Qwen/Qwen3-1.7B" \
    --lambda_align 1.0 \
    --output_dir "$CKPT_L1" \
    --train_file "$TRAIN_DATA" \
    --device cuda:1 \
    > logs/overnight/train_arm3_k6_lambda1.0.log 2>&1 &
PID_L1=$!

# Job 2: Lambda = 3.0 on GPU 0 (with gradient checkpointing for safe 6.6GB VRAM)
.venv/bin/python scripts/86_train_qwen3_arms_v1_1.py \
    --arm arm3 \
    --k_tokens 6 \
    --model_id "Qwen/Qwen3-1.7B" \
    --lambda_align 3.0 \
    --gradient_checkpointing \
    --output_dir "$CKPT_L3" \
    --train_file "$TRAIN_DATA" \
    --device cuda:0 \
    > logs/overnight/train_arm3_k6_lambda3.0.log 2>&1 &
PID_L3=$!

echo "Job 1 (lambda=1.0, GPU 1) PID: $PID_L1"
echo "Job 2 (lambda=3.0, GPU 0) PID: $PID_L3"
echo "Waiting for concurrent training jobs to finish..."

wait $PID_L1
STATUS_L1=$?
echo "Job 1 (lambda=1.0) finished with exit code $STATUS_L1 at $(date)."

wait $PID_L3
STATUS_L3=$?
echo "Job 2 (lambda=3.0) finished with exit code $STATUS_L3 at $(date)."

if [ $STATUS_L1 -ne 0 ] || [ $STATUS_L3 -ne 0 ]; then
    echo "ERROR: One or both training jobs failed! Halting."
    exit 1
fi

echo "======================================================================"
echo "[Step 2] Executing Gate 2 Causal Activation Patching on lambda=1.0..."
echo "======================================================================"

REPORT_L1="data/gate2_causal_patching_report_lambda1.0.json"
set +e
if [ ! -f "$REPORT_L1" ]; then
    .venv/bin/python scripts/88_gate2_causal_patching_v1_1.py \
        --model_id "Qwen/Qwen3-1.7B" \
        --checkpoint "$CKPT_L1" \
        --output_file "$REPORT_L1" \
        --device cuda:1 \
        --n_problems 100 \
        > logs/overnight/gate2_causal_patching_lambda1.0.log 2>&1
    STATUS_GATE_L1=$?
else
    echo "Gate 2 (lambda=1.0) report already exists at $REPORT_L1. Skipping re-run."
    STATUS_GATE_L1=0
fi
set -e

echo "Gate 2 (lambda=1.0) finished with status code $STATUS_GATE_L1."

echo "======================================================================"
echo "[Step 3] Executing Gate 2 Causal Activation Patching on lambda=3.0..."
echo "======================================================================"

REPORT_L3="data/gate2_causal_patching_report_lambda3.0.json"
set +e
.venv/bin/python scripts/88_gate2_causal_patching_v1_1.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --checkpoint "$CKPT_L3" \
    --output_file "$REPORT_L3" \
    --device cuda:1 \
    --n_problems 100 \
    > logs/overnight/gate2_causal_patching_lambda3.0.log 2>&1
STATUS_GATE_L3=$?
set -e

echo "Gate 2 (lambda=3.0) finished with status code $STATUS_GATE_L3."

echo "======================================================================"
echo "[Step 4] Summarizing Gate 2 Outcomes & Deciding Downstream Action"
echo "======================================================================"

.venv/bin/python -c "
import json, sys

print('=== GATE 2 SUMMARY REPORT ===')
for name, path in [('Lambda 1.0', '$REPORT_L1'), ('Lambda 3.0', '$REPORT_L3')]:
    try:
        with open(path) as f:
            d = json.load(f)
        steer = d.get('delta_steer', 0.0)
        p_chance = d.get('p_chance', 0.0)
        p_steered = d.get('p_steered', 0.0)
        amp = d.get('mean_amplification', 0.0)
        verdict = d.get('verdict', 'UNKNOWN')
        print(f'{name}: Delta Steer = {steer:+.2f}% (P_chance={p_chance:.2f}%, P_steered={p_steered:.2f}%, Amp={amp:.2f}x) -> Verdict: {verdict}')
    except Exception as e:
        print(f'{name}: Error loading {path}: {e}')
"

echo "Pipeline script completed at $(date)."
