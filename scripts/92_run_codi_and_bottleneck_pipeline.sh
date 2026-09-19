#!/usr/bin/env bash
set -e

REPO_DIR="."

mkdir -p logs/overnight
MASTER_LOG="logs/overnight/codi_and_bottleneck_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "======================================================================"
echo "STARTING CONCURRENT CODI-FIXED-TARGET & BOTTLENECK TRAINING PIPELINE"
echo "Start Time: $(date)"
echo "Master Log: $MASTER_LOG"
echo "======================================================================"

TRAIN_DATA="data/curated_train_v1_1_dilution10pct_qwen3_1.7b.jsonl"
CKPT_CODI="checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_codi_fallback"
CKPT_BOTTLENECK="checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_bottleneck"

echo "[Step 1] Launching concurrent training on GPU 1 (CODI fixed-target) and GPU 0 (Bottleneck)..."

# Job 1: Frozen-Teacher Answer-Position Distillation (CODI fixed-target) on GPU 1 (RTX PRO 4500 32GB)
.venv/bin/python scripts/86_train_qwen3_arms_v1_1.py \
    --arm arm3 \
    --k_tokens 6 \
    --model_id "Qwen/Qwen3-1.7B" \
    --lambda_align 1.0 \
    --use_codi_fallback \
    --output_dir "$CKPT_CODI" \
    --train_file "$TRAIN_DATA" \
    --device cuda:1 \
    > logs/overnight/train_arm3_k6_codi_fallback.log 2>&1 &
PID_CODI=$!

# Job 2: Question-Token Attention Bottleneck Variant on GPU 0 (RTX 4080 16GB)
.venv/bin/python -u scripts/86_train_qwen3_arms_v1_1.py \
    --arm arm3 \
    --k_tokens 6 \
    --model_id "Qwen/Qwen3-1.7B" \
    --lambda_align 1.0 \
    --use_bottleneck_mask \
    --bottleneck_relax_steps 100 \
    --max_target_len 1024 \
    --output_dir "$CKPT_BOTTLENECK" \
    --train_file "$TRAIN_DATA" \
    --device cuda:0 \
    > logs/overnight/train_arm3_k6_bottleneck.log 2>&1 &
PID_BOTTLENECK=$!

echo "Job 1 (CODI fixed-target, GPU 1) PID: $PID_CODI"
echo "Job 2 (Bottleneck, GPU 0) PID: $PID_BOTTLENECK"
echo "Waiting for concurrent training jobs to finish..."

wait $PID_CODI
STATUS_CODI=$?
echo "Job 1 (CODI fixed-target) finished with exit code $STATUS_CODI at $(date)."

wait $PID_BOTTLENECK
STATUS_BOTTLENECK=$?
echo "Job 2 (Bottleneck) finished with exit code $STATUS_BOTTLENECK at $(date)."

if [ $STATUS_CODI -ne 0 ] || [ $STATUS_BOTTLENECK -ne 0 ]; then
    echo "ERROR: One or both training jobs failed! Halting."
    exit 1
fi

echo "======================================================================"
echo "[Step 2] Executing Gate 2 Causal Patching Matrix (t=3 and t=6 on GPU 1)..."
echo "======================================================================"

# 1. CODI at t=3
echo "--- Running Gate 2: CODI fixed-target at t=3 ---"
set +e
.venv/bin/python scripts/88_gate2_causal_patching_v1_1.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --checkpoint "$CKPT_CODI" \
    --patch_step 3 \
    --output_file "data/gate2_report_codi_fallback_t3.json" \
    --device cuda:1 \
    --batch_size 16 \
    --n_problems 100 \
    > logs/overnight/gate2_codi_fallback_t3.log 2>&1

# 2. CODI at t=6 (Hand-off position)
echo "--- Running Gate 2: CODI fixed-target at t=6 ---"
.venv/bin/python scripts/88_gate2_causal_patching_v1_1.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --checkpoint "$CKPT_CODI" \
    --patch_step 6 \
    --output_file "data/gate2_report_codi_fallback_t6.json" \
    --device cuda:1 \
    --batch_size 16 \
    --n_problems 100 \
    > logs/overnight/gate2_codi_fallback_t6.log 2>&1

# 3. Bottleneck at t=3
echo "--- Running Gate 2: Bottleneck at t=3 ---"
.venv/bin/python scripts/88_gate2_causal_patching_v1_1.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --checkpoint "$CKPT_BOTTLENECK" \
    --patch_step 3 \
    --output_file "data/gate2_report_bottleneck_t3.json" \
    --device cuda:1 \
    --batch_size 16 \
    --n_problems 100 \
    > logs/overnight/gate2_bottleneck_t3.log 2>&1

# 4. Bottleneck at t=6 (Hand-off position)
echo "--- Running Gate 2: Bottleneck at t=6 ---"
.venv/bin/python scripts/88_gate2_causal_patching_v1_1.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --checkpoint "$CKPT_BOTTLENECK" \
    --patch_step 6 \
    --output_file "data/gate2_report_bottleneck_t6.json" \
    --device cuda:1 \
    --batch_size 16 \
    --n_problems 100 \
    > logs/overnight/gate2_bottleneck_t6.log 2>&1
set -e

echo "======================================================================"
echo "[Step 3] Summarizing Gate 2 Outcomes Across Both Read-Side Interventions"
echo "======================================================================"

.venv/bin/python -c "
import json

runs = [
    ('CODI Fixed-Target (t=3)', 'data/gate2_report_codi_fallback_t3.json'),
    ('CODI Fixed-Target (t=6, Hand-off)', 'data/gate2_report_codi_fallback_t6.json'),
    ('Bottleneck Mask (t=3)', 'data/gate2_report_bottleneck_t3.json'),
    ('Bottleneck Mask (t=6, Hand-off)', 'data/gate2_report_bottleneck_t6.json')
]

print('\n=== READ-SIDE INTERVENTION GATE 2 MATRIX ===')
print(f'{\"Configuration\":<38} | {\"P_chance\":<9} | {\"P_steered\":<9} | {\"Delta Steer\":<12} | {\"Amplification\":<13} | {\"Verdict\"}')
print('-' * 98)
for name, path in runs:
    try:
        with open(path) as f:
            d = json.load(f)
        steer = d.get('delta_steer', 0.0)
        p_chance = d.get('p_chance', 0.0)
        p_steered = d.get('p_steered', 0.0)
        amp = d.get('mean_amplification', 0.0)
        verdict = d.get('verdict', 'UNKNOWN')
        print(f'{name:<38} | {p_chance:>8.2f}% | {p_steered:>8.2f}% | {steer:>+11.2f}% | {amp:>12.2f}x | {verdict}')
    except Exception as e:
        print(f'{name:<38} | ERROR: {e}')
"

echo "Pipeline script completed at $(date)."
