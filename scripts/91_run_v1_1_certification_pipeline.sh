#!/usr/bin/env bash
# scripts/91_run_v1_1_certification_pipeline.sh
# End-to-end autonomous pipeline for Continuous Latent Recurrence v1.1 on Qwen/Qwen3-1.7B
set -euo pipefail

LAB_DIR="."
source .venv/bin/activate

echo "=========================================================================="
echo "=== Continuous Latent Recurrence v1.1 Certification Pipeline ==="
echo "Model: Qwen/Qwen3-1.7B"
echo "Compute Device: NVIDIA RTX PRO 4500 (cuda:1)"
echo "Start Time: $(date -u)"
echo "=========================================================================="

DATA_DIR="$LAB_DIR/data"
FINAL_TRAIN="$DATA_DIR/curated_train_v1_1_final_qwen3_1.7b.jsonl"
FINAL_DEV="$DATA_DIR/curated_dev_v1_1_final_qwen3_1.7b.jsonl"
MANIFEST="$DATA_DIR/curated_v1_1_final_manifest.json"
TEACHER_CACHE="$DATA_DIR/teacher_cot_states_qwen3_1.7b.pt"

# --- STEP 1: Await Component 1 Curation Completion ---
echo -e "\n>>> Step 1: Checking Component 1 Curation Status..."
while [ ! -f "$MANIFEST" ]; do
    echo "[$(date '+%H:%M:%S')] Waiting for scripts/90_curate_live_problem_targets.py to finish writing manifest..."
    sleep 30
done

echo "Component 1 Curation Complete!"
python -c "
import json
with open('$MANIFEST') as f:
    m = json.load(f)
print(f'Train Traces: {m[\"total_train_examples\"]} (Live Math: {m[\"live_examples\"]}, Solved-Direct: {m[\"solved_direct_examples\"]}, General: {m[\"general_instruction_examples\"]})')
print(f'Live Ratio: {m[\"actual_live_ratio\"]*100:.1f}%')
print('Curation verified!')
"

# --- STEP 2: Extract Teacher CoT States ---
echo -e "\n>>> Step 2: Extracting Teacher CoT States (Step-Level State Distillation)..."
if [ ! -f "$TEACHER_CACHE" ]; then
    python scripts/85_extract_teacher_cot_states.py \
      --model_id "Qwen/Qwen3-1.7B" \
      --device "cuda:1" \
      --train_file "$FINAL_TRAIN" \
      --dev_file "$FINAL_DEV" \
      --output_file "$TEACHER_CACHE"
else
    echo "Teacher state cache already exists at $TEACHER_CACHE, skipping extraction."
fi

# --- STEP 3: Train Arm 1b (Direct SFT Control, K=0) ---
echo -e "\n>>> Step 3: Training Arm 1b (Direct SFT Control, K=0)..."
ARM1B_DIR="$LAB_DIR/checkpoints/lora_arm1b_v1_1_qwen_qwen3-1.7b_k0"
if [ ! -f "$ARM1B_DIR/adapter_model.safetensors" ]; then
    python scripts/86_train_qwen3_arms_v1_1.py \
      --arm arm1b \
      --k_tokens 0 \
      --model_id "Qwen/Qwen3-1.7B" \
      --device "cuda:1" \
      --epochs 2 \
      --lr 2e-4 \
      --grad_accum_steps 8 \
      --train_file "$FINAL_TRAIN" \
      --dev_file "$FINAL_DEV" \
      --output_dir "$ARM1B_DIR"
else
    echo "Arm 1b checkpoint already exists at $ARM1B_DIR, skipping training."
fi

# --- STEP 4: Gate 1 Health Assertion ---
echo -e "\n>>> Step 4: Executing Gate 1 Health Assertion on MATH-500 Levels 3-5 (N=240)..."
python scripts/87_gate1_eval_arm1b_health.py \
  --model_id "Qwen/Qwen3-1.7B" \
  --checkpoint "$ARM1B_DIR" \
  --device "cuda:1" \
  --batch_size 16

python -c "
import json
with open('$DATA_DIR/gate1_arm1b_health_report.json') as f:
    rep = json.load(f)
acc = rep['pass_at_1']
print(f'GATE 1 CERTIFICATION AUDIT: Pass@1 = {acc:.2f}% (Threshold >= 68.0%)')
if acc < 68.0:
    print('ERROR: Gate 1 failed! Halting pipeline.')
    exit(1)
else:
    print('GATE 1 PASSED! Base floor is healthy.')
"

# --- STEP 5: Train Arm 3 (Continuous Latent Recurrence, K=6) ---
echo -e "\n>>> Step 5: Training Arm 3 (Continuous Latent Recurrence, K=6)..."
ARM3_DIR="$LAB_DIR/checkpoints/lora_arm3_v1_1_qwen_qwen3-1.7b_k6"
python scripts/86_train_qwen3_arms_v1_1.py \
  --arm arm3 \
  --k_tokens 6 \
  --model_id "Qwen/Qwen3-1.7B" \
  --device "cuda:1" \
  --epochs 2 \
  --lr 2e-4 \
  --grad_accum_steps 8 \
  --lambda_sweep \
  --train_file "$FINAL_TRAIN" \
  --dev_file "$FINAL_DEV" \
  --output_dir "$ARM3_DIR"

# --- STEP 6: Gate 2 Causal Steering Assertion ---
echo -e "\n>>> Step 6: Executing Gate 2 Causal Patching Assertion (N=100)..."
python scripts/88_gate2_causal_patching_v1_1.py \
  --model_id "Qwen/Qwen3-1.7B" \
  --checkpoint "$ARM3_DIR" \
  --device "cuda:1" \
  --n_problems 100 \
  --k_steps 6

python -c "
import json
with open('$DATA_DIR/gate2_causal_patching_report.json') as f:
    rep = json.load(f)
delta = rep['delta_steer']
verdict = rep['verdict']
print(f'GATE 2 CAUSAL PATCHING AUDIT: Delta Steer = {delta:.2f}% | Verdict: {verdict}')
if verdict == 'FAIL':
    print('ERROR: Gate 2 failed (< 10.0% steering). Latent channel mechanically uncoupled!')
    exit(1)
print(f'Gate 2 status: {verdict}. Proceeding to Decision Gate 1b benchmarks.')
"

echo -e "\n=========================================================================="
echo "=== Continuous Latent Recurrence v1.1 Certification Pipeline COMPLETE ==="
echo "End Time: $(date -u)"
echo "=========================================================================="
