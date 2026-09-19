#!/usr/bin/env bash
# studies/dynamic_registers/scripts/run_study_pipeline.sh
# End-to-end training and evaluation pipeline for Staged Dynamic Registers
set -e

STUDY_DIR="studies/dynamic_registers"
PYTHON=".venv/bin/python"

echo "================================================================================"
echo "STAGED DYNAMIC REGISTERS STUDY: MASTER AUTOMATED EXECUTION PIPELINE"
echo "================================================================================"

# 1. Train Control 1: Matched Headers-Only SFT on GPU 1
echo ""
echo "=== Step 1: Training Control 1 (Matched Headers-Only SFT) on GPU 1 ==="
$PYTHON $STUDY_DIR/scripts/02_train_register_models.py \
  --mode headers_only \
  --data_dir $STUDY_DIR/data \
  --output_dir $STUDY_DIR/checkpoints/lora_control1_headers_only_qwen3_1.7b \
  --device cuda:1 \
  --epochs 2 \
  --lr 1e-4 \
  --accum_steps 16

# 2. Train Arm 1: Dynamic Discrete Registers on GPU 1
echo ""
echo "=== Step 2: Training Arm 1 (Dynamic Discrete Registers) on GPU 1 ==="
$PYTHON $STUDY_DIR/scripts/02_train_register_models.py \
  --mode dynamic_registers \
  --data_dir $STUDY_DIR/data \
  --output_dir $STUDY_DIR/checkpoints/lora_arm1_dynamic_registers_qwen3_1.7b \
  --device cuda:1 \
  --epochs 2 \
  --lr 1e-4 \
  --accum_steps 16

# 3. Train Arm 2: Empirically-Grounded Mismatched Value Filler on GPU 1
echo ""
echo "=== Step 3: Training Arm 2 (Mismatched Value Filler Control) on GPU 1 ==="
$PYTHON $STUDY_DIR/scripts/02_train_register_models.py \
  --mode matched_filler \
  --data_dir $STUDY_DIR/data \
  --output_dir $STUDY_DIR/checkpoints/lora_arm2_matched_filler_qwen3_1.7b \
  --device cuda:1 \
  --epochs 2 \
  --lr 1e-4 \
  --accum_steps 16

# 4. Evaluate Control 1 on 250-Suite (N=1,000 across 4 seeds)
echo ""
echo "=== Step 4: Evaluating Control 1 (Headers-Only) on 250 Suite ==="
$PYTHON $STUDY_DIR/scripts/03_eval_register_models.py \
  --checkpoint $STUDY_DIR/checkpoints/lora_control1_headers_only_qwen3_1.7b/best_checkpoint \
  --tag control1_headers_only \
  --output_file $STUDY_DIR/data/eval_results_control1_headers_only.json \
  --streaming_file $STUDY_DIR/data/streaming_control1_headers_only.jsonl \
  --device cuda:1 \
  --batch_size 8

# 5. Evaluate Arm 1 on 250-Suite (N=1,000 across 4 seeds)
echo ""
echo "=== Step 5: Evaluating Arm 1 (Dynamic Registers) on 250 Suite ==="
$PYTHON $STUDY_DIR/scripts/03_eval_register_models.py \
  --checkpoint $STUDY_DIR/checkpoints/lora_arm1_dynamic_registers_qwen3_1.7b/best_checkpoint \
  --tag arm1_dynamic_registers \
  --output_file $STUDY_DIR/data/eval_results_arm1_dynamic_registers.json \
  --streaming_file $STUDY_DIR/data/streaming_arm1_dynamic_registers.jsonl \
  --device cuda:1 \
  --batch_size 8

# 6. Evaluate Arm 2 on 250-Suite (N=1,000 across 4 seeds)
echo ""
echo "=== Step 6: Evaluating Arm 2 (Mismatched Filler) on 250 Suite ==="
$PYTHON $STUDY_DIR/scripts/03_eval_register_models.py \
  --checkpoint $STUDY_DIR/checkpoints/lora_arm2_matched_filler_qwen3_1.7b/best_checkpoint \
  --tag arm2_matched_filler \
  --output_file $STUDY_DIR/data/eval_results_arm2_matched_filler.json \
  --streaming_file $STUDY_DIR/data/streaming_arm2_matched_filler.jsonl \
  --device cuda:1 \
  --batch_size 8

# 7. Paired Hierarchical Bootstrap Matrix Analysis (B=10,000 iterations)
echo ""
echo "=== Step 7: Paired Bootstrap Matrix Analysis ==="
# Delta_1: vs Control 1 (Matched Headers-Only SFT baseline)
$PYTHON $STUDY_DIR/scripts/04_paired_bootstrap_registers.py \
  --arm1_file $STUDY_DIR/data/streaming_arm1_dynamic_registers.jsonl \
  --control_file $STUDY_DIR/data/streaming_control1_headers_only.jsonl \
  --comparator_name "Control 1 (Matched Headers-Only SFT)" \
  --output_file $STUDY_DIR/data/bootstrap_arm1_vs_control1.json

# Delta_2: vs Arm 2 (Empirical Mismatched Value Filler)
$PYTHON $STUDY_DIR/scripts/04_paired_bootstrap_registers.py \
  --arm1_file $STUDY_DIR/data/streaming_arm1_dynamic_registers.jsonl \
  --control_file $STUDY_DIR/data/streaming_arm2_matched_filler.jsonl \
  --comparator_name "Arm 2 (Mismatched Value Filler)" \
  --output_file $STUDY_DIR/data/bootstrap_arm1_vs_arm2.json

# Delta_3: vs Untrained Base Direct (Arm 1 baseline from sealed study)
$PYTHON $STUDY_DIR/scripts/04_paired_bootstrap_registers.py \
  --arm1_file $STUDY_DIR/data/streaming_arm1_dynamic_registers.jsonl \
  --control_file data/streaming_arm1_qwen_qwen3-1.7b.jsonl \
  --comparator_name "Untrained Base Direct (Arm 1: 73.90%)" \
  --output_file $STUDY_DIR/data/bootstrap_arm1_vs_untrained_base.json

# Cross-Study Reference: vs Historical Arm 1b-R
$PYTHON $STUDY_DIR/scripts/04_paired_bootstrap_registers.py \
  --arm1_file $STUDY_DIR/data/streaming_arm1_dynamic_registers.jsonl \
  --control_file data/streaming_arm1b_register_headers_only.jsonl \
  --comparator_name "Historical Arm 1b-R (73.20%)" \
  --output_file $STUDY_DIR/data/bootstrap_arm1_vs_historical_arm1b_r.json

echo ""
echo "================================================================================"
echo "STAGED DYNAMIC REGISTERS STUDY PIPELINE COMPLETE!"
echo "================================================================================"
