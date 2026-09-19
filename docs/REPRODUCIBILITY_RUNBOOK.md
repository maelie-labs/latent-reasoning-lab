# Continuous Latent Recurrence Lab: End-to-End Reproducibility Runbook

**Purpose**: This runbook provides the definitive, self-contained, reproducible execution guide for the Continuous Latent Recurrence project. Any researcher or agent starting with **zero prior context** can execute the commands below in sequence to reproduce every phase, dataset, trained adapter, and benchmark result identically.

---

## 1. System Architecture & Hardware Allocation

| Physical Device | Role | VRAM | Configuration |
| :--- | :--- | :---: | :--- |
| **GPU 1 (`cuda:1`, RTX PRO 4500 Blackwell)** | **Primary Dedicated Compute** | **32 GB** | SGLang high-throughput serving (`--mem-fraction-static 0.90`), trace generation, baseline benchmarks. |
| **GPU 0 (`cuda:0`, RTX 4080 Ada)** | **Host Desktop / Interactive** | **16 GB** | Desktop compositor (~3.5 GB), LoRA adapter training with gradient checkpointing (peak VRAM ~7.6 GB). |

### Environment Setup
```bash
# Repository Root

# Dedicated Python Virtualenv (PyTorch 2.5+, Transformers, PEFT, math_verify)
source .venv/bin/activate

# SGLang Serving Environment (GPU 1)
# SGLang binary located at: sglang
```

---

## 2. Experimental Invariant: "One Suite, One Harness, Every Model"

To guarantee that cross-model synthesis tables remain directly comparable, interpretable, and reproducible:

| Component | Policy | Details |
| :--- | :---: | :--- |
| **Benchmark Suite** | **Fixed** | Always `data/benchmark_suite_250.json` (150 GSM8K + 100 MATH-500 L3–L5). |
| **Baseline Floor/Ceiling** | **Fixed** | Always MATH-500 & GSM8K in thinking (Arm 4 ceiling) and non-thinking (Arm 1 floor) modes. |
| **Experimental Arms** | **Fixed** | **Arm 0** (Neutrality), **Arm 1** (Base Direct), **Arm 1b** (Trained Direct), **Arm 2** (Minimal Discrete $K$), **Arm 2b** (Trained Pause), **Arm 3** (Fixed-$K$ Recurrent with $P(\text{</think>})$ logging), **Arm 3a** (Adaptive Halting Pareto), **Arm 4** (Base CoT). *(Arm 5a wall-clock & Arm 6 CODI deferred).* |
| **Seeds & Determinism** | **Fixed** | Always `SEEDS = [42, 123, 456, 789]`. |
| **Verifier & Grader** | **Fixed** | Always `math_verify` with canonical `\boxed{}` extraction and symbolic check. |
| **Token Budgets & Scoring** | **Fixed** | Thinking budget: 32,768 tokens; Answer budget: 8,192 tokens. **Truncations scored strictly as wrong (`is_correct = False`), NEVER dropped.** (The 4k-token drop rule applies strictly to training data curation). |
| **Telemetry & 7 Pillars** | **Fixed** | Pass@1 (%), Latency ms, Peak VRAM MiB, KV Cache KiB/tok, Recurrent State MiB, Throughput tok/s, Truncation %. Logged with **Seven Timing Pillars**: Card, Concurrency $C^*$, Mem Fraction 0.90, Engine Version, Prefix-Cache Regime, Warmup Exclusion, Retraction Tracking. |
| **Statistical Analysis** | **Fixed** | Paired bootstrap analysis ($B=10,000$ iterations) defining: $\Delta_1 = \text{Arm 3} - \text{Arm 1b}$ (vs direct), $\Delta_2 = \text{Arm 3} - \text{Arm 2b}$ (vs pause), and $\text{Arm 3} - \text{Arm 2}$ (vs discrete FLOP-matched). |
| **Sampling Config** | *Model-Specific* | Official upstream benchmark hyperparameters (e.g. Qwen 3: `0.6/0.95`; Qwen 3.5: `temp=1.0, top_p=0.95, top_k=20, pp=1.5` for both modes for 51.6/61.2 calibration). |
| **Calibration Anchor** | *Model-Specific* | Upstream published target (e.g. MATH-500 for 1.7B; GPQA Diamond 51.6% & IFEval 61.2% for 2B). |
| **Training Curriculum** | *Model-Specific* | Self-generated traces from train splits; zero cross-model distillation. |
| **Architecture Hooks** | *Model-Specific* | $\alpha = \mathbb{E}[\|W_E\|] / \mathbb{E}[\|h_0\|]$, PLE, attention vs. linear cache state handling. |

---

## 3. Phase 0: SGLang High-Throughput Serving on GPU 1

Always launch SGLang with `--mem-fraction-static 0.90` on GPU 1. This dedicates 90% (~30.2 GB) to model weights and dynamic KV cache pool, leaving a 2.4 GB buffer for CUDA runtime:

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID sglang serve \
  --model-path Qwen/Qwen3-1.7B \
  --host 127.0.0.1 \
  --port 30000 \
  --dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --disable-radix-cache \
  --disable-decode-cuda-graph \
  --disable-prefill-cuda-graph \
  --trust-remote-code
```
*Health Check*: `curl http://127.0.0.1:30000/v1/models` returns model JSON.

---

## 4. Concurrency Smoke Test & Calibration (Mandatory for Every New Model)

Before running benchmarks or trace generation on any new model architecture, execute a 30–60s concurrency sweep:

```bash
.venv/bin/python scripts/56_concurrency_smoke_test.py \
  --server_url http://127.0.0.1:30000 \
  --concurrency_grid "8,16,24,32,48,64" \
  --requests_per_level 16 \
  --mode both \
  --register_suite "benchmark_suite_250"
```
*Outputs*: Optimal concurrency knee ($C_{\text{model}}^*$) logged to `data/concurrency_policy_registry.json`.  
*Policy Rule*: All compared arms within this model must be evaluated at this exact locked concurrency setting.

---

## 5. Phase 1: Self-Distillation Trace Generation (GPU 1)

Generates 3,000 candidate reasoning traces (1,500 GSM8K + 1,500 MATH) from disjoint `train` splits using SGLang async concurrency:

```bash
.venv/bin/python scripts/54_generate_training_traces_sglang.py \
  --server_url http://127.0.0.1:30000 \
  --model_id Qwen/Qwen3-1.7B \
  --concurrency 32 \
  --target_per_set 1500
```
*Outputs*: `data/self_distill_train_traces_qwen3_1.7b.jsonl` (contains >2,000 verified correct traces).

---

## 6. Phase 2: Data Curation, Filtering, and Step Segmentation

Applies the rigorous methodology checklist:
1. Filters via `math_verify` and strictly drops truncated traces and any missing `\boxed{}`.
2. Hard 4,096-token cap: **DROPS** any sequence exceeding 4k tokens rather than truncating.
3. Records `data/kept_trace_ids.json` and asserts **0 overlap** with test benchmarks.
4. Holds out exactly 100 traces in `data/curated_dev_traces_*.jsonl` for validation loss monitoring.
5. Decomposes `think` into discrete reasoning steps via paragraph breaks (`\n\n`) and discourse markers.

```bash
.venv/bin/python scripts/58_curate_training_dataset.py \
  --raw_traces data/self_distill_train_traces_qwen3_1.7b.jsonl \
  --model_id Qwen/Qwen3-1.7B \
  --max_total_tokens 4096 \
  --dev_size 100
```
*Outputs*:
- `data/curated_train_traces_qwen_qwen3-1.7b.jsonl` (training pool)
- `data/curated_dev_traces_qwen_qwen3-1.7b.jsonl` (100 dev samples)
- `data/kept_trace_ids.json` (disjointness evidence)
- `data/ablation_subsets_manifest.json` (stratified 500, 1000, full subsets)

---

## 7. Phase 3: LoRA Training on GPU 0 (RTX 4080)

Trains experimental arms on identical data, ordering, and schedule with gradient checkpointing and batched latent unrolling.
*Timing Calibration*: ~2,000 examples × 2 epochs at effective batch 8 is ~500 steps per stage. With 4k-token sequences and gradient checkpointing on the shared RTX 4080, steps take seconds:
- Arm 1b ($K=0$): ~30–40 minutes
- Arm 2b ($K=6$): ~30–40 minutes
- Arm 3 ($K=6$): ~40–50 minutes
- Arm 2b ($K=32$): ~45–60 minutes
- Arm 3 ($K=32$): ~60–80 minutes

*Pipeline Synchronization*: GPU 1 (RTX PRO 4500) executes all base evaluations (Arms 1, 4, 2-6, 2-32) concurrently over SGLang while GPU 0 trains Arm 1b, guaranteeing zero GPU 1 idle time.

### Step 1: Decision Gate 1 Training ($K=6$ and $K=32$)
```bash
# Arm 1b: Direct SFT / No-CoT Control (K=0) [~35 min]
.venv/bin/python scripts/41_train_qwen3_arms.py --arm arm1b --model_id Qwen/Qwen3-1.7B --device cuda:0

# Arm 2b: Pause-Token Control (K=6 and K=32)
.venv/bin/python scripts/41_train_qwen3_arms.py --arm arm2b --model_id Qwen/Qwen3-1.7B --k_tokens 6 --device cuda:0
.venv/bin/python scripts/41_train_qwen3_arms.py --arm arm2b --model_id Qwen/Qwen3-1.7B --k_tokens 32 --device cuda:0

# Arm 3: Continuous Latent Recurrence (K=6 and K=32)
.venv/bin/python scripts/41_train_qwen3_arms.py --arm arm3 --model_id Qwen/Qwen3-1.7B --k_tokens 6 --device cuda:0
.venv/bin/python scripts/41_train_qwen3_arms.py --arm arm3 --model_id Qwen/Qwen3-1.7B --k_tokens 32 --device cuda:0
```

### Step 2: Empirical $\alpha_{128}$ Stability & Stop Probability Sweep
While Gate 1 evaluations run on GPU 1, execute the horizon $K=128$ stability check on the $K=6$ adapter on GPU 0:
```bash
.venv/bin/python scripts/36_stress_test_alpha.py \
  --model_id Qwen/Qwen3-1.7B \
  --adapter_path checkpoints/lora_arm3_qwen_qwen3-1.7b_k6_selfdistill \
  --max_k 128 \
  --device cuda:0
```
*Telemetry Output*: Logs $P(\text{</think>})$, residual norm trajectory $\|h_k\|$, and cosine drift out to $k=128$ to verify bounded representation and measure Lyapunov sensitivity growth.

### Step 3: Deep Reasoning Training ($K=128$): CONTINGENCY ONLY
> [!IMPORTANT]
> **Contingency-Only Policy**: Training at $K=128$ is **NOT** part of the standard execution path. We test zero-shot depth extrapolation out to $K=128$ using the trained $K=6$ and $K=32$ adapters first.
> 
> Training at $K=128$ is executed **strictly as a contingency** if:
> 1. Zero-shot unrolling of the $K=6$ or $K=32$ adapters out to $K=128$ shows instability, expansiveness ($\lambda > 0$), or representational drift, **AND**
> 2. Gate 1 ($K=6$ or $K=32$) has successfully cleared both controls ($\Delta_1 > 0$ and $\Delta_2 > 0$, $p < 0.05$).
> 
> If zero-shot unrolling is stable/contractive or if Gate 1 does not clear the controls, $K=128$ training is omitted entirely to avoid unnecessary compute expenditure.
```bash
# [CONTINGENCY ONLY] Arm 2b: Pause-Token Control (K=128)
.venv/bin/python scripts/41_train_qwen3_arms.py --arm arm2b --model_id Qwen/Qwen3-1.7B --k_tokens 128 --device cuda:0

# [CONTINGENCY ONLY] Arm 3: Continuous Latent Recurrence (K=128)
.venv/bin/python scripts/41_train_qwen3_arms.py --arm arm3 --model_id Qwen/Qwen3-1.7B --k_tokens 128 --device cuda:0
```

---

## 8. Phase 4: High-Throughput Base Evaluations on SGLang (GPU 1)

While GPU 0 trains Arm 1b, execute all base-model arms on GPU 1 over SGLang (`--mem-fraction-static 0.90`):

### Arm 1: Base Direct (enable_thinking=False, 250 problems x 4 seeds = 1,000 runs)
```bash
.venv/bin/python scripts/53_sglang_eval_arm1.py \
  --server_url http://127.0.0.1:30000 \
  --concurrency 48
```
*Runtime*: ~2.5 minutes.

### Arm 4: Unconstrained Discrete CoT (enable_thinking=True, up to 32k tokens)
```bash
.venv/bin/python scripts/57_sglang_eval_arm4.py \
  --server_url http://127.0.0.1:30000 \
  --concurrency 32
```
*Runtime*: ~12–15 minutes.

### Arm 2: Minimal Discrete Thinking Tokens ($K=6$ and $K=32$)
Near-FLOP-matched discrete comparator. Generates $K$ discrete thinking tokens, appends `\n</think>\n\n`, then generates answer up to 8192 tokens:
```bash
.venv/bin/python scripts/60_sglang_eval_arm2.py --k_tokens 6 --concurrency 32
.venv/bin/python scripts/60_sglang_eval_arm2.py --k_tokens 32 --concurrency 32
```
*Runtime*: ~3–5 minutes per horizon.

### Baseline Calibration Anchor (MATH-500 N=500 + GSM8K N=1,319)
```bash
.venv/bin/python scripts/40_sglang_full_baseline_calibration.py \
  --server_url http://127.0.0.1:30000 \
  --model_id Qwen/Qwen3-1.7B \
  --concurrency 32
```
*Teardown*: Once base arms finish, terminate SGLang to free physical VRAM on GPU 1: `pkill -f "sglang serve"`.

---

## 9. Phase 5: Adapter Neutrality, Spot-Check, and Decision Gate Evaluation (GPU 1)

Executed on GPU 1 in dedicated uncompressed `bfloat16` PyTorch:

### Step 1: Arm 0 Surgical Neutrality Check (MMLU / IFEval Slice at $K=0$)
Runs right after each adapter is trained, ahead of benchmark evaluation (~5 min per adapter):
```bash
.venv/bin/python scripts/39_surgical_neutrality_check.py \
  --model_id Qwen/Qwen3-1.7B \
  --lora_arm1b_path checkpoints/lora_arm1b_qwen_qwen3-1.7b_k0 \
  --lora_arm3_path checkpoints/lora_arm3_qwen_qwen3-1.7b_k6 \
  --device cuda:1
```
*Requirement*: Non-inferiority bound $\Delta(\text{Retrofitted} - \text{Base}) \ge -1.5\%$ on general knowledge / instruction following.

### Step 2: Extractor Spot-Check (20 Arm 1b Outputs Read by Hand)
Before computing paired statistics, inspect 20 Arm 1b outputs against `math_verify` to confirm output formatting:
```bash
.venv/bin/python scripts/61_extractor_spot_check.py \
  --adapter_path checkpoints/lora_arm1b_qwen_qwen3-1.7b_k0 \
  --device cuda:1 \
  --num_samples 20
```

### Step 3: Batched Benchmark Suite 250 Evaluation
Evaluates adapter arms across the 250-problem suite with 4 seeds (`SEEDS = [42, 123, 456, 789]`, 1,000 runs per arm) using batched PyTorch ($B=16$ on `cuda:1`):
```bash
.venv/bin/python scripts/55_batched_eval_arms.py \
  --arm all \
  --model_id Qwen/Qwen3-1.7B \
  --k_tokens 6 \
  --batch_size 16 \
  --device cuda:1

.venv/bin/python scripts/55_batched_eval_arms.py \
  --arm all \
  --model_id Qwen/Qwen3-1.7B \
  --k_tokens 32 \
  --batch_size 16 \
  --device cuda:1
```

---

## 10. Phase 6: Decision Gate 1 Statistical Analysis & Cold-Timing Pass

### Decision Gate 1 (Accuracy-Only)
Computes $B=10,000$ paired hierarchical bootstrap differences across matched problems and seeds:
$$\Delta_1 = \text{Arm 3} - \text{Arm 1b}, \quad \Delta_2 = \text{Arm 3} - \text{Arm 2b}$$
- **Gate Criteria**:
  1. $\Delta_1 > 0$ and $p < 0.05$ (Paired gains over direct SFT).
  2. $\Delta_2 > 0$ and $p < 0.05$ (Paired gains over discrete pause tokens).
  3. Discrete comparator check: $\Delta_{\text{FLOP}} = \text{Arm 3} - \text{Arm 2}$.
*Runtime Note*: Gate 1 uses accuracy only (unbiased by cross-runtime serving differences).

### Dedicated Cold-Timing Pass (Unified Runtime)
True wall-clock speedups and throughput comparisons across all arms are benchmarked in a single, unified PyTorch/SGLang runtime pass under identical batching and warmup exclusion.

---

## 11. Phase 6b: Statistical Telemetry Suite & Offline Analysis (CPU Only)

All analysis, bootstrapping, stratification tables, compute trade-off metrics, and publication figure generation execute strictly on **CPU** with zero GPU contention (`CUDA_VISIBLE_DEVICES=""`):

### Step 1: Paired Hierarchical Bootstrapping ($B=10,000$ Problem Clusters)
```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/analysis/paired_bootstrap.py \
  --file_a data/streaming_arm4_thinking_qwen_qwen3-1.7b.jsonl \
  --file_b data/streaming_arm2_minimal_discrete_qwen_qwen3-1.7b_k32.jsonl \
  --name_a "Arm 4 (Ceiling CoT)" \
  --name_b "Arm 2 (Discrete K=32)" \
  --n_bootstrap 10000 \
  --output_json data/analysis_paired_bootstrap_results.json
```

### Step 2: Per-Stratum Telemetry & Stratification Tables
```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/analysis/generate_strata_tables.py \
  --eval_files "Arm4_Ceiling=data/streaming_arm4_thinking_qwen_qwen3-1.7b.jsonl" \
               "Arm2_K32=data/streaming_arm2_minimal_discrete_qwen_qwen3-1.7b_k32.jsonl" \
               "Arm2_K6=data/streaming_arm2_minimal_discrete_qwen_qwen3-1.7b_k6.jsonl" \
  --output_md data/strata_telemetry_table.md \
  --output_tex data/strata_telemetry_table.tex \
  --output_json data/strata_telemetry_results.json
```

### Step 3: Compute-to-Threshold & Resource Trade-Offs
```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/analysis/compute_to_threshold.py \
  --arms "Arm4_Unconstrained=data/streaming_arm4_thinking_qwen_qwen3-1.7b.jsonl:0:0" \
         "Arm2_K32=data/streaming_arm2_minimal_discrete_qwen_qwen3-1.7b_k32.jsonl:32:0" \
         "Arm2_K6=data/streaming_arm2_minimal_discrete_qwen_qwen3-1.7b_k6.jsonl:6:0" \
  --output_md data/compute_threshold_table.md \
  --output_json data/compute_threshold_results.json
```

### Step 4: Publication Figure Generation
```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/analysis/plot_figures.py \
  --output_dir docs/figures
```

### Step 5: Reasoning Step-Count Histogram & Regime Extrapolation
```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/analysis/step_histogram.py \
  --train_traces data/curated_train_traces_qwen_qwen3-1.7b.jsonl \
  --dev_traces data/curated_dev_traces_qwen_qwen3-1.7b.jsonl \
  --output_json data/step_histogram_analysis_qwen3_1.7b.json \
  --output_memo data/step_histogram_findings_memo.md \
  --output_fig_dir docs/figures
```

---

## 12. Phase 7: Register-Bundle Ladder Architecture & Phase 0 Validation Suite

### 1. Goal & Architectural Focus
- **Primary Objective**: Demonstrate either:
  1. **Matched intelligence at $10\times\text{ to }15\times$ lower compute** by replacing verbose 500–1,000 token discrete CoT with ~20 discrete register tokens and 24 continuous latent passes (~44–50 total passes).
  2. **Higher intelligence at matched compute** by operating in continuous vector space ($\mathbb{R}^{2048}$), avoiding the irreversible 17-bit vocabulary collapse of discrete token generation and maintaining superposition across multi-constraint equations.
- **The Core Mechanistic Hypothesis**:
  $$\textbf{Headers act as semantic anchors in the KV cache that the answer attends to, with continuous latent bundles ($K=8$) computing the local non-verbal reasoning between them.}$$
  - Headers provide readable keys that pretrained query heads ($W_Q$) require to identify subgoals (remedying the $1.80\%$ attentional invisibility of raw latents).
  - Latent bundles ($K=8$ per bundle, 24 total) execute intermediate state transitions, constraint propagation, and non-verbal reasoning without generating syntactic English filler.

### 2. Prototype Training (GPU 1)
```bash
# Train prototype checkpoint with 3 rungs x 8 latents = 24 latents
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/94_train_register_bundle_ladder.py \
  --device cuda:1 \
  --batch_size 4 \
  --grad_accum 8 \
  --num_steps 8 \
  --k_per_bundle 8 \
  --num_rungs 3 \
  --mask_prompt_prob 0.5 \
  --output_dir checkpoints/test_register_ladder_quick
```

### 3. Phase 0 Prototype Validation Suite (GPU 1)
Executes four rigorous falsification checks before committing compute to production training:
```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/95_phase0_prototype_validation.py \
  --device cuda:1 \
  --checkpoint checkpoints/test_register_ladder_quick
```

#### Protocol & Decision Rules:
1. **Check A (Attention Split)**:
   - Decomposes recurrent-zone attention across all 28 layers on 60 MATH L3–5 problems under eager attention.
   - **Decision Rule**: Latent mass $\ge 10.0\% \to \textbf{PASS}$ (cleared empirically at $12.83\%$).
2. **Check C (4-Way Matched Inference Ablation & The Pause-Token Acid Test)**:
   - Evaluates: (1) Full Ladder, (2) Headers Only, (3) Latents Only, (4) Headers + 24 Pause Tokens.
   - **Decision Rule**:
     - If Cond 1 $\approx$ Cond 4: Latents are interchangeable delay lines; gain is 100% text-subgoal scaffolding (Arm 2c). **HALT latent training**.
     - Only if Cond 1 > Cond 4 AND Cond 1 > Cond 2: Continuous vector reasoning is confirmed load-bearing. **PROCEED to production training**.
3. **Check D (Multi-Bundle Sensitivity & Localized Damage)**:
   - Injects Gaussian noise into Bundle 1, 2, and 3 separately.
   - Requires divergence rate $\ge 20\%$ and localized step damage (corrupting Bundle 2 alters Step 2 while leaving Step 1 intact).
4. **Check B (Causal Activation Patching with Control 0 Verification)**:
   - Evaluates $N=94$ counterfactual donor pairs.
   - **Mandatory Prerequisite**: Control 0 self-patch identity must be $\ge 95.0\%$. If identity fails, harness bug is present-halt before interpreting $\Delta \text{Steer}$.

### 4. Production Training & Benchmark Evaluation (Completed & Certified)
```bash
# 1. Production Training on 1,367 curated traces (1,267 train / 100 dev) on GPU 1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/96_train_register_ladder_production.py \
  --device cuda:1 \
  --output_dir checkpoints/lora_arm3_register_ladder_production_qwen3_1.7b \
  --num_epochs 2 \
  --lr 1e-4 \
  --lambda_align 0.1

# 2. Benchmark Evaluation across 4 seeds (N=1,000 queries, seeds [42, 123, 456, 789]) on GPU 1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/97_eval_arm3_register_ladder.py \
  --checkpoint checkpoints/lora_arm3_register_ladder_production_qwen3_1.7b/best_checkpoint \
  --device cuda:1 \
  --batch_size 8 \
  --tag arm3_register_ladder_production_best \
  --max_ans_tokens 8192

# 3. Official Paired Hierarchical Bootstrapping (B=10,000 iterations)
python3 scripts/99_run_production_paired_bootstrap.py \
  --arm3_file data/streaming_arm3_register_ladder_production_best.jsonl \
  --arm1b_file data/streaming_arm1b_qwen_qwen3-1.7b.jsonl \
  --output_file data/paired_bootstrap_arm3_register_ladder_vs_arm1b.json

# 4. Production Acid Test Suite (Condition 1 vs Condition 4 on 60 MATH L3-5) on GPU 0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/98_acid_test_production_model.py \
  --checkpoint checkpoints/lora_arm3_register_ladder_production_qwen3_1.7b/best_checkpoint \
  --device cuda:0 \
  --output_file data/acid_test_production_model_results.json
```

#### Certified Milestone Results:
- **Benchmark Overall Pass@1**: **71.30%** (713 / 1,000) vs. Arm 1b (**71.20%**, $\Delta_1 = \mathbf{+0.10\%}$, $p=0.9698$ [**Statistical Null Parity**]).
- **MATH-500 Overall**: **74.00%** vs. Arm 1b (**71.50%**, $\Delta = \mathbf{+2.50\%}$, $p=0.2366$, Not Significant).
- **MATH Level 4**: **68.75%** vs. Arm 1b (**58.75%**, $\Delta = \mathbf{+10.00\%}$, 95% CI: `[+1.25%, +18.75%]`, $p = 0.0171$ [**Exploratory Sub-Slice**; 1 of 9 comparisons]). Across all hard math (L3–5), Arm 3-R (**65.00%**) is beaten by both trained discrete scaffolding controls (Arm 1b-R at **67.08%** and Arm 2b-R at **68.33%**).
- **Acid Test**: The previously reported $+3.33\%$ gain (45.00% vs. 41.67%) was measured against an *untrained* pause filler control. When evaluated against properly trained matched controls (Arm 1b-R at **73.20%** and Arm 2b-R at **72.90%**), continuous latents (Arm 3-R at **71.30%**) lose decisively: $\Delta = \mathbf{-1.90\%}$ overall and $\Delta = \mathbf{-3.33\%}$ on hard math. On the seed-42 $N=60$ hard math test, latents show a **$-10.00\%$ deficit** against trained pause tokens (Arm 3-R: 61.67% vs. Arm 2b-R: 71.67%).

---

### 5. Depth Extrapolation, Matched Controls, & Latent Information Bottleneck

```bash
# 1. Depth Extrapolation Scaling Study (K in [4, 6, 8, 12, 16] per bundle) on GPU 0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/99b_test_latent_depth_extrapolation.py \
  --checkpoint checkpoints/lora_arm3_register_ladder_production_qwen3_1.7b/best_checkpoint \
  --device cuda:0 \
  --output_file data/depth_extrapolation_study_qwen3_1.7b.json

# 2. Train Matched Discrete Controls (Arm 1b-R Headers Only & Arm 2b-R Pause Tokens) on GPU 0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/99_train_register_ladder_controls.py \
  --mode headers_only \
  --device cuda:0 \
  --output_dir checkpoints/lora_arm1b_register_ladder_headers_only_qwen3_1.7b

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/99_train_register_ladder_controls.py \
  --mode headers_pause \
  --device cuda:0 \
  --output_dir checkpoints/lora_arm2b_register_ladder_headers_pause_qwen3_1.7b

# 3. Evaluate Trained Controls on 60 MATH L3-5 Problems on GPU 0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/103_eval_trained_controls.py \
  --device cuda:0 \
  --output_file data/trained_controls_acid_test_results.json

# 4. Full 250-Suite Benchmark (N=1,000, 4 Seeds) for Arm 1b-R and Arm 2b-R on GPU 1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/105_eval_register_controls_250suite.py \
  --mode headers_only \
  --checkpoint checkpoints/lora_arm1b_register_ladder_headers_only_qwen3_1.7b/best_checkpoint \
  --device cuda:1 \
  --batch_size 16

CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/105_eval_register_controls_250suite.py \
  --mode headers_pause \
  --checkpoint checkpoints/lora_arm2b_register_ladder_headers_pause_qwen3_1.7b/best_checkpoint \
  --device cuda:1 \
  --batch_size 16

# 5. Multivariate Paired Bootstrap Matrix Analysis (B=10,000 Problem Clusters) on CPU
CUDA_VISIBLE_DEVICES="" python3 scripts/106_multivariate_bootstrap_analysis.py \
  --output_json data/multivariate_scaffolding_bootstrap_matrix.json

# 6. Latent Information Bottleneck (LIB) Ladder Training on GPU 1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/102_train_register_ladder_bottleneck.py \
  --device cuda:1 \
  --output_dir checkpoints/lora_arm3_register_ladder_bottleneck_qwen3_1.7b \
  --epochs 2 \
  --lr 1e-4

# 7. Evaluate LIB Model under Masked and Unmasked Conditions on GPU 0/1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/104_eval_register_ladder_bottleneck.py \
  --checkpoint checkpoints/lora_arm3_register_ladder_bottleneck_qwen3_1.7b/best_checkpoint \
  --device cuda:0 \
  --output_file data/eval_results_register_ladder_bottleneck_qwen3_1.7b.json

# 8. Full 250-Suite Benchmark of Missing Scaffolding Control (Arm 1-R) on GPU 1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/107_eval_untrained_base_headers_250suite.py \
  --device cuda:1 \
  --batch_size 16

# 9. Full 250-Suite Benchmark of Latent Information Bottleneck Model (Arm 3-LIB) on GPU 1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/97_eval_arm3_register_ladder.py \
  --checkpoint checkpoints/lora_arm3_register_ladder_bottleneck_qwen3_1.7b/best_checkpoint \
  --output_file data/eval_results_arm3_register_ladder_bottleneck_qwen3_1.7b.json \
  --streaming_file data/streaming_arm3_register_ladder_bottleneck_qwen3_1.7b.jsonl \
  --batch_size 8 \
  --device cuda:1

# 10. Complete 4-Part Acid Test Suite on Latent Information Bottleneck Model on GPU 0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/98_acid_test_production_model.py \
  --checkpoint checkpoints/lora_arm3_register_ladder_bottleneck_qwen3_1.7b/best_checkpoint \
  --device cuda:0 \
  --output_file data/acid_test_lib_bottleneck_results.json
```

#### Certified Scaffolding & LIB Milestone Results:
- **Arm 1-R (Untrained Base + Headers, $N=1,000$)**:
  - **Pass@1**: **70.20%** (GSM8K: 68.83%, MATH-500: 72.25%, Hard MATH: 63.33%).
  - **vs. Arm 1b-R (Trained Headers 73.20%)**: $\Delta = \mathbf{-3.00\%}$ ($p = 0.0230$).
  - **vs. Arm 1 (Clean Base Direct 73.90%)**: $\Delta = \mathbf{-3.70\%}$ (Hard MATH: $63.33\%$ vs. $70.00\%$).
  - *Mechanism*: Scaffolding is a learned repair for narrow SFT damage, not an intrinsic reasoning booster over the clean base floor.
- **Arm 3-LIB (Latent Information Bottleneck Full Benchmark, $N=1,000$)**:
  - **Pass@1**: **65.00%** (GSM8K: 65.33%, MATH-500: 64.50%, Hard MATH: 53.75%).
  - **vs. Arm 1b-R (73.20%)**: $\Delta = \mathbf{-8.20\%}$ ($p = 0.0000$, 95% CI `[-10.90%, -5.50%]`).
  - **vs. Arm 3-R Standard Ladder (71.30%)**: $\Delta = \mathbf{-6.30\%}$ ($p = 0.0000$, 95% CI `[-9.10%, -3.50%]`).
  - *Mechanism*: Compelling 125:1 compression into 24 latents during training causes severe general degradation across reasoning tasks.
- **LIB Acid Test Suite (Checks A, C, D, B)**:
  - **Check A (Attention Mass)**: Prompt: 62.34%, Headers: 14.37%, Latents: 3.21%, Transition: 11.00%. Latent attention remains $<5\%$.
  - **Check C (Matched Ablation on 60 MATH L3–5)**: Full Ladder **41.67%** vs. Headers Only **41.67%** ($\Delta = \mathbf{0.00\%}$).
  - **Check D (Gaussian Noise Sensitivity)**: Divergence: Bundle 1 **95.00%**, Bundle 2 **90.00%**, Bundle 3 **95.00%**.
  - **Check B (Causal Activation Patching)**: Control 0 Gate: **70/70 (100.00% Identity)**. Counterfactual Donor Steering ($N=70$): $P(\text{chance}) = \mathbf{2.86\%}$, $P(\text{steered}) = \mathbf{2.86\%}$, **Net Delta Steer: $\mathbf{+0.00\%}$ [INERT]**.

---

## 12. Study 2: Staged Dynamic Discrete Registers (`studies/dynamic_registers/`)

Reproduces discrete key-value register training (`<|reg|>k=v<|/reg|>`) and counterfactual donor corruption:

```bash
# 1. Prepare curriculum arms (Arm 1 Dynamic Registers, Ctrl 1 Headers-Only, Arm 2 Filler)
.venv/bin/python studies/dynamic_registers/scripts/prepare_curriculum_arms.py

# 2. Train LoRA adapters on GPU 1
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python studies/dynamic_registers/scripts/02_train_register_models.py --arm dynamic_registers
.venv/bin/python studies/dynamic_registers/scripts/02_train_register_models.py --arm headers_only
.venv/bin/python studies/dynamic_registers/scripts/02_train_register_models.py --arm matched_filler

# 3. Evaluate 250 suite x 4 seeds (N=1,000 queries per arm)
.venv/bin/python studies/dynamic_registers/scripts/03_eval_register_models.py --arm dynamic_registers --batch_size 16
.venv/bin/python studies/dynamic_registers/scripts/03_eval_register_models.py --arm headers_only --batch_size 16
.venv/bin/python studies/dynamic_registers/scripts/03_eval_register_models.py --arm matched_filler --batch_size 16

# 4. Paired Bootstrapping and Counterfactual Swap Audit
.venv/bin/python studies/dynamic_registers/scripts/04_paired_bootstrap_registers.py
.venv/bin/python studies/dynamic_registers/scripts/05_causal_register_corruption.py
```

---

## 13. Study 3: Telegraphic Propositional CoT Scaling (`studies/telegraphic_cot/`)

Reproduces dense 4B vs. hybrid Gated DeltaNet scaling on compressed discrete propositions:

```bash
# 1. Generate telegraphic curriculum from teacher traces
.venv/bin/python studies/telegraphic_cot/scripts/01_distill_telegraphic_curriculum.py \
  --model_id "Qwen/Qwen3-4B"
.venv/bin/python studies/telegraphic_cot/scripts/01_distill_telegraphic_curriculum.py \
  --model_id "Qwen/Qwen3.5-4B"

# 2. Train LoRA adapters on GPU 1 (CUDA_VISIBLE_DEVICES=1)
.venv/bin/python studies/telegraphic_cot/scripts/02_train_telegraphic_models.py \
  --model_id "Qwen/Qwen3-4B" --arm headers_only
.venv/bin/python studies/telegraphic_cot/scripts/02_train_telegraphic_models.py \
  --model_id "Qwen/Qwen3-4B" --arm telegraphic_cot
.venv/bin/python studies/telegraphic_cot/scripts/02_train_telegraphic_models.py \
  --model_id "Qwen/Qwen3.5-4B" --arm headers_only
.venv/bin/python studies/telegraphic_cot/scripts/02_train_telegraphic_models.py \
  --model_id "Qwen/Qwen3.5-4B" --arm telegraphic_cot

# 3. Evaluate 250 Suite x 4 seeds (N=1,000 queries per checkpoint)
bash scripts/124_run_study3_eval_all.sh

# 4. Evaluate Arm 0 IFEval Surgical Neutrality (541 prompts per checkpoint)
.venv/bin/python scripts/126_eval_ifeval_study3_4b.py \
  --model_id "Qwen/Qwen3-4B" \
  --checkpoint "checkpoints/lora_headers_only_qwen3_4b/best_checkpoint" \
  --tag "headers_only_qwen3_4b" \
  --output_file "data/ifeval_headers_only_qwen3_4b.json"
```

---

## 14. Study 4: Tool-Grounded In-Place Working Memory Scratchpad (`studies/tool_grounded_scratchpad/`)

Reproduces native tool-calling scratchpad curriculum, training, causal steering audit, and benchmark evaluation:

```bash
# 1. Curate Tool-Grounded Scratchpad Curriculum (1,267 train / 100 dev)
.venv/bin/python studies/tool_grounded_scratchpad/scripts/01_curate_tool_scratchpad_curriculum.py

# 2. Train In-Place Working Memory Scratchpad LoRA Adapter (GPU 1)
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python studies/tool_grounded_scratchpad/scripts/02_train_tool_scratchpad.py \
  --model_id "Qwen/Qwen3-4B" \
  --train_file "studies/tool_grounded_scratchpad/data/train_tool_scratchpad.jsonl" \
  --dev_file "studies/tool_grounded_scratchpad/data/dev_tool_scratchpad.jsonl" \
  --output_dir "checkpoints/lora_tool_scratchpad_qwen3_4b" \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 1e-4

# 3. Gate 2 Counterfactual Patching & Steering Audit (N=100)
.venv/bin/python studies/tool_grounded_scratchpad/scripts/03_gate2_patching_scratchpad.py \
  --model_id "Qwen/Qwen3-4B" \
  --checkpoint "checkpoints/lora_tool_scratchpad_qwen3_4b/best_checkpoint"

# 4. Multi-Turn Batched Benchmark Evaluation (250 Suite x 4 seeds = 1,000 queries)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
.venv/bin/python studies/tool_grounded_scratchpad/scripts/04_eval_tool_scratchpad.py \
  --model_id "Qwen/Qwen3-4B" \
  --checkpoint "checkpoints/lora_tool_scratchpad_qwen3_4b/best_checkpoint" \
  --device "cuda:0" \
  --batch_size 8

# 5. Post-Evaluation Pipeline (Paired Bootstrapping B=10,000 + Arm 0 IFEval)
bash studies/tool_grounded_scratchpad/scripts/06_post_eval_pipeline.sh
```



