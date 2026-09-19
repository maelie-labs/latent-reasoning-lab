# Post-Gate Follow-Up Investigations & Scientific Checks (Non-Model Ladder)

This document formalizes all follow-up investigations, mechanistic probes, and robustness checks planned for Continuous Latent Recurrence once the primary Decision Gate 1 (Fixed $K=6, 32$ on the 250-suite) is certified. 

This scope explicitly excludes the multi-model architecture scaling ladder (e.g. Qwen3.5-2B, Gemma 4, 9B), focusing exclusively on algorithmic depth, mechanistic interpretability, and generalization checks.

---

## 0. Check 0: Dedicated Cold-Timing Baseline & Latency Grounding
* **Purpose**: Establish the authoritative latency and memory x-axis for Checks 5 and 6 under the Seven Timing Pillars.
* **Protocol**:
  - Dedicated single-seed run on physical compute GPU (`cuda:1`, NVIDIA RTX PRO 4500 32GB).
  - Static configuration: `--disable-radix-cache`, `--mem-fraction-static 0.90`, dedicated compute environment (0 compositor / desktop overhead).
  - Warmup exclusion: First $N=5$ queries discarded from timing percentiles.
  - Retraction tracking: Explicit assertion of 0 retractions / memory evictions.
* **Telemetry Pillars Measured**:
  1. Mean & median latency per query (ms).
  2. Sustained throughput (tokens/second).
  3. Incremental KV cache footprint per token (KiB/tok).
  4. Recurrent latent state footprint (MiB).
* **Execution Script**: `scripts/91_benchmark_cold_timing.py`.

---

## 1. Check 1: Recurrent Depth Extrapolation & Native Scaling Gap ($K \to 128$)
* **Scientific Question**: Does an adapter trained on a fixed horizon generalize out to deeper recurrent horizons ($K \in [6, 12, 16, 24, 32, 48, 64, 96, 128]$) at inference time without retraining?
* **Dual Adapter Protocol**:
  - Evaluate BOTH the $K=6$ adapter AND the $K=32$ adapter unrolled across the entire grid $K \in [6 \dots 128]$ on `data/benchmark_suite_250.json`.
  - **The Native vs. Extrapolated Gap**: At shared budget points (specifically $K=32$), directly compare the performance of the extrapolated $K=6 \to 32$ adapter against the natively trained $K=32$ adapter.
* **Trained-vs-Untrained Latent Sensitivity**:
  - Measure per-step latent amplification factor via norm ratio ($\|h_k\|_2 / \|h_{k-1}\|_2$) and finite-difference perturbation sensitivity (perturbing $h_{k-1}$ by a small $\epsilon$ with $\|\epsilon\|_2 = 10^{-3} \|h_{k-1}\|_2$, running one forward step, and measuring $\|\Delta h_k\|_2 / \|\epsilon\|_2$ at the cost of one extra forward pass) to directly quantify trained vs. untrained local contraction without expensive 28-layer Jacobian computation.
* **Telemetry to Measure**:
  1. Pass@1 accuracy curve as a function of $K$.
  2. Hidden state $L_2$ norm growth: $\|h_k\|_2$ per step $k \in [1 \dots K]$.
  3. Step-to-step cosine similarity: $\cos(h_k, h_{k-1})$ to quantify representational drift vs. convergence.
  4. Per-step $P(\text{</think>})$ trajectory across easy vs. hard problems.
* **Execution Script**: `scripts/86_eval_depth_extrapolation.py`.

---

## 2. Check 2: Scale Factor ($\alpha$) Stress Testing & Attractor Basin
* **Scientific Question**: How wide is the stable recurrent attractor basin around the calibrated empirical scale factor $\alpha$?
* **Background**:
  - Calibrated $\alpha = \mathbb{E}[\|W_E\|] / \mathbb{E}[\|h_0\|] = 0.011440$ on `Qwen3-1.7B`.
  - Legacy preliminary experiments ran with $\alpha_{\text{legacy}} \approx 3\alpha$ without collapsing, suggesting substantial stability margins (The Stability Margin Hypothesis).
* **Ablation Grid**:
  $$\alpha \in \{0.25\alpha, 0.50\alpha, 1.0\alpha\ (\text{calibrated}), 2.0\alpha, 3.0\alpha\ (\text{legacy}), 1.0\ (\text{unscaled}), \text{SpiralThinker RMS Norm}\}$$
* **SpiralThinker Prior Art Comparison (arXiv 2511.08983)**:
  - *Mechanism*: SpiralThinker applies per-vector target-RMS normalization derived from the embedding matrix ($h_{\text{norm}} = h / \text{RMS}(h) \cdot \text{RMS}(W_E)$).
  - *Distinction*: Unlike empirical $\alpha$ (which is a fixed scalar multiplier that preserves relative amplitude dynamics across steps), SpiralThinker clamps every vector to a fixed magnitude.
  - *Measurement*: Run side-by-side in Check 2 to empirically state the performance delta between relative-norm preservation ($\alpha$) and uniform per-vector RMS clamping.
* **Measurements (Inference-Only Sweep)**:
  1. NaN / degeneration rate (% of sequences producing NaN/Inf activations or repetitive collapse) out to $K=128$.
  2. Latent state $L_2$ norm trajectory $\|h_k\|_2$ and step-to-step cosine drift per pass.
  3. Downstream Pass@1 on stratified MATH-500 Levels 3–5 across the grid.
* **Execution Script**: `scripts/36_stress_test_alpha.py`.

---

## 2b. Check 2b: Training-Volume Ablation ($K=6$)
* **Scientific Question**: Does adapter performance saturate at current training volume, or is Continuous Latent Recurrence undertrained?
* **Protocol**:
  - Train Arm 3 ($K=6$) at fixed 2 epochs across stratified subsets of 500, 1,000, and 2,000 self-distillation traces.
  - Subsets drawn stratified by source (GSM8K vs. MATH) and MATH difficulty level, pre-registered in `data/ablation_subsets_manifest.json`.
  - The 2,000-trace point uses the primary verified training set (identical to the Decision Gate adapter; zero extra compute).
* **Measurements**:
  1. Dev loss convergence on $N=100$ held-out traces (`data/curated_dev_traces_qwen_qwen3-1.7b.jsonl`).
  2. Downstream Pass@1 on `data/benchmark_suite_250.json`.
  3. Scaling returns curve: accuracy delta per 1,000 curated traces.

---

## 3. Check 3: Causal Activation Patching & Semantic Donor Steering ($N \ge 100$)
* **Scientific Question**: Do intermediate continuous latent vectors $h_k$ encode authentic mathematical computation, or are they merely functioning as passive compute delays?
* **Experimental Controls & Pre-Registered Criteria**:
  1. **Control 0 (Corrected Null Patch / Mechanical Sanity)**:
     - Re-inject problem $i$'s own latent vector $h_t$ at recurrent step $t$.
     - **Pass Criterion**: Assert $\cos(h_{\text{post}}, h_{\text{clean}}) \ge 0.999$ on the post-patch latent state **AND** bit-identical generated answer text. *(Replaces legacy $10^{-3}$ logit delta which is invalid under bf16 precision).*
  2. **Control A (Norm-Matched Gaussian Noise)**:
     - Replace latent vector $h_t$ at fractional steps $t \in \{\lfloor 0.25K \rfloor, \lfloor 0.50K \rfloor, \lfloor 0.75K \rfloor\}$ with isotropic Gaussian noise scaled to $\|h_t\|_2$.
     - Measure damage to final answer accuracy (quantifying information destruction).
  3. **Control B (Filtered Shuffled-Problem Donor Steering)**:
     - **Donor Value Filtering**: Candidate donor problem $j$ must have an answer value $x \in V_{\text{donor}}^{\text{filtered}}$ satisfying:
       $$x > 20 \quad \text{and} \quad x \notin V_{\text{recipient}}^{\text{prompt}} \cup V_{\text{recipient}}^{\text{solution}}$$
     - **Empirical Chance Baseline ($P_{\text{chance}}$)**: Measured by scoring *unpatched* recipient model outputs against random donor answer sets.
     - Patch donor latent $h_t^{(j)}$ into recipient problem $i$'s latent loop at step $t$.
     - **Pre-Registered Steering Criterion**:
       $$\Delta \text{Steer} = P_{\text{steered}}(\text{donor answer}) - P_{\text{chance}} \ge +40\%$$
* **Execution Script**: `scripts/37_causal_patching_scaled.py`.

---

## 4. Check 4: Out-of-Band Latent Thought Stream Decoding (Observability)
* **Scientific Question**: Can intermediate latent states $h_k$ be translated into interpretable human-readable thought streams asynchronously without emitting discrete tokens into the autoregressive KV cache?
* **Methods**:
  1. **Direct Logit Lens (No Double-RMSNorm)**:
     - Because $h_k = \text{RMSNorm}(u_k)$ is already the normalized output of the final RMSNorm layer (which is scaled by $\alpha$ and fed back), the direct logit lens projects directly via the unembedding matrix without re-normalizing:
       $$\text{logits}_k = W_U \cdot h_k \quad (\text{or } \text{model.lm\_head}(h_k))$$
     - Log top-5 vocabulary tokens and token entropies per recurrent step.
  2. **Auxiliary Reasoning-Step Probe**:
     - Architecture: Lightweight 2-layer MLP probe ($d \to d \to |V|$).
     - **Loss & Training Target**: Trained to predict the next reasoning-step tokens from the teacher trace at the corresponding step position (cross-entropy loss).
* **First-Match & Stable-Match Telemetry (arXiv 2604.04902)**:
  - At each step $k \in [1 \dots K]$, log top-1 decoded token.
  - Record:
    * `first_match`: The earliest step $k$ where the top-1 decoded token matches the final parsed answer.
    * `stable_match`: The earliest step $k$ after which top-1 remains identical to the final answer for all subsequent passes.
  - Telemetry report: Log both per problem, partitioned by **correct vs. incorrect final answers**.
  - Scientific value: Measures premature commitment and provides an auxiliary halting indicator alongside $P(\text{</think>})$ with zero extra GPU compute.
* **The ThoughtSteer Caveat**:
  - *Observational Constraint*: Readouts from the logit lens or auxiliary probe are purely observational; semantic representation validity is grounded exclusively by causal intervention (Check 3), not by superficial human-perceived text coherence.
* **Execution Script**: `scripts/87_decode_thought_stream.py`.

---

## 5. Check 5: Dynamic Halting Pareto Frontier (Arm 3a vs. Mirrored CoT)
* **Scientific Question**: Does adaptive halting ($P(\text{</think>}) \ge \tau$) establish an accuracy-latency Pareto frontier strictly dominating both fixed-$K$ and early-stopped discrete CoT?
* **Axes & Experimental Setup**:
  - **X-Axis (Latency)**: Latency percentiles (mean/median ms) derived strictly from the **Check 0 Cold-Timing Baseline** under the Seven Timing Pillars.
  - **Y-Axis (Accuracy)**: Pass@1 on `data/benchmark_suite_250.json`.
* **Comparators on Identical Axes**:
  1. **Arm 3a**: Halting threshold sweep $\tau \in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]$ with $K_{\min}=2, K_{\max}=32$ (and $128$).
  2. **Fixed-$K$ Arm 3**: Fixed points $K \in \{6, 16, 32\}$.
  3. **Mirrored Discrete CoT Baseline (Arm 4 Early-Stopped)**: Arm 4 evaluated under early-stopping rules:
     - Fixed thinking token budgets: $B_{\text{think}} \in \{32, 64, 128, 256, 512\}$ tokens.
     - Answer-probability early exit threshold.
* **Execution Script**: `scripts/80_eval_arm3a_adaptive_halting.py`.

---

## 6. Check 6: Breadth Generalization & Boundary Stress Benchmarks
* **Scientific Question**: Does continuous latent recurrence transfer effectively to non-arithmetic reasoning domains requiring search, constraint satisfaction, perturbation invariance, and high-difficulty competition?
* **Benchmark Suite**:
  1. **GPQA Diamond (Headline Efficiency Anchor)**:
     - 198 problems $\times$ 8 samples = 1,584 queries under official thinking protocol.
     - Pre-registered non-inferiority bound against base thinking mode.
  2. **ProsQA (4-Hop Graph Search: Split Zero-Shot vs. In-Distribution)**:
     - **ProsQA Zero-Shot**: Evaluates existing math-trained adapters on synthetic logical graph traversal to test out-of-distribution reasoning transfer.
     - **ProsQA In-Distribution**: Evaluates ProsQA-trained adapters trained on graph traces under the exact same parity self-distillation recipe.
  3. **GSM-Symbolic (1,000 Problems)**:
     - Systematic variable, name, and numerical perturbations of arithmetic questions to test whether latent recurrence is robust against prompt phrasing perturbations compared to discrete CoT.
  4. **ZebraLogic (300 Problems)**:
     - Multi-variable grid constraint elimination (logic grid puzzles).
  5. **AIME 2024 + 2025 Bounding (60 Problems $\times$ 16 Seeds)**:
     - High-difficulty competition mathematics boundary check.
* **Execution Scripts**: `scripts/45_eval_gpqa.py`, `scripts/88_eval_breadth_prosqa.py`, `scripts/89_eval_breadth_zebra.py`, `scripts/90_eval_breadth_symbolic.py`.

---

## 7. Check 7: Quantization Stage: Static vs. Dynamic Activation Quantization
* **Source & Scientific Grounding**: *LoopQ: Quantization for Recursive Transformers* (arXiv 2605.16343).
* **Key Finding**: In looped/recurrent architectures, static post-training quantization (PTQ) activation clipping matrices accumulate error exponentially across loop transitions, causing representational divergence.
* **Experimental Protocol**:
  - For each target precision format (e.g. W8A8 FP8, NVFP4):
    1. Evaluate a **static-calibrated PTQ** variant (fixed activation scales derived from prefill).
    2. Evaluate a **dynamic per-token / per-step activation quantization** variant.
  - Profile out to $K=32$ unrolled steps, measuring:
    * Per-step hidden state norm $\|h_k\|_2$.
    * Step-to-step cosine drift $\cos(h_k, h_{k-1})$.
    * Survival rate (% of sequences that complete $K$ passes without divergence or NaN).
* **Objective**: Decisively explain the earlier FP8 collapse mechanism and certify which quantization recipe (static vs. dynamic activation scaling) is production-viable for deployment.

---

## 8. Check 8: Canonical v2 Training Plan Enhancements (Non-Study Adapters)
*Note: These enhancements apply strictly to future canonical v2 production adapters and do NOT alter the frozen v1 experimental study adapters.*
* **1. Optimizer State Reset at Curriculum Boundaries**:
  - *Source*: Deng et al., *From Explicit CoT to Implicit CoT* (arXiv 2405.14838, Sections 3 & 5.2).
  - *Mechanism*: AdamW maintains second-order gradient moments ($\beta_2$). Shifting the discrete token removal boundary induces sudden loss jumps that produce corrupted gradient estimates. Resetting the optimizer state ($\beta_1, \beta_2$ moments) at each curriculum stage transition avoids catastrophic optimization spikes.
* **2. Removal Smoothing (Exact Parameterization)**:
  - *Source*: Deng et al. (arXiv 2405.14838, Eq. 6).
  - *Mechanism*: Add a non-negative integer random offset $o \in \mathbb{Z}_{\ge 0}$ to the scheduled token removal count $s(t)$, yielding $s(t)^* = s(t) + o$, where $P(o) \propto \exp(-\lambda o)$. Parameterized by $\lambda = 4$, yielding $P(o=0) = 1 - e^{-4} \approx 98.17\%$ and $P(o \ge 1) = e^{-4} \approx 1.83\% \approx 2\%$. About 2% of training batches remove additional tokens ahead of schedule, preventing loss cliffs at stage transitions.
* **3. Random-$K$ Horizon Sampling**:
  - *Source*: arXiv 2606.27538.
  - *Mechanism*: Randomize unroll depth $K \sim \mathcal{U}[K_{\min}, K_{\max}]$ during Stage 2 training to prevent the adapter from overfitting to a single static step horizon.
* **4. General-Domain Instruction Mixing (IFEval Neutrality Preservation)**:
  - *Motivation*: Math-only self-distillation causes a domain specialization shift on conversational formatting prompts (Arm 1b drops from 67.65% to 59.70% strict), while deep recurrence exhibits $K$-dependent formatting drift (54.34% at $K=6 \to 50.28\%$ at $K=32$).
  - *Mechanism*: Mix 300–500 general-domain instruction-following examples (the model's own non-thinking answers to diverse IFEval-style constraint prompts, paired with latent unroll spans) directly into the curriculum alongside math traces.
  - *Objective*: Prevent representation drift on non-math formatting constraints and ensure $K$-independent surgical neutrality across all unroll depths.

---

## 9. Register Design Rationale Status (ResearchGate 408236268)
* **Status**: **UNVERIFIED: DO NOT CITE**.
* **Audit**: Attempted retrieval of ResearchGate publication 408236268 (*"Do Models Read What They Write?"*, July 2026) via automated web fetch and DOI registry (Crossref). Retrieval failed (HTTP 403 / unindexed).
* **Policy Compliance**: Per explicit directive (*"Do not cite until verified. If not, remove the reference"*), all citations and formal references to this paper are omitted from specification sections (§7 / §2.3) until independent verification and peer-indexed access are established.

---

## 10. Master Execution Matrix Summary

| Follow-Up Package | Core Metric / Target | Hardware Role | Est Runtime | Primary Script / Reference |
| :--- | :--- | :--- | :---: | :--- |
| **0. Cold-Timing Baseline** | Mean/med ms, tok/s, KiB/tok (Seven Pillars) | GPU 1 (RTX PRO 4500) | ~20 min | `scripts/91_benchmark_cold_timing.py` |
| **1. Depth Extrapolation ($K \le 128$)** | Pass@1 curve, native vs. extrapolated gap, amplification factor | GPU 1 (RTX PRO 4500) | ~50 min | `scripts/86_eval_depth_extrapolation.py` |
| **2. $\alpha$ Scale Factor Stress Test** | NaN rate, norm/cosine drift across $0.25\alpha \dots 3.0\alpha$ + SpiralThinker | GPU 1 (RTX PRO 4500) | ~30 min | `scripts/36_stress_test_alpha.py` |
| **2b. Training-Volume Ablation** | Dev loss & Pass@1 across 500, 1000, 2000 traces | GPU 0 (RTX 4080) | ~1.5 hours | `scripts/76_train_volume_ablation.py` |
| **3. Causal Activation Patching** | $\cos \ge 0.999$ (Ctrl 0); $\Delta \text{Steer} \ge +40\%$ (Ctrl B filtered) | GPU 0 or 1 | ~25 min | `scripts/37_causal_patching_scaled.py` |
| **4. Thought Stream Decoding** | First/stable-match telemetry; next reasoning token loss; $W_U \cdot h_k$ | GPU 0 (RTX 4080) | ~15 min | `scripts/87_decode_thought_stream.py` |
| **5. Arm 3a Pareto vs. Mirrored CoT** | Pareto dominance over fixed-$K$ & early-stopped Arm 4 | GPU 1 (RTX PRO 4500) | ~50 min | `scripts/80_eval_arm3a_adaptive_halting.py` |
| **6. Breadth Generalization** | GPQA Diamond, ProsQA (zero-shot & in-dist), Zebra, Symbolic | GPU 1 (RTX PRO 4500) | ~3.0 hours | `scripts/88_eval_breadth_*.py` |
| **7. Activation Quantization (LoopQ)** | Static vs. Dynamic FP8 / NVFP4 stability across loop transitions | GPU 1 (RTX PRO 4500) | ~45 min | arXiv 2605.16343 |
| **8. v2 Training Enhancements** | Optimizer reset, removal smoothing ($\lambda=4$), random-$K$, IF mixing | Training Phase | - | arXiv 2405.14838, 2606.27538 |
