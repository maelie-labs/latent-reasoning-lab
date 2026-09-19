# Official Benchmark & Concurrency Policy: Intra-Model Evaluation Framework

**Effective Date**: 2026-09-12  
**Core Scientific Thesis**: Continuous Latent Recurrence Lab evaluates **Trained Model vs. Untrained Base Model** (and within-model controls) independently for each model architecture. We do NOT evaluate across models (e.g., 1.7B vs. 2B) as competitors.

---

## 1. The Intra-Model Evaluation Framework
For every model family investigated ($M \in \{\text{Qwen3-1.7B}, \text{Qwen3.5-2B}, \text{Qwen3-4B}, \dots\}$), the experimental evaluation matrix isolates the treatment effect of continuous latent recurrence within that model:
1. **Arm 3 (Continuous Latent Recurrence)**: Trained adapter with $K$ latent unrolling passes.
2. **Arm 1 (Base Direct Baseline)**: Untrained base model with thinking disabled (`enable_thinking=False`).
3. **Arm 4 (Base Discrete CoT Baseline)**: Untrained base model with full unconstrained thinking (`enable_thinking=True`, up to 32k tokens).
4. **Arm 1b (Trained Direct Control)**: Direct SFT adapter trained on identical targets without CoT.
5. **Arm 2b (Trained Pause Control)**: Pause-token adapter trained with $K$ discrete pause tokens on identical targets.

The primary scientific metric is:
$$\Delta = \text{Accuracy}(\text{Arm 3}) - \text{Accuracy}(\text{Untrained Base Model})$$
and the paired difference against matched controls:
$$\Delta_{\text{controls}} = \text{Accuracy}(\text{Arm 3}) - \text{Accuracy}(\text{Arm 1b / Arm 2b})$$

---

## 2. Dynamic Model-Specific Concurrency Profiling & Locked Invariant
- **Model-Specific Profiling**: Each model architecture has different parameter sizes, KV cache growth rates, and compute requirements. We run `scripts/56_concurrency_smoke_test.py` to discover each model's throughput knee ($C_{\text{model}}^*$).
- **Intra-Model Concurrency Invariant**: Within that model's evaluation suite, **the exact same concurrency level ($C$) MUST be held constant across all compared arms** (Arm 1, Arm 1b, Arm 2b, Arm 3, Arm 4) so that differences in performance reflect true algorithmic gains rather than hardware scheduling or queue latency variance.

---

## 3. High-Throughput Execution Invariants
1. **Zero Sequential Loops**: No batch-size-1 loops for benchmark suites.
2. **SGLang Serving Arms**: Run via async concurrency at $C_{\text{model}}^*$ (e.g. `scripts/53_sglang_eval_arm1.py`, `scripts/57_sglang_eval_arm4.py`).
3. **Batched PyTorch Adapter Arms**: Run via vectorized batch inference ($B \ge 8\text{--}32$ via `scripts/55_batched_eval_arms.py`).
4. **RTX PRO 4500 (32GB) Static VRAM**: Always launched with `--mem-fraction-static 0.90`.

---

## 4. Training Data Curation & Step-Segmented Curriculum Protocol
1. **Filtering Requirements**:
   - `math_verify` correctness required; drop truncated traces and any lacking `\boxed{}` in the answer.
   - Traces exceeding 4,096 tokens total are **dropped** rather than truncated to avoid incomplete derivations.
   - Deduplication by problem ID; disjointness proof recorded in `data/kept_trace_ids.json` (0 test-set overlap).
   - Exactly 100 traces held out in `data/curated_dev_traces_*.jsonl` for validation loss monitoring.
2. **Three-Field Architecture**:
   - `prompt`: exact evaluation prompt with chat template ending in `<think>\n`.
   - `think`: full trajectory between `<think>` and `</think>`.
   - `answer`: solution text after `</think>\n\n`, ending with `<|im_end|>`.
3. **Step Segmentation Rule**:
   - Segment `think` on double newlines (`\n\n`) and discourse transition markers (`Wait`, `So`, `Therefore`, `Now`, `Next`, `First`, `Then`, `Let's`, `Alternatively`, `Step N:`).
   - Stage 1 replaces first $\lfloor S/2 \rfloor$ steps with $\lfloor K/2 \rfloor$ latents; predicts remaining steps + `\n</think>\n\n` + answer.
   - Stage 2 replaces all $S$ steps with $K$ latents; predicts `\n</think>\n\n` + answer.
4. **Target Sequences & Transition Invariant**:
   - `\n</think>\n\n` transition tokens are strictly included in the loss target for all arms.
   - Arm 1b: `prompt` + `\n</think>\n\n` + `answer`. Loss on transition + answer.
   - Arm 2b: `prompt` + $K$ `<pause>` tokens + `\n</think>\n\n` + `answer`. Loss on transition + answer.
   - No loss on prompt or latent positions; latents never attend to replaced think tokens.
5. **Experimental Parity**:
   - All three arms train on identical example list, same order, same seed (`42`), and same learning schedule.
   - Save `arm_training_meta.json` logging `math_verify` version, sampling config, kept IDs, segmentation rule, step counts, and $\alpha$.

---

## 5. "One Suite, One Harness, Every Model" Invariant

To maintain pristine scientific comparability across diverse model architectures (e.g., pure attention vs. hybrid linear attention, varying parameter scales from 1.7B to 27B), we enforce a strict separation between universal constants and model-specific variables:

### Universal Fixed Constants (Never Change Across Models)
1. **The Benchmark Suite**: Evaluated on `data/benchmark_suite_250.json` (150 GSM8K + 100 MATH-500 stratified Levels 1–5, 20 per level), with full baseline floor/ceiling runs on MATH-500 and GSM8K.
2. **The Experimental Arms**:
   - **Arm 0**: Surgical Neutrality on IFEval / MMLU-Pro (checks base capability preservation before interpreting downstream results).
   - **Arm 1**: Untrained Base Direct (`enable_thinking=False`).
   - **Arm 1b**: Trained Direct No-CoT Control.
   - **Arm 2**: Minimal Discrete Thinking Tokens ($K$ discrete tokens; near-FLOP-matched discrete comparator).
   - **Arm 2b**: Trained Pause-Token Control ($K$ `<pause>` tokens).
   - **Arm 3**: Continuous Latent Recurrence (Fixed $K$ unrolled latent states; instrumented to log $P(\text{</think>})$, $\|h_k\|$, and cosine drift per step).
   - **Arm 3a**: Adaptive Halting Recurrence ($K_{\min} \le K \le K_{\max}$ based on calibrated stop threshold; evaluated on the fixed-$K$ Pareto frontier).
   - **Arm 4**: Untrained Base Unconstrained CoT (`enable_thinking=True`).
   - *(Arms 5a wall-clock-matched discrete CoT and Arm 6 CODI baseline defer to breadth stage).*
3. **Deterministic Evaluation Seeds**: `SEEDS = [42, 123, 456, 789]`.
4. **Grading & Verification**: Formal symbolic verification via `math_verify` requiring exact boxed match or mathematical equivalence.
5. **Token Budgets & Strict Truncation Protocol**:
   - Thinking budget: 32,768 tokens (for Arm 4 unconstrained thinking).
   - Answer budget: 8,192 tokens (identical across all arms).
   - **Scoring Rule**: Truncations hitting generation ceilings without reaching `<|im_end|>` or `</think>` are **strictly scored as incorrect (`is_correct = False`)**, NEVER dropped. Dropping truncated evaluations removes the hardest problems and severely biases comparisons. (The 4,096-token cap with drop rule applies strictly to training data curation to fit GPU batches, never to benchmark evaluation).
6. **Telemetry Columns & The Seven Timing Pillars**:
   - Telemetry tracks: Pass@1 Accuracy (%), Latency per query (mean/median ms), Peak VRAM Allocation (MiB), KV Cache Memory per Token (KiB), Recurrent State Size (MiB), Serving Throughput (tok/s), Truncation Rate (%).
   - **The Seven Timing Pillars**: (1) Card (physical GPU / bus ID), (2) Concurrency ($C_{\text{model}}^*$), (3) Static memory fraction (`--mem-fraction-static 0.90`), (4) Engine version (SGLang/PyTorch/Transformers), (5) Prefix-cache regime (`--disable-radix-cache`), (6) Warmup exclusion (first $N$ queries omitted from latency percentiles), (7) Retraction tracking.
7. **Statistical Inference & Delta Definitions**:
   - $\Delta_1 = \text{Arm 3} - \text{Arm 1b}$ (algorithmic gain of latent recurrence over direct SFT).
   - $\Delta_2 = \text{Arm 3} - \text{Arm 2b}$ (algorithmic gain of latent recurrence over pause tokens).
   - Discrete comparator check: $\text{Arm 3} - \text{Arm 2}$ (evaluates whether continuous latent states beat discrete tokens at matched step budget $K$).
   - Paired hierarchical bootstrapping ($B=10,000$ iterations) with 95% confidence intervals.

### Allowed Model-Specific Variables (Only These Four Vary)
1. **Official Sampling Configuration**:
   - `Qwen/Qwen3-1.7B`: `temperature = 0.6`, `top_p = 0.95`.
   - `Qwen/Qwen3.5-2B`: Model card benchmark settings are `temperature = 1.0`, `top_p = 0.95`, `top_k = 20`, `presence_penalty = 1.5` for **both** modes (used for calibration against 51.6% GPQA and 61.2% IFEval); `temp = 1.0, top_p = 1.0, pp = 2.0` is the general non-thinking usage recommendation. Hold sampling strictly identical across arms and report explicitly.
2. **Upstream Calibration Anchor Target**: The official benchmark used to verify the harness matches published numbers (e.g. MATH-500 for Qwen3-1.7B; GPQA Diamond 51.6% / IFEval 61.2% for Qwen3.5-2B).
3. **Self-Generated Curriculum**: Traces generated strictly from train splits by the model under study. Traces are never transferred or shared across models.
4. **Architecture-Specific Hooks**:
   - Model-calibrated scale factor $\alpha_{\text{model}} = \mathbb{E}[\|W_E\|] / \mathbb{E}[\|h_0\|]$.
   - Positional Latent Embeddings (PLE).
   - State/Cache management (Transformer static/dynamic KV cache vs. hybrid Gated DeltaNet state).


