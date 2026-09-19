# Model Results & Telemetry: `Qwen/Qwen3.5-2B`

**Architecture**: Hybrid Gated DeltaNet (Linear Attention) + Full Self-Attention (24 Layers: 18 DeltaNet + 6 Attention)  
**Calibrated Empirical Scale Factor**: $\alpha_{\text{model}} = \mathbf{0.925143}$ (Measured via `scripts/64_preflight_qwen3.5_2b.py`: $\mathbb{E}[\|W_E\|] = 0.675691$, $\mathbb{E}[\|h_0\|] = 0.730364$)  
**Memory Footprint**: $9.84\text{ MiB}$ fixed recurrent state + $12.0\text{ KiB/token}$ dynamic KV cache ($9.33\times$ lower KV growth than pure transformer)  
**Base Weights VRAM**: $3,589.3\text{ MiB}$ in pure `bfloat16`  
**Vocab Size**: 248,077 tokens  
**Calibrated Concurrency Knee**: $C_{\text{model}}^*$ *(To be determined via `scripts/56_concurrency_smoke_test.py`)*  
**Official Sampling Configuration**: `temperature = 1.0, top_p = 0.95, top_k = 20, presence_penalty = 1.5` for both thinking and direct modes  
**Official Published Reference Anchors**:
- GPQA Diamond: **51.6%** (Thinking)
- IFEval: **78.6%** (Thinking) / **61.2%** (Non-Thinking)

### Section 10 Pre-Flight Verification Audit
- **Default Mode**: Non-Thinking mode. Calling `apply_chat_template` without kwargs renders `<think>\n\n</think>\n\n` automatically.
- **Thinking Mode Activation**: Explicitly requires `enable_thinking=True` to emit `<think>\n`.
- **Pre-Flight Evidence**: [`data/preflight_qwen3.5_2b.json`](../data/preflight_qwen3.5_2b.json) (Verified 6 unrolled latent steps under `SDPBackend.MATH`, 0 errors, 0 NaNs).

---

## 1. Executive Telemetry & Comparison Matrix (250 Benchmark Suite)

> Evaluated on `data/benchmark_suite_250.json` (150 GSM8K + 100 MATH-500 stratified Levels 1–5, 20 per level) across 4 deterministic seeds: `SEEDS = [42, 123, 456, 789]` ($N=1,000$ total evaluations per arm).  
> Grader: Canonical `math_verify` with symbolic equivalence.  
> Truncation protocol: Any sequence exceeding token ceilings without `<|im_end|>` or `</think>` is strictly scored as incorrect (`is_correct = False`).

| Experimental Arm | Thinking Mode / Configuration | Pass@1 Accuracy (%) | Mean Latency (ms) | Median Latency (ms) | Peak VRAM (MiB) | KV Cache / Token | Throughput (tok/s) | Truncation Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 0** | Surgical Neutrality (IFEval / MMLU) | *Pending* | - | - | - | 12 KiB | - | - |
| **Arm 1** | Untrained Base Direct (Non-Thinking) | *Pending* | - | - | - | 12 KiB | - | - |
| **Arm 1b** | Trained Direct No-CoT Control ($K=0$) | *Pending* | - | - | - | 12 KiB | - | - |
| **Arm 2 ($K=6$)** | Base Discrete Tokens ($K=6$ discrete tokens) | *Pending* | - | - | - | 12 KiB | - | - |
| **Arm 2 ($K=32$)** | Base Discrete Tokens ($K=32$ discrete tokens) | *Pending* | - | - | - | 12 KiB | - | - |
| **Arm 2b ($K=6$)** | Trained Pause-Token Control ($K=6$ `<pause>`) | *Pending* | - | - | - | 12 KiB | - | - |
| **Arm 2b ($K=32$)** | Trained Pause-Token Control ($K=32$ `<pause>`) | *Pending* | - | - | - | 12 KiB | - | - |
| **Arm 3 ($K=6$)** | Continuous Latent Recurrence ($K=6$ latents) | *Pending* | - | - | - | 12 KiB | - | - |
| **Arm 3 ($K=32$)**| Continuous Latent Recurrence ($K=32$ latents) | *Pending* | - | - | - | 12 KiB | - | - |
| **Arm 3a** | Adaptive Halting Recurrence ($K_{\min} \dots K_{\max}$) | *Pending* | - | - | - | 12 KiB | - | - |
| **Arm 4** | Untrained Base Unconstrained CoT | *Pending* | - | - | - | 12 KiB | - | - |

---

## 2. Phase 0: Baseline Runs & Anchor Calibration Targets

| Benchmark | Mode | Target Metric / Published Anchor | Evaluated Setting | Measured Pass@1 | Status |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Concurrency Sweep** | Throughput | Discover knee $C_{\text{2B}}^*$ | Concurrency sweep $C \in [8..64]$ | - | **PENDING** |
| **GPQA Diamond** | Thinking | **51.6%** | `temp=1.0, top_p=0.95, top_k=20, pp=1.5` | - | **PENDING** |
| **MATH-500** | Thinking (Arm 4 Ceiling) | Measured baseline | `temp=1.0, top_p=0.95, top_k=20, pp=1.5` | - | **PENDING** |
| **MATH-500** | Non-Thinking (Arm 1 Floor) | Measured baseline | Direct template exit | - | **PENDING** |
| **GSM8K** | Thinking (Arm 4 Ceiling) | Measured baseline | `temp=1.0, top_p=0.95, top_k=20, pp=1.5` | - | **PENDING** |
| **GSM8K** | Non-Thinking (Arm 1 Floor) | Measured baseline | Direct template exit | - | **PENDING** |
| **IFEval** | Thinking / Non-Thinking | **78.6%** / **61.2%** | Standard IFEval prompt protocol | - | **PENDING** |

---

## 3. Architecture Hooks & Recurrent Cache State
Because Qwen3.5-2B utilizes hybrid Gated DeltaNet layers, latent recurrence operates by preserving both:
1. Conventional transformer KV cache states for the 6 full-attention layers.
2. Linear RNN recurrent hidden states ($S_t = S_{t-1} \odot \dots$) for the 18 Gated DeltaNet layers.
This architecture enables linear scaling and bounded memory footprint ($9.84\text{ MiB}$ fixed state size) regardless of reasoning depth $K$.

---

## 4. Phase 0b: Arm 0 Surgical Neutrality (MMLU Ground-Truth Baseline)

Evaluated across $N=1,000$ stratified MMLU questions across 57 subjects in pure `bfloat16` on GPU 0 (`cuda:0`, RTX 4080 16GB) using 5-shot prompts, logit evaluation on last token position:

| Category | Total Questions | Correct | Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Social Sciences** | 221 | 147 | **66.52%** |
| **Other** | 243 | 152 | **62.55%** |
| **STEM** | 193 | 99 | **51.30%** |
| **Humanities** | 343 | 173 | **50.44%** |
| **OVERALL BASELINE** | **1,000** | **571** | **57.10%** |

* **Execution Speed**: $129.8\text{ ms/q}$ ($100\%$ GPU compute utilization at $246\text{W}$).
* **VRAM Allocated**: $3,589.3\text{ MiB}$ (~$3.6\text{ GB}$).
* **Pre-Registered Non-Inferiority Gate**: All future adapters for Qwen3.5-2B (Arm 1b, Arm 2b, Arm 3) will be evaluated on these exact 1,000 queries with $B=10,000$ paired bootstrap iterations against this reference score ($57.10\%$) asserting $\Delta(\text{Adapter} - \text{Base}) \ge -1.5\%$.
* **Evidence Artifact**: [`data/surgical_neutrality_base_qwen3.5_2b.json`](../data/surgical_neutrality_base_qwen3.5_2b.json).

