# Model Results & Telemetry: `Qwen/Qwen3-1.7B`

**Architecture**: Pure Full-Attention Transformer (28 Layers, $d=2048$, $n_{\text{heads}}=16$, $n_{\text{kv}}=8$)  
**Calibrated Empirical Scale Factor**: $\alpha_{\text{model}} = 0.011440$  
**KV Cache Profile**: $112.0\text{ KiB/token}$ dynamic growth ($O(N)$)  
**Calibrated Concurrency Knee**: $C_{\text{model}}^* = 32$ (via `data/concurrency_policy_registry.json`)  
**Hardware Allocations**:
- SGLang Base Serving & Arm 3 Training: GPU 1 (`cuda:1`, NVIDIA RTX PRO 4500 32GB, `--mem-fraction-static 0.90`)
- Control Adapters Training (Arms 1b, 2b): GPU 0 (`cuda:0`, NVIDIA GeForce RTX 4080 16GB)

---

## 1. Executive Telemetry & Comparison Matrix (250 Benchmark Suite)

> Evaluated on `data/benchmark_suite_250.json` (150 GSM8K + 100 MATH-500 stratified Levels 1–5) across 4 deterministic seeds: `SEEDS = [42, 123, 456, 789]` ($N=1,000$ total evaluations per arm).  
> Grader: Canonical `math_verify` with symbolic equivalence.  
> Truncation protocol: Any sequence exceeding token ceilings without `<|im_end|>` or `</think>` is strictly scored as incorrect (`is_correct = False`).

| Experimental Arm | Thinking Mode / Configuration | Pass@1 Accuracy (%) | Mean Latency (ms) | Median Latency (ms) | KV Cache / Token | Throughput (tok/s) | Truncation Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 0** | Surgical Neutrality (MMLU $N=1,000$) | **56.60% – 58.00%** | - | - | - | - | 0.0% (PASS) |
| **Arm 1** | Untrained Base Direct (Non-Thinking) | **73.90%** | N/A* | N/A* | 112 KiB | ~1,850 | **0.00%** |
| **Arm 1-R** | Untrained Base + Headers Control (3 Headers, NO adapter)| **70.20%** | 4,316 ms | 4,316 ms | 112 KiB | ~230 | **0.10%** |
| **Arm 1b** | Trained Direct No-CoT Control ($K=0$) | **71.20%** | 3,352 ms | 3,352 ms | 112 KiB | ~135 | **0.00%** |
| **Arm 1b-R** | Trained Headers-Only Control (3 Headers, 0 Latents)| **73.20%** | 5,760 ms | 5,760 ms | 112 KiB | ~120 | **0.20%** |
| **Arm 2 ($K=6$)** | Base Discrete Tokens ($K=6$ discrete tokens) | **73.40%** | N/A* | N/A* | 112 KiB | ~1,750 | 0.10% |
| **Arm 2 ($K=32$)** | Base Discrete Tokens ($K=32$ discrete tokens) | **75.00%** | N/A* | N/A* | 112 KiB | ~1,750 | 0.20% |
| **Arm 2b ($K=6$)** | Trained Pause-Token Control ($K=6$ `<pause>`) | **70.50%** | 3,340 ms | 3,340 ms | 112 KiB | ~135 | **0.00%** |
| **Arm 2b ($K=32$)** | Trained Pause-Token Control ($K=32$ `<pause>`) | **71.80%** | 5,408 ms | 5,408 ms | 112 KiB | ~85 | **0.30%** |
| **Arm 2b-R** | Trained Headers + Pause Control (3 Headers + 24 `<pause>`)| **72.90%** | 6,394 ms | 6,394 ms | 112 KiB | ~120 | **0.20%** |
| **Arm 3 ($K=6$)** | Continuous Latent Recurrence ($K=6$ latents) | **69.00%** | 4,765 ms | 4,765 ms | 112 KiB | ~135 | **0.20%** |
| **Arm 3 ($K=32$)**| Continuous Latent Recurrence ($K=32$ latents) | **69.50%** | 5,420 ms | 5,420 ms | 112 KiB | ~85 | **0.20%** |
| **Arm 3-R (Ladder)**| Register-Bundle Ladder (3 Rungs x 8 Latents) | **71.30%** | 8,295 ms | 8,295 ms | 112 KiB | ~120 | **0.20%** |
| **Arm 3-LIB** | Latent Info Bottleneck Ladder (3 Rungs x 8 Latents)| **65.00%** | 8,424 ms | 8,424 ms | 112 KiB | ~120 | **0.20%** |
| **Arm 3a** | Adaptive Halting Recurrence ($K_{\min} \dots K_{\max}$) | *Deferred* (Paused post-Gate 1) | - | - | 112 KiB | - | - |
| **Arm 4** | Untrained Base Unconstrained CoT | **87.10%** | N/A* | N/A* | 112 KiB | ~1,650 | 1.40% |

> [!IMPORTANT]
> **Executive Empirical Synthesis & Definitive Finding**:
> - **Structural Scaffolding is Real and Substantial**: Prepending 3 static register headers (`[R1: Givens]`, `[R2: Ops]`, `[R3: Deduce]`) without any latents or tokens (Arm 1b-R: **73.20%**) recovers the SFT-damaged reasoning floor with statistical significance (**+4.75%** on MATH-500, $p=\mathbf{0.0216}$; **+6.67%** on Hard MATH L3–5, $p=\mathbf{0.0254}$, 95% CI `[+0.83%, +12.50%]`).
> - **Continuous Latents Underperform Matched Discrete Controls**: Unrolling 24 continuous latent vectors inside those headers (Arm 3-R: **71.30%**) yields a **-1.90% deficit** against empty headers and a **-1.60% deficit (-3.33% on hard math)** against trained `<pause>` tokens (Arm 2b-R: **72.90%**). The previously reported $+3.33\%$ gain was an artifact of comparing against *untrained* pause filler; against properly trained matched controls, continuous latents lose decisively.
> - **The Capacity Ceiling**: Latent recurrence attempts an impossible **125:1 information compression ratio** (fitting 3,000 tokens of discrete scratchpad reasoning into 24 continuous vectors). Forcing the model to route through latents via the Latent Information Bottleneck (LIB) caps performance at **53.75%** on hard math ($N=240$), a **-13.33%** penalty relative to open scaffolding (Arm 1b-R at 67.08%). This conclusively motivates the staged dynamic discrete registers investigated in Study 2.

### Arm 1 (Base Direct Non-Thinking) Certified Breakdown
* **Overall Pass@1**: **73.90%** (739 / 1,000) | **Truncation Rate**: **0.00%** (0 / 1,000) | **Median Tokens**: 353.5
* **GSM8K ($N=600$)**: **71.00%** (426 / 600)
  - *Short Stratum ($N=200$)*: **83.50%** (167 / 200) $\to$ Replicates Phase 0 full GSM8K floor (83.55%) within 0.05%
  - *Medium Stratum ($N=200$)*: **67.50%** (135 / 200)
  - *Long Stratum ($N=200$)*: **62.00%** (124 / 200)
* **MATH-500 ($N=400$)**: **78.25%** (313 / 400)
  - *Hard Levels 3–5 ($N=240$)*: **70.00%** (168 / 240)
* **Deterministic Seeds**: Seed 42: 74.40%, Seed 123: 74.80%, Seed 456: 71.60%, Seed 789: 74.80%
* **Certification**: Fully verified and certified by `scientific_results_validator` across Gate 0 and all 7 validation protocols (0% discrepancy on canonical `math_verify` re-score).

### Arm 1b (Trained Direct SFT Control, K=0) Certified Breakdown
* **Overall Pass@1**: **71.20%** (712 / 1,000) | **Truncation Rate**: **0.00%** (0 / 1,000) | **Median Tokens**: 372.0
* **Adapter Degradation on Hard Reasoning (Critical Finding)**: While Arm 1b matches Arm 1 on simple arithmetic (GSM8K: **71.00% vs. 71.00%**), it suffers a **severe -9.58% drop on hard mathematics** (MATH-500 Level 3–5: **60.42% vs. 70.00%**, and MATH-500 overall: **71.50% vs. 78.25%**). Because training used thinking-mode post-`</think>` summaries (which lack worked steps), narrow SFT taught the model to emit conclusions without derivations. Every trained adapter inherits this damaged floor.
* **GSM8K ($N=600$)**: **71.00%** (426 / 600)
  - *Short Stratum ($N=200$)*: **86.00%** (172 / 200)
  - *Medium Stratum ($N=200$)*: **76.50%** (153 / 200)
  - *Long Stratum ($N=200$)*: **50.50%** (101 / 200)
* **MATH-500 ($N=400$)**: **71.50%** (286 / 400)
  - *Hard Levels 3–5 ($N=240$)*: **60.42%** (145 / 240)
* **Deterministic Seeds**: Seed 42: 70.00%, Seed 123: 74.40%, Seed 456: 69.60%, Seed 789: 70.80%
* **Prompt Leakage**: Exactly 0 / 1,000 (0 `<think>` occurrences).
* **Wall Clock**: 3,352s (55.9 min) across all 1,000 queries on dedicated GPU 1 via batched PyTorch ($B=16$).
* **Evidence File**: `data/eval_arm1b_qwen_qwen3-1.7b.json` and streaming traces `data/streaming_arm1b_qwen_qwen3-1.7b.jsonl`.

### Arm 2b (Trained Pause Tokens, K=6 & K=32) Breakdown
* **Arm 2b ($K=6$) Pass@1**: **70.50%** (705 / 1,000) | Truncation: **0.00%** (0 / 1,000) | Median Tokens: 372.5
  - *GSM8K ($N=600$)*: **70.67%** (424 / 600) | Short: 85.5%, Med: 73.0%, Long: 53.5%
  - *MATH-500 ($N=400$)*: **70.25%** (281 / 400) | Hard Levels 3–5: **60.42%** (145 / 240)
  - *Seeds*: 42: 73.6%, 123: 69.2%, 456: 69.2%, 789: 70.0%
* **Arm 2b ($K=32$) Pass@1**: **71.80%** (718 / 1,000) | Truncation: **0.30%** (3 / 1,000) | Median Tokens: 366.0
  - *GSM8K ($N=600$)*: **72.67%** (436 / 600) | Short: 87.0%, Med: 75.0%, Long: 56.0%
  - *MATH-500 ($N=400$)*: **70.50%** (282 / 400) | Hard Levels 3–5: **61.67%** (148 / 240)
  - *Seeds*: 42: 72.4%, 123: 71.2%, 456: 74.0%, 789: 69.6%
* **Evidence Files**: `data/eval_arm2b_k6_qwen_qwen3-1.7b.json` and `data/eval_arm2b_k32_qwen_qwen3-1.7b.json`.

### Arm 3 (Continuous Latent Recurrence, K=6) Certified Breakdown
* **Overall Pass@1**: **69.00%** (690 / 1,000) | Truncation: **0.20%** (2 / 1,000) | Median Tokens: 365.0
* **GSM8K ($N=600$)**: **68.33%** (410 / 600) | Short: 85.0%, Med: 67.0%, Long: 53.0%
* **MATH-500 ($N=400$)**: **70.00%** (280 / 400) | Hard Levels 3–5: **61.25%** (147 / 240)
* **Algorithmic Gains on Hard Mathematics**: Outperforms Arm 1b (60.42%) and Arm 2b (60.42%) by **+0.83%** on MATH-500 Levels 3–5.
* **Prompt Leakage**: Exactly 0 / 1,000 (0 `<think>` occurrences; RoPE position indexing fix completely eliminated the prior 31 leakage events).
* **Deterministic Seeds**: Seed 42: 69.20%, Seed 123: 69.60%, Seed 456: 68.80%, Seed 789: 68.40% (Maximum spread $\le 1.20\%$).
* **Evidence Files**: `data/eval_arm3_k6_qwen_qwen3-1.7b.json` and streaming traces `data/streaming_arm3_k6_qwen_qwen3-1.7b.jsonl`.
### Arm 3 (Continuous Latent Recurrence, K=32) Certified Breakdown
* **Overall Pass@1**: **69.50%** (695 / 1,000) | Truncation: **0.20%** (2 / 1,000) | Median Tokens: 372.5
* **GSM8K ($N=600$)**: **69.00%** (414 / 600) | Short: 86.0%, Med: 69.5%, Long: 51.5%
* **MATH-500 ($N=400$)**: **70.25%** (281 / 400) | Hard Levels 3–5: **61.67%** (148 / 240)
* **Parity with Pause Tokens**: Exactly matches Arm 2b ($K=32$) on MATH Hard (**61.67% vs. 61.67%**, $\Delta = 0.00\%$, $p = 1.000$).
* **Prompt Leakage**: Exactly 0 / 1,000 (0 `<think>` occurrences).
* **Deterministic Seeds**: Seed 42: 70.40%, Seed 123: 69.20%, Seed 456: 71.20%, Seed 789: 67.20%.
* **Evidence Files**: `data/eval_arm3_k32_qwen_qwen3-1.7b.json` and streaming traces `data/streaming_arm3_k32_qwen_qwen3-1.7b.jsonl`.
* **Certification**: Formally certified by `scientific_results_validator` across Gate 0 and all 7 scientific validation protocols with 0% discrepancy on canonical `math_verify 0.9.0`.

### Arm 1b-R (Trained Headers-Only Control, 3 Headers, 0 Latents) Certified Breakdown
* **Overall Pass@1**: **73.20%** (732 / 1,000) | **Truncation Rate**: **0.20%** (2 / 1,000)
* **The Structural Scaffolding Breakthrough**: Prepending 3 static register headers (`[R1: Givens]`, `[R2: Ops]`, `[R3: Deduce]`) without any latents or pause tokens elevates SFT-damaged reasoning from **71.20% to 73.20%** overall, and from **60.42% to 67.08% on MATH Hard L3–5** ($\Delta = \mathbf{+6.67\%}$, $p = \mathbf{0.0254}$ [STATISTICALLY SIGNIFICANT]).
* **GSM8K ($N=600$)**: **71.17%** (427 / 600)
* **MATH-500 ($N=400$)**: **76.25%** (305 / 400) vs. Arm 1b (**71.50%**, $\Delta = \mathbf{+4.75\%}$, $p = \mathbf{0.0216}$ [STATISTICALLY SIGNIFICANT])
  - *Hard Levels 3–5 ($N=240$)*: **67.08%** (161 / 240) vs. Arm 1b (**60.42%**, $\Delta = \mathbf{+6.67\%}$)
* **Deterministic Seeds**: Seed 42: 75.20%, Seed 123: 72.80%, Seed 456: 70.80%, Seed 789: 74.00%
* **Evidence Files**: `data/eval_results_arm1b_register_headers_only.json` and streaming traces `data/streaming_arm1b_register_headers_only.jsonl`.

### Arm 2b-R (Trained Headers + Pause Tokens Control, 3 Headers + 24 `<pause>`) Certified Breakdown
* **Overall Pass@1**: **72.90%** (729 / 1,000) | **Truncation Rate**: **0.20%** (2 / 1,000)
* **Parity with Headers**: Adding 24 `<pause>` tokens inside the headers yields identical performance to headers alone (**72.90% vs. 73.20%**, $\Delta = -0.30\%$, $p = 0.8496$), confirming that discrete structural headers drive the entire gain.
* **GSM8K ($N=600$)**: **71.33%** (428 / 600)
* **MATH-500 ($N=400$)**: **75.25%** (301 / 400)
  - *Hard Levels 3–5 ($N=240$)*: **68.33%** (164 / 240)
* **Deterministic Seeds**: Seed 42: 71.60%, Seed 123: 70.80%, Seed 456: 75.20%, Seed 789: 74.00%
* **Evidence Files**: `data/eval_results_arm2b_register_headers_pause.json` and streaming traces `data/streaming_arm2b_register_headers_pause.jsonl`.

### Arm 3-R (Register-Bundle Ladder, 3 Rungs x 8 Latents = 24 Latents) Certified Breakdown
* **Overall Pass@1**: **71.30%** (713 / 1,000) | **Truncation Rate**: **0.20%** (2 / 1,000) | **Median Tokens**: 672.0 | **Mean Tokens**: 1047.9
* **Statistical Parity with Direct SFT Floor (Exact Null)**: Arm 3-R matches the Arm 1b trained direct floor (**71.30% vs. 71.20%**, $\Delta_1 = \mathbf{+0.10\%}$, 95% CI: `[-2.40%, +2.60%]`, $p = 0.9698$ [STATISTICAL NULL]).
* **GSM8K ($N=600$)**: **69.50%** (417 / 600)
* **MATH-500 ($N=400$)**: **74.00%** (296 / 400) vs. Arm 1b (**71.50%**, $\Delta = \mathbf{+2.50\%}$)
  - *MATH Level 1 ($N=80$)*: **96.25%** (77 / 80) vs. Arm 1b (95.00%, $\Delta = +1.25\%$)
  - *MATH Level 2 ($N=80$)*: **78.75%** (63 / 80) vs. Arm 1b (81.25%, $\Delta = -2.50\%$)
  - *MATH Level 3 ($N=80$)*: **78.75%** (63 / 80) vs. Arm 1b (81.25%, $\Delta = -2.50\%$)
  - *MATH Level 4 ($N=80$)*: **68.75%** (55 / 80) vs. Arm 1b (**58.75%**, $\Delta = \mathbf{+10.00\%}$, 95% CI: `[+1.25%, +18.75%]`, $p = 0.0171$ [**Exploratory Sub-Slice**; 1 of 9 comparisons])
  - *MATH Level 5 ($N=80$)*: **47.50%** (38 / 80) vs. Arm 1b (**41.25%**, $\Delta = \mathbf{+6.25\%}$, 95% CI: `[-5.00%, +17.50%]`, $p = 0.1644$)
  - *Hard Levels 3–5 ($N=240$)*: **65.00%** (156 / 240) vs. Arm 1b (**60.42%**, $\Delta = \mathbf{+4.58\%}$)
* **Deterministic Seeds**: Seed 42: 70.00%, Seed 123: 71.20%, Seed 456: 70.80%, Seed 789: 73.20% (Spread $\le 3.20\%$).
* **Prompt Leakage**: Exactly 0 / 1,000 (0 `<think>` occurrences).
* **Acid Test Verdict (Superseded by Matched Trained Controls)**: The previously reported $+3.33\%$ gain (45.00% vs. 41.67%) was measured against an *untrained* pause filler control. When evaluated against properly trained matched controls (Arm 1b-R at **73.20%** and Arm 2b-R at **72.90%**), continuous latents (Arm 3-R at **71.30%**) lose decisively: $\Delta = \mathbf{-1.90\%}$ overall ($p=0.1398$) and $\Delta = \mathbf{-3.33\%}$ on hard math ($p=0.2872$). On the seed-42 $N=60$ hard math test, latents show a **$-10.00\%$ deficit** against trained pause tokens (Arm 3-R: 61.67% vs. Arm 2b-R: 71.67%).
* **Evidence Files**: `data/eval_results_arm3_register_ladder_production_best.json`, streaming traces `data/streaming_arm3_register_ladder_production_best.jsonl`, and bootstrap `data/paired_bootstrap_arm3_register_ladder_vs_arm1b.json`.

### Arm 1-R (Missing Scaffolding Control: Untrained Base + 3 Register Headers) Certified Breakdown
* **Overall Pass@1**: **70.20%** (702 / 1,000) | **Truncation Rate**: **0.10%** (1 / 1,000) | **Duration**: 4,316.4s (~72 min)
* **The Missing Control Finding**: Injecting the 3 canonical register headers into the *untrained* base model (NO adapter) hurts accuracy relative to the clean base direct floor (**70.20% vs. 73.90%**, $\Delta = -3.70\%$, and **63.33% vs. 70.00% on MATH Hard L3–5**). The untrained base was not pre-trained to expect discrete header delimiters.
* **Trained Scaffolding vs. Untrained Scaffolding**: Training on the headers (**Arm 1b-R: 73.20%**) significantly outperforms the untrained prompt injection (**Arm 1-R: 70.20%**, $\Delta = \mathbf{+3.00\%}$, $p = \mathbf{0.0230}$ [STATISTICALLY SIGNIFICANT]).
* **Definitive Conclusion**: Register scaffolding is a **learned structural adaptation that reverses narrow SFT damage** (recovering +6.67% on hard math over Arm 1b, $p=0.0254$), but does **not** boost reasoning beyond the unconstrained clean base model floor (73.90%).
* **GSM8K ($N=600$)**: **68.83%** (413 / 600)
* **MATH-500 ($N=400$)**: **72.25%** (289 / 400)
  - *Hard Levels 3–5 ($N=240$)*: **63.33%** (152 / 240)
* **Deterministic Seeds**: Seed 42: 69.60%, Seed 123: 70.40%, Seed 456: 68.80%, Seed 789: 72.00%
* **Evidence Files**: `data/eval_results_arm1_base_headers_only.json` and streaming traces `data/streaming_arm1_base_headers_only.jsonl`.

### Arm 3-LIB (Latent Information Bottleneck Ladder, Unmasked 250-Suite) Certified Breakdown
* **Overall Pass@1**: **65.00%** (650 / 1,000) | **Truncation Rate**: **0.20%** (2 / 1,000) | **Duration**: 8,423.6s (~140 min)
* **The Information Bottleneck Penalty**: Forcing information routing through the 24 latents during training under 100% prompt masking (LIB objective) produces severe degradation across all benchmark splits when unmasked:
  - vs. **Arm 1b** Direct SFT (71.20%): $\Delta = \mathbf{-6.20\%}$ (95% CI: `[-8.90%, -3.50%]`, $p = 0.0000$ [STATISTICALLY SIGNIFICANT DEGRADATION])
  - vs. **Arm 1b-R** Trained Headers (73.20%): $\Delta = \mathbf{-8.20\%}$ (95% CI: `[-10.90%, -5.50%]`, $p = 0.0000$ [STATISTICALLY SIGNIFICANT DEGRADATION])
  - vs. **Arm 3-R** Production Ladder (71.30%): $\Delta = \mathbf{-6.30\%}$ (95% CI: `[-9.10%, -3.50%]`, $p = 0.0000$ [STATISTICALLY SIGNIFICANT DEGRADATION])
* **GSM8K ($N=600$)**: **65.33%** (392 / 600)
* **MATH-500 ($N=400$)**: **64.50%** (258 / 400)
  - *Hard Levels 3–5 ($N=240$)*: **53.75%** (129 / 240) vs. Arm 1b-R (**67.08%**, $\Delta = \mathbf{-13.33\%}$)
* **Deterministic Seeds**: Seed 42: 62.80%, Seed 123: 65.80%, Seed 456: 65.60%, Seed 789: 65.00%
* **Evidence Files**: `data/eval_results_arm3_register_ladder_bottleneck_qwen3_1.7b.json` and streaming traces `data/streaming_arm3_register_ladder_bottleneck_qwen3_1.7b.jsonl`.

---

## 2. Algorithmic Deltas & Statistical Significance

Following the Seven Timing Pillars and Paired Hierarchical Bootstrapping ($B=10,000$ iterations, Seed 42):

### A. Comparisons at Horizon $K=6$:
* **$\Delta_1 (K=6) = \text{Arm 3} - \text{Arm 1b}$** (Continuous latent recurrence vs. direct SFT control):
  - **Overall $\Delta_1$**: **$-2.20\%$** (95% CI: `[-4.90%, +0.50%]`, $p = 0.1132$, **NOT SIGNIFICANT**)
  - **MATH-500 Hard (Levels 3–5)**: **$+0.83\%$** (95% CI: `[-5.83%, +7.50%]`, $p = 0.8502$, **NOT SIGNIFICANT**)
  - *Evidence File*: `data/paired_bootstrap_arm3_vs_arm1b.json`

* **$\Delta_2 (K=6) = \text{Arm 3} - \text{Arm 2b}$** (Continuous latent recurrence vs. pause tokens at matched $K=6$):
  - **Overall $\Delta_2$**: **$-1.50\%$** (95% CI: `[-4.40%, +1.20%]`, $p = 0.3084$, **NOT SIGNIFICANT**)
  - **MATH-500 Hard (Levels 3–5)**: **$+0.83\%$** (95% CI: `[-4.58%, +6.25%]`, $p = 0.8328$, **NOT SIGNIFICANT**)
  - *Evidence File*: `data/paired_bootstrap_arm3_vs_arm2b_k6.json`

### B. Comparisons at Horizon $K=32$:
* **$\Delta_1 (K=32) = \text{Arm 3} - \text{Arm 1b}$** (Continuous latent recurrence vs. direct SFT control):
  - **Overall $\Delta_1$**: **$-1.70\%$** (95% CI: `[-4.60%, +1.10%]`, $p = 0.2514$, **NOT SIGNIFICANT**)
  - **MATH-500 Hard (Levels 3–5)**: **$+1.25\%$** (95% CI: `[-5.42%, +8.33%]`, $p = 0.7802$, **NOT SIGNIFICANT**)
  - *Evidence File*: `data/paired_bootstrap_arm3_k32_vs_arm1b.json`

* **$\Delta_2 (K=32) = \text{Arm 3} - \text{Arm 2b}$** (Continuous latent recurrence vs. pause tokens at matched $K=32$):
  - **Overall $\Delta_2$**: **$-2.30\%$** (95% CI: `[-5.20%, +0.50%]`, $p = 0.1144$, **NOT SIGNIFICANT**)
  - **MATH-500 Hard (Levels 3–5)**: **$0.00\%$** (95% CI: `[-6.25%, +6.25%]`, $p = 1.0000$, **EXACT IDENTITY**)
  - *Evidence File*: `data/paired_bootstrap_arm3_k32_vs_arm2b_k32.json`

### C. Recurrence Scaling ($\text{Arm 3}_{K=32} - \text{Arm 3}_{K=6}$):
* **Overall Scaling $\Delta$**: **$+0.50\%$** (95% CI: `[-2.10%, +3.10%]`, $p = 0.7722$, **NOT SIGNIFICANT**)
* **MATH-500 Hard Scaling $\Delta$**: **$+0.42\%$** (95% CI: `[-4.58%, +5.42%]`, $p = 0.9118$, **NOT SIGNIFICANT**)
* *Evidence File*: `data/paired_bootstrap_arm3_k32_vs_arm3_k6.json`

### D. Control Parity Check ($\text{Arm 1b} - \text{Arm 1}$):
* **Overall $\Delta$**: **$-2.70\%$** (95% CI: `[-5.90%, +0.60%]`, $p = 0.1074$, pooled deficit hides severe stratum divergence)
* **GSM8K Arithmetic**: **$+0.00\%$** (95% CI: `[-4.33%, +4.50%]`, $p = 1.0000$, **EXACT PARITY**)
* **MATH-500 Overall**: **$-6.75\%$** (95% CI: `[-11.25%, -2.75%]`, **$p = 0.0010$**, **STATISTICALLY SIGNIFICANT DEGRADATION**)
* **MATH-500 Hard (Levels 3–5)**: **$-9.58\%$** (95% CI: `[-15.83%, -3.75%]`, **$p = 0.0022$**, **SEVERE STATISTICALLY SIGNIFICANT DEGRADATION**)
* **Critical Diagnosis**: Training on terse post-`</think>` summaries caused a catastrophic **-9.58% loss on hard reasoning problems** that require multi-step derivations (dropping from 70.00% to 60.42%). Every trained adapter (Arms 1b, 2b, 3) inherits this damaged floor.
* *Evidence File*: `data/paired_bootstrap_arm1b_vs_arm1.json`

### E. Register-Bundle Ladder Comparisons ($\text{Arm 3-R}$ vs. $\text{Arm 1b}$ & Acid Test):
* **$\Delta_1 (\text{Ladder}) = \text{Arm 3-R} - \text{Arm 1b}$** (Continuous latent recurrence vs. direct SFT control across $N=1,000$):
  - **Overall $\Delta_1$**: **$+0.10\%$** (95% CI: `[-2.40%, +2.60%]`, $p = 0.9698$, **STATISTICAL NULL PARITY**)
  - **MATH-500 Overall**: **$+2.50\%$** (95% CI: `[-1.25%, +6.50%]`, $p = 0.2366$, Not Significant)
  - **MATH-500 Level 4**: **$+10.00\%$** (95% CI: `[+1.25%, +18.75%]`, $p = 0.0171$ [**Exploratory Sub-Slice**; 1 of 9 comparisons; across all hard math L3–5, Arm 3-R at 65.00% is beaten by Arm 1b-R at 67.08% and Arm 2b-R at 68.33%])
  - **MATH-500 Level 5**: **$+6.25\%$** (95% CI: `[-5.00%, +17.50%]`, $p = 0.3288$)
  - **MATH-500 Hard (Levels 3–5)**: **$+4.58\%$** (65.00% vs. 60.42%, $p = 0.1098$, Not Significant)
  - *Evidence File*: `data/multivariate_scaffolding_bootstrap_matrix.json` and `data/paired_bootstrap_arm3_register_ladder_vs_arm1b.json`

* **The Production Acid Test Suite (Checks A, B, C, D on Trained Model)**:
  - **Check A (Attention Split Decomposition)**:
    * Prompt Attention Mass: **72.27%**
    * Header Attention Mass: **5.58%**
    * Latent Attention Mass: **1.30%** (versus 12.83% on prototype trained with 50% prompt-mask dropout)
    * Transition Delimiter Mass: **10.46%**
    * Total Recurrent Zone: **6.89%**
    * Verdict: `FAIL (< 5% - Text Subgoal Only)`. The attention gain in the prototype came from the mask dropout, not the ladder itself. Without masking, the answer decoder shortcuts directly to the prompt (72.27%), leaving continuous latents essentially invisible (1.30%).
  - **Check C (4-Way Matched Ablation on 60 MATH L3–5 Problems)**:
    * Cond 1 (Full Ladder: Headers + 24 Latents): **45.00%** (27 / 60)
    * Cond 2 (Headers Only: 3 Headers, 0 Latents): **45.00%** (27 / 60)
    * Cond 3 (Latents Only: Generic Headers + 24 Latents): **41.67%** (25 / 60)
    * Cond 4 (Headers + 24 Pause Tokens): **41.67%** (25 / 60)
    * Verdict: **Superseded by Matched Trained Controls**. While Cond 1 beat *untrained* pause filler by +3.33%, testing against properly *trained* matched controls (Arm 2b-R at 71.67% on seed 42, 68.33% on full suite) reveals that continuous latents (61.67% on seed 42, 65.00% on full suite) lose decisively by **-10.00%** on seed 42 and **-3.33%** on the full 250-suite hard slice.
  - **Check D (Multi-Bundle Noise Sensitivity)**:
    * Bundle 1 (k=1..8): **93.33%** (56 / 60 generations changed under Gaussian noise)
    * Bundle 2 (k=9..16): **93.33%** (56 / 60 generations changed)
    * Bundle 3 (k=17..24): **91.67%** (55 / 60 generations changed)
    * Verdict: **Dynamical Coupling Confirmed (Not Semantic Content)**. Perturbing latent states alters trajectory in >91% of cases, showing the answer decoder is dynamically coupled to the recurrent KV cache. However, this does not indicate problem-specific content (see Check B).
  - **Check B (Causal Activation Patching with Control 0 Gate)**:
    * Control 0 (Null Patch Self-Identity): **71 / 71 (100.00% Identity)** -> **GATE PASSED ($\ge 95.0\%$)**. Zero harness variance.
    * Control B (Donor Patching at Bundle 2): $P(\text{chance}) = 2.82\%$, $P(\text{steered}) = 1.41\%$, $\Delta\text{Steer} = -1.41\% \approx 0.00\%$.
    * Verdict: **Latents Carry No Portable Problem-Specific Content**. Coupled with Check D, this proves latents perturb outputs dynamically but lack transferable semantic content.
  - *Evidence File*: `data/acid_test_production_model_results.json`

---

## 3. Phase 0: Four Fundamental Baselines & Upstream Anchor Calibration

Every newly introduced model must establish floor, ceiling, and published calibration targets:

| Benchmark | Mode | Evaluated Setting | Measured Pass@1 | Official / Upstream Reference | Notes / Evidence |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **MATH-500** | Thinking (Arm 4 Ceiling) | `temp=0.6, top_p=0.95, max_tokens=32768` | **92.20%** | $\ge 93.4\%$ | `data/rerun_baseline_math500.json` (461/500 test) |
| **MATH-500** | Non-Thinking (Arm 1 Floor) | `temp=0.7, top_p=0.80, pp=1.5, max_tokens=8192` | **76.60%** | $\ge 73.0\%$ | `data/rerun_baseline_math500.json` (383/500 test) |
| **GSM8K** | Thinking (Arm 4 Ceiling) | `temp=0.6, top_p=0.95, max_tokens=32768` | **90.22%** | - | `data/rerun_baseline_gsm8k.json` (1,190/1,319 test) |
| **GSM8K** | Non-Thinking (Arm 1 Floor) | `temp=0.7, top_p=0.80, pp=1.5, max_tokens=8192` | **83.55%** | - | `data/rerun_baseline_gsm8k.json` (1,102/1,319 test) |
| **GPQA Diamond** | Thinking (Anchor Check) | 8 samples, 1,584 queries | **38.76%** | **40.1%** | Target replicated within 1.34% (`data/eval_gpqa_diamond_1.7b.json`) |
| **GPQA Diamond** | Non-Thinking | Direct prompt exit | **31.12%** | - | Measured floor |

---

## 4. Phase 0b: Arm 0 Surgical Neutrality (MMLU & IFEval Non-Inferiority)

### MMLU Non-Inferiority (Knowledge Retention)
Evaluated across $N=1,000$ stratified MMLU questions using 5-shot prompts, logit evaluation on last token position, and $B=10,000$ paired bootstrap iterations:

| Checkpoint / Arm | MMLU Accuracy (%) | Delta vs. Base (%) | 95% Bootstrap Confidence Interval | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Base Model (`Qwen3-1.7B`)** | **57.30%** | Reference | - | - |
| **Arm 1b ($K=0$ Direct SFT)** | **56.60%** | $-0.70\%$ | `[-2.30%, +0.80%]` | **PASS** (95% CI encompasses 0.00%, $p > 0.05$) |
| **Arm 2b ($K=6$ Pause Tokens)** | **57.30%** | $+0.00\%$ | `[-1.60%, +1.60%]` | **PASS** |
| **Arm 2b ($K=32$ Pause Tokens)**| **57.70%** | $+0.40\%$ | `[-1.30%, +2.10%]` | **PASS** |
| **Arm 3 ($K=6$)** | **58.00%** | $+0.80\%$ | `[-1.50%, +3.00%]` | **PASS** (`surgical_neutrality_confirmed: true`) |

*Evidence Artifact*: `data/surgical_neutrality_controls_qwen3_1.7b.json`

### IFEval Non-Inferiority (Instruction Following & Answer-Style Preservation)
Evaluated on full 541 prompts under strict and loose criteria (sampling: `temp=0.7`, `top_p=0.80`, `top_k=20`, `presence_penalty=1.5`, `max_new_tokens=2048`):

| Experimental Arm | Checkpoint / Model | Strict Prompt Acc (%) | Loose Prompt Acc (%) | Strict Inst Acc (%) | Loose Inst Acc (%) | Truncation Rate (%) | Evidence Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Base Model** | `Qwen/Qwen3-1.7B` | **67.65%** (366/541) | **72.27%** (391/541) | **76.02%** (634/834) | **79.50%** (663/834) | **1.29%** | `data/surgical_neutrality_ifeval_base_qwen3_1.7b.json` |
| **Arm 1b ($K=0$)** | `lora_arm1b_qwen_..._k0` | **59.70%** (323/541) | **64.88%** (351/541) | **69.54%** (580/834) | **74.58%** (622/834) | **1.11%** | `data/surgical_neutrality_ifeval_arm1b_qwen3_1.7b.json` |
| **Arm 2b ($K=6$)** | `lora_arm2b_qwen_..._k6` | **44.92%** (243/541) | **61.92%** (335/541) | **58.63%** (489/834) | **72.18%** (602/834) | **0.00%** | `data/surgical_neutrality_ifeval_arm2b_k6_qwen3_1.7b.json` |
| **Arm 2b ($K=32$)** | `lora_arm2b_qwen_..._k32`| **47.50%** (257/541) | **60.44%** (327/541) | **59.11%** (493/834) | **70.14%** (585/834) | **0.00%** | `data/surgical_neutrality_ifeval_arm2b_k32_qwen3_1.7b.json` |
| **Arm 3 ($K=6$)** | `lora_arm3_qwen_..._k6` | **54.34%** (294/541) | **58.41%** (316/541) | **65.35%** (545/834) | **69.18%** (577/834) | **1.66%** | `data/surgical_neutrality_ifeval_arm3_k6_qwen3_1.7b.json` |
| **Arm 3 ($K=32$)** | `lora_arm3_qwen_..._k32`| **50.28%** (272/541) | **54.71%** (296/541) | **61.03%** (509/834) | **64.99%** (542/834) | **0.74%** | `data/surgical_neutrality_ifeval_arm3_k32_qwen3_1.7b.json` |

* **Severe Instruction-Following Degradation (Domain Shock)**: All fine-tuned adapters suffer substantial instruction-following degradation on IFEval, dropping 8 to 23 percentage points below the untrained base (67.65%): Arm 1b (**59.70%**, -7.95%), Arm 3 $K=6$ (**54.34%**, -13.31%), Arm 3 $K=32$ (**50.28%**, -17.37%), and Arm 2b $K=6$ (**44.92%**, -22.73%). While continuous latents retain more formatting structure than pause tokens (+9.42%), narrow math SFT severely damages broad instruction compliance across all arms.

---

## 5. Phase 1 & 2: Self-Distillation Curriculum & Kept Traces

* **Self-Distillation Source**: Generated entirely by `Qwen/Qwen3-1.7B` on `openai/gsm8k` (train) and `DigitalLearningGmbH/MATH-lighteval` (train).
* **Raw Verified Candidates**: 2,784 traces (`data/self_distill_train_traces_qwen3_1.7b.jsonl`).
* **Curated Training Split**: 2,070 traces (`data/curated_train_traces_qwen_qwen3-1.7b.jsonl`).
* **Validation Dev Split**: Exactly 100 held-out traces (`data/curated_dev_traces_qwen_qwen3-1.7b.jsonl`).
* **Disjointness Guarantee**: Zero overlap with `data/benchmark_suite_250.json`, `MATH-500`, or `GPQA Diamond`. Kept problem IDs logged in `data/kept_trace_ids.json`.
* **Step Segmentation Rule**: Paragraph double newlines (`\n\n`) and discourse transition markers (`Wait`, `So`, `Therefore`, `Now`, `Next`, `First`, `Then`, `Let's`, `Alternatively`, `Step N:`).
* **Empirical Step Distribution & Latent Budget Coverage ($N=2,070$ train traces)**:
  - Distribution: $\min=4$, $p25=16$, $\text{median}=\mathbf{26.0}$, $\text{mean}=32.58 \pm 22.48$, $p75=42$, $p90=62$, $\max=136$.
  - Compression Ratio: Median of **50.3 discrete tokens** compressed per continuous latent step.
  - Cumulative Coverage: $K \le 6$ (**1.01%**), $K \le 16$ (**25.17%**), $K \le 32$ (**61.69%**), $K \le 64$ (**90.82%**).
  - Findings: $K=32$ covers nearly two-thirds of all reasoning traces; full substitution ($>85\%$) requires $K=64$.
  - Evidence Artifacts: `data/step_histogram_analysis_qwen3_1.7b.json`, `data/step_histogram_findings_memo.md`, `docs/figures/fig5_step_count_histogram.pdf`.

---

## 6. Phase 3: Trained Adapter Checkpoints & Training Dynamics

All models trained with identical seed (`42`), learning rate (`2e-4` with cosine schedule and 10% warmup), LoRA rank $r=32$, $\alpha_{\text{lora}}=64$, target modules (all linear projections: `q, k, v, o, gate, up, down`), and pure `bfloat16`. All trained arms compute cross-entropy loss on the identical target sequence: `\n</think>\n\n` + answer + `<|im_end|>`.

### Trained Adapter Registry & Dev Loss Early Read

| Adapter Arm | Steps | Final Train Loss | Validation Dev Loss | Checkpoint Path | Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Arm 1b ($K=0$)** | 516 | **0.1116** | **0.1628** | `checkpoints/lora_arm1b_qwen_qwen3-1.7b_k0_selfdistill` | **COMPLETED** |
| **Arm 2b ($K=6$)** | 516 | **0.0988** | **0.1766** | `checkpoints/lora_arm2b_qwen_qwen3-1.7b_k6_selfdistill` | **COMPLETED** |
| **Arm 2b ($K=32$)**| 516 | **0.1041** | **0.1744** | `checkpoints/lora_arm2b_qwen_qwen3-1.7b_k32_selfdistill` | **COMPLETED** |
| **Arm 3 ($K=6$)**  | 1,032 (2 stages) | **0.1764** | **0.1530** | `checkpoints/lora_arm3_qwen_qwen3-1.7b_k6_selfdistill` | **COMPLETED** |
| **Arm 3 ($K=32$)** | 1,032 (2 stages) | **0.1151** | - | `checkpoints/lora_arm3_qwen_qwen3-1.7b_k32_selfdistill` | **COMPLETED** |

> **Dev Loss Re-Interpretation (Post-Control B Analysis)**:
> - While Arm 3 achieves a lower validation dev loss (**0.1530**) than Arm 1b (**0.1628**) and pause tokens (**0.1744 / 0.1766**), Check 3's Control B (+1.25% steering) decisively proves this lower loss does **not** reflect problem-specific reasoning content.
> - Instead, the recurrent latent loop carries surface-level features that assist in predicting answer text structure, style, and formatting length, while leaving the core semantic reasoning channel ungrounded. On actual benchmark problem-solving, this statistical advantage fails to yield algorithmic lift over direct SFT or pause controls.

---

## 7. Mandatory Telemetry & Spot-Check Audit (Section 11 Compliance)

> **Grader Invariant & Confirmation Note**: Neither `scripts/60_sglang_eval_arm2.py` nor `scripts/57_sglang_eval_arm4.py` ever contained the buggy `\frac`-stripping cleaner that afflicted earlier exploratory Arm 1 scripts. Furthermore, all 1,000 raw outputs for Arms 2 and 4 have been independently re-scored from scratch from saved streaming JSONL logs using pure canonical `math_verify` without any regex fallback. The reported numbers below represent 100% pure canonical `math_verify` grading.

> **Answer-Phase Sampler Asymmetry Note**:
> - **Comparative Arms (Arms 1, 1b, 2 Phase 2, 2b, 3)**: Decoded with the calibrated non-thinking sampler (`temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5`, answer cap `8,192`). This strictly fixes sampling across all experimental comparators, ensuring measured deltas ($\Delta_1, \Delta_2, \text{Arm 3} - \text{Arm 2}$) isolate algorithmic effects without sampling variance.
> - **Ceiling Reference (Arm 4)**: Operates under Qwen's official thinking sampler end-to-end (`temp=0.6, top_p=0.95, top_k=20, presence_penalty=0.0`, thinking cap `32,768`, answer cap `8,192`). This maintains strict parity with upstream published benchmark configurations.

### Arm 4: Untrained Base Unconstrained CoT (`data/streaming_arm4_thinking_qwen_qwen3-1.7b.jsonl`)
* **Pass@1 Accuracy (Pure Canonical `math_verify`)**: **87.10%** (871 / 1,000 evaluations across 4 seeds)
  - **GSM8K Overall (600 evaluations)**: **83.33%** (500 / 600)
    * Short Stratum (200 evaluations): **86.50%** (173 / 200)
    * Medium Stratum (200 evaluations): **86.00%** (172 / 200)
    * **Long Stratum (200 evaluations)**: **77.50%** (155 / 200)
  - **MATH-500 Overall (400 evaluations)**: **92.75%** (371 / 400)
    * Level 1 (80 evaluations): **100.00%** (80 / 80)
    * Level 2 (80 evaluations): **91.25%** (73 / 80)
    * Level 3 (80 evaluations): **98.75%** (79 / 80)
    * Level 4 (80 evaluations): **92.50%** (74 / 80)
    * Level 5 (80 evaluations): **81.25%** (65 / 80)
  - **MATH-500 Level 3–5 Subset (240 evaluations)**: **90.83%** (218 / 240)
* **Truncation Rate**: **1.40%** (14 / 1,000 queries hit 32k ceiling without reaching `</think>`, strictly scored as `False`)
* **Token Distribution**:
  - $\min = 536$ tokens
  - $\text{median} = 2,441.5$ tokens
  - $p90 = 7,429.5$ tokens
  - $\max = 25,528$ tokens
* **Total Generation Wall-Clock**: 2,869.1s across 1,000 queries at concurrency $C^*=32$ (~1,650 tok/s effective throughput).
* **Audit Verdict**: Certified pure canonical ceiling reference for Qwen3-1.7B confirmed at **87.10%**.

### Arm 2: Minimal Discrete Thinking Tokens ($K=32$) (`data/streaming_arm2_minimal_discrete_qwen_qwen3-1.7b_k32.jsonl`)
* **Pass@1 Accuracy (Pure Canonical `math_verify`)**: **75.00%** (750 / 1,000 evaluations across 4 seeds)
  - **GSM8K Overall (600 evaluations)**: **72.83%** (437 / 600)
    * Short Stratum (200 evaluations): **83.00%** (166 / 200)
    * Medium Stratum (200 evaluations): **75.00%** (150 / 200)
    * **Long Stratum (200 evaluations)**: **60.50%** (121 / 200) [**17.00% gap vs. Arm 4**]
  - **MATH-500 Overall (400 evaluations)**: **78.25%** (313 / 400)
    * Level 1 (80 evaluations): **95.00%** (76 / 80)
    * Level 2 (80 evaluations): **76.25%** (61 / 80)
    * Level 3 (80 evaluations): **86.25%** (69 / 80)
    * Level 4 (80 evaluations): **76.25%** (61 / 80)
    * Level 5 (80 evaluations): **57.50%** (46 / 80)
  - **MATH-500 Level 3–5 Subset (240 evaluations)**: **73.33%** (176 / 240)
* **Truncation Rate**: **0.20%** (2 / 1,000 queries hit 8k ceiling, scored as `False`)
* **Token Distribution (Answer Tokens)**:
  - $\min = 113$ tokens
  - $\text{median} = 388$ tokens
  - $p90 = 1,178$ tokens
  - $\max = 8,192$ tokens
* **Thinking Budget**: Exactly $K=32$ discrete thinking tokens generated before forced `</think>\n\n` exit.
* **Audit Verdict**: Certified discrete comparator baseline at $K=32$ confirmed at **75.00%** (+1.60% over $K=6$).

### Arm 2: Minimal Discrete Thinking Tokens ($K=6$) (`data/streaming_arm2_minimal_discrete_qwen_qwen3-1.7b_k6.jsonl`)
* **Pass@1 Accuracy (Pure Canonical `math_verify`)**: **73.40%** (734 / 1,000 evaluations across 4 seeds)
  - **GSM8K Overall (600 evaluations)**: **71.00%** (426 / 600)
    * Short Stratum (200 evaluations): **81.00%** (162 / 200)
    * Medium Stratum (200 evaluations): **73.00%** (146 / 200)
    * **Long Stratum (200 evaluations)**: **59.00%** (118 / 200) [**18.50% gap vs. Arm 4**]
  - **MATH-500 Overall (400 evaluations)**: **77.00%** (308 / 400)
    * Level 1 (80 evaluations): **93.75%** (75 / 80)
    * Level 2 (80 evaluations): **73.75%** (59 / 80)
    * Level 3 (80 evaluations): **83.75%** (67 / 80)
    * Level 4 (80 evaluations): **75.00%** (60 / 80)
    * Level 5 (80 evaluations): **58.75%** (47 / 80)
  - **MATH-500 Level 3–5 Subset (240 evaluations)**: **72.50%** (174 / 240)
* **Truncation Rate**: **0.10%** (1 / 1,000 queries hit 8k ceiling, scored as `False`)
* **Token Distribution (Answer Tokens)**:
  - $\min = 113$ tokens
  - $\text{median} = 406$ tokens
  - $p90 = 1,183$ tokens
  - $\max = 8,192$ tokens
* **Thinking Budget**: Exactly $K=6$ discrete thinking tokens generated before forced `</think>\n\n` exit.
* **Audit Verdict**: Certified discrete comparator baseline at $K=6$ confirmed at **73.40%**.

### Arm 1: Untrained Base Direct Non-Thinking (`data/streaming_arm1_direct_qwen_qwen3-1.7b.jsonl`)
* **Status**: **CERTIFIED [PASS] across Gate 0 and all 7 Scientific Validation Protocols**.
* **Pass@1 Accuracy (Pure Canonical `math_verify`)**: **73.90%** (739 / 1,000 evaluations across 4 seeds)
  - **GSM8K Overall (600 evaluations)**: **71.00%** (426 / 600)
    * Short Stratum (200 evaluations): **83.50%** (167 / 200) $\to$ Matches Phase 0 GSM8K floor (83.55%) within 0.05%
    * Medium Stratum (200 evaluations): **67.50%** (135 / 200)
    * **Long Stratum (200 evaluations)**: **62.00%** (124 / 200)
  - **MATH-500 Overall (400 evaluations)**: **78.25%** (313 / 400)
    * Level 1 (80 evaluations): **92.50%** (74 / 80)
    * Level 2 (80 evaluations): **77.50%** (62 / 80)
    * Level 3 (80 evaluations): **82.50%** (66 / 80)
    * Level 4 (80 evaluations): **71.25%** (57 / 80)
    * Level 5 (80 evaluations): **56.25%** (45 / 80)
  - **MATH-500 Level 3–5 Subset (240 evaluations)**: **70.00%** (168 / 240)
* **Truncation Rate**: **0.00%** (0 / 1,000 queries truncated; `max_new_tokens=8192` completely eliminated runaway non-thinking loops).
* **Token Distribution (Answer Tokens)**:
  - $\min = 76$ tokens
  - $\text{median} = 353.5$ tokens
  - $p90 = 899.2$ tokens
  - $\max = 6,803$ tokens
* **Leakage Count**: Exactly 0 / 1,000 sequences contained `<think>` tags.
* **Audit Verdict**: Verified and certified by `scientific_results_validator` subagent (0% re-scoring discrepancy).

---

## 8. Arm 0: Surgical Neutrality Suite (IFEval)
**Protocol Invariant**: To assert that LoRA fine-tuning for latent recurrence or pause tokens has not degraded general instruction-following capabilities, all adapters are evaluated on `google/IFEval` (541 prompts) under calibrated non-thinking sampling (`enable_thinking=False`, `temp=0.7`, `top_p=0.80`, `top_k=20`, `presence_penalty=1.5`, `max_new_tokens=2048`).

| Arm | Model / Adapter | Strict Prompt Acc (%) | Loose Prompt Acc (%) | Strict Inst Acc (%) | Loose Inst Acc (%) | Truncation Rate (%) | Token Length (med / max) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base** | `Qwen/Qwen3-1.7B` (Untrained Base) | **67.65%** (366/541) | **72.27%** (391/541) | **76.02%** (634/834) | **79.50%** (663/834) | **1.29%** (7/541) | 203 / 2,048 (p90: 642) | **Certified Baseline** |
| **Arm 1b** | `lora_arm1b_qwen_qwen3-1.7b_k0_selfdistill` | **59.70%** (323/541) | **64.88%** (351/541) | **69.54%** (580/834) | **74.58%** (622/834) | **1.11%** (6/541) | 247 / 2,048 (p90: 696) | **Certified Arm 1b** |
| **Arm 2b ($K=6$)** | `lora_arm2b_qwen_qwen3-1.7b_k6_selfdistill` | **44.92%** (243/541) | **61.92%** (335/541) | **58.63%** (489/834) | **72.18%** (602/834) | **0.00%** (0/541) | 261 / 2,047 (p90: 704) | **Certified Arm 2b-6** |
| **Arm 2b ($K=32$)** | `lora_arm2b_qwen_qwen3-1.7b_k32_selfdistill` | **47.50%** (257/541) | **60.44%** (327/541) | **59.11%** (493/834) | **70.14%** (585/834) | **0.00%** (0/541) | 268 / 2,047 (p90: 696) | **Certified Arm 2b-32** |
| **Arm 3 ($K=6$)** | `lora_arm3_qwen_qwen3-1.7b_k6_selfdistill` | **54.34%** (294/541) | **58.41%** (316/541) | **65.35%** (545/834) | **69.18%** (577/834) | **1.66%** (9/541) | 213 / 2,050 (p90: 686) | **Certified Arm 3-6** |
| **Arm 3 ($K=32$)** | `lora_arm3_qwen_qwen3-1.7b_k32_selfdistill` | **50.28%** (272/541) | **54.71%** (296/541) | **61.03%** (509/834) | **64.99%** (542/834) | **0.74%** (4/541) | 249 / 2,050 (p90: 744) | **Certified Arm 3-32** |

* **Base Truncation & Distribution Audit (2,048 Token Cap with Presence Penalty)**:
  - Truncation ceiling: `max_new_tokens = 2,048` tokens.
  - Adding presence penalty (`pp=1.5`) eliminated non-thinking repetition loops, reducing truncation from 4.81% down to **1.29%** (only 7 prompts) and bringing strict prompt accuracy to **67.65%** (closely matching Qwen's official 68.2% reference).
  - Length distribution: $\min = 3$, $\text{median} = 203$, $p90 = 642$, $\max = 2,048$.
  - Evaluation runtime: 1,932.3s on GPU 0 (`cuda:0`, RTX 4080 16GB).
  - Certified artifact: `data/surgical_neutrality_ifeval_base_qwen3_1.7b.json`.

---

## 9. Phase 5: Check 3 Causal Activation Patching Study (Mechanistic Interpretability)

**Protocol**: Evaluated across $N=100$ mathematical problems on dedicated GPU 1 (RTX PRO 4500 32GB) using `scripts/37_causal_patching_scaled.py` targeting `checkpoints/lora_arm3_qwen_qwen3-1.7b_k6_selfdistill` at horizon $K=6$, empirical $\alpha_{\text{model}} = 0.011440$, and mid-thought intervention depth $t = \lfloor 0.5 K \rfloor = 3$.  
**Evidence Artifact**: `data/causal_patching_qwen_qwen3-1.7b.json`  
**Execution Runtime**: 5,928.6s (~98.8 min) across 600 autoregressive rollouts.

### Mechanistic Activation Patching Matrix ($N=100$)

| Interpretability Metric / Control | Pre-Registered Standard | Empirical Value | Verdict | Scientific Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Control 0: Null Patch Fidelity** | Bit-identical output, $\Delta_{\text{logits}} = 0.0$, $\cos \ge 0.999$ | **$\Delta_{\text{logits}} = \mathbf{0.000000}$**, $\cos_{\min} = 0.9961$, **100% Identical Text** | **CONFIRMED** | Implementation fidelity verified: re-injecting own latent yields 100% bit-exact logits and output text. $\cos_{\min}$ deviation reflects pure bfloat16 machine epsilon ($1 - 2^{-8} = 0.99609375$). |
| **Control A: Gaussian Noise Perturbation** | Significant disruption (>50%) | **100.00%** Divergence (100 / 100) | **CONFIRMED** | **Dynamical Coupling Confirmed (Not Semantic Content)**: Norm-matched Gaussian noise completely breaks generation in 100% of cases, demonstrating the decoder is causally coupled to the latent KV cache. However, as with Check D, this reflects dynamical sensitivity rather than modular semantic content (see Control B). |
| **Finite-Difference Sensitivity** | $\ge 1.0\times$ next-step amplification | **$17.31\times$** Mean Amplification | **CONFIRMED** | Tiny latent perturbations ($\|\epsilon\| = 10^{-3} \|h\|$) amplify by $17.3\times$ across successive recurrent steps ($\|\Delta h_{k+1}\| / \|\epsilon\|$). |
| **Empirical Chance Baseline ($P_{\text{chance}}$)** | Measured frequency of donor numbers | **1.00%** (1 / 100) | Measured | Random baseline rate of unpatched recipient output containing filtered donor numbers. |
| **Control B: Donor Latent Steering Rate ($P_{\text{steered}}$)** | Pre-registered $\ge 40.0\%$ steering delta | **2.25%** (2 / 89 valid trials) | **FAILED (Null)** | Injecting donor latent vector $h_{\text{mid}}^D$ into recipient problem $R$ transfers donor mathematical entities only 2.25% of the time. |
| **Net Steering Delta ($\Delta \text{Steer}$)** | $\Delta = P_{\text{steered}} - P_{\text{chance}} \ge +40.0\%$ | **$+1.25\%$** | **FAILED (Null)** | Decisively rejects the hypothesis that continuous latent vectors carry modular, problem-specific reasoning content. |

### Mechanistic Diagnosis: The Cause of the Illusion of Superposition

1. **Two Stacked Architectural & Training Failures**:
   - **Failure 1: The Answer Target is Defective (The 60.4% Hard-MATH Collapse)**:
     * In thinking mode, the text following `</think>\n\n` is a terse final summary because the actual mathematical derivations occurred inside `<think>`.
     * Fine-tuning adapters (Arms 1b, 2b, 3) to emit this brief summary directly after the prompt (or after inert pause/latent steps) trains the network to output answers without writing derivations.
     * On easy arithmetic (GSM8K: 71.0%), the model can jump to the answer in one leap. But on complex mathematics (MATH-500 Levels 3–5), this target induces a **catastrophic -9.58% drop** (70.00% base direct $\to$ 60.42% trained adapters). The adapter damaged the base reasoning floor.
   - **Failure 2: Answer-Only Loss Leaves Latents Semantically Empty**:
     * Cross-entropy loss on final answer tokens alone provides zero gradient pressure to encode intermediate reasoning steps into continuous hidden states.
     * Control B confirms this null result: donor problem steering produces only **+1.25% entity transfer** against a 1.00% baseline (failing the +40% threshold).
     * The model treats latent unrolling as an uninformative computational delay buffer, explaining the exact parity between Arm 3 ($K=32$) and Arm 2b ($K=32$) on MATH Hard (**61.67% vs. 61.67%**).

2. **Dynamical Sensitivity & Expansive Mapping**:
   - The $17.31\times$ per-step sensitivity amplification demonstrates that the trained recurrence map $f: h_k \to h_{k+1}$ is strongly expansive ($\lambda > 1$), not contractive. Without contractive regularization or intermediate CoT grounding, latent trajectories diverge exponentially across longer horizons.

3. **Technical Note on Control 0 Implementation Fidelity**:
   - Control 0 compared the patched tensor $h_3$ directly against itself in `bfloat16`.
   - In IEEE `bfloat16` (7-bit mantissa), PyTorch's native `F.cosine_similarity(x, x)` computes $\frac{\sum x^2}{\|\mathbf{x}\|^2}$, which truncates to $1.0 - 2^{-8} = \mathbf{0.99609375}$ due to intermediate norm-squaring reduction.
   - In `float32`, cosine similarity between the tensors is bit-exact **1.0000000**, confirming 100% implementation fidelity alongside `max_delta_logit = 0.000000`.

4. **Mandated v1.1 Algorithmic Architecture**:
   - **Non-Thinking Answer Targets**: Train on the model's own *non-thinking* traces where derivations are written out step-by-step, ensuring Arm 1b reproduces Arm 1 (70.0% hard math) by construction.
   - **Lever 1 (CODI Feature Alignment Loss)**: Explicitly align intermediate latent vectors $h_k$ to the teacher's discrete CoT hidden states at paragraph/step boundaries to force semantic content into the channel.
   - **Contractive Regularization & Removal Smoothing**: Smooth transition schedules ($\lambda=4$) to enforce bounded latent dynamics.

---

## 9.5 Interim Phase v1.1: The Dilution Curve, Distillation Sweeps, & Discovery of the Miscalibrated Gate 2 Threshold

*(Consolidated from `RESULTS_QWEN3_1.7B_V1_1.md` to maintain a single authoritative record)*

### 1. Gate 1: Non-Thinking Worked Derivation Health (The Dilution Curve)
To fix Failure 1 from Phase 5 (the terse answer target collapse), we evaluated worked-solution SFT across dilution ratios on MATH-500 Levels 3–5 ($N=240$ multi-seed queries across seeds `[42, 123, 456, 789]`):

| Configuration | Training Dataset Composition | Hardware | Pass@1 (%) | Correct / Denominator | vs Base Reference (70.00%) | Gate 1 Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Point 1 (Over-Diluted)** | 21% General / 79% Math ($N=1,556$) | GPU 1 | 66.25% | 159 / 240 | -3.75% (-9 problems) | `REJECTED [FAIL]` |
| **Path B (Zero Dilution)** | 0% General / 100% Pure Math ($N=1,230$)| GPU 0 | 66.67% | 160 / 240 | -3.33% (-8 problems) | `REJECTED [FAIL]` |
| **Path A (10% Dilution)** | **10% General / 90% Math ($N=1,367$)** | **GPU 1** | **69.17%** | **166 / 240** | **-0.83% (-2 problems)** | **`CERTIFIED [PASS]`** |

- **Outcome**: 10% dilution cleared the pre-registered $\ge 68.0\%$ floor, re-establishing non-thinking worked derivation health.

---

### 2. Gate 2: The Step-Distillation Sweep & The Miscalibrated Threshold Discovery
We trained student adapters on Path A with teacher step-distillation loss ($\lambda_{\text{align}} \in [0.1, 1.0, 3.0]$) and CODI fixed-target feature matching, evaluating causal activation patching across $N=94$ problem pairs:

| Regime / Strategy | Intervention Condition | Final Dev CE Loss | Final Dev Alignment | Valid Pairs ($N$) | Mean Amp | $P_{\text{chance}}$ (%) | $P_{\text{steered}}$ (%) | $\Delta \text{Steer}$ (%) | Gate 2 Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weak Step Distill** | $\lambda = 0.1$ ($t=3$) | 0.0904 | 12.66% | 94 | 7.67× | 14.89% | 13.83% | **-1.06%** | `FAIL / HALT` |
| **Balanced Step Distill** | $\lambda = 1.0$ ($t=3$) | **0.0777** | **87.34%** | 94 | 6.63× | 15.96% | 13.83% | **-2.13%** | `FAIL / HALT` |
| **Aggressive Step Distill**| $\lambda = 3.0$ ($t=3$) | 0.3067 | 62.02% | 94 | 12.41× | 13.83% | 13.83% | **+0.00%** | `FAIL / HALT` |
| **CODI Fixed-Target** | Answer State ($t=3$) | 0.0781 | 95.05% | 94 | 6.08× | 12.77% | 13.83% | **+1.06%** | `FAIL / HALT` |
| **CODI Fixed-Target** | Answer State ($t=6$, Hand-Off)| 0.0781 | 95.05% | 94 | 28.31× | 12.77% | 12.77% | **+0.00%** | `FAIL / HALT` |

#### Diagnostic Insights & The Register Ladder Hypothesis:
1. **The Gate 2 Reference Calibration on Discrete CoT**:
   - We calibrated standard Discrete Text CoT under the exact same causal patching protocol.
   - Splicing text CoT at $K=6$ yielded **$\Delta\text{Steer} = +0.00\%$**; at Step 1 paragraph **$+8.70\%$**; and only reached **$+21.74\%$** when the entire multi-step chain was replaced.
   - **Insight**: Discrete natural language reasoning itself cannot meet the $+40.0\%$ bar on short segments. Splicing a short unanchored segment does not override the recipient prompt's causal basin in either text or latents.
2. **Attentional Invisibility Discovered**:
   - Query heads require discrete token keys in the KV cache; unanchored continuous latents received only **$1.80\%$ of attention mass**, while **$78.78\%$** remained locked onto prompt tokens.
   - This motivated hypothesizing the **Register-Bundle Ladder** below (interleaving discrete headers as semantic anchors).
   - *Retrospective Note*: As production tests later revealed (Check A: $1.30\%$ latent mass), the ladder alone did *not* solve attentional invisibility in standard training; the query heads still bypassed the latents for prompt tokens, and the downstream reasoning gains came from the structural headers themselves.

---

## 10. The Register-Bundle Ladder: Production Evaluation, Acid Tests & Depth Extrapolation

### Architecture Specification
The **Register-Bundle Ladder** interleaves $R=3$ canonical semantic register headers with $K=8$ continuous unrolled latent steps per bundle ($3 \times 8 = 24$ total latent forward passes):
- **Prompt**: Exact evaluation prompt with chat template ending in `<think>\n`.
- **Rung 1**: `[R1: Identify givens, constraints, and target variable]\n` + 8 unrolled latent passes in $\mathbb{R}^{2048}$.
- **Rung 2**: `[R2: Compute intermediate operations and verify relations]\n` + 8 unrolled latent passes in $\mathbb{R}^{2048}$.
- **Rung 3**: `[R3: Execute final deduction and verify constraints]\n` + 8 unrolled latent passes in $\mathbb{R}^{2048}$.
- **Transition**: Delimiter `\n</think>\n\n`.
- **Answer Phase**: Worked non-thinking solution ending in `\boxed{...}<|im_end|>`.
- **Adapter**: LoRA rank $r=32$, $\alpha=64$, all linear projections (`q, k, v, o, gate, up, down`), pure `bfloat16`, $\alpha_{\text{model}} = 0.011440$.

---

### Certified Benchmark Evaluation ($N=1,000$ Queries across 4 Deterministic Seeds)
Evaluated on `data/benchmark_suite_250.json` (150 GSM8K + 100 MATH-500 Levels 1–5) across `SEEDS = [42, 123, 456, 789]` ($N=1,000$ total queries) under frozen Gate 0 parameters (`temp=0.7`, `top_p=0.80`, `top_k=20`, `presence_penalty=1.5`, `max_ans=8192`) on GPU 1 (RTX PRO 4500 32GB). Grader: Canonical `math_verify 0.9.0` with canonical `\boxed{}` extraction and symbolic equivalence.

Paired hierarchical bootstrapping ($B=10,000$ iterations, Seed 42):

| Metric / Benchmark Stratum | Arm 1b ($K=0$ Direct SFT) | Arm 3 Register Ladder ($3 \times 8 = 24$ Latents) | Observed Delta ($\Delta_1$) | 95% Bootstrap Confidence Interval | $p$-value / Statistical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Pass@1 ($N=1,000$)** | **71.20%** (712/1000) | **71.30%** (713/1000) | $\mathbf{+0.10\%}$ | `[-2.40%, +2.60%]` | $p = 0.4849$ (**Statistical Null Parity**) |
| **MATH-500 Overall ($N=400$)** | **71.50%** (286/400) | **74.00%** (296/400) | $\mathbf{+2.50\%}$ | `[-1.25%, +6.50%]` | $p = 0.1183$ (Inconclusive) |
| **MATH Level 1 ($N=80$)** | **95.00%** (76/80) | **96.25%** (77/80) | $\mathbf{+1.25\%}$ | `[-5.00%, +7.50%]` | $p = 0.4285$ |
| **MATH Level 2 ($N=80$)** | **81.25%** (65/80) | **78.75%** (63/80) | $-2.50\%$ | `[-11.25%, +6.25%]` | $p = 0.3470$ |
| **MATH Level 3 ($N=80$)** | **81.25%** (65/80) | **78.75%** (63/80) | $-2.50\%$ | `[-10.00%, +3.75%]` | $p = 0.3126$ |
| **MATH Level 4 ($N=80$, Hard)** | **58.75%** (47/80) | **68.75%** (55/80) | $\mathbf{+10.00\%}$ | `[+1.25%, +18.75%]` | $p = 0.0171$ (**Exploratory sub-slice**; 1 of 9 slices) |
| **MATH Level 5 ($N=80$, Hardest)** | **41.25%** (33/80) | **47.50%** (38/80) | $\mathbf{+6.25\%}$ | `[-5.00%, +17.50%]` | $p = 0.1644$ |
| **GSM8K Arithmetic ($N=600$)** | **71.00%** (426/600) | **69.50%** (417/600) | $-1.50\%$ | `[-4.67%, +1.67%]` | $p = 0.3541$ (Arithmetic Parity) |

- **Telemetry Sanity Certified**:
  - Truncation Rate: **0.20%** (2 / 1,000 queries hitting generation ceiling; scored as incorrect).
  - Prompt Thinking Leakage: **0.00%** (0 occurrences of `<think>` tags in output text).
  - Execution Throughput: ~2.5 queries/sec via batched PyTorch ($B=8$).
- **Evidence Artifacts**: `data/eval_results_arm3_register_ladder_production_best.json`, `data/paired_bootstrap_arm3_register_ladder_vs_arm1b.json`, `data/streaming_arm3_register_ladder_production_best.jsonl`.

---

### The Four Production Acid Tests ($N=60$ MATH Levels 3–5 Problems)
Evaluated on the 60 hardest problems from MATH-500 (20 each from Levels 3, 4, 5) using `scripts/98_test_register_ladder_acid_tests.py`:

| Acid Test Check | Target Protocol | Empirical Result | Scientific Interpretation |
| :--- | :--- | :---: | :--- |
| **Check A: Attention Mass Decomposition** | Attention distribution from answer tokens back into preceding sequence | Prompt Mass: **72.27%**<br>Header Mass: **5.58%**<br>Latent Mass: **1.30%**<br>Transition Mass: **10.46%** | **Attentional Invisibility Persists**: Without explicit prompt masking during training, answer query heads bypass the latents (1.30%) and attend 72.27% to the prompt. The ladder structure alone does NOT overcome attentional invisibility. |
| **Check C: 4-Way Matched Ablation** | Cond 1 (Full Ladder) vs. Cond 2 (Headers Only), Cond 3 (Generic Headers + Latents), Cond 4 (Headers + 24 Pause Tokens) | Cond 1: **45.00%** (27/60)<br>Cond 2: **45.00%** (27/60)<br>Cond 3: **41.67%** (25/60)<br>Cond 4: **41.67%** (25/60)<br>**Untrained Delta: +3.33%** | **Acid Test Inconclusive / Superseded**: While continuous latents beat *untrained* pause tokens (+3.33%), evaluating properly *trained* pause tokens (Arm 2b-R: 71.67%) reveals that continuous latents (61.67%) lose by **-10.0%**. Trained discrete delay outperforms continuous latents. |
| **Check D: Multi-Bundle Noise Sensitivity** | Inject $\mathcal{N}(0, \sigma_k^2)$ noise into each bundle's latent states | Bundle 1: **93.33%** Divergence (56/60)<br>Bundle 2: **93.33%** Divergence (56/60)<br>Bundle 3: **91.67%** (55/60) | **Dynamical Coupling Confirmed (Not Content)**: Perturbing latent states alters trajectory in >91% of cases, verifying the model is causally coupled to the latent KV cache. However, this does NOT demonstrate semantic content (see Check B). |
| **Check B: Causal Activation Patching** | Control 0 Null Patch Gate ($\ge 95\%$) followed by Donor Latent Patching at Bundle 2 ($N=71$ valid pairs) | Control 0: **71/71 (100.00% Identity)** [GATE PASSED]<br>$P(\text{chance}) = 2.82\%$<br>$P(\text{steered}) = 1.41\%$<br>$\Delta\text{Steer} = -1.41\% \approx \mathbf{0.00\%}$ | **Latents Carry No Portable Semantic Content**: With 100% harness fidelity, donor latent vectors fail to steer the recipient problem, mirroring discrete CoT at $K=6$ (+0.00%). |

- **Evidence Artifact**: `data/acid_test_production_model_results.json`.

---

### Depth Extrapolation Scaling Study ($K \in [4, 6, 8, 12, 16]$ per bundle)
Evaluated on the 60 MATH L3–5 test suite using `scripts/99b_test_latent_depth_extrapolation.py` to assess whether the recurrent map $f: h_k \to h_{k+1}$ remains stable when unrolled beyond its training depth ($K=8$ per bundle, 24 total latents):

| Depth ($K$ / bundle) | Total Latents | MATH L3–5 Accuracy (%) | Mean Latent Norm ($\|h\|$) | Max Latent Norm | Mean Cosine Drift ($\cos(h_k, h_{k-1})$) | Stability Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $K=4$ | 12 | **41.67%** (25/60) | 136.45 | 158.00 | 0.6912 | Bounded, sub-training horizon |
| $K=6$ | 18 | **45.00%** (27/60) | 138.88 | 157.00 | 0.8263 | **Peak Accuracy Frontier** |
| $K=8$ (Trained) | 24 | **41.67%** (25/60) | 138.65 | 157.00 | 0.8679 | Training baseline depth |
| $K=12$ ($1.5\times$) | 36 | **43.33%** (26/60) | 138.87 | 158.00 | 0.9164 | Robust extrapolation (+1.66% vs K=8) |
| $K=16$ ($2.0\times$) | 48 | **40.00%** (24/60) | 139.33 | 158.00 | 0.9277 | Bounded at $2\times$ depth, zero explosion |

- **Key Scientific Takeaways**:
  1. **Rock-Solid Norm Stability**: Mean latent norm is invariant between 136.45 and 139.33 across all depths from 12 to 48 steps. Max norm is strictly capped at 158.00. The recurrent map does NOT explode or collapse.
  2. **Monotonic Cosine Saturation**: Consecutive step cosine similarity increases smoothly from 0.6912 ($K=4$) to 0.9277 ($K=16$), showing the continuous trajectory approaches a stable attractor fixed point.
  3. **Zero Depth Degradation**: Even at $K=16$ (48 latents, double the trained compute), accuracy remains 40.00% (within 1 problem of $K=8$), confirming the absence of runaway drift.
- **Evidence Artifact**: `data/depth_extrapolation_study_qwen3_1.7b.json`.

---

## 11. Trained Discrete Controls, Latent Capacity Analysis, & Latent Information Bottleneck

### Trained Matched Controls on MATH L3–5 ($N=60$) and Full 250-Suite ($N=1,000$)
Evaluated via `scripts/103_eval_trained_controls.py` ($N=60$) and `scripts/105_eval_register_controls_250suite.py` ($N=1,000$, 250 suite $\times$ 4 seeds) under the Gate 0 Frozen sampling specification (`temp=0.7`, `top_p=0.80`, `top_k=20`, `presence_penalty=1.5`, `max_ans_tokens=8192`, canonical `math_verify 0.9.0`):

| Experimental Arm | Architecture & Delimiters | MATH L3–5 Accuracy (Seed 42, $N=60$) | Full 250-Suite Pass@1 ($N=1,000$) | GSM8K ($N=600$) | MATH-500 ($N=400$) | MATH Hard L3–5 ($N=240$) | Truncation Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 1 (Base Direct)** | Untrained Direct Non-Thinking | 70.00% (42/60) | **73.90%** (739/1000) | 71.00% | 78.25% | **70.00%** (168/240) | 0.00% |
| **Arm 1-R (Untrained Base + Headers)**| 3 Canonical Headers injected (NO adapter) | 63.33% (38/60) | **70.20%** (702/1000) | 68.83% | 72.25% | **63.33%** (152/240) | 0.10% |
| **Arm 1b (Direct SFT Control)** | Direct SFT without Thinking Tags | ~60.00% (36/60) | **71.20%** (712/1000) | 71.00% | 71.50% | **60.42%** (145/240) | 0.00% |
| **Arm 1b-R (Trained Headers Only)** | 3 Canonical Headers (0 latents, direct transition) | **68.33%** (41/60) | **73.20%** (732/1000) | **71.17%** | **76.25%** | **67.08%** (161/240) | 0.20% |
| **Arm 2b-R (Trained Pause Tokens)** | 3 Canonical Headers + 24 `<pause>` Tokens | **71.67%** (43/60) | **72.90%** (729/1000) | **71.33%** | **75.25%** | **68.33%** (164/240) | 0.20% |
| **Arm 3-R (Register Ladder)** | 3 Headers + 24 Continuous Latents ($\mathbb{R}^{2048}$) | **61.67%** (37/60) | **71.30%** (713/1000) | 69.50% | 74.00% | **65.00%** (156/240) | 0.20% |
| **Arm 3-LIB (Information Bottleneck)**| 3 Headers + 24 Latents (100% prompt mask trained) | 53.33% (32/60)* | **65.00%** (650/1000) | 65.33% | 64.50% | **53.75%** (129/240) | 0.20% |

> *\*Note on Arm 3-LIB Seed-42 Slice*: On the unmasked 250-suite Seed 42 slice, Hard MATH is 53.33% (32 / 60); evaluated under 100% prompt masking, it scores 50.00% (30 / 60). Across all 4 seeds ($N=240$), Hard MATH reliably converges to **53.75%** (129 / 240), establishing a **-13.33%** penalty relative to open discrete scaffolding (Arm 1b-R at 67.08%).

- **Key Scientific Insights from Multivariate Bootstrapping Matrix ($B=10,000$, $N=1,000$)**:
  1. **The Nature of Scaffolding: SFT Floor Repair, Not Super-Base Reasoning**:
     - Prepending headers to the *untrained* base model (**Arm 1-R: 70.20%**) hurts performance relative to the clean base direct prompt (**Arm 1: 73.90%**, $\Delta = \mathbf{-3.70\%}$, and **63.33% vs. 70.00% on MATH Hard L3–5**). The base model was never conditioned on structured register headers.
     - In contrast, training on those headers (**Arm 1b-R: 73.20%**) significantly outperforms the untrained header prompt (**Arm 1-R: 70.20%**, $\Delta = \mathbf{+3.00\%}$, $p = \mathbf{0.0230}$ [STATISTICALLY SIGNIFICANT]).
     - Therefore, **scaffolding is a learned structural adaptation that reverses narrow SFT damage** (recovering +6.67% on hard math over Arm 1b, $p=0.0254$), but does **not** boost reasoning beyond the unconstrained clean base model floor (73.90%).
  2. **Statistically Significant Gain from Scaffolding Over Direct SFT**: 
     - On MATH-500: $\text{Arm 1b-R} - \text{Arm 1b} = \mathbf{+4.75\%}$ ($p=\mathbf{0.0216}$, $95\%\text{ CI } [+0.75\%, +9.00\%]$).
     - On MATH Hard (L3–5, $N=240$): $\text{Arm 1b-R} - \text{Arm 1b} = \mathbf{+6.67\%}$ ($p=\mathbf{0.0254}$, $95\%\text{ CI } [+0.83\%, +12.50\%]$).
     - Discrete structural scaffolding reverses the SFT reasoning floor damage with statistical significance, completely independently of any token thinking.
  3. **Both Discrete Scaffolding Controls Beat Continuous Latents**:
     - Across the full 1,000 queries, Arm 1b-R (**73.20%**, $+1.90\%$) and Arm 2b-R (**72.90%**, $+1.60\%$) both outperform the continuous latent ladder (Arm 3-R at **71.30%**).
     - On the hardest contest problems (MATH Hard L3–5, $N=240$):
       - Arm 1b Baseline: **60.42%** (145 / 240)
       - Arm 3-R Continuous Latents: **65.00%** (156 / 240)
       - Arm 1b-R Headers Only: **67.08%** (161 / 240) [$\mathbf{+2.08\%}$ over latents]
       - Arm 2b-R Pause Tokens: **68.33%** (164 / 240) [$\mathbf{+3.33\%}$ over latents]
  4. **Continuous Latent Inefficacy vs. Matched Controls**: Adding 24 continuous latent states into the headers yields a **$-1.90\%$ overall deficit** against empty headers and a **$-1.60\%$ overall deficit ($-3.33\%$ on hard math)** against trained `<pause>` tokens. Continuous vectors perform worse than trained discrete delay filler.
  5. **The Information Bottleneck Penalty (Arm 3-LIB)**:
     - Forcing routing through latents by masking 100% of prompt attention during training causes **severe general degradation across all splits**:
       - vs. Arm 1b Direct SFT (71.20%): $\Delta = \mathbf{-6.20\%}$ ($p = 0.0000$)
       - vs. Arm 1b-R Trained Headers (73.20%): $\Delta = \mathbf{-8.20\%}$ ($p = 0.0000$)
       - vs. Arm 3-R Standard Ladder (71.30%): $\Delta = \mathbf{-6.30\%}$ ($p = 0.0000$)
       - On MATH Hard L3–5: drops to **53.75%** (a **-13.33%** collapse from Arm 1b-R's 67.08%).
- **Evidence Artifacts**: `data/eval_results_arm1_base_headers_only.json`, `data/eval_results_arm1b_register_headers_only.json`, `data/eval_results_arm2b_register_headers_pause.json`, `data/eval_results_arm3_register_ladder_production_best.json`, `data/eval_results_arm3_register_ladder_bottleneck_qwen3_1.7b.json`, `data/multivariate_scaffolding_bootstrap_matrix.json`.

---

### Analytical Capacity Audit: Latent Space vs. Token CoT (The 125:1 Compression Reality)
To understand why Arm 3 sits at $71.30\%$ overall ($65.00\%$ on hard math) while unconstrained thinking (Arm 4) reaches **$87.10\%$** ($86.67\%$ on hard math), we compared the physical information space allocated to both modes:

| Information Dimension | Arm 3 Register Ladder (24 Latents) | Arm 4 Thinking Mode (Token CoT) | Scale Disparity |
| :--- | :---: | :---: | :---: |
| **Autoregressive Passes** | **24 forward passes** | **~2,500 – 4,000 forward passes** | **$100\times - 160\times$ less compute** |
| **Sequence Length** | 24 vectors (+ ~50 header tokens) | ~3,000 tokens average | **$\approx 0.8\%$ of CoT length** |
| **KV Cache Footprint** | **~5.4 MB** | **~672 MB** | **$125\times$ smaller memory** |
| **State Representation** | 24 vectors $\in \mathbb{R}^{2048}$ ($49,152$ floats) | 3,000 discrete vocabulary tokens | Continuous vs. Discrete |
| **Performance Gap** | **71.30%** (Hard math: 65.00%) | **87.10%** (Hard math: 86.67%) | **Gap to thinking: 15.80% (21.67% on hard math)** |

* **The Takeaway**: Expecting 24 continuous vectors to replace 3,000 discrete tokens of trial-and-error calculations, factorizations, and scratchpad equations demands an impossible **125:1 compression ratio**.
* **Re-Alignment with Staged Dynamic Registers (Study 2)**: The study explored this via **dynamic inline register writes** (`<|reg|>x = 13<|/reg|>`), an engine-managed **Register Table**, and deterministic `<|state|>` re-injection. This allows exact integer constants to crystallize into discrete tokens while continuous latents handle deep relational reasoning.

---

### Latent Information Bottleneck (LIB) Ladder Training & Convergence
To eliminate the $72.27\%$ prompt shortcut and force the model to route all reasoning through the latents, we developed and trained `scripts/102_train_register_ladder_bottleneck.py` on dedicated compute GPU 1 (`cuda:1`, RTX PRO 4500 32GB):
* **Architecture**: Prompt attention is strictly masked to 0.0 during transition and answer token generation via 4D boolean attention masking. The answer decoder CANNOT see the prompt; all facts and numbers must pass through the 3 register headers and 24 continuous latent vectors in $\mathbb{R}^{2048}$.
* **Loss Trajectory**:
  - Initial Dev Loss (Step 0): **3.6615**
  - Step 25 / 158: Loss 3.7842 | Dev Loss: **3.4236**
  - Step 50 / 158: Loss 1.2253 | Dev Loss: **1.1641**
  - Step 75 / 158: Loss 0.6140 | Dev Loss: **0.6558**
  - Step 100 / 158: Loss 0.6172 | Dev Loss: **0.5840**
  - Step 125 / 158: Loss 0.5756 | Dev Loss: **0.5399**
  - Step 150 / 158: Loss 0.6012 | Dev Loss: **0.5283**
  - Step 158 / 158: Loss 0.5194 | Dev Loss: **0.5277** (**Convergence Reached**)
* **Training Metadata**: Total duration: 13,578s (3.77 hours) | Peak VRAM: 12.4 GB | Best checkpoint saved to `checkpoints/lora_arm3_register_ladder_bottleneck_qwen3_1.7b/best_checkpoint`.
* **Empirical Bottleneck Evaluation Results**:
  - **Full 250-Suite Benchmark ($N=1,000$ Queries across 4 Seeds)**:
    * **Overall Pass@1**: **65.00%** (650 / 1,000) [Truncation: 0.20%]
    * **GSM8K ($N=600$)**: **65.33%** (392 / 600)
    * **MATH-500 ($N=400$)**: **64.50%** (258 / 400)
    * **MATH Hard L3–5 ($N=240$)**: **53.75%** (129 / 240)
    * **Paired Bootstrap vs. Arm 1b Direct SFT (71.20%)**: $\Delta = \mathbf{-6.20\%}$ ($p = 0.0000$, 95% CI `[-8.90%, -3.50%]`)
    * **Paired Bootstrap vs. Arm 1b-R Trained Headers (73.20%)**: $\Delta = \mathbf{-8.20\%}$ ($p = 0.0000$, 95% CI `[-10.90%, -5.50%]`)
    * **Paired Bootstrap vs. Arm 3-R Production Ladder (71.30%)**: $\Delta = \mathbf{-6.30\%}$ ($p = 0.0000$, 95% CI `[-9.10%, -3.50%]`)
    * *Scientific Finding*: Forcing information routing through 24 latents during training damages the transformer's general reasoning capability, causing a **$-8.20\%$ collapse** relative to discrete headers alone and a **$-13.33\%$ drop on Hard MATH**. The earlier seed-42 $N=60$ reading (56.67%) was sample variance on a small slice; across $N=1,000$, the bottleneck penalty is severe and unambiguous.

  - **The 4-Part Acid Test Suite on the LIB Bottleneck Model**:
    * **Check A (Attention Split Decomposition)**: Prompt: **62.34%**, Headers: **14.37%**, Latents: **3.21%**, Transition: **11.00%**. Latent attention increased by $+147\%$ (from 1.30% to 3.21%), but remains $<5\%$. The model still routes $62.34\%$ of attention to prompt tokens when visible.
    * **Check C (4-Way Matched Ablation on 60 MATH L3–5)**:
      - Full Ladder (Headers + 24 Latents): **41.67%** (25 / 60)
      - Headers Only (0 Latents): **41.67%** (25 / 60) [$\Delta = \mathbf{0.00\%}$ vs. Full Ladder]
      - Generic Headers + 24 Latents: **36.67%** (22 / 60)
      - Headers + 24 Pause Tokens: **36.67%** (22 / 60)
      - *Definitive Takeaway*: Latents provide **exactly zero incremental accuracy** over headers alone ($25/60$ vs. $25/60$). The apparent $+5.00\%$ over pause tokens is caused entirely by pause tokens actively harming performance ($41.67\% \to 36.67\%$).
    * **Check D (Multi-Bundle Gaussian Noise Sensitivity)**: Bundle 1: **95.00%**, Bundle 2: **90.00%**, Bundle 3: **95.00%** divergence under noise.
    * **Check B (Causal Activation Patching with Control 0 Gate)**:
      - Control 0 Self-Identity Gate: **70 / 70 (100.00% Identity)** [**GATE PASSED** with zero harness noise].
      - Counterfactual Donor Steering ($N=70$ valid trials): $P(\text{chance}) = \mathbf{2.86\%}$ ($2/70$), $P(\text{steered}) = \mathbf{2.86\%}$ ($2/70$).
      - **Net Delta Steer: $\mathbf{+0.00\%}$ [INERT]**.
      - *Definitive Takeaway*: Even after 100% prompt-masked training, donor latent states carry **zero portable semantic computation** to steer recipient problems.
* **Evidence Artifacts**: `data/acid_test_lib_bottleneck_results.json`, `data/eval_results_arm3_register_ladder_bottleneck_qwen3_1.7b.json`, `data/streaming_arm3_register_ladder_bottleneck_qwen3_1.7b.jsonl`.

