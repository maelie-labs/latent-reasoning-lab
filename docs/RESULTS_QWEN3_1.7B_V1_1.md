# Continuous Latent Recurrence v1.1 Results: Qwen/Qwen3-1.7B

> [!WARNING]
> **CONSOLIDATED & SUPERSEDED HISTORICAL DOCUMENT**:
> This document records the interim v1.1 dilution experiments and unanchored step-distillation diagnostics. All certified, authoritative results for `Qwen/Qwen3-1.7B` (including the resolution of the miscalibrated Gate 2 threshold, the Register-Bundle Ladder production benchmarks, trained controls, and the Latent Information Bottleneck) have been consolidated into the canonical master document:
> **[`docs/RESULTS_QWEN3_1.7B.md`](RESULTS_QWEN3_1.7B.md)**.
> Please refer to `RESULTS_QWEN3_1.7B.md` for current results.

**Model Architecture**: `Qwen/Qwen3-1.7B` ($d_{\text{model}} = 2048$, 28 layers, 16 Q heads, 8 KV heads)  
**Calibrated Empirical Scale Factor**: $\alpha = 0.011440$  
**Harness Version**: Continuous Latent Recurrence v1.1 (Grounded Latent Recurrence & Calibrated Dilution)  
**Hardware Infrastructure**: Dedicated Compute GPU 1 (NVIDIA RTX PRO 4500 Blackwell 32GB, `--mem-fraction-static 0.90`) & GPU 0 (RTX 4080 16GB)  
**Supervisor Log**: [`logs/overnight/master_20260914_003704.log`](../logs/overnight/master_20260914_003704.log)  

---

## 1. Executive Summary & Core Scientific Findings

Continuous Latent Recurrence v1.1 was deployed to test two primary scientific hypotheses:
1. **The Dilution Hypothesis**: Non-thinking worked derivation health on competition math (MATH-500 Levels 3–5) is evaluated across dilution levels (0% Pure Math, 10% Dilution, 21% Dilution).
2. **The Grounding Hypothesis**: Augmenting the latent unroll loss with step-level teacher state distillation ($\lambda_{\text{align}} \cdot L_{\text{distill}}$) tests whether continuous latents can be forced to encode causal, semantic reasoning trajectories rather than acting as ungrounded representation drift or passive pause tokens.

### Key Empirical Findings:
1. **Gate 1 Dilution Curve (10% Cleared Gate; Curve Unresolved at $n=240$)**:
   - **Point 1 (21% Dilution)**: 66.25% (159/240): `REJECTED [FAIL]`.
   - **Point 2 (Path A: 10% Dilution)**: **69.17% (166/240)**: **`CERTIFIED [PASS]`**. Cleared the pre-registered $\ge 68.0\%$ floor.
   - **Point 3 (Path B: 0% Pure Math)**: 66.67% (160/240): `REJECTED [FAIL]`.
   - *Scientific Honesty on Curve*: While Path A cleared the pre-registered floor, the curve remains unresolved at $n=240$ ($p=0.20$ difference between 69.17% and 66.67% is within statistical variation of a 6-to-8 problem delta). The v1.1 training set is also smaller than v1.0 (1,230 math traces vs 2,070), and the worked-solution targets are self-rewritten derivations the model has not natively produced before. Either factor could account for the Level 4 deficit as easily as dilution.
2. **Gate 2 Causal Activation Patching at $\lambda_{\text{align}}=0.1$ ($\Delta \text{Steer} = -1.06\%$, Pre-Registered FAIL/HALT)**:
   - Evaluated across 94 valid counterfactual problem pairs ($D \to R$ hidden state injection at step $t=3$).
   - Control 0 (Null Patch Identity): **100.0% (94/94)** deterministic execution.
   - Finite-difference amplification: **7.67×** (down from $17.31\times$ in v1.0), providing mild empirical evidence that step-level distillation alters the dynamical landscape by contracting the expansive map.
   - $P_{\text{chance}} = 14.89\%$ vs. $P_{\text{steered}} = 13.83\%$. (Note: $P_{\text{chance}}$ jumped from 1.0% in v1.0 to 14.89% because multi-step worked-solution targets introduce substantially higher numerical token density).
   - **Net Steering Delta**: $\Delta \text{Steer} = \mathbf{-1.06\%}$ (Pre-registered threshold: $\ge +40.0\%$ for PASS, $\ge +10.0\%$ for PARTIAL). Verdict: `REJECTED [FAIL / HALT]`.
3. **Procedural Correction & $\lambda_{\text{align}}$ Diagnostic**:
   - **Flawed Sweep Selection**: The prior 200-step sweep chose $\lambda=0.1$ by evaluating total loss ($L_{\text{CE}} + \lambda L_{\text{distill}}$), which mathematically biased selection to the smallest multiplier $\lambda$. Consequently, the tested Arm 3 adapter had the distillation term effectively dialed down to noise ($0.1 \times 7.23 \approx 0.72$ vs CE $\approx 4.65$), yielding ungrounded latents with higher dev loss (0.0904) than pause tokens (0.0774).
   - **Enforcing Terminal Gate Halts**: The overnight runner erroneously progressed past Gate 2 FAIL to train controls and start the 250-suite benchmark. The benchmark suite was immediately terminated. The runner and evaluator scripts have been strictly updated so that Gate 2 failure is terminal (`sys.exit(1)`).
   - **Active Retraining**: Arm 3 ($K=6$) is being retrained at full alignment weights ($\lambda=1.0$ and $\lambda=3.0$) on Path A data. Gate 2 will be re-evaluated on both before any downstream suite is allowed to run.

---

## 2. Gate 1: Non-Thinking Worked Derivation Health (Dilution Curve)

**Pre-Registered Assertion**: MATH-500 Levels 3–5 multi-seed evaluation ($N=240$ queries across seeds `[42, 123, 456, 789]`) must achieve $\text{Pass@1} \ge \mathbf{68.00\%}$ to clear the gate and prove non-thinking worked derivation health before downstream adapter training.

| Configuration | Training Dataset Composition | Hardware | Pass@1 (%) | Correct / Denominator | vs Base Reference (70.00%) | Gate 1 Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Point 1 (Over-Diluted)** | 21% General / 79% Math ($N=1,556$) | GPU 1 | 66.25% | 159 / 240 | -3.75% (-9 problems) | `REJECTED [FAIL]` |
| **Path B (Zero Dilution)** | 0% General / 100% Pure Math ($N=1,230$)| GPU 0 | 66.67% | 160 / 240 | -3.33% (-8 problems) | `REJECTED [FAIL]` |
| **Path A (10% Dilution)** | **10% General / 90% Math ($N=1,367$)** | **GPU 1** | **69.17%** | **166 / 240** | **-0.83% (-2 problems)** | **`CERTIFIED [PASS]`** |

*Note*: 10% cleared the pre-registered gate floor; the curve is unresolved at $n=240$.

**Artifact References**:
- Official Certification Report: [`docs/GATE1_DILUTION_CURVE_CERTIFICATION_REPORT.md`](GATE1_DILUTION_CURVE_CERTIFICATION_REPORT.md)
- Path A Report: [`data/gate1_arm1b_dilution10pct_report.json`](../data/gate1_arm1b_dilution10pct_report.json)
- Path B Report: [`data/gate1_arm1b_puremath_report.json`](../data/gate1_arm1b_puremath_report.json)
- Point 1 Report: [`data/gate1_arm1b_health_report.json`](../data/gate1_arm1b_health_report.json)

---

## 3. Gate 2: Causal Donor Steering Assertion

**Evaluator**: [`scripts/88_gate2_causal_patching_v1_1.py`](../scripts/88_gate2_causal_patching_v1_1.py)  
**Report Artifact**: [`data/gate2_causal_patching_report.json`](../data/gate2_causal_patching_report.json)  
**Official Certification Report**: [`docs/GATE2_CAUSAL_PATCHING_CERTIFICATION_REPORT.md`](GATE2_CAUSAL_PATCHING_CERTIFICATION_REPORT.md)  
**Pre-Registered Criteria**:
- $\Delta \text{Steer} \ge +40.0\% \implies \mathbf{PASS}$ (Strong semantic grounding)
- $10.0\% \le \Delta \text{Steer} < 40.0\% \implies \mathbf{PARTIAL}$ (Partial grounding)
- $\Delta \text{Steer} < +10.0\% \implies \mathbf{FAIL / HALT}$ (Latent channel mechanically bypassed)

#### Empirical Comparison Across All Alignment & Read-Side Regimes:

| Regime / Strategy | Intervention Condition | Final Dev CE Loss | Final Dev Alignment | Valid Pairs ($N$) | Mean Amp | $P_{\text{chance}}$ (%) | $P_{\text{steered}}$ (%) | $\Delta \text{Steer}$ (%) | Gate 2 Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Weak Step Distill** | $\lambda = 0.1$ ($t=3$) | 0.0904 | 12.66% | 94 | 7.67× | 14.89% | 13.83% | **-1.06%** | `FAIL / HALT` |
| **Balanced Step Distill** | $\lambda = 1.0$ ($t=3$) | **0.0777** | **87.34%** | 94 | 6.63× | 15.96% | 13.83% | **-2.13%** | `FAIL / HALT` |
| **Aggressive Step Distill**| $\lambda = 3.0$ ($t=3$) | 0.3067 | 62.02% | 94 | 12.41× | 13.83% | 13.83% | **+0.00%** | `FAIL / HALT` |
| **CODI Fixed-Target** | Answer State ($t=3$) | 0.0781 | 95.05% | 94 | 6.08× | 12.77% | 13.83% | **+1.06%** | `FAIL / HALT` |
| **CODI Fixed-Target** | Answer State ($t=6$, Hand-Off)| 0.0781 | 95.05% | 94 | 28.31× | 12.77% | 12.77% | **+0.00%** | `FAIL / HALT` |
| **Attention Bottleneck** | Prompt Masked ($t=3$) | 0.0977 | 86.55% | 94 | 5.70× | 13.83% | 13.83% | **+0.00%** | `FAIL / HALT` |
| **Attention Bottleneck** | Prompt Masked ($t=6$, Hand-Off)| 0.0977 | 86.55% | 94 | 26.46× | 13.83% | 13.83% | **+0.00%** | `FAIL / HALT` |

#### Scientific Deductions & Negative Mechanism Certification on Qwen3-1.7B:
1. **The "Passive Alignment" Paradox Certified**:
   - In CODI fixed-target training, the student latent reached **95.05% cosine alignment** to the teacher's answer-position state.
   - Despite this near-identical representation and a 28.31x finite-difference amplification at $t=6$, patching donor latents produced **identically 0.00% net steering** ($P_{\text{steered}} = 12.77\% \equiv P_{\text{chance}} = 12.77\%$).
2. **Attention Bottlenecking Ineffective at 1.7B Parameter Scale**:
   - Forcibly zeroing out prompt attention during training (Steps 0–241) and relaxing it over the final 100 steps trained the model to generate fluent answers from latents during masked steps, but upon relaxation, the attention heads immediately defaulted back to prompt tokens.
   - At both $t=3$ and $t=6$, donor latent injection produced $\Delta \text{Steer} = \mathbf{+0.00\%}$.
3. **Formal Negative Mechanism Certification**:
   - Across 7 independent experimental conditions spanning loss weighting, target position, and architectural attention masking, the continuous latent recurrence channel on `Qwen/Qwen3-1.7B` cannot be causally bound to downstream answer generation.
   - The decoder systematically bypasses the latent vectors, treating them as computational delay buffers (interchangeable with `<pause>` tokens).
4. **Terminal Action**:
   - In accordance with the pre-registered decision boundary, Gate 2 is officially classified as **TERMINAL FAIL [HALT]**.
   - The downstream 1,000-query 250-suite benchmark is **permanently aborted** for Qwen3-1.7B to prevent decorative benchmarking on a non-functional channel.
   - The research program formally pivots to pilot **`Qwen/Qwen3-4B`**.

---

## 4. Final Adapter Registry

| Arm | Description | Alignment / Mechanism | Checkpoint Path | Status / Dev Loss | Gate 2 Status |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **Arm 1b** | Direct Worked SFT Control | N/A | [`checkpoints/lora_arm1b_v1_1_dilution10pct_qwen3_1.7b_k0`](../checkpoints/lora_arm1b_v1_1_dilution10pct_qwen3_1.7b_k0) | **Certified** (Gate 1: 69.17%) | N/A |
| **Arm 2b** | Pause Tokens ($K=6$) | N/A | [`checkpoints/lora_arm2b_v1_1_qwen3_1.7b_k6`](../checkpoints/lora_arm2b_v1_1_qwen3_1.7b_k6) | **Trained** (Dev: 0.0774) | N/A |
| **Arm 2b** | Pause Tokens ($K=32$) | N/A | [`checkpoints/lora_arm2b_v1_1_qwen3_1.7b_k32`](../checkpoints/lora_arm2b_v1_1_qwen3_1.7b_k32) | **Trained** (Dev: 0.0797) | N/A |
| **Arm 3** | Grounded Latents ($K=6$) | $\lambda=0.1$ | [`checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6`](../checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6) | Trained (Dev CE: 0.0904) | `FAIL` ($\Delta = -1.06\%$) |
| **Arm 3** | Grounded Latents ($K=6$) | $\lambda=1.0$ | [`checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_lambda1.0`](../checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_lambda1.0) | Trained (Dev CE: 0.0777, Cos: 87.3%) | `FAIL` ($\Delta = -2.13\%$) |
| **Arm 3** | Grounded Latents ($K=6$) | $\lambda=3.0$ | [`checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_lambda3.0`](../checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_lambda3.0) | Trained (Dev CE: 0.3067, Cos: 62.0%) | `FAIL` ($\Delta = +0.00\%$) |
| **Arm 3** | CODI Fixed-Target ($K=6$) | $\lambda=1.0$, $h_6 \to \text{teacher}$ | [`checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_codi_fallback`](../checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_codi_fallback) | Trained (Dev CE: 0.0781, Cos: 95.1%) | `FAIL` ($\Delta = +0.00\%$) |
| **Arm 3** | Attention Bottleneck ($K=6$) | $\lambda=1.0$, Prompt Masked | [`checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_bottleneck`](../checkpoints/lora_arm3_v1_1_qwen3_1.7b_k6_bottleneck) | Trained (Dev CE: 0.0977, Cos: 86.6%) | `FAIL` ($\Delta = +0.00\%$) |

---

## 5. Downstream Gate 1b Benchmark Suite (TERMINATED & VOIDED)

The downstream 250-suite benchmark is **permanently terminated and voided** for Qwen3-1.7B.
- **Rule Enforcement**: In accordance with Project Invariants and explicit user directives, evaluating a 1,000-query benchmark matrix on an adapter whose channel is demonstrably bypassed produces ungrounded, deceptive metrics.
- **Conclusion**: The 1.7B parameter architecture cannot sustain causal continuous latent recurrence under decoder-only causal attention. The model must scale to $\ge \text{4B}$ to test whether increased channel capacity and deeper attention heads resolve the read-side bottleneck.

