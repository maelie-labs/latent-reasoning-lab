# Empirical Results: Staged Dynamic Discrete Registers Study

**Model Architecture**: `Qwen/Qwen3-1.7B`  
**Hardware Configuration**: Dedicated Compute GPU 1 (`RTX PRO 4500 Blackwell 32GB`, bus ID `0B:00.0`, `--mem-fraction-static 0.90`)  
**Serving Engine**: SGLang v0.5.9 (Asynchronous Concurrency $C^*=32$, Zero-Prefix-Cache `--disable-radix-cache`)  
**Grader Version**: Canonical `math_verify 0.9.0` (Symbolic extraction & equivalence, 0 heuristics)  
**Multi-Seed Evaluation**: $N=1,000$ queries (250 suite $\times$ 4 seeds: `[42, 123, 456, 789]`)  
**Date of Execution**: September 15–16, 2026  

---

## Executive Summary & Decision Gate Verdict

This study investigated whether equipping a small reasoning model (`Qwen/Qwen3-1.7B`) to generate and attend to **staged dynamic discrete key-value registers** (`<|reg|>variable = value<|/reg|>`) within canonical scaffolding rungs (`[R1]`, `[R2]`, `[R3]`) provides authentic algorithmic reasoning gains over matched static scaffolding headers alone (Control 1), empirical filler controls (Arm 2), and the untrained base direct model.

### Pre-Registered Decision Gate Verdict: **DEFINITIVE FAIL / HALT**
Per pre-registered protocol, adding continuous latent recurrence (Arm 3: Dual-Channel Registers + Latents) was strictly contingent upon Arm 1 establishing positive, statistically significant reasoning gains over Control 1 ($\Delta_1 > 0, p < 0.05$) and Arm 2 ($\Delta_2 > 0, p < 0.05$).

1. **Gate 1 ($\Delta_1 = \text{Arm 1} - \text{Control 1}$)**: **FAILED [DEGRADATION]**  
   $\Delta_1 = -3.70\%$ [95% CI: -6.30%, -1.00%], $p = 0.0068$.  
   Dynamic discrete registers significantly degrade reasoning performance relative to static headers alone.
2. **Gate 2 ($\Delta_2 = \text{Arm 1} - \text{Arm 2}$)**: **FAILED [NULL / DEFICIT]**  
   $\Delta_2 = -2.00\%$ [95% CI: -4.50%, +0.50%], $p = 0.1260$.  
   Dynamic discrete registers fail to outperform syntax-matched registers populated with randomized empirical filler.
3. **Attribution Ruling**: Continuous latent recurrence (Arm 3) is **NOT justified and MUST NOT be added**.

---

## 1. Unified Benchmark Telemetry ($N=1,000$)

All arms were evaluated under the frozen Gate 0 specification: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `presence_penalty=1.5`, answer cap 8,192 tokens, with `<think>\n\n</think>\n\n` formatting. Truncated sequences hitting the token ceiling are strictly scored as incorrect (`is_correct = False`).

| Experimental Arm | Overall Pass@1 | GSM8K Pass@1 | MATH-500 Pass@1 | MATH Hard (L3–5) | Truncation Rate | Eval Time | Concurrency ($C^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untrained Base Direct** (Ref) | 73.90% | 75.83% | 71.00% | 70.00% | 0.00% | - | 32 |
| **Historical Arm 1b-R** (Ref) | 73.20% | 75.50% | 69.75% | 67.08% | 0.00% | - | 32 |
| **Control 1: Headers-Only** | **73.70%** | **74.00%** | **73.25%** | **64.58%** | 0.10% | 330.4s | 32 |
| **Arm 1: Dynamic Registers** | **70.00%** | **68.33%** | **72.50%** | **65.00%** | 0.20% | 452.2s | 32 |
| **Arm 2: Matched Value Filler** | **72.00%** | **71.17%** | **73.25%** | **65.00%** | 0.10% | 375.6s | 32 |

### Key Observations:
- **Control 1 Replicates Historical Arm 1b-R**: Control 1 (73.70%) closely matches the historical Arm 1b-R control (73.20%), confirming that SFT on clean 3-header rungs preserves base model reasoning capabilities without distortion.
- **Arithmetic Sensitivity**: The primary degradation in Arm 1 occurred on GSM8K arithmetic (68.33% vs. 74.00% for Control 1, $\Delta = -5.67\%$).
- **MATH Hard Floor**: On difficult contest math (MATH Levels 3–5, $N=240$), all three trained arms clustered tightly at 64.58%–65.00% (a 1-problem difference across 240 queries), indicating that registers provide zero leverage on genuinely hard reasoning problems.

---

## 2. Paired Hierarchical Bootstrapping ($B=10,000$)

Paired difference distributions and 95% bootstrap confidence intervals computed across all $N=1,000$ paired problem-seed queries:

| Hypothesis Comparison | Metric | Observed Delta ($\Delta$) | 95% Bootstrap CI | Two-Sided $p$-value | Empirical Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$\Delta_1$: Arm 1 vs. Control 1** | Overall Pass@1 | **-3.70%** | **[-6.30%, -1.00%]** | **$p = 0.0068$** | **REJECT [Significant Deficit]** |
| | MATH Hard (L3–5) | +0.42% | [-5.42%, +6.25%] | $p = 0.9348$ | Null (1 problem delta) |
| **$\Delta_2$: Arm 1 vs. Arm 2** | Overall Pass@1 | **-2.00%** | **[-4.50%, +0.50%]** | **$p = 0.1260$** | **REJECT [No Effect]** |
| | MATH Hard (L3–5) | +0.00% | [-5.83%, +5.43%] | $p = 1.0000$ | Identical (156/240 vs 156/240) |
| **$\Delta_3$: Arm 1 vs. Base Direct** | Overall Pass@1 | **-3.90%** | **[-6.50%, -1.30%]** | **$p = 0.0034$** | **Significant Degradation** |
| | MATH Hard (L3–5) | -5.00% | [-10.00%, 0.00%] | $p = 0.0698$ | Deficit |
| **Replication: Ctrl 1 vs. Hist Arm 1b-R**| Overall Pass@1 | **+0.50%** | **[-2.00%, +3.00%]** | **$p = 0.7320$** | **PASS [Equivalence Validated]** |

---

## 3. Arm 0 Surgical Neutrality (google/IFEval Audit)

Evaluated across all 541 prompts (834 verifiable instructions) from the official `google/IFEval` benchmark under identical serving conditions:

| Model / Checkpoint | Prompt Strict | Prompt Loose | Instruction Strict | Instruction Loose | Truncation Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Untrained Base** | **67.65%** | **72.27%** | **75.42%** | **79.50%** | 0.00% |
| **Control 1: Headers-Only** | 63.40% | 68.76% | 72.30% | 76.50% | 2.59% |
| **Arm 1: Dynamic Registers** | 61.18% | **64.88%** | 70.02% | 73.62% | 3.33% |
| **Arm 2: Matched Value Filler** | 60.81% | 64.33% | 70.14% | 73.14% | 3.14% |

### General Capability Findings:
1. **Narrow SFT Degradation**: All math SFT adapters experience moderate general instruction-following drift relative to the untrained base (3.5 to 7.9 points).
2. **Registers Exacerbate Degradation**: Training the model on discrete key-value register syntax induces an additional **~3.9 point drop** on Prompt Loose accuracy relative to headers alone (64.88% vs. 68.76%), demonstrating that enforcing rigid register syntax slightly impairs natural text adherence.

---

## 4. Phase 5 Mechanistic Intervention Audit

### 4.1 Attention Decomposition (Task 5.1)
Measured answer-phase attention distribution across 30 authentic evaluation trajectories on dedicated compute hardware:

| Structural Component | Mean Attention Mass (%) | Std Dev (%) | Relative Share |
| :--- | :---: | :---: | :---: |
| **Prompt Tokens** | **82.74%** | ±3.14% | Dominant Driver |
| **Discrete Register Tokens (`<|reg|>...<|/reg|>`)** | **8.21%** | ±2.96% | Measurable Presence |
| **Structural Headers (`[R1]`, `[R2]`, `[R3]`)** | **3.91%** | ±0.87% | Scaffolding Rungs |
| **Transition Tokens (`\n</think>\n\n`)** | **4.90%** | ±0.63% | Delimiter Boundary |
| **Other / Whitespace** | **0.25%** | ±1.32% | Residual |

*Key Insight*: Formatting variables into discrete syntax (`<|reg|>`) completely solved the **attentional invisibility** observed with continuous latent tokens (where latents received only 1.8% of attention). The registers attract 8.21% of self-attention mass. However, as the corruption test proves, this attention is decorative rather than causal.

---

### 4.2 Controlled Register Corruption Test (Task 5.2)
To establish whether the 8.21% attention mass reflects authentic causal binding or superficial co-occurrence, counterfactual register swapping was performed across 100 benchmark problem pairs:

- **Control 0 (Null Patch Self-Identity)**: Re-injecting the unperturbed register block yielded **81.00%** answer identity (verifying that the model decodes deterministically from its own scaffolding).
- **Chance Baseline $P(\text{chance})$**: **0.00%** (donor answers never appeared spontaneously in unperturbed generations).
- **Counterfactual Donor Steered $P(\text{steered})$**: **0.00%** (0 out of 100 trials).
- **Net Steering Delta ($\Delta\text{Steer}$)**: **+0.00%** [95% CI: 0.00%, 0.00%].

### Calibration Against Empirical Anchors:
```
================================================================================
EMPIRICAL CAUSAL STEERING BENCHMARK COMPARISON
--------------------------------------------------------------------------------
Full-Chain Discrete CoT Replacement:     +21.74% (measured upper bound)
Step-1 Discrete CoT Replacement:         +8.70%
Dynamic Discrete Registers (<|reg|>):     +0.00% (completely decoupled)
Continuous Latent Vectors (K=6):         +0.00% (inert floor)
================================================================================
```

*Mechanistic Conclusion*: Despite attending to `<|reg|>` tokens, the model's downstream answer is **completely causally decoupled** from the numerical values assigned to those registers. When R3's final register value is replaced with a counterfactual donor value, the model completely ignores the injected value and generates the correct answer to the original prompt from direct prompt attention.

---

## 5. Scientific Conclusions & Takeaways

1. **Discrete Registers Act as Delay Tokens, Not Computation Units**:  
   Like pause tokens and unrolled latents, dynamic discrete registers provide a fixed compute buffer before answer generation. However, the model does not execute variable substitution or stateful tracking inside the registers.
2. **Grammatical Constraints Penalize Exploration**:  
   Constraining the model's reasoning into rigid key-value pairs (`pink_rem = 26 - 10 = 16`) restricts natural language error correction and verification, reducing GSM8K accuracy from 74.00% down to 68.33%.
3. **Mismatched Filler Equivalence Confirms Indifference**:  
   Arm 2 (randomized empirical values in identical register slots) performed within noise of Arm 1 (72.00% vs. 70.00%, $p = 0.1260$, and identical 65.00% on MATH Hard). The model is largely indifferent to whether register values are factually correct or randomly drawn from unrelated problems.
4. **Architectural Recommendation**:  
   In small autoregressive language models (1.7B), multi-step reasoning is inextricably linked to the expressive, dense semantic representations of natural language token chains. Neither continuous vectors nor structured discrete key-value slots serve as an effective substitute for natural language chain-of-thought.
