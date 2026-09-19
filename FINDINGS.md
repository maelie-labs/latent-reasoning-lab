# Master Findings: Continuous Latent Recurrence, Discrete Scaffolding, and Working Memory

**Continuous Latent Recurrence Laboratory**  
**Spanning Models**: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, and `Qwen/Qwen3.5-4B`  
**Date**: September 2026  
**Master Publication Paper**: [`docs/PAPER_CONTINUOUS_LATENT_RECURRENCE_LAB.md`](docs/PAPER_CONTINUOUS_LATENT_RECURRENCE_LAB.md)

---

## 1. Executive Summary: Core Results Across the Four Studies

Across four pre-registered experimental studies spanning dense Transformers and hybrid linear-attention architectures, evaluated across $N=1,000$ queries per arm on a balanced 250-problem suite with canonical CAS grading (`math_verify 0.9.0`), we report three mechanistic negative results and an empirical trade-off:

1. **Study 1: Continuous Latent Recurrence is Inert**:
   - Unrolled latent vectors ($h_k \to h_{k+1}$) fail to carry causal reasoning. On `Qwen3-1.7B`, Pass@1 matches direct SFT controls within $\Delta_1 \le +0.50\%$ ($p > 0.60$), while falling $-2.20\%$ below untrained base direct generation and matching dummy `<pause>` tokens bit-for-bit on hard math. Counterfactual donor patching yields near-zero steering ($\Delta\text{Steer} \le +1.25\%$).
   - *Dense Transformers*: Downstream decoders bypass latents (<1.8% read attention).
   - *Hybrid Gated DeltaNet (`Qwen3.5-4B`)*: Linear-attention delta innovation collapses $5.0\times$ ($3.55 \to 0.71$) and recurrent memory freezes into an invariant attractor ($\cos \ge 0.9997$).
   - *Full Fine-Tuning*: Training all 28 layers of 1.7B confirmed flat steering ($\Delta\text{Steer} = -1.06\%$ / $+1.06\%$), though qualified by base generation destabilization (Control 0 falling to 72.3%).
2. **Study 2: Dynamic Discrete Registers Are Decoupled**:
   - Explicit key-value register slots (`<|reg|>k = v<|/reg|>`) achieve 70.00% Pass@1 on `Qwen3-1.7B`, underperforming the empty headers control by $-3.70\%$ ($p = 0.0068$) and matching random numerical filler.
   - *Attention Decoupling*: Registers receive 8.21% of self-attention mass, but counterfactual value corruption yields $\Delta\text{Steer} = 0.00\%$, demonstrating that high attention weight does not imply causal information utilization.
3. **Study 3: Compression Is Task-Dependent**:
   - Stripping conversational narration into telegraphic propositional rungs induces severe arithmetic collapse on dense Transformers ($-8.33\%$ on 1.7B, $-8.00\%$ on 4B vs. headers; widening to $-36.00\%$ on long multi-step execution vs. verbose CoT).
   - *Hybrid GDN Stability*: On Gated DeltaNet (`Qwen3.5-4B`), the arithmetic penalty vanishes ($\Delta = -0.17\%$, n.s.), consistent with recurrent matrix state ($S_t \in \mathbb{R}^{d \times d}$) buffering intermediate transitions (though equally consistent with an SFT-damage floor at ~87–88%). However, telegraphic reasoning on the hybrid remains $-4.40\%$ below native direct non-thinking mode.
4. **Study 4: Tool-Grounded In-Place Working Memory & The Pareto Frontier**:
   - Native tool-calling state mutations (`update_scratchpad`) recover $+5.80\%$ over telegraphic prose on `Qwen3-4B`, but fail to surpass native base direct inference.
   - **The Pareto Frontier: Direct Generation Dominates Intermediate Compression**:
     * **Untrained Base Direct**: **86.10% Pass@1 at 184.0 median tokens** (the efficient baseline).
     * **Untrained Verbose CoT**: **94.30% Pass@1 at 2,455.0 median tokens** (high-accuracy ceiling, +8.20% for 277 tok/pt).
     * **Intermediate compression is strictly dominated**: Telegraphic (73.30% at 533 tok), Minimal Scratchpad (76.80% at 612 tok), and Tool Scratchpad (79.10% at 792 tok) sit 7.0 to 12.8 points below base direct while consuming 2.9× to 4.3× more tokens.
5. **Practical Implications**:
   - If token budget permits, run unconstrained verbose CoT (+8.20% ceiling at 277 tok/pt).
   - If budget is constrained, **turn thinking off completely** (native direct is $4.3\times$ cheaper and 7.0 points more accurate than the best scratchpad adapter).
   - Avoid intermediate compression.

---

## 2. Master Paper & Research Dossiers Reference

For full derivations, mathematical formulations, kernel equations, and statistical bootstrap tables, refer to the master research paper and literature dossiers:
* **[`docs/PAPER_CONTINUOUS_LATENT_RECURRENCE_LAB.md`](docs/PAPER_CONTINUOUS_LATENT_RECURRENCE_LAB.md)**:  
  *Title*: Continuous Latent Recurrence, Discrete Scaffolding, and Working Memory: An Empirical and Mechanistic Study Across Language Model Architectures  
  *Authors*: Continuous Latent Recurrence Laboratory  
  *Artifact Path*: `docs/PAPER_CONTINUOUS_LATENT_RECURRENCE_LAB.md`
* **[`docs/OUTSIDE_VIEW_RESEARCH.md`](docs/OUTSIDE_VIEW_RESEARCH.md)**:  
  *Title*: Outside View: Replacing Chain-of-Thought with Recurrence in Open-Weight Models  
  *Scope*: Comprehensive literature survey of looped transformers (Huginn, LoopUS, Ouro), latent CoT (Coconut, CODI), pause tokens, and empirical scaling.
* **[`docs/SCIENTIFIC_LANDSCAPE_AND_CONTRIBUTIONS.md`](docs/SCIENTIFIC_LANDSCAPE_AND_CONTRIBUTIONS.md)**:  
  *Title*: Scientific Landscape, Architectural Differentiation & Research Roadmap  
  *Scope*: Foundational taxonomy contrasting depth recurrence against sequence-extending latent recurrence, causal patching methodology, and cross-architecture analysis.

---

## 3. Study 1: Continuous Latent Recurrence Breakdown

### 3.1 Empirical Null Across Architectures ($N=1,000$ queries)
| Model Architecture | Base Direct (Arm 1) | Direct SFT (Arm 1b) | Pause Tokens (Arm 2b) | Latent Recurrence (Arm 3) | Verbose CoT (Arm 4) | Causal Steering $\Delta\text{Steer}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-1.7B** (Dense 28L) | 73.90% | 71.20% | 71.80% | **71.70%** | **87.10%** | **+1.25%** (Inert) |
| **Qwen3-4B** (Dense 36L) | 86.10% | -* | -* | -* | **94.30%** | **+0.00% / +1.06%** (Inert) |
| **Qwen3.5-4B** (Hybrid 32L) | 91.50% | -* | -* | -* | **95.10%** | **0.00%** (Inert) |

*\*Note*: Fast-fail triggered on 4B and 3.5-4B because Gate 2 Counterfactual Patching failed decisively ($\Delta\text{Steer} \le +1.06\%$ against pre-registered $+10.0\%$ threshold).

### 3.2 Mechanistic Breakdown
1. **The Write/Read Asymmetry in Dense Transformers**:
   - Continuous vectors amplify $13.6\times$ through early self-attention layers.
   - However, the downstream answer decoder directs $<1.8\%$ attention mass to latent positions, routing cross-attention directly back to the static prompt tokens.
2. **Gated DeltaNet State Freeze (`Qwen3.5-4B`)**:
   - In hybrid linear-attention layers, write gates remain open ($\beta_t = 0.352 \pm 0.03$).
   - But because continuous vectors lack token discretization, $S_{t-1} k_t \approx v_t$. The innovation vector $(v_t - S_{t-1} k_t)$ collapses $5.0\times$ ($3.55 \to 0.71$).
   - The state update $\Delta S_t$ collapses $6.3\times$, and recurrent memory freezes into an invariant attractor ($\cos(S_t, S_{t-1}) \ge 0.9997$).
3. **1.7B Full Fine-Tuning Retrofit Test**:
   - Updating all $1.72 \times 10^9$ parameters confirmed flat donor steering ($\Delta\text{Steer} = -1.06\%$ at $t=3$, $+1.06\%$ at $t=6$).
   - However, uncorrupted Control 0 recovery dropped to 72.3% (indicating base generation instability), making the full-FT run scientifically inconclusive on its own and establishing LoRA as the primary causal evidence.

---

## 4. Study 2: Staged Dynamic Discrete Registers

Testing explicit discrete key-value slots (`<|reg|>variable = value<|/reg|>`) within canonical scaffolding rungs (`[R1]`, `[R2]`, `[R3]`):

| Arm / Configuration | Benchmark Pass@1 | GSM8K ($N=600$) | MATH-500 ($N=400$) | Hard MATH L3–5 ($N=240$) | Paired $\Delta$ vs. Ctrl 1 | Bootstrap $p$-value | Causal Steering $\Delta\text{Steer}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untrained Verbose CoT (Ceiling)** | **87.10%** | 87.17% | 87.00% | 90.83% | $+13.40\%$ | $p < 0.0001$ | - |
| **Untrained Base Direct (Floor)** | **73.90%** | 71.00% | 78.25% | 70.00% | $+0.20\%$ | $p = 0.732$ | - |
| **Control 1 (Headers Only)** | **73.70%** | 74.00% | 73.25% | 64.58% | Reference | - | - |
| **Arm 2 (Matched Filler)** | **72.00%** | 71.17% | 73.25% | 65.00% | $-1.70\%$ | $p = 0.126$ | **0.00%** ($0 / 100$) |
| **Arm 1 (Authentic Registers)** | **70.00%** | 68.33% | 72.50% | 65.00% | **-3.70%** | $p = 0.007$ | **0.00%** ($0 / 100$) |

- **Key Takeaway**: Registers degraded performance by $-3.70\%$ ($p = 0.0068$). Although registers received 8.21% of self-attention mass, replacing register values with donor numbers produced $\Delta\text{Steer} = 0.00\%$. The decoder fetched parameters directly from the prompt prefill.

---

## 5. Study 3: Telegraphic Propositional CoT

Stripping conversational self-talk into concise deductive propositions:

| Model Architecture | Arm | Pass@1 Overall | GSM8K ($N=600$) | GSM8K-Long ($N=200$) | MATH-500 ($N=400$) | Median Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-1.7B** (Dense) | Verbose CoT (Ceiling) | **87.10%** | 87.17% | 84.50% | 87.00% | 2,114.0 |
| | Headers Control | **73.70%** | 74.00% | 72.00% | 73.25% | 350.0 |
| | Telegraphic CoT | **69.00%** | **65.67%** | **61.50%** | 74.00% | 446.5 |
| | *Drop vs. Headers Control* | *-4.70%* | **-8.33%** | *-10.50%* | *+0.75%* | *p < 0.0001* |
| **Qwen3-4B** (Dense) | Verbose CoT (Ceiling) | **94.30%** | 96.00% | 93.00% | 91.75% | 2,455.0 |
| | Headers Control | **83.00%** | 83.17% | 73.00% | 82.75% | 409.0 |
| | Telegraphic CoT | **73.30%** | **75.17%** | **57.00%** | 70.50% | 533.0 |
| | *Drop vs. Headers Control* | *-9.70%* | **-8.00%** | *-16.00%* | *-12.25%* | *p < 0.0001* |
| **Qwen3.5-4B** (Hybrid GDN) | Verbose CoT (Ceiling) | **95.10%** | 93.17% | - | 98.00% | - |
| | Base Direct (Floor) | **91.50%** | 91.00% | - | 92.25% | - |
| | Headers Control | **88.40%** | 87.00% | - | 90.50% | - |
| | Telegraphic CoT | **87.10%** | **86.83%** | - | 87.50% | - |
| | *Drop vs. Headers Control* | *-1.30%* | **-0.17%** | - | *-3.00%* | *p = 0.96 (n.s.)* |

- **Dense Collapse**: Removing conversational filler starved dense attention layers of intermediate arithmetic working memory ($-8.00\%$ to $-8.33\%$ drop on GSM8K; $-36.00\%$ on 4B long problems).
- **Hybrid GDN Parity**: On `Qwen3.5-4B`, telegraphic reasoning matched headers-only control ($\Delta = -0.17\%$, n.s.), consistent with recurrent matrix state buffering intermediate calculations. However, it still sat $-4.40\%$ below base direct non-thinking mode.

---

## 6. Study 4: Tool-Grounded In-Place Working Memory & The Pareto Frontier

Mapping the complete test-time compute frontier on `Qwen3-4B` ($N=1,000$ queries):

| Evaluation Arm | Pass@1 Overall | GSM8K Overall | GSM8K-Long ($N=200$) | MATH-500 Overall | Median Tokens | Status / Frontier Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untrained Base Direct** (Arm 1) | **86.10%** | 88.00% | 86.50% | 83.25% | **184.0** | **Native Efficient Point (Floor)** |
| **Telegraphic CoT** (Study 3) | **73.30%** | 88.00% | 57.00% | 51.25% | **533.0** | Dominated (-12.80% acc, 2.9× tok) |
| **Minimal Tool Scratchpad** (Dial) | **76.80%** | 86.50% | 66.00% | 62.25% | **612.0** | Dominated (-9.30% acc, 3.3× tok) |
| **Tool Scratchpad** (Study 4) | **79.10%** | **80.33%** | **71.00%** | **77.25%** | **792.0** | Dominated (-7.00% acc, 4.3× tok) |
| **Untrained Verbose CoT** (Arm 4) | **94.30%** | **96.00%** | **93.00%** | **91.75%** | **2,455.0** | **High-Accuracy Ceiling** |

### Marginal Cost per Point Analysis:
- **Direct $\to$ Verbose CoT**: $+8.20\%$ accuracy for $+2,271$ median tokens = **277.0 tokens / point** (The authentic frontier).
- **Direct $\to$ Any Compressed Representation**: **Negative Return** (Loses 7.0 to 12.8 points of accuracy while consuming 2.9× to 4.3× more tokens).
- **Sub-Space Recovery**: Tool Scratchpad recovers $+5.80\%$ over Telegraphic CoT at 44.7 tok/pt within the compressed subspace, but remains strictly Pareto-dominated by native zero-thought generation.

---

## 7. Methodological Contributions

Beyond empirical benchmark measurements, this work highlights two methodological findings:

1. **Discrete-CoT Calibration for Activation Patching**:
   - Arbitrary steering thresholds (e.g. asserting that patched vectors must steer $\ge 40\%$) are invalid without calibrating natural language text CoT under identical prompt geometry.
   - Splicing $K=6$ discrete tokens yields $\Delta\text{Steer} = +0.00\%$; splicing an entire paragraph yields $+8.70\%$; only replacing the full multi-step chain yields $+21.74\%$.
2. **Attention Mass $\neq$ Semantic Grounding (Coupling vs. Content)**:
   - High attention weights (>8%) and high noise sensitivity (>90%) reflect dynamical coupling within transformer KV cache graphs, NOT semantic computation.
   - Asserting cognitive computation requires counterfactual donor patching and matched delay controls.

---

## 8. Scope Boundaries

- **Retrofitting vs. Pre-Training from Scratch**: Results apply to retrofitting pre-trained foundation models. Whether training from token zero with continuous latent recurrence could develop native continuous reasoning channels remains an open question.
- **Search vs. Deductive Tracking**: Combinatorial proof search (MATH Level 5) resisted compression across all formats tested, requiring expansive exploration trees.
- **Reinforcement Learning vs. Supervised SFT**: RL outcome training (PPO/GRPO) remains to be investigated, though continuous credit assignment presents significant optimization obstacles.
- **Workload Scope**: The Pareto frontier was measured on mathematical and arithmetic deduction benchmarks on `Qwen3-4B`.

---

## 9. Release Tag & Milestone Map

- **`v2.0-master-paper-final`**: Complete four-study laboratory release, master paper, and unified evaluation artifacts.
