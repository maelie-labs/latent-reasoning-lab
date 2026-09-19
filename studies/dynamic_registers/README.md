# Study: Staged Dynamic Discrete Registers

**Namespace**: `studies/dynamic_registers`  
**Status**: **COMPLETED (FINAL)**  
**Parent Investigation**: Sealed at `v1.0-qwen3-1.7b-final` (see [`FINDINGS.md`](../../FINDINGS.md))  
**Detailed Empirical Report**: [`RESULTS_DYNAMIC_REGISTERS.md`](RESULTS_DYNAMIC_REGISTERS.md)  

---

## 1. Core Hypothesis & Attribution Invariant

The prior investigation proved that while static discrete scaffolding rungs (`[R1]`, `[R2]`, `[R3]`) partially protect against narrow SFT collapse (+6.67% on Hard MATH), continuous latent recurrence vectors remained causally inert (+0.00% steering, 1.8% attention mass).

**Hypothesis Under Test**: Giving the model the ability to write and attend to **explicit, dynamic discrete key-value registers** (`<|reg|>variable = value<|/reg|>`) within the structural rungs allows symbolic entities to crystallize into discrete tokens, improving reasoning accuracy over static boilerplate headers alone.

**Attribution Rule**: Continuous latents (Arm 3) were strictly deferred until Arm 1 demonstrated statistically significant gains over Control 1 ($\Delta_1 > 0, p < 0.05$) and Arm 2 ($\Delta_2 > 0, p < 0.05$).

---

## 2. Experimental Arms & Results Summary ($N=1,000$, 4 Seeds)

Evaluated across the standardized 250 benchmark suite (150 GSM8K + 100 MATH-500) under Gate 0 Frozen Spec (`0.7 / 0.80 / 20 / 1.5`, 8,192 cap, canonical `math_verify 0.9.0`):

| Arm | Model Checkpoint | Overall Pass@1 | GSM8K | MATH-500 | MATH Hard (L3–5) | Truncation | IFEval Prompt Loose |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control 1** | Headers-Only SFT | **73.70%** | **74.00%** | **73.25%** | **64.58%** | 0.10% | **68.76%** |
| **Arm 1** | Dynamic Discrete Registers | **70.00%** | **68.33%** | **72.50%** | **65.00%** | 0.20% | **64.88%** |
| **Arm 2** | Matched Value Filler Control | **72.00%** | **71.17%** | **73.25%** | **65.00%** | 0.10% | **64.33%** |
| *Ref* | Untrained Base Direct | 73.90% | 75.83% | 71.00% | 70.00% | 0.00% | 72.27% |
| *Ref* | Historical Arm 1b-R | 73.20% | 75.50% | 69.75% | 67.08% | 0.00% | - |

---

## 3. Paired Bootstrapping ($B=10,000$) & Decision Gate Verdict

- **Gate 1 ($\Delta_1 = \text{Arm 1} - \text{Control 1}$)**: **-3.70%** [95% CI: -6.30%, -1.00%], $p = 0.0068$ (**REJECT [Definitive Degradation]**)
- **Gate 2 ($\Delta_2 = \text{Arm 1} - \text{Arm 2}$)**: **-2.00%** [95% CI: -4.50%, +0.50%], $p = 0.1260$ (**REJECT [No Benefit over Mismatched Filler]**)
- **Gate Verdict**: **DEFINITIVE HALT**. Arm 3 (Continuous Latents) is **not justified and must not be added**.

---

## 4. Mechanistic Findings

1. **Attentional Visibility**: Register tokens attract **8.21%** of self-attention mass (solving the 1.8% latent invisibility problem).
2. **Causal Decoupling**: In the Controlled Counterfactual Register Corruption Test, swapping R3 final register values with donor problem values yielded **$\Delta\text{Steer} = +0.00\%$** (0/100 shifted answers), compared to $+21.74\%$ for full-chain discrete CoT. The model's answer decoder is completely decoupled from register numerical contents.
3. **General Capability**: Register conditioning exacerbates IFEval instruction-following collapse by ~3.9 points compared to headers alone (64.88% vs. 68.76%).

For complete tables, method details, and raw data paths, see [`RESULTS_DYNAMIC_REGISTERS.md`](RESULTS_DYNAMIC_REGISTERS.md).
