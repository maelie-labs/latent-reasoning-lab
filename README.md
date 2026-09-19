# Continuous Latent Recurrence, Discrete Scaffolding, and Working Memory

An empirical and mechanistic investigation across language model architectures into whether test-time thought trajectories can be compressed into continuous latent recurrence vectors, explicit structured token registers, or dense propositional summaries.

Evaluations span dense Transformers (`Qwen3-1.7B`, `Qwen3-4B`) and hybrid linear-attention architectures (`Qwen3.5-4B`, combining Gated DeltaNet with full self-attention), evaluated across $N=1,000$ queries per arm on a balanced 250-problem suite with canonical CAS grading (`math_verify 0.9.0`).

---

## Paper

The complete research paper is available at:
* **[`docs/PAPER_CONTINUOUS_LATENT_RECURRENCE_LAB.md`](docs/PAPER_CONTINUOUS_LATENT_RECURRENCE_LAB.md)**:  
  *Continuous Latent Recurrence, Discrete Scaffolding, and Working Memory: An Empirical and Mechanistic Study Across Language Model Architectures*  
  Details all four studies, the write/read attention asymmetry, the Gated DeltaNet innovation collapse, the discrete register decoupling, and the token-compute Pareto frontier.

---

## Overview & Experimental Results

This repository investigates four approaches to compressing intermediate thoughts:

1. **[Study 1: Continuous Latent Recurrence](studies/continuous_recurrence/)**:
   - Evaluated four latent objectives (sequence-level SFT, step-level distillation across three $\lambda$ values, CODI-style hand-off, and prompt attention bottlenecks) alongside matched delay controls.
   - **Result**: Continuous channels are causally inert ($\Delta_1 \le +0.50\%$, $p > 0.60$; falling $-2.20\%$ below base direct on 1.7B and matching dummy `<pause>` tokens on hard math). Counterfactual donor activation patching yields near-zero steering ($\Delta\text{Steer} \le +1.25\%$).
   - **Mechanism**: In dense models, downstream decoders bypass latents (<1.8% attention mass). In hybrid Gated DeltaNet (`Qwen3.5-4B`), linear-attention delta innovation collapses $5.0\times$ ($3.55 \to 0.71$) and recurrent state freezes into an invariant attractor ($\cos \ge 0.9997$).
   - **Full Fine-Tuning**: Training 100% of parameters on 1.7B confirmed flat steering ($\Delta\text{Steer} = -1.06\%$ / $+1.06\%$), though qualified by base generation degradation (Control 0 falling to 72.3%).
2. **[Study 2: Staged Dynamic Discrete Registers](studies/dynamic_registers/)**:
   - Evaluated explicit discrete key-value slots (`<|reg|>k = v<|/reg|>`) against empty headers-only controls and matched numerical filler.
   - **Result**: Authentic registers achieve 70.00% Pass@1 on `Qwen3-1.7B`, underperforming the empty headers control by $-3.70\%$ ($p = 0.0068$) and matching random filler.
   - **Mechanism**: Registers receive 8.21% of self-attention mass, but counterfactual value corruption yields $\Delta\text{Steer} = 0.00\%$, demonstrating that high attention mass does not imply causal information utilization.
3. **[Study 3: Telegraphic Propositional CoT](studies/telegraphic_cot/)**:
   - Stripped conversational narration into concise deductive propositions.
   - **Result**: Dense models suffer arithmetic degradation ($-8.33\%$ on 1.7B, $-8.00\%$ on 4B vs. headers; widening to $-36.00\%$ on long multi-step execution vs. verbose CoT).
   - **Hybrid Architecture**: On Gated DeltaNet (`Qwen3.5-4B`), the arithmetic penalty vanishes ($\Delta = -0.17\%$, n.s.), consistent with the recurrent state matrix buffering intermediate transitions (or both arms reaching an SFT floor at ~87–88%).
4. **[Study 4: Tool-Grounded In-Place Working Memory](studies/tool_grounded_scratchpad/)**:
   - Evaluated native tool-calling state mutations (`update_scratchpad`) and mapped the full Token-Accuracy Pareto Frontier on `Qwen3-4B`.
   - **The Pareto Frontier: Direct Generation Dominates Intermediate Compression**:
     * **Untrained Base Direct**: **86.10% Pass@1 at 184.0 median tokens** (the efficient baseline).
     * **Untrained Verbose CoT**: **94.30% Pass@1 at 2,455.0 median tokens** (high-accuracy ceiling, +8.20% for 277 tok/pt).
     * **Intermediate compression is strictly dominated**: Telegraphic (73.30% at 533 tok), Minimal Scratchpad (76.80% at 612 tok), and Tool Scratchpad (79.10% at 792 tok) sit 7.0 to 12.8 points below base direct while consuming 2.9× to 4.3× more tokens.
     * **Practical Implication**: If compute budget permits, run full verbose CoT; if budget is constrained, **turn thinking off completely**.

---

## Release Tag & Milestones

- **`v2.0-master-paper-final`**: Complete four-study laboratory release, master paper, and unified evaluation artifacts.

---

## Documentation Index

- **[`docs/PAPER_CONTINUOUS_LATENT_RECURRENCE_LAB.md`](docs/PAPER_CONTINUOUS_LATENT_RECURRENCE_LAB.md)**: Full research paper.
- **[`docs/OUTSIDE_VIEW_RESEARCH.md`](docs/OUTSIDE_VIEW_RESEARCH.md)**: Literature survey and prior art dossier on looped transformers, depth recurrence, and latent reasoning.
- **[`docs/SCIENTIFIC_LANDSCAPE_AND_CONTRIBUTIONS.md`](docs/SCIENTIFIC_LANDSCAPE_AND_CONTRIBUTIONS.md)**: Architectural comparison taxonomy (depth recurrence vs. sequence-extending latent recurrence) and theoretical positioning.
- **[`docs/REPRODUCIBILITY_RUNBOOK.md`](docs/REPRODUCIBILITY_RUNBOOK.md)**: Step-by-step reproduction instructions for all models, training runs, and evaluations.
- **[`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md)**: Chronological laboratory notebook covering Milestones 1 through 52.
- **[`docs/RESULTS_QWEN3_4B.md`](docs/RESULTS_QWEN3_4B.md)**: Telemetry and Gate 2 reports for `Qwen3-4B`.
- **[`docs/RESULTS_QWEN3.5_4B.md`](docs/RESULTS_QWEN3.5_4B.md)**: Telemetry and GDN kernel measurements for `Qwen3.5-4B`.
- **[`FINDINGS.md`](FINDINGS.md)**: Summary of empirical results across all four studies.

---

## License

This project is licensed under the Apache License, Version 2.0. See the [`LICENSE`](LICENSE) file for details.
