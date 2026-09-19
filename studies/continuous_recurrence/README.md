# Study: Continuous Latent Recurrence (Sealed v1.0)

**Namespace**: `studies/continuous_recurrence`  
**Status**: Completed & Sealed  
**Tag Anchor**: [`v1.0-qwen3-1.7b-final`](https://github.com/maelie-labs/latent-reasoning-lab/releases/tag/v1.0-qwen3-1.7b-final)  
**Parent Investigation**: Qwen 1.7B and Qwen 4B Continuous Latent Recurrence  
**Primary Summary**: [`FINDINGS.md`](../../FINDINGS.md)  
**Full Telemetry & Results**: [`docs/RESULTS_QWEN3_1.7B.md`](../../docs/RESULTS_QWEN3_1.7B.md)  

---

## 1. Summary of Findings
- **Continuous Latents Are Causally Inert**: Across 14 evaluated configurations ($N=1,000$ per arm, multi-seed), continuous vectors yielded $\Delta\text{Steer} = +0.00\%$ under causal activation patching and $0.00\%$ incremental accuracy over empty headers.
- **Scaffolding vs. Recurrence**: Structural discrete headers (`[R1: Givens]`, `[R2: Ops]`, `[R3: Deduce]`) recovered the reasoning floor damaged by narrow direct SFT (+6.67% on Hard MATH, $p=0.0254$), but did not beat the untrained base floor (73.20% vs. 73.90%).
- **The 125:1 Compression Bottleneck**: Enforcing prompt masking during answer decoding collapsed reasoning performance by $-13.33\%$ on Hard MATH (53.75%).
- **Scale Independence**: Three distillation variants and two recurrent depths on Qwen3-4B also yielded $\Delta\text{Steer} \approx 0.00\%$, confirming that latent inertia is not a sub-2B capacity artifact.

All original scripts, logs, and artifacts for this study are frozen in git tag `v1.0-qwen3-1.7b-final`.
