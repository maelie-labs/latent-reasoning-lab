# Outside View: Replacing Chain-of-Thought with Recurrence in Open-Weight Models

*Compiled 2026-09-11 from a literature scan. Purpose: an independent reference for checking assumptions in a project that aims to reduce VRAM / thinking-token cost on a ~27B open-weight model (Qwen-class, NVFP4, SGLang, Blackwell, low concurrency). Treat every claim here as something to verify against the primary source before relying on it.*

---

## 1. Short answer

**Yes, people are retrofitting recurrence into open-weight models. No, nobody has shown recurrence replacing explicit CoT at parity on hard tasks, at any scale that matters.**

The field has two mostly separate threads:

1. **Depth recurrence / looped transformers**: reapply a shared block of layers N times per token. Adds serial compute *per token* without adding tokens.
2. **Latent (continuous) CoT**: feed hidden states back as input embeddings so "thinking" happens as a short sequence of non-text positions between prompt and answer. Adds compute *per sequence* via a few dense latent tokens instead of thousands of text tokens.

Retrofits onto existing open weights exist for both, but almost all of them are **additive** (base capability at loop 1, gains at higher loops) rather than **substitutive** (drop the thinking block).

---

## 2. What exists (retrofits onto open weights)

| Work | Base model(s) | Approach | Status |
|---|---|---|---|
| Bartoldson et al., *Retrofitting Language Models with Depth Recurrence* (Nov 2025, arXiv 2511.07384) | Llama-3.2-1B, OLMo-2-1B | Model surgery into prelude / recurrent block / coda; linear adapter concatenates prelude + recurrent output. GSM8K/MATH gains of a few points. | Paper + code |
| **LoopUS**: *Recasting Pretrained LLMs into Looped Latent Refinement Models* (May 2026, arXiv 2605.11011) | Qwen3-1.7B / 4B / 8B, Phi-4 | Encoder → looped reasoning block → decoder. Block selection via representation dynamics; input-dependent selective gate against hidden-state drift; random deep supervision; confidence head for early exit. KV caching per loop implemented on HF DynamicCache. | Paper + HF checkpoints (e.g. `Thrillcrazyer/Phi4_LoopUS`) |
| *Retrofitting Recurrent Depth into a Pretrained LM* (Aug 2026, arXiv 2608.11233) | Qwen2.5-0.5B-Instruct | Identity-preserving one-loop path + re-entry bridge. Installs at 6M-param LoRA-style adapter over frozen base **or** 180M full block. Shows one "task step per loop" that persists after outcome-only RL annealing. | Paper |
| *Looped LMs Improve Compositional Tool Calling* (Aug 2026, arXiv 2608.18171) | Llama-3.2-1B, OLMo-2-1B retrofits; Ouro | Fixed 8-loop inference; gains on BFCL / NESTful vs non-looped parents fine-tuned identically. | Paper |
| `maxyu1115/qwen3-recurrent` (GitHub) | Qwen3-0.6B | Hobbyist: loop layers 8–20, adapter merges loop output back into layer-7 output, initialized to identity. | Code only |
| Coconut (Hao et al., Dec 2024), CODI (Shen et al., 2025) | GPT-2 / small Llama | Feed last-layer hidden state back as next input embedding; curriculum (Coconut) or self-distillation (CODI) replaces CoT tokens with k latent positions. | Papers + code |
| *Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy RL* (Jun 2026, arXiv 2606.13106) | Small open models | Same model can reason in text or in latent; RL over latent thoughts. Closest thing to true "replace CoT". | Paper |

From-scratch looped models for reference: **Huginn-3.5B** (Geiping et al. 2025), **Ouro** (Zhu et al. 2025), **Parcae** scaling laws (2026).

Curated list: `github.com/EIT-NLP/Awesome-Latent-CoT`.

---

## 3. Limitations of current work (the checklist)

### 3.1 Accuracy does not reach explicit CoT
- Huginn-3.5B probe (arXiv 2507.02199): on GSM8K, more recurrent steps improve only marginally and fall short of explicit reasoning.
- Recurrence adds a handful of "effective layers" per token. A 10k-token thinking trace adds orders of magnitude more serial compute *and* lets the model revisit earlier intermediate states via attention. Looped depth cannot recover the second property.
- Coconut/CODI/Abstract-CoT are validated on GSM8K-scale arithmetic and ProsQA/ProntoQA logic, not AIME, LiveCodeBench, or agentic tasks.
- Structured-data comparison (arXiv 2605.11262): looped baselines with no extra tokens underperform latent-CoT with appended feedback tokens (i.e. *sequence-extending* latent compute beats *depth-extending* latent compute in a head-to-head).

**Check:** does the project have a benchmark where explicit-CoT Qwen 27B is the baseline, and is the target "match" or "acceptable degradation"? Be explicit about which.

### 3.2 Scale
- Nearly all retrofit results are 0.5B–8B. Largest is LoopUS on Qwen3-8B / Phi-4 (~14B).
- No published retrofit at 27B+ with capability retention demonstrated. Scaling laws for stable loops (Parcae) are for from-scratch training, not surgery.

**Check:** any result the agent cites from ≤8B should be flagged as unproven at 27B.

### 3.3 Retrofits are fragile
- Naively looping a block trained once tends to diverge or wash out. LoopUS needs a selective gate to stop hidden-state drift and explicitly guards against representation collapse.
- Which layers to loop is an open problem; LoopUS picks by inspecting representation dynamics, others pick by hand.
- Retrofits are trained to be **non-inferior at loop 1**; gains at higher loops are modest and task-specific.

**Check:** does the project measure base-task regression at loop 1 (e.g. MMLU, IFEval, tool-use), not just gains on the target task?

### 3.4 Interpretability and verifiability
- You lose the trace. Huginn probing shows discontinuities and inconsistent probes across recurrent cycles (arXiv 2507.02199).
- *Observable Patterns Are Not Explanations* (Jun 2026, arXiv 2606.12689) argues the structure people observe in continuous-thought models is not causally load-bearing.
- For agentic / tool-calling use this is a real cost: no way to inspect *why* an action was chosen.

### 3.5 Efficiency is not free: this is the one most relevant to a VRAM project
- Recurrence multiplies **FLOPs per token** by the loop count. Token count goes down; compute per token goes up. Net latency at 1–2 streams is not obviously better.
- **KV cache per loop**: depending on design, each loop iteration has its own KV. LoopUS implements a separate DynamicCache per loop; *Memory-Efficient Looped Transformer* (2026) exists specifically because the naive design multiplies KV memory. The VRAM win over explicit CoT is smaller than it appears unless the loop shares KV or only caches the final iteration.
- Speculative decoding interaction is unexplored: a DFlash/EAGLE-style drafter assumes fixed per-token depth. Adaptive loop counts (early exit via confidence head) break the draft/verify symmetry.
- NVFP4 interaction is unexplored: repeated application of a quantized block compounds quantization error; none of the papers quantize the looped block.

**Check:** if the project's motivation is VRAM, compute the actual KV bytes per loop under the chosen design before assuming a win. Also confirm whether spec decode is still usable.

### 3.6 Training recipe
- RL for explicit CoT works because the trace is sampleable text. RL over latent thoughts is only just starting (arXiv 2606.13106) and has not been shown at scale.
- Intermediate-step supervision (arXiv 2608.11233) is what makes the loop learn a reusable procedure; outcome-only training tends to install answer lookup rather than iteration.
- From-scratch looped models need pretraining-scale data; not feasible outside a lab. Retrofit is the only realistic route, and retrofit gains are bounded by §3.1–3.3.

---

## 4. Questions the agent should be able to answer about its own work

1. Which of the two threads (depth recurrence vs latent CoT) is it pursuing, and why that one given §3.1's head-to-head?
2. What is the explicit-CoT baseline number on the target task, and what degradation is acceptable?
3. What is the base-task regression at loop 1?
4. What is the measured KV memory and FLOPs at the intended loop count, in bytes/FLOPs, not "fewer tokens"?
5. Does it still work with DFlash / spec decode, or is that being given up?
6. Has the looped block been tested under NVFP4, or only BF16?
7. Which layers are looped, and how was that chosen?
8. Is there any evidence of iteration (state changes per loop) rather than answer lookup (e.g. the per-loop probing of arXiv 2608.11233)?

---

## 5. Key references

- Geiping et al. 2025: *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Huginn-3.5B)
- Lu et al. 2025: *Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer*: arXiv 2507.02199
- Bartoldson et al. 2025: *Retrofitting Language Models with Depth Recurrence*: arXiv 2511.07384
- *LoopUS* 2026: arXiv 2605.11011
- *Retrofitting Recurrent Depth into a Pretrained LM* 2026: arXiv 2608.11233
- *Looped LMs Improve Compositional Tool Calling* 2026: arXiv 2608.18171
- Hao et al. 2024: *Training LLMs to Reason in a Continuous Latent Space* (Coconut)
- *Demystifying Hidden-State Recurrence* 2026: arXiv 2606.13106
- *Observable Patterns Are Not Explanations* 2026: arXiv 2606.12689
- *Latent CoT Improves Structured-Data Transformers* 2026: arXiv 2605.11262
- Zhu et al. 2025: *Scaling Latent Reasoning via Looped Language Models* (Ouro)
- Awesome-Latent-CoT list: github.com/EIT-NLP/Awesome-Latent-CoT

---

## 6. Follow-up Guidelines: Subjects, MoE Routing, and Quantization Isolation

*Compiled 2026-09-11 from external research advisor review.*

### 6.1 On the R1-Distill Family vs. Qwen3 Ladder
- **R1-Distill caveats**: Pure SFT on R1 traces with no RL and no native non-thinking mode. "No-CoT on base" is an unnatural condition for them: they will often attempt to think anyway or degrade severely, which artificially inflates the apparent gain from recurrence. Also vintage Jan 2025 (1.5B has substantial headroom).
- **Add Qwen3 ladder (1.7B / 8B / 14B)**:
  - Qwen3 features a native `/no_think` mode, making the non-thinking baseline an intended operational mode.
  - LoopUS already retrofitted Qwen3-1.7B / 4B / 8B, providing published baseline benchmarks to reproduce before validating our custom pipeline.
  - Cap the ladder with the 27B model (the actual production deployment target) rather than assuming 14B extrapolates.
  - **Unpublished empirical comparison**: Test whether a model that was RL'd to think in text (Qwen3) is harder or easier to adapt into latent iteration than a model trained purely via distillation (R1-Distill).

### 6.2 On MoE Architectures (Qwen3-30B-A3B)
- Natural candidate: close to 27B capability while computationally cheap per token.
- **The Core Scientific Question: Dynamic Routing Under Recurrence**:
  - In an MoE, each recurrent loop re-evaluates the top-$k$ router on a drifting hidden state:
    $$\text{Router}(h_t) \to \text{Expert Selection at Loop } t$$
  - Does the router select different experts across loops?
    - *Potential feature*: The recurrent loop dynamically specializes across reasoning steps.
    - *Potential failure mode*: State drift compounds through chaotic expert switching.
  - *Recommendation*: Establish clean dense baseline results first; test 30B-A3B as a targeted experiment specifically logging expert selection trajectories across recurrent passes.

### 6.3 On Isolating Quantization Variables
- Size and quantization must not be confounded (e.g. comparing 14B FP8 vs 7B BF16 confounds size with precision).
- **Methodology**: Fix model scale (e.g. 7B) and evaluate across `bfloat16`, `fp8`, and `NVFP4` under identical loop settings to isolate quantization error accumulation through the recurrent loop.
- If NVFP4 holds up across 8–16 loops on 7B, then evaluate 14B and 27B in NVFP4 (the target Blackwell deployment format).
- **Training Order**: Distinguish between:
  1. *Quantizing a trained recurrent retrofit* (recommended first).
  2. *Training a recurrent loop adapter directly on a quantized base* (substantially harder due to gradient quantization noise).

---

## 7. The Three-Point Validation Triad

*Compiled 2026-09-11 from external research advisor review.*

To conclusively prove that continuous recurrence provides true reasoning benefits rather than an artifact of baseline distortion or memorization, three targeted tests must be conducted:

### 7.1 Test 1: Compute-Matched Comparison at Loop N
- **The Pitfall**: Comparing *recurrent at 8 loops vs. base model with no thinking* is an uneven comparison because the recurrent model has received 8 extra forward passes of compute.
- **The Rigorous Bar**: Compare against a **compute-matched baseline**:
  - Retrofit at $K=6$ or $K=8$ loops vs. the base model restricted to generating roughly the *equivalent budget of thinking tokens* (e.g. 6 to 8 discrete tokens), or vs. best-of-$k$ sampling at identical total FLOPs.
  - *Current Laboratory State*: Our benchmark evaluates recurrent models directly against the **full discrete Chain-of-Thought teacher** generating 600–1,200 tokens (where recurrence uses $\sim 1/100\text{th}$ the generation compute). Adding a compute-matched 6-token discrete baseline will isolate the specific expressivity of continuous latent superposition vs. 6 discrete tokens.

### 7.2 Test 2: Causal Intervention & Activation Patching (Load-Bearing vs. Epiphenomenon)
- **The Core Question**: Are per-loop decodes causally load-bearing, or merely readable epiphenomena (as argued in *Observable Patterns Are Not Explanations*, arXiv 2606.12689)?
- **The Patching Protocol**:
  1. At loop $k \in [2..4]$, extract the intermediate hidden state $h_k$.
  2. Decode $h_k$ using the auxiliary probe into its top-1 token $T_{\text{probe}}$.
  3. Re-embed $T_{\text{probe}}$ via $W_E$, and patch $h_k \leftarrow \alpha \cdot \text{embed}(T_{\text{probe}})$ (or patch with an intermediate latent from a distinct counterfactual problem).
  4. Continue recurrence from loop $k+1 \to K$ and examine the final answer:
     - If the final answer is insensitive to the patch $\implies$ the recurrent hidden states are correlated but not causally active.
     - If the final answer systematically shifts in direction of the patch $\implies$ **conclusive proof of causally load-bearing, interpretable latent CoT**.

### 7.3 Test 3: GSM-Symbolic / GSM-Plus Perturbations (Preventing Memorization Bias)
- **The Threat**: Because the R1-Distill family was trained on extensive R1 synthetic traces, standard GSM8K questions risk test-set contamination or memorization.
- **The Protocol**: Spot-check the recurrent retrofit against **GSM-Symbolic** (renamed entity names, shifted numerical constants, perturbed clause order).
- If the recurrent model maintains its accuracy delta over the discrete baseline under symbolic perturbations, it proves genuine algorithmic execution rather than trace recall.

---

## 8. MoE Subjects for Recurrence Retrofit & Routing Diagnostics

*Compiled 2026-09-11 from external research advisor review.*

*Hardware Constraint: pure `bfloat16`, $\le \sim 14\text{B}$ total parameters (must fit comfortably within 32 GB VRAM on NVIDIA RTX PRO 4500 Blackwell).*

### 8.1 Two Distinct Experimental Roles: Instrumentation vs. Results
A critical methodological distinction is separating models used for **instrumentation** (mechanistic router stability) from models used for **headline results** (top-tier reasoning benchmark numbers):
- **Never use one model for both**: A model ideal for router probing (open data, standard architecture) is often obsolete or weak at reasoning, which inflates apparent recurrence gains.
- **Instrumentation Role**: Answers: *Is hidden-state recurrence mathematically stable when a dynamic router sits inside the loop?*
- **Results Role**: Answers: *Does a modern, competitive MoE achieve state-of-the-art reasoning under continuous latent recurrence?*

### 8.2 Candidate MoE Models

| Model | Total / Active Params | Role | Suitability & Strategic Value |
| :--- | :--- | :--- | :--- |
| **OLMoE-1B-7B** | 7B total / 1B active (16 layers, 64 experts, top-8) | **Instrumentation** | 2024 vintage, weaker at reasoning (accuracy gains will not transfer directly). Core value: **100% open-source** (weights, data, architecture) and the standard benchmark subject in recent MoE mechanistic interpretability literature. Provides an established router baseline. |
| **Qwen1.5-MoE-A2.7B** | 14.3B total / 2.7B active | **Transfer** | Upcycled from Qwen1.5-1.8B; closest direct architectural cousin to our dense Qwen subjects (Qwen 1.5B / 7B / 14B / 27B). Directly tests whether the exact same recurrence surgery transfers seamlessly from dense $\to$ MoE within one model lineage. |
| **LFM2-8B-A1B** | 8B total / 1B active | **Results** | Strongest modern, highly competitive small MoE that fits natively in 16 GB / 32 GB BF16. The primary subject for publication-grade headline accuracy numbers. |
| **Granite 3.1 3B-A800M / 1B-A400M** | Extremely compact | Optional | Fast, lightweight sanity check to verify if the recurrence loop converges with a router at all before scaling. |
| *DeepSeek-V2-Lite* | 15.7B total / 2.4B active (~31 GB BF16) | Over limit | Shared + routed experts, MLA (Multi-Head Latent Attention). Closest small analogue to DeepSeek-V3/R1. Marginally fits on 32 GB; reserve for later. |
| *gpt-oss-20b* | 21B total / 3.6B active | Later | Native reasoning MoE, but experts ship in MXFP4 (~42 GB upcast); requires custom dequantization. |

### 8.3 Recommended Execution Pipeline
$$\text{OLMoE-1B-7B (Instrumentation)} \longrightarrow \text{Qwen1.5-MoE-A2.7B (Transfer)} \longrightarrow \text{LFM2-8B-A1B (Headline Results)}$$
*(Note: Skipping OLMoE and instrumenting directly on Qwen1.5-MoE or LFM2 is acceptable; it only foregoes the prior-art mechanistic comparison baseline, not the validity of the experiment itself).*

### 8.4 Mandatory Telemetry: What to Log on Every MoE Recurrent Pass
*None of the existing retrofit literature covers this phenomenon; measuring it constitutes the primary novel contribution of the MoE experiment:*
1. **Expert Selection Trajectory**: Log top-$k$ expert IDs per token at every recurrent loop ($k = 1 \dots K$).
2. **Inter-Loop Routing Change Rate (Jaccard Drift)**: Compute the fraction of tokens whose active top-$k$ expert set changes between loop $t$ and loop $t+1$:
   $$\Delta_{\text{routing}}^{(t)} = 1 - \frac{|E_t \cap E_{t+1}|}{|E_t \cup E_{t+1}|}$$
3. **Hidden-State Drift Gate vs. Routing Stability**: Determine whether gating the hidden state norm $\|h_t\|$ simultaneously dampens chaotic expert thrashing, or whether router drift compounds independently of representation norm.
4. **Router Logit Entropy Evolution**: Track the softmax entropy of the router gating network:
   $$H(\text{Router}_t) = - \sum_{i=1}^{E} p_i \log p_i$$
   Measure whether the router **collapses** toward a static expert set across loops (specialization / stabilization) or **spreads** chaotically (instability).


