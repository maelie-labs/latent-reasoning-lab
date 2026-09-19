# Continuous Latent Recurrence, Discrete Scaffolding, and Working Memory: An Empirical and Mechanistic Study Across Language Model Architectures

**Continuous Latent Recurrence Laboratory**  
**Pre-Registration & Certification Protocol**  
**Repository Artifact**: `docs/PAPER_CONTINUOUS_LATENT_RECURRENCE_LAB.md`

---

## Abstract

Language models require extensive test-time computation, often thousands of chain-of-thought (CoT) tokens, to solve complex mathematical, algorithmic, and planning problems. We investigate whether test-time thought trajectories can be compressed into continuous latent recurrence vectors, explicit structured token registers, or dense propositional summaries, and analyze why prior approaches routinely report mixed or brittle gains.

Through four pre-registered experimental studies spanning pure dense Transformers (`Qwen3-1.7B`, `Qwen3-4B`) and hybrid linear-attention architectures (`Qwen3.5-4B`, combining Gated DeltaNet with full self-attention), evaluated across $N=1,000$ queries on a balanced 250-problem suite with canonical CAS grading (`math_verify 0.9.0`), we report three mechanistic negative results and an empirical trade-off:

1. **Continuous Channels Are Inert**: Unrolled latent vectors inserted between prompt and answer fail to transmit causal reasoning across three model architectures and four latent training objectives (answer-only SFT, step-level distillation at three $\lambda$, CODI-style hand-off, and prompt bottleneck) alongside matched controls. On `Qwen3-1.7B`, Pass@1 matches matched direct SFT controls within $\Delta_1 \le +0.50\%$ ($p > 0.60$), while falling strictly $-2.20\%$ below untrained base direct generation and matching dummy pause-token controls bit-for-bit on hard math. Counterfactual donor patching demonstrates near-zero steering ($\Delta\text{Steer} \le +1.25\%$). Mechanistically, while continuous injections amplify $13.6\times$ through self-attention, they are completely bypassed by downstream decoders (<1.8% attention mass). In the hybrid architecture, Gated DeltaNet (GDN) linear-attention layers freeze out the latent input: despite active write gates ($\beta_t = 0.35$), delta-rule innovation collapses $5.0\times$ ($3.55 \to 0.71$) and recurrent state cosine similarity drifts by $<0.03\%$ ($\cos \ge 0.9997$), trapping the recurrent memory in an invariant attractor. Full parameter fine-tuning across all 28 layers of `Qwen3-1.7B` yields flat steering ($\Delta\text{Steer} = -1.06\%$ at $t=3$, $+1.06\%$ at $t=6$), but this result is scientifically inconclusive because uncorrupted unroll recovery failed (Control 0 falling to 72.3%), leaving parameter-efficient LoRA runs as the primary causal evidence.
2. **Structured Tokens Are Decoupled**: Explicit key-value register slots (`<|reg|>key = value<|/reg|>`) achieve 70.00% Pass@1 on `Qwen3-1.7B`, falling below an empty headers-only control ($\Delta = -3.70\%$). Despite receiving 8.21% of the model's self-attention mass, counterfactual corruption of register values produces $\Delta\text{Steer} = 0.00\%$, demonstrating that high attention weight does not imply causal information utilization.
3. **Compression Is Task-Dependent**: Stripping conversational narration into telegraphic propositional rungs induces severe arithmetic collapse on multi-step reasoning in dense Transformers ($-8.00\%$ on GSM8K, worsening to $-36.00\%$ on long problems). In contrast, on the hybrid Gated DeltaNet architecture, telegraphic reasoning matches its scaffolding control ($\Delta = -0.17\%$, n.s.). While this stability is consistent with the hypothesis that persistent recurrent matrix state ($S_t \in \mathbb{R}^{d \times d}$) buffers intermediate calculations, it is equally consistent with both scaffolding arms simply landing on the same SFT-damage floor without headroom for further degradation. Telegraphic reasoning on the hybrid remains $-4.40\%$ below native direct non-thinking mode (87.10% vs. 91.50%), indicating that compressed traces provide no algorithmic benefit over base direct generation.
4. **The Pareto Frontier: Direct Generation Dominates Intermediate Compression**: On `Qwen3-4B`, native direct non-thinking inference is already the efficient operating point: **86.10% Pass@1 at 184.0 median tokens**. Unconstrained verbose thinking serves as the high-accuracy ceiling: **94.30% Pass@1 at 2,455.0 median tokens** (+8.20 points for a $13.3\times$ token expenditure). Every intermediate compressed representation evaluated (including Telegraphic CoT at 73.30% with 533 tokens, Minimal Scratchpad at 76.80% with 612 tokens, and Tool Scratchpad at 79.10% with 792 tokens) is strictly Pareto-dominated by native direct generation, which is both 7.0 to 12.8 points more accurate and 2.9× to 4.3× cheaper. While tool-grounded memory recovers $+5.80\%$ over telegraphic prose within the compressed subspace, the practical implication is direct: if compute budget cannot support the $13\times$ token cost of unconstrained CoT, the optimal strategy is to disable thinking entirely rather than deploy intermediate compression.

---

## 1. Setup and Controls: The Paired Experimental Harness

Scientific investigation into latent and compressed reasoning has historically suffered from subtle methodological confounds: uncalibrated sampling hyper-parameters, mismatched concurrency during serving, inconsistent prompt templates, leaky evaluators, and cross-model distillation. To isolate true algorithmic effects, this project enforced seven non-negotiable architectural invariants across all models and arms.

```
+-----------------------------------------------------------------------------------------+
|                               UNIFIED EVALUATION HARNESS                                |
|                                                                                         |
|  [250-Problem Stratified Suite] (150 GSM8K + 100 MATH-500 L1-L5) x 4 Seeds = 1,000 Qs  |
|                                            |                                            |
|                                            v                                            |
|             +-------------------------------------------------------------+             |
|             | Gate 0: Per-Model Official Config                           |             |
|             +-------------------------------------------------------------+             |
|                                            |                                            |
|                      +---------------------+---------------------+                      |
|                      |                                           |                      |
|                      v                                           v                      |
|         [Untrained Base Direct / CoT]            [Trained Recurrent / Controls]         |
|         (SGLang Concurrency C*=32)               (Batched PyTorch B=16)                 |
|                      |                                           |                      |
|                      +---------------------+---------------------+                      |
|                                            |                                            |
|                                            v                                            |
|                  +---------------------------------------------------+                  |
|                  | Gate 2: Causal Patching Steering Reference        |                  |
|                  | (Delta Steer: Counterfactual Donor Injection)     |                  |
|                  +---------------------------------------------------+                  |
|                                            |                                            |
|                                            v                                            |
|                  +---------------------------------------------------+                  |
|                  | Canonical Grader: math_verify 0.9.0 CAS Sympy     |                  |
|                  +---------------------------------------------------+                  |
|                                            |                                            |
|                                            v                                            |
|                  +---------------------------------------------------+                  |
|                  | Paired Hierarchical Bootstrapping (B=10,000 iter) |                  |
|                  +---------------------------------------------------+                  |
+-----------------------------------------------------------------------------------------+
```

### 1.1 Intra-Model Paired Evaluation Design
All comparisons are strictly intra-model: a trained adapter or recurrent loop is compared exclusively against the untrained base model of the exact same architecture, size, and pre-training checkpoint:
- **Arm 1**: Untrained Base Direct non-thinking (`enable_thinking=False`).
- **Arm 1b**: Trained Direct No-CoT Control (SFT directly on `prompt -> answer`).
- **Arm 2**: Minimal Discrete CoT ($K$ discrete tokens).
- **Arm 2b**: Trained Pause-Token Control (Prompt + $K$ `<pause>` dummy tokens).
- **Arm 3**: Continuous Latent Recurrence ($K$ unrolled latent states in $\mathbb{R}^d$).
- **Arm 4**: Untrained Base Unconstrained CoT (`enable_thinking=True`).

To prevent teacher-student distillation artifacts, every model generated its own training curriculum strictly from its own verified thinking traces on disjoint `train` splits (`openai/gsm8k` train and `DigitalLearningGmbH/MATH-lighteval` train). All test problems remained 100% unseen.

### 1.2 Benchmark Composition and Stratification
The evaluation harness executes across `data/benchmark_suite_250.json`, consisting of exactly 250 problems:
- **150 GSM8K test problems**: Multi-step elementary and intermediate arithmetic.
- **100 MATH-500 test problems**: Rigorously stratified across competition difficulty Levels 1 through 5 (exactly 20 problems per level).

Evaluated across four deterministic seeds (`SEEDS = [42, 123, 456, 789]`), every experimental arm yields exactly $N = 1,000$ evaluated queries.

### 1.3 Canonical CAS Symbolic Verification (`math_verify 0.9.0`)
Heuristic regex scrapers and relaxed string matching fail on mathematical equivalence (e.g., coordinate tuples, fractional expressions, algebraic polynomials). All grading in this work is performed exclusively by canonical `math_verify 0.9.0` utilizing SymPy Computer Algebra System (CAS) parsing. The extraction target is strictly `\boxed{...}` or trailing `#### ...`. Heuristic fallbacks are banned.

### 1.4 Frozen Config Gates (Gate 0 Compliance)
Before any evaluation score is recorded, each run must pass **Gate 0** verification against its architecture's official pre-registered specification:
- **Dense Transformer Family (`Qwen3-1.7B`, `Qwen3-4B`)**:
  * *Direct, Scaffolding & Recurrent Arms*: `temperature = 0.7`, `top_p = 0.80`, `top_k = 20`, `presence_penalty = 1.5`, `max_tokens = 8192`. Presence penalty is implemented via exact additive logit subtraction (`PresencePenaltyLogitsProcessor`).
  * *Unconstrained Thinking (Arm 4)*: `temperature = 0.6`, `top_p = 0.95`, `top_k = 20`, `presence_penalty = 0.0`, thinking ceiling 32,768 tokens, answer ceiling 8,192 tokens.
- **Hybrid Linear-Attention Family (`Qwen3.5-4B`)**:
  * *Official Model Card Benchmark Configuration*: `temperature = 1.0`, `top_p = 0.95`, `top_k = 20`, `presence_penalty = 1.5`, answer cap 8,192 tokens across all evaluation arms.
  * *Thinking Budget*: Audited and set to an 81,920 token ceiling in SGLang (`--max-total-tokens 131072`) following the Step-0 pre-flight truncation audit to guarantee zero artificial truncation on multi-step traces.
- **Strict Truncation Protocol**: Across all models, any sequence hitting generation limits without emitting `<|im_end|>` or `</think>` is strictly scored as incorrect (`is_correct = False`). Truncated queries are never dropped, avoiding survivor-selection bias.

### 1.5 The Calibrated Causal Steering Reference (Gate 2)
Top-line accuracy on downstream benchmarks cannot distinguish whether an adapter actively reads reasoning from an intermediate channel or simply utilizes the channel as a computational delay line while reading the answer from the prompt prefill.

To resolve this, we formulated **Gate 2: Counterfactual Causal Patching**. For $N=100$ verified correct problems, we inject corrupted or counterfactual representations from a foreign donor problem into the candidate channel (continuous latent vector, register token, or scratchpad variable). The causal steering sensitivity is quantified as:
$$\Delta\text{Steer} = \text{Accuracy}_{\text{corrupted}} - \text{Accuracy}_{\text{donor\_match}}$$
A causally grounded channel will follow the injected representation ($\Delta\text{Steer} \gg 0\%$). An uncoupled, bypassed channel will ignore the injection and continue generating the authentic answer derived directly from the uncorrupted prompt prefill ($\Delta\text{Steer} \approx 0\%$).

### 1.6 Statistical Rigor and Bootstrapping
All accuracy deltas ($\Delta$) are verified using paired hierarchical bootstrapping ($B = 10,000$ resamples) clustered by problem ID across seeds, reporting exact two-sided bootstrap $p$-values and true 95% confidence intervals.

---

## 2. Continuous Channels Are Inert: Failures Across Architectures, Objectives, and Ranks

The primary hypothesis of continuous latent recurrence is that replacing discrete language tokens with unrolled continuous vectors $h_k \in \mathbb{R}^d$ allows the model to compute intermediate reasoning in an unconstrained, high-dimensional latent space, reducing latency and KV cache memory.

```
Prompt tokens: [x_1, x_2, ..., x_T]
                     |
                     v
   +------------------------------------+
   |     Continuous Latent Unroll       |
   |  h_1  --->  h_2  ---> ... ---> h_K |  (Hypothesis: Holds intermediate reasoning)
   +------------------------------------+
                     |
                     v  (Read Asymmetry / Attractor Freeze)
           Answer Generation: y_1, y_2, ..., y_M
           (Empirical Reality: Decoder reads prompt directly; ignores h_k)
```

### 2.1 Experimental Scope: Three Architectures, Four Latent Objectives, and Controls
We investigated continuous latent recurrence across three foundation model families:
1. **`Qwen/Qwen3-1.7B`**: Pure dense Transformer (28 layers, hidden dim $d=2048$, 16 attention heads, 8 KV heads).
2. **`Qwen/Qwen3-4B`**: Pure dense Transformer (36 layers, hidden dim $d=2560$, 32 attention heads, 8 KV heads).
3. **`Qwen/Qwen3.5-4B`**: Hybrid Gated DeltaNet + Transformer (32 layers: 24 linear attention layers with $S_t \in \mathbb{R}^{d \times d}$ state + 8 full attention layers).

Across these architectures, we evaluated four latent training objectives alongside matched discrete and scaffolding controls:
- **Objective 1 (Sequence-Level SFT)**: Autoregressive cross-entropy on final answers conditioned on $K$ unrolled latent states ($K \in \{6, 32\}$).
- **Objective 2 (Step-Level Distillation)**: Aligning student latents $h_k$ to post-final-norm teacher CoT vectors ($\lambda_{\text{align}} \in \{0.1, 1.0, 3.0\}$) with continuous removal smoothing ($\lambda_{\text{smooth}}=4$).
- **Objective 3 (CODI-Style Hand-Off)**: Aligning terminal latent $h_K$ directly to the teacher's answer-initiation state.
- **Objective 4 (Prompt Bottleneck)**: Masking prompt prefill tokens during answer generation to force causal routing through the latent channel.
- **Matched Controls**: Paired direct SFT (Arm 1b), matched pause tokens (Arm 2b, $K$ dummy `<pause>` tokens), discrete key-value registers (Study 2), and headers-only scaffolding (Study 3).

### 2.2 Empirical Null Results Across Models
Table 1 summarizes the performance of continuous latent recurrence against base and control arms across $N=1,000$ queries.

#### Table 1: Intra-Model Evaluation of Continuous Latent Recurrence ($N=1,000$ queries, 4 seeds)

| Model Architecture | Arm 1 (Untrained Base Direct) | Arm 1b (Trained Direct SFT) | Arm 2b (Pause Tokens $K=32$) | Arm 3 (Latent Recurrence $K=32$) | Arm 4 (Untrained Verbose CoT) | $\Delta_1$ (Arm 3 - Arm 1b) | $\Delta_2$ (Arm 3 - Arm 2b) | Causal Steering $\Delta\text{Steer}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-1.7B** (Dense 28L) | 73.90% | 71.20% | 71.80% | **71.70%** | **87.10%** | $+0.50\%$ ($p=0.68$) | $\mathbf{-0.10\%}$ ($p=0.92$) | $\mathbf{+1.25\%}$ (Inert) |
| **Qwen3-4B** (Dense 36L) | 86.10% | -* | -* | -* | **94.30%** | -* | -* | $\mathbf{+0.00\%}$ / $\mathbf{+1.06\%}$ (Inert) |
| **Qwen3.5-4B** (Hybrid 32L) | 91.50% | -* | -* | -* | **95.10%** | -* | -* | $\mathbf{0.00\%}$ (Inert) |

*\*Note on Fast-Fail Protocol*: In accordance with the pre-registered fast-fail mandate, full 250-suite benchmark evaluations for Arms 1b, 2b, and 3 on `Qwen3-4B` and `Qwen3.5-4B` were permanently halted because Gate 2 Counterfactual Patching failed decisively ($\Delta\text{Steer} \le +1.06\%$ on 4B; $\Delta\text{Steer} = 0.00\%$ on Qwen3.5-4B with GDN innovation collapse, against the $+10.0\%$ pre-registered threshold). All trained checkpoints and causal patching reports are preserved and cited below.

*Data Provenance & File Citations*:
- `Qwen3-1.7B`:
  * Arm 1 floor: `data/streaming_arm1_direct_qwen_qwen3-1.7b.jsonl` (73.90%)
  * Arm 1b control: `data/eval_arm1b_qwen_qwen3-1.7b.json`, `data/streaming_arm1b_trained_direct_qwen_qwen3-1.7b.jsonl` (71.20%)
  * Arm 2b pause control: `data/streaming_arm2b_k32_pause_qwen3_1.7b.jsonl` (71.80%)
  * Arm 3 latent recurrence: `data/eval_arm3_k32_qwen_qwen3-1.7b.json`, `data/streaming_arm3_k32_qwen_qwen3-1.7b.jsonl` (71.70%)
  * Arm 4 ceiling: `data/streaming_arm4_thinking_qwen_qwen3-1.7b.jsonl` (87.10% canonical; independent replication 86.70% in `data/eval_arm4_qwen_qwen3-1.7b.json`)
  * Bootstraps: `data/paired_bootstrap_arm3_k32_vs_arm1b.json`, `data/paired_bootstrap_arm3_k32_vs_arm2b_k32.json`
- `Qwen3-4B`:
  * Arm 1 floor: `data/eval_250suite_qwen_qwen3-4b_arm1.jsonl` (86.10%)
  * Arm 4 ceiling: `data/eval_250suite_qwen_qwen3-4b_arm4.jsonl` (94.30%)
  * Trained Checkpoints: `checkpoints/lora_arm3_v1_1_qwen_qwen3-4b_k6_lambda1.0`, `checkpoints/lora_arm3_v1_1_qwen_qwen3-4b_k6_codi`
  * Gate 2 Steering Reports: `data/gate2_causal_patching_report_qwen3_4b_lambda1.0.json` ($\Delta\text{Steer} = +0.00\%$), `data/gate2_causal_patching_report_qwen3_4b_codi_t3.json` ($\Delta\text{Steer} = +0.00\%$), `data/gate2_causal_patching_report_qwen3_4b_codi_t6.json` ($\Delta\text{Steer} = +1.06\%$).
- `Qwen3.5-4B`:
  * Arm 1 floor: `data/eval_250suite_qwen_qwen3.5-4b_arm1.jsonl` (91.50%)
  * Arm 4 ceiling: `data/eval_250suite_qwen_qwen3.5-4b_arm4.jsonl` (95.10%)
  * Trained Checkpoint: `checkpoints/lora_arm3_k32_qwen3.5_4b`
  * Gate 2 Steering Report: `data/gate2_causal_patching_report_qwen3.5_4b.json` ($\Delta\text{Steer} = 0.00\%$)
  * GDN Telemetry Report: `data/gdn_gate_telemetry_qwen3.5_4b.json` (Innovation collapse $5.0\times$).

Across all three architectures, continuous latent recurrence fails to demonstrate causal reasoning. On `Qwen3-1.7B`, Arm 3 fails to produce any statistically significant gain over trained direct SFT ($\Delta_1 = +0.50\%$, $p = 0.68$), remains strictly $-2.20\%$ below the untrained base direct model (71.70% vs. 73.90%), and is statistically indistinguishable from dummy `<pause>` tokens ($\Delta_2 = -0.10\%$, $p = 0.92$). On the hardest mathematical strata (MATH Levels 4 and 5), Arm 3 and Arm 2b converge to **bit-exact identity** (61.67% vs. 61.67%). The continuous latent loop acts strictly as a computational delay line.

### 2.3 The Write/Read Asymmetry in Dense Transformers
Probing internal layer representations during unrolling revealed a severe **Write/Read Asymmetry**:
- **Write Sensitivity**: Injecting latent vectors into early transformer layers alters key and value activations significantly, amplifying $13.6\times$ in Euclidean norm across subsequent attention layers.
- **Read Bypass**: However, when the model transitions to generating the answer tokens, the softmax attention distribution over the unrolled latent positions drops below 1.8%. The autoregressive decoder routes attention directly back to the static prompt tokens. The model satisfies the final answer loss by reading the prompt and ignoring the recurrent channel.

### 2.4 The Mechanistic Breakdown: Gated DeltaNet State Freeze
In the hybrid `Qwen3.5-4B` architecture (comprising 24 Gated DeltaNet linear-attention layers and 8 full self-attention layers), linear-attention layers update an explicit recurrent memory matrix $S_t \in \mathbb{R}^{d \times d}$ via the delta rule:
$$S_t = \alpha_t S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^T$$
where $\alpha_t \in [0, 1]$ is the recurrent decay gate, $\beta_t \in [0, 1]$ is the write gate, $k_t$ is the key projection, and $v_t$ is the value projection.

To test whether the continuous recurrence failure on the hybrid model was caused by zero gate activations or projection degeneration, we instrumented and logged the GDN kernel dynamics during latent steps versus adjacent text steps (`scripts/127_measure_gdn_gates_at_latents.py`). The measurements revealed the mechanistic breakdown:

#### Table 2: Mechanistic Telemetry of Gated DeltaNet Layers During Latent Injection

| Metric / Kernel Parameter | Adjacent Prompt Text ($t_{\text{text}}$) | Unrolled Latent Steps ($t_{\text{latent}}$) | Ratio / Mechanistic Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Write Gate ($\beta_t$)** | 0.384 $\pm$ 0.04 | 0.352 $\pm$ 0.03 | $0.92\times$ (Active) | Gate is wide open |
| **Decay Gate ($\alpha_t$)** | 0.832 $\pm$ 0.02 | 0.841 $\pm$ 0.01 | $1.01\times$ (Invariant) | Memory retention normal |
| **Key Projection Norm ($\|k_t\|$)** | 52.1 $\pm$ 4.2 | 49.8 $\pm$ 3.6 | $0.96\times$ (Healthy) | Non-degenerate projection |
| **Value Projection Norm ($\|v_t\|$)** | 68.4 $\pm$ 5.1 | 64.2 $\pm$ 4.8 | $0.94\times$ (Healthy) | Non-degenerate projection |
| **Delta Innovation Norm ($\|v_t - S_{t-1} k_t\|$)** | **3.55 $\pm$ 0.42** | **0.71 $\pm$ 0.11** | **$0.20\times$ ($5.0\times$ collapse)** | **Innovation collapsed** |
| **Matrix State Drift ($\|S_t - S_{t-1}\|_F$)** | **14.2 $\pm$ 1.8** | **2.25 $\pm$ 0.3** | **$0.16\times$ ($6.3\times$ reduction)**| **State frozen** |
| **Recurrent State Cosine ($\cos(S_t, S_{t-1})$)**| 0.942 $\pm$ 0.03 | **0.99974 $\pm$ 0.0001** | **$\Delta\text{Drift} < 0.03\%$** | **Attractor Freeze** |

The mechanistic cause of this freeze stems directly from the update dynamics:
1. The write gate $\beta_t$ does not shut off; it remains fully active at 0.35.
2. The key and value projections are non-degenerate ($\|k\| \approx 50, \|v\| \approx 64$).
3. **The Innovation Collapses**: Because continuous latent vectors are unrolled without discrete token discretization, the term $S_{t-1} k_t$ closely predicts $v_t$. The innovation vector $(v_t - S_{t-1} k_t)$ collapses by $5.0\times$ ($3.55 \to 0.71$).
4. **Attractor Freeze**: With innovation near zero, the state update $\Delta S_t$ drops by $6.3\times$, and the recurrent memory matrix freezes into an invariant attractor ($\cos \ge 0.9997$ across all layers, reaching $1.000000$ at upper layers). The recurrent memory that holds 75% of the model's sequence state treats continuous latents as zero-surprise inputs and refuses to update.

### 2.5 The "Last Retrofit Test": Full Fine-Tuning at 1.7B
To test whether continuous latent bypass was an artifact of parameter-efficient low-rank adaptation (LoRA rank bottleneck) or parameter freezing, we executed the pre-registered **"Last Retrofit Test"**: updating all $1.72 \times 10^9$ parameters across all 28 layers of `Qwen3-1.7B` under identical curriculum supervision (`scripts/129_train_full_ft_qwen3_1.7b.py`).

The full fine-tuning run confirmed the negative result across every metric, with an important qualification:
- **Optimization & Alignment**: Trained across 341 steps at $\text{lr}=1\times 10^{-5}$ (`AdamW8bit`), achieving a final dev cross-entropy loss of **0.0865** (near parity with LoRA's 0.0777) and dev alignment cosine similarity of **77.21%** against teacher CoT states.
- **Dynamical Amplification**: Internal state sensitivity through unrolling remained active and amplified by **$13.04\times$** at $t=3$ (mid-thought) and **$23.15\times$** at $t=6$ (terminal projection), confirming dynamical sensitivity.
- **Causal Steering Invariance**: Causal donor steering across $N=94$ valid problem pairs remained completely flat at the natural chance floor: **$\Delta\text{Steer} = -1.06\%$** at $t=3$ (11.70% steered vs. 12.77% chance) and **$\Delta\text{Steer} = +1.06\%$** at $t=6$ (13.83% steered vs. 12.77% chance), falling far short of the $+10.0\%$ partial grounding threshold (`REJECTED [FAIL / HALT]`).
- **Attention Bypass**: The autoregressive answer decoder allocated $<2\%$ of attention mass to the full-rank unrolled latent vectors, routing cross-attention directly back to the static prompt tokens.
- **Inconclusive Harness Caveat**: Full fine-tuning of all base weights yielded flat donor steering, but this evaluation must be qualified by an essential experimental caveat: the uncorrupted Control 0 null patch identity fell from the healthy ~95–100% baseline observed in clean LoRA runs down to 72.34% ($t=3$) and 73.40% ($t=6$). A 72% null identity represents a failed harness sanity check, indicating that full-parameter unrolling degraded base generation stability. Consequently, the Gate 2 result on the full fine-tuning checkpoint is scientifically inconclusive on its own; the parameter-efficient LoRA runs (where Control 0 passed cleanly without base degradation) remain the primary, robust causal evidence that continuous channels are an architectural bypass rather than an adapter capacity bottleneck.

---

## 3. Structured Tokens Are Decoupled: Dynamic Registers Are Attended to but Ignored

Having established that continuous vectors fail, we investigated whether restricting the communication channel to explicit, structured discrete token slots could enforce causal information transfer.

```
Prompt: [Question text]
           |
           v
[Dynamic Registers]: <|reg|> var_1 = 42 <|/reg|> <|reg|> factor = 7 <|/reg|>
           |
           +---> Attention Mass: 8.21% (Prompt Pre-fill Retains 82.74%)
           |
           +---> Causal Steering: 0.00% (Model ignores register contents)
           |
           v
Answer: Derived entirely from prompt prefill
```

### 3.1 Architecture of Explicit Register Slots
In Study 2 (`studies/dynamic_registers`), we defined explicit discrete register tags:
$$\text{Prompt} \longrightarrow \langle|\text{reg}|\rangle \text{key}_1 = \text{val}_1 \langle|/\text{reg}|\rangle \dots \langle|\text{reg}|\rangle \text{key}_m = \text{val}_m \langle|/\text{reg}|\rangle \longrightarrow \text{Answer}$$
To prevent syntactic OOD failure, we designed three strictly aligned, 1:1 problem-matched curricula across 1,267 training and 100 dev records:
1. **Arm 1 (Authentic Dynamic Registers)**: Verified intermediate numeric variables and equations extracted from ground-truth thinking trajectories.
2. **Control 1 (Headers Only)**: Register blocks stripped entirely, retaining only formatting header delimiters (`Step 1:`, `Step 2:`).
3. **Arm 2 (Matched Filler Registers)**: Identical register keys and syntax as Arm 1, but populated with non-degenerate numeric values drawn randomly from an empirical math pool.

### 3.2 Controlled Benchmark Results
Table 3 shows the evaluation across $N=1,000$ queries on `Qwen3-1.7B`:

#### Table 3: Controlled Evaluation of Structured Dynamic Registers on `Qwen3-1.7B` ($N=1,000$)

| Arm / Configuration | Benchmark Pass@1 | GSM8K ($N=600$) | MATH-500 ($N=400$) | Hard MATH L3–5 ($N=240$) | Paired $\Delta$ vs. Ctrl 1 | Bootstrap $p$-value | Causal Steering $\Delta\text{Steer}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untrained Verbose CoT (Ceiling)** | **87.10%** | 87.17% | 87.00% | 90.83% | $+13.40\%$ | $p < 0.0001$ | - |
| **Untrained Base Direct (Floor)** | **73.90%** | 71.00% | 78.25% | 70.00% | $+0.20\%$ | $p = 0.732$ | - |
| **Control 1 (Headers Only)** | **73.70%** | 74.00% | 73.25% | 64.58% | Reference | - | - |
| **Arm 2 (Matched Filler)** | **72.00%** | 71.17% | 73.25% | 65.00% | $-1.70\%$ | $p = 0.126$ | $\mathbf{0.00\%}$ ($0 / 100$) |
| **Arm 1 (Authentic Registers)** | **70.00%** | 68.33% | 72.50% | 65.00% | **-3.70%** | $p = 0.007$ | $\mathbf{0.00\%}$ ($0 / 100$) |

*Data Provenance*: Control 1 (`studies/dynamic_registers/data/eval_results_control1_headers_only.json`); Arm 1 (`studies/dynamic_registers/data/eval_results_arm1_dynamic_registers.json`); Arm 2 (`studies/dynamic_registers/data/eval_results_arm2_matched_filler.json`); Bootstraps (`studies/dynamic_registers/data/bootstrap_arm1_vs_control1.json`, `studies/dynamic_registers/data/bootstrap_arm1_vs_arm2.json`); Counterfactual Steering (`studies/dynamic_registers/data/register_corruption_results.json`).

The results demonstrate a complete lack of semantic grounding:
- Authentic registers underperform the empty headers-only control by **$-3.70\%$** ($p = 0.0068$), driven by arithmetic degradation on GSM8K (68.33% vs. 74.00%).
- Authentic registers fail to outperform filler registers populated with random numbers from an empirical pool (70.00% vs. 72.00%, $\Delta = -2.00\%$, $p = 0.1260$, n.s.).
- When foreign donor register values are patched into Arm 1, the model achieves **0.00% causal steering**: in 100 out of 100 counterfactual injections, the model ignores the corrupted register numbers and derives the authentic solution directly from the prompt text.

### 3.3 The Attention Mass Dissociation
Probing the layer-by-layer attention distributions revealed that the decoder allocates exactly **8.21% (±2.96%)** of its attention mass to the register tokens, while the static prompt tokens retain **82.74% (±3.14%)** (with headers taking 3.91% and delimiters taking 4.90%).

This establishes a critical interpretability principle: **attention weight is not evidence of causal use**. The Transformer allocates attention to structured register tokens to satisfy positional and syntactic routing, but the semantic values inside the registers are causally inert.

---

## 4. Compression Is Task-Dependent: Execution Loss vs. Recurrent Architectural Memory

In Study 3 (`studies/telegraphic_cot`), we investigated whether natural language reasoning can be compressed by stripping conversational narration and meta-commentary (`"Let's see...", "Wait, let me think about this..."`), retaining only concise propositional logic rungs:
```
<think>
- Given: car travels 60 miles in 1.5 hours.
- Speed = 60 / 1.5 = 40 mph.
- Remaining distance = 140 - 60 = 80 miles.
- Remaining time = 80 / 40 = 2 hours.
- Total time = 1.5 + 2 = 3.5 hours.
</think>
```

### 4.1 Dense Model Collapse on Multi-Step Execution
On dense models (`Qwen3-1.7B` and `Qwen3-4B`), compressing reasoning into telegraphic propositions produced a stark arithmetic collapse relative to both matched scaffolding controls and unconstrained verbose CoT:

#### Table 4: Telegraphic CoT vs. Verbose CoT and Matched Controls on Dense Transformers ($N=1,000$)

| Model Architecture | Evaluated Arm | Pass@1 Overall | GSM8K ($N=600$) | GSM8K-Long ($N=200$) | MATH-500 ($N=400$) | Median Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-1.7B** (Dense 28L) | Untrained Verbose CoT (Ceiling) | **87.10%** | 87.17% | 84.50% | 87.00% | 2,114.0 |
| | Untrained Base Direct (Floor) | **73.90%** | 71.00% | 62.00% | 78.25% | 145.0 |
| | Headers-Only Control | **73.70%** | 74.00% | 72.00% | 73.25% | 350.0 |
| | Telegraphic CoT | **69.00%** | **65.67%** | **61.50%** | 74.00% | 446.5 |
| | *Drop vs. Headers Control* | *-4.70%* | **-8.33%** | *-10.50%* | *+0.75%* | *p < 0.0001* |
| | *Drop vs. Base Direct Floor* | *-4.90%* | **-5.33%** | *-0.50%* | *-4.25%* | *3.1× tok increase* |
| | *Drop vs. Verbose Ceiling* | *-18.10%* | **-21.50%** | *-23.00%* | *-13.00%* | *4.7× reduction* |
| **Qwen3-4B** (Dense 36L) | Untrained Verbose CoT (Ceiling) | **94.30%** | 96.00% | 93.00% | 91.75% | 2,455.0 |
| | Untrained Base Direct (Floor) | **86.10%** | 88.00% | 86.50% | 83.25% | 184.0 |
| | Headers-Only Control | **83.00%** | 83.17% | 73.00% | 82.75% | 409.0 |
| | Telegraphic CoT | **73.30%** | **75.17%** | **57.00%** | 70.50% | 533.0 |
| | *Drop vs. Headers Control* | *-9.70%* | **-8.00%** | *-16.00%* | *-12.25%* | *p < 0.0001* |
| | *Drop vs. Base Direct Floor* | *-12.80%* | **-12.83%** | **-29.50%** | *-12.75%* | *2.9× tok increase* |
| | *Drop vs. Verbose Ceiling* | *-21.00%* | **-20.83%** | **-36.00%** | *-21.25%* | *4.6× reduction* |

*Data Provenance*: Qwen3-1.7B (Arm 1: `data/streaming_arm1_direct_qwen_qwen3-1.7b.jsonl`; Arm 4: `data/streaming_arm4_thinking_qwen_qwen3-1.7b.jsonl` / `studies/telegraphic_cot/RESULTS_TELEGRAPHIC_COT.md`; Telegraphic: `studies/telegraphic_cot/data/eval_results_arm1_telegraphic_cot.json`); Qwen3-4B (Arm 1: `data/eval_250suite_qwen_qwen3-4b_arm1.jsonl`; Arm 4: `data/eval_250suite_qwen_qwen3-4b_arm4.jsonl`; Headers: `data/eval_study3_headers_only_qwen3_4b.json`; Telegraphic: `data/eval_study3_telegraphic_cot_qwen3_4b.json`).

On both dense models, stripping conversational scaffolding induces a nearly identical collapse on arithmetic reasoning relative to matched headers-only scaffolding: **$-8.33\%$ on `Qwen3-1.7B`** (74.00% $\to$ 65.67%) and **$-8.00\%$ on `Qwen3-4B`** (83.17% $\to$ 75.17%). When evaluated against the unconstrained verbose CoT ceiling, this execution deficit widens on long multi-step problems, reaching a **$-36.00\%$ deficit on 4B** (93.00% $\to$ 57.00%).

These empirical deficits are consistent with the hypothesis that autoregressive self-attention layers in dense Transformers rely on intermediate conversational tokens as temporary working-memory registers to store and carry forward intermediate arithmetic states. When these tokens are removed, the model appears to suffer working-memory starvation during multi-step execution.

### 4.2 Hybrid Gated DeltaNet: Scaffolding Penalty Eliminated, But No Gain Over Direct Mode
When the identical telegraphic distillation curriculum was evaluated on the hybrid linear-attention architecture (`Qwen3.5-4B`), the $-8.00\%$ arithmetic penalty against matched scaffolding controls completely vanished:

#### Table 5: Telegraphic CoT on Hybrid Gated DeltaNet (`Qwen3.5-4B`, $N=1,000$)

| Evaluated Arm | Pass@1 Overall | GSM8K ($N=600$) | MATH-500 ($N=400$) | $\Delta$ vs. Headers Control | $\Delta$ vs. Base Direct Floor | $\Delta$ vs. Verbose Ceiling |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untrained Verbose CoT (Ceiling)** | **95.10%** | 93.17% | 98.00% | $+6.70\%$ | $+3.60\%$ | Reference |
| **Untrained Base Direct (Floor)** | **91.50%** | 91.00% | 92.25% | $+3.10\%$ | Reference | $-3.60\%$ |
| **Headers-Only Control (Ctrl 1)** | **88.40%** | 87.00% | 90.50% | Reference | $-3.10\%$ | $-6.70\%$ |
| **Telegraphic Propositional CoT** | **87.10%** | **86.83%** | 87.50% | $\mathbf{-1.30\%}$ ($p=0.28$) | $\mathbf{-4.40\%}$ | $\mathbf{-8.00\%}$ |
| *Arithmetic Delta (GSM8K)* | - | - | - | $\mathbf{-0.17\%}$ ($p=0.96$) | $\mathbf{-4.17\%}$ | $\mathbf{-6.34\%}$ |

*Data Provenance*: `data/eval_250suite_qwen_qwen3.5-4b_arm4.jsonl`, `data/eval_250suite_qwen_qwen3.5-4b_arm1.jsonl`, `data/eval_study3_headers_only_qwen3_5_4b.json`, `data/eval_study3_telegraphic_cot_qwen3_5_4b.json`.

Two critical empirical deductions follow:
1. **Scaffolding Parity Consistent with Working Memory**: Relative to the matched headers-only control, the hybrid model exhibits an arithmetic delta of only **$-0.17\%$** ($p = 0.9626$, n.s.), compared to $-8.00\%$ in dense Transformers. While this stability is consistent with the hypothesis that the recurrent state matrix $S_t \in \mathbb{R}^{d \times d}$ buffers intermediate arithmetic transitions, it is equally consistent with an alternative explanation: both scaffolding arms (headers-only and telegraphic) may have landed on the same SFT-damage floor (~87–88%) relative to the base direct model (91.50%), leaving no room for a further arithmetic penalty.
2. **Compression Adds No Accuracy Over Direct Generation**: Telegraphic CoT (87.10%) remains **$-4.40\%$ below Untrained Base Direct (91.50%)** and **$-8.00\%$ below Verbose CoT (95.10%)**. The fact that telegraphic reasoning matches its scaffolding control on the hybrid does not indicate that recurrent memory enabled new reasoning capability; rather, the compressed trace added nothing over native direct generation, consistent with the findings across the laboratory.

---

## 5. Intermediate Reasoning Compression: Working Memory, the Monotonic Dial, and Non-Thinking Dominance

The findings of Studies 1–3 define the precise boundary of test-time compression:
- Continuous vectors fail because the pre-trained decoder cannot read them.
- Discrete registers fail because the model routes around ungrounded text slots.
- Dense models collapse when stripped of working memory for intermediate calculations.

In Study 4 (`studies/tool_grounded_scratchpad`), we investigated whether an explicit, native tool-grounded working memory scratchpad could restore intermediate calculation buffers without conversational bloat.

```
User Query: [Complex Math / Multi-Step Reasoning Problem]
                     |
                     v
   +---------------------------------------------------+
   |             In-Place State Mutation               |
   | <tool_call>                                       |
   | {"name": "update_scratchpad",                     |
   |  "arguments": {"x": 13, "remainder": 5}}          |
   | </tool_call>                                      |
   +---------------------------------------------------+
                     |
                     v  (Environment returns updated working memory map)
   +---------------------------------------------------+
   |               Ground-Truth Execution              |
   | - Working state: {x: 13, remainder: 5}            |
   | - Final Answer: \boxed{65}                        |
   +---------------------------------------------------+
```

### 5.1 Architecture and Training
Rather than inventing artificial tokens, the model invokes native tool-calling syntax pre-trained in foundation models:
$$\langle\text{tool\_call}\rangle\backslash\text{n}\{\text{"name": "update\_scratchpad", "arguments": \{...\}}\}\backslash\text{n}\langle/\text{tool\_call}\rangle$$
The environment executes the state update and returns the compact working memory map as a tool message. The assistant then completes the solution conditioned on the verified state.

We curated 1,267 training and 100 held-out dev traces, training a rank-16 LoRA adapter on `Qwen3-4B` with loss masking applied to prompt tokens and tool responses, supervising only tool calls and answer derivations.

### 5.2 Empirical Evaluation Across Strata
Table 6 compares the Tool Scratchpad against all baseline and compressed arms on `Qwen3-4B` across $N=1,000$ queries.

#### Table 6: The Full Compression Frontier on `Qwen3-4B` ($N=1,000$)

| Evaluation Arm | Pass@1 Overall | GSM8K Overall | GSM8K-Long ($N=200$) | MATH-500 Overall | MATH Level 4 ($N=80$) | MATH Level 5 ($N=80$) | Median Tokens | P90 Tokens | Mean Tokens | Status / Frontier Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untrained Base Direct** (Arm 1) | **86.10%** | 88.00% | 86.50% | 83.25% | 76.25% | 61.25% | **184.0** | 247.0 | 186.2 | **Native Efficient Point (Floor)** |
| **Telegraphic CoT** (Study 3) | **73.30%** | 88.00% | 57.00% | 51.25% | 60.00% | 50.00% | **533.0** | 1,067.6 | 654.4 | Dominated (-12.80% acc, 2.9× tok) |
| **Minimal Tool Scratchpad** (Dial) | **76.80%** | 86.50% | 66.00% | 62.25% | 65.00% | 50.00% | **612.0** | 1,420.5 | 788.1 | Dominated (-9.30% acc, 3.3× tok) |
| **Tool Scratchpad** (Study 4) | **79.10%** | **80.33%** | **71.00%** | **77.25%** | **68.75%** | **51.25%** | **792.0** | **2,013.3** | **1,181.2** | Dominated (-7.00% acc, 4.3× tok) |
| **Untrained Verbose CoT** (Arm 4) | **94.30%** | **96.00%** | **93.00%** | **91.75%** | **90.00%** | **91.25%** | **2,455.0** | **6,763.7** | **3,214.6** | **High-Accuracy Ceiling** |

*Data Provenance*:
- Untrained Base Direct: `data/eval_250suite_qwen_qwen3-4b_arm1.jsonl`
- Telegraphic CoT: `data/eval_study3_telegraphic_cot_qwen3_4b.json`
- Minimal Tool Scratchpad: `data/eval_study4_minimal_scratchpad_dial_qwen3_4b.json`
- Tool Scratchpad: `data/eval_study4_tool_scratchpad_qwen3_4b.json`
- Untrained Verbose CoT: `data/eval_250suite_qwen_qwen3-4b_arm4.jsonl`

### 5.3 Non-Thinking Dominance: Why Intermediate Compression Fails to Beat Base Direct
The central empirical reality revealed by Table 6 is that **every intermediate compressed representation is strictly Pareto-dominated by native zero-thought execution**:

1. **The Native Baseline Is Already the Efficient Point**: On `Qwen3-4B`, Untrained Base Direct non-thinking mode achieves **86.10% Pass@1 at 184.0 median tokens**. The authentic efficient frontier connects this point directly to Untrained Verbose CoT (**94.30% at 2,455.0 median tokens**), which purchases +8.20 points of accuracy at a 13.3× token expenditure.
2. **Strict Pareto Domination**: Nothing we trained lies on or above the line connecting the two native modes. The Tool Scratchpad achieves **79.10%** ($-7.00\%$ accuracy) while consuming **792.0 median tokens** (+608 tokens, a $4.3\times$ inflation). The model spends over 600 additional tokens per query to arrive at a solution that is 7 percentage points less accurate than simply generating the answer directly with thinking disabled.
3. **Pervasive Across All Problem Strata**: The base model's direct execution dominates the tool scratchpad across every single problem stratum:
   - **GSM8K Overall**: 88.00% (Base Direct) vs. 80.33% (Tool Scratchpad): $+7.67\%$ higher accuracy at $4.3\times$ lower token cost.
   - **GSM8K-Long**: 86.50% vs. 71.00%: $+15.50\%$ higher accuracy.
   - **MATH-500 Overall**: 83.25% vs. 77.25%: $+6.00\%$ higher accuracy.
   - **MATH Level 4**: 76.25% vs. 68.75%: $+7.50\%$ higher accuracy.
   - **MATH Level 5**: 61.25% vs. 51.25%: $+10.00\%$ higher accuracy.
4. **The Monotonic Dial**: To determine whether an optimal trade-off exists between telegraphic reasoning and tool scratchpads, we evaluated the **Minimal Tool Scratchpad** (collapsing prose to compact bullets while retaining every computing line). It achieves **76.80% at 612 tokens**. This intermediate point does not identify an efficient knee; instead, it establishes a strictly monotonic descent connecting the Tool Scratchpad (79.10%, 792 tok) down to Telegraphic CoT (73.30%, 533 tok).
5. **The Measured Empirical Constraint**: Constrained intermediate representations (including telegraphic bullet points, explicit token registers, and tool scratchpad updates) consistently degrade accuracy while inflating token expenditures relative to the base model's native direct mode. Generating structured intermediate artifacts truncates or disrupts the native execution paths without purchasing back the lost accuracy. While the precise internal mechanistic cause remains open, the measured fact is definitive: on `Qwen3-4B`, you cannot compress execution below the base model's zero-thought capability without losing accuracy, and intermediate scratchpad tokens do not buy back what is lost by truncating unconstrained reasoning.

### 5.4 Cost per Point Metric and Practitioner Trade-Offs
To formalize the economic trade-offs of test-time compute, we compute the marginal cost per point of accuracy:
$$\text{Marginal Cost per Point} = \frac{\Delta \text{Tokens}}{\Delta \text{Pass@1}}$$

#### Table 7: Marginal and Cumulative Cost per Point on `Qwen3-4B` Relative to True Baselines

| Transition / Configuration | $\Delta$ Pass@1 (%) | $\Delta$ Median Tokens | Marginal Cost (Tok / Point) | P90 Marginal Cost (P90 Tok / Point) | Deployment Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Untrained Direct $\to$ Untrained Verbose CoT** | **$+8.20\%$** | **$+2,271.0$** | **277.0 tok / pt** | **794.7 tok / pt** | **Authentic Frontier: Purchases +8.20% ceiling** |
| **Untrained Direct $\to$ Tool Scratchpad** | **$-7.00\%$** | **$+608.0$** | **Negative Return** | **Negative Return** | **Pareto-Dominated: 4.3× more tokens, -7.00% acc** |
| **Untrained Direct $\to$ Minimal Scratchpad** | **$-9.30\%$** | **$+428.0$** | **Negative Return** | **Negative Return** | **Pareto-Dominated: 3.3× more tokens, -9.30% acc** |
| **Untrained Direct $\to$ Telegraphic CoT** | **$-12.80\%$** | **$+349.0$** | **Negative Return** | **Negative Return** | **Pareto-Dominated: 2.9× more tokens, -12.80% acc** |
| *Sub-Space Recovery: Telegraphic $\to$ Minimal* | $+3.50\%$ | $+79.0$ | 22.6 tok / pt | 100.8 tok / pt | Internal recovery within compressed subspace |
| *Sub-Space Recovery: Minimal $\to$ Scratchpad* | $+2.30\%$ | $+180.0$ | 78.3 tok / pt | 257.7 tok / pt | Monotonic recovery toward native capabilities |
| *Sub-Space Recovery: Telegraphic $\to$ Scratchpad* | $+5.80\%$ | $+259.0$ | 44.7 tok / pt | 163.0 tok / pt | Tool grounding restores partial arithmetic loss |

*Practical Recommendations*:
1. **If token budget permits**: Run unconstrained verbose CoT. It purchases the true ceiling (+8.20% accuracy over direct mode) at a marginal cost of 277 tokens per point.
2. **If token budget is constrained**: **Turn thinking off.** Native direct generation (`enable_thinking=False`) is the true efficient operating point: it is 7.0 points more accurate than the best scratchpad adapter and $4.3\times$ cheaper.
3. **Avoid intermediate compression**: Intermediate representations (telegraphic CoT, register tokens, tool scratchpads) introduce adapter overhead and reasoning truncation, degrading accuracy while inflating token count relative to the native base direct model.

---

## 6. Attention Mass Is Not Grounding: Unifying the Three Negative Mechanisms

Across our three negative studies, the central finding is the dissociation between **Attentional Allocation** and **Causal Utilization**.

```
+-----------------------------------------------------------------------------------------+
|                   ATTENTION VS. CAUSAL UTILIZATION SUMMARY                              |
|                                                                                         |
|  1. Continuous Latents:                                                                 |
|     - Amplifies 13.6x in Self-Attention                                                 |
|     - Downstream Decoder Read: < 1.8%                                                   |
|     - Gated DeltaNet State Update: cos >= 0.9997 (Zero-Surprise Freeze)                  |
|                                                                                         |
|  2. Dynamic Registers:                                                                  |
|     - Attracts 8.21% of Attention Mass (Prompt Retains 82.74%)                          |
|     - Counterfactual Causal Steering: Delta Steer = 0.00% (Prompt prefill bypass)       |
|                                                                                         |
|  3. Telegraphic Scaffolding:                                                            |
|     - Attended to by Dense Layers, but numbers lost without discrete buffers            |
|     - Penalty Vanishes on Hybrid GDN (Delta = -0.17% vs Ctrl, but -4.40% vs Base)       |
+-----------------------------------------------------------------------------------------+
```

1. **Continuous Latents**: The model expends significant attention bandwidth routing continuous vectors, amplifying their norm by $13.6\times$ through self-attention layers. Yet downstream decoders ignore them, and linear-attention recurrent states treat them as zero-surprise invariant attractors ($\cos \ge 0.9997$).
2. **Dynamic Registers**: The model allocates 8.21% (±2.96%) of its cross-attention mass directly to register tokens (with prompt prefill retaining 82.74%). Yet replacing register contents with counterfactual donor numbers produces $\Delta\text{Steer} = 0.00\%$. The attention heads attend to the registers to maintain sequence positional alignment, while the factual parameters are fetched directly from the static prefill prompt.
3. **Scaffolding Parity on Recurrent Hybrids**: On linear-attention hybrids, the $-8.00\%$ arithmetic drop between telegraphic prose and headers-only scaffolding disappears ($\Delta = -0.17\%$). However, because both scaffolding arms remain ~4 percentage points below native base direct generation (87.10% vs. 91.50%), this parity cannot be claimed as true algorithmic reasoning synthesis; it is equally consistent with both formats landing on a common SFT degradation floor.

**Conclusion**: High attention weight is a necessary prerequisite for routing, but it is completely insufficient to prove causal grounding. Transformers routinely look at tokens and vectors they do not use, and refuse to write to channels they do look at.

---

## 7. Related Work and Prior Art

Our investigation intersects four primary bodies of prior research on neural reasoning, recurrence, and interpretability:

1. **Continuous Latent Reasoning and Soft Thinking**:
   Prior work has investigated replacing discrete chain-of-thought tokens with continuous vector representations. Hao et al. (2024) proposed Coconut, training language models on synthetic logic tasks (ProsQA) by progressively replacing CoT tokens with continuous hidden states through multi-stage curricula. Shen et al. (2025) introduced CODI, employing sequence-level distillation to compress thoughts into continuous representations. Goyal et al. (2023) explored pause tokens, inserting dummy `<pause>` tokens to allocate test-time compute prior to emission. Our work evaluates these approaches on standardized mathematical benchmarks (GSM8K and MATH-500) under rigorous paired controls, establishing that unrolled continuous vectors act as computational delay lines rather than semantic reasoning channels in pre-trained transformer decoders.

2. **Depth Recurrence and Looped Transformers**:
   A distinct paradigm applies recurrent computation across network depth rather than extending the sequence dimension. Geiping et al. (2025) pre-trained Huginn-3.5B from scratch with a recurrent block, showing that additional recurrent iterations provide modest gains on arithmetic but fail to reach discrete chain-of-thought performance. Lu et al. (2025) probed depth-recurrent models and identified representational discontinuities across recurrent cycles. Bartoldson et al. (2025) and Park et al. (2026, LoopUS) retrofitted depth recurrence into open-weight models (Llama and Qwen architectures), observing that looped blocks require stabilization mechanisms (such as selective gates) to avoid representation collapse. Unlike depth recurrence, which loops over all prompt tokens, our Study 1 evaluates single-token sequence-extending recurrence where the prompt cache is frozen once.

3. **Causal Interpretability and Activation Patching**:
   Our mechanistic audit builds on causal mediation analysis in transformers (Meng et al., 2022; Wang et al., 2022; Geva et al., 2023). While earlier studies often relied on perturbation sensitivity or attention visualization to infer representation content, recent methodological analyses emphasize that observable attention patterns do not constitute causal explanations. Our counterfactual donor patching protocol and attention-decomposition analyses (Studies 1 and 2) confirm this dissociation empirically: high attention mass does not imply causal information utilization.

4. **Discrete Working Memory and Scaffolding**:
   Nye et al. (2021) introduced scratchpads for intermediate algorithm execution in language models. Subsequent research has explored constrained prompt scaffolding and compressed reasoning formats. Our Studies 3 and 4 evaluate the limits of intermediate reasoning compression, demonstrating that on dense models, removing natural language intermediate tokens degrades multi-step arithmetic, and that native base direct generation Pareto-dominates intermediate compressed representations.

5. **Hybrid Linear-Attention Architectures**:
   Recent hybrid models combine linear attention with standard softmax attention to achieve sub-quadratic sequence scaling. The Gated DeltaNet architecture (Schlag et al., 2021; Yang et al., 2024) maintains an explicit recurrent memory matrix updated via the delta rule. In Study 1 and Study 3, we evaluated hybrid models (Qwen3.5-4B), discovering that continuous latents cause delta-rule innovation collapse, but that the recurrent state matrix provides working memory buffers that eliminate the arithmetic penalty of telegraphic text.

For an extensive literature survey and background analysis, refer to [`docs/OUTSIDE_VIEW_RESEARCH.md`](OUTSIDE_VIEW_RESEARCH.md) and [`docs/SCIENTIFIC_LANDSCAPE_AND_CONTRIBUTIONS.md`](SCIENTIFIC_LANDSCAPE_AND_CONTRIBUTIONS.md).

---

## 8. Scope and What Remains Open

To maintain strict scientific integrity, we explicitly demarcate the boundaries of these findings:

1. **Pre-Trained Retrofits vs. Pre-Training from Scratch**:
   - Our findings apply to **retrofitting pre-trained foundation models** (via LoRA, full fine-tuning, or curriculum distillation). A model pre-trained on billions of tokens of natural language has its entire internal representation organized around discrete linguistic tokens.
   - It remains an open question whether a model trained **from token zero (pre-training from scratch)** with continuous latent recurrence could develop native representations that read and write causally to continuous channels.
2. **Search vs. Deductive Tracking**:
   - The Tool Scratchpad recovers structured state on deductive and arithmetic problems (GSM8K, MATH Levels 1–4), though it remains dominated by native base direct execution.
   - However, **combinatorial proof search** (MATH Level 5) resists compression: achieving >90% accuracy on Level 5 competition math appears to require expansive discrete exploration trees to test and reject candidate proofs. In our experiments, proof search was not compressed by any representation we tested.
3. **Reinforcement Learning vs. Supervised SFT**:
   - In this work, we focused on supervised curriculum alignment and state matching. Reinforcement learning (e.g., PPO or GRPO with pure outcome rewards) might discover non-trivial latent utilization strategies that supervised gradients fail to discover, though credit assignment through continuous unrolling remains an acute theoretical obstacle.
4. **Model Scaling Limits**:
   - We verified these invariants across 1.7B and 4B models (with Qwen3.5-2B evaluated during Phase 0 baseline calibration). Whether massive frontier models (>32B, 70B+) exhibit emergent continuous channel grounding without architectural modification remains an empirical question for future high-compute laboratories.
5. **Workload and Model Scope of the Pareto Frontier**:
   - The empirical Pareto frontier, which demonstrates that native direct non-thinking mode strictly dominates intermediate compressed representations, was measured on mathematical and arithmetic deduction benchmarks on `Qwen3-4B`. Agentic and tool-orchestration workloads (such as environments requiring dynamic observations and multi-turn planning) were not evaluated in the certified benchmark suite following the withdrawal of preliminary uncalibrated runs, and remain an open regime for future investigation.

---

## 9. Summary of Completed Laboratory Artifacts

| Study Area | Primary Script / Tool | Empirical Artifact | Certified Status |
| :--- | :--- | :--- | :---: |
| **Uniform Suite** | `scripts/generate_stratified_250_suite.py` | `data/benchmark_suite_250.json` | **VERIFIED** ($N=1,000$, 0 test overlap) |
| **Study 1: Continuous Latents** | `scripts/41_train_qwen3_arms.py` | `data/eval_arm3_k32_qwen_qwen3-1.7b.json` | **CERTIFIED** ($\Delta_2 = -0.10\%$, n.s.) |
| **Study 1: 1.7B Full Fine-Tuning**| `scripts/129_train_full_ft_qwen3_1.7b.py` | `docs/RESEARCH_LOG.md` (Milestone 51) | **CERTIFIED** (Dev Cosine 77.2%, $\Delta\text{Steer}$ -1.06% / +1.06%, Inconclusive Harness Caveat) |
| **Study 1: GDN Mechanism** | `scripts/127_measure_gdn_gates_at_latents.py` | `docs/RESEARCH_LOG.md` (Milestone 46) | **CERTIFIED** (Innovation collapse $5\times$, $\cos \ge 0.9997$) |
| **Study 2: Dynamic Registers** | `studies/dynamic_registers/scripts/03_eval.py` | `studies/dynamic_registers/data/eval_results_arm1_dynamic_registers.json` | **CERTIFIED** (70.00% Pass@1, $\Delta = -3.70\%$, $\Delta\text{Steer} = 0.00\%$) |
| **Study 3: Telegraphic CoT** | `studies/telegraphic_cot/scripts/03_eval.py` | `data/eval_study3_telegraphic_cot_qwen3_4b.json` | **CERTIFIED** ($-8.00\%$ on dense, $-0.17\%$ on GDN) |
| **Study 4: Tool Scratchpad** | `studies/tool_grounded_scratchpad/scripts/04_eval.py` | `data/eval_study4_tool_scratchpad_qwen3_4b.json` | **CERTIFIED** (79.10% Pass@1, 792 med tok, $+14\%$ recovery) |
| **Bootstrap Analysis** | `studies/tool_grounded_scratchpad/scripts/05_boot.py` | `data/paired_bootstrap_study4_tool_scratchpad.json` | **CERTIFIED** ($B=10,000$ iterations) |

---

## References

1. Bartoldson, B. R., et al. (2025). *Retrofitting Language Models with Depth Recurrence*. arXiv preprint arXiv:2511.07384.
2. Geiping, J., et al. (2025). *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*. arXiv preprint arXiv:2502.05171.
3. Geva, M., et al. (2023). *Dissecting Recall of Factual Associations in Auto-Regressive Language Models*. In Proceedings of EMNLP 2023.
4. Goyal, S., et al. (2023). *Think before you speak: Training Language Models with Pause Tokens*. In Proceedings of ICLR 2024.
5. Hao, S., Gu, Y., Ma, H., Hong, J. J., Wang, Z., Wang, D. Z., & Hu, Z. (2024). *Training Large Language Models to Reason in a Continuous Latent Space*. arXiv preprint arXiv:2412.06769.
6. Lu, K., et al. (2025). *Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer*. arXiv preprint arXiv:2507.02199.
7. Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and Editing Factual Associations in GPT*. In Advances in Neural Information Processing Systems (NeurIPS 2022).
8. Nye, M., Andreassen, A. J., Gambardella, G., et al. (2021). *Show Your Work: Scratchpads for Intermediate Computation with Language Models*. arXiv preprint arXiv:2112.00114.
9. Park, J., et al. (2026). *LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models*. arXiv preprint arXiv:2605.11011.
10. Schlag, I., Irie, K., & Schmidhuber, J. (2021). *Linear Transformers Are Secretly Fast Weight Programmers*. In Proceedings of ICML 2021.
11. Shen, Y., et al. (2025). *Chain-of-Thought Distillation in Continuous Space*. arXiv preprint.
12. Wang, K., et al. (2022). *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 small*. In Advances in Neural Information Processing Systems (NeurIPS 2022).
13. Wei, J., Wang, X., Schuurmans, D., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. In Advances in Neural Information Processing Systems (NeurIPS 2022).
14. Yang, A., et al. (2025). *Qwen2.5 Technical Report*. arXiv preprint arXiv:2412.15115.
15. Yang, S., et al. (2024). *Gated Linear Attention Transformers with Hardware-Efficient Training*. In Proceedings of ICML 2024.
16. Zhu, Z., et al. (2025). *Scaling Latent Reasoning via Looped Language Models*. arXiv preprint.
