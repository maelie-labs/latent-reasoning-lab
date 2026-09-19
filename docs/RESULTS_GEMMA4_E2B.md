# Model Results & Telemetry: `google/gemma-4-E2B-it`

**Architecture**: Gemma-4 Multimodal Foundation (35 Text Layers, $d=1536$, $n_{\text{heads}}=8$, $n_{\text{kv}}=1$, Heterogeneous Per-Layer Configuration)  
**Calibrated Empirical Scale Factor**: $\alpha_{\text{gemma4\_e2b}} = 0.027496$ ($\mathbb{E}[\|W_E\|] = 1.193636$, $\mathbb{E}[\|h_0\|] = 43.410700$)  
**Thinking Delimiter Protocol**:
- System turn thinking activation: `<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n...<turn|>\n<|turn>model\n`
- Direct non-thinking: `<bos><|turn>user\n...<turn|>\n<|turn>model\n`
- Thinking channel within response: `<|channel>thought\n...<channel|>`
**Hardware Allocations**:
- Step-0 Preflight & Arm 0 Baseline: GPU 0 (`cuda:0`, NVIDIA GeForce RTX 4080 16GB)
- SGLang High-Throughput Serving & Trace Generation: GPU 1 (`cuda:1`, NVIDIA RTX PRO 4500 32GB)

---

## 1. Executive Telemetry & Comparison Matrix (250 Benchmark Suite)

> Evaluated on `data/benchmark_suite_250.json` (150 GSM8K + 100 MATH-500 stratified Levels 1–5, 20 per level) across 4 deterministic seeds: `SEEDS = [42, 123, 456, 789]` ($N=1,000$ total evaluations per arm).  
> Grader: Canonical `math_verify` with symbolic equivalence.  
> Truncation protocol: Any sequence exceeding token ceilings without `<turn|>` or `<channel|>` is strictly scored as incorrect (`is_correct = False`).

| Experimental Arm | Thinking Mode / Configuration | Pass@1 Accuracy (%) | Mean Latency (ms) | Median Latency (ms) | Peak VRAM (MiB) | KV Cache / Token | Throughput (tok/s) | Truncation Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 0** | Surgical Neutrality (MMLU $N=1,000$) | **50.40%** | 60.8 ms | - | 9,889.6 | - | - | 0.00% |
| **Arm 1** | Untrained Base Direct (Non-Thinking) | *Queued (GPU 1 post-1.7B)* | - | - | - | - | - | - |
| **Arm 1b** | Trained Direct No-CoT Control ($K=0$) | *Queued* | - | - | - | - | - | - |
| **Arm 2 ($K=6$)** | Base Discrete Tokens ($K=6$ tokens) | *Queued* | - | - | - | - | - | - |
| **Arm 2 ($K=32$)** | Base Discrete Tokens ($K=32$ tokens) | *Queued* | - | - | - | - | - | - |
| **Arm 2b ($K=6$)** | Trained Pause-Token Control ($K=6$ `<pause>`) | *Queued* | - | - | - | - | - | - |
| **Arm 2b ($K=32$)** | Trained Pause-Token Control ($K=32$ `<pause>`) | *Queued* | - | - | - | - | - | - |
| **Arm 3 ($K=6$)** | Continuous Latent Recurrence ($K=6$ latents) | *Queued* | - | - | - | - | - | - |
| **Arm 3 ($K=32$)**| Continuous Latent Recurrence ($K=32$ latents) | *Queued* | - | - | - | - | - | - |
| **Arm 3a** | Adaptive Halting Recurrence ($K_{\min} \dots K_{\max}$) | *Queued* | - | - | - | - | - | - |
| **Arm 4** | Untrained Base Unconstrained CoT | *Queued (GPU 1 post-1.7B)* | - | - | - | - | - | - |

---

## 2. Phase 0: Step-0 Preflight & Architectural Profiling (COMPLETED)
- **Tokenizer**: 262,144 vocab size with multimodal token reservation (`<|image|>`, `<|audio|>`, `<|video|>`).
- **Jinja Chat Template Analysis**:
  - `enable_thinking=True`: Injects `<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n...<turn|>\n<|turn>model\n`.
  - `enable_thinking=False`: Renders `<bos><|turn>user\n...<turn|>\n<|turn>model\n` directly.
- **Model Topology**:
  - 35 text layers (`Gemma4TextDecoderLayer`) inside `model.language_model.layers`.
  - Hidden size $d=1536$, 8 attention heads, 1 KV head.
  - Intermediate size $d_{\text{mlp}}=8192$.
- **Per-Layer Embeddings (PLE) Discovery**:
  - Gemma 4 incorporates per-layer text embeddings (`embed_tokens_per_layer`, `hidden_size_per_layer_input = 256`).
  - Latent recurrence requires explicit `per_layer_inputs` of shape `[batch, seq_len, 35, 256]` (or zeros/learned PLE vectors).
  - Context projection from `inputs_embeds` combines with `per_layer_inputs` via $(P_{\text{context}} + P_{\text{identity}}) \times \frac{1}{\sqrt{2}}$.
- **Recurrence Loop Stability Verification**:
  - Tested 6 unrolled latent passes under pinned `SDPBackend.MATH` on `cuda:0`.
  - Norm sequence: Step 1: 380.87, Step 2: 372.10, Step 3: 356.39, Step 4: 387.80, Step 5: 375.99, Step 6: 268.11.
  - Zero divergence, zero autograd errors, pure `bfloat16`.
  - Decoded token exit: `' toán'` (token ID: 49110).
- **Calibrated Empirical Scale Factor**:
  - $\mathbb{E}[\|W_E\|] = 1.193636$
  - $\mathbb{E}[\|h_0\|] = 43.410700$
  - $\alpha_{\text{gemma4\_e2b}} = 0.027496$
- **Artifact**: `data/preflight_gemma4_e2b.json`

---

## 3. Phase 0b: Arm 0 Surgical Neutrality Baseline (COMPLETED)
- **Script**: `scripts/71_surgical_neutrality_base_gemma4_e2b.py`
- **Benchmark**: MMLU 1,000 stratified questions across 57 subjects (5-shot prompt).
- **Execution**: `cuda:0` (RTX 4080 16GB), duration: 60.8 seconds (60.8 ms per query).
- **Headline Baseline Accuracy**: **50.40% (504 / 1,000)**
- **Category Breakdown**:
  - **Humanities**: 56.25% (126 / 224)
  - **Social Sciences**: 60.95% (128 / 210)
  - **Other**: 50.22% (113 / 225)
  - **STEM**: 40.18% (137 / 341)
- **Non-Inferiority Target Bound for Trained Adapters**:
  - Any future adapter (Arm 1b, Arm 2b, Arm 3) must achieve $\text{Accuracy} \ge 48.90\%$ ($\Delta \ge -1.5\%$) on this exact 1,000-question distribution to verify zero catastrophic forgetting of base pre-trained knowledge.
- **Artifact**: `data/surgical_neutrality_base_gemma4_e2b.json`

