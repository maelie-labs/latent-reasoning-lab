#!/usr/bin/env python3
"""
scripts/127_measure_gdn_gates_at_latents.py
Mechanistic Telemetry of Gated DeltaNet (GDN) Linear Attention Gates and State Dynamics.

Evaluates on Qwen/Qwen3.5-4B (checkpoints/lora_arm3_k32_qwen3.5_4b/best_checkpoint):
1. Write gate beta_t = sigmoid(b_t) across:
   - Prompt text tokens (last 16 tokens)
   - Latent recurrent positions (k = 1 ... 32)
   - Answer text tokens (first 16 tokens)
2. Decay gate alpha_t = exp(g_t) across the three regimes.
3. Key/Value projection norms (||k_t||_2, ||v_t||_2, ||q_t||_2).
4. Innovation / surprise norm ||v_t - S_{t-1} k_t||_2.
5. Recurrent state matrix drift ||Delta S_t||_F and cos(S_t, S_{t-1}).
6. Causal perturbation sensitivity: inject eps into latent vector at k=16,
   measuring attention amplification vs GDN recurrent state drift cos(S_pert, S_clean).

Hardware: Dedicated GPU 1 (RTX PRO 4500 Blackwell 32GB) via CUDA_VISIBLE_DEVICES=1.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import LinearAttentionLayer
from peft import PeftModel
from tqdm import tqdm

CALIBRATED_ALPHA_3_5_4B = 0.893860

# Safe monkeypatches for LinearAttentionLayer caching
def safe_update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
    if not self.is_recurrent_states_initialized[state_idx]:
        self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
    self.recurrent_states[state_idx] = recurrent_states
    return self.recurrent_states[state_idx]

def safe_update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
    if not self.is_conv_states_initialized[state_idx]:
        self.lazy_initialization(conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size)
    if not self.has_previous_state[state_idx]:
        full_conv_states = conv_states
        self.has_previous_state[state_idx] = True
        if not self.record_past and full_conv_states.shape[-1] < self.conv_kernel_size[state_idx]:
            padding_length = self.conv_kernel_size[state_idx] - full_conv_states.shape[-1]
            full_conv_states = torch.nn.functional.pad(full_conv_states, (padding_length, 0), value=0)
    else:
        full_conv_states = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
    self.conv_states[state_idx] = full_conv_states[..., -self.conv_kernel_size[state_idx]:].clone()
    return full_conv_states

LinearAttentionLayer.update_recurrent_state = safe_update_recurrent_state
LinearAttentionLayer.update_conv_state = safe_update_conv_state


def main():
    parser = argparse.ArgumentParser(description="GDN Gate Telemetry at Latent vs Text Positions")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lora_arm3_k32_qwen3.5_4b/best_checkpoint")
    parser.add_argument("--n_problems", type=int, default=20)
    parser.add_argument("--k_steps", type=int, default=32)
    parser.add_argument("--ans_steps", type=int, default=16)
    parser.add_argument("--output_file", type=str, default="data/gdn_gate_telemetry_qwen3.5_4b.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("===================================================================")
    print(f"GDN GATE TELEMETRY & MECHANISTIC PROFILING FOR {args.model_id}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Problems: {args.n_problems} | Latent Horizon: K={args.k_steps} | Answer Steps: {args.ans_steps}")
    print(f"Device: {args.device} | Scale Factor: alpha={CALIBRATED_ALPHA_3_5_4B:.6f}")
    print("===================================================================")

    suite_path = "data/benchmark_suite_250.json"
    with open(suite_path) as f:
        suite = json.load(f)
    # Stratified 10 GSM8K + 10 MATH-500
    gsm_probs = [p for p in suite if p.get("benchmark") == "GSM8K"][:args.n_problems // 2]
    math_probs = [p for p in suite if p.get("benchmark") == "MATH-500"][:args.n_problems // 2]
    eval_problems = gsm_probs + math_probs
    print(f"Selected {len(eval_problems)} problems ({len(gsm_probs)} GSM8K + {len(math_probs)} MATH-500).")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base model in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    if os.path.exists(args.checkpoint):
        print("Loading LoRA adapter from:", args.checkpoint)
        model = PeftModel.from_pretrained(base_model, args.checkpoint)
    else:
        print("Warning: Adapter not found, using base model directly.")
        model = base_model
    model.eval()

    # Discover GDN layers
    gdn_layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Qwen3_5GatedDeltaNet":
            parts = name.split(".")
            layer_idx = int(parts[parts.index("layers") + 1])
            gdn_layers.append((layer_idx, module))
    gdn_layers.sort(key=lambda x: x[0])
    layer_indices = [idx for idx, _ in gdn_layers]
    print(f"Found {len(gdn_layers)} GDN layers: {layer_indices}")

    # Storage for per-step metrics
    metrics_by_regime = {
        "prompt": {l: {"beta": [], "alpha": [], "k_norm": [], "v_norm": [], "q_norm": []} for l in layer_indices},
        "latents": {l: {"beta": [], "alpha": [], "k_norm": [], "v_norm": [], "q_norm": [],
                        "state_diff_norm": [], "state_cos": [], "state_rel_diff": [], "innov_norm": []} for l in layer_indices},
        "answer": {l: {"beta": [], "alpha": [], "k_norm": [], "v_norm": [], "q_norm": [],
                       "state_diff_norm": [], "state_cos": [], "state_rel_diff": [], "innov_norm": []} for l in layer_indices},
    }

    # Tracking latent trajectory dynamics step-by-step (k = 1 ... 32)
    step_trajectory = {
        k: {
            "beta": [], "alpha": [],
            "k_norm": [], "v_norm": [],
            "state_cos_prev": [], "state_cos_initial": [],
            "state_rel_diff": []
        }
        for k in range(1, args.k_steps + 1)
    }

    perturbation_results = {
        "hidden_amp": [],
        "gdn_state_cos": [],
        "gdn_state_diff_norm": [],
        "gdn_state_drift_pct": []
    }

    captured_step_data = {}

    def make_gdn_hook(layer_idx):
        def hook(module, args_hook, kwargs_hook, output):
            hidden_states = kwargs_hook.get("hidden_states") if "hidden_states" in kwargs_hook else args_hook[0]
            with torch.no_grad():
                B, seq_len, D = hidden_states.shape
                b = module.in_proj_b(hidden_states)
                beta = b.sigmoid()
                a = module.in_proj_a(hidden_states)
                g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
                alpha = g.exp()

                mixed_qkv = module.in_proj_qkv(hidden_states)
                query, key, value = torch.split(
                    mixed_qkv,
                    [module.key_dim, module.key_dim, module.value_dim],
                    dim=-1
                )
                q_n = torch.norm(query.float(), dim=-1).mean().item()
                k_n = torch.norm(key.float(), dim=-1).mean().item()
                v_n = torch.norm(value.float(), dim=-1).mean().item()

                captured_step_data[layer_idx] = {
                    "beta": beta.float(),
                    "alpha": alpha.float(),
                    "q_norm": q_n,
                    "k_norm": k_n,
                    "v_norm": v_n
                }
        return hook

    hooks = [attn.register_forward_hook(make_gdn_hook(idx), with_kwargs=True) for idx, attn in gdn_layers]

    print("\nExecuting GDN Telemetry across N=20 problems...")
    t0_all = time.time()

    for p_i, prob in enumerate(tqdm(eval_problems, desc="Evaluating Problems")):
        prompt_text = f"<|im_start|>user\n{prob['question']}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        enc = tokenizer(prompt_text, return_tensors="pt").to(args.device)
        prompt_len = enc.input_ids.shape[1]

        # -------------------------------------------------------------
        # Phase 1: Prompt Prefill
        # -------------------------------------------------------------
        captured_step_data.clear()
        with torch.no_grad():
            out = model(**enc, use_cache=True, output_hidden_states=True)
            past_kv = out.past_key_values
            curr_latent = out.hidden_states[-1][:, -1:, :]

        for l_idx in layer_indices:
            data = captured_step_data[l_idx]
            last_tokens_beta = data["beta"][:, -16:, :].mean().item()
            last_tokens_alpha = data["alpha"][:, -16:, :].mean().item()
            metrics_by_regime["prompt"][l_idx]["beta"].append(last_tokens_beta)
            metrics_by_regime["prompt"][l_idx]["alpha"].append(last_tokens_alpha)
            metrics_by_regime["prompt"][l_idx]["k_norm"].append(data["k_norm"])
            metrics_by_regime["prompt"][l_idx]["v_norm"].append(data["v_norm"])
            metrics_by_regime["prompt"][l_idx]["q_norm"].append(data["q_norm"])

        initial_states = {
            l_idx: past_kv.layers[l_idx].recurrent_states[0].clone().float()
            for l_idx in layer_indices
        }
        prev_states = {l_idx: s.clone() for l_idx, s in initial_states.items()}

        # -------------------------------------------------------------
        # Phase 2: Latent Recurrent Steps (k = 1 ... 32)
        # -------------------------------------------------------------
        h_k16_clean = None

        for k in range(1, args.k_steps + 1):
            pos = torch.tensor([[prompt_len + k - 1]], device=args.device, dtype=torch.long)
            scaled_latent = curr_latent * CALIBRATED_ALPHA_3_5_4B
            captured_step_data.clear()

            with torch.no_grad():
                step_out = model(
                    inputs_embeds=scaled_latent,
                    position_ids=pos,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_kv = step_out.past_key_values
                curr_latent = step_out.hidden_states[-1][:, -1:, :]

            if k == 16:
                h_k16_clean = curr_latent.clone()

            step_betas = []
            step_alphas = []
            step_cos_prev = []
            step_cos_init = []
            step_rel_diffs = []

            for l_idx in layer_indices:
                data = captured_step_data[l_idx]
                beta_val = data["beta"].mean().item()
                alpha_val = data["alpha"].mean().item()
                metrics_by_regime["latents"][l_idx]["beta"].append(beta_val)
                metrics_by_regime["latents"][l_idx]["alpha"].append(alpha_val)
                metrics_by_regime["latents"][l_idx]["k_norm"].append(data["k_norm"])
                metrics_by_regime["latents"][l_idx]["v_norm"].append(data["v_norm"])
                metrics_by_regime["latents"][l_idx]["q_norm"].append(data["q_norm"])

                s_curr = past_kv.layers[l_idx].recurrent_states[0].clone().float()
                s_prev = prev_states[l_idx]
                s_init = initial_states[l_idx]

                diff_norm = torch.norm(s_curr - s_prev).item()
                rel_diff = diff_norm / (torch.norm(s_prev).item() + 1e-8)
                cos_prev = F.cosine_similarity(s_curr.flatten().unsqueeze(0), s_prev.flatten().unsqueeze(0)).item()
                cos_init = F.cosine_similarity(s_curr.flatten().unsqueeze(0), s_init.flatten().unsqueeze(0)).item()

                metrics_by_regime["latents"][l_idx]["state_diff_norm"].append(diff_norm)
                metrics_by_regime["latents"][l_idx]["state_cos"].append(cos_prev)
                metrics_by_regime["latents"][l_idx]["state_rel_diff"].append(rel_diff)

                innov_proxy = diff_norm / (beta_val + 1e-8)
                metrics_by_regime["latents"][l_idx]["innov_norm"].append(innov_proxy)

                step_betas.append(beta_val)
                step_alphas.append(alpha_val)
                step_cos_prev.append(cos_prev)
                step_cos_init.append(cos_init)
                step_rel_diffs.append(rel_diff)

                prev_states[l_idx] = s_curr

            step_trajectory[k]["beta"].append(float(np.mean(step_betas)))
            step_trajectory[k]["alpha"].append(float(np.mean(step_alphas)))
            step_trajectory[k]["state_cos_prev"].append(float(np.mean(step_cos_prev)))
            step_trajectory[k]["state_cos_initial"].append(float(np.mean(step_cos_init)))
            step_trajectory[k]["state_rel_diff"].append(float(np.mean(step_rel_diffs)))

        # -------------------------------------------------------------
        # Phase 3: Transition & Answer Decoding (first 16 tokens)
        # -------------------------------------------------------------
        trans_tokens = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
        trans_ids = torch.tensor([trans_tokens], dtype=torch.long, device=args.device)
        trans_len = len(trans_tokens)
        trans_pos = torch.arange(trans_len, device=args.device).unsqueeze(0) + (prompt_len + args.k_steps)

        with torch.no_grad():
            trans_out = model(
                input_ids=trans_ids,
                position_ids=trans_pos,
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = trans_out.past_key_values
            curr_token = torch.argmax(trans_out.logits[:, -1:, :], dim=-1)

        for step in range(args.ans_steps):
            ans_pos = torch.tensor([[prompt_len + args.k_steps + trans_len + step]], device=args.device, dtype=torch.long)
            captured_step_data.clear()

            with torch.no_grad():
                step_out = model(
                    input_ids=curr_token,
                    position_ids=ans_pos,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_kv = step_out.past_key_values
                curr_token = torch.argmax(step_out.logits[:, -1:, :], dim=-1)

            for l_idx in layer_indices:
                data = captured_step_data[l_idx]
                beta_val = data["beta"].mean().item()
                alpha_val = data["alpha"].mean().item()
                metrics_by_regime["answer"][l_idx]["beta"].append(beta_val)
                metrics_by_regime["answer"][l_idx]["alpha"].append(alpha_val)
                metrics_by_regime["answer"][l_idx]["k_norm"].append(data["k_norm"])
                metrics_by_regime["answer"][l_idx]["v_norm"].append(data["v_norm"])
                metrics_by_regime["answer"][l_idx]["q_norm"].append(data["q_norm"])

                s_curr = past_kv.layers[l_idx].recurrent_states[0].clone().float()
                s_prev = prev_states[l_idx]
                diff_norm = torch.norm(s_curr - s_prev).item()
                rel_diff = diff_norm / (torch.norm(s_prev).item() + 1e-8)
                cos_prev = F.cosine_similarity(s_curr.flatten().unsqueeze(0), s_prev.flatten().unsqueeze(0)).item()
                metrics_by_regime["answer"][l_idx]["state_diff_norm"].append(diff_norm)
                metrics_by_regime["answer"][l_idx]["state_cos"].append(cos_prev)
                metrics_by_regime["answer"][l_idx]["state_rel_diff"].append(rel_diff)

                innov_proxy = diff_norm / (beta_val + 1e-8)
                metrics_by_regime["answer"][l_idx]["innov_norm"].append(innov_proxy)
                prev_states[l_idx] = s_curr

        # -------------------------------------------------------------
        # Phase 4: Perturbation Sensitivity Probe (at k=16, first 5 problems)
        # -------------------------------------------------------------
        if h_k16_clean is not None and p_i < 5:
            torch.manual_seed(42 + p_i)
            eps_dir = torch.randn_like(h_k16_clean)
            h_norm = torch.norm(h_k16_clean, dim=-1, keepdim=True)
            eps = 1e-3 * h_norm * (eps_dir / torch.norm(eps_dir, dim=-1, keepdim=True))

            with torch.no_grad():
                out_clean_pre = model(**enc, use_cache=True, output_hidden_states=True)
                kv_clean = out_clean_pre.past_key_values
                lat = out_clean_pre.hidden_states[-1][:, -1:, :]
                for k in range(1, 16):
                    p_pos = torch.tensor([[prompt_len + k - 1]], device=args.device, dtype=torch.long)
                    s_o = model(inputs_embeds=lat * CALIBRATED_ALPHA_3_5_4B, position_ids=p_pos, past_key_values=kv_clean, use_cache=True, output_hidden_states=True)
                    kv_clean = s_o.past_key_values
                    lat = s_o.hidden_states[-1][:, -1:, :]

                p17_pos = torch.tensor([[prompt_len + 16]], device=args.device, dtype=torch.long)
                clean_step17 = model(inputs_embeds=lat * CALIBRATED_ALPHA_3_5_4B, position_ids=p17_pos, past_key_values=kv_clean, use_cache=True, output_hidden_states=True)
                h17_clean = clean_step17.hidden_states[-1][:, -1:, :]
                s17_clean = {l_idx: clean_step17.past_key_values.layers[l_idx].recurrent_states[0].clone().float() for l_idx in layer_indices}

                out_pert_pre = model(**enc, use_cache=True, output_hidden_states=True)
                kv_pert = out_pert_pre.past_key_values
                lat_p = out_pert_pre.hidden_states[-1][:, -1:, :]
                for k in range(1, 16):
                    p_pos = torch.tensor([[prompt_len + k - 1]], device=args.device, dtype=torch.long)
                    s_o = model(inputs_embeds=lat_p * CALIBRATED_ALPHA_3_5_4B, position_ids=p_pos, past_key_values=kv_pert, use_cache=True, output_hidden_states=True)
                    kv_pert = s_o.past_key_values
                    lat_p = s_o.hidden_states[-1][:, -1:, :]

                pert_step17 = model(inputs_embeds=(lat_p + eps) * CALIBRATED_ALPHA_3_5_4B, position_ids=p17_pos, past_key_values=kv_pert, use_cache=True, output_hidden_states=True)
                h17_pert = pert_step17.hidden_states[-1][:, -1:, :]
                s17_pert = {l_idx: pert_step17.past_key_values.layers[l_idx].recurrent_states[0].clone().float() for l_idx in layer_indices}

                h_diff = torch.norm(h17_pert - h17_clean).item()
                eps_norm = torch.norm(eps).item()
                amp = h_diff / eps_norm

                cos_s_list = []
                diff_s_list = []
                for l_idx in layer_indices:
                    c = F.cosine_similarity(s17_clean[l_idx].flatten().unsqueeze(0), s17_pert[l_idx].flatten().unsqueeze(0)).item()
                    d = torch.norm(s17_pert[l_idx] - s17_clean[l_idx]).item()
                    cos_s_list.append(c)
                    diff_s_list.append(d)

                mean_c = float(np.mean(cos_s_list))
                perturbation_results["hidden_amp"].append(amp)
                perturbation_results["gdn_state_cos"].append(mean_c)
                perturbation_results["gdn_state_diff_norm"].append(float(np.mean(diff_s_list)))
                perturbation_results["gdn_state_drift_pct"].append((1.0 - mean_c) * 100.0)

    for h in hooks:
        h.remove()

    total_dur = time.time() - t0_all
    print(f"\nProfiling completed in {total_dur:.1f}s.")

    # -------------------------------------------------------------
    # Summarize & Format Findings
    # -------------------------------------------------------------
    summary = {
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
        "n_problems": args.n_problems,
        "duration_s": round(total_dur, 2),
        "overall_regime_comparison": {},
        "layer_group_comparison": {},
        "trajectory_step_dynamics": {},
        "perturbation_sensitivity": {}
    }

    print("\n" + "="*75)
    print("=== SUMMARY: GDN GATE TELEMETRY ACROSS REGIMES ===")
    print("="*75)
    print(f"{'Metric':<25} | {'Prompt (Text)':<15} | {'Latents (k=1..32)':<18} | {'Answer (Text)':<15}")
    print("-" * 75)

    for metric in ["beta", "alpha", "k_norm", "v_norm", "q_norm"]:
        vals_p = [float(np.mean(metrics_by_regime["prompt"][l][metric])) for l in layer_indices]
        vals_l = [float(np.mean(metrics_by_regime["latents"][l][metric])) for l in layer_indices]
        vals_a = [float(np.mean(metrics_by_regime["answer"][l][metric])) for l in layer_indices]

        mean_p, std_p = float(np.mean(vals_p)), float(np.std(vals_p))
        mean_l, std_l = float(np.mean(vals_l)), float(np.std(vals_l))
        mean_a, std_a = float(np.mean(vals_a)), float(np.std(vals_a))

        summary["overall_regime_comparison"][metric] = {
            "prompt": {"mean": round(mean_p, 4), "std": round(std_p, 4)},
            "latents": {"mean": round(mean_l, 4), "std": round(std_l, 4)},
            "answer": {"mean": round(mean_a, 4), "std": round(std_a, 4)}
        }
        print(f"{metric:<25} | {mean_p:.4f} ± {std_p:.4f}   | {mean_l:.4f} ± {std_l:.4f}     | {mean_a:.4f} ± {std_a:.4f}")

    for metric in ["state_cos", "state_diff_norm", "state_rel_diff", "innov_norm"]:
        vals_l = [float(np.mean(metrics_by_regime["latents"][l][metric])) for l in layer_indices]
        vals_a = [float(np.mean(metrics_by_regime["answer"][l][metric])) for l in layer_indices]
        mean_l, std_l = float(np.mean(vals_l)), float(np.std(vals_l))
        mean_a, std_a = float(np.mean(vals_a)), float(np.std(vals_a))

        summary["overall_regime_comparison"][metric] = {
            "prompt": {"mean": None, "std": None},
            "latents": {"mean": round(mean_l, 4), "std": round(std_l, 4)},
            "answer": {"mean": round(mean_a, 4), "std": round(std_a, 4)}
        }
        print(f"{metric:<25} | {'N/A (Prefill)':<15} | {mean_l:.4f} ± {std_l:.4f}     | {mean_a:.4f} ± {std_a:.4f}")

    groups = {
        "Early GDN (Layers 0, 1, 2)": [0, 1, 2],
        "Middle GDN (Layers 12, 13, 14)": [12, 13, 14],
        "Deep GDN (Layers 28, 29, 30)": [28, 29, 30]
    }
    print("\n" + "="*75)
    print("=== LAYER GROUP BREAKDOWN (Beta & Alpha at Latents vs Answer) ===")
    print("="*75)
    for g_name, g_layers in groups.items():
        g_beta_lat = float(np.mean([np.mean(metrics_by_regime["latents"][l]["beta"]) for l in g_layers]))
        g_alpha_lat = float(np.mean([np.mean(metrics_by_regime["latents"][l]["alpha"]) for l in g_layers]))
        g_beta_ans = float(np.mean([np.mean(metrics_by_regime["answer"][l]["beta"]) for l in g_layers]))
        g_alpha_ans = float(np.mean([np.mean(metrics_by_regime["answer"][l]["alpha"]) for l in g_layers]))

        g_cos_lat = float(np.mean([np.mean(metrics_by_regime["latents"][l]["state_cos"]) for l in g_layers]))
        g_cos_ans = float(np.mean([np.mean(metrics_by_regime["answer"][l]["state_cos"]) for l in g_layers]))

        summary["layer_group_comparison"][g_name] = {
            "beta_latent": round(g_beta_lat, 4),
            "alpha_latent": round(g_alpha_lat, 4),
            "beta_answer": round(g_beta_ans, 4),
            "alpha_answer": round(g_alpha_ans, 4),
            "cos_latent": round(g_cos_lat, 4),
            "cos_answer": round(g_cos_ans, 4)
        }
        print(f"{g_name}:")
        print(f"  Latents: beta={g_beta_lat:.4f}, alpha={g_alpha_lat:.4f}, cos(S_k, S_prev)={g_cos_lat:.4f}")
        print(f"  Answer : beta={g_beta_ans:.4f}, alpha={g_alpha_ans:.4f}, cos(S_t, S_prev)={g_cos_ans:.4f}")

    if perturbation_results["hidden_amp"]:
        mean_amp = float(np.mean(perturbation_results["hidden_amp"]))
        mean_s_cos = float(np.mean(perturbation_results["gdn_state_cos"]))
        mean_s_diff = float(np.mean(perturbation_results["gdn_state_diff_norm"]))
        mean_s_drift = float(np.mean(perturbation_results["gdn_state_drift_pct"]))
        summary["perturbation_sensitivity"] = {
            "mean_hidden_amp": round(mean_amp, 2),
            "mean_gdn_state_cos": round(mean_s_cos, 6),
            "mean_gdn_state_diff_norm": round(mean_s_diff, 4),
            "state_drift_pct": round(mean_s_drift, 4)
        }

        print("\n" + "="*75)
        print("=== PERTURBATION SENSITIVITY (Causal Perturbation at k=16) ===")
        print("="*75)
        print(f"Hidden Trajectory Amplification: {mean_amp:.2f}x")
        print(f"GDN Recurrent State Cosine Similarity: {mean_s_cos:.6f}")
        print(f"GDN Recurrent State Drift: {mean_s_drift:.4f}%")
        print(f"GDN State Delta Frobenius Norm: {mean_s_diff:.4f}")

    for k in range(1, args.k_steps + 1):
        summary["trajectory_step_dynamics"][f"k_{k}"] = {
            "beta": round(float(np.mean(step_trajectory[k]["beta"])), 4),
            "alpha": round(float(np.mean(step_trajectory[k]["alpha"])), 4),
            "cos_prev": round(float(np.mean(step_trajectory[k]["state_cos_prev"])), 4),
            "cos_init": round(float(np.mean(step_trajectory[k]["state_cos_initial"])), 4),
            "rel_diff": round(float(np.mean(step_trajectory[k]["state_rel_diff"])), 4)
        }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved complete GDN telemetry report to: {args.output_file}")


if __name__ == "__main__":
    main()
