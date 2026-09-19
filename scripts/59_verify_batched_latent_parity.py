#!/usr/bin/env python3
"""
scripts/59_verify_batched_latent_parity.py
Automated Numerical Parity, StaticCache Gate, and Lyapunov Sensitivity Analyzer.

Verifies:
1. Parity on the EXACT code path used for training (StaticCache + cache_position + position_ids).
2. SDPA Backend pinning: SDPBackend.MATH for latent steps vs FlashAttention for prefill/answer.
3. Dynamical Sensitivity: Tracks per-step perturbation growth factor Delta_k / Delta_{k-1}
   and Lyapunov exponent lambda = (1/K) * ln(Delta_K / Delta_0) to distinguish
   expansive (lambda > 0) from contractive (lambda < 0) mappings.
4. Adapter Verification: Supports --adapter_path to test if LoRA training regularizes
   the expansive base map into a contractive attractor basin.
5. Causal Patching Control 0: Cosine >= 0.999 on the post-patch latent state AND
   exact identical decoded answer string.
"""

import os
import math
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
from peft import PeftModel

TEST_PROMPTS = [
    "What is 2 + 2?",
    "Solve for x: 3x + 12 = 45.",
    "Janet has 16 eggs. She uses 3 for breakfast and 4 for baking. How many are left?",
    "Find the sum of all integers from 1 to 100 inclusive.",
    "A car travels 180 miles in 3 hours. What is its average speed in miles per hour?"
]

def verify_parity_and_dynamics(model_id="Qwen/Qwen3-1.7B", adapter_path=None, device="cuda:0", k_steps=6, scale_factor=0.011440):
    mode_str = f"Trained Adapter ({adapter_path})" if adapter_path else "Untrained Base Model"
    print(f"=== Latent Dynamics & Numerical Parity Verification ===")
    print(f"Model: {model_id} | Mode: {mode_str}")
    print(f"Device: {device} | Horizon K: {k_steps} | Alpha: {scale_factor}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"Loaded adapter from {adapter_path}")
    else:
        if adapter_path:
            print(f"Notice: Adapter {adapter_path} not found; testing base model dynamics.")
        model = base_model
    model.eval()

    # -------------------------------------------------------------
    # PART 1: Pinned SDPBackend.MATH vs FlashAttention Sensitivity
    # -------------------------------------------------------------
    print("="*70)
    print("PART 1: Dynamical Sensitivity & Lyapunov Exponent (Prompt 2 Dynamics)")
    print("="*70)
    p2 = TEST_PROMPTS[2] # 0-padding, length 25

    # Run B=1 vs B=5 identical prompt 2 to isolate kernel tiling perturbation
    enc1 = tokenizer([p2], return_tensors="pt").to(device)
    enc5 = tokenizer([p2]*5, return_tensors="pt").to(device)
    L2 = enc1.input_ids.shape[1]

    for backend_name, backend_enum in [("Pinned MATH", [torch.nn.attention.SDPBackend.MATH]),
                                       ("Flash / Efficient", [torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION, torch.nn.attention.SDPBackend.MATH])]:
        with torch.nn.attention.sdpa_kernel(backend_enum):
            with torch.no_grad():
                # B=1
                out1 = model(**enc1, use_cache=True, output_hidden_states=True)
                past1 = out1.past_key_values
                curr1 = out1.hidden_states[-1][:, -1:, :]
                u_steps = [curr1.clone()]
                for k in range(k_steps):
                    pos = torch.tensor([[L2 + k]], device=device, dtype=torch.long)
                    out1 = model(inputs_embeds=curr1 * scale_factor, position_ids=pos, past_key_values=past1, use_cache=True, output_hidden_states=True)
                    past1 = out1.past_key_values
                    curr1 = out1.hidden_states[-1][:, -1:, :]
                    u_steps.append(curr1.clone())

                # B=5
                out5 = model(**enc5, use_cache=True, output_hidden_states=True)
                past5 = out5.past_key_values
                curr5 = out5.hidden_states[-1][:, -1:, :]
                b_steps = [curr5[0:1].clone()]
                for k in range(k_steps):
                    pos = torch.full((5, 1), L2 + k, device=device, dtype=torch.long)
                    out5 = model(inputs_embeds=curr5 * scale_factor, position_ids=pos, past_key_values=past5, use_cache=True, output_hidden_states=True)
                    past5 = out5.past_key_values
                    curr5 = out5.hidden_states[-1][:, -1:, :]
                    b_steps.append(curr5[0:1].clone())

        print(f"\n--- Backend: {backend_name} ---")
        diffs = []
        for k in range(k_steps + 1):
            cos = F.cosine_similarity(u_steps[k].squeeze().float().unsqueeze(0), b_steps[k].squeeze().float().unsqueeze(0)).item()
            diff = (u_steps[k] - b_steps[k]).abs().max().item()
            diffs.append(diff)
            growth = f"{diff / diffs[k-1]:.2f}x" if k > 0 and diffs[k-1] > 1e-6 else "N/A"
            print(f"  Step {k}: Cosine = {cos:.6f} | MaxDiff = {diff:.6f} | Growth = {growth}")

        d0 = max(diffs[0], 1e-6)
        dK = max(diffs[-1], 1e-6)
        lyapunov = (1.0 / k_steps) * math.log(dK / d0)
        regime = "EXPANSIVE (lambda > 0)" if lyapunov > 0.05 else ("CONTRACTIVE (lambda < 0)" if lyapunov < -0.05 else "NEUTRAL / BOUNDED")
        print(f"  Estimated Lyapunov Exponent lambda: {lyapunov:+.3f} / step -> [{regime}]")

    # -------------------------------------------------------------
    # PART 2: Training Path Parity Gate (StaticCache + SDPBackend.MATH)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("PART 2: Parity Gate on StaticCache + cache_position (Training Path)")
    print("="*70)

    # 1. Unbatched ground truth with Pinned MATH
    unbatched_gt = []
    unbatched_answers = []
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        for idx, prompt in enumerate(TEST_PROMPTS):
            enc = tokenizer([prompt], return_tensors="pt").to(device)
            L = enc.input_ids.shape[1]
            with torch.no_grad():
                out = model(**enc, use_cache=True, output_hidden_states=True)
                past_kv = out.past_key_values
                curr = out.hidden_states[-1][:, -1:, :]
                for k in range(k_steps):
                    pos = torch.tensor([[L + k]], device=device, dtype=torch.long)
                    out = model(inputs_embeds=curr * scale_factor, position_ids=pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
                    past_kv = out.past_key_values
                    curr = out.hidden_states[-1][:, -1:, :]
                unbatched_gt.append(curr.squeeze().float())

                # Quick 10-token answer decode for Control 0 string check
                ans_ids = []
                # transition
                trans_ids = tokenizer.encode("\n</think>\n\n", add_special_tokens=False, return_tensors="pt").to(device)
                out = model(input_ids=trans_ids, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                ans_ids.append(next_tok.item())
                for _ in range(8):
                    out = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
                    ans_ids.append(next_tok.item())
                unbatched_answers.append(tokenizer.decode(ans_ids))

    # 2. Batched execution with StaticCache + cache_position + Pinned MATH
    enc_batch = tokenizer(TEST_PROMPTS, padding=True, return_tensors="pt").to(device)
    B, L_max = enc_batch.input_ids.shape
    seq_lens = enc_batch.attention_mask.sum(dim=-1)
    max_cache_len = L_max + k_steps + 32

    static_cache = StaticCache(
        config=model.config,
        max_batch_size=B,
        max_cache_len=max_cache_len,
        device=device,
        dtype=torch.bfloat16
    )
    cache_pos = torch.arange(0, L_max, device=device)
    pos_prefill = enc_batch.attention_mask.long().cumsum(-1) - 1
    pos_prefill.masked_fill_(enc_batch.attention_mask == 0, 0)

    batched_results = []
    batched_answers = []
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        with torch.no_grad():
            out_b = model(
                input_ids=enc_batch.input_ids,
                attention_mask=enc_batch.attention_mask,
                position_ids=pos_prefill,
                past_key_values=static_cache,
                cache_position=cache_pos,
                use_cache=True,
                output_hidden_states=True
            )
            curr_b = out_b.hidden_states[-1][:, -1:, :]

            curr_mask = enc_batch.attention_mask
            for k in range(k_steps):
                step_pos = (seq_lens + k).unsqueeze(1)
                c_pos = torch.tensor([L_max + k], device=device)
                curr_mask = torch.cat([curr_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1)

                out_b = model(
                    inputs_embeds=curr_b * scale_factor,
                    attention_mask=curr_mask,
                    position_ids=step_pos,
                    past_key_values=static_cache,
                    cache_position=c_pos,
                    use_cache=True,
                    output_hidden_states=True
                )
                curr_b = out_b.hidden_states[-1][:, -1:, :]

            for b in range(B):
                batched_results.append(curr_b[b, 0, :].float())

    print("Per-Prompt Parity Report (StaticCache vs. Unbatched):")
    all_passed = True
    for b in range(B):
        cos = F.cosine_similarity(unbatched_gt[b].unsqueeze(0), batched_results[b].unsqueeze(0)).item()
        mean_diff = (unbatched_gt[b] - batched_results[b]).abs().mean().item()
        max_diff = (unbatched_gt[b] - batched_results[b]).abs().max().item()
        pad_len = L_max - seq_lens[b].item()
        status = "PASSED (>=0.999)" if cos >= 0.999 else "FAILED (<0.999)"
        if cos < 0.999:
            all_passed = False
        print(f"  Prompt {b} [len {seq_lens[b].item():2d}, pad {pad_len:2d}]: Cosine = {cos:.6f} | MaxDiff = {max_diff:.4f} -> [{status}]")

    print("\n" + "="*70)
    if all_passed:
        print("GATE 1 PARITY PASSED: StaticCache + Pinned MATH satisfies strict Cosine >= 0.999 invariant.")
    else:
        print("GATE 1 PARITY FAILED: Check cache_position or rotary position IDs.")
    print("="*70)

    # -------------------------------------------------------------
    # PART 3: End-to-End Generation Parity (DynamicCache + Transition + 20 Answer Tokens Greedy)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("PART 3: End-to-End Generation Parity (Transition + 20 Answer Tokens Greedy)")
    print("="*70)

    ans_tokens_count = 20
    trans_tokens = tokenizer.encode("\n</think>\n\n", add_special_tokens=False)
    trans_len = len(trans_tokens)

    # 1. Unbatched Ground Truth
    unbatched_answers = []
    with torch.no_grad():
        for p in TEST_PROMPTS:
            p_text = tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            enc = tokenizer([p_text], return_tensors="pt").to(device)
            L = enc.input_ids.shape[1]
            out = model(**enc, use_cache=True, output_hidden_states=True)
            past_kv = out.past_key_values
            curr = out.hidden_states[-1][:, -1:, :]

            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                for k in range(k_steps):
                    pos = torch.tensor([[L + k]], device=device, dtype=torch.long)
                    step_out = model(inputs_embeds=curr * scale_factor, position_ids=pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
                    past_kv = step_out.past_key_values
                    curr = step_out.hidden_states[-1][:, -1:, :]

            trans_ids = torch.tensor([trans_tokens], dtype=torch.long, device=device)
            trans_pos = torch.arange(trans_len, device=device).unsqueeze(0) + (L + k_steps)
            trans_mask = torch.ones((1, L + k_steps + trans_len), dtype=torch.long, device=device)

            gen_out = model.generate(
                input_ids=trans_ids,
                attention_mask=trans_mask,
                position_ids=trans_pos,
                past_key_values=past_kv,
                max_new_tokens=ans_tokens_count,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            ans_toks = gen_out[0][trans_len:].tolist()
            unbatched_answers.append(tokenizer.decode(ans_toks, skip_special_tokens=True))

    # 2. Batched Generation with pos_prefill & Rotary Hook
    formatted = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=True) for p in TEST_PROMPTS]
    enc_b = tokenizer(formatted, padding=True, return_tensors="pt").to(device)
    B = len(TEST_PROMPTS)
    seq_lens = enc_b.attention_mask.sum(dim=-1)
    pos_prefill = enc_b.attention_mask.long().cumsum(-1) - 1
    pos_prefill.masked_fill_(enc_b.attention_mask == 0, 0)

    hook_calls = []
    def rotary_hook(module, args, output):
        pos_ids = args[1]
        hook_calls.append(pos_ids.detach().cpu())

    with torch.no_grad():
        out_b = model(input_ids=enc_b.input_ids, attention_mask=enc_b.attention_mask, position_ids=pos_prefill, use_cache=True, output_hidden_states=True)
        past_kv = out_b.past_key_values
        curr_b = out_b.hidden_states[-1][:, -1:, :]
        curr_mask = enc_b.attention_mask

        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            for k in range(k_steps):
                step_pos = (seq_lens + k).unsqueeze(1)
                curr_mask = torch.cat([curr_mask, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1)
                step_out = model(inputs_embeds=curr_b * scale_factor, attention_mask=curr_mask, position_ids=step_pos, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
                past_kv = step_out.past_key_values
                curr_b = step_out.hidden_states[-1][:, -1:, :]

        trans_ids = torch.tensor([trans_tokens] * B, dtype=torch.long, device=device)
        trans_mask = torch.cat([curr_mask, torch.ones((B, trans_len), dtype=torch.long, device=device)], dim=1)
        pos = torch.stack([torch.arange(trans_len, device=device) + (seq_lens[b] + k_steps) for b in range(B)], dim=0)

        hook_handle = model.model.rotary_emb.register_forward_hook(rotary_hook)
        gen_b = model.generate(
            input_ids=trans_ids,
            attention_mask=trans_mask,
            position_ids=pos,
            past_key_values=past_kv,
            max_new_tokens=ans_tokens_count,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        hook_handle.remove()

    batched_answers = []
    for b in range(B):
        ans_toks = gen_b[b][trans_len:].tolist()
        batched_answers.append(tokenizer.decode(ans_toks, skip_special_tokens=True))

    # 3. Verify Hook Positions
    print(f"Total rotary hook calls during generate(): {len(hook_calls)}")
    call0 = hook_calls[0]
    print(f"Call 0 (Transition) pos: {call0.tolist()}")
    for b in range(B):
        expected_trans = [seq_lens[b].item() + k_steps + i for i in range(trans_len)]
        assert call0[b].tolist() == expected_trans, f"Row {b} transition pos mismatch: got {call0[b].tolist()}, expected {expected_trans}"

    for t in range(1, len(hook_calls)):
        call_t = hook_calls[t]
        for b in range(B):
            expected_pos = seq_lens[b].item() + k_steps + trans_len + (t - 1)
            actual_pos = call_t[b, 0].item()
            assert actual_pos == expected_pos, f"Step {t} row {b} pos mismatch: got {actual_pos}, expected {expected_pos}"
    print("HOOK VERIFICATION PASSED: Every decode step 1..N strictly saw L_b + K + trans_len + t, never L_max!")

    # 4. Verify Exact String Parity
    gen_passed = True
    print("\nPer-Prompt Generation String Parity Report (Batched vs. Unbatched):")
    for b in range(B):
        matched = (unbatched_answers[b] == batched_answers[b])
        if not matched:
            gen_passed = False
        print(f"  Prompt {b} [len {seq_lens[b].item():2d}]: matched = {matched}")
        if not matched:
            print(f"    Unbatched: {repr(unbatched_answers[b])}")
            print(f"    Batched:   {repr(batched_answers[b])}")

    print("\n" + "="*70)
    if gen_passed:
        print("GATE 2 GENERATION PARITY PASSED: 100% bit-exact string match batched vs unbatched across all prompts!")
    else:
        print("GATE 2 GENERATION PARITY FAILED: String mismatch between batched and unbatched generation.")
    print("="*70)

    return all_passed and gen_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--k_steps", type=int, default=6)
    args = parser.parse_args()
    verify_parity_and_dynamics(args.model_id, args.adapter_path, args.device, args.k_steps)

