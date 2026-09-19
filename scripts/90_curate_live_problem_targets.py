#!/usr/bin/env python3
"""
scripts/90_curate_live_problem_targets.py
Component 1: Live-Problem Targets for Continuous Latent Recurrence v1.1.

1. Classifies training problems from data/curated_train_traces_qwen_qwen3-1.7b.jsonl:
   - Solved-Direct: Non-thinking correct (from scripts/84 curated math outputs, N=1,464).
   - Live Candidate: Thinking correct (in v1.0 traces), non-thinking failed (N=606).
   Logs IDs to data/train_problem_classification_qwen3_1.7b.json.

2. Generates self-rewritten worked solutions for live problems:
   - Prompts base Qwen/Qwen3-1.7B with its own thinking trace in user prompt.
   - Generates in non-thinking mode (temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5).
   - Verifies with canonical math_verify against ground_truth.
   - Enforces regex operation checks (intermediate numeric values accompanied by operations,
     substantive length >= 40 words, no bare assertions).

3. Samples and Generates 500 Clean UltraChat General Instruction Examples:
   - 13-gram and 50-char substring zero-contamination assertion against google/ifeval.
   - Fixed end-token truncation detection (<|im_end|> or eos_token_id).

4. Stratified Mixture Assembly:
   - Stratifies by MATH level (Levels 1–5 + GSM8K).
   - Constructs target mix of ~35% live problems and ~65% solved-direct problems.
   - Appends 500 clean UltraChat general instruction examples.
   - Saves final unified training set to data/curated_train_v1_1_final_qwen3_1.7b.jsonl.
   - Saves dev set to data/curated_dev_v1_1_final_qwen3_1.7b.jsonl.
"""

import os
import re
import json
import argparse
import random
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from math_verify import parse, verify

def get_ngrams(text, n):
    words = re.findall(r"\w+", text.lower())
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))

def get_char_substrings(text, length=50):
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    return set(cleaned[i:i+length] for i in range(len(cleaned) - length + 1))

def check_ifeval_contamination(prompt, ifeval_13grams, ifeval_substrings):
    p_13grams = get_ngrams(prompt, 13)
    if p_13grams.intersection(ifeval_13grams):
        return True
    p_substrings = get_char_substrings(prompt, 50)
    if p_substrings.intersection(ifeval_substrings):
        return True
    return False

def check_math_correct(pred_text, ground_truth):
    if not ground_truth:
        return False
    try:
        c_gold = parse(ground_truth)
        c_pred = parse(pred_text)
        return bool(verify(c_gold, c_pred))
    except Exception:
        return False

def parse_level(lvl):
    if isinstance(lvl, int):
        return lvl
    if isinstance(lvl, str):
        nums = re.findall(r'\d+', lvl)
        if nums:
            return int(nums[0])
    return 1

def check_operations_in_derivation(text):
    words = text.split()
    if len(words) < 40:
        return False, "Too short (< 40 words)"
        
    op_pattern = re.compile(r'(=|\+|-|\\times|\\cdot|\*|/|\\div|\\approx|\\equiv|\\le|\\ge|\^|\\sqrt)')
    op_matches = op_pattern.findall(text)
    if len(op_matches) < 3:
        return False, f"Too few mathematical operations ({len(op_matches)} < 3)"
        
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    lines_with_numbers = [l for l in lines if re.search(r'\d', l)]
    
    op_lines = [l for l in lines_with_numbers if op_pattern.search(l)]
    if len(lines_with_numbers) > 0 and (len(op_lines) / len(lines_with_numbers)) < 0.4:
        return False, f"Low operator density in numeric lines ({len(op_lines)}/{len(lines_with_numbers)})"
        
    assertion_only = re.search(r'^(Therefore, |So, |Hence, |The answer is |Thus, )?(\$?\\boxed\{.*\}\$?\.?)$', text.strip(), re.IGNORECASE)
    if assertion_only:
        return False, "Pure assertion without derivation"
        
    return True, "Passed"

def generate_batch(model, tokenizer, prompts, device, max_new_tokens=2048, batch_size=16, pbar_desc="Generating"):
    results = []
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    eos_id = tokenizer.eos_token_id

    pbar = tqdm(total=len(prompts), desc=pbar_desc)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        prompt_lens = [enc.attention_mask[b].sum().item() for b in range(enc.input_ids.shape[0])]
        
        class PPWrapper:
            def __init__(self, pp, p_lens):
                self.pp = pp
                self.p_lens = p_lens
            def __call__(self, input_ids, scores):
                for b in range(input_ids.shape[0]):
                    p_len = self.p_lens[b]
                    if input_ids.shape[1] > p_len:
                        gen_ids = input_ids[b, p_len:]
                        u_ids = torch.unique(gen_ids)
                        scores[b, u_ids] -= self.pp
                return scores

        pp_proc = PPWrapper(1.5, prompt_lens)
        
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.80,
                top_k=20,
                logits_processor=[pp_proc]
            )
            
        for b in range(len(batch_prompts)):
            p_len = prompt_lens[b]
            gen_ids = out[b, enc.input_ids.shape[1] - (enc.attention_mask.shape[1] - p_len):]
            gen_slice = out[b, enc.input_ids.shape[1]:]
            text = tokenizer.decode(gen_slice, skip_special_tokens=True).strip()
            
            # Robust end-of-sequence detection
            has_end = (gen_slice == im_end_id).any().item() or (gen_slice == eos_id).any().item()
            is_trunc = not has_end
            results.append({"text": text, "tokens": len(gen_ids), "is_truncated": is_trunc})
            
        del enc, out
        torch.cuda.empty_cache()
        pbar.update(len(batch_prompts))
    pbar.close()
    return results

def main():
    parser = argparse.ArgumentParser(description="Curate Live-Problem Targets for v1.1")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--target_live_ratio", type=float, default=0.35)
    parser.add_argument("--general_count", type=int, default=500)
    args = parser.parse_args()

    tag = args.model_id.replace("/", "_").lower()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    v1_train_path = os.path.join(data_dir, "curated_train_traces_qwen_qwen3-1.7b.jsonl")
    step84_train_path = os.path.join(data_dir, "curated_train_v1_1_qwen3_1.7b.jsonl")
    step84_dev_path = os.path.join(data_dir, "curated_dev_v1_1_qwen3_1.7b.jsonl")
    
    classification_out_path = os.path.join(data_dir, "train_problem_classification_qwen3_1.7b.json")
    final_train_path = os.path.join(data_dir, "curated_train_v1_1_final_qwen3_1.7b.jsonl")
    final_dev_path = os.path.join(data_dir, "curated_dev_v1_1_final_qwen3_1.7b.jsonl")
    manifest_path = os.path.join(data_dir, "curated_v1_1_final_manifest.json")

    print(f"=== Component 1: Live-Problem Target Curation for {args.model_id} ===")

    # 1. Load All 2,070 v1.0 Verified Math Traces
    with open(v1_train_path) as f:
        all_math_traces = [json.loads(line) for line in f]
    print(f"Loaded {len(all_math_traces)} verified thinking traces from train split.")

    # 2. Load Solved-Direct Problems from scripts/84 output
    if not os.path.exists(step84_train_path):
        raise FileNotFoundError(f"scripts/84 output not found at: {step84_train_path}.")
    
    with open(step84_train_path) as f:
        step84_records = [json.loads(line) for line in f]
    
    solved_direct_math = [r for r in step84_records if r.get("dataset") != "ultrachat"]
    solved_direct_ids = set(r["id"] for r in solved_direct_math)
    print(f"Loaded {len(solved_direct_math)} solved-direct math problems.")

    # 3. Classify: Solved-Direct vs Live Candidates
    live_candidates = [s for s in all_math_traces if s["id"] not in solved_direct_ids]
    live_candidate_ids = [s["id"] for s in live_candidates]
    
    print(f"\n--- Classification Results ---")
    print(f"Total Train Math Traces: {len(all_math_traces)}")
    print(f"Solved-Direct (non-thinking correct): {len(solved_direct_math)} ({len(solved_direct_math)/len(all_math_traces)*100:.1f}%)")
    print(f"Live Candidates (thinking correct, non-thinking wrong): {len(live_candidates)} ({len(live_candidates)/len(all_math_traces)*100:.1f}%)")

    # Save classification registry
    classification_data = {
        "model_id": args.model_id,
        "total_traces": len(all_math_traces),
        "solved_direct_count": len(solved_direct_math),
        "live_candidate_count": len(live_candidates),
        "solved_direct_ids": list(solved_direct_ids),
        "live_candidate_ids": live_candidate_ids
    }
    with open(classification_out_path, "w") as f:
        json.dump(classification_data, f, indent=2)
    print(f"Saved problem classification registry to: {classification_out_path}")

    # 4. Load Model and Tokenizer
    print(f"\n--- Loading {args.model_id} on {args.device} in pure bfloat16 ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    # 5. Generate Self-Rewritten Worked Solutions for Live Candidates
    print(f"\n--- Generating Self-Rewritten Worked Solutions for {len(live_candidates)} Live Problems ---")
    rewrite_prompts = []
    for s in live_candidates:
        p = (
            f"<|im_start|>user\n"
            f"{s['question']}\n\n"
            f"Here are scratchpad reasoning notes for this problem:\n"
            f"{s.get('think', '')}\n\n"
            f"Using the reasoning above, write out a complete, clean, step-by-step worked solution. "
            f"Show every calculation and mathematical operation explicitly. "
            f"End your solution with the final answer in \\boxed{{}}.<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<think>\n\n</think>\n\n"
        )
        rewrite_prompts.append(p)

    live_cache_file = os.path.join(data_dir, f"cache_live_rewrites_{tag}.jsonl")
    if os.path.exists(live_cache_file):
        print(f"Loading cached live rewrites from: {live_cache_file}")
        gen_results = [json.loads(line) for line in open(live_cache_file)]
    else:
        gen_results = generate_batch(model, tokenizer, rewrite_prompts, args.device, max_new_tokens=2048, batch_size=args.batch_size, pbar_desc="Rewriting Live Problems")
        with open(live_cache_file, "w") as f_cache:
            for r in gen_results:
                f_cache.write(json.dumps(r) + "\n")

    # Filter rewrites via math_verify and operation regex check
    verified_live = []
    filter_stats = defaultdict(int)

    for s, res in zip(live_candidates, gen_results):
        text = res["text"]
        is_trunc = res["is_truncated"]
        
        if is_trunc:
            filter_stats["truncated"] += 1
            continue
        if "\\boxed{" not in text:
            filter_stats["missing_boxed"] += 1
            continue
            
        is_correct = check_math_correct(text, s.get("ground_truth", ""))
        if not is_correct:
            filter_stats["incorrect_answer"] += 1
            continue
            
        passed_ops, reason = check_operations_in_derivation(text)
        if not passed_ops:
            filter_stats[f"op_fail: {reason}"] += 1
            continue

        filter_stats["passed"] += 1
        rec = {
            "id": s["id"],
            "dataset": s.get("dataset", "math"),
            "level": s.get("level", 1),
            "subject": s.get("subject", "math"),
            "question": s["question"],
            "ground_truth": s.get("ground_truth", ""),
            "prompt": f"<|im_start|>user\n{s['question']}<|im_end|>\n<|im_start|>assistant\n<think>\n",
            "nonthinking_answer": text + "<|im_end|>",
            "teacher_think": s.get("think", ""),
            "teacher_steps": s.get("steps", []),
            "num_teacher_steps": s.get("num_steps", len(s.get("steps", []))),
            "answer_tokens": res["tokens"],
            "is_live": True
        }
        verified_live.append(rec)

    print(f"\n--- Live Problem Filtering Statistics ---")
    for k, v in filter_stats.items():
        print(f"  {k}: {v}")
    print(f"Verified & Operation-Checked Live Problems: {len(verified_live)} / {len(live_candidates)} ({len(verified_live)/max(len(live_candidates),1)*100:.1f}%)")

    # 6. Sample and Generate N=500 Clean UltraChat General Instruction Prompts
    print(f"\n--- Step 6: Curating {args.general_count} General Instruction Examples from UltraChat ---")
    ifeval = load_dataset("google/ifeval", split="train")
    ifeval_13grams = set()
    ifeval_substrings = set()
    for row in ifeval:
        p = row["prompt"]
        ifeval_13grams.update(get_ngrams(p, 13))
        ifeval_substrings.update(get_char_substrings(p, 50))
    print(f"Indexed {len(ifeval_13grams)} IFEval 13-grams for contamination assertion.")

    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    general_prompts = []
    for sample in uc:
        if len(general_prompts) >= args.general_count:
            break
        msgs = sample.get("messages", [])
        if not msgs or msgs[0].get("role") != "user":
            continue
        p = msgs[0]["content"].strip()
        if len(p.split()) < 5 or len(p.split()) > 250:
            continue
        if check_ifeval_contamination(p, ifeval_13grams, ifeval_substrings):
            continue
        general_prompts.append({
            "id": f"ultrachat_{sample.get('prompt_id', len(general_prompts))}",
            "prompt": p,
            "dataset": "ultrachat"
        })
    print(f"Sampled {len(general_prompts)} clean UltraChat prompts (0% IFEval overlap verified).")

    gen_prompts_formatted = [
        f"<|im_start|>user\n{g['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        for g in general_prompts
    ]
    uc_cache_file = os.path.join(data_dir, f"cache_ultrachat_gen_{tag}.jsonl")
    if os.path.exists(uc_cache_file):
        print(f"Loading cached general instructions from: {uc_cache_file}")
        gen_results = [json.loads(line) for line in open(uc_cache_file)]
    else:
        gen_results = generate_batch(model, tokenizer, gen_prompts_formatted, args.device, max_new_tokens=1024, batch_size=args.batch_size, pbar_desc="Generating General Instructions")
        with open(uc_cache_file, "w") as f_cache:
            for r in gen_results:
                f_cache.write(json.dumps(r) + "\n")
    
    general_curated = []
    for g, res in zip(general_prompts, gen_results):
        if not res["is_truncated"] and len(res["text"].split()) >= 15:
            rec = {
                "id": g["id"],
                "dataset": "ultrachat",
                "level": 0,
                "subject": "general_instruction",
                "question": g["prompt"],
                "ground_truth": "",
                "prompt": f"<|im_start|>user\n{g['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n",
                "nonthinking_answer": res["text"] + "<|im_end|>",
                "teacher_think": "",
                "teacher_steps": [],
                "num_teacher_steps": 0,
                "answer_tokens": res["tokens"],
                "is_live": False,
                "is_general": True
            }
            general_curated.append(rec)
    print(f"General instruction responses generated & kept: {len(general_curated)} / {len(general_prompts)}")

    # 7. Stratified Mixture Assembly (~35% Live, ~65% Solved-Direct)
    print(f"\n--- Stratifying and Mixing Dataset (~{args.target_live_ratio*100:.0f}% Live) ---")
    for r in solved_direct_math:
        r["is_live"] = False
        
    n_live = len(verified_live)
    if n_live == 0:
        print("WARNING: No live problems passed verification. Using solved-direct only.")
        final_math = solved_direct_math
    else:
        n_direct_target = int(round(n_live * (1.0 - args.target_live_ratio) / args.target_live_ratio))
        n_direct_target = min(n_direct_target, len(solved_direct_math))
        
        direct_by_level = defaultdict(list)
        for r in solved_direct_math:
            lvl = parse_level(r.get("level", 1))
            r["level"] = lvl
            direct_by_level[lvl].append(r)
            
        live_by_level = defaultdict(list)
        for r in verified_live:
            lvl = parse_level(r.get("level", 1))
            r["level"] = lvl
            live_by_level[lvl].append(r)
            
        print("Level distribution of verified live problems:")
        for lvl in sorted(live_by_level.keys()):
            print(f"  Level {lvl}: {len(live_by_level[lvl])} live")

        sampled_direct = []
        random.seed(42)
        total_direct = len(solved_direct_math)
        for lvl in sorted(direct_by_level.keys()):
            items = direct_by_level[lvl]
            lvl_fraction = len(items) / total_direct
            lvl_target = int(round(n_direct_target * lvl_fraction))
            lvl_target = min(lvl_target, len(items))
            sampled_direct.extend(random.sample(items, lvl_target))
            
        print(f"Sampled {len(sampled_direct)} solved-direct problems (target was {n_direct_target}).")
        final_math = verified_live + sampled_direct

    actual_live_ratio = len(verified_live) / len(final_math) if final_math else 0
    print(f"Final Math Split: {len(final_math)} problems ({len(verified_live)} live [{actual_live_ratio*100:.1f}%], {len(final_math)-len(verified_live)} solved-direct)")

    full_train_dataset = final_math + general_curated
    random.seed(42)
    random.shuffle(full_train_dataset)

    # Dev split
    with open(step84_dev_path) as f:
        dev_dataset = [json.loads(line) for line in f]
    for d in dev_dataset:
        d["is_live"] = False

    # 8. Save Outputs
    print(f"\n--- Step 8: Saving Final Datasets ---")
    with open(final_train_path, "w") as f:
        for r in full_train_dataset:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(full_train_dataset)} training examples to: {final_train_path}")

    with open(final_dev_path, "w") as f:
        for r in dev_dataset:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(dev_dataset)} dev examples to: {final_dev_path}")

    manifest = {
        "model_id": args.model_id,
        "total_train_examples": len(full_train_dataset),
        "math_examples": len(final_math),
        "live_examples": len(verified_live),
        "solved_direct_examples": len(final_math) - len(verified_live),
        "actual_live_ratio": actual_live_ratio,
        "general_instruction_examples": len(general_curated),
        "dev_examples": len(dev_dataset),
        "filter_stats": filter_stats,
        "classification": {
            "total_original_traces": len(all_math_traces),
            "solved_direct_count": len(solved_direct_math),
            "live_candidate_count": len(live_candidates)
        }
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved final manifest to: {manifest_path}")

    print("\n=== Component 1 Live-Problem Curation Complete [SUCCESS] ===")

if __name__ == "__main__":
    main()
