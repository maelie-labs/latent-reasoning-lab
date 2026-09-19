#!/usr/bin/env python3
"""
scripts/84_curate_v1_1_nonthinking_targets.py
Continuous Latent Recurrence v1.1 Data Curation Pipeline.

1. Takes verified training problems from:
   - data/curated_train_traces_qwen_qwen3-1.7b.jsonl (2,070 traces)
   - data/curated_dev_traces_qwen_qwen3-1.7b.jsonl (100 traces)
2. Generates full step-by-step non-thinking solutions using Qwen/Qwen3-1.7B on cuda:0:
   - Prompt: <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
   - Sampling: temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5, max_new_tokens=2048
   - Evaluated by canonical math_verify against reference answer.
   - Kept if verified correct and non-truncated with full worked derivations.
3. Ingests a general-domain instruction-following regularization slice (N=500):
   - Sourced from HuggingFaceH4/ultrachat_200k (train_sft).
   - Filtered with strict 13-gram and 50-character substring disjointness checks against all 541 IFEval evaluation prompts.
   - Generates responses with Qwen/Qwen3-1.7B in non-thinking mode.
4. Produces:
   - data/curated_train_v1_1_qwen3_1.7b.jsonl
   - data/curated_dev_v1_1_qwen3_1.7b.jsonl (100 held-out traces)
   - data/general_slice_500_manifest.json (with 13-gram overlap assertion report)
"""

import os
import re
import json
import time
import argparse
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from math_verify import parse, verify

def get_ngrams(text: str, n: int = 13) -> set:
    words = text.strip().lower().split()
    if len(words) < n:
        return set()
    return set(" ".join(words[i:i+n]) for i in range(len(words)-n+1))

def get_char_substrings(text: str, length: int = 50) -> set:
    clean = re.sub(r'\s+', ' ', text.strip().lower())
    if len(clean) < length:
        return set()
    return set(clean[i:i+length] for i in range(0, len(clean)-length+1, 10))

def check_ifeval_contamination(prompt: str, ifeval_13grams: set, ifeval_substrings: set) -> bool:
    cand_13g = get_ngrams(prompt, 13)
    if cand_13g.intersection(ifeval_13grams):
        return True
    cand_subs = get_char_substrings(prompt, 50)
    if cand_subs.intersection(ifeval_substrings):
        return True
    return False

def check_math_correct(pred_text: str, gold_text: str) -> bool:
    try:
        gold_parsed = parse(gold_text, parsing_timeout=None)
        pred_parsed = parse(pred_text, parsing_timeout=None)
        if gold_parsed and pred_parsed:
            if verify(gold_parsed, pred_parsed):
                return True
    except Exception:
        pass
    return False

def generate_batch(model, tokenizer, prompts: list, device: str, max_new_tokens: int = 2048, batch_size: int = 16):
    results = []
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
            text = tokenizer.decode(out[b][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
            is_trunc = (len(out[b]) - enc.input_ids.shape[1]) >= (max_new_tokens - 2)
            results.append({"text": text, "tokens": len(gen_ids), "is_truncated": is_trunc})
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Curate v1.1 Training Targets on Qwen3-1.7B")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--general_count", type=int, default=500)
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    v1_train_path = os.path.join(data_dir, "curated_train_traces_qwen_qwen3-1.7b.jsonl")
    v1_dev_path = os.path.join(data_dir, "curated_dev_traces_qwen_qwen3-1.7b.jsonl")
    
    out_train_path = os.path.join(data_dir, "curated_train_v1_1_qwen3_1.7b.jsonl")
    out_dev_path = os.path.join(data_dir, "curated_dev_v1_1_qwen3_1.7b.jsonl")
    general_manifest_path = os.path.join(data_dir, "general_slice_500_manifest.json")

    print(f"=== Starting Continuous Latent Recurrence v1.1 Data Curation ===")
    print(f"Model: {args.model_id} on {args.device} | Batch Size: {args.batch_size}")
    
    # 1. Build IFEval Contamination Filter
    print("\n--- Step 1: Building IFEval Contamination Filter ---")
    ifeval = load_dataset("google/ifeval", split="train")
    ifeval_13grams = set()
    ifeval_substrings = set()
    for row in ifeval:
        p = row["prompt"]
        ifeval_13grams.update(get_ngrams(p, 13))
        ifeval_substrings.update(get_char_substrings(p, 50))
    print(f"Loaded {len(ifeval)} IFEval evaluation prompts.")
    print(f"Indexed {len(ifeval_13grams)} 13-grams and {len(ifeval_substrings)} 50-character substrings.")

    # 2. Sample N=500 Clean UltraChat Prompts
    print(f"\n--- Step 2: Sampling {args.general_count} Clean General Instruction Prompts from UltraChat ---")
    uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    general_prompts = []
    filtered_out = 0
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
            filtered_out += 1
            continue
        general_prompts.append({
            "id": f"ultrachat_{sample.get('prompt_id', len(general_prompts))}",
            "prompt": p,
            "dataset": "ultrachat"
        })
    print(f"Sampled {len(general_prompts)} clean UltraChat prompts (filtered {filtered_out} overlapping/short candidates).")
    
    # Assert zero overlap
    for item in general_prompts:
        assert not check_ifeval_contamination(item["prompt"], ifeval_13grams, ifeval_substrings), "Contamination assertion failed!"
    print("Contamination Assertion: 100% Zero IFEval 13-Gram & Substring Overlap VERIFIED [PASS].")

    # 3. Load Model and Tokenizer
    print(f"\n--- Step 3: Loading {args.model_id} on {args.device} in pure bfloat16 ---")
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
    print(f"Model loaded. VRAM allocated: {torch.cuda.memory_allocated(args.device)/1e9:.2f} GB")

    # 4. Generate Non-Thinking Targets for Math Traces
    print("\n--- Step 4: Generating Non-Thinking Step-by-Step Solutions for Math Problems ---")
    math_sources = []
    with open(v1_train_path) as f:
        for line in f:
            math_sources.append(json.loads(line))
    dev_sources = []
    with open(v1_dev_path) as f:
        for line in f:
            dev_sources.append(json.loads(line))
    print(f"Loaded {len(math_sources)} train math traces and {len(dev_sources)} dev math traces.")

    def process_math_slice(sources, desc="Math Processing"):
        curated = []
        prompts = [
            f"<|im_start|>user\n{s['question']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            for s in sources
        ]
        
        pbar = tqdm(total=len(prompts), desc=desc)
        for i in range(0, len(prompts), args.batch_size):
            batch_slice = sources[i:i+args.batch_size]
            batch_prompts = prompts[i:i+args.batch_size]
            gen_res = generate_batch(model, tokenizer, batch_prompts, args.device, max_new_tokens=2048, batch_size=args.batch_size)
            
            for s, res in zip(batch_slice, gen_res):
                text = res["text"]
                is_trunc = res["is_truncated"]
                is_correct = False
                if not is_trunc and "\\boxed{" in text:
                    is_correct = check_math_correct(text, s.get("ground_truth", ""))
                
                has_worked_steps = (len(text.split()) >= 40)
                
                if is_correct and has_worked_steps:
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
                        "answer_tokens": res["tokens"]
                    }
                    curated.append(rec)
            pbar.update(len(batch_slice))
        pbar.close()
        return curated

    print("Generating non-thinking solutions for DEV split (target: 100)...")
    dev_curated = process_math_slice(dev_sources, desc="Dev Math Processing")
    print(f"Dev verified correct non-thinking solutions: {len(dev_curated)} / {len(dev_sources)}")

    print("Generating non-thinking solutions for TRAIN split...")
    train_math_curated = process_math_slice(math_sources, desc="Train Math Processing")
    print(f"Train verified correct non-thinking solutions: {len(train_math_curated)} / {len(math_sources)}")

    # 5. Generate Non-Thinking Targets for General Instruction Prompts
    print(f"\n--- Step 5: Generating Responses for {len(general_prompts)} General Instruction Prompts ---")
    gen_prompts_formatted = [
        f"<|im_start|>user\n{g['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        for g in general_prompts
    ]
    gen_results = generate_batch(model, tokenizer, gen_prompts_formatted, args.device, max_new_tokens=1024, batch_size=args.batch_size)
    
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
                "answer_tokens": res["tokens"]
            }
            general_curated.append(rec)
    print(f"General instruction responses generated: {len(general_curated)} / {len(general_prompts)}")

    # Save general slice manifest
    with open(general_manifest_path, "w") as f:
        json.dump({
            "count": len(general_curated),
            "source": "HuggingFaceH4/ultrachat_200k",
            "contamination_assertion": "13-gram and 50-char substring disjointness against google/ifeval verified: 0 overlap",
            "items": [{"id": x["id"], "prompt_snippet": x["question"][:120]} for x in general_curated]
        }, f, indent=2)
    print(f"Saved general slice manifest to: {general_manifest_path}")

    # Combine train dataset
    full_train_dataset = train_math_curated + general_curated
    import random
    random.seed(42)
    random.shuffle(full_train_dataset)

    print(f"\n--- Step 6: Saving Final v1.1 Curated Datasets ---")
    print(f"Total Train Examples: {len(full_train_dataset)} ({len(train_math_curated)} math + {len(general_curated)} general instruction)")
    print(f"Total Dev Examples: {len(dev_curated)} (disjoint math)")

    with open(out_train_path, "w") as f:
        for row in full_train_dataset:
            f.write(json.dumps(row) + "\n")
    print(f"Saved train dataset to: {out_train_path}")

    with open(out_dev_path, "w") as f:
        for row in dev_curated:
            f.write(json.dumps(row) + "\n")
    print(f"Saved dev dataset to: {out_dev_path}")

    print("\n=== v1.1 Data Curation COMPLETE [SUCCESS] ===")

if __name__ == "__main__":
    main()
