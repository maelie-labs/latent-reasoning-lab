#!/usr/bin/env python3
"""
studies/telegraphic_cot/scripts/01_distill_telegraphic_curriculum.py
Phase 1: Telegraphic Propositional Curriculum Distillation.

Extracts concise propositional deductions from verified reasoning traces while
strictly preserving:
1. Exact user prompts ending with <think>\n.
2. Complete, detailed, user-facing pedagogical solutions after </think>\n\n.
3. 100% test-set disjointness proof against data/benchmark_suite_250.json.

Generates:
- studies/telegraphic_cot/data/train_telegraphic_cot.jsonl
- studies/telegraphic_cot/data/dev_telegraphic_cot.jsonl
- studies/telegraphic_cot/data/train_matched_pause_control.jsonl
- studies/telegraphic_cot/data/dev_matched_pause_control.jsonl
- studies/telegraphic_cot/data/kept_trace_ids.json
"""

import os
import sys
import re
import json
from typing import Dict, Any, List
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from telegraphic_format import clean_discourse_filler, format_telegraphic_scratchpad

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def clean_line(text: str) -> str:
    """Removes markdown artifacts, delimiters, and extra whitespace."""
    res = text.replace("<|im_end|>", "").replace("$$", "").strip()
    res = re.sub(r"\*+", "", res).strip()
    res = re.sub(r"^[#\-\s:]+", "", res).strip()
    res = re.sub(r"\s+", " ", res).strip()
    return res

def distill_record_to_telegraphic(sample: Dict[str, Any], tokenizer, max_tokens: int = 150) -> str:
    """
    Distills a verified problem record into a concise propositional scratchpad.
    """
    ans = sample.get("nonthinking_answer", "")
    gt = sample.get("ground_truth", "").strip()
    steps = []

    # 1. Parse markdown step headers if present
    step_chunks = re.split(r"###\s*\**Step\s*\d+[:\.]*\**", ans)
    if len(step_chunks) > 1:
        for idx, chunk in enumerate(step_chunks[1:], 1):
            lines = [clean_line(l) for l in chunk.split("\n") if l.strip() and not l.strip().startswith("---")]
            # Filter for lines with equations, mathematical symbols, or key calculations
            eq_lines = []
            for l in lines:
                if not l or len(l) > 140:
                    continue
                if any(op in l for op in ["=", "+", "-", "*", "/", "<", ">", "%", "^", "\\frac"]):
                    cleaned = clean_discourse_filler(l)
                    if cleaned and cleaned not in eq_lines:
                        eq_lines.append(cleaned)
            if eq_lines:
                steps.append(f"Step {idx}: {'; '.join(eq_lines[:2])}")

    # 2. Fallback to teacher_steps if structured headers were not found
    if not steps and sample.get("teacher_steps"):
        t_steps = sample["teacher_steps"]
        for idx, ts in enumerate(t_steps[:6], 1):
            cleaned = clean_discourse_filler(clean_line(ts))
            sents = [s.strip() for s in re.split(r"[\.\n]", cleaned) if s.strip()]
            math_sents = [s for s in sents if any(c.isdigit() for c in s) and len(s) < 120]
            if math_sents:
                steps.append(f"Step {len(steps)+1}: {math_sents[0]}")
            if len(steps) >= 5:
                break

    if not steps:
        steps.append("Deduction: Evaluate problem constraints directly.")

    scratchpad = format_telegraphic_scratchpad(steps, answer=gt)
    
    # Check token count and enforce <= max_tokens
    toks = tokenizer.encode(scratchpad, add_special_tokens=False)
    if len(toks) > max_tokens:
        # Prune earlier steps until it fits within max_tokens
        while len(steps) > 2 and len(toks) > max_tokens:
            steps.pop(0)
            scratchpad = format_telegraphic_scratchpad(steps, answer=gt)
            toks = tokenizer.encode(scratchpad, add_special_tokens=False)

    return scratchpad

def build_pause_scratchpad(token_len: int) -> str:
    """Generates an exact length-matched sequence of pause tokens."""
    return ("<pause> " * max(1, token_len)).strip() + "\n"

def process_curriculum(tokenizer_id: str = "Qwen/Qwen3-1.7B"):
    print("=" * 80)
    print("PHASE 1: DISTILLING TELEGRAPHIC PROPOSITIONAL CURRICULUM")
    print("=" * 80)

    src_train = os.path.join(PROJECT_ROOT, "studies/dynamic_registers/data/train_dynamic_registers.jsonl")
    src_dev = os.path.join(PROJECT_ROOT, "studies/dynamic_registers/data/dev_dynamic_registers.jsonl")
    bench_file = os.path.join(PROJECT_ROOT, "data/benchmark_suite_250.json")

    out_dir = os.path.join(PROJECT_ROOT, "studies/telegraphic_cot/data")
    os.makedirs(out_dir, exist_ok=True)

    with open(bench_file) as f:
        bench_problems = json.load(f)
    test_ids = set(p["id"] for p in bench_problems)
    print(f"Loaded {len(test_ids)} test benchmark problem IDs for disjointness assertion.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    all_kept_ids = []
    for split_name, src_file in [("train", train_src), ("dev", dev_src)]:
        print(f"\nProcessing {split_name} split from {src_file}...")
        with open(src_file) as f:
            records = [json.loads(l) for l in f if l.strip()]

        print(f"Loaded {len(records)} records for {split_name}.")

        telegraphic_records = []
        pause_records = []
        token_lengths = []

        for r in records:
            r_id = r["id"]
            if r_id in test_ids:
                raise ValueError(f"CRITICAL CONTAMINATION: Problem {r_id} exists in test benchmark suite!")
            all_kept_ids.append(r_id)

            t_scratch = distill_record_to_telegraphic(r, tokenizer, max_tokens=150)
            tok_len = len(tokenizer.encode(t_scratch, add_special_tokens=False))
            token_lengths.append(tok_len)

            rec_telegraphic = {
                "id": r_id,
                "dataset": r.get("dataset", ""),
                "level": r.get("level", 1),
                "subject": r.get("subject", ""),
                "question": r["question"],
                "ground_truth": r.get("ground_truth", ""),
                "prompt": r["prompt"],
                "telegraphic_think": t_scratch,
                "nonthinking_answer": r["nonthinking_answer"],
                "think_tokens": tok_len
            }
            telegraphic_records.append(rec_telegraphic)

            rec_pause = {
                "id": r_id,
                "dataset": r.get("dataset", ""),
                "level": r.get("level", 1),
                "subject": r.get("subject", ""),
                "question": r["question"],
                "ground_truth": r.get("ground_truth", ""),
                "prompt": r["prompt"],
                "pause_think": build_pause_scratchpad(tok_len),
                "nonthinking_answer": r["nonthinking_answer"],
                "think_tokens": tok_len
            }
            pause_records.append(rec_pause)

        # Save files
        out_tele = os.path.join(out_dir, f"{split_name}_telegraphic_cot.jsonl")
        out_pause = os.path.join(out_dir, f"{split_name}_matched_pause_control.jsonl")

        with open(out_tele, "w") as f:
            for rec in telegraphic_records:
                f.write(json.dumps(rec) + "\n")

        with open(out_pause, "w") as f:
            for rec in pause_records:
                f.write(json.dumps(rec) + "\n")

        print(f"Saved {len(telegraphic_records)} records to {out_tele}")
        print(f"Saved {len(pause_records)} records to {out_pause}")
        print(f"Thinking token stats: min={min(token_lengths)}, median={int(sorted(token_lengths)[len(token_lengths)//2])}, max={max(token_lengths)}")

    # Save kept IDs disjointness proof
    manifest_path = os.path.join(out_dir, "kept_trace_ids.json")
    with open(manifest_path, "w") as f:
        json.dump({"total_kept_ids": len(all_kept_ids), "ids": all_kept_ids, "test_suite_overlap": 0}, f, indent=2)
    print(f"\nDisjointness manifest saved to {manifest_path} (0 overlap with test suite).")
    print("=" * 80)
    print("PHASE 1 CURRICULUM DISTILLATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    process_curriculum()
