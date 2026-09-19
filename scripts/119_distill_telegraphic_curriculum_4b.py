#!/usr/bin/env python3
"""
scripts/119_distill_telegraphic_curriculum_4b.py
Distill Telegraphic Propositional CoT and Headers-Only Curricula for 4B Models.

Processes self-distilled train/dev traces for:
1. Qwen/Qwen3-4B (Dense Pure Transformer)
2. Qwen/Qwen3.5-4B (Hybrid Gated DeltaNet)

Outputs:
- data/train_telegraphic_cot_qwen3_4b.jsonl
- data/dev_telegraphic_cot_qwen3_4b.jsonl
- data/train_headers_only_qwen3_4b.jsonl
- data/dev_headers_only_qwen3_4b.jsonl
- data/train_telegraphic_cot_qwen3_5_4b.jsonl
- data/dev_telegraphic_cot_qwen3_5_4b.jsonl
- data/train_headers_only_qwen3_5_4b.jsonl
- data/dev_headers_only_qwen3_5_4b.jsonl

Preserves:
1. 100% test-set disjointness proof against data/benchmark_suite_250.json.
2. Exact prompt matching with model-specific delimiters.
3. Untouched canonical nonthinking answer with \\boxed{}.
"""

import os
import re
import sys
import json
from typing import Dict, Any, List
from transformers import AutoTokenizer

DISCOURSE_FILLER_PATTERNS = [
    r"(?i)\b(okay|alright|well|hmm|um|let me see|let's see|let me think|let's think|i think|wait|hold on)\b",
    r"(?i)\b(let me check|let's check|let me re-read|let me double check|to be sure|just to be sure)\b",
    r"(?i)\b(i should probably|maybe i can|could it be that|what if we|how about we)\b",
    r"(?i)\b(first of all|now let's move on to|got that part|that seems right|yes, that makes sense)\b",
]

CANONICAL_HEADERS = (
    "[R1: Identify givens, constraints, and target variable]\n"
    "[R2: Compute intermediate operations and verify relations]\n"
    "[R3: Execute final deduction and verify constraints]\n"
)

def clean_discourse_filler(text: str) -> str:
    res = text
    for pat in DISCOURSE_FILLER_PATTERNS:
        res = re.sub(pat, "", res)
    res = re.sub(r"[ \t]+", " ", res)
    res = re.sub(r"\s*,\s*,", ",", res)
    res = re.sub(r"^\s*[,;\.]\s*", "", res)
    return res.strip()

def clean_line(text: str) -> str:
    res = text.replace("<|im_end|>", "").replace("$$", "").strip()
    res = re.sub(r"\*+", "", res).strip()
    res = re.sub(r"^[#\-\s:]+", "", res).strip()
    res = re.sub(r"\s+", " ", res).strip()
    return res

def extract_telegraphic_scratchpad(sample: Dict[str, Any], tokenizer, max_tokens: int = 150) -> str:
    ans = sample.get("answer", "")
    gt_match = re.search(r"\\boxed\{([^}]+)\}", ans)
    gt = gt_match.group(1) if gt_match else ""
    
    steps = []
    step_chunks = re.split(r"(?:###|\*\*|Step)\s*\d+[:\.]*", ans)
    if len(step_chunks) > 1:
        for idx, chunk in enumerate(step_chunks[1:], 1):
            lines = [clean_line(l) for l in chunk.split("\n") if l.strip() and not l.strip().startswith("---")]
            eq_lines = []
            for l in lines:
                if not l or len(l) > 140:
                    continue
                if any(op in l for op in ["=", "+", "-", "*", "/", "<", ">", "%", "^", "\\frac"]):
                    cleaned = clean_discourse_filler(l)
                    if cleaned and cleaned not in eq_lines:
                        eq_lines.append(cleaned)
            if eq_lines:
                steps.append(f"- Step {idx}: {'; '.join(eq_lines[:2])}")

    if not steps and sample.get("steps"):
        t_steps = sample["steps"]
        for idx, ts in enumerate(t_steps[:6], 1):
            cleaned = clean_discourse_filler(clean_line(ts))
            sents = [s.strip() for s in re.split(r"[\.\n]", cleaned) if s.strip()]
            math_sents = [s for s in sents if any(c.isdigit() for c in s) and len(s) < 120]
            if math_sents:
                steps.append(f"- Step {len(steps)+1}: {math_sents[0]}")
            if len(steps) >= 4:
                break

    if not steps:
        steps.append("- Deduction: Evaluate problem constraints and solve directly.")

    if gt:
        steps.append(f"- Ans: \\boxed{{{gt}}}")

    scratchpad = "\n".join(steps) + "\n"
    toks = tokenizer.encode(scratchpad, add_special_tokens=False)
    while len(toks) > max_tokens and len(steps) > 2:
        steps.pop(1)
        scratchpad = "\n".join(steps) + "\n"
        toks = tokenizer.encode(scratchpad, add_special_tokens=False)

    return scratchpad

def process_model(model_tag: str, tokenizer_id: str, train_in: str, dev_in: str, test_suite_path: str):
    print(f"\n========================================================")
    print(f"=== Distilling Telegraphic & Headers Curricula for {model_tag} ===")
    print(f"========================================================")

    with open(test_suite_path) as f:
        suite = json.load(f)
    test_ids = {p["id"] for p in suite}
    print(f"Loaded {len(test_ids)} benchmark test IDs for disjointness enforcement.")

    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    for split_name, in_path in [("train", train_in), ("dev", dev_in)]:
        out_tele_path = f"data/{split_name}_telegraphic_cot_{model_tag}.jsonl"
        out_hdr_path = f"data/{split_name}_headers_only_{model_tag}.jsonl"

        records_tele = []
        records_hdr = []
        token_counts = []

        with open(in_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                pid = rec.get("id")
                assert pid not in test_ids, f"FATAL: Test set contamination detected for ID {pid}!"

                prompt = rec["prompt"]
                ans_text = rec["answer"]
                if not ans_text.endswith("<|im_end|>"):
                    ans_text += "<|im_end|>"

                # Telegraphic scratchpad
                tele_thought = extract_telegraphic_scratchpad(rec, tok, max_tokens=150)
                t_count = len(tok.encode(tele_thought, add_special_tokens=False))
                token_counts.append(t_count)

                # Format record for Telegraphic CoT
                rec_tele = {
                    "id": pid,
                    "dataset": rec.get("dataset", "unknown"),
                    "prompt": prompt,
                    "thought": tele_thought,
                    "answer": ans_text,
                    "tokens": {
                        "thought": t_count,
                        "answer": len(tok.encode(ans_text, add_special_tokens=False))
                    }
                }
                records_tele.append(rec_tele)

                # Format record for Headers-Only Control
                rec_hdr = {
                    "id": pid,
                    "dataset": rec.get("dataset", "unknown"),
                    "prompt": prompt,
                    "thought": CANONICAL_HEADERS,
                    "answer": ans_text,
                    "tokens": {
                        "thought": len(tok.encode(CANONICAL_HEADERS, add_special_tokens=False)),
                        "answer": len(tok.encode(ans_text, add_special_tokens=False))
                    }
                }
                records_hdr.append(rec_hdr)

        # Write telegraphic
        with open(out_tele_path, "w") as f:
            for r in records_tele:
                f.write(json.dumps(r) + "\n")

        # Write headers-only
        with open(out_hdr_path, "w") as f:
            for r in records_hdr:
                f.write(json.dumps(r) + "\n")

        median_toks = float(sorted(token_counts)[len(token_counts) // 2]) if token_counts else 0.0
        print(f"[{split_name.upper()}] Processed {len(records_tele)} records | Telegraphic Median Tokens: {median_toks:.1f}")
        print(f"  -> Saved Telegraphic: {out_tele_path}")
        print(f"  -> Saved Headers-Only: {out_hdr_path}")

def main():
    test_suite_path = "data/benchmark_suite_250.json"
    
    # 1. Qwen3-4B
    process_model(
        model_tag="qwen3_4b",
        tokenizer_id="Qwen/Qwen3-4B",
        train_in="data/curated_train_traces_qwen_qwen3-4b.jsonl",
        dev_in="data/curated_dev_traces_qwen_qwen3-4b.jsonl",
        test_suite_path=test_suite_path
    )

    # 2. Qwen3.5-4B
    process_model(
        model_tag="qwen3_5_4b",
        tokenizer_id="Qwen/Qwen3.5-4B",
        train_in="data/curated_train_traces_qwen_qwen3_5-4b.jsonl",
        dev_in="data/curated_dev_traces_qwen_qwen3_5-4b.jsonl",
        test_suite_path=test_suite_path
    )

    print("\nAll 4B curricula successfully distilled and verified with zero test contamination!")

if __name__ == "__main__":
    main()
