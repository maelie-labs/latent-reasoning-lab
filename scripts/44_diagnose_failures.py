#!/usr/bin/env python3
"""
scripts/44_diagnose_failures.py
Automated Diagnostic Analyzer for Benchmark Failures.

Classifies failure cases into:
1. 'wrong': Actual reasoning / mathematical calculation error.
2. 'extraction_miss': Model arrived at the correct answer in text or thought, but parser missed it.
3. 'no_answer': Model truncated, looped, or produced no final answer.
"""

import os
import sys
import json
import argparse
import re
from math_verify import parse, verify

def classify_failure(record):
    sol = record.get("solution", "")
    content = record.get("content_text", "")
    thinking = record.get("thinking_text", "")
    is_truncated = record.get("is_truncated", False)

    # Check for empty or truncated output with no answer
    if is_truncated and len(content.strip()) < 5:
        return "no_answer", "Truncated before generating final content", None

    if not content.strip():
        return "no_answer", "Empty content text", None

    # Try deep parse on content
    parsed_gold = parse(sol)
    parsed_pred = parse(content)

    if parsed_gold and parsed_pred and verify(parsed_gold, parsed_pred):
        return "extraction_miss", "math_verify succeeds on re-parse", parsed_pred

    # Check if gold answer appears literally anywhere in content
    # Extract gold candidate
    gold_nums = re.findall(r"[-+]?\d+(?:\.\d+)?", sol)
    if gold_nums:
        gold_val = gold_nums[-1]
        # Look for gold_val in content as a whole word
        if re.search(r'\b' + re.escape(gold_val) + r'\b', content):
            return "extraction_miss", f"Gold value '{gold_val}' appears in content text but unextracted", parsed_pred

    # Check if gold answer appeared in thinking text
    if gold_nums:
        gold_val = gold_nums[-1]
        if re.search(r'\b' + re.escape(gold_val) + r'\b', thinking[-500:]):
            return "extraction_miss", f"Gold value '{gold_val}' reached at end of thinking block", parsed_pred

    return "wrong", "Genuine incorrect reasoning or calculation", parsed_pred

def main():
    parser = argparse.ArgumentParser(description="Diagnose benchmark failures from streaming JSONL")
    parser.add_argument("--file", type=str, required=True, help="Path to streaming JSONL file")
    parser.add_argument("--num_cases", type=int, default=20, help="Number of failure cases to inspect")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    records = []
    with open(args.file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    failures = [r for r in records if not r.get("is_correct", False)]
    print(f"\n=======================================================")
    print(f"DIAGNOSTIC REPORT: {args.file}")
    print(f"Total problems: {len(records)} | Total failures: {len(failures)}")
    print(f"Accuracy: {((len(records) - len(failures)) / max(1, len(records))) * 100:.2f}%")
    print(f"=======================================================\n")

    counts = {"wrong": 0, "extraction_miss": 0, "no_answer": 0}
    sample_cases = []

    for i, fail in enumerate(failures):
        cat, reason, parsed = classify_failure(fail)
        counts[cat] += 1
        if len(sample_cases) < args.num_cases:
            sample_cases.append({
                "idx": i + 1,
                "id": fail.get("id"),
                "category": cat,
                "reason": reason,
                "solution": fail.get("solution", "")[:120],
                "content": fail.get("content_text", "")[:200],
                "thinking_snippet": fail.get("thinking_text", "")[-200:] if fail.get("thinking_text") else ""
            })

    print(f"--- FAILURE BREAKDOWN ({len(failures)} total failures) ---")
    for cat, cnt in counts.items():
        pct = (cnt / max(1, len(failures))) * 100
        print(f"  {cat.upper():18s}: {cnt:4d} ({pct:5.1f}%)")

    print(f"\n--- FIRST {len(sample_cases)} FAILURE INSPECTION ---")
    for c in sample_cases:
        print(f"\n[{c['idx']}] ID: {c['id']} | Category: {c['category'].upper()} | Reason: {c['reason']}")
        print(f"  Gold Solution: {c['solution']}")
        print(f"  Model Content: {c['content']}")
        if c['thinking_snippet']:
            print(f"  Think Ending : ...{c['thinking_snippet']}")

if __name__ == "__main__":
    main()
