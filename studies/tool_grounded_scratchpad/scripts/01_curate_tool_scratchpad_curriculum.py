#!/usr/bin/env python3
"""
studies/tool_grounded_scratchpad/scripts/01_curate_tool_scratchpad_curriculum.py
Generates the Tool-Grounded In-Place Working Memory Scratchpad curriculum.

Uses verified training traces and converts intermediate reasoning steps into
native function calls: `update_scratchpad(variables={...})`.
The environment returns `<tool_response>` containing the updated in-place
cumulative working memory table.

Outputs:
- studies/tool_grounded_scratchpad/data/train_tool_scratchpad.jsonl
- studies/tool_grounded_scratchpad/data/dev_tool_scratchpad.jsonl
- studies/tool_grounded_scratchpad/data/kept_trace_ids.json
"""

import os
import sys
import json
import re
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
DATA_DIR = os.path.join(SCRIPT_DIR, "../data")

REG_PATTERN = re.compile(r"<\|reg\|>\s*([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*?)\s*<\|/reg\|>", re.DOTALL)

SCRATCHPAD_TOOL = {
    "name": "update_scratchpad",
    "description": "Updates key-value entries in the working memory scratchpad.",
    "parameters": {
        "type": "object",
        "properties": {
            "variables": {
                "type": "object",
                "description": "Key-value variable map to update"
            }
        },
        "required": ["variables"]
    }
}

SYSTEM_PROMPT = "You are a helpful assistant with an in-place working memory scratchpad."

def parse_rungs(reg_block: str) -> List[Dict[str, str]]:
    """Splits register block into rungs and extracts (key, value) pairs per rung."""
    rungs = []
    current_vars = {}
    lines = reg_block.split("\n")
    for line in lines:
        line_s = line.strip()
        if line_s.startswith("[R") and "]" in line_s:
            if current_vars:
                rungs.append(dict(current_vars))
                current_vars = {}
        else:
            matches = REG_PATTERN.findall(line)
            for k, v in matches:
                clean_k = k.strip()
                clean_v = v.strip().replace("<|/reg|>", "").replace("<|reg|>", "")
                current_vars[clean_k] = clean_v
    if current_vars:
        rungs.append(dict(current_vars))
    return rungs

def format_scratchpad_record(
    record: Dict[str, Any],
    tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    question = record["question"]
    reg_block = record.get("register_block", "")
    answer_text = record.get("nonthinking_answer", "")
    if not answer_text and "answer" in record:
        answer_text = record["answer"]

    rungs = parse_rungs(reg_block)
    if not rungs:
        # Fallback if no register block
        rungs = [{"goal": "solve_problem", "status": "active"}]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    cumulative_memory = {}
    for rung_idx, rung_vars in enumerate(rungs):
        cumulative_memory.update(rung_vars)
        call_args = json.dumps({"variables": rung_vars}, ensure_ascii=False)
        response_content = json.dumps({"scratchpad": dict(cumulative_memory)}, ensure_ascii=False)

        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "update_scratchpad",
                    "arguments": call_args
                }
            }]
        })
        messages.append({
            "role": "tool",
            "name": "update_scratchpad",
            "content": response_content
        })

    # Final assistant turn: nonthinking answer
    clean_ans = answer_text.strip()
    if clean_ans.startswith("<|im_start|>assistant"):
        clean_ans = clean_ans.split("<|im_start|>assistant")[-1].strip()
    if "<think>\n\n</think>\n\n" not in clean_ans:
        clean_ans = f"<think>\n\n</think>\n\n{clean_ans}"
    if clean_ans.endswith("<|im_end|>"):
        clean_ans = clean_ans[:-len("<|im_end|>")].strip()

    messages.append({
        "role": "assistant",
        "content": clean_ans
    })

    rendered_text = tokenizer.apply_chat_template(
        messages,
        tools=[SCRATCHPAD_TOOL],
        tokenize=False
    )

    return {
        "id": record["id"],
        "dataset": record.get("dataset", "unknown"),
        "level": record.get("level", 0),
        "subject": record.get("subject", "unknown"),
        "question": question,
        "ground_truth": record.get("ground_truth", ""),
        "messages": messages,
        "full_text": rendered_text,
        "final_scratchpad": cumulative_memory
    }

def main():
    print("=" * 80)
    print("CURATING TOOL-GROUNDED SCRATCHPAD CURRICULUM")
    print("=" * 80)

    os.makedirs(DATA_DIR, exist_ok=True)

    model_id = "Qwen/Qwen3-4B"
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    input_train = os.path.join(PROJECT_ROOT, "studies/dynamic_registers/data/train_dynamic_registers.jsonl")
    input_dev = os.path.join(PROJECT_ROOT, "studies/dynamic_registers/data/dev_dynamic_registers.jsonl")

    if not os.path.exists(input_train):
        raise FileNotFoundError(f"Input train file not found: {input_train}")

    with open(input_train) as f:
        train_raw = [json.loads(line) for line in f if line.strip()]
    with open(input_dev) as f:
        dev_raw = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(train_raw)} train and {len(dev_raw)} dev raw register records.")

    # Convert to tool scratchpad format
    train_out = []
    for r in train_raw:
        item = format_scratchpad_record(r, tokenizer)
        train_out.append(item)

    dev_out = []
    for r in dev_raw:
        item = format_scratchpad_record(r, tokenizer)
        dev_out.append(item)

    # Verify disjointness with benchmark suite
    bench_file = os.path.join(PROJECT_ROOT, "data/benchmark_suite_250.json")
    with open(bench_file) as f:
        bench_suite = json.load(f)
    bench_ids = {p["id"] for p in bench_suite}

    train_ids = [r["id"] for r in train_out]
    dev_ids = [r["id"] for r in dev_out]
    overlap_train = set(train_ids).intersection(bench_ids)
    overlap_dev = set(dev_ids).intersection(bench_ids)

    assert len(overlap_train) == 0, f"Contamination detected! Train overlaps bench: {overlap_train}"
    assert len(overlap_dev) == 0, f"Contamination detected! Dev overlaps bench: {overlap_dev}"
    print("Disjointness check PASSED: Exactly 0 overlap with benchmark suite 250.")

    # Write output files
    train_path = os.path.join(DATA_DIR, "train_tool_scratchpad.jsonl")
    dev_path = os.path.join(DATA_DIR, "dev_tool_scratchpad.jsonl")
    kept_ids_path = os.path.join(DATA_DIR, "kept_trace_ids.json")

    with open(train_path, "w") as f:
        for item in train_out:
            f.write(json.dumps(item) + "\n")

    with open(dev_path, "w") as f:
        for item in dev_out:
            f.write(json.dumps(item) + "\n")

    with open(kept_ids_path, "w") as f:
        json.dump({"train_ids": train_ids, "dev_ids": dev_ids, "count": len(train_ids) + len(dev_ids)}, f, indent=2)

    print(f"Saved {len(train_out)} train records to {train_path}")
    print(f"Saved {len(dev_out)} dev records to {dev_path}")
    print(f"Saved kept IDs to {kept_ids_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
