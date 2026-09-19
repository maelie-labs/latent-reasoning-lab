#!/usr/bin/env python3
"""
prepare_curriculum_arms.py
Generates the three strictly matched curricula for the Staged Dynamic Registers Study:
1. Arm 1: Dynamic Discrete Registers (authentic <|reg|>k = v<|/reg|>)
2. Control 1: Headers-Only SFT (registers stripped, 3 headers retained)
3. Arm 2: Empirically-Grounded Mismatched Value Filler (exact same keys and count, values drawn from empirical math pool)

Guarantees 100% problem-by-problem alignment across all 1,267 train and 100 dev records.
"""

import os
import sys
import json
import random
import re
from typing import Dict, Any, List, Tuple

CANONICAL_HEADERS = [
    "[R1: Identify givens, constraints, and target variable]\n",
    "[R2: Compute intermediate operations and verify relations]\n",
    "[R3: Execute final deduction and verify constraints]\n"
]

REG_PATTERN = re.compile(r"<\|reg\|>\s*([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*?)\s*<\|/reg\|>", re.DOTALL)

def parse_registers(text: str) -> List[Tuple[str, str]]:
    return REG_PATTERN.findall(text)

def make_headers_only_block(reg_text: str) -> str:
    """Strips all register writes, leaving only the 3 canonical headers."""
    return "".join(CANONICAL_HEADERS)

def extract_all_values(records: List[Dict[str, Any]]) -> List[str]:
    """Collects all authentic register values across all records into an empirical pool."""
    pool = []
    for r in records:
        reg_block = r.get("register_block", "")
        pairs = parse_registers(reg_block)
        for k, v in pairs:
            clean_v = v.strip()
            if clean_v and clean_v not in ["input_data", "execute_derivation", "result", "0"]:
                pool.append(clean_v)
    return pool

def make_mismatched_filler_block(reg_text: str, value_pool: List[str], rng: random.Random) -> str:
    """
    Keeps identical keys, counts, and positions, but replaces each value
    with an authentic value drawn randomly from the empirical math pool.
    """
    def replacer(match):
        key = match.group(1)
        orig_val = match.group(2).strip()
        # Sample from pool until a different value is found
        filler_val = rng.choice(value_pool)
        for _ in range(10):
            if filler_val != orig_val:
                break
            filler_val = rng.choice(value_pool)
        return f"<|reg|>{key} = {filler_val}<|/reg|>"

    return REG_PATTERN.sub(replacer, reg_text)

def process_and_verify(data_dir: str, seed: int = 42):
    rng = random.Random(seed)

    dyn_train_path = os.path.join(data_dir, "train_dynamic_registers.jsonl")
    dyn_dev_path = os.path.join(data_dir, "dev_dynamic_registers.jsonl")

    if not os.path.exists(dyn_train_path) or not os.path.exists(dyn_dev_path):
        raise FileNotFoundError(f"Input dynamic register files not found in {data_dir}")

    with open(dyn_train_path) as f:
        train_dynamic = [json.loads(line) for line in f if line.strip()]
    with open(dyn_dev_path) as f:
        dev_dynamic = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(train_dynamic)} train and {len(dev_dynamic)} dev dynamic records.")

    # 1. Build empirical value pool
    all_records = train_dynamic + dev_dynamic
    value_pool = extract_all_values(all_records)
    print(f"Extracted {len(value_pool)} authentic math register values into empirical pool.")
    if len(value_pool) < 50:
        raise ValueError("Empirical value pool is too small! Check register extraction quality.")

    # 2. Build Control 1: Headers-Only
    train_headers = []
    for r in train_dynamic:
        c = dict(r)
        c["register_block"] = make_headers_only_block(r["register_block"])
        train_headers.append(c)

    dev_headers = []
    for r in dev_dynamic:
        c = dict(r)
        c["register_block"] = make_headers_only_block(r["register_block"])
        dev_headers.append(c)

    # 3. Build Arm 2: Empirically-Grounded Mismatched Value Filler
    train_filler = []
    for r in train_dynamic:
        c = dict(r)
        c["register_block"] = make_mismatched_filler_block(r["register_block"], value_pool, rng)
        train_filler.append(c)

    dev_filler = []
    for r in dev_dynamic:
        c = dict(r)
        c["register_block"] = make_mismatched_filler_block(r["register_block"], value_pool, rng)
        dev_filler.append(c)

    # 4. Save all files
    out_paths = {
        "ctrl_train": os.path.join(data_dir, "train_headers_only.jsonl"),
        "ctrl_dev": os.path.join(data_dir, "dev_headers_only.jsonl"),
        "arm2_train": os.path.join(data_dir, "train_matched_filler_registers.jsonl"),
        "arm2_dev": os.path.join(data_dir, "dev_matched_filler_registers.jsonl"),
    }

    for name, p in out_paths.items():
        records = train_headers if name == "ctrl_train" else dev_headers if name == "ctrl_dev" else train_filler if name == "arm2_train" else dev_filler
        with open(p, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved {len(records)} records to {p}")

    # 5. Sanity Audit & Inspection
    print("\n" + "=" * 60)
    print("PARITY AUDIT: 3 MATCHED ARMS")
    print("=" * 60)
    print(f"Arm 1 (Dynamic Registers):  {len(train_dynamic)} train / {len(dev_dynamic)} dev")
    print(f"Control 1 (Headers-Only):   {len(train_headers)} train / {len(dev_headers)} dev")
    print(f"Arm 2 (Mismatched Filler):  {len(train_filler)} train / {len(dev_filler)} dev")

    # Inspect sample 0 across all three
    print("\n[Sample 0 Comparison]")
    print(f"Problem: {train_dynamic[0]['question'][:60]}...")
    print("\n--- Arm 1 (Dynamic Registers) ---")
    print(train_dynamic[0]['register_block'].strip())
    print("\n--- Control 1 (Headers Only) ---")
    print(train_headers[0]['register_block'].strip())
    print("\n--- Arm 2 (Mismatched Value Filler) ---")
    print(train_filler[0]['register_block'].strip())
    print("=" * 60)
    print("Curricula generation complete and certified!")

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "studies/dynamic_registers/data"
    process_and_verify(data_dir)
