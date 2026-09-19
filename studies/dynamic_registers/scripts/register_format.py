#!/usr/bin/env python3
"""
studies/dynamic_registers/scripts/register_format.py
Format, parse, and validate dynamic key-value registers for Study 2.

Grammar:
  register_write ::= "<|reg|>" reg_key "=" reg_val "<|/reg|>"
  reg_key        ::= [A-Za-z_][A-Za-z0-9_.]*
  reg_val        ::= { any text except "<|/reg|>" }
"""

import re
from typing import Dict, List, Tuple, Optional

REG_OPEN = "<|reg|>"
REG_CLOSE = "<|/reg|>"

CANONICAL_HEADERS = [
    "[R1: Identify givens, constraints, and target variable]\n",
    "[R2: Compute intermediate operations and verify relations]\n",
    "[R3: Execute final deduction and verify constraints]\n"
]

# Regex to match a single register write
REG_PATTERN = re.compile(r"<\|reg\|>\s*([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*?)\s*<\|/reg\|>", re.DOTALL)

def parse_registers(text: str) -> List[Tuple[str, str]]:
    """
    Extracts all (key, value) pairs from a text string in order.
    """
    return REG_PATTERN.findall(text)

def extract_registers(text: str) -> List[Dict[str, str]]:
    """
    Extracts all register writes as dictionaries with 'key' and 'value'.
    """
    return [{"key": k, "value": v} for k, v in parse_registers(text)]

def format_register(key: str, val: str) -> str:
    """
    Formats a single key-value register entry.
    """
    clean_key = re.sub(r"[^A-Za-z0-9_.]", "_", key.strip())
    if not clean_key or not clean_key[0].isalpha() and clean_key[0] != '_':
        clean_key = "var_" + clean_key
    clean_val = val.strip().replace("<|/reg|>", "").replace("<|reg|>", "")
    return f"{REG_OPEN}{clean_key} = {clean_val}{REG_CLOSE}"

def format_rung(rung_header: str, registers: List[Tuple[str, str]]) -> str:
    """
    Formats a complete rung with its header and register writes.
    """
    lines = [rung_header.rstrip("\n")]
    for k, v in registers:
        lines.append(format_register(k, v))
    return "\n".join(lines) + "\n"

def make_filler_registers(reg_text: str) -> str:
    """
    Generates syntax-matched non-informative filler for Arm 2.
    Preserves delimiters and key structure, but replaces semantic values
    with non-informative characters matched in length.
    """
    def replace_val(match):
        key = match.group(1)
        val = match.group(2)
        # Generate length-matched non-informative digits/dots
        length = max(1, len(val))
        filler_val = "0" * length
        return f"{REG_OPEN}{key} = {filler_val}{REG_CLOSE}"

    return REG_PATTERN.sub(replace_val, reg_text)

if __name__ == "__main__":
    # Self-test
    sample = (
        "[R1: Identify givens, constraints, and target variable]\n"
        "<|reg|>pink_hats = 26<|/reg|>\n"
        "<|reg|>green_hats = 15<|/reg|>\n"
        "[R2: Compute intermediate operations and verify relations]\n"
        "<|reg|>pink_rem = 26 - 10 = 16<|/reg|>\n"
        "[R3: Execute final deduction and verify constraints]\n"
        "<|reg|>ans = 43<|/reg|>\n"
    )
    parsed = parse_registers(sample)
    print("Parsed registers:", parsed)
    assert len(parsed) == 4, f"Expected 4, got {len(parsed)}"
    assert parsed[0] == ("pink_hats", "26")
    assert parsed[2] == ("pink_rem", "26 - 10 = 16")

    filler = make_filler_registers(sample)
    print("\nSyntax-matched filler:\n", filler)
    parsed_filler = parse_registers(filler)
    assert len(parsed_filler) == 4
    assert parsed_filler[0] == ("pink_hats", "00")
    print("\nSelf-test PASSED!")
