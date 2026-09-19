#!/usr/bin/env python3
"""
studies/telegraphic_cot/scripts/telegraphic_format.py
Format, parse, and validate Telegraphic Propositional CoT traces.

Propositional Schema:
- Premise: [Given facts, constraints, and target goal]
- Step 1:  [First mathematical or logical deduction]
- Step 2..N: [Subsequent algebraic transformations / lemma applications]
- Check:   [Sanity check / constraint verification / edge case test]
- Ans:     [Crystallized result ready for answer rendering]
"""

import re
from typing import List, Dict, Any, Optional

DISCOURSE_FILLER_PATTERNS = [
    r"(?i)\b(okay|alright|well|hmm|um|let me see|let's see|let me think|let's think|i think|wait|hold on)\b",
    r"(?i)\b(let me check|let's check|let me re-read|let me double check|to be sure|just to be sure)\b",
    r"(?i)\b(i should probably|maybe i can|could it be that|what if we|how about we)\b",
    r"(?i)\b(first of all|now let's move on to|got that part|that seems right|yes, that makes sense)\b",
]

def clean_discourse_filler(text: str) -> str:
    """Strips conversational self-talk and conversational hedges."""
    res = text
    for pat in DISCOURSE_FILLER_PATTERNS:
        res = re.sub(pat, "", res)
    # Collapse multiple spaces and clean punctuation artifacts
    res = re.sub(r"[ \t]+", " ", res)
    res = re.sub(r"\s*,\s*,", ",", res)
    res = re.sub(r"^\s*[,;\.]\s*", "", res)
    return res.strip()

def format_telegraphic_scratchpad(steps: List[str], check: Optional[str] = None, answer: Optional[str] = None) -> str:
    """
    Renders a list of deduction steps into a clean bulleted propositional scratchpad.
    """
    lines = []
    for s in steps:
        clean_s = s.strip()
        if not clean_s.startswith("-"):
            clean_s = f"- {clean_s}"
        lines.append(clean_s)
    
    if check and not any("check:" in l.lower() for l in lines):
        lines.append(f"- Check: {check.strip()}")
    
    if answer and not any(l.lower().startswith(("- ans:", "- result:", "- conclusion:")) for l in lines):
        lines.append(f"- Ans: {answer.strip()}")
        
    return "\n".join(lines) + "\n"

def parse_telegraphic_scratchpad(text: str) -> List[str]:
    """
    Parses a telegraphic thought block into individual propositional bullets.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return [l.lstrip("-* ").strip() for l in lines if l.startswith(("-", "*"))]

if __name__ == "__main__":
    test_sample = [
        "Premise: f(x) = x^2 - 2x + m, g(x) = x^2 - 2x + 4m. Condition: 2f(4) = g(4).",
        "Step 1: f(4) = 16 - 8 + m = 8 + m.",
        "Step 2: g(4) = 16 - 8 + 4m = 8 + 4m.",
        "Step 3: 2(8 + m) = 8 + 4m => 16 + 2m = 8 + 4m => 2m = 8 => m = 4.",
    ]
    rendered = format_telegraphic_scratchpad(test_sample, check="2(12) = 24 = 8 + 16. Consistent.", answer="m = 4")
    print("Rendered scratchpad:\n" + rendered)
    parsed = parse_telegraphic_scratchpad(rendered)
    print("Parsed steps count:", len(parsed))
    assert len(parsed) == 6
    print("Self-test PASSED!")
