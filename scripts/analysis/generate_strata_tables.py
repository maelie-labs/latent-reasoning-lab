#!/usr/bin/env python3
"""
scripts/analysis/generate_strata_tables.py

Generates detailed per-stratum telemetry tables for benchmark runs on data/benchmark_suite_250.json.
Reports Accuracy, Mean & Median Latency, Median Answer Tokens, and Truncation Rate.
Outputs both formatted Markdown tables and LaTeX tabular snippets.

Execution Environment: CPU only (CUDA_VISIBLE_DEVICES="")
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def load_suite_strata(suite_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads data/benchmark_suite_250.json and indexes problem metadata:
    - benchmark: GSM8K or MATH-500
    - stratum: Short, Medium, Long, or Level 1..5
    - level: int (for MATH-500)
    """
    if not os.path.exists(suite_path):
        raise FileNotFoundError(f"Benchmark suite file not found: {suite_path}")

    with open(suite_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    meta = {}
    for p in problems:
        pid = p["id"]
        bench = p.get("benchmark", "")
        stratum = p.get("stratum", "")
        level = p.get("level")
        meta[pid] = {
            "id": pid,
            "benchmark": bench,
            "stratum": stratum,
            "level": level,
        }
    return meta

def load_eval_records(filepath: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    Loads eval JSONL file and extracts per-(problem_id, seed) metrics.
    """
    records = {}
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Eval file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            pid = str(data.get("problem_id", data.get("id", ""))).strip()
            if not pid:
                continue

            seed = int(data.get("seed", 42))

            # Correctness
            is_correct_raw = data.get("is_correct")
            if is_correct_raw is None and "run" in data and isinstance(data["run"], dict):
                is_correct_raw = data["run"].get("is_correct")
            is_correct = bool(is_correct_raw == True or is_correct_raw == 1)

            # Truncation
            is_trunc_raw = data.get("is_truncated")
            if is_trunc_raw is None and "run" in data and isinstance(data["run"], dict):
                is_trunc_raw = data["run"].get("is_truncated")
            is_truncated = bool(is_trunc_raw == True or is_trunc_raw == 1)

            if is_truncated:
                is_correct = False

            # Latency (in ms)
            dur_ms = None
            if "total_time_ms" in data and data["total_time_ms"] is not None:
                dur_ms = float(data["total_time_ms"])
            elif "run" in data and isinstance(data["run"], dict) and "total_time_ms" in data["run"]:
                dur_ms = float(data["run"]["total_time_ms"])
            elif "dur" in data and data["dur"] is not None:
                dur_ms = float(data["dur"]) * 1000.0
            elif "run" in data and isinstance(data["run"], dict) and "total_time_s" in data["run"]:
                dur_ms = float(data["run"]["total_time_s"]) * 1000.0
            elif "run" in data and isinstance(data["run"], dict) and "dur" in data["run"]:
                dur_ms = float(data["run"]["dur"]) * 1000.0

            # Answer tokens
            ans_tokens = None
            if "ans_tokens" in data and data["ans_tokens"] is not None:
                ans_tokens = int(data["ans_tokens"])
            elif "run" in data and isinstance(data["run"], dict) and "ans_tokens" in data["run"]:
                ans_tokens = int(data["run"]["ans_tokens"])
            elif "total_tokens" in data and data["total_tokens"] is not None:
                ans_tokens = int(data["total_tokens"])

            records[(pid, seed)] = {
                "problem_id": pid,
                "seed": seed,
                "is_correct": is_correct,
                "is_truncated": is_truncated,
                "dur_ms": dur_ms,
                "ans_tokens": ans_tokens,
            }
    return records


def compute_stratum_metrics(
    records: Dict[Tuple[str, int], Dict[str, Any]],
    problem_ids: List[str],
) -> Dict[str, Any]:
    matched_evals = [records[(pid, s)] for pid in problem_ids for s in [42, 123, 456, 789] if (pid, s) in records]
    
    if not matched_evals:
        # Check any seeds
        matched_evals = [rec for (pid, s), rec in records.items() if pid in problem_ids]

    n = len(matched_evals)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "mean_latency_ms": None,
            "median_latency_ms": None,
            "median_tokens": None,
            "truncation_rate": 0.0,
        }

    correct_count = sum(1 for e in matched_evals if e["is_correct"])
    trunc_count = sum(1 for e in matched_evals if e["is_truncated"])
    acc = (correct_count / n) * 100.0
    trunc_rate = (trunc_count / n) * 100.0

    durations = [e["dur_ms"] for e in matched_evals if e["dur_ms"] is not None]
    tokens = [e["ans_tokens"] for e in matched_evals if e["ans_tokens"] is not None]

    mean_lat = float(np.mean(durations)) if durations else None
    med_lat = float(np.median(durations)) if durations else None
    med_tok = float(np.median(tokens)) if tokens else None

    return {
        "n": n,
        "n_problems": len(problem_ids),
        "correct": correct_count,
        "accuracy": acc,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": med_lat,
        "median_tokens": med_tok,
        "truncation_rate": trunc_rate,
    }


def generate_report(
    meta: Dict[str, Dict[str, Any]],
    eval_dict: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]],
    output_md: Optional[str] = None,
    output_tex: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes full stratum tables across all loaded arms.
    """
    strata_definitions = [
        ("GSM8K: All", [p for p, d in meta.items() if d["benchmark"] == "GSM8K"]),
        ("  - Short (<=3 sent)", [p for p, d in meta.items() if d["benchmark"] == "GSM8K" and d["stratum"] == "Short"]),
        ("  - Medium (4-6 sent)", [p for p, d in meta.items() if d["benchmark"] == "GSM8K" and d["stratum"] == "Medium"]),
        ("  - Long (>=7 sent)", [p for p, d in meta.items() if d["benchmark"] == "GSM8K" and d["stratum"] == "Long"]),
        ("MATH-500: All", [p for p, d in meta.items() if d["benchmark"] == "MATH-500"]),
        ("  - Level 1", [p for p, d in meta.items() if d["benchmark"] == "MATH-500" and d["stratum"] == "Level 1"]),
        ("  - Level 2", [p for p, d in meta.items() if d["benchmark"] == "MATH-500" and d["stratum"] == "Level 2"]),
        ("  - Level 3", [p for p, d in meta.items() if d["benchmark"] == "MATH-500" and d["stratum"] == "Level 3"]),
        ("  - Level 4", [p for p, d in meta.items() if d["benchmark"] == "MATH-500" and d["stratum"] == "Level 4"]),
        ("  - Level 5", [p for p, d in meta.items() if d["benchmark"] == "MATH-500" and d["stratum"] == "Level 5"]),
        ("MATH-500: Hard Subset (L3-5)", [p for p, d in meta.items() if d["benchmark"] == "MATH-500" and d.get("level", 0) >= 3]),
        ("Overall Suite (250 Problems)", list(meta.keys())),
    ]

    results = {}
    for arm_name, records in eval_dict.items():
        results[arm_name] = {}
        for s_name, p_ids in strata_definitions:
            results[arm_name][s_name] = compute_stratum_metrics(records, p_ids)

    # Build Markdown table
    md_lines = []
    md_lines.append("# Per-Stratum Telemetry & Stratification Table\n")
    
    headers = ["Stratum / Subset", "Problems (N)"]
    for arm_name in eval_dict.keys():
        headers.extend([f"{arm_name} Acc (%)", f"{arm_name} Trunc (%)"])
    
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")

    for s_name, p_ids in strata_definitions:
        row = [s_name.replace("  - ", "&nbsp;&nbsp;↳ "), f"{len(p_ids)} ({len(p_ids)*4})"]
        for arm_name in eval_dict.keys():
            m = results[arm_name][s_name]
            acc_str = f"**{m['accuracy']:5.2f}%**" if "Overall" in s_name or "All" in s_name else f"{m['accuracy']:5.2f}%"
            trunc_str = f"{m['truncation_rate']:4.1f}%"
            row.extend([acc_str, trunc_str])
        md_lines.append("| " + " | ".join(row) + " |")

    # Latency & Token breakdown table
    md_lines.append("\n### Latency & Answer Token Breakdown\n")
    tok_headers = ["Stratum / Subset"]
    for arm_name in eval_dict.keys():
        tok_headers.extend([f"{arm_name} Med Lat (ms)", f"{arm_name} Med Toks"])
    md_lines.append("| " + " | ".join(tok_headers) + " |")
    md_lines.append("| " + " | ".join([":---"] + [":---:"] * (len(tok_headers) - 1)) + " |")

    for s_name, _ in strata_definitions:
        row = [s_name.replace("  - ", "&nbsp;&nbsp;↳ ")]
        for arm_name in eval_dict.keys():
            m = results[arm_name][s_name]
            lat_str = f"{m['median_latency_ms']:.1f}" if m['median_latency_ms'] is not None else "N/A*"
            tok_str = f"{m['median_tokens']:.0f}" if m['median_tokens'] is not None else "-"
            row.extend([lat_str, tok_str])
        md_lines.append("| " + " | ".join(row) + " |")

    md_output = "\n".join(md_lines)

    # Build LaTeX tabular snippet
    tex_lines = []
    tex_lines.append("% Auto-generated by scripts/analysis/generate_strata_tables.py")
    tex_lines.append("\\begin{table*}[t]")
    tex_lines.append("\\centering")
    tex_lines.append("\\small")
    tex_cols = "l r " + " ".join(["r r" for _ in eval_dict.keys()])
    tex_lines.append(f"\\begin{{tabular}}{{{tex_cols}}}")
    tex_lines.append("\\toprule")
    col_heads = " & ".join([f"\\multicolumn{{2}}{{c}}{{{name}}}" for name in eval_dict.keys()])
    tex_lines.append(f"Stratum & $N$ & {col_heads} \\\\")
    sub_heads = " & ".join(["Acc (\\%) & Trunc (\\%)" for _ in eval_dict.keys()])
    tex_lines.append(f"\\cmidrule(r){{1-2}} & & {sub_heads} \\\\")
    tex_lines.append("\\midrule")

    for s_name, p_ids in strata_definitions:
        clean_name = s_name.replace("  - ", "~~").replace("<=", "$\\le$").replace(">=", "$\\ge$").replace("%", "\\%")
        row = [clean_name, f"{len(p_ids)*4}"]
        for arm_name in eval_dict.keys():
            m = results[arm_name][s_name]
            row.extend([f"{m['accuracy']:.2f}", f"{m['truncation_rate']:.1f}"])
        tex_lines.append(" & ".join(row) + " \\\\")

    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\caption{Per-stratum evaluation accuracy and truncation rate across experimental arms on \\texttt{benchmark\\_suite\\_250.json}.}")
    tex_lines.append("\\label{tab:strata_telemetry}")
    tex_lines.append("\\end{table*}")
    tex_output = "\n".join(tex_lines)

    if output_md:
        os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(md_output)
        print(f"Saved Markdown strata table to {output_md}")

    if output_tex:
        os.makedirs(os.path.dirname(output_tex) or ".", exist_ok=True)
        with open(output_tex, "w", encoding="utf-8") as f:
            f.write(tex_output)
        print(f"Saved LaTeX strata table to {output_tex}")

    return {
        "metrics": results,
        "markdown": md_output,
        "latex": tex_output,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate per-stratum telemetry tables")
    parser.add_argument("--suite_file", type=str, default="data/benchmark_suite_250.json")
    parser.add_argument("--eval_files", type=str, nargs="+", required=True, help="List of 'Label=path/to/eval.jsonl'")
    parser.add_argument("--output_md", type=str, default="data/strata_telemetry_table.md")
    parser.add_argument("--output_tex", type=str, default="data/strata_telemetry_table.tex")
    parser.add_argument("--output_json", type=str, default="data/strata_telemetry_results.json")
    args = parser.parse_args()

    meta = load_suite_strata(args.suite_file)
    print(f"Loaded {len(meta)} problems from {args.suite_file}")

    eval_dict = {}
    for item in args.eval_files:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            label = os.path.basename(item).replace(".jsonl", "")
            path = item
        print(f"Loading {label} from {path}...")
        eval_dict[label] = load_eval_records(path)

    report = generate_report(meta, eval_dict, output_md=args.output_md, output_tex=args.output_tex)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report["metrics"], f, indent=2)
    print(f"Saved JSON metrics to {args.output_json}")

    print("\n" + report["markdown"] + "\n")

if __name__ == "__main__":
    main()
