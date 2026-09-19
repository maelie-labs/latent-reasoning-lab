#!/usr/bin/env python3
"""
09_eval_qwen27b.py
Evaluates Qwen 3.8 27B (served via SGLang at http://localhost:9000/v1)
across the exact 25 benchmark problems (15 stratified GSM8K + 10 MATH-500 Level 4/5).
Measures reasoning latency, total latency, token counts, and accuracy.
Saves results to data/eval_results_qwen27b.json.
"""

import os
import re
import json
import time
import requests
from datasets import load_dataset

SGLANG_URL = "http://localhost:9000/v1/chat/completions"
MODEL_NAME = "qwen3.8-27b"
OUTPUT_FILE = "data/eval_results_qwen27b.json"

def clean_and_extract_candidate(cand):
    """Clean LaTeX macros and isolate float value from candidate string."""
    if not cand:
        return None
    cand = re.sub(r"\\(?:text|mathbf|mathrm)\{([^}]+)\}", r"\1", cand)
    cand = re.sub(r"[\$\\%!\s]", "", cand)
    cand = cand.replace(",", "").strip().rstrip(".")
    m_frac = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", cand)
    if m_frac:
        try:
            return float(m_frac.group(1)) / float(m_frac.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    m_div = re.fullmatch(r"(-?\d+)/(-?\d+)", cand)
    if m_div:
        try:
            return float(m_div.group(1)) / float(m_div.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(cand)
    except ValueError:
        nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", cand)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                pass
    return None

def extract_math_boxed_expression(text):
    """Extracts content inside the last \\boxed{...}, correctly handling nested braces and equalities."""
    if not text or "\\boxed{" not in text:
        return ""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    end = start
    while end < len(text) and depth > 0:
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1
    if depth == 0:
        res = text[start:end-1].strip()
        if "=" in res:
            res = res.split("=")[-1].strip()
        return res.rstrip(".")
    return ""

def normalize_math_str(s):
    """Normalizes a LaTeX mathematical string for comparison."""
    if not s:
        return ""
    s = s.strip()
    boxed = extract_math_boxed_expression(s)
    if boxed:
        s = boxed
    if "=" in s:
        s = s.split("=")[-1].strip()
    s = re.sub(r"\\(?:text|mathbf|mathrm)\{([^}]+)\}", r"\1", s)
    s = re.sub(r"[\$\s]", "", s)
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"^{\circ}", "").replace(r"^\circ", "")
    return s.rstrip(".")

def try_eval_numeric(s):
    """Attempts to evaluate a normalized math string as a float."""
    norm = normalize_math_str(s)
    if not norm:
        return None
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", norm)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", norm)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(norm)
    except ValueError:
        return None

def math_equal(pred_str, gt_str):
    """Robust equivalence check for mathematical answers (LaTeX strings and numbers)."""
    if not pred_str or not gt_str:
        return False
    p_norm = normalize_math_str(pred_str)
    g_norm = normalize_math_str(gt_str)
    if p_norm == g_norm:
        return True
    p_num = try_eval_numeric(p_norm)
    g_num = try_eval_numeric(g_norm)
    if p_num is not None and g_num is not None:
        return abs(p_num - g_num) < 1e-4
    return False

def extract_numeric_answer(text, is_answer_only=False):
    """
    Extracts numerical answer from text, supporting GSM8K and MATH formats.
    """
    if not text:
        return None
        
    # 1. \\boxed{...}
    boxed = extract_math_boxed_expression(text)
    if boxed:
        val = clean_and_extract_candidate(boxed)
        if val is not None:
            return val
            
    # 2. #### <num>
    hash_match = re.findall(r'####\s*([-\d.,]+)', text)
    if hash_match:
        val = clean_and_extract_candidate(hash_match[-1])
        if val is not None:
            return val
            
    # 3. Bolded answers e.g. **18** or **$70,000** or **3 bolts in total**
    bold_matches = re.findall(r"\*\*(?:[A-Za-z\s:]*)?([\$]?[-\d.,]+)(?:[A-Za-z\s]*)\*\*", text)
    if bold_matches:
        val = clean_and_extract_candidate(bold_matches[-1])
        if val is not None:
            return val

    # 4. Explicit phrasing
    ans_match = re.findall(r'(?:the answer is|final answer is|total is|equals|equal to|in total|profit:)\s*([\$]?[-\d.,]+)', text, re.IGNORECASE)
    if ans_match:
        val = clean_and_extract_candidate(ans_match[-1])
        if val is not None:
            return val
            
    # 5. Fallback to last number with commas handled
    if is_answer_only:
        nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", text)
        if nums:
            val = clean_and_extract_candidate(nums[-1])
            if val is not None:
                return val
            
    return None

def check_match(pred_num, pred_text, gt_num, gt_raw, is_math=False):
    """Checks if predicted answer matches ground truth."""
    if pred_num is not None and gt_num is not None:
        if abs(pred_num - gt_num) < 1e-4:
            return True
            
    if gt_raw:
        pred_box = extract_math_boxed_expression(pred_text)
        if pred_box and math_equal(pred_box, gt_raw):
            return True
        if math_equal(pred_text, gt_raw):
            return True
            
    return False

def get_benchmark_problems():
    """Load the exact 25 benchmark problems used in dynamic_halting_results.json."""
    ds_gsm = load_dataset("openai/gsm8k", "main", split="test")
    easy, med, hard = [], [], []
    for i, item in enumerate(ds_gsm):
        ops = re.findall(r"<<.*?>>", item["answer"])
        op_count = len(ops)
        if op_count <= 2 and len(easy) < 5:
            easy.append({"id": f"gsm8k_easy_{i}", "ds_idx": i, "category": "Easy", "op_count": op_count, "question": item["question"], "answer": item["answer"]})
        elif 3 <= op_count <= 4 and len(med) < 5:
            med.append({"id": f"gsm8k_med_{i}", "ds_idx": i, "category": "Medium", "op_count": op_count, "question": item["question"], "answer": item["answer"]})
        elif op_count >= 5 and len(hard) < 5:
            hard.append({"id": f"gsm8k_hard_{i}", "ds_idx": i, "category": "Hard", "op_count": op_count, "question": item["question"], "answer": item["answer"]})
        if len(easy) >= 5 and len(med) >= 5 and len(hard) >= 5:
            break
            
    ds_math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math_samples = []
    for i, item in enumerate(ds_math):
        if item["level"] in [4, 5]:
            math_samples.append({
                "id": f"math500_lvl{item['level']}_{i}",
                "ds_idx": i,
                "category": f"MATH-L{item['level']}",
                "level": item["level"],
                "subject": item["subject"],
                "question": item["problem"],
                "answer": item["answer"]
            })
            if len(math_samples) >= 10:
                break
                
    return easy + med + hard + math_samples

def query_qwen27b(question, max_tokens=2048, temperature=0.6):
    """
    Queries SGLang server at http://localhost:9000/v1/chat/completions using streaming
    to precisely time the reasoning phase vs answer phase.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": question}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    t0 = time.time()
    t_reasoning_start = None
    t_reasoning_end = None
    t_content_start = None
    t_end = None
    
    reasoning_chunks = []
    content_chunks = []
    usage = None
    
    response = requests.post(SGLANG_URL, json=payload, stream=True, timeout=180)
    response.raise_for_status()
    
    for line in response.iter_lines():
        if not line:
            continue
        lstr = line.decode('utf-8')
        if lstr.startswith('data: '):
            data_str = lstr[6:].strip()
            if data_str == '[DONE]':
                break
            data = json.loads(data_str)
            if 'usage' in data and data['usage'] is not None:
                usage = data['usage']
            if not data.get('choices'):
                continue
            delta = data['choices'][0].get('delta', {})
            rc = delta.get('reasoning_content')
            c = delta.get('content')
            now = time.time()
            if rc:
                if t_reasoning_start is None:
                    t_reasoning_start = now
                reasoning_chunks.append(rc)
                t_reasoning_end = now
            if c:
                if t_content_start is None:
                    t_content_start = now
                content_chunks.append(c)
                
    t_end = time.time()
    total_time_s = t_end - t0
    
    reasoning_text = "".join(reasoning_chunks)
    content_text = "".join(content_chunks)
    
    # Fallback to non-streaming usage counts if available
    reasoning_tokens = usage.get('reasoning_tokens', 0) if usage else 0
    completion_tokens = usage.get('completion_tokens', 0) if usage else 0
    answer_tokens = max(0, completion_tokens - reasoning_tokens)
    
    # Compute thinking time
    if t_reasoning_end is not None:
        think_time_ms = (t_reasoning_end - t0) * 1000.0
    elif completion_tokens > 0 and reasoning_tokens > 0:
        think_time_ms = (reasoning_tokens / completion_tokens) * total_time_s * 1000.0
    else:
        think_time_ms = 0.0
        
    return {
        "reasoning_content": reasoning_text,
        "content": content_text,
        "think_time_ms": round(think_time_ms, 2),
        "total_time_s": round(total_time_s, 3),
        "think_tokens": reasoning_tokens,
        "answer_tokens": answer_tokens,
        "usage": usage
    }

def main():
    print(f"=== Stage 1: Evaluating Qwen 3.8 27B on 25 Benchmark Problems ===")
    print(f"SGLang Endpoint: {SGLANG_URL} (Model: {MODEL_NAME})")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    problems = get_benchmark_problems()
    print(f"Loaded {len(problems)} benchmark problems.")
    
    runs = []
    for idx, item in enumerate(problems):
        q = item["question"]
        is_math = "math500" in item["id"]
        gt_num = extract_numeric_answer(item["answer"], is_answer_only=True)
        
        print(f"[{idx+1}/{len(problems)}] [{item['category']}] {item['id']}: {q[:55]}... (GT: {gt_num or item['answer'][:15]})")
        
        res = query_qwen27b(q, max_tokens=2048, temperature=0.6)
        
        pred_num = extract_numeric_answer(res["content"], is_answer_only=True)
        is_correct = check_match(pred_num, res["content"], gt_num, item["answer"], is_math=is_math)
        
        record = {
            "question_id": item["id"],
            "category": item["category"],
            "ground_truth_num": gt_num,
            "ground_truth_raw": item["answer"],
            "predicted_num": pred_num,
            "reasoning_content": res["reasoning_content"],
            "content": res["content"],
            "raw_text": f"<think>\n{res['reasoning_content']}\n</think>\n\n{res['content']}",
            "think_time_ms": res["think_time_ms"],
            "total_time_s": res["total_time_s"],
            "think_tokens": res["think_tokens"],
            "answer_tokens": res["answer_tokens"],
            "correct": is_correct
        }
        runs.append(record)
        print(f"      -> Pred: {pred_num} | Correct: {is_correct} | ThinkTime: {res['think_time_ms']:.1f}ms | TotTime: {res['total_time_s']:.2f}s | ThinkTok: {res['think_tokens']} | AnsTok: {res['answer_tokens']}")
        
    # Calculate statistics
    total = len(runs)
    correct_count = sum(1 for r in runs if r["correct"])
    gsm_runs = [r for r in runs if "gsm8k" in r["question_id"]]
    math_runs = [r for r in runs if "math500" in r["question_id"]]
    
    easy_runs = [r for r in runs if r["category"] == "Easy"]
    med_runs = [r for r in runs if r["category"] == "Medium"]
    hard_runs = [r for r in runs if r["category"] == "Hard"]
    
    summary = {
        "name": "Qwen 3.8 27B (SGLang FP4/BF16)",
        "overall_accuracy": round(correct_count / total * 100.0, 2) if total else 0.0,
        "gsm8k_accuracy": round(sum(1 for r in gsm_runs if r["correct"]) / len(gsm_runs) * 100.0, 2) if gsm_runs else 0.0,
        "gsm8k_easy_acc": round(sum(1 for r in easy_runs if r["correct"]) / len(easy_runs) * 100.0, 2) if easy_runs else 0.0,
        "gsm8k_med_acc": round(sum(1 for r in med_runs if r["correct"]) / len(med_runs) * 100.0, 2) if med_runs else 0.0,
        "gsm8k_hard_acc": round(sum(1 for r in hard_runs if r["correct"]) / len(hard_runs) * 100.0, 2) if hard_runs else 0.0,
        "math500_accuracy": round(sum(1 for r in math_runs if r["correct"]) / len(math_runs) * 100.0, 2) if math_runs else 0.0,
        "mean_think_time_ms": round(sum(r["think_time_ms"] for r in runs) / total, 2) if total else 0.0,
        "mean_total_time_s": round(sum(r["total_time_s"] for r in runs) / total, 3) if total else 0.0,
        "mean_think_tokens": round(sum(r["think_tokens"] for r in runs) / total, 1) if total else 0.0,
        "mean_answer_tokens": round(sum(r["answer_tokens"] for r in runs) / total, 1) if total else 0.0,
        "correct_count": correct_count,
        "total_count": total
    }
    
    output_data = {
        "metadata": {
            "model": MODEL_NAME,
            "backend": "SGLang",
            "endpoint": SGLANG_URL,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "summary": summary,
        "detailed_runs": runs
    }
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    print("\n" + "="*80)
    print("STAGE 1 EVALUATION SUMMARY (Qwen 3.8 27B)")
    print("="*80)
    print(f"Overall Accuracy: {summary['overall_accuracy']}% ({correct_count}/{total})")
    print(f"GSM8K Accuracy:   {summary['gsm8k_accuracy']}% ({sum(1 for r in gsm_runs if r['correct'])}/{len(gsm_runs)})")
    print(f"  - Easy:         {summary['gsm8k_easy_acc']}%")
    print(f"  - Medium:       {summary['gsm8k_med_acc']}%")
    print(f"  - Hard:         {summary['gsm8k_hard_acc']}%")
    print(f"MATH-500 Accuracy:{summary['math500_accuracy']}% ({sum(1 for r in math_runs if r['correct'])}/{len(math_runs)})")
    print(f"Mean Think Time:  {summary['mean_think_time_ms']:.1f} ms")
    print(f"Mean Total Time:  {summary['mean_total_time_s']:.2f} s")
    print(f"Mean Think Tok:   {summary['mean_think_tokens']}")
    print(f"Mean Answer Tok:  {summary['mean_answer_tokens']}")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
