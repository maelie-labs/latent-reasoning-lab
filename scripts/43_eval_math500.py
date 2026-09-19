#!/usr/bin/env python3
"""
scripts/43_eval_math500.py
Standalone Modular Benchmark Runner for MATH-500 (500 problems).

Evaluates models via SGLang server using official sampling parameters:
- Thinking Mode: enable_thinking=True, temp=0.6, top_p=0.95, top_k=20, max_tokens=32768
- Non-Thinking Mode: enable_thinking=False, temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5, max_tokens=2048

Uses math_verify (CAS / SymPy algebraic equivalence) as the primary evaluator
with regex fallback. Computes 95% bootstrap confidence intervals (B=10,000).
"""

import os
import re
import json
import time
import asyncio
import argparse
import numpy as np
from datasets import load_dataset
import httpx
from math_verify import parse, verify

THINK_END_TOKEN = "</think>"

def extract_math_boxed_expression(text):
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

def clean_and_extract_candidate(cand):
    if not cand:
        return None
    cand = re.sub(r"\\(?:text|mathbf|mathrm)\{([^}]+)\}", r"\1", str(cand))
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

def check_math500_match(pred_text, gold_text):
    # 1. Primary: math_verify symbolic parser
    try:
        gold_parsed = parse(gold_text, parsing_timeout=None)
        pred_parsed = parse(pred_text, parsing_timeout=None)
        if gold_parsed and pred_parsed:
            if verify(gold_parsed, pred_parsed):
                return True
    except Exception:
        pass

    # 2. Secondary fallback: Boxed / numeric normalization
    pred_boxed = extract_math_boxed_expression(pred_text)
    gold_boxed = extract_math_boxed_expression(gold_text) or gold_text

    pred_val = clean_and_extract_candidate(pred_boxed or pred_text)
    gold_val = clean_and_extract_candidate(gold_boxed)

    if pred_val is not None and gold_val is not None:
        if abs(pred_val - gold_val) < 1e-4:
            return True

    if pred_boxed and gold_boxed:
        c_pred = re.sub(r"[\$\\%!\s]", "", pred_boxed).replace(",", "").strip().lower()
        c_gold = re.sub(r"[\$\\%!\s]", "", gold_boxed).replace(",", "").strip().lower()
        if c_pred == c_gold:
            return True

    return False

def get_sampling_params(model_name, enable_thinking, max_tokens):
    is_qwen35 = "3.5" in model_name.lower() or "3_5" in model_name.lower()
    if is_qwen35:
        temp = 1.0
        top_p = 0.95
        top_k = 20
        pres_pen = 1.5
    else:
        temp = 0.6 if enable_thinking else 0.7
        top_p = 0.95 if enable_thinking else 0.80
        top_k = 20
        pres_pen = 0.0 if enable_thinking else 1.5
    actual_max = max_tokens if enable_thinking else min(max_tokens, 8192)
    return {
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": pres_pen,
        "max_new_tokens": actual_max
    }

def format_prompt(raw_prompt, model_name, enable_thinking):
    if enable_thinking:
        return f"<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    else:
        return f"<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

async def query_model(client, server_url, prompt, enable_thinking, max_tokens, model_name):
    sampling_params = get_sampling_params(model_name, enable_thinking, max_tokens)
    prompt_text = format_prompt(prompt, model_name, enable_thinking)
    payload = {
        "text": prompt_text,
        "sampling_params": sampling_params
    }
    t0 = time.time()
    try:
        resp = await client.post(f"{server_url}/generate", json=payload, timeout=600.0)
        dur = time.time() - t0
        if resp.status_code != 200:
            return {"error": resp.text, "dur": dur, "is_truncated": True, "thinking_text": "", "content_text": ""}
        data = resp.json()
        meta = data.get("meta_info", {})
        finish_reason = meta.get("finish_reason", {})
        fr_type = finish_reason.get("type", "") if isinstance(finish_reason, dict) else str(finish_reason)
        is_truncated = (fr_type == "length")
        completion_tokens = meta.get("completion_tokens", 0)
        text = data.get("text", "")
        
        if enable_thinking:
            if THINK_END_TOKEN in text:
                parts = text.split(THINK_END_TOKEN, 1)
                thinking_text = parts[0].strip()
                content_text = parts[1].strip()
            else:
                thinking_text = text.strip()
                content_text = ""
        else:
            thinking_text = ""
            content_text = text.strip()
            
        return {
            "dur": round(dur, 2),
            "completion_tokens": completion_tokens,
            "is_truncated": is_truncated,
            "thinking_text": thinking_text,
            "content_text": content_text
        }
    except Exception as e:
        return {"error": str(e), "dur": time.time() - t0, "is_truncated": True, "thinking_text": "", "content_text": ""}

async def run_evaluation(server_url, problems, enable_thinking, concurrency, model_name, max_tokens=32768, jsonl_stream_path=None):
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    if jsonl_stream_path:
        os.makedirs(os.path.dirname(os.path.abspath(jsonl_stream_path)), exist_ok=True)
        with open(jsonl_stream_path, "w") as f_init:
            pass

    async with httpx.AsyncClient() as client:
        async def worker(idx, prob):
            async with semaphore:
                raw_q = prob["question"]
                gold_sol = prob["solution"]
                # Append standard Qwen mathematical reasoning instruction if not already present
                if "\\boxed" not in raw_q:
                    q = raw_q.rstrip() + "\nPlease reason step by step, and put your final answer within \\boxed{}."
                else:
                    q = raw_q

                try:
                    res = await query_model(client, server_url, q, enable_thinking, max_tokens, model_name)
                    is_correct = False
                    if not res.get("is_truncated", False):
                        is_correct = check_math500_match(res.get("content_text", ""), gold_sol)
                    res["is_correct"] = is_correct
                    res["id"] = prob["id"]
                    results.append(res)

                    if jsonl_stream_path:
                        rec = {
                            "id": prob["id"],
                            "raw_question": raw_q,
                            "prompt": q,
                            "solution": gold_sol,
                            "is_correct": is_correct,
                            "is_truncated": res.get("is_truncated", False),
                            "dur": res.get("dur", 0.0),
                            "completion_tokens": res.get("completion_tokens", 0),
                            "thinking_text": res.get("thinking_text", ""),
                            "content_text": res.get("content_text", "")
                        }
                        with open(jsonl_stream_path, "a") as f_stream:
                            f_stream.write(json.dumps(rec) + "\n")
                except Exception as e:
                    err_res = {"id": prob["id"], "is_correct": False, "is_truncated": True, "error": str(e)}
                    results.append(err_res)
                    if jsonl_stream_path:
                        rec = {
                            "id": prob["id"],
                            "raw_question": raw_q,
                            "prompt": q,
                            "solution": gold_sol,
                            "is_correct": False,
                            "is_truncated": True,
                            "error": str(e)
                        }
                        with open(jsonl_stream_path, "a") as f_stream:
                            f_stream.write(json.dumps(rec) + "\n")

                if len(results) % 25 == 0 or len(results) == len(problems):
                    acc = np.mean([r["is_correct"] for r in results]) * 100.0
                    trunc = np.mean([r.get("is_truncated", False) for r in results]) * 100.0
                    print(f"  [{len(results)}/{len(problems)}] MATH-500 Accuracy: {acc:.2f}% | Truncation: {trunc:.2f}%")

        tasks = [worker(i, p) for i, p in enumerate(problems)]
        await asyncio.gather(*tasks)

    return results

def bootstrap_ci(results, n_boot=10000):
    scores = np.array([1.0 if r.get("is_correct", False) else 0.0 for r in results])
    n = len(scores)
    boot_means = [np.mean(scores[np.random.choice(n, size=n, replace=True)]) for _ in range(n_boot)]
    return {
        "mean_pct": round(float(np.mean(scores) * 100.0), 2),
        "ci_95_low": round(float(np.percentile(boot_means, 2.5) * 100.0), 2),
        "ci_95_high": round(float(np.percentile(boot_means, 97.5) * 100.0), 2),
        "total_evaluated": n,
        "correct_count": int(np.sum(scores))
    }

def main():
    parser = argparse.ArgumentParser(description="Standalone MATH-500 Benchmark Runner")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30001", help="SGLang server endpoint")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mode", type=str, choices=["thinking", "nothinking", "both"], default="thinking")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max output tokens (default: 32768 for both modes per Qwen official eval)")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    # Discover model from server
    model_name = args.model_id
    try:
        r = httpx.get(f"{args.server_url}/v1/models", timeout=5.0)
        if r.status_code == 200:
            m_data = r.json()
            if "data" in m_data and len(m_data["data"]) > 0:
                model_name = m_data["data"][0]["id"]
                print(f"Discovered server model: '{model_name}'")
    except Exception as e:
        print(f"Notice: Could not query /v1/models ({e}), using '{model_name}'")

    print(f"Loading MATH-500 test dataset...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [{"id": f"math500_{i}", "question": ex["problem"], "solution": ex["solution"]} for i, ex in enumerate(ds)]
    if args.max_problems:
        problems = problems[:args.max_problems]
    print(f"Loaded {len(problems)} MATH-500 problems.")

    modes_to_run = ["thinking"] if args.mode == "thinking" else (["nothinking"] if args.mode == "nothinking" else ["thinking", "nothinking"])

    report = {
        "benchmark": "MATH-500",
        "model_id": args.model_id,
        "server_model_name": model_name,
        "server_url": args.server_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": {}
    }

    for m in modes_to_run:
        is_think = (m == "thinking")
        print(f"\n" + "="*60)
        print(f"EVALUATING MATH-500: {m.upper()} MODE (concurrency={args.concurrency}, max_tokens={args.max_tokens})")
        print("="*60)
        tag = args.model_id.replace("/", "_").lower()
        stream_path = os.path.join(os.path.dirname(os.path.abspath(args.output_file or "data/eval.json")), f"streaming_math500_{tag}_{m}.jsonl")
        print(f"Streaming outputs to: {stream_path}")
        t0 = time.time()
        results = asyncio.run(run_evaluation(args.server_url, problems, is_think, args.concurrency, model_name, max_tokens=args.max_tokens, jsonl_stream_path=stream_path))
        dur = time.time() - t0
        stats = bootstrap_ci(results)
        trunc_rate = np.mean([r.get("is_truncated", False) for r in results]) * 100.0

        print(f"\n--- MATH-500 {m.upper()} Summary ({dur:.1f}s) ---")
        print(f"Pass@1: {stats['mean_pct']}% (95% CI: [{stats['ci_95_low']}%, {stats['ci_95_high']}%])")
        print(f"Correct: {stats['correct_count']} / {stats['total_evaluated']}")
        print(f"Truncation: {trunc_rate:.2f}%")

        report["runs"][m] = {
            "duration_s": round(dur, 1),
            "stats": stats,
            "truncation_pct": round(trunc_rate, 2),
            "results": results
        }

    out_file = args.output_file
    if not out_file:
        tag = args.model_id.replace("/", "_").lower()
        out_file = os.path.join(os.path.dirname(__file__), "..", "data", f"eval_math500_{tag}_{args.mode}.json")

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved MATH-500 report to: {out_file}")

if __name__ == "__main__":
    main()
