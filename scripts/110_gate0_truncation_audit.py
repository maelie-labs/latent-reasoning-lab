#!/usr/bin/env python3
"""
scripts/110_gate0_truncation_audit.py
Gate 0 Truncation Audit for MATH-500 in Thinking Mode.

Evaluates models via SGLang high-throughput serving:
- Qwen3.5-4B: 81,920 token cap, official 3.5 sampler (temp=1.0, top_p=0.95, top_k=20, pp=1.5)
- Qwen3-4B: 32,768 token cap, official Qwen3 sampler (temp=0.6, top_p=0.95, top_k=20, pp=0.0)

Grader: Canonical math_verify 0.9.0 (CAS symbolic parser, 0 heuristic fallbacks).
Strict Truncation Rule: Hit max_tokens without emitting </think> or <|im_end|> => is_correct = False.

Thresholds:
- truncation <= 5.0%: PASS [VALID UNCONSTRAINED CEILING]
- 5.0% < truncation <= 15.0%: QUALIFIED PASS [CEILING VALID WITH STATED CAVEAT]
- truncation > 15.0%: FAIL [REPETITIVE LOOP ARTIFACT, CEILING INVALID]
"""

import os
import sys
import json
import time
import asyncio
import argparse
import numpy as np
from collections import defaultdict
from datasets import load_dataset
import httpx
from math_verify import parse, verify

THINK_END_TOKEN = "</think>"

def get_sampling_params(model_name, max_tokens):
    is_qwen35 = "3.5" in model_name.lower() or "3_5" in model_name.lower()
    if is_qwen35:
        temp = 1.0
        top_p = 0.95
        top_k = 20
        pres_pen = 1.5
    else:
        temp = 0.6
        top_p = 0.95
        top_k = 20
        pres_pen = 0.0
    return {
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": pres_pen,
        "max_new_tokens": max_tokens
    }

def format_prompt(problem_text):
    raw_q = problem_text.rstrip()
    if "\\boxed" not in raw_q:
        q = raw_q + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    else:
        q = raw_q
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"

async def query_model(client, server_url, prompt_text, sampling_params):
    payload = {
        "text": prompt_text,
        "sampling_params": sampling_params
    }
    t0 = time.time()
    try:
        resp = await client.post(f"{server_url}/generate", json=payload, timeout=1800.0)
        dur = time.time() - t0
        if resp.status_code != 200:
            return {
                "error": resp.text[:300],
                "dur": dur,
                "is_truncated": True,
                "completion_tokens": 0,
                "thinking_text": "",
                "answer_text": "",
                "raw_text": ""
            }
        data = resp.json()
        meta = data.get("meta_info", {})
        finish_reason = meta.get("finish_reason", {})
        fr_type = finish_reason.get("type", "") if isinstance(finish_reason, dict) else str(finish_reason)
        completion_tokens = meta.get("completion_tokens", 0)
        raw_text = data.get("text", "")

        is_truncated = (fr_type == "length")
        if THINK_END_TOKEN in raw_text:
            parts = raw_text.split(THINK_END_TOKEN, 1)
            thinking_text = parts[0].strip()
            answer_text = parts[1].strip()
        else:
            thinking_text = raw_text.strip()
            answer_text = ""
            is_truncated = True

        return {
            "dur": round(dur, 2),
            "completion_tokens": completion_tokens,
            "is_truncated": is_truncated,
            "finish_reason": fr_type,
            "thinking_text": thinking_text,
            "answer_text": answer_text,
            "raw_text": raw_text,
            "error": None
        }
    except Exception as e:
        return {
            "error": str(e),
            "dur": time.time() - t0,
            "is_truncated": True,
            "completion_tokens": 0,
            "thinking_text": "",
            "answer_text": "",
            "raw_text": ""
        }

def grade_solution(answer_text, gold_answer, gold_solution):
    gold_target = f"\\boxed{{{gold_answer}}}" if "\\boxed" not in gold_answer else gold_answer
    try:
        gold_parsed = parse(gold_target)
        if not gold_parsed:
            gold_parsed = parse(gold_solution)
        pred_parsed = parse(answer_text)
        if gold_parsed and pred_parsed:
            return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        pass
    return False

async def run_audit(server_url, model_id, max_tokens, concurrency, streaming_file):
    print(f"\n========================================================")
    print(f"=== Gate 0 Truncation Audit: MATH-500 (500 problems) ===")
    print(f"Model ID: {model_id}")
    print(f"Server URL: {server_url}")
    print(f"Max Tokens Cap: {max_tokens:,}")
    print(f"Concurrency: {concurrency}")
    sampling_params = get_sampling_params(model_id, max_tokens)
    print(f"Sampling Specification: {sampling_params}")
    print(f"Streaming Output: {streaming_file}")
    print(f"========================================================\n")

    print("Loading HuggingFaceH4/MATH-500 dataset (test split)...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = list(dataset)
    total_problems = len(problems)
    print(f"Loaded {total_problems} problems.")

    if streaming_file:
        os.makedirs(os.path.dirname(os.path.abspath(streaming_file)), exist_ok=True)
        with open(streaming_file, "w") as f_init:
            pass

    semaphore = asyncio.Semaphore(concurrency)
    results = []
    completed_count = 0
    t_start = time.time()

    async with httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)) as client:
        async def worker(idx, prob):
            nonlocal completed_count
            async with semaphore:
                prompt_text = format_prompt(prob["problem"])
                gold_answer = prob["answer"]
                gold_solution = prob["solution"]
                level = prob.get("level", 0)
                subject = prob.get("subject", "Unknown")
                uid = prob.get("unique_id", f"math500_{idx}")

                res = await query_model(client, server_url, prompt_text, sampling_params)

                is_correct = False
                if not res["is_truncated"] and res["answer_text"]:
                    is_correct = grade_solution(res["answer_text"], gold_answer, gold_solution)

                record = {
                    "idx": idx,
                    "unique_id": uid,
                    "level": level,
                    "subject": subject,
                    "dur": res["dur"],
                    "completion_tokens": res["completion_tokens"],
                    "is_truncated": res["is_truncated"],
                    "finish_reason": res.get("finish_reason"),
                    "is_correct": is_correct,
                    "gold_answer": gold_answer,
                    "error": res.get("error"),
                    "prompt": prompt_text,
                    "thinking_snippet": res["thinking_text"][:200] if res["thinking_text"] else "",
                    "answer_text": res["answer_text"][:500] if res["answer_text"] else ""
                }
                results.append(record)
                completed_count += 1

                if streaming_file:
                    with open(streaming_file, "a") as f_stream:
                        f_stream.write(json.dumps(record) + "\n")

                if completed_count % 10 == 0 or completed_count == total_problems:
                    acc = np.mean([r["is_correct"] for r in results]) * 100.0
                    trunc = np.mean([r["is_truncated"] for r in results]) * 100.0
                    toks = [r["completion_tokens"] for r in results if r["completion_tokens"] > 0]
                    med_tok = np.median(toks) if toks else 0
                    elapsed = time.time() - t_start
                    qps = completed_count / elapsed
                    print(f"[{completed_count:3d}/{total_problems}] Acc: {acc:5.2f}% | Truncation: {trunc:5.2f}% | Median Tok: {med_tok:.0f} | QPS: {qps:.2f} ({elapsed:.1f}s)")

        tasks = [worker(i, p) for i, p in enumerate(problems)]
        await asyncio.gather(*tasks)

    # Compile Final Comprehensive Audit Report
    tokens_list = [r["completion_tokens"] for r in results]
    trunc_count = sum(1 for r in results if r["is_truncated"])
    trunc_rate = (trunc_count / total_problems) * 100.0
    correct_count = sum(1 for r in results if r["is_correct"])
    pass_rate = (correct_count / total_problems) * 100.0

    # Paired Bootstrap 95% CI on Accuracy
    rng = np.random.default_rng(42)
    acc_array = np.array([1.0 if r["is_correct"] else 0.0 for r in results])
    boot_means = [np.mean(rng.choice(acc_array, size=len(acc_array), replace=True)) * 100.0 for _ in range(10000)]
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    # Stratified by Level
    by_level = defaultdict(list)
    for r in results:
        by_level[r["level"]].append(r)
    level_stats = {}
    for lvl in sorted(by_level.keys()):
        l_recs = by_level[lvl]
        l_acc = np.mean([r["is_correct"] for r in l_recs]) * 100.0
        l_trunc = np.mean([r["is_truncated"] for r in l_recs]) * 100.0
        l_toks = [r["completion_tokens"] for r in l_recs]
        level_stats[f"level_{lvl}"] = {
            "n": len(l_recs),
            "accuracy_pct": round(l_acc, 2),
            "truncation_pct": round(l_trunc, 2),
            "median_tokens": round(float(np.median(l_toks)), 1),
            "p90_tokens": round(float(np.percentile(l_toks, 90)), 1)
        }

    # Stratified by Subject
    by_subj = defaultdict(list)
    for r in results:
        by_subj[r["subject"]].append(r)
    subj_stats = {}
    for s in sorted(by_subj.keys()):
        s_recs = by_subj[s]
        s_acc = np.mean([r["is_correct"] for r in s_recs]) * 100.0
        s_trunc = np.mean([r["is_truncated"] for r in s_recs]) * 100.0
        subj_stats[s] = {
            "n": len(s_recs),
            "accuracy_pct": round(s_acc, 2),
            "truncation_pct": round(s_trunc, 2)
        }

    # Verdict
    if trunc_rate <= 5.0:
        verdict = "PASS [VALID UNCONSTRAINED CEILING]"
    elif trunc_rate <= 15.0:
        verdict = "QUALIFIED PASS [CEILING VALID WITH STATED CAVEAT]"
    else:
        verdict = "FAIL [REPETITIVE LOOP ARTIFACT, CEILING INVALID]"

    report = {
        "benchmark": "MATH-500",
        "model_id": model_id,
        "server_url": server_url,
        "max_tokens_cap": max_tokens,
        "concurrency": concurrency,
        "sampling_params": sampling_params,
        "total_problems": total_problems,
        "verdict": verdict,
        "metrics": {
            "accuracy_pct": round(pass_rate, 2),
            "ci_95_low": round(ci_low, 2),
            "ci_95_high": round(ci_high, 2),
            "truncation_count": trunc_count,
            "truncation_rate_pct": round(trunc_rate, 2),
            "total_duration_s": round(time.time() - t_start, 2),
            "mean_duration_s": round(float(np.mean([r['dur'] for r in results])), 2)
        },
        "token_distribution": {
            "min": int(np.min(tokens_list)),
            "p25": round(float(np.percentile(tokens_list, 25)), 1),
            "median": round(float(np.median(tokens_list)), 1),
            "p75": round(float(np.percentile(tokens_list, 75)), 1),
            "p90": round(float(np.percentile(tokens_list, 90)), 1),
            "p99": round(float(np.percentile(tokens_list, 99)), 1),
            "max": int(np.max(tokens_list))
        },
        "level_breakdown": level_stats,
        "subject_breakdown": subj_stats,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    return report

def main():
    parser = argparse.ArgumentParser(description="Gate 0 Truncation Audit Runner")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max_tokens", type=int, default=81920)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--streaming_file", type=str, default=None)
    args = parser.parse_args()

    if args.output_file is None:
        safe_name = args.model_id.replace("/", "_").replace(".", "_").lower()
        args.output_file = f"data/gate0_math500_{safe_name}.json"
    if args.streaming_file is None:
        safe_name = args.model_id.replace("/", "_").replace(".", "_").lower()
        args.streaming_file = f"data/streaming_gate0_math500_{safe_name}.jsonl"

    report = asyncio.run(run_audit(
        server_url=args.server_url,
        model_id=args.model_id,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        streaming_file=args.streaming_file
    ))

    with open(args.output_file, "w") as f_out:
        json.dump(report, f_out, indent=2)

    print("\n========================================================")
    print(f"=== Gate 0 Final Verdict: {report['verdict']} ===")
    print(f"Truncation Rate: {report['metrics']['truncation_rate_pct']}% ({report['metrics']['truncation_count']}/{report['total_problems']})")
    print(f"Pass@1 Accuracy: {report['metrics']['accuracy_pct']}% (95% CI: [{report['metrics']['ci_95_low']}%, {report['metrics']['ci_95_high']}%])")
    print(f"Token Lengths: Median={report['token_distribution']['median']} | P90={report['token_distribution']['p90']} | Max={report['token_distribution']['max']}")
    print(f"Saved full audit report to: {args.output_file}")
    print("========================================================\n")

if __name__ == "__main__":
    main()
