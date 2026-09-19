#!/usr/bin/env python3
"""
scripts/45_eval_gpqa.py
Official Shared Ground-Truth Anchor Runner: GPQA Diamond (198 problems x 8 samples).

Official Published Reference Numbers:
- Qwen3-1.7B: 40.1% (Thinking Mode, multi-sample average)
- Qwen3.5-2B: 51.6% (Thinking Mode)

Evaluation Invariants:
1. Qwen's official MCQ JSON format: {"answer": "LETTER"}.
2. 8 samples per problem with pseudo-random option shuffling using deterministic fixed seeds
   to eliminate option-position bias and balance A/B/C/D distribution.
3. Per-domain logging (Physics, Chemistry, Biology) and cluster bootstrap 95% CI.
4. Native sampling parameters per model family:
   - Qwen3-1.7B: temp=0.6, top_p=0.95, top_k=20, presence_penalty=0.0 (thinking)
                 temp=0.7, top_p=0.80, top_k=20, presence_penalty=1.5 (non-thinking)
   - Qwen3.5-2B: temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5 (thinking)
                 temp=1.0, top_p=1.0, top_k=20, presence_penalty=2.0 (non-thinking)
"""

import os
import re
import json
import time
import random
import asyncio
import argparse
from collections import defaultdict
import numpy as np
from datasets import load_dataset
import httpx

THINK_END_TOKEN = "</think>"
LETTERS = ["A", "B", "C", "D"]

def extract_mcq_letter(content_text):
    """
    Extract multiple choice answer letter (A, B, C, or D) from model response.
    Prioritizes Qwen JSON format {"answer": "LETTER"}, with fallbacks for boxed and raw letters.
    """
    if not content_text:
        return None

    # 1. Look for JSON format: "answer": "X" or 'answer': 'X'
    m = re.search(r"\"answer\"\s*:\s*\"?([A-Da-d])\"?", content_text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\x27answer\x27\s*:\s*\x27?([A-Da-d])\x27?", content_text)
    if m:
        return m.group(1).upper()

    # 2. Try JSON parsing of last lines / code blocks
    for line in reversed(content_text.strip().splitlines()):
        line = line.strip()
        if "{" in line and "}" in line:
            try:
                sub = line[line.find("{"):line.rfind("}")+1]
                obj = json.loads(sub)
                if "answer" in obj and str(obj["answer"]).strip().upper() in LETTERS:
                    return str(obj["answer"]).strip().upper()
            except Exception:
                pass

    # 3. Fallback: \boxed{X}
    m = re.search(r"\\boxed\{([A-Da-d])\}", content_text)
    if m:
        return m.group(1).upper()

    # 4. Fallback: Search last line for standalone letter
    last_line = content_text.strip().splitlines()[-1] if content_text.strip() else ""
    m = re.search(r"\b([A-Da-d])\b", last_line)
    if m:
        return m.group(1).upper()

    # 5. Fallback: Last occurrence of letter option in text
    all_letters = re.findall(r"\b([A-Da-d])\b", content_text)
    if all_letters:
        return all_letters[-1].upper()

    return None

def load_gpqa_diamond_dataset():
    """
    Load canonical 198 GPQA Diamond problems with options and domain metadata.
    Uses hendrydong/gpqa_diamond matched against ankner/gpqa.
    """
    print("Loading GPQA Diamond dataset...")
    ds_diamond = load_dataset("hendrydong/gpqa_diamond", split="test")
    ds_ankner = load_dataset("ankner/gpqa", split="train")

    ankner_by_q = {r["Question"].strip(): r for r in ds_ankner}

    problems = []
    for i, r in enumerate(ds_diamond):
        raw_q = r["problem"].strip()
        match = ankner_by_q.get(raw_q)
        if not match:
            for aq, ar in ankner_by_q.items():
                if raw_q[:50] in aq or aq[:50] in raw_q:
                    match = ar
                    break
        assert match is not None, f"Could not match problem {i}: {raw_q[:50]}"

        problems.append({
            "problem_idx": i,
            "id": f"gpqa_diamond_{i:03d}",
            "question": match["Question"].strip(),
            "correct_answer": match["Correct Answer"].strip(),
            "incorrect_1": match["Incorrect Answer 1"].strip(),
            "incorrect_2": match["Incorrect Answer 2"].strip(),
            "incorrect_3": match["Incorrect Answer 3"].strip(),
            "domain": match.get("High-level domain", "Unknown"),
            "subdomain": match.get("Subdomain", "Unknown")
        })

    domain_counts = defaultdict(int)
    for p in problems:
        domain_counts[p["domain"]] += 1
    print(f"Loaded {len(problems)} GPQA Diamond problems: {dict(domain_counts)}")
    return problems

def build_shuffled_sample(problem, sample_idx, base_seed=42):
    """
    Build a deterministically shuffled MCQ sample.
    Returns: prompt, gold_letter, letter_to_choice_map
    """
    rng = random.Random(base_seed + problem["problem_idx"] * 1000 + sample_idx)
    raw_choices = [
        ("correct", problem["correct_answer"]),
        ("incorrect_1", problem["incorrect_1"]),
        ("incorrect_2", problem["incorrect_2"]),
        ("incorrect_3", problem["incorrect_3"]),
    ]
    rng.shuffle(raw_choices)

    options_text_lines = []
    gold_letter = None
    choice_map = {}
    for letter, (opt_type, text) in zip(LETTERS, raw_choices):
        if opt_type == "correct":
            gold_letter = letter
        choice_map[letter] = text
        options_text_lines.append(f"({letter}) {text}")

    options_block = "\n".join(options_text_lines)
    prompt = (
        f"Answer the following multiple choice question. The last line of your response should be of the following format: '{{\"answer\": \"$LETTER\"}}' (without quotes) where LETTER is one of A, B, C, or D. Think step by step before answering.\n\n"
        f"{problem['question']}\n\n"
        f"{options_block}"
    )
    return prompt, gold_letter, choice_map

def format_gpqa_prompt(raw_prompt, enable_thinking):
    if enable_thinking:
        return f"<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    else:
        return f"<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

async def query_model(client, server_url, prompt, enable_thinking, max_tokens, model_name, temp, top_p, top_k, pres_pen):
    prompt_text = format_gpqa_prompt(prompt, enable_thinking)
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": pres_pen,
            "max_new_tokens": max_tokens
        }
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

async def run_gpqa_evaluation(
    server_url,
    problems,
    enable_thinking,
    concurrency,
    model_name,
    temp,
    top_p,
    top_k,
    pres_pen,
    num_samples=8,
    max_tokens=32768,
    jsonl_stream_path=None
):
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    if jsonl_stream_path:
        os.makedirs(os.path.dirname(os.path.abspath(jsonl_stream_path)), exist_ok=True)
        with open(jsonl_stream_path, "w") as f_init:
            pass

    # Build all evaluation queries: 198 problems x num_samples
    eval_items = []
    for prob in problems:
        for s_idx in range(num_samples):
            prompt, gold_letter, choice_map = build_shuffled_sample(prob, s_idx)
            eval_items.append({
                "problem_idx": prob["problem_idx"],
                "prob_id": prob["id"],
                "sample_idx": s_idx,
                "domain": prob["domain"],
                "subdomain": prob["subdomain"],
                "question": prob["question"],
                "prompt": prompt,
                "gold_letter": gold_letter,
                "choice_map": choice_map
            })

    total_queries = len(eval_items)
    print(f"Generated {total_queries} evaluation tasks ({len(problems)} problems x {num_samples} samples).")

    async with httpx.AsyncClient() as client:
        async def worker(item):
            async with semaphore:
                try:
                    res = await query_model(
                        client, server_url, item["prompt"], enable_thinking,
                        max_tokens, model_name, temp, top_p, top_k, pres_pen
                    )
                    pred_letter = None
                    is_correct = False
                    if not res.get("is_truncated", False):
                        pred_letter = extract_mcq_letter(res.get("content_text", ""))
                        is_correct = (pred_letter == item["gold_letter"]) if pred_letter else False

                    res_record = {
                        "problem_idx": item["problem_idx"],
                        "id": item["prob_id"],
                        "sample_idx": item["sample_idx"],
                        "domain": item["domain"],
                        "subdomain": item["subdomain"],
                        "gold_letter": item["gold_letter"],
                        "pred_letter": pred_letter,
                        "is_correct": is_correct,
                        "is_truncated": res.get("is_truncated", False),
                        "dur": res.get("dur", 0.0),
                        "completion_tokens": res.get("completion_tokens", 0),
                        "thinking_text": res.get("thinking_text", ""),
                        "content_text": res.get("content_text", "")
                    }
                    results.append(res_record)

                    if jsonl_stream_path:
                        with open(jsonl_stream_path, "a") as f_stream:
                            f_stream.write(json.dumps(res_record) + "\n")

                except Exception as e:
                    err_record = {
                        "problem_idx": item["problem_idx"],
                        "id": item["prob_id"],
                        "sample_idx": item["sample_idx"],
                        "domain": item["domain"],
                        "subdomain": item["subdomain"],
                        "gold_letter": item["gold_letter"],
                        "pred_letter": None,
                        "is_correct": False,
                        "is_truncated": True,
                        "error": str(e)
                    }
                    results.append(err_record)
                    if jsonl_stream_path:
                        with open(jsonl_stream_path, "a") as f_stream:
                            f_stream.write(json.dumps(err_record) + "\n")

                if len(results) % 50 == 0 or len(results) == total_queries:
                    acc = np.mean([r["is_correct"] for r in results]) * 100.0
                    trunc = np.mean([r.get("is_truncated", False) for r in results]) * 100.0
                    print(f"  [{len(results)}/{total_queries}] GPQA Multi-Sample Acc: {acc:.2f}% | Truncation: {trunc:.2f}%")

        tasks = [worker(item) for item in eval_items]
        await asyncio.gather(*tasks)

    return results

def compute_cluster_bootstrap_stats(results, num_problems=198, num_boot=10000):
    """
    Compute overall pass@1, problem-level mean accuracy, majority voting,
    per-domain metrics, and 95% cluster bootstrap confidence intervals.
    """
    # Group results by problem_idx
    by_prob = defaultdict(list)
    for r in results:
        by_prob[r["problem_idx"]].append(r)

    prob_accs = []
    prob_majority = []
    prob_domains = []

    domain_prob_accs = defaultdict(list)
    domain_sample_correct = defaultdict(list)

    for p_idx in sorted(by_prob.keys()):
        p_res = by_prob[p_idx]
        acc_p = np.mean([1.0 if r["is_correct"] else 0.0 for r in p_res])
        prob_accs.append(acc_p)

        # Majority vote
        votes = defaultdict(int)
        for r in p_res:
            if r["pred_letter"]:
                votes[r["pred_letter"]] += 1
        gold = p_res[0]["gold_letter"]
        if votes:
            top_choice = max(votes.items(), key=lambda x: x[1])[0]
            prob_majority.append(1.0 if top_choice == gold else 0.0)
        else:
            prob_majority.append(0.0)

        dom = p_res[0]["domain"]
        prob_domains.append(dom)
        domain_prob_accs[dom].append(acc_p)
        for r in p_res:
            domain_sample_correct[dom].append(1.0 if r["is_correct"] else 0.0)

    prob_accs = np.array(prob_accs)
    mean_pass_rate = float(np.mean(prob_accs) * 100.0)
    mean_maj_rate = float(np.mean(prob_majority) * 100.0)

    # Cluster bootstrap across problems
    n = len(prob_accs)
    boot_means = [np.mean(prob_accs[np.random.choice(n, size=n, replace=True)]) for _ in range(num_boot)]
    ci_low = float(np.percentile(boot_means, 2.5) * 100.0)
    ci_high = float(np.percentile(boot_means, 97.5) * 100.0)

    domain_summary = {}
    for dom in sorted(domain_prob_accs.keys()):
        p_accs = np.array(domain_prob_accs[dom])
        s_accs = np.array(domain_sample_correct[dom])
        domain_summary[dom] = {
            "num_problems": len(p_accs),
            "total_samples": len(s_accs),
            "sample_pass_rate": round(float(np.mean(s_accs) * 100.0), 2),
            "problem_mean_acc": round(float(np.mean(p_accs) * 100.0), 2)
        }

    return {
        "pass_rate_pct": round(mean_pass_rate, 2),
        "ci_95_low": round(ci_low, 2),
        "ci_95_high": round(ci_high, 2),
        "majority_vote_pct": round(mean_maj_rate, 2),
        "total_problems": n,
        "total_samples": len(results),
        "domains": domain_summary
    }

def main():
    parser = argparse.ArgumentParser(description="GPQA Diamond Shared Ground-Truth Anchor Runner")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mode", type=str, choices=["thinking", "nothinking", "both"], default="thinking")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples per problem with shuffled options")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    # Discover model name from server
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

    is_qwen35 = "3.5" in args.model_id.lower() or "3_5" in args.model_id.lower()
    problems = load_gpqa_diamond_dataset()

    modes_to_run = ["thinking"] if args.mode == "thinking" else (["nothinking"] if args.mode == "nothinking" else ["thinking", "nothinking"])

    report = {
        "benchmark": "GPQA-Diamond",
        "model_id": args.model_id,
        "server_model_name": model_name,
        "server_url": args.server_url,
        "num_samples_per_problem": args.num_samples,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": {}
    }

    for m in modes_to_run:
        is_think = (m == "thinking")
        if is_qwen35:
            # Qwen3.5-2B official settings: temp=1.0, top_p=0.95, top_k=20, pp=1.5 for BOTH modes
            temp = 1.0
            top_p = 0.95
            top_k = 20
            pres_pen = 1.5
        else:
            # Qwen3-1.7B official settings
            temp = 0.6 if is_think else 0.7
            top_p = 0.95 if is_think else 0.80
            top_k = 20
            pres_pen = 0.0 if is_think else 1.5

        print(f"\n" + "="*65)
        print(f"EVALUATING GPQA DIAMOND: {m.upper()} MODE ({args.num_samples} samples/problem)")
        print(f"Params: temp={temp}, top_p={top_p}, top_k={top_k}, pres_pen={pres_pen}")
        print("="*65)

        tag = args.model_id.replace("/", "_").lower()
        stream_path = os.path.join(
            os.path.dirname(os.path.abspath(args.output_file or "data/eval.json")),
            f"streaming_gpqa_{tag}_{m}.jsonl"
        )
        print(f"Streaming outputs to: {stream_path}")

        t0 = time.time()
        results = asyncio.run(run_gpqa_evaluation(
            args.server_url, problems, is_think, args.concurrency,
            model_name, temp, top_p, top_k, pres_pen,
            num_samples=args.num_samples, max_tokens=args.max_tokens,
            jsonl_stream_path=stream_path
        ))
        dur = time.time() - t0

        stats = compute_cluster_bootstrap_stats(results, num_problems=len(problems))
        trunc_rate = np.mean([r.get("is_truncated", False) for r in results]) * 100.0

        print(f"\n--- GPQA DIAMOND {m.upper()} Results ({dur:.1f}s) ---")
        print(f"Pass@1: {stats['pass_rate_pct']}% (95% CI: [{stats['ci_95_low']}%, {stats['ci_95_high']}%])")
        print(f"Majority Vote (Consensus): {stats['majority_vote_pct']}%")
        print(f"Truncation: {trunc_rate:.2f}%")
        print(f"Domain Breakdown:")
        for dom, d_data in stats["domains"].items():
            print(f"  - {dom:<12}: Pass Rate: {d_data['sample_pass_rate']}% | Mean Acc: {d_data['problem_mean_acc']}% (N={d_data['num_problems']})")

        report["runs"][m] = {
            "duration_s": round(dur, 1),
            "sampling_params": {
                "temp": temp, "top_p": top_p, "top_k": top_k,
                "presence_penalty": pres_pen, "max_tokens": args.max_tokens,
                "num_samples": args.num_samples
            },
            "stats": stats,
            "truncation_pct": round(trunc_rate, 2),
            "results": results
        }

    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f_out:
            json.dump(report, f_out, indent=2)
        print(f"Saved complete GPQA Diamond report to: {args.output_file}")

if __name__ == "__main__":
    main()
