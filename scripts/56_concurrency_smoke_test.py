#!/usr/bin/env python3
"""
scripts/56_concurrency_smoke_test.py
Automated Concurrency Smoke Test & Calibration Tool.

POLICY REQUIREMENT:
Before running full-scale benchmarks on any new model architecture, run this quick (~30-60s)
smoke test to determine the optimal concurrency sweet spot.
The calibrated concurrency must be held CONSTANT across all benchmark sets that will be
directly compared (e.g. GSM8K vs MATH-500, or Qwen3-1.7B vs Qwen3.5-2B, or Arm 1 vs Arm 4).
"""

import os
import sys
import json
import time
import asyncio
import argparse
import numpy as np
import httpx

BENCHMARK_PROMPTS = [
    # GSM8K style
    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars will she make every day?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 into repairs. He sells it for $150,000. What is his net profit?",
    "James decides to run 3 miles every day. How many miles does he run in 2 weeks?",
    # MATH style
    "Compute the sum of all positive integers n such that n^2 + 2n + 1 divides n^3 + 3n^2 + 3n + 1.",
    "Let f(x) = 2x^3 - 5x^2 + 4x - 1. Find the sum of the roots of f(x) = 0.",
    "A regular hexagon has perimeter 36. What is its area in simplest radical form?",
    # GPQA / Science style
    "What is the change in entropy when one mole of an ideal gas expands isothermally from a volume of 10 L to 20 L at 300 K?",
]

async def send_req(client, server_url, model_name, prompt, enable_thinking, max_tokens):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.6 if enable_thinking else 0.7,
        "top_p": 0.95 if enable_thinking else 0.8,
        "presence_penalty": 0.0 if enable_thinking else 1.5,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "top_k": 20,
            "presence_penalty": 0.0 if enable_thinking else 1.5
        }
    }
    t0 = time.time()
    try:
        resp = await client.post(f"{server_url}/v1/chat/completions", json=payload, timeout=120.0)
        dur = time.time() - t0
        if resp.status_code != 200:
            return {"error": resp.text[:200], "dur": dur, "tokens": 0}
        data = resp.json()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return {"dur": dur, "tokens": tokens, "error": None}
    except Exception as e:
        return {"error": str(e), "dur": time.time() - t0, "tokens": 0}

async def run_sweep_step(server_url, model_name, concurrency, enable_thinking, num_requests):
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency + 10)
    
    async with httpx.AsyncClient(limits=limits) as client:
        async def worker(idx):
            prompt = BENCHMARK_PROMPTS[idx % len(BENCHMARK_PROMPTS)]
            max_tokens = 1024 if enable_thinking else 256
            async with semaphore:
                return await send_req(client, server_url, model_name, prompt, enable_thinking, max_tokens)
                
        t_start = time.time()
        tasks = [worker(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        wall_clock = time.time() - t_start
        
    valid_results = [r for r in results if not r.get("error")]
    errors = len(results) - len(valid_results)
    total_tokens = sum(r["tokens"] for r in valid_results)
    throughput = total_tokens / wall_clock if wall_clock > 0 else 0
    durations = [r["dur"] for r in valid_results]
    mean_lat = float(np.mean(durations)) if durations else 0.0
    p95_lat = float(np.percentile(durations, 95)) if durations else 0.0
    
    return {
        "concurrency": concurrency,
        "throughput_tok_s": round(throughput, 1),
        "wall_clock_s": round(wall_clock, 2),
        "mean_latency_s": round(mean_lat, 2),
        "p95_latency_s": round(p95_lat, 2),
        "total_tokens": total_tokens,
        "errors": errors
    }

async def main_async(args):
    print(f"===============================================================")
    print(f"   CONCURRENCY SMOKE TEST & CALIBRATION POLICY HARNESS         ")
    print(f"===============================================================")
    print(f"Target Server : {args.server_url}")
    
    # Auto-detect model if not passed
    model_name = args.model_id
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{args.server_url}/v1/models", timeout=5.0)
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    model_name = data[0]["id"]
                    print(f"Detected Model: {model_name}")
    except Exception as e:
        print(f"Notice: Could not query /v1/models: {e}")
        
    if not model_name:
        model_name = "default"
        
    modes = ["thinking", "direct"] if args.mode == "both" else [args.mode]
    grid = [int(c.strip()) for c in args.concurrency_grid.split(",")]
    
    # Warmup
    print("\nWarming up connection...")
    async with httpx.AsyncClient() as client:
        await send_req(client, args.server_url, model_name, BENCHMARK_PROMPTS[0], False, 64)
    print("Warmup complete. Beginning sweep.\n")
    
    optimal_results = {}
    
    for m in modes:
        is_thinking = (m == "thinking")
        print(f"--- Sweeping {m.upper()} Mode across concurrencies: {grid} ---")
        step_results = []
        for c in grid:
            res = await run_sweep_step(args.server_url, model_name, c, is_thinking, args.requests_per_level)
            step_results.append(res)
            print(f"  Concurrency {c:2d} -> {res['throughput_tok_s']:6.1f} tok/s | Mean Latency: {res['mean_latency_s']:5.2f}s | P95: {res['p95_latency_s']:5.2f}s | Errors: {res['errors']}")
            await asyncio.sleep(0.5)
            
        # Find peak throughput and knee of the curve (>= 90% peak throughput with lowest concurrency)
        peak_th = max(r["throughput_tok_s"] for r in step_results)
        valid_steps = [r for r in step_results if r["errors"] == 0 and r["throughput_tok_s"] >= 0.90 * peak_th]
        best_c = valid_steps[0]["concurrency"] if valid_steps else step_results[-1]["concurrency"]
        
        optimal_results[m] = {
            "tested_grid": step_results,
            "peak_throughput_tok_s": peak_th,
            "recommended_concurrency": best_c
        }
        print(f"--> Optimal {m.upper()} Concurrency: {best_c} (Peak: {peak_th:.1f} tok/s)\n")

    # Policy Enforcement: Register concurrency
    if args.register_suite:
        registry_path = os.path.join(os.path.dirname(__file__), "..", "data", "concurrency_policy_registry.json")
        registry = {}
        if os.path.exists(registry_path):
            try:
                with open(registry_path) as f:
                    registry = json.load(f)
            except Exception:
                registry = {}
                
        rec_c = optimal_results.get("thinking", optimal_results.get("direct"))["recommended_concurrency"]
        registry[args.register_suite] = {
            "model_id": model_name,
            "locked_concurrency": rec_c,
            "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "details": optimal_results
        }
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"Policy Enforced: Locked concurrency {rec_c} registered for comparison suite '{args.register_suite}'")
        print(f"Registry saved to: {registry_path}")

def main():
    parser = argparse.ArgumentParser(description="Automated Concurrency Smoke Test & Calibration")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--concurrency_grid", type=str, default="8,16,24,32,48,64")
    parser.add_argument("--requests_per_level", type=int, default=16)
    parser.add_argument("--mode", type=str, choices=["thinking", "direct", "both"], default="both")
    parser.add_argument("--register_suite", type=str, default=None, help="Benchmark suite name to lock concurrency for comparisons")
    args = parser.parse_args()
    
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
