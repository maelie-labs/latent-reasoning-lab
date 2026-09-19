import argparse
import asyncio
import time
import httpx
import numpy as np

PROMPTS = [
    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars will she make every day?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 into repairs. He sells it for $150,000. What is his net profit?",
    "James decides to run 3 miles every day. How many miles does he run in 2 weeks?",
    "Every day, Wendi feeds each of her chickens 3 cups of feed. If she has 20 chickens, how many cups of feed does she use in 7 days?",
    "A deep-sea monster at 100 feet below sea level dives down 400 feet, then rises 250 feet. What is its new depth below sea level in feet?",
    "A bakery makes 400 loaves of bread. 25% are whole wheat, 35% are rye, and the rest are white. How many loaves are white?",
    "A car travels 60 miles per hour for 3 hours, then 75 miles per hour for 2 hours. What is the total distance traveled in miles?",
]

async def send_req(client, server_url, prompt, enable_thinking, max_tokens):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": "Qwen/Qwen3-4B",
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
    resp = await client.post(f"{server_url}/v1/chat/completions", json=payload, timeout=300.0)
    dur = time.time() - t0
    if resp.status_code != 200:
        return {"error": resp.text, "dur": dur, "tokens": 0}
    data = resp.json()
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return {"dur": dur, "tokens": tokens, "error": None}

async def benchmark_concurrency(server_url, concurrency, enable_thinking, num_requests=32):
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency + 10)
    
    async with httpx.AsyncClient(limits=limits) as client:
        async def worker(idx):
            prompt = PROMPTS[idx % len(PROMPTS)]
            max_tokens = 4096 if enable_thinking else 512
            async with semaphore:
                return await send_req(client, server_url, prompt, enable_thinking, max_tokens)
                
        t_start = time.time()
        tasks = [worker(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        total_wall_clock = time.time() - t_start
        
    total_tokens = sum(r["tokens"] for r in results if not r.get("error"))
    errors = sum(1 for r in results if r.get("error"))
    mean_lat = np.mean([r["dur"] for r in results if not r.get("error")])
    throughput = total_tokens / total_wall_clock if total_wall_clock > 0 else 0
    
    mode_str = "Thinking (max 4k)" if enable_thinking else "Non-Thinking (max 512)"
    print(f"[{mode_str}] Concurrency: {concurrency:2d} | Requests: {num_requests:2d} | Total Toks: {total_tokens:5d} | Wall-Clock: {total_wall_clock:5.2f}s | Throughput: {throughput:6.1f} tok/s | Mean Latency: {mean_lat:5.2f}s | Errors: {errors}")
    return {
        "concurrency": concurrency,
        "mode": mode_str,
        "throughput_tok_s": round(throughput, 1),
        "wall_clock_s": round(total_wall_clock, 2),
        "mean_latency_s": round(float(mean_lat), 2),
        "errors": errors
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:30001")
    args = parser.parse_args()

    print(f"=== Qwen3-4B Concurrency & Throughput Sweep on RTX PRO 4500 (32GB) ===")
    print(f"Target: {args.server_url}\n")
    
    # Warmup
    print("Sending warmup request...")
    limits = httpx.Limits(max_connections=5, max_keepalive_connections=5)
    async with httpx.AsyncClient(limits=limits) as client:
        await send_req(client, args.server_url, PROMPTS[0], False, 128)
    print("Warmup complete.\n")

    # 1. Non-Thinking Concurrency Sweep (16, 32, 48, 64)
    print("--- 1. Non-Thinking Mode Concurrency Sweep (32 requests each) ---")
    for c in [16, 32, 48, 64]:
        await benchmark_concurrency(args.server_url, concurrency=c, enable_thinking=False, num_requests=32)
        await asyncio.sleep(1)

    print("\n--- 2. Thinking Mode Concurrency Sweep (32 requests each) ---")
    for c in [16, 24, 32, 48]:
        await benchmark_concurrency(args.server_url, concurrency=c, enable_thinking=True, num_requests=32)
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
