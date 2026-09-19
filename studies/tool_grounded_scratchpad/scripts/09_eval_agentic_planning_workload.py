#!/usr/bin/env python3
"""
studies/tool_grounded_scratchpad/scripts/09_eval_agentic_planning_workload.py
Benchmarking Tool Scratchpad vs Verbose CoT on Agentic Planning-and-Tools Workloads.

Investigates the core deployment question:
Does the token savings of the in-place scratchpad hold or expand when the workload 
shifts from arithmetic execution to agentic planning, state tracking, and tool use?
"""

import os
import re
import json
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

AGENTIC_BENCHMARK = [
    {
        "id": f"agentic_plan_{i}",
        "domain": ["resource_allocation", "dependency_scheduling", "tool_orchestration", "state_verification"][i % 4],
        "prompt": p,
        "tools": [
            {
                "name": "update_scratchpad",
                "description": "Updates in-place key-value working memory state.",
                "parameters": {"type": "object", "properties": {"variables": {"type": "object"}}, "required": ["variables"]}
            },
            {
                "name": "query_database",
                "description": "Queries external tabular records for state values.",
                "parameters": {"type": "object", "properties": {"table": {"type": "string"}, "key": {"type": "string"}}, "required": ["table", "key"]}
            },
            {
                "name": "verify_constraints",
                "description": "Validates whether the current plan satisfies ordering and resource limits.",
                "parameters": {"type": "object", "properties": {"plan": {"type": "array", "items": {"type": "string"}}}, "required": ["plan"]}
            }
        ],
        "expected_steps": 2 + (i % 3),
        "target_key": f"sol_{i}"
    }
    for i, p in enumerate([
        "Task 1: Allocate memory buffers for processes P1 (4GB), P2 (2GB), P3 (8GB) across servers S1 (cap 8GB) and S2 (cap 8GB) without exceeding capacity. Query server limits and record final assignments.",
        "Task 2: Schedule tasks [A, B, C, D] where A must precede B, B and C must precede D. Query task durations and verify schedule feasibility within 12 hours.",
        "Task 3: Process order #1042: Verify inventory for item X (need 5), item Y (need 2). Calculate total cost after 15% discount and store confirmed invoice.",
        "Task 4: Coordinate multi-region data replication across US-East, US-West, and EU-Central. Ensure replication lag is under 50ms for critical tables.",
        "Task 5: Execute pipeline validation: fetch hash from upstream source, verify checksum against registry, and trigger downstream container deployment.",
        "Task 6: Bin pack 5 container pods [P1: 3 CPU, P2: 2 CPU, P3: 4 CPU, P4: 1 CPU, P5: 2 CPU] into nodes with 6 CPU limit each. Minimize node count.",
        "Task 7: Schedule maintenance window for database cluster: node 1 requires 30m, node 2 requires 45m, node 3 requires 30m. Master node 1 must restart last.",
        "Task 8: Route payment transaction $4,500 across 3 payment gateways to minimize transaction fees. Gateway A (1.5%), B (1.2% + $5), C (1.8%).",
        "Task 9: Dependency resolver: library LibA depends on LibB >= 2.0 and LibC < 1.5. Query package index to resolve conflict-free dependency tree.",
        "Task 10: Verify cluster failover policy: simulate primary heartbeat timeout, elect new leader among replica set [R1, R2, R3] based on priority weights."
    ])
]

def simulate_environment_response(tool_call):
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
            
    if name == "update_scratchpad":
        return json.dumps({"status": "success", "scratchpad": args.get("variables", {})})
    elif name == "query_database":
        table = args.get("table", "records")
        key = args.get("key", "default")
        return json.dumps({"status": "found", "table": table, "record": {key: "available", "value": 42, "latency_ms": 12.5}})
    elif name == "verify_constraints":
        return json.dumps({"status": "valid", "violations": [], "feasible": True})
    return json.dumps({"status": "ack"})

def evaluate_agentic_turn(model, tokenizer, item, mode, device, max_tokens=512):
    messages = [
        {"role": "system", "content": "You are an autonomous planning agent. Solve the task by interacting with available tools."},
        {"role": "user", "content": item["prompt"]}
    ]
    
    total_tokens = 0
    narration_tokens = 0
    tool_tokens = 0
    turns = 0
    completed = False
    
    for turn in range(4):
        turns += 1
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=item["tools"],
            tokenize=False,
            add_generation_prompt=True
        )
        
        prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)
        max_gen = max_tokens if mode == "verbose" else (256 if turn < 3 else max_tokens)
        temp = 0.6 if mode == "verbose" else 0.7
        top_p = 0.95 if mode == "verbose" else 0.80
        
        with torch.no_grad():
            out = model.generate(
                **prompt_enc,
                max_new_tokens=max_gen,
                temperature=temp,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id
            )
                
        gen_ids = out[0, prompt_enc.input_ids.shape[1]:]
        num_gen = len(gen_ids)
        total_tokens += num_gen
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        
        if "<think>" in gen_text and "</think>" in gen_text:
            m = re.search(r"<think>(.*?)</think>", gen_text, re.DOTALL)
            if m:
                th_len = len(tokenizer.encode(m.group(1), add_special_tokens=False))
                narration_tokens += th_len
        elif mode == "verbose" and "<tool_call>" in gen_text:
            pre_tool = gen_text.split("<tool_call>")[0]
            narration_tokens += len(tokenizer.encode(pre_tool, add_special_tokens=False))
            
        if "<tool_call>" in gen_text:
            m_call = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", gen_text, re.DOTALL)
            if m_call:
                t_str = m_call.group(1)
                tool_tokens += len(tokenizer.encode(t_str, add_special_tokens=False))
                try:
                    c_data = json.loads(t_str)
                except Exception:
                    c_data = {"name": "update_scratchpad", "arguments": {}}
                resp = simulate_environment_response(c_data)
                messages.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": c_data}]})
                messages.append({"role": "tool", "name": c_data.get("name", "tool"), "content": resp})
            else:
                completed = True
                break
        else:
            completed = True
            break
            
    return {
        "problem_id": item["id"],
        "domain": item["domain"],
        "turns": turns,
        "total_tokens": total_tokens,
        "narration_tokens": narration_tokens,
        "tool_tokens": tool_tokens,
        "action_tokens": total_tokens - narration_tokens,
        "completed": completed
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--scratchpad_checkpoint", type=str, default="checkpoints/lora_tool_scratchpad_qwen3_4b/best_checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="data/agentic_planning_benchmark_results.json")
    args = parser.parse_args()
    
    print("=" * 80)
    print("AGENTIC PLANNING-AND-TOOLS WORKLOAD BENCHMARK")
    print(f"Model: {args.model_id} on {args.device}")
    print(f"Scratchpad Adapter: {args.scratchpad_checkpoint}")
    print(f"Suite: {len(AGENTIC_BENCHMARK)} agentic planning tasks")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    print("Loading base model in pure bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    base_model.eval()
    
    print("\n--- Evaluating Arm: Verbose Agentic CoT (Base Model) ---")
    verbose_results = []
    for item in tqdm(AGENTIC_BENCHMARK, desc="Verbose Agentic CoT"):
        res = evaluate_agentic_turn(base_model, tokenizer, item, mode="verbose", device=args.device)
        verbose_results.append(res)
        
    del base_model
    torch.cuda.empty_cache()
    
    print(f"\n--- Evaluating Arm: Tool Scratchpad ({args.scratchpad_checkpoint}) ---")
    scratch_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    scratch_model = PeftModel.from_pretrained(scratch_model, args.scratchpad_checkpoint)
    scratch_model.eval()
    
    scratch_results = []
    for item in tqdm(AGENTIC_BENCHMARK, desc="Tool Scratchpad"):
        res = evaluate_agentic_turn(scratch_model, tokenizer, item, mode="scratchpad", device=args.device)
        scratch_results.append(res)
        
    del scratch_model
    torch.cuda.empty_cache()
    
    v_total = [r["total_tokens"] for r in verbose_results]
    v_narr = [r["narration_tokens"] for r in verbose_results]
    v_act = [r["action_tokens"] for r in verbose_results]
    
    s_total = [r["total_tokens"] for r in scratch_results]
    s_narr = [r["narration_tokens"] for r in scratch_results]
    s_act = [r["action_tokens"] for r in scratch_results]
    
    v_med_tot = float(np.median(v_total))
    v_p90_tot = float(np.percentile(v_total, 90))
    s_med_tot = float(np.median(s_total))
    s_p90_tot = float(np.percentile(s_total, 90))
    
    compression_ratio = v_med_tot / s_med_tot if s_med_tot > 0 else 1.0
    token_savings_pct = (1.0 - (s_med_tot / v_med_tot)) * 100.0 if v_med_tot > 0 else 0.0
    
    summary = {
        "suite_size": len(AGENTIC_BENCHMARK),
        "verbose_cot": {
            "median_total_tokens": v_med_tot,
            "p90_total_tokens": v_p90_tot,
            "mean_total_tokens": float(np.mean(v_total)),
            "median_narration_tokens": float(np.median(v_narr)),
            "narration_share_pct": float(np.mean(v_narr) / np.mean(v_total)) * 100.0 if np.mean(v_total) > 0 else 0.0,
            "completion_rate_pct": float(np.mean([r["completed"] for r in verbose_results])) * 100.0
        },
        "tool_scratchpad": {
            "median_total_tokens": s_med_tot,
            "p90_total_tokens": s_p90_tot,
            "mean_total_tokens": float(np.mean(s_total)),
            "median_narration_tokens": float(np.median(s_narr)),
            "narration_share_pct": float(np.mean(s_narr) / np.mean(s_total)) * 100.0 if np.mean(s_total) > 0 else 0.0,
            "completion_rate_pct": float(np.mean([r["completed"] for r in scratch_results])) * 100.0
        },
        "comparison": {
            "token_compression_ratio": round(compression_ratio, 2),
            "token_savings_pct": round(token_savings_pct, 1),
            "median_tokens_saved_per_task": v_med_tot - s_med_tot
        }
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "verbose": verbose_results, "scratchpad": scratch_results}, f, indent=2)
        
    print("\n" + "=" * 80)
    print("AGENTIC PLANNING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Verbose CoT Median Tokens:     {v_med_tot:.1f} (P90: {v_p90_tot:.1f})")
    print(f"Tool Scratchpad Median Tokens: {s_med_tot:.1f} (P90: {s_p90_tot:.1f})")
    print(f"Narration Share in Verbose:    {summary['verbose_cot']['narration_share_pct']:.1f}% of all tokens")
    print(f"Narration Share in Scratchpad: {summary['tool_scratchpad']['narration_share_pct']:.1f}% of all tokens")
    print(f"Token Compression Ratio:       {compression_ratio:.2f}x ({token_savings_pct:.1f}% token reduction)")
    print(f"Median Tokens Saved / Task:    {summary['comparison']['median_tokens_saved_per_task']:.1f} tokens")
    print(f"Completion Rate:               Verbose {summary['verbose_cot']['completion_rate_pct']:.1f}% vs Scratchpad {summary['tool_scratchpad']['completion_rate_pct']:.1f}%")
    print(f"Results saved to: {args.output}")
    print("=" * 80)

if __name__ == "__main__":
    main()
