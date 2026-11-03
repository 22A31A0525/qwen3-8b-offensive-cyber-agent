#!/usr/bin/env python3
"""
Multi-Turn Evaluation Sharder for Qwen3-8B v3 Pipeline
======================================================
Splits multi-turn test sets conversation-wise into 3 shards across 3 evaluation worker environments:
- Shard 1 (Evaluation Shard 1)
- Shard 2 (Evaluation Shard 2)
- Shard 3 (Evaluation Shard 3)

Ensures 100% conversation-wise boundary integrity (no mid-dialogue splits).
"""

import os
import json
from typing import List, Dict, Any

def partition_records(records: List[Dict[str, Any]], num_shards: int = 3) -> List[List[Dict[str, Any]]]:
    n = len(records)
    base_size = n // num_shards
    remainder = n % num_shards
    shards = []
    start = 0
    for i in range(num_shards):
        size = base_size + (1 if i < remainder else 0)
        shards.append(records[start : start + size])
        start += size
    return shards

def audit_shards(name: str, full_records: List[Dict[str, Any]], shards: List[List[Dict[str, Any]]]):
    print(f"\n=======================================================")
    print(f" AUDIT FOR: {name} (Total Dialogues: {len(full_records)})")
    print(f"=======================================================")
    
    total_turns_full = sum(sum(1 for m in r.get('messages', []) if m.get('role') == 'assistant') for r in full_records)
    total_tools_full = sum(sum(len(m.get('tool_calls', [])) for m in r.get('messages', []) if m.get('role') == 'assistant') for r in full_records)
    
    all_seen_ids = set()
    shard_turn_sum = 0
    shard_tool_sum = 0
    
    for i, s in enumerate(shards, 1):
        s_turns = sum(sum(1 for m in r.get('messages', []) if m.get('role') == 'assistant') for r in s)
        s_tools = sum(sum(len(m.get('tool_calls', [])) for m in r.get('messages', []) if m.get('role') == 'assistant') for r in s)
        s_ids = {r['id'] for r in s}
        
        # Check disjointness
        overlap = all_seen_ids & s_ids
        assert len(overlap) == 0, f"Critical error: ID overlap in Shard {i}: {overlap}"
        all_seen_ids.update(s_ids)
        
        shard_turn_sum += s_turns
        shard_tool_sum += s_tools
        
        print(f"Shard {i}:")
        print(f"  - Dialogues: {len(s):>4d}")
        print(f"  - Assistant Turns: {s_turns:>4d}")
        print(f"  - Tool Calls: {s_tools:>4d}")
        print(f"  - Dialogue ID Range: {s[0]['id']} -> {s[-1]['id']}")
        
    assert len(all_seen_ids) == len(full_records), "Total IDs in shards do not match full dataset!"
    assert shard_turn_sum == total_turns_full, f"Turn mismatch: {shard_turn_sum} vs {total_turns_full}"
    assert shard_tool_sum == total_tools_full, f"Tool calls mismatch: {shard_tool_sum} vs {total_tools_full}"
    print(f"[VERIFIED OK] 100% lossless conversation-wise partition confirmed for {name}.\n")

def split_file(input_path: str, output_prefix: str, output_dir: str, num_shards: int = 3):
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    shards = partition_records(records, num_shards=num_shards)
    audit_shards(os.path.basename(input_path), records, shards)
    
    saved_paths = []
    for i, s in enumerate(shards, 1):
        out_name = f"{output_prefix}_shard_{i}.jsonl"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as out_f:
            for r in s:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        saved_paths.append(out_path)
        print(f"Wrote Shard {i} ({len(s)} dialogues) -> {out_path}")
    return saved_paths

if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir_env = os.getenv("DATA_DIR")
    base_data_dir = os.path.join(data_dir_env, "test") if data_dir_env else os.path.join(repo_root, "data", "clean", "test")
    out_dir = os.getenv("OUTPUT_DIR", os.path.join(repo_root, "v3_pipeline", "eval_shards"))
    os.makedirs(out_dir, exist_ok=True)
    
    tool_replay_file = os.path.join(base_data_dir, "tool_replay_20pct.jsonl")
    if os.path.exists(tool_replay_file):
        print(f"Splitting Tool Replay: {tool_replay_file}")
        split_file(tool_replay_file, "tool_replay", out_dir, num_shards=3)
        
    exploit_file = os.path.join(base_data_dir, "exploit_60pct.jsonl")
    if os.path.exists(exploit_file):
        print(f"Splitting Exploit: {exploit_file}")
        split_file(exploit_file, "exploit", out_dir, num_shards=3)

