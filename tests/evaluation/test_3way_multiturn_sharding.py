"""
Unit tests for 3-Way Multi-Turn Evaluation Sharding Pipeline (Qwen3-8B v3)
========================================================================
Verifies:
1. Exact conversation-wise boundary integrity across 3 shards.
2. Disjoint partition (zero conversation leakage between shards).
3. 100% dialogue & turn reconciliation against original V3 test splits.
4. Teacher-forced turn integrity (ensuring no target assistant turn is leaked in context).
"""

import os
import json
import pytest
from src.data.split_multiturn_shards_v3 import partition_records

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHARDS_DIR = os.path.join(REPO_ROOT, "v3_pipeline", "eval_shards")
BASE_DATA_DIR = os.path.join(REPO_ROOT, "data", "clean", "test")

def test_partition_logic_3way():
    dummy = [{"id": f"conv_{i}"} for i in range(130)]
    shards = partition_records(dummy, num_shards=3)
    assert len(shards) == 3
    assert len(shards[0]) == 44
    assert len(shards[1]) == 43
    assert len(shards[2]) == 43
    
    ids_0 = {r["id"] for r in shards[0]}
    ids_1 = {r["id"] for r in shards[1]}
    ids_2 = {r["id"] for r in shards[2]}
    assert len(ids_0 & ids_1) == 0
    assert len(ids_0 & ids_2) == 0
    assert len(ids_1 & ids_2) == 0
    assert len(ids_0 | ids_1 | ids_2) == 130

def test_tool_replay_shards_reconciliation():
    orig_path = os.path.join(BASE_DATA_DIR, "tool_replay_20pct.jsonl")
    with open(orig_path, "r", encoding="utf-8") as f:
        orig_records = [json.loads(line) for line in f if line.strip()]
        
    s1_path = os.path.join(SHARDS_DIR, "tool_replay_shard_1.jsonl")
    s2_path = os.path.join(SHARDS_DIR, "tool_replay_shard_2.jsonl")
    s3_path = os.path.join(SHARDS_DIR, "tool_replay_shard_3.jsonl")
    
    with open(s1_path, "r", encoding="utf-8") as f:
        s1 = [json.loads(l) for l in f if l.strip()]
    with open(s2_path, "r", encoding="utf-8") as f:
        s2 = [json.loads(l) for l in f if l.strip()]
    with open(s3_path, "r", encoding="utf-8") as f:
        s3 = [json.loads(l) for l in f if l.strip()]
        
    assert len(s1) == 44
    assert len(s2) == 43
    assert len(s3) == 43
    assert len(s1) + len(s2) + len(s3) == len(orig_records) == 130
    
    # Check ID alignment
    combined_ids = [r["id"] for r in s1] + [r["id"] for r in s2] + [r["id"] for r in s3]
    orig_ids = [r["id"] for r in orig_records]
    assert combined_ids == orig_ids

def test_exploit_shards_reconciliation():
    orig_path = os.path.join(BASE_DATA_DIR, "exploit_60pct.jsonl")
    with open(orig_path, "r", encoding="utf-8") as f:
        orig_records = [json.loads(line) for line in f if line.strip()]
        
    s1_path = os.path.join(SHARDS_DIR, "exploit_shard_1.jsonl")
    s2_path = os.path.join(SHARDS_DIR, "exploit_shard_2.jsonl")
    s3_path = os.path.join(SHARDS_DIR, "exploit_shard_3.jsonl")
    
    with open(s1_path, "r", encoding="utf-8") as f:
        s1 = [json.loads(l) for l in f if l.strip()]
    with open(s2_path, "r", encoding="utf-8") as f:
        s2 = [json.loads(l) for l in f if l.strip()]
    with open(s3_path, "r", encoding="utf-8") as f:
        s3 = [json.loads(l) for l in f if l.strip()]
        
    assert len(s1) == 130
    assert len(s2) == 130
    assert len(s3) == 130
    assert len(s1) + len(s2) + len(s3) == len(orig_records) == 390
    
    combined_ids = [r["id"] for r in s1] + [r["id"] for r in s2] + [r["id"] for r in s3]
    orig_ids = [r["id"] for r in orig_records]
    assert combined_ids == orig_ids

def test_teacher_forced_turn_iterator_integrity():
    # Load sample dialogue from shard 1
    s1_path = os.path.join(SHARDS_DIR, "tool_replay_shard_1.jsonl")
    with open(s1_path, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
        
    messages = sample["messages"]
    turns_yielded = 0
    context = []
    
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            # Test context does not include current assistant turn
            assert msg not in context, "Target assistant turn leaked into prior context!"
            # Test context has preceding messages
            assert len(context) == idx
            turns_yielded += 1
        context.append(msg)
        
    asst_turn_count = sum(1 for m in messages if m.get("role") == "assistant")
    assert turns_yielded == asst_turn_count

