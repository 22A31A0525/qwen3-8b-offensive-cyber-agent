"""Detailed Clinical Reviewer for 39 Critical and 6 High Cross-Split Pairs.

Inspects full dialogues from original dataset files, clusters connected pairs,
categorizes into genuine answer/eval leakage vs generic mock tool outputs vs
scaffolding, and produces an audit review report.
"""
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Set

EVIDENCE_FILE = "v3_pipeline/audit/output/audit_evidence.jsonl"
DATA_DIR = "v3_pipeline/data"


def load_dataset_records() -> Dict[str, Dict[str, Any]]:
    """Loads all records from train, val, and test indexed by ID."""
    records_by_id = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(DATA_DIR, f"{split}.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = str(rec.get("id") or rec.get("conv_id"))
                rec["__split"] = split
                records_by_id[rid] = rec
    return records_by_id


def main():
    records_by_id = load_dataset_records()
    print(f"Loaded {len(records_by_id)} total dataset records into memory.")

    # 1. Read all cross-split critical and high pairs
    flagged_pairs = []
    with open(EVIDENCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("left_split") != item.get("right_split"):
                if item.get("severity") in ("critical", "high"):
                    flagged_pairs.append(item)

    print(f"Total cross-split critical + high pairs: {len(flagged_pairs)}")

    # 2. Extract unique IDs per split
    unique_ids_by_split = {"train": set(), "val": set(), "test": set()}
    for p in flagged_pairs:
        unique_ids_by_split[p["left_split"]].add(p["left_id"])
        unique_ids_by_split[p["right_split"]].add(p["right_id"])

    print("\nUnique Affected IDs:")
    for sp, ids in unique_ids_by_split.items():
        print(f"  {sp.upper()} ({len(ids)} unique IDs): {sorted(list(ids))[:10]}...")

    # 3. Analyze each pair deeply
    findings_by_cluster = defaultdict(list)

    for p in flagged_pairs:
        lid = p["left_id"]
        rid = p["right_id"]
        l_rec = records_by_id.get(lid, {})
        r_rec = records_by_id.get(rid, {})

        l_cat = l_rec.get("metadata", {}).get("category") or l_rec.get("category", "unknown")
        r_cat = r_rec.get("metadata", {}).get("category") or r_rec.get("category", "unknown")
        l_src = l_rec.get("metadata", {}).get("source_dataset") or "unknown"
        r_src = r_rec.get("metadata", {}).get("source_dataset") or "unknown"

        cluster_key = f"{l_src}::{r_src}::{p['classification']}::{p['field_matched']}"
        findings_by_cluster[cluster_key].append({
            "pair": p,
            "left_rec": l_rec,
            "right_rec": r_rec,
            "left_cat": l_cat,
            "right_cat": r_cat,
        })

    print(f"\nDiscovered {len(findings_by_cluster)} distinct clusters across the 45 pairs:")
    for c_key, items in findings_by_cluster.items():
        print(f"\n--- Cluster: {c_key} ({len(items)} pairs) ---")
        sample = items[0]
        p = sample["pair"]
        l_rec = sample["left_rec"]
        r_rec = sample["right_rec"]

        print(f"  Sample pair: {p['left_id']} ({p['left_split']}) -> {p['right_id']} ({p['right_split']})")
        print(f"  Reason: {p.get('reason')}")
        print(f"  Field Matched: {p.get('field_matched')}")
        print(f"  Meaningful Jaccard: {p.get('meaningful_jaccard')}")
        
        # Print snippet of messages
        l_msgs = l_rec.get("messages", [])
        r_msgs = r_rec.get("messages", [])
        
        l_user = next((m["content"] for m in l_msgs if m.get("role") == "user"), "")[:120]
        r_user = next((m["content"] for m in r_msgs if m.get("role") == "user"), "")[:120]
        print(f"  Left User Prompt:  {l_user!r}")
        print(f"  Right User Prompt: {r_user!r}")

        # If tool observation matched, check tool outputs
        l_tools = [m["content"] for m in l_msgs if m.get("role") == "tool"]
        r_tools = [m["content"] for m in r_msgs if m.get("role") == "tool"]
        if l_tools and r_tools:
            print(f"  Left Tool Obs sample:  {l_tools[0][:100]!r}")
            print(f"  Right Tool Obs sample: {r_tools[0][:100]!r}")


if __name__ == "__main__":
    main()

