"""Creates the versioned derived clean dataset data/clean/.

Leaves frozen original v3_pipeline/data/ untouched.
Replaces the 2 confirmed leaked training records and 1 validation record with
unused pristine dialogues from data/deduplicated/hermes.jsonl, preserving
exact 60/20/20 balance and 80/10/10 split counts.
"""
import hashlib
import json
import os
import shutil
from typing import Any, Dict, List, Set

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ORIGINAL_DATA_DIR = os.getenv("ORIGINAL_DATA_DIR", os.path.join(REPO_ROOT, "v3_pipeline", "data"))
DERIVED_DATA_DIR = os.getenv("DATA_DIR", os.path.join(REPO_ROOT, "data", "clean"))
HERMES_POOL = os.getenv("HERMES_POOL_PATH", os.path.join(REPO_ROOT, "data", "deduplicated", "hermes.jsonl"))

# Records confirmed for quarantine from clinical review
TRAIN_RECORDS_TO_REPLACE = {
    "hermes_7e118bec-7d0b-462a-950e-a652743bf76e",
    "hermes_3e2bdf27-eb65-4f7f-856b-a25e1bb69a8e",
}
VAL_RECORDS_TO_REPLACE = {
    "hermes_bfc12eb2-fea8-4c51-b695-33c16030d4c9",
}


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def main():
    print("==================================================================")
    print("   CREATING VERSIONED DERIVED DATASET: data/clean  ")
    print("==================================================================")

    # 1. Collect all currently used IDs across the 6,500 master dataset
    used_ids: Set[str] = set()
    for s in ["train", "val", "test"]:
        with open(os.path.join(ORIGINAL_DATA_DIR, f"{s}.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    used_ids.add(rec["id"])

    # 2. Find clean replacement candidates from unused Hermes pool
    replacement_candidates: List[Dict[str, Any]] = []
    with open(HERMES_POOL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec["id"]
            if rid not in used_ids:
                # Ensure it has tools and valid structure
                if rec.get("tools") and len(rec.get("messages", [])) >= 2:
                    replacement_candidates.append(rec)
                    if len(replacement_candidates) >= 10:
                        break

    print(f"Discovered {len(replacement_candidates)} eligible replacement candidates from unused pool.")
    assert len(replacement_candidates) >= 3, "Insufficient replacement candidates in pool!"

    rep_train_1 = replacement_candidates[0]
    rep_train_2 = replacement_candidates[1]
    rep_val_1 = replacement_candidates[2]

    # Ensure metadata category and source_dataset match expected schema
    for r in [rep_train_1, rep_train_2, rep_val_1]:
        r.setdefault("metadata", {})
        r["metadata"]["source_dataset"] = "hermes-function-calling-v1"
        r["metadata"]["category"] = "tool_calling_replay"
        r["metadata"]["derivation_note"] = "remediated_decontaminated_v3_1"

    # Create directories
    os.makedirs(DERIVED_DATA_DIR, exist_ok=True)
    for sp in ["train", "val", "test"]:
        os.makedirs(os.path.join(DERIVED_DATA_DIR, sp), exist_ok=True)

    # 3. Process Train Split
    train_clean: List[Dict[str, Any]] = []
    train_exploit: List[Dict[str, Any]] = []
    train_reasoning: List[Dict[str, Any]] = []
    train_replay: List[Dict[str, Any]] = []

    with open(os.path.join(ORIGINAL_DATA_DIR, "train.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec["id"]
            if rid == "hermes_7e118bec-7d0b-462a-950e-a652743bf76e":
                print(f"  [REPLACE] Train: Replacing {rid} with {rep_train_1['id']}")
                rec = rep_train_1
            elif rid == "hermes_3e2bdf27-eb65-4f7f-856b-a25e1bb69a8e":
                print(f"  [REPLACE] Train: Replacing {rid} with {rep_train_2['id']}")
                rec = rep_train_2

            train_clean.append(rec)
            cat = rec.get("metadata", {}).get("category") or rec.get("category", "")
            src = rec.get("metadata", {}).get("source_dataset", "")
            if "exploit" in cat or "red_team" in src or "pentestr1" in src:
                train_exploit.append(rec)
            elif "reasoning" in cat or "cybersecurity" in src:
                train_reasoning.append(rec)
            elif "replay" in cat or "tool" in cat or "hermes" in src or "nemotron" in src:
                train_replay.append(rec)
            else:
                train_replay.append(rec)

    # 4. Process Validation Split
    val_clean: List[Dict[str, Any]] = []
    val_exploit: List[Dict[str, Any]] = []
    val_reasoning: List[Dict[str, Any]] = []
    val_replay: List[Dict[str, Any]] = []

    with open(os.path.join(ORIGINAL_DATA_DIR, "val.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec["id"]
            if rid == "hermes_bfc12eb2-fea8-4c51-b695-33c16030d4c9":
                print(f"  [REPLACE] Val:   Replacing {rid} with {rep_val_1['id']}")
                rec = rep_val_1

            val_clean.append(rec)
            cat = rec.get("metadata", {}).get("category") or rec.get("category", "")
            src = rec.get("metadata", {}).get("source_dataset", "")
            if "exploit" in cat or "red_team" in src or "pentestr1" in src:
                val_exploit.append(rec)
            elif "reasoning" in cat or "cybersecurity" in src:
                val_reasoning.append(rec)
            elif "replay" in cat or "tool" in cat or "hermes" in src or "nemotron" in src:
                val_replay.append(rec)
            else:
                val_replay.append(rec)

    # 5. Process Test Split (100% UNTOUCHED, exact bit-for-bit copy)
    test_clean: List[Dict[str, Any]] = []
    test_exploit: List[Dict[str, Any]] = []
    test_reasoning: List[Dict[str, Any]] = []
    test_replay: List[Dict[str, Any]] = []

    with open(os.path.join(ORIGINAL_DATA_DIR, "test.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            test_clean.append(rec)
            cat = rec.get("metadata", {}).get("category") or rec.get("category", "")
            src = rec.get("metadata", {}).get("source_dataset", "")
            if "exploit" in cat or "red_team" in src or "pentestr1" in src:
                test_exploit.append(rec)
            elif "reasoning" in cat or "cybersecurity" in src:
                test_reasoning.append(rec)
            elif "replay" in cat or "tool" in cat or "hermes" in src or "nemotron" in src:
                test_replay.append(rec)
            else:
                test_replay.append(rec)

    # Assert exact count integrity
    assert len(train_clean) == 5200, f"Expected 5200 train, got {len(train_clean)}"
    assert len(train_exploit) == 3120, f"Expected 3120 train exploit, got {len(train_exploit)}"
    assert len(train_reasoning) == 1040, f"Expected 1040 train reasoning, got {len(train_reasoning)}"
    assert len(train_replay) == 1040, f"Expected 1040 train replay, got {len(train_replay)}"

    assert len(val_clean) == 650, f"Expected 650 val, got {len(val_clean)}"
    assert len(val_exploit) == 390, f"Expected 390 val exploit, got {len(val_exploit)}"
    assert len(val_reasoning) == 130, f"Expected 130 val reasoning, got {len(val_reasoning)}"
    assert len(val_replay) == 130, f"Expected 130 val replay, got {len(val_replay)}"

    assert len(test_clean) == 650, f"Expected 650 test, got {len(test_clean)}"
    assert len(test_exploit) == 390, f"Expected 390 test exploit, got {len(test_exploit)}"
    assert len(test_reasoning) == 130, f"Expected 130 test reasoning, got {len(test_reasoning)}"
    assert len(test_replay) == 130, f"Expected 130 test replay, got {len(test_replay)}"

    # Write files
    def write_jsonl(path: str, records: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(DERIVED_DATA_DIR, "train.jsonl"), train_clean)
    write_jsonl(os.path.join(DERIVED_DATA_DIR, "val.jsonl"), val_clean)
    write_jsonl(os.path.join(DERIVED_DATA_DIR, "test.jsonl"), test_clean)

    write_jsonl(os.path.join(DERIVED_DATA_DIR, "train", "exploit_60pct.jsonl"), train_exploit)
    write_jsonl(os.path.join(DERIVED_DATA_DIR, "train", "reasoning_20pct.jsonl"), train_reasoning)
    write_jsonl(os.path.join(DERIVED_DATA_DIR, "train", "tool_replay_20pct.jsonl"), train_replay)

    write_jsonl(os.path.join(DERIVED_DATA_DIR, "val", "exploit_60pct.jsonl"), val_exploit)
    write_jsonl(os.path.join(DERIVED_DATA_DIR, "val", "reasoning_20pct.jsonl"), val_reasoning)
    write_jsonl(os.path.join(DERIVED_DATA_DIR, "val", "tool_replay_20pct.jsonl"), val_replay)

    write_jsonl(os.path.join(DERIVED_DATA_DIR, "test", "exploit_60pct.jsonl"), test_exploit)
    write_jsonl(os.path.join(DERIVED_DATA_DIR, "test", "reasoning_20pct.jsonl"), test_reasoning)
    write_jsonl(os.path.join(DERIVED_DATA_DIR, "test", "tool_replay_20pct.jsonl"), test_replay)

    # Compute checksums
    print("\n--- Derived Dataset SHA-256 Checksums ---")
    for fname in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        fpath = os.path.join(DERIVED_DATA_DIR, fname)
        csum = compute_sha256(fpath)
        print(f"  {fname:<12}: {csum}")

    # Check test.jsonl checksum against original
    orig_test_csum = compute_sha256(os.path.join(ORIGINAL_DATA_DIR, "test.jsonl"))
    derived_test_csum = compute_sha256(os.path.join(DERIVED_DATA_DIR, "test.jsonl"))
    assert orig_test_csum == derived_test_csum, "Test set was modified! Violation of Rule 1!"
    print(f"\n[PASS] test.jsonl is 100% byte-for-byte identical to original frozen test split: {orig_test_csum}")

    print("\nDerived clean dataset successfully created at data/clean/")


if __name__ == "__main__":
    main()

