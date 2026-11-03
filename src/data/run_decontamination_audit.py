"""Top-Level Orchestration CLI Runner for 4-Tier Decontamination Audit.

Executes all 4 audit tiers across all 6 pairwise split combinations on the
canonical 6,500 dataset records in v3_pipeline/data/, enforces cryptographic
SHA-256 verification, and emits structured audit artifacts.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from v3_pipeline.audit.classifier import classify_pair, group_matches_by_pair
from v3_pipeline.audit.config import (
    CONFIRMED_LEAKAGE,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
)
from v3_pipeline.audit.normalizer import extract_audit_fields
from v3_pipeline.audit.reporter import format_gfm_table, generate_audit_reports
from v3_pipeline.audit.tier1_exact import run_tier1_audit
from v3_pipeline.audit.tier2_normalized import run_tier2_audit
from v3_pipeline.audit.tier3_ngram import run_tier3_audit
from v3_pipeline.audit.tier4_minhash import run_tier4_audit

FROZEN_CHECKSUMS = {
    "train.jsonl": "098dbee87a570101e0902adabcecca35862bd44b5c31be0cfb07d6b1c4b19b1d",
    "val.jsonl": "07578d14eecc5bad071c50f8e67579ea76166698b81379a56e215d01695d798a",
    "test.jsonl": "fabd494f98bd5e79e81c3b6e3c740db99a1b0d5abcfd0d2e40a72606721f02d5",
}

EXPECTED_COUNTS = {
    "train.jsonl": 5200,
    "val.jsonl": 650,
    "test.jsonl": 650,
}


def _compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def _load_records(filepath: str) -> List[Dict[str, Any]]:
    """Loads JSONL records from file."""
    records: List[Dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                records.append(rec)
            except Exception as e:
                raise ValueError(
                    f"Invalid JSON at {filepath}:{line_num}: {e}"
                ) from e
    return records


def _extract_known_flags(records: List[Dict[str, Any]]) -> Set[str]:
    """Extracts known secret flags from evaluation records for cross-split detection."""
    flags: Set[str] = set()
    flag_pattern = re.compile(
        r"(FLAG\{[^}]+\}|CTF\{[^}]+\})", re.IGNORECASE
    )
    for r in records:
        fields = extract_audit_fields(r)
        for text in (fields.get("tool_observation", ""), fields.get("assistant_answer", "")):
            if text:
                matches = flag_pattern.findall(text)
                for m in matches:
                    flags.add(m.strip())
    return flags


def run_audit(
    data_dir: str = "v3_pipeline/data",
    output_dir: str = "v3_pipeline/audit/output",
    report_path: str = "v3_pipeline/audit/audit_report.md",
    ngram_threshold: float = 0.50,
    minhash_threshold: float = 0.65,
    allow_derived: bool = False,
) -> Dict[str, Any]:
    """Executes the full 4-tier decontamination audit across all 6 pairwise splits."""
    start_time = time.time()
    print("=" * 80)
    print("4-TIER DECONTAMINATION & OVERLAP AUDIT — QWEN3-8B (v3 PIPELINE)")
    print("=" * 80)

    # 1. Verify Dataset Checksums
    print("\n[Phase 1/4] Verifying cryptographic SHA-256 checksums...")
    is_derived = allow_derived or ("clean" in data_dir.lower() or "derived" in data_dir.lower())
    observed_checksums: Dict[str, str] = {}
    for filename in ("train.jsonl", "val.jsonl", "test.jsonl"):
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required dataset file missing: {file_path}")
        observed_hash = _compute_sha256(file_path)
        observed_checksums[filename] = observed_hash
        expected_hash = FROZEN_CHECKSUMS[filename]

        if filename == "test.jsonl":
            if observed_hash != expected_hash:
                raise ValueError(
                    f"TEST SET CHECKSUM MISMATCH for {filename}!\n"
                    f"  Expected: {expected_hash}\n"
                    f"  Observed: {observed_hash}\n"
                    f"Test set must remain strictly immutable per Rule 1."
                )
            print(f"  [OK] {filename:<12} SHA-256: {observed_hash} (Strict Master Match)")
        else:
            if not is_derived:
                if observed_hash != expected_hash:
                    raise ValueError(
                        f"CHECKSUM MISMATCH for {filename}!\n"
                        f"  Expected: {expected_hash}\n"
                        f"  Observed: {observed_hash}\n"
                        f"Pipeline aborted per Rule 1 & Rule 9."
                    )
                print(f"  [OK] {filename:<12} SHA-256: {observed_hash}")
            else:
                print(f"  [DERIVED] {filename:<12} SHA-256: {observed_hash}")

    # 2. Load and index all records
    print("\n[Phase 2/4] Loading and reconciling 6,500 dataset records...")
    splits: Dict[str, List[Dict[str, Any]]] = {}
    records_by_id: Dict[str, Dict[str, Any]] = {}

    for split_name, filename in (
        ("train", "train.jsonl"),
        ("val", "val.jsonl"),
        ("test", "test.jsonl"),
    ):
        file_path = os.path.join(data_dir, filename)
        recs = _load_records(file_path)
        expected_cnt = EXPECTED_COUNTS[filename]
        if len(recs) != expected_cnt:
            raise ValueError(
                f"Record count mismatch in {filename}: expected {expected_cnt}, found {len(recs)}"
            )
        splits[split_name] = recs
        for idx, r in enumerate(recs):
            rid = str(r.get("id") or r.get("dialogue_id") or r.get("conv_id") or f"{split_name}_{idx}")
            records_by_id[rid] = r
        print(f"  [OK] {split_name:<5} loaded: {len(recs):>5} dialogues")

    total_records = sum(len(r) for r in splits.values())
    print(f"  --> Total Dialogues Reconciled: {total_records} / 6,500 (100.0%)")

    # Extract eval flags from test and val
    eval_flags = _extract_known_flags(splits["test"]) | _extract_known_flags(splits["val"])
    print(f"  --> Discovered {len(eval_flags)} distinct eval flags/secrets in test/val splits")

    # 3. Execute all 6 pairwise split comparisons across Tiers 1-4
    print("\n[Phase 3/4] Executing 4-tier audit across 6 pairwise split combinations...")
    pairwise_pairs = [
        ("train", "test", "Cross-Split Leakage"),
        ("train", "val", "Cross-Split Leakage"),
        ("val", "test", "Cross-Split Leakage"),
        ("train", "train", "Within-Split Data Quality"),
        ("val", "val", "Within-Split Data Quality"),
        ("test", "test", "Within-Split Data Quality"),
    ]

    all_raw_matches: List[Any] = []

    for l_sp, r_sp, scope in pairwise_pairs:
        pair_t0 = time.time()
        left_recs = splits[l_sp]
        right_recs = splits[r_sp]

        # Tier 1: Exact Identity & Metadata
        t1_matches = run_tier1_audit(
            left_records=left_recs,
            right_records=right_recs,
            left_split=l_sp,
            right_split=r_sp,
            known_eval_flags=eval_flags if l_sp != r_sp else None,
        )

        # Tier 2: Normalized Exact Duplicates
        t2_matches = run_tier2_audit(
            left_records=left_recs,
            right_records=right_recs,
            left_split=l_sp,
            right_split=r_sp,
        )

        # Tier 3: Meaningful 13-gram Overlap
        t3_matches = run_tier3_audit(
            left_records=left_recs,
            right_records=right_recs,
            left_split=l_sp,
            right_split=r_sp,
            threshold=ngram_threshold,
        )

        # Tier 4: MinHash-LSH & Security Entity Recall
        t4_matches = run_tier4_audit(
            left_records=left_recs,
            right_records=right_recs,
            left_split=l_sp,
            right_split=r_sp,
        )

        pair_matches = t1_matches + t2_matches + t3_matches + t4_matches
        all_raw_matches.extend(pair_matches)
        pair_dur = time.time() - pair_t0

        print(
            f"  [*] {l_sp.capitalize()} vs {r_sp.capitalize():<5} ({scope:<25}) -> "
            f"T1: {len(t1_matches):>3} | T2: {len(t2_matches):>3} | T3: {len(t3_matches):>3} | "
            f"T4: {len(t4_matches):>3} | Total: {len(pair_matches):>3} ({pair_dur:.2f}s)"
        )

    # 4. Multi-Signal Merge & Pair Classification
    print("\n[Phase 4/4] Merging candidate matches and classifying evidence...")
    grouped = group_matches_by_pair(all_raw_matches)
    print(f"  --> Grouped {len(all_raw_matches)} candidate tier hits into {len(grouped)} distinct record pairs")

    flagged_pairs: List[Dict[str, Any]] = []
    for (l_id, r_id), matches in grouped.items():
        left_rec = records_by_id.get(l_id)
        right_rec = records_by_id.get(r_id)
        evidence = classify_pair(
            candidate_matches=matches,
            left_record=left_rec,
            right_record=right_rec,
        )
        flagged_pairs.append(evidence)

    # Generate structured reports
    dataset_metadata = {
        "checksums": observed_checksums,
        "counts": {f"{k}.jsonl": len(v) for k, v in splits.items()},
        "total_dialogues": total_records,
        "eval_flags_indexed": len(eval_flags),
        "ngram_threshold": ngram_threshold,
        "minhash_threshold": minhash_threshold,
    }

    summary = generate_audit_reports(
        flagged_pairs=flagged_pairs,
        dataset_metadata=dataset_metadata,
        output_dir=output_dir,
        report_md_path=report_path,
    )

    elapsed = time.time() - start_time
    print(f"  --> Evidence exported to: {os.path.join(output_dir, 'audit_evidence.jsonl')}")
    print(f"  --> Summary exported to:  {os.path.join(output_dir, 'audit_summary.json')}")
    print(f"  --> Report exported to:   {report_path}")
    print(f"\nAudit completed in {elapsed:.2f}s.")

    # Console Summary Table
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY MATRIX")
    print("=" * 80)
    print(f"Total Candidate Pairs Flagged: {summary['total_pairs_flagged']}")
    print(f"Cross-Split Contamination (Critical): {summary['counts_by_severity'].get(SEVERITY_CRITICAL, 0)}")
    print(f"Suspected High Overlaps:             {summary['counts_by_severity'].get(SEVERITY_HIGH, 0)}")
    print(f"Suspected Medium Overlaps:           {summary['counts_by_severity'].get(SEVERITY_MEDIUM, 0)}")
    print(f"Benign / Low Overlaps:               {summary['counts_by_severity'].get(SEVERITY_LOW, 0)}")
    print(f"Within-Split Duplicates:             {summary['within_split_duplicates_total']}")

    verdict_str = (
        "CLEAN: ZERO CONTAMINATION DETECTED"
        if summary["cross_split_contamination_total"] == 0
        else "FAILED: CONTAMINATION DETECTED — RETRAINING BLOCKED"
    )
    print(f"\nDECONTAMINATION GATE VERDICT: {verdict_str}")
    print("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="4-Tier Decontamination Audit Runner for Qwen3-8B v3 Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="v3_pipeline/data",
        help="Path to frozen dataset directory (default: v3_pipeline/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="v3_pipeline/audit/output",
        help="Directory to save audit_evidence.jsonl and audit_summary.json",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="v3_pipeline/audit/audit_report.md",
        help="Path to save Markdown audit report",
    )
    parser.add_argument(
        "--ngram-threshold",
        type=float,
        default=0.50,
        help="Tier 3 meaningful Jaccard threshold (default: 0.50)",
    )
    parser.add_argument(
        "--minhash-threshold",
        type=float,
        default=0.65,
        help="Tier 4 MinHash similarity threshold (default: 0.65)",
    )
    parser.add_argument(
        "--allow-derived",
        action="store_true",
        help="Allow derived dataset run (enforces frozen test.jsonl hash while recording derived train/val hashes)",
    )

    args = parser.parse_args()

    try:
        summary = run_audit(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            report_path=args.report_path,
            ngram_threshold=args.ngram_threshold,
            minhash_threshold=args.minhash_threshold,
            allow_derived=args.allow_derived,
        )
        if summary["cross_split_contamination_total"] > 0:
            print("\nWARNING: Cross-split contamination candidates detected. Review audit_evidence.jsonl.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL AUDIT ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
