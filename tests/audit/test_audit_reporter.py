"""Unit tests for Structured Report Generator (src.audit.reporter)."""
import json
import os
import re
import pytest
from src.audit.config import (
    CONFIRMED_LEAKAGE,
    SUSPECTED_HIGH,
    WITHIN_SPLIT_DUPLICATE,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
)
from src.audit.reporter import (
    format_gfm_table,
    generate_audit_reports,
)


def test_format_gfm_table_vertical_alignment():
    """Verifies format_gfm_table generates space-padded, vertically aligned GFM tables per Rule 11."""
    headers = ["Split", "Records", "Status"]
    rows = [
        ["train.jsonl", 5200, "Verified"],
        ["val.jsonl", 650, "Verified"],
        ["test.jsonl", 650, "Verified"],
    ]
    alignments = ["left", "right", "center"]

    table_str = format_gfm_table(headers, rows, alignments)
    lines = [line.strip() for line in table_str.strip().split("\n") if line.strip()]

    assert len(lines) == 5  # header, separator, 3 data rows

    # Rule 11 check: pipe positions must be identical on EVERY line
    pipe_indices_per_line = [
        [i for i, char in enumerate(line) if char == "|"] for line in lines
    ]
    first_line_pipes = pipe_indices_per_line[0]
    for idx, p_indices in enumerate(pipe_indices_per_line):
        assert p_indices == first_line_pipes, (
            f"Line {idx} pipe positions {p_indices} do not match header {first_line_pipes}:\n{lines[idx]}"
        )


def test_generate_audit_reports_creates_artifacts(tmp_path):
    """Verifies that generate_audit_reports creates jsonl, json summary, and md report."""
    output_dir = tmp_path / "output"
    report_md_path = tmp_path / "audit_report.md"

    mock_pairs = [
        {
            "audit_id": "audit_t01_e01",
            "left_id": "t01",
            "right_id": "e01",
            "left_split": "train",
            "right_split": "test",
            "left_source": "exploit_v3",
            "right_source": "eval_cyber_v1",
            "left_category": "exploit",
            "right_category": "reasoning",
            "match_type": "tier1_exact",
            "field_matched": "eval_flag",
            "raw_jaccard": 0.0,
            "meaningful_jaccard": 0.0,
            "containment_left_in_right": 0.0,
            "containment_right_in_left": 0.0,
            "matching_spans": [{"preview": "Found FLAG{secret_token_12345} in obs"}],
            "flag_or_proof_present": True,
            "scenario_id_collision": False,
            "tool_target_collision": False,
            "tool_observation_collision": True,
            "severity": SEVERITY_CRITICAL,
            "classification": CONFIRMED_LEAKAGE,
            "reason": "Eval flag detected in training tool observation",
            "review_status": "pending",
        },
        {
            "audit_id": "audit_t02_t03",
            "left_id": "t02",
            "right_id": "t03",
            "left_split": "train",
            "right_split": "train",
            "left_source": "exploit_v3",
            "right_source": "exploit_v3",
            "left_category": "exploit",
            "right_category": "exploit",
            "match_type": "tier1_exact",
            "field_matched": "canonical_hash",
            "raw_jaccard": 1.0,
            "meaningful_jaccard": 1.0,
            "containment_left_in_right": 1.0,
            "containment_right_in_left": 1.0,
            "matching_spans": [],
            "flag_or_proof_present": False,
            "scenario_id_collision": False,
            "tool_target_collision": False,
            "tool_observation_collision": False,
            "severity": SEVERITY_MEDIUM,
            "classification": WITHIN_SPLIT_DUPLICATE,
            "reason": "Internal full record duplicate in train",
            "review_status": "pending",
        },
    ]

    mock_metadata = {
        "checksums": {
            "train.jsonl": "098dbee87a570101e0902adabcecca35862bd44b5c31be0cfb07d6b1c4b19b1d",
            "val.jsonl": "07578d14eecc5bad071c50f8e67579ea76166698b81379a56e215d01695d798a",
            "test.jsonl": "fabd494f98bd5e79e81c3b6e3c740db99a1b0d5abcfd0d2e40a72606721f02d5",
        },
        "counts": {"train.jsonl": 5200, "val.jsonl": 650, "test.jsonl": 650},
    }

    result = generate_audit_reports(
        flagged_pairs=mock_pairs,
        dataset_metadata=mock_metadata,
        output_dir=str(output_dir),
        report_md_path=str(report_md_path),
    )

    evidence_file = output_dir / "audit_evidence.jsonl"
    summary_file = output_dir / "audit_summary.json"

    assert evidence_file.exists()
    assert summary_file.exists()
    assert report_md_path.exists()

    # Verify summary JSON
    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    assert summary_data["total_pairs_flagged"] == 2
    assert summary_data["counts_by_classification"][CONFIRMED_LEAKAGE] == 1
    assert summary_data["counts_by_classification"][WITHIN_SPLIT_DUPLICATE] == 1
    assert summary_data["counts_by_severity"][SEVERITY_CRITICAL] == 1
    assert summary_data["counts_by_severity"][SEVERITY_MEDIUM] == 1

    # Verify evidence JSONL and flag redaction
    with open(evidence_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 2
    # Verify Rule 5: secret flag should be redacted in evidence preview
    preview_text = lines[0]["matching_spans"][0]["preview"]
    assert "secret_token_12345" not in preview_text
    assert "[REDACTED]" in preview_text

    # Verify Markdown report content and Rule 11
    report_content = report_md_path.read_text(encoding="utf-8")
    assert "Executive Summary" in report_content
    assert "Frozen Dataset Checksum" in report_content
    assert "Pairwise Split Matrix" in report_content
    assert "Scope & Limitations" in report_content
    assert "098dbee87a570101e0902adabcecca35862bd44b5c31be0cfb07d6b1c4b19b1d" in report_content

    # Verify all tables in markdown report have vertically aligned pipes
    table_pattern = re.compile(r"((?:\|[^\n]+\|\n)+)", re.MULTILINE)
    tables = table_pattern.findall(report_content)
    assert len(tables) >= 5, f"Expected at least 5 markdown tables, found {len(tables)}"

    for t_idx, table_str in enumerate(tables):
        t_lines = [l.strip() for l in table_str.strip().split("\n") if l.strip()]
        if len(t_lines) < 2:
            continue
        pipe_locs = [[i for i, c in enumerate(l) if c == "|"] for l in t_lines]
        target = pipe_locs[0]
        for l_idx, locs in enumerate(pipe_locs):
            assert locs == target, (
                f"Table {t_idx} line {l_idx} pipe alignment failed:\n{t_lines[l_idx]}\nExpected pipes at {target}, got {locs}"
            )
