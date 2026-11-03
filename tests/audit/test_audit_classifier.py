"""Unit tests for Multi-Signal Pair Classifier (src.audit.classifier)."""
import pytest
from src.audit.config import (
    CONFIRMED_LEAKAGE,
    SUSPECTED_HIGH,
    SUSPECTED_MEDIUM,
    SUSPECTED_LOW,
    BENIGN_OVERLAP,
    WITHIN_SPLIT_DUPLICATE,
    CROSS_SPLIT_DUPLICATE,
    TOOL_OBSERVATION_LEAKAGE,
    METADATA_COLLISION,
    NEEDS_HUMAN_REVIEW,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    FIELD_USER_INTENT,
    FIELD_ASSISTANT_ANSWER,
    FIELD_TOOL_TARGET,
    FIELD_TOOL_OBSERVATION,
    FIELD_FULL_CONVERSATION,
)
from src.audit.tier1_exact import AuditMatch
from src.audit.classifier import classify_pair, group_matches_by_pair


REQUIRED_SECTION_6_FIELDS = {
    "audit_id",
    "left_id",
    "right_id",
    "left_split",
    "right_split",
    "left_source",
    "right_source",
    "left_category",
    "right_category",
    "match_type",
    "field_matched",
    "raw_jaccard",
    "meaningful_jaccard",
    "containment_left_in_right",
    "containment_right_in_left",
    "matching_spans",
    "flag_or_proof_present",
    "scenario_id_collision",
    "tool_target_collision",
    "tool_observation_collision",
    "severity",
    "classification",
    "reason",
    "review_status",
}


def test_classify_pair_section6_schema_compliance():
    """Verifies that classify_pair emits all 24 required fields with proper defaults."""
    match = AuditMatch(
        left_id="train_001",
        right_id="test_001",
        left_split="train",
        right_split="test",
        match_tier="tier1_exact",
        field_matched="id",
        details={"matched_id": "test_001"},
    )
    result = classify_pair([match])

    assert isinstance(result, dict)
    missing = REQUIRED_SECTION_6_FIELDS - set(result.keys())
    assert not missing, f"Missing Section 6 fields: {missing}"
    assert result["review_status"] == "pending"
    assert result["left_id"] == "train_001"
    assert result["right_id"] == "test_001"


def test_classify_pair_confirmed_leakage_eval_flag():
    """Eval flag present in train assistant or tool output must be CONFIRMED_LEAKAGE."""
    match = AuditMatch(
        left_id="train_flag_leak",
        right_id="test_target_01",
        left_split="train",
        right_split="test",
        match_tier="tier1_exact",
        field_matched="eval_flag",
        details={"flag": "FLAG{eval_secret_123}", "preview": "FLAG{...}"},
    )
    res = classify_pair([match])
    assert res["classification"] == CONFIRMED_LEAKAGE
    assert res["severity"] == SEVERITY_CRITICAL
    assert res["flag_or_proof_present"] is True


def test_classify_pair_confirmed_leakage_identical_id():
    """Identical evaluation ID in training set must be CONFIRMED_LEAKAGE."""
    match = AuditMatch(
        left_id="id_match_cross",
        right_id="id_match_cross",
        left_split="train",
        right_split="test",
        match_tier="tier1_exact",
        field_matched="id",
        details={"matched_id": "id_match_cross"},
    )
    res = classify_pair([match])
    assert res["classification"] == CONFIRMED_LEAKAGE
    assert res["severity"] == SEVERITY_CRITICAL


def test_classify_pair_confirmed_leakage_identical_conversation():
    """Near-identical conversation (J >= 0.95) across splits must be CONFIRMED_LEAKAGE."""
    match = AuditMatch(
        left_id="train_near_dup",
        right_id="test_target",
        left_split="train",
        right_split="test",
        match_tier="tier3_ngram",
        field_matched=FIELD_FULL_CONVERSATION,
        details={"meaningful_jaccard": 0.96, "raw_jaccard": 0.97},
    )
    res = classify_pair([match])
    assert res["classification"] == CONFIRMED_LEAKAGE
    assert res["severity"] == SEVERITY_CRITICAL


def test_classify_pair_tool_observation_leakage():
    """Cross-split tool observation collision must be TOOL_OBSERVATION_LEAKAGE."""
    match = AuditMatch(
        left_id="train_tool_obs",
        right_id="test_target",
        left_split="train",
        right_split="test",
        match_tier="tier2_normalized",
        field_matched=FIELD_TOOL_OBSERVATION,
        details={"preview": "Database dumped output"},
    )
    res = classify_pair([match])
    assert res["classification"] == TOOL_OBSERVATION_LEAKAGE
    assert res["severity"] == SEVERITY_CRITICAL
    assert res["tool_observation_collision"] is True


def test_classify_pair_cross_split_duplicate():
    """Canonical hash collision across splits must be CROSS_SPLIT_DUPLICATE."""
    match = AuditMatch(
        left_id="train_exact_dup",
        right_id="test_exact_dup",
        left_split="train",
        right_split="test",
        match_tier="tier1_exact",
        field_matched="canonical_hash",
        details={"canonical_hash": "deadbeef1234"},
    )
    res = classify_pair([match])
    assert res["classification"] == CROSS_SPLIT_DUPLICATE
    assert res["severity"] == SEVERITY_CRITICAL


def test_classify_pair_metadata_collision():
    """Scenario or group ID collision across splits without flag/answer leakage is METADATA_COLLISION."""
    match = AuditMatch(
        left_id="train_scen_01",
        right_id="val_scen_01",
        left_split="train",
        right_split="val",
        match_tier="tier1_exact",
        field_matched="scenario_id",
        details={"scenario_id": "cve_2024_1234_scenario"},
    )
    res = classify_pair([match])
    assert res["classification"] == METADATA_COLLISION
    assert res["severity"] == SEVERITY_HIGH
    assert res["scenario_id_collision"] is True


def test_classify_pair_within_split_duplicate():
    """Internal duplicates within same split must be WITHIN_SPLIT_DUPLICATE."""
    match = AuditMatch(
        left_id="train_dup_a",
        right_id="train_dup_b",
        left_split="train",
        right_split="train",
        match_tier="tier1_exact",
        field_matched="canonical_hash",
        details={"canonical_hash": "aabbccdd1122"},
    )
    res = classify_pair([match])
    assert res["classification"] == WITHIN_SPLIT_DUPLICATE
    assert res["severity"] == SEVERITY_MEDIUM


def test_classify_pair_suspected_high():
    """Cross-split meaningful Jaccard >= 0.80 must be SUSPECTED_HIGH."""
    match = AuditMatch(
        left_id="train_high_jaccard",
        right_id="test_high_jaccard",
        left_split="train",
        right_split="test",
        match_tier="tier3_ngram",
        field_matched=FIELD_FULL_CONVERSATION,
        details={"meaningful_jaccard": 0.84, "raw_jaccard": 0.86},
    )
    res = classify_pair([match])
    assert res["classification"] == SUSPECTED_HIGH
    assert res["severity"] == SEVERITY_HIGH


def test_classify_pair_suspected_medium():
    """Cross-split meaningful Jaccard between 0.65 and 0.80 must be SUSPECTED_MEDIUM."""
    match = AuditMatch(
        left_id="train_med_jaccard",
        right_id="val_med_jaccard",
        left_split="train",
        right_split="val",
        match_tier="tier3_ngram",
        field_matched=FIELD_FULL_CONVERSATION,
        details={"meaningful_jaccard": 0.72, "raw_jaccard": 0.75},
    )
    res = classify_pair([match])
    assert res["classification"] == SUSPECTED_MEDIUM
    assert res["severity"] == SEVERITY_MEDIUM


def test_classify_pair_suspected_low():
    """Low Jaccard with shared entity or tool target across splits must be SUSPECTED_LOW."""
    match = AuditMatch(
        left_id="train_low",
        right_id="test_low",
        left_split="train",
        right_split="test",
        match_tier="tier4_minhash",
        field_matched=FIELD_FULL_CONVERSATION,
        details={"estimated_jaccard": 0.45, "shared_entities": {"cves": ["CVE-2023-1111"]}},
    )
    res = classify_pair([match])
    assert res["classification"] == SUSPECTED_LOW
    assert res["severity"] == SEVERITY_LOW


def test_classify_pair_benign_overlap():
    """Boilerplate scaffolding overlap with zero/negligible meaningful content must be BENIGN_OVERLAP."""
    match = AuditMatch(
        left_id="train_benign",
        right_id="test_benign",
        left_split="train",
        right_split="test",
        match_tier="tier3_ngram",
        field_matched=FIELD_FULL_CONVERSATION,
        details={"meaningful_jaccard": 0.02, "raw_jaccard": 0.65},
    )
    res = classify_pair([match])
    assert res["classification"] == BENIGN_OVERLAP
    assert res["severity"] == SEVERITY_LOW


def test_classify_pair_merges_multiple_matches():
    """Multiple matches across tiers for the same pair must be merged into one enriched evidence record."""
    m1 = AuditMatch(
        left_id="pair_l",
        right_id="pair_r",
        left_split="train",
        right_split="test",
        match_tier="tier1_exact",
        field_matched="scenario_id",
        details={"scenario_id": "sec_eval_01"},
    )
    m2 = AuditMatch(
        left_id="pair_l",
        right_id="pair_r",
        left_split="train",
        right_split="test",
        match_tier="tier3_ngram",
        field_matched=FIELD_FULL_CONVERSATION,
        details={
            "meaningful_jaccard": 0.88,
            "raw_jaccard": 0.90,
            "containment_left_in_right": 0.91,
            "containment_right_in_left": 0.89,
            "matched_spans": [{"left_char_start": 0, "preview": "nmap scan..."}],
        },
    )
    res = classify_pair([m1, m2])
    assert "tier1_exact" in res["match_type"]
    assert "tier3_ngram" in res["match_type"]
    assert res["scenario_id_collision"] is True
    assert res["meaningful_jaccard"] == 0.88
    assert res["containment_left_in_right"] == 0.91
    assert len(res["matching_spans"]) == 1
    # Meaningful Jaccard 0.88 overrides metadata collision into SUSPECTED_HIGH
    assert res["classification"] == SUSPECTED_HIGH


def test_classify_pair_record_metadata_enrichment():
    """Passing left_record and right_record enriches source and category fields."""
    match = AuditMatch(
        left_id="rec_l",
        right_id="rec_r",
        left_split="train",
        right_split="test",
        match_tier="tier2_normalized",
        field_matched=FIELD_USER_INTENT,
        details={"preview": "Run exploit"},
    )
    left_rec = {
        "id": "rec_l",
        "metadata": {"source_dataset": "exploit_v3", "domain": "exploit"},
    }
    right_rec = {
        "id": "rec_r",
        "metadata": {"source_dataset": "eval_cyber_v1", "domain": "reasoning"},
    }
    res = classify_pair([match], left_record=left_rec, right_record=right_rec)
    assert res["left_source"] == "exploit_v3"
    assert res["right_source"] == "eval_cyber_v1"
    assert res["left_category"] == "exploit"
    assert res["right_category"] == "reasoning"


def test_group_matches_by_pair():
    """group_matches_by_pair correctly groups matches by (left_id, right_id)."""
    m1 = AuditMatch("a", "b", "train", "test", "tier1_exact", "id")
    m2 = AuditMatch("a", "b", "train", "test", "tier3_ngram", "full")
    m3 = AuditMatch("c", "d", "train", "test", "tier2_normalized", "intent")

    grouped = group_matches_by_pair([m1, m2, m3])
    assert ("a", "b") in grouped
    assert ("c", "d") in grouped
    assert len(grouped[("a", "b")]) == 2
    assert len(grouped[("c", "d")]) == 1
