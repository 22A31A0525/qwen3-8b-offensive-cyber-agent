"""Unit tests for src.audit.config."""
import pytest
from src.audit.config import (
    JACCARD_HIGH_PRIORITY,
    JACCARD_MEDIUM_PRIORITY,
    NGRAM_SIZE,
    MINHASH_NUM_PERM,
    MINHASH_NUM_BANDS,
    MINHASH_ROWS_PER_BAND,
    DETERMINISTIC_SEED,
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
    FIELD_RAW_RECORD,
    AuditConfig,
    AuditClassification,
    AuditSeverity,
    AuditField,
)


def test_audit_constants_values():
    assert JACCARD_HIGH_PRIORITY == 0.80
    assert JACCARD_MEDIUM_PRIORITY == 0.65
    assert NGRAM_SIZE == 13
    assert MINHASH_NUM_PERM == 128
    assert MINHASH_NUM_BANDS == 16
    assert MINHASH_ROWS_PER_BAND == 8
    assert DETERMINISTIC_SEED == 42


def test_audit_classification_constants():
    assert CONFIRMED_LEAKAGE == "confirmed_leakage"
    assert SUSPECTED_HIGH == "suspected_high"
    assert SUSPECTED_MEDIUM == "suspected_medium"
    assert SUSPECTED_LOW == "suspected_low"
    assert BENIGN_OVERLAP == "benign_overlap"
    assert WITHIN_SPLIT_DUPLICATE == "within_split_duplicate"
    assert CROSS_SPLIT_DUPLICATE == "cross_split_duplicate"
    assert TOOL_OBSERVATION_LEAKAGE == "tool_observation_leakage"
    assert METADATA_COLLISION == "metadata_collision"
    assert NEEDS_HUMAN_REVIEW == "needs_human_review"


def test_audit_severity_constants():
    assert SEVERITY_CRITICAL == "critical"
    assert SEVERITY_HIGH == "high"
    assert SEVERITY_MEDIUM == "medium"
    assert SEVERITY_LOW == "low"


def test_audit_target_fields():
    assert FIELD_USER_INTENT == "user_intent"
    assert FIELD_ASSISTANT_ANSWER == "assistant_answer"
    assert FIELD_TOOL_TARGET == "tool_call_target"
    assert FIELD_TOOL_OBSERVATION == "tool_observation"
    assert FIELD_FULL_CONVERSATION == "full_meaningful_conversation"
    assert FIELD_RAW_RECORD == "raw_full_record"


def test_audit_enums_and_dataclass():
    assert AuditClassification.CONFIRMED_LEAKAGE == CONFIRMED_LEAKAGE
    assert AuditSeverity.CRITICAL == SEVERITY_CRITICAL
    assert AuditField.USER_INTENT == FIELD_USER_INTENT

    cfg = AuditConfig()
    assert cfg.jaccard_high_priority == 0.80
    assert cfg.jaccard_medium_priority == 0.65
    assert cfg.ngram_size == 13
    assert cfg.minhash_num_perm == 128
    assert cfg.minhash_num_bands == 16
    assert cfg.minhash_rows_per_band == 8
    assert cfg.deterministic_seed == 42
