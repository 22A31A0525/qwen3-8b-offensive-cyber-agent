"""Audit Foundation and Configuration Module.

Defines thresholds, classification labels, severity levels, target fields,
and configuration dataclasses for the 4-tier decontamination audit pipeline.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

# Thresholds & Parameters
JACCARD_HIGH_PRIORITY: float = 0.80
JACCARD_MEDIUM_PRIORITY: float = 0.65
NGRAM_SIZE: int = 13
MINHASH_NUM_PERM: int = 128
MINHASH_NUM_BANDS: int = 16
MINHASH_ROWS_PER_BAND: int = 8
DETERMINISTIC_SEED: int = 42

# Classification Labels
CONFIRMED_LEAKAGE: str = "confirmed_leakage"
SUSPECTED_HIGH: str = "suspected_high"
SUSPECTED_MEDIUM: str = "suspected_medium"
SUSPECTED_LOW: str = "suspected_low"
BENIGN_OVERLAP: str = "benign_overlap"
WITHIN_SPLIT_DUPLICATE: str = "within_split_duplicate"
CROSS_SPLIT_DUPLICATE: str = "cross_split_duplicate"
TOOL_OBSERVATION_LEAKAGE: str = "tool_observation_leakage"
METADATA_COLLISION: str = "metadata_collision"
NEEDS_HUMAN_REVIEW: str = "needs_human_review"

# Severity Levels
SEVERITY_CRITICAL: str = "critical"
SEVERITY_HIGH: str = "high"
SEVERITY_MEDIUM: str = "medium"
SEVERITY_LOW: str = "low"

# Target Fields for Auditing
FIELD_USER_INTENT: str = "user_intent"
FIELD_ASSISTANT_ANSWER: str = "assistant_answer"
FIELD_TOOL_TARGET: str = "tool_call_target"
FIELD_TOOL_OBSERVATION: str = "tool_observation"
FIELD_FULL_CONVERSATION: str = "full_meaningful_conversation"
FIELD_RAW_RECORD: str = "raw_full_record"


class AuditClassification(str, Enum):
    CONFIRMED_LEAKAGE = CONFIRMED_LEAKAGE
    SUSPECTED_HIGH = SUSPECTED_HIGH
    SUSPECTED_MEDIUM = SUSPECTED_MEDIUM
    SUSPECTED_LOW = SUSPECTED_LOW
    BENIGN_OVERLAP = BENIGN_OVERLAP
    WITHIN_SPLIT_DUPLICATE = WITHIN_SPLIT_DUPLICATE
    CROSS_SPLIT_DUPLICATE = CROSS_SPLIT_DUPLICATE
    TOOL_OBSERVATION_LEAKAGE = TOOL_OBSERVATION_LEAKAGE
    METADATA_COLLISION = METADATA_COLLISION
    NEEDS_HUMAN_REVIEW = NEEDS_HUMAN_REVIEW


class AuditSeverity(str, Enum):
    CRITICAL = SEVERITY_CRITICAL
    HIGH = SEVERITY_HIGH
    MEDIUM = SEVERITY_MEDIUM
    LOW = SEVERITY_LOW


class AuditField(str, Enum):
    USER_INTENT = FIELD_USER_INTENT
    ASSISTANT_ANSWER = FIELD_ASSISTANT_ANSWER
    TOOL_TARGET = FIELD_TOOL_TARGET
    TOOL_OBSERVATION = FIELD_TOOL_OBSERVATION
    FULL_CONVERSATION = FIELD_FULL_CONVERSATION
    RAW_RECORD = FIELD_RAW_RECORD


@dataclass(frozen=True)
class AuditConfig:
    jaccard_high_priority: float = JACCARD_HIGH_PRIORITY
    jaccard_medium_priority: float = JACCARD_MEDIUM_PRIORITY
    ngram_size: int = NGRAM_SIZE
    minhash_num_perm: int = MINHASH_NUM_PERM
    minhash_num_bands: int = MINHASH_NUM_BANDS
    minhash_rows_per_band: int = MINHASH_ROWS_PER_BAND
    deterministic_seed: int = DETERMINISTIC_SEED


@dataclass
class AuditMatch:
    left_id: str
    right_id: str
    left_split: str
    right_split: str
    match_tier: str
    field_matched: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the AuditMatch to a serializable dictionary."""
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "left_split": self.left_split,
            "right_split": self.right_split,
            "match_tier": self.match_tier,
            "field_matched": self.field_matched,
            "details": self.details,
        }

