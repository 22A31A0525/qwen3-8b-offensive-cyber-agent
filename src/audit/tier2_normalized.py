"""Tier 2 Normalized Exact Duplicate Audit Module.

Detects exact collisions on normalized text across individual conversation components:
- user_intent
- assistant_answer
- tool_call_target
- tool_observation
- full_meaningful_conversation

Uses inverted hash index (SHA-256 over normalized text) for fast O(N) candidate detection.
Skips empty fields to prevent false positive collisions on missing/uninvoked tools.
"""
from typing import Any, Dict, List, Optional
import hashlib
import re

from src.audit.config import (
    FIELD_USER_INTENT,
    FIELD_ASSISTANT_ANSWER,
    FIELD_TOOL_TARGET,
    FIELD_TOOL_OBSERVATION,
    FIELD_FULL_CONVERSATION,
)
from src.audit.normalizer import extract_audit_fields, normalize_text
from src.audit.tier1_exact import AuditMatch

AUDIT_FIELDS = [
    FIELD_USER_INTENT,
    FIELD_ASSISTANT_ANSWER,
    FIELD_TOOL_TARGET,
    FIELD_TOOL_OBSERVATION,
    FIELD_FULL_CONVERSATION,
]


def _get_record_id(record: Dict[str, Any], default: str = "") -> str:
    """Extracts record ID with fallback to conv_id."""
    rid = record.get("id") or record.get("conv_id")
    return str(rid).strip() if rid else default


def _create_preview(text: str, max_chars: int = 150) -> str:
    """Creates a truncated, redacted preview string of matched content."""
    # Redact sensitive tokens per Rule 5
    redacted = re.sub(
        r"(FLAG\{[^}]+\}|CTF\{[^}]+\}|Bearer\s+[A-Za-z0-9_\-\.]+)",
        "[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    if len(redacted) <= max_chars:
        return redacted
    return redacted[:max_chars] + "..."


def run_tier2_audit(
    left_records: List[Dict[str, Any]],
    right_records: List[Dict[str, Any]],
    left_split: str,
    right_split: str,
) -> List[AuditMatch]:
    """Executes Tier 2 normalized exact duplicate audit across all 5 target fields.

    Args:
        left_records: Records from left split.
        right_records: Records from right split.
        left_split: Label for left split (e.g. "train").
        right_split: Label for right split (e.g. "test", "val", "train").

    Returns:
        List of AuditMatch objects with match_tier="tier2_normalized".
    """
    matches: List[AuditMatch] = []
    is_within_split = left_split == right_split

    if not left_records or not right_records:
        return matches

    # 1. Build inverted hash index on right_records for each field
    # inverted_indexes[field_name][sha256_hash] = list of right_ids
    inverted_indexes: Dict[str, Dict[str, List[str]]] = {f: {} for f in AUDIT_FIELDS}

    for r in right_records:
        rid = _get_record_id(r)
        if not rid:
            continue

        r_fields = extract_audit_fields(r) if "messages" in r else r

        for field_name in AUDIT_FIELDS:
            raw_text = r_fields.get(field_name, "")
            norm = normalize_text(raw_text)
            if not norm:
                # Never index empty fields to prevent false positive collisions
                continue

            h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            inverted_indexes[field_name].setdefault(h, []).append(rid)

    # 2. Scan left_records against inverted index
    for l in left_records:
        lid = _get_record_id(l)
        if not lid:
            continue

        l_fields = extract_audit_fields(l) if "messages" in l else l

        for field_name in AUDIT_FIELDS:
            raw_text = l_fields.get(field_name, "")
            norm = normalize_text(raw_text)
            if not norm:
                continue

            h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            if h in inverted_indexes[field_name]:
                for rid in inverted_indexes[field_name][h]:
                    if is_within_split and lid >= rid:
                        continue

                    matches.append(
                        AuditMatch(
                            left_id=lid,
                            right_id=rid,
                            left_split=left_split,
                            right_split=right_split,
                            match_tier="tier2_normalized",
                            field_matched=field_name,
                            details={
                                "hash": h,
                                "preview": _create_preview(norm),
                                "matched_length": len(norm),
                            },
                        )
                    )

    return matches
