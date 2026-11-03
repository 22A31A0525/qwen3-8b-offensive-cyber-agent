"""Multi-Signal Pair Classifier Module for 4-Tier Decontamination Audit.

Evaluates and merges candidate matches across Tiers 1-4 for a record pair,
distinguishing cross-split contamination from within-split data quality,
and assigns deterministic classification labels and severity rankings.
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import re

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
    FIELD_RAW_RECORD,
    JACCARD_HIGH_PRIORITY,
    JACCARD_MEDIUM_PRIORITY,
)
from src.audit.tier1_exact import AuditMatch


def _redact_text(text: str) -> str:
    """Redacts sensitive security tokens (flags, credentials) per Rule 5."""
    return re.sub(
        r"(FLAG\{[^}]+\}|CTF\{[^}]+\}|Bearer\s+[A-Za-z0-9_\-\.]+)",
        "[REDACTED]",
        str(text),
        flags=re.IGNORECASE,
    )


def _extract_source_category(rec: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """Extracts provenance source dataset and domain category from record."""
    if not rec or not isinstance(rec, dict):
        return "unknown", "unknown"
    meta = rec.get("metadata", {}) if isinstance(rec.get("metadata"), dict) else {}
    source = meta.get("source_dataset") or rec.get("source") or "unknown"
    category = (
        meta.get("domain")
        or rec.get("domain")
        or rec.get("category")
        or rec.get("track")
        or "unknown"
    )
    return str(source), str(category)


def group_matches_by_pair(
    matches: List[AuditMatch],
) -> Dict[Tuple[str, str], List[AuditMatch]]:
    """Groups a list of candidate AuditMatch objects by (left_id, right_id) pair."""
    grouped: Dict[Tuple[str, str], List[AuditMatch]] = {}
    for m in matches:
        key = (str(m.left_id), str(m.right_id))
        grouped.setdefault(key, []).append(m)
    return grouped


def classify_pair(
    candidate_matches: Union[List[AuditMatch], AuditMatch, Dict[str, Any]],
    left_record: Optional[Dict[str, Any]] = None,
    right_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merges and classifies multi-tier matches for a single pair of records.

    Applies deterministic rule order:
    1. CONFIRMED_LEAKAGE (CRITICAL): Eval flag present in train, identical eval ID, or verbatim answer (J >= 0.95).
    2. TOOL_OBSERVATION_LEAKAGE (CRITICAL): Tool observation collision across splits.
    3. CROSS_SPLIT_DUPLICATE (CRITICAL): Canonical full record hash match across splits.
    4. SUSPECTED_HIGH (HIGH): Meaningful Jaccard >= 0.80 or assistant answer duplicate.
    5. METADATA_COLLISION (HIGH): Scenario/group ID match across splits without confirmed answer leakage.
    6. WITHIN_SPLIT_DUPLICATE (MEDIUM): Match where left_split == right_split (data quality issue).
    7. SUSPECTED_MEDIUM (MEDIUM): Meaningful Jaccard 0.65 <= J < 0.80.
    8. SUSPECTED_LOW (LOW): Overlap with shared security entity or tool target across splits.
    9. BENIGN_OVERLAP (LOW): Public boilerplate scaffolding only with low meaningful overlap.
    10. NEEDS_HUMAN_REVIEW (MEDIUM): Ambiguous evidence.

    Returns:
        Enriched evidence dict conforming to Section 6 of the audit specification.
    """
    if isinstance(candidate_matches, AuditMatch):
        matches = [candidate_matches]
    elif isinstance(candidate_matches, dict):
        if "left_id" in candidate_matches:
            matches = [
                AuditMatch(
                    left_id=str(candidate_matches["left_id"]),
                    right_id=str(candidate_matches.get("right_id", "")),
                    left_split=str(candidate_matches.get("left_split", "unknown")),
                    right_split=str(candidate_matches.get("right_split", "unknown")),
                    match_tier=str(candidate_matches.get("match_tier", "unknown")),
                    field_matched=str(candidate_matches.get("field_matched", "")),
                    details=candidate_matches.get("details", {}),
                )
            ]
        else:
            matches = []
    elif isinstance(candidate_matches, list):
        matches = candidate_matches
    else:
        matches = []

    if not matches:
        return {
            "audit_id": "audit_unknown_unknown",
            "left_id": "unknown",
            "right_id": "unknown",
            "left_split": "unknown",
            "right_split": "unknown",
            "left_source": "unknown",
            "right_source": "unknown",
            "left_category": "unknown",
            "right_category": "unknown",
            "match_type": "none",
            "field_matched": "none",
            "raw_jaccard": 0.0,
            "meaningful_jaccard": 0.0,
            "containment_left_in_right": 0.0,
            "containment_right_in_left": 0.0,
            "matching_spans": [],
            "flag_or_proof_present": False,
            "scenario_id_collision": False,
            "tool_target_collision": False,
            "tool_observation_collision": False,
            "severity": SEVERITY_MEDIUM,
            "classification": NEEDS_HUMAN_REVIEW,
            "reason": "Empty candidate match list",
            "review_status": "pending",
        }

    left_id = str(matches[0].left_id)
    right_id = str(matches[0].right_id)
    left_split = str(matches[0].left_split)
    right_split = str(matches[0].right_split)
    is_cross_split = left_split != right_split

    left_source, left_category = _extract_source_category(left_record)
    right_source, right_category = _extract_source_category(right_record)

    # Aggregate signals across all matches for this pair
    match_tiers: List[str] = list(
        dict.fromkeys(m.match_tier for m in matches if getattr(m, "match_tier", None))
    )
    matched_fields: List[str] = list(
        dict.fromkeys(
            m.field_matched for m in matches if getattr(m, "field_matched", None)
        )
    )

    flag_or_proof_present = False
    scenario_id_collision = False
    tool_target_collision = False
    tool_observation_collision = False
    canonical_hash_collision = False
    id_collision = False

    raw_jaccard = 0.0
    meaningful_jaccard = 0.0
    containment_left_in_right = 0.0
    containment_right_in_left = 0.0
    matching_spans: List[Dict[str, Any]] = []

    for m in matches:
        fld = getattr(m, "field_matched", "")
        tier = getattr(m, "match_tier", "")
        dtl = getattr(m, "details", {}) or {}

        if fld == "id":
            id_collision = True
        elif fld in ("scenario_id", "group_id"):
            scenario_id_collision = True
        elif fld == "canonical_hash":
            canonical_hash_collision = True
            raw_jaccard = max(raw_jaccard, 1.0)
            meaningful_jaccard = max(meaningful_jaccard, 1.0)
            containment_left_in_right = max(containment_left_in_right, 1.0)
            containment_right_in_left = max(containment_right_in_left, 1.0)
        elif fld == "eval_flag":
            flag_or_proof_present = True
            if dtl.get("matched_field") == FIELD_TOOL_OBSERVATION:
                tool_observation_collision = True
        elif fld == FIELD_TOOL_TARGET:
            tool_target_collision = True
        elif fld == FIELD_TOOL_OBSERVATION:
            tool_observation_collision = True
        elif fld == FIELD_FULL_CONVERSATION and tier == "tier2_normalized":
            meaningful_jaccard = max(meaningful_jaccard, 1.0)

        # Inspect details
        if "flag" in dtl or dtl.get("flag_or_proof_present"):
            flag_or_proof_present = True

        if "meaningful_jaccard" in dtl:
            meaningful_jaccard = max(
                meaningful_jaccard, float(dtl["meaningful_jaccard"])
            )
        elif "estimated_jaccard" in dtl:
            meaningful_jaccard = max(
                meaningful_jaccard, float(dtl["estimated_jaccard"])
            )

        if "raw_jaccard" in dtl:
            raw_jaccard = max(raw_jaccard, float(dtl["raw_jaccard"]))

        if "containment_left_in_right" in dtl:
            containment_left_in_right = max(
                containment_left_in_right, float(dtl["containment_left_in_right"])
            )
        if "containment_right_in_left" in dtl:
            containment_right_in_left = max(
                containment_right_in_left, float(dtl["containment_right_in_left"])
            )

        if "matched_spans" in dtl and isinstance(dtl["matched_spans"], list):
            for sp in dtl["matched_spans"]:
                if isinstance(sp, dict):
                    sp_copy = dict(sp)
                    if "preview" in sp_copy:
                        sp_copy["preview"] = _redact_text(sp_copy["preview"])
                    matching_spans.append(sp_copy)

        if "preview" in dtl and not matching_spans:
            matching_spans.append({"preview": _redact_text(dtl["preview"])})

    # Redact any leftover spans
    for sp in matching_spans:
        if "preview" in sp:
            sp["preview"] = _redact_text(sp["preview"])

    # Classification Decision Logic
    classification: str
    severity: str
    reason: str

    # 1. CROSS_SPLIT_DUPLICATE (CRITICAL): Exact full record canonical hash match across splits
    if is_cross_split and canonical_hash_collision:
        classification = CROSS_SPLIT_DUPLICATE
        severity = SEVERITY_CRITICAL
        reason = "Canonical full record SHA-256 hash collision across splits."

    # 2. CONFIRMED_LEAKAGE (CRITICAL)
    elif is_cross_split and (
        flag_or_proof_present
        or id_collision
        or meaningful_jaccard >= 0.95
        or (
            FIELD_FULL_CONVERSATION in matched_fields
            and "tier2_normalized" in match_tiers
        )
    ):
        classification = CONFIRMED_LEAKAGE
        severity = SEVERITY_CRITICAL
        reason = (
            "Known evaluation flag, identical evaluation ID, or verbatim answer/conversation "
            "(meaningful Jaccard >= 0.95) detected across splits."
        )

    # 3. TOOL_OBSERVATION_LEAKAGE (CRITICAL)
    elif is_cross_split and tool_observation_collision:
        classification = TOOL_OBSERVATION_LEAKAGE
        severity = SEVERITY_CRITICAL
        reason = "Tool observation collision detected across splits."

    # 4. SUSPECTED_HIGH (HIGH) - High content overlap overrides pure metadata
    elif is_cross_split and (
        meaningful_jaccard >= JACCARD_HIGH_PRIORITY
        or (
            FIELD_ASSISTANT_ANSWER in matched_fields
            and "tier2_normalized" in match_tiers
        )
    ):
        classification = SUSPECTED_HIGH
        severity = SEVERITY_HIGH
        reason = (
            f"High meaningful conversation overlap (Jaccard = {meaningful_jaccard:.4f} >= "
            f"{JACCARD_HIGH_PRIORITY}) or assistant answer duplicate across splits."
        )

    # 5. METADATA_COLLISION (HIGH)
    elif is_cross_split and scenario_id_collision:
        classification = METADATA_COLLISION
        severity = SEVERITY_HIGH
        reason = "Scenario or group ID collision across splits without confirmed answer leakage."

    # 6. WITHIN_SPLIT_DUPLICATE (MEDIUM)
    elif not is_cross_split and (
        canonical_hash_collision
        or id_collision
        or scenario_id_collision
        or tool_target_collision
        or tool_observation_collision
        or meaningful_jaccard >= 0.50
        or any(
            f in matched_fields
            for f in (
                FIELD_USER_INTENT,
                FIELD_ASSISTANT_ANSWER,
                FIELD_FULL_CONVERSATION,
            )
        )
    ):
        classification = WITHIN_SPLIT_DUPLICATE
        severity = SEVERITY_MEDIUM
        reason = (
            f"Internal duplicate or near-duplicate within split '{left_split}' "
            "(data quality issue, not cross-split contamination)."
        )

    # 7. SUSPECTED_MEDIUM (MEDIUM)
    elif is_cross_split and meaningful_jaccard >= JACCARD_MEDIUM_PRIORITY:
        classification = SUSPECTED_MEDIUM
        severity = SEVERITY_MEDIUM
        reason = (
            f"Medium meaningful conversation overlap ({JACCARD_MEDIUM_PRIORITY} <= "
            f"Jaccard = {meaningful_jaccard:.4f} < {JACCARD_HIGH_PRIORITY}) across splits."
        )

    # 8. SUSPECTED_LOW (LOW)
    elif is_cross_split and (
        tool_target_collision
        or "tier4_minhash" in match_tiers
        or any(
            bool(m.details.get("shared_entities"))
            for m in matches
            if hasattr(m, "details") and isinstance(m.details, dict)
        )
    ):
        classification = SUSPECTED_LOW
        severity = SEVERITY_LOW
        reason = "Low similarity overlap with shared security entity or tool target across splits."

    # 9. BENIGN_OVERLAP (LOW)
    elif raw_jaccard > 0 and meaningful_jaccard < 0.10:
        classification = BENIGN_OVERLAP
        severity = SEVERITY_LOW
        reason = "Shared tool schema or public boilerplate scaffolding only with low meaningful overlap."

    # 10. NEEDS_HUMAN_REVIEW (MEDIUM)
    else:
        classification = NEEDS_HUMAN_REVIEW
        severity = SEVERITY_MEDIUM
        reason = "Ambiguous multi-signal overlap requiring human review."

    audit_id = f"audit_{left_id}_{right_id}"

    return {
        "audit_id": audit_id,
        "left_id": left_id,
        "right_id": right_id,
        "left_split": left_split,
        "right_split": right_split,
        "left_source": left_source,
        "right_source": right_source,
        "left_category": left_category,
        "right_category": right_category,
        "match_type": ", ".join(match_tiers) if match_tiers else "none",
        "field_matched": ", ".join(matched_fields) if matched_fields else "none",
        "raw_jaccard": round(raw_jaccard, 4),
        "meaningful_jaccard": round(meaningful_jaccard, 4),
        "containment_left_in_right": round(containment_left_in_right, 4),
        "containment_right_in_left": round(containment_right_in_left, 4),
        "matching_spans": matching_spans,
        "flag_or_proof_present": flag_or_proof_present,
        "scenario_id_collision": scenario_id_collision,
        "tool_target_collision": tool_target_collision,
        "tool_observation_collision": tool_observation_collision,
        "severity": severity,
        "classification": classification,
        "reason": reason,
        "review_status": "pending",
    }
