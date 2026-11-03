"""Tier 1 Exact Identity and Metadata Audit Module.

Detects exact collisions across dataset records:
- Exact Record ID collisions
- Scenario / Group ID collisions (from top-level or metadata)
- Canonical full record SHA-256 hash collisions
- Verbatim evaluation flag / proof string leakage in tool observations or assistant prose
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
import hashlib

from src.audit.config import (
    FIELD_ASSISTANT_ANSWER,
    FIELD_TOOL_OBSERVATION,
)
from src.audit.normalizer import canonical_record_hash, extract_audit_fields


@dataclass
class AuditMatch:
    """Represents an audit candidate match between two records or a record and an eval token."""

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


def _get_record_id(record: Dict[str, Any], default: str = "") -> str:
    """Extracts record ID with fallback to conv_id."""
    rid = record.get("id") or record.get("conv_id")
    return str(rid).strip() if rid else default


def _get_scenario_id(record: Dict[str, Any]) -> Optional[str]:
    """Extracts scenario_id from top-level or metadata dict."""
    sid = record.get("scenario_id")
    if not sid and isinstance(record.get("metadata"), dict):
        sid = record["metadata"].get("scenario_id")
    if sid:
        sid_str = str(sid).strip()
        return sid_str if sid_str else None
    return None


def _get_group_id(record: Dict[str, Any]) -> Optional[str]:
    """Extracts group_id from top-level or metadata dict."""
    gid = record.get("group_id")
    if not gid and isinstance(record.get("metadata"), dict):
        gid = record["metadata"].get("group_id")
    if gid:
        gid_str = str(gid).strip()
        return gid_str if gid_str else None
    return None


def _extract_flag_preview(text: str, flag: str, context_len: int = 40) -> str:
    """Extracts a truncated window of text around the matched flag."""
    idx = text.find(flag)
    if idx == -1:
        return text[:100]
    start = max(0, idx - context_len)
    end = min(len(text), idx + len(flag) + context_len)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def run_tier1_audit(
    left_records: List[Dict[str, Any]],
    right_records: List[Dict[str, Any]],
    left_split: str,
    right_split: str,
    known_eval_flags: Optional[Union[Set[str], List[str], Dict[str, Any]]] = None,
) -> List[AuditMatch]:
    """Executes Tier 1 exact identity, metadata, hash, and flag collision audit.

    Args:
        left_records: Records from left split (e.g. train).
        right_records: Records from right split (e.g. test, val, or train).
        left_split: Label for left split (e.g. "train").
        right_split: Label for right split (e.g. "test", "val", "train").
        known_eval_flags: Optional set, list, or dict of evaluation flags/secrets.

    Returns:
        List of AuditMatch objects for detected collisions.
    """
    matches: List[AuditMatch] = []
    is_within_split = left_split == right_split

    if not left_records or (not right_records and not known_eval_flags):
        return matches

    # 1. Build inverted indexes on right_records
    right_by_id: Dict[str, List[str]] = {}
    right_by_scenario: Dict[str, List[str]] = {}
    right_by_group: Dict[str, List[str]] = {}
    right_by_hash: Dict[str, List[str]] = {}
    right_flag_map: Dict[str, List[str]] = {}

    eval_flag_set: Set[str] = set()
    if known_eval_flags:
        eval_flag_set = {str(f).strip() for f in known_eval_flags if str(f).strip()}

    for r in right_records:
        rid = _get_record_id(r)
        if not rid:
            continue

        right_by_id.setdefault(rid, []).append(rid)

        sid = _get_scenario_id(r)
        if sid:
            right_by_scenario.setdefault(sid, []).append(rid)

        gid = _get_group_id(r)
        if gid:
            right_by_group.setdefault(gid, []).append(rid)

        chash = canonical_record_hash(r)
        right_by_hash.setdefault(chash, []).append(rid)

        if eval_flag_set:
            r_fields = extract_audit_fields(r)
            r_obs = r_fields.get(FIELD_TOOL_OBSERVATION, "")
            r_ans = r_fields.get(FIELD_ASSISTANT_ANSWER, "")
            for f in eval_flag_set:
                if f in r_obs or f in r_ans:
                    right_flag_map.setdefault(f, []).append(rid)

    # 2. Audit left_records
    for l in left_records:
        lid = _get_record_id(l)
        if not lid:
            continue

        # Check a: Exact Record ID collision (only cross-split; within-split requires lid < rid)
        if not is_within_split:
            if lid in right_by_id:
                for rid in right_by_id[lid]:
                    matches.append(
                        AuditMatch(
                            left_id=lid,
                            right_id=rid,
                            left_split=left_split,
                            right_split=right_split,
                            match_tier="tier1_exact",
                            field_matched="id",
                            details={"matched_id": lid},
                        )
                    )

        # Check b: Scenario ID collision
        sid = _get_scenario_id(l)
        if sid and sid in right_by_scenario:
            for rid in right_by_scenario[sid]:
                if is_within_split and lid >= rid:
                    continue
                matches.append(
                    AuditMatch(
                        left_id=lid,
                        right_id=rid,
                        left_split=left_split,
                        right_split=right_split,
                        match_tier="tier1_exact",
                        field_matched="scenario_id",
                        details={"scenario_id": sid},
                    )
                )

        # Check b2: Group ID collision
        gid = _get_group_id(l)
        if gid and gid in right_by_group:
            for rid in right_by_group[gid]:
                if is_within_split and lid >= rid:
                    continue
                matches.append(
                    AuditMatch(
                        left_id=lid,
                        right_id=rid,
                        left_split=left_split,
                        right_split=right_split,
                        match_tier="tier1_exact",
                        field_matched="group_id",
                        details={"group_id": gid},
                    )
                )

        # Check c: Canonical full record SHA-256 hash collision
        chash = canonical_record_hash(l)
        if chash in right_by_hash:
            for rid in right_by_hash[chash]:
                if is_within_split and lid >= rid:
                    continue
                matches.append(
                    AuditMatch(
                        left_id=lid,
                        right_id=rid,
                        left_split=left_split,
                        right_split=right_split,
                        match_tier="tier1_exact",
                        field_matched="canonical_hash",
                        details={"canonical_hash": chash},
                    )
                )

        # Check d: Eval flag exact value collision
        if eval_flag_set:
            l_fields = extract_audit_fields(l)
            l_obs = l_fields.get(FIELD_TOOL_OBSERVATION, "")
            l_ans = l_fields.get(FIELD_ASSISTANT_ANSWER, "")

            for f in eval_flag_set:
                matched_field: Optional[str] = None
                preview: Optional[str] = None

                if f in l_obs:
                    matched_field = FIELD_TOOL_OBSERVATION
                    preview = _extract_flag_preview(l_obs, f)
                elif f in l_ans:
                    matched_field = FIELD_ASSISTANT_ANSWER
                    preview = _extract_flag_preview(l_ans, f)

                if matched_field:
                    paired_rids = right_flag_map.get(f, [])
                    if paired_rids:
                        for rid in paired_rids:
                            if is_within_split and lid >= rid:
                                continue
                            matches.append(
                                AuditMatch(
                                    left_id=lid,
                                    right_id=rid,
                                    left_split=left_split,
                                    right_split=right_split,
                                    match_tier="tier1_exact",
                                    field_matched="eval_flag",
                                    details={
                                        "flag": f,
                                        "matched_field": matched_field,
                                        "preview": preview,
                                    },
                                )
                            )
                    else:
                        rid = f if not isinstance(known_eval_flags, dict) else str(known_eval_flags.get(f, f))
                        matches.append(
                            AuditMatch(
                                left_id=lid,
                                right_id=rid,
                                left_split=left_split,
                                right_split=right_split,
                                match_tier="tier1_exact",
                                field_matched="eval_flag",
                                details={
                                    "flag": f,
                                    "matched_field": matched_field,
                                    "preview": preview,
                                },
                            )
                        )

    return matches
