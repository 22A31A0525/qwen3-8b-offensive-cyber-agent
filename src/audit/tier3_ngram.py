"""Tier 3 Meaningful 13-Gram Overlap Audit Module.

Calculates continuous 13-gram overlap, Jaccard similarity, and containment metrics
across meaningful conversation components (excluding system prompts and tool declarations),
with inverted index candidate pruning and matched span extraction.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import re

from src.audit.config import (
    AuditMatch,
    NGRAM_SIZE,
    FIELD_USER_INTENT,
    FIELD_ASSISTANT_ANSWER,
    FIELD_TOOL_TARGET,
    FIELD_TOOL_OBSERVATION,
    FIELD_FULL_CONVERSATION,
    FIELD_RAW_RECORD,
)
from src.audit.normalizer import (
    normalize_text,
    extract_audit_fields,
)


def _tokenize(text: Optional[str]) -> List[str]:
    """Tokenizes text into normalized word tokens."""
    if not text:
        return []
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def extract_ngrams(tokens: List[str], n: int = NGRAM_SIZE) -> Set[Tuple[str, ...]]:
    """Extracts continuous sliding-window n-grams of size n from a sequence of tokens.

    Args:
        tokens: List of token strings.
        n: Window size (default 13).

    Returns:
        Set of n-gram tuples. Empty set if len(tokens) < n or n < 1.
    """
    if not tokens or n < 1 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def compute_ngram_overlap(
    left_text: str,
    right_text: str,
    n: int = NGRAM_SIZE,
) -> Dict[str, Any]:
    """Computes Jaccard similarity, containment, and shared n-gram counts between two texts.

    Args:
        left_text: Source text (A).
        right_text: Target text (B).
        n: Window size (default 13).

    Returns:
        Dictionary with jaccard, containment_left_in_right, containment_right_in_left,
        and shared_ngrams_count.
    """
    tokens_left = _tokenize(left_text)
    tokens_right = _tokenize(right_text)

    ngrams_left = extract_ngrams(tokens_left, n=n)
    ngrams_right = extract_ngrams(tokens_right, n=n)

    if not ngrams_left and not ngrams_right:
        return {
            "jaccard": 0.0,
            "containment_left_in_right": 0.0,
            "containment_right_in_left": 0.0,
            "shared_ngrams_count": 0,
        }

    shared = ngrams_left & ngrams_right
    shared_count = len(shared)
    union_count = len(ngrams_left | ngrams_right)

    jaccard = (shared_count / union_count) if union_count > 0 else 0.0
    containment_left = (shared_count / len(ngrams_left)) if ngrams_left else 0.0
    containment_right = (shared_count / len(ngrams_right)) if ngrams_right else 0.0

    return {
        "jaccard": jaccard,
        "containment_left_in_right": containment_left,
        "containment_right_in_left": containment_right,
        "shared_ngrams_count": shared_count,
    }


def _find_matching_spans(
    left_text: str,
    shared_ngrams: Set[Tuple[str, ...]],
    max_spans: int = 3,
) -> List[Dict[str, Any]]:
    """Extracts character offsets and truncated previews for shared n-grams in left_text."""
    spans: List[Dict[str, Any]] = []
    if not left_text or not shared_ngrams:
        return spans

    lower_text = left_text.lower()
    for ngram in list(shared_ngrams)[:max_spans]:
        phrase = " ".join(ngram)
        idx = lower_text.find(phrase)
        if idx != -1:
            preview = left_text[idx : idx + len(phrase)]
            if len(preview) > 150:
                preview = preview[:147] + "..."
            spans.append({
                "start": idx,
                "end": idx + len(phrase),
                "preview": preview,
            })
        else:
            first_tok = ngram[0]
            last_tok = ngram[-1]
            first_idx = lower_text.find(first_tok)
            if first_idx != -1:
                last_idx = lower_text.find(last_tok, first_idx)
                end = (last_idx + len(last_tok)) if last_idx != -1 else (first_idx + len(first_tok))
                preview = left_text[first_idx:end]
                if len(preview) > 150:
                    preview = preview[:147] + "..."
                spans.append({
                    "start": first_idx,
                    "end": end,
                    "preview": preview,
                })
    return spans


def run_tier3_audit(
    left_records: List[Dict[str, Any]],
    right_records: List[Dict[str, Any]],
    left_split: str,
    right_split: str,
    threshold: float = 0.50,
) -> List[AuditMatch]:
    """Executes Tier 3 Meaningful 13-Gram overlap audit between two splits.

    Uses an inverted n-gram index over right records to prune candidate pairs sharing
    0 n-grams before computing full meaningful and field-level Jaccard similarities.

    Args:
        left_records: Records from source split.
        right_records: Records from comparison split.
        left_split: Name of source split (e.g. "train").
        right_split: Name of comparison split (e.g. "test").
        threshold: Minimum Jaccard threshold for flagging an AuditMatch (default 0.50).

    Returns:
        List of AuditMatch objects meeting the threshold.
    """
    matches: List[AuditMatch] = []
    is_within_split = (left_split == right_split)

    # 1. Preprocess and extract audit fields for right_records
    right_extracted: List[Tuple[str, Dict[str, str], Set[Tuple[str, ...]]]] = []
    inverted_index: Dict[Tuple[str, ...], List[int]] = {}

    for idx, rec in enumerate(right_records):
        r_id = str(rec.get("id") or rec.get("dialogue_id") or rec.get("conv_id") or f"r_{idx}")
        fields = extract_audit_fields(rec)
        meaningful_tokens = _tokenize(fields[FIELD_FULL_CONVERSATION])
        r_ngrams = extract_ngrams(meaningful_tokens, n=NGRAM_SIZE)
        right_extracted.append((r_id, fields, r_ngrams))

        for ng in r_ngrams:
            if ng not in inverted_index:
                inverted_index[ng] = []
            inverted_index[ng].append(idx)

    # 2. Iterate through left_records and query inverted index
    for l_idx, l_rec in enumerate(left_records):
        l_id = str(l_rec.get("id") or l_rec.get("dialogue_id") or l_rec.get("conv_id") or f"l_{l_idx}")
        l_fields = extract_audit_fields(l_rec)
        l_meaningful_tokens = _tokenize(l_fields[FIELD_FULL_CONVERSATION])
        l_ngrams = extract_ngrams(l_meaningful_tokens, n=NGRAM_SIZE)

        if not l_ngrams:
            continue

        # Inverted index lookup: find right candidate indices sharing >= 1 n-gram
        candidate_indices: Set[int] = set()
        for ng in l_ngrams:
            if ng in inverted_index:
                candidate_indices.update(inverted_index[ng])

        for r_idx in candidate_indices:
            r_id, r_fields, r_ngrams = right_extracted[r_idx]

            # Enforce within-split duplicate prevention
            if is_within_split:
                if l_id >= r_id:
                    continue

            # Shared n-grams on meaningful conversation
            shared_meaningful = l_ngrams & r_ngrams
            if not shared_meaningful:
                continue

            shared_count = len(shared_meaningful)
            union_count = len(l_ngrams | r_ngrams)
            meaningful_jaccard = (shared_count / union_count) if union_count > 0 else 0.0
            containment_left = shared_count / len(l_ngrams) if l_ngrams else 0.0
            containment_right = shared_count / len(r_ngrams) if r_ngrams else 0.0

            # Raw conversation overlap (including system and schemas)
            raw_overlap = compute_ngram_overlap(
                l_fields[FIELD_RAW_RECORD],
                r_fields[FIELD_RAW_RECORD],
                n=NGRAM_SIZE,
            )

            # Field-level meaningful overlaps
            field_overlaps: Dict[str, Dict[str, Any]] = {}
            max_field_jaccard = 0.0
            field_name_max = FIELD_FULL_CONVERSATION

            for fld in (
                FIELD_USER_INTENT,
                FIELD_ASSISTANT_ANSWER,
                FIELD_TOOL_TARGET,
                FIELD_TOOL_OBSERVATION,
            ):
                fo = compute_ngram_overlap(l_fields[fld], r_fields[fld], n=NGRAM_SIZE)
                field_overlaps[fld] = fo
                if fo["jaccard"] > max_field_jaccard:
                    max_field_jaccard = fo["jaccard"]

            # Check if threshold is met
            if meaningful_jaccard >= threshold or max_field_jaccard >= threshold:
                matched_spans = _find_matching_spans(l_fields[FIELD_FULL_CONVERSATION], shared_meaningful)
                primary_field = (
                    FIELD_FULL_CONVERSATION
                    if meaningful_jaccard >= threshold
                    else field_name_max
                )

                matches.append(
                    AuditMatch(
                        left_id=l_id,
                        right_id=r_id,
                        left_split=left_split,
                        right_split=right_split,
                        match_tier="tier3_ngram",
                        field_matched=primary_field,
                        details={
                            "meaningful_jaccard": meaningful_jaccard,
                            "raw_jaccard": raw_overlap["jaccard"],
                            "containment_left_in_right": containment_left,
                            "containment_right_in_left": containment_right,
                            "shared_ngrams_count": shared_count,
                            "field_overlaps": field_overlaps,
                            "matched_spans": matched_spans,
                        },
                    )
                )

    return matches
