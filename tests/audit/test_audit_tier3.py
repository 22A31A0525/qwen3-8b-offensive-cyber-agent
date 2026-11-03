"""Unit tests for src.audit.tier3_ngram."""
import pytest
from src.audit.config import (
    AuditMatch,
    FIELD_FULL_CONVERSATION,
    FIELD_ASSISTANT_ANSWER,
    FIELD_USER_INTENT,
    FIELD_TOOL_TARGET,
    FIELD_TOOL_OBSERVATION,
)
from src.audit.tier3_ngram import (
    extract_ngrams,
    compute_ngram_overlap,
    run_tier3_audit,
)


def test_extract_ngrams_empty_or_too_short():
    assert extract_ngrams([], n=13) == set()
    tokens = ["token"] * 5
    assert extract_ngrams(tokens, n=13) == set()


def test_extract_ngrams_exact_and_sliding():
    tokens = [f"word_{i}" for i in range(13)]
    ngrams = extract_ngrams(tokens, n=13)
    assert len(ngrams) == 1
    assert tuple(tokens) in ngrams

    tokens_15 = [f"word_{i}" for i in range(15)]
    ngrams_15 = extract_ngrams(tokens_15, n=13)
    assert len(ngrams_15) == 3


def test_compute_ngram_overlap_empty_and_identical():
    res_empty = compute_ngram_overlap("", "")
    assert res_empty["jaccard"] == 0.0
    assert res_empty["containment_left_in_right"] == 0.0
    assert res_empty["containment_right_in_left"] == 0.0
    assert res_empty["shared_ngrams_count"] == 0

    text_long = " ".join([f"token_{i}" for i in range(25)])
    res_ident = compute_ngram_overlap(text_long, text_long, n=13)
    assert res_ident["jaccard"] == 1.0
    assert res_ident["containment_left_in_right"] == 1.0
    assert res_ident["containment_right_in_left"] == 1.0
    assert res_ident["shared_ngrams_count"] == 25 - 13 + 1


def test_compute_ngram_overlap_partial_and_disjoint():
    words_a = [f"alpha_{i}" for i in range(20)]
    words_b = [f"beta_{i}" for i in range(20)]
    res_disjoint = compute_ngram_overlap(" ".join(words_a), " ".join(words_b), n=5)
    assert res_disjoint["jaccard"] == 0.0
    assert res_disjoint["shared_ngrams_count"] == 0

    # Overlapping text
    shared = [f"common_{i}" for i in range(10)]
    words_left = ["pre_left_1", "pre_left_2"] + shared + ["post_left_1"]
    words_right = ["pre_right_1", "pre_right_2"] + shared + ["post_right_1"]
    res_partial = compute_ngram_overlap(" ".join(words_left), " ".join(words_right), n=5)
    assert 0.0 < res_partial["jaccard"] < 1.0
    assert res_partial["shared_ngrams_count"] > 0
    assert 0.0 < res_partial["containment_left_in_right"] <= 1.0
    assert 0.0 < res_partial["containment_right_in_left"] <= 1.0


def test_run_tier3_audit_scaffolding_immunity():
    # Two records sharing identical system prompt boilerplate and schema,
    # but having completely disjoint user queries and assistant responses.
    boilerplate_system = "You are an autonomous penetration testing assistant equipped with bash and python tools."
    tool_schema = "<tools>def run_cmd(cmd: str): ...</tools>"

    rec_train = {
        "id": "train_001",
        "messages": [
            {"role": "system", "content": f"{boilerplate_system}\n{tool_schema}"},
            {"role": "user", "content": "How do I perform an ARP spoofing attack on local subnet 192.168.1.0/24?"},
            {"role": "assistant", "content": "Use arpspoof or ettercap with targeted IP forwarding enabled on interface eth0."},
        ]
    }
    rec_test = {
        "id": "test_001",
        "messages": [
            {"role": "system", "content": f"{boilerplate_system}\n{tool_schema}"},
            {"role": "user", "content": "Explain how buffer overflow attacks overwrite the instruction pointer register."},
            {"role": "assistant", "content": "In x86 architectures, exceeding stack buffer boundaries corrupts the saved EIP address."},
        ]
    }

    matches = run_tier3_audit([rec_train], [rec_test], "train", "test", threshold=0.30)
    # Since meaningful content excludes system and tool schemas, meaningful_jaccard is 0.0
    assert len(matches) == 0


def test_run_tier3_audit_meaningful_overlap_detection():
    # Long overlapping exploit text (> 13 tokens identical)
    exploit_prompt = (
        "analyze vulnerable heap chunk allocation in glibc malloc where chunk consolidation "
        "leads to arbitrary memory write via corrupted forward and backward pointers"
    )
    exploit_answer = (
        "to exploit unsafe unlink craft fake chunk with fd pointing to target address minus 12 "
        "and bk pointing to fake chunk address to bypass corruption verification checks"
    )

    rec_train = {
        "id": "train_exploit_01",
        "messages": [
            {"role": "user", "content": exploit_prompt},
            {"role": "assistant", "content": exploit_answer},
        ]
    }
    rec_test = {
        "id": "test_exploit_01",
        "messages": [
            {"role": "user", "content": exploit_prompt},
            {"role": "assistant", "content": exploit_answer},
        ]
    }

    matches = run_tier3_audit([rec_train], [rec_test], "train", "test", threshold=0.50)
    assert len(matches) == 1
    m = matches[0]
    assert isinstance(m, AuditMatch)
    assert m.left_id == "train_exploit_01"
    assert m.right_id == "test_exploit_01"
    assert m.match_tier == "tier3_ngram"
    assert m.left_split == "train"
    assert m.right_split == "test"
    assert m.details["meaningful_jaccard"] >= 0.50
    assert m.details["shared_ngrams_count"] > 0
    assert "matched_spans" in m.details
    assert len(m.details["matched_spans"]) > 0


def test_run_tier3_audit_within_split_deduplication():
    # Within split deduplication avoids self match and duplicate pair
    rec_a = {
        "id": "rec_001",
        "messages": [
            {"role": "user", "content": "identical long phrase for testing within split deduplication logic in tier 3 audit engine"},
            {"role": "assistant", "content": "confirming identical long response phrase to satisfy thirteen token sliding window boundary requirement"},
        ]
    }
    rec_b = {
        "id": "rec_002",
        "messages": [
            {"role": "user", "content": "identical long phrase for testing within split deduplication logic in tier 3 audit engine"},
            {"role": "assistant", "content": "confirming identical long response phrase to satisfy thirteen token sliding window boundary requirement"},
        ]
    }

    # Within split: train vs train
    matches = run_tier3_audit([rec_a, rec_b], [rec_a, rec_b], "train", "train", threshold=0.50)
    # Only rec_001 vs rec_002 should be returned, not rec_001 vs rec_001 or rec_002 vs rec_001
    assert len(matches) == 1
    assert matches[0].left_id == "rec_001"
    assert matches[0].right_id == "rec_002"
