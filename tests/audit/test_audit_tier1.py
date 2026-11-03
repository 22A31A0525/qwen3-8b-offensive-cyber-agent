"""Unit tests for Tier 1 Exact Identity & Metadata Audit (src.audit.tier1_exact)."""
import pytest
from src.audit.tier1_exact import AuditMatch, run_tier1_audit
from src.audit.config import (
    FIELD_USER_INTENT,
    FIELD_ASSISTANT_ANSWER,
    FIELD_TOOL_TARGET,
    FIELD_TOOL_OBSERVATION,
    FIELD_FULL_CONVERSATION,
)


def test_audit_match_dataclass():
    match = AuditMatch(
        left_id="art_001",
        right_id="art_002",
        left_split="train",
        right_split="test",
        match_tier="tier1_exact",
        field_matched="id",
        details={"matched_id": "art_001"},
    )
    assert match.left_id == "art_001"
    assert match.right_id == "art_002"
    assert match.left_split == "train"
    assert match.right_split == "test"
    assert match.match_tier == "tier1_exact"
    assert match.field_matched == "id"
    assert match.details == {"matched_id": "art_001"}

    d = match.to_dict()
    assert isinstance(d, dict)
    assert d["left_id"] == "art_001"
    assert d["field_matched"] == "id"


def test_tier1_exact_record_id_collision_cross_split():
    left_records = [
        {
            "id": "collision_001",
            "messages": [{"role": "user", "content": "Query A"}],
        }
    ]
    right_records = [
        {
            "id": "collision_001",
            "messages": [{"role": "user", "content": "Query B with distinct content"}],
        }
    ]
    matches = run_tier1_audit(left_records, right_records, "train", "test")
    id_matches = [m for m in matches if m.field_matched == "id"]
    assert len(id_matches) == 1
    assert id_matches[0].left_id == "collision_001"
    assert id_matches[0].right_id == "collision_001"
    assert id_matches[0].match_tier == "tier1_exact"
    assert id_matches[0].left_split == "train"
    assert id_matches[0].right_split == "test"


def test_tier1_no_collision_when_ids_and_data_differ():
    left_records = [
        {
            "id": "rec_001",
            "messages": [{"role": "user", "content": "How to scan ports?"}],
        }
    ]
    right_records = [
        {
            "id": "rec_002",
            "messages": [{"role": "user", "content": "How to configure firewall?"}],
        }
    ]
    matches = run_tier1_audit(left_records, right_records, "train", "test")
    assert len(matches) == 0


def test_tier1_scenario_id_collision():
    left_records = [
        {
            "id": "train_scen_1",
            "metadata": {"scenario_id": "sqli_blind_scenario_01"},
            "messages": [{"role": "user", "content": "Run sqlmap"}],
        }
    ]
    right_records = [
        {
            "id": "test_scen_1",
            "scenario_id": "sqli_blind_scenario_01",
            "messages": [{"role": "user", "content": "Exploit blind SQL injection"}],
        }
    ]
    matches = run_tier1_audit(left_records, right_records, "train", "test")
    scen_matches = [m for m in matches if m.field_matched == "scenario_id"]
    assert len(scen_matches) == 1
    assert scen_matches[0].left_id == "train_scen_1"
    assert scen_matches[0].right_id == "test_scen_1"
    assert scen_matches[0].details["scenario_id"] == "sqli_blind_scenario_01"


def test_tier1_group_id_collision():
    left_records = [
        {
            "id": "train_grp_1",
            "metadata": {"group_id": "group_buffer_overflow_22"},
            "messages": [{"role": "user", "content": "Generate exploit payload"}],
        }
    ]
    right_records = [
        {
            "id": "test_grp_1",
            "metadata": {"group_id": "group_buffer_overflow_22"},
            "messages": [{"role": "user", "content": "Buffer overflow testing"}],
        }
    ]
    matches = run_tier1_audit(left_records, right_records, "train", "test")
    grp_matches = [m for m in matches if m.field_matched == "group_id"]
    assert len(grp_matches) == 1
    assert grp_matches[0].left_id == "train_grp_1"
    assert grp_matches[0].right_id == "test_grp_1"
    assert grp_matches[0].details["group_id"] == "group_buffer_overflow_22"


def test_tier1_canonical_hash_collision():
    record_content = {
        "id": "cross_split_dup_01",
        "messages": [
            {"role": "user", "content": "Explain AS-REP roasting."},
            {"role": "assistant", "content": "AS-REP roasting targets accounts without preauth."},
        ]
    }
    left_records = [{**record_content, "_timestamp": 12345}]
    right_records = [{**record_content, "index": 99}]

    matches = run_tier1_audit(left_records, right_records, "train", "test")
    hash_matches = [m for m in matches if m.field_matched == "canonical_hash"]
    assert len(hash_matches) == 1
    assert hash_matches[0].left_id == "cross_split_dup_01"
    assert hash_matches[0].right_id == "cross_split_dup_01"
    assert "canonical_hash" in hash_matches[0].details or "hash" in hash_matches[0].details


def test_tier1_eval_flag_in_tool_observation():
    flag_val = "FLAG{sql_bypass_super_secret_token_12345}"
    left_records = [
        {
            "id": "train_flag_leak",
            "messages": [
                {"role": "user", "content": "Test tool response"},
                {"role": "assistant", "content": "Running tool..."},
                {
                    "role": "tool",
                    "content": f"Database query completed. Value: {flag_val}",
                },
            ],
        }
    ]
    right_records = [
        {
            "id": "test_eval_target",
            "messages": [
                {"role": "user", "content": "Extract the secret flag."},
                {"role": "tool", "content": f"Output: {flag_val}"},
            ],
        }
    ]
    matches = run_tier1_audit(
        left_records,
        right_records,
        "train",
        "test",
        known_eval_flags={flag_val},
    )
    flag_matches = [m for m in matches if m.field_matched == "eval_flag"]
    assert len(flag_matches) >= 1
    assert flag_matches[0].left_id == "train_flag_leak"
    assert flag_matches[0].details["flag"] == flag_val
    assert flag_matches[0].details["matched_field"] == FIELD_TOOL_OBSERVATION


def test_tier1_eval_flag_in_assistant_answer():
    flag_val = "FLAG{kerberos_silver_ticket_privesc_888}"
    left_records = [
        {
            "id": "train_flag_prose",
            "messages": [
                {"role": "user", "content": "What is the key?"},
                {
                    "role": "assistant",
                    "content": f"The captured ticket flag is {flag_val}. We can forge tickets now.",
                },
            ],
        }
    ]
    right_records = []
    matches = run_tier1_audit(
        left_records,
        right_records,
        "train",
        "test",
        known_eval_flags={flag_val},
    )
    flag_matches = [m for m in matches if m.field_matched == "eval_flag"]
    assert len(flag_matches) == 1
    assert flag_matches[0].left_id == "train_flag_prose"
    assert flag_matches[0].details["flag"] == flag_val
    assert flag_matches[0].details["matched_field"] == FIELD_ASSISTANT_ANSWER


def test_tier1_eval_flag_no_leakage():
    left_records = [
        {
            "id": "train_clean_001",
            "messages": [
                {"role": "user", "content": "Normal security advice."},
                {"role": "assistant", "content": "Always enable MFA and patch systems."},
            ],
        }
    ]
    right_records = []
    matches = run_tier1_audit(
        left_records,
        right_records,
        "train",
        "test",
        known_eval_flags={"FLAG{completely_absent_eval_flag}"},
    )
    flag_matches = [m for m in matches if m.field_matched == "eval_flag"]
    assert len(flag_matches) == 0


def test_tier1_within_split_prevents_self_matches_and_duplicates():
    # Same split: train vs train
    rec_a = {
        "id": "rec_001",
        "metadata": {"scenario_id": "shared_scen_x"},
        "messages": [{"role": "user", "content": "Dialogue A"}],
    }
    rec_b = {
        "id": "rec_002",
        "metadata": {"scenario_id": "shared_scen_x"},
        "messages": [{"role": "user", "content": "Dialogue B"}],
    }
    records = [rec_a, rec_b]

    matches = run_tier1_audit(records, records, "train", "train")
    scen_matches = [m for m in matches if m.field_matched == "scenario_id"]

    # Only rec_001 < rec_002 should be reported
    assert len(scen_matches) == 1
    assert scen_matches[0].left_id == "rec_001"
    assert scen_matches[0].right_id == "rec_002"

    # Verify no self-match (rec_001, rec_001) or reverse pair (rec_002, rec_001)
    for m in matches:
        assert m.left_id < m.right_id


def test_tier1_empty_records():
    matches = run_tier1_audit([], [], "train", "test")
    assert matches == []
