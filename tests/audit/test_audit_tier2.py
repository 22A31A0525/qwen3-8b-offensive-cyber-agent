"""Unit tests for Tier 2 Normalized Duplicate Audit (src.audit.tier2_normalized)."""
import pytest
from src.audit.tier1_exact import AuditMatch
from src.audit.tier2_normalized import run_tier2_audit
from src.audit.config import (
    FIELD_USER_INTENT,
    FIELD_ASSISTANT_ANSWER,
    FIELD_TOOL_TARGET,
    FIELD_TOOL_OBSERVATION,
    FIELD_FULL_CONVERSATION,
)


def test_tier2_user_intent_exact_match_normalized():
    # Differences in casing, trailing/leading whitespace, and internal spacing
    left_records = [
        {
            "id": "train_norm_01",
            "messages": [
                {
                    "role": "user",
                    "content": "  How to configure   ProxyChains with TOR for pivoting? \n\t ",
                },
                {"role": "assistant", "content": "Distinct response alpha."},
            ],
        }
    ]
    right_records = [
        {
            "id": "test_norm_01",
            "messages": [
                {
                    "role": "user",
                    "content": "how to configure proxychains with tor for pivoting?",
                },
                {"role": "assistant", "content": "Distinct response beta."},
            ],
        }
    ]
    matches = run_tier2_audit(left_records, right_records, "train", "test")
    user_intent_matches = [m for m in matches if m.field_matched == FIELD_USER_INTENT]
    assert len(user_intent_matches) == 1
    assert user_intent_matches[0].left_id == "train_norm_01"
    assert user_intent_matches[0].right_id == "test_norm_01"
    assert user_intent_matches[0].match_tier == "tier2_normalized"
    assert "preview" in user_intent_matches[0].details
    assert "hash" in user_intent_matches[0].details


def test_tier2_assistant_answer_exact_match_normalized():
    left_records = [
        {
            "id": "train_ans_01",
            "messages": [
                {"role": "user", "content": "Different user question 1."},
                {
                    "role": "assistant",
                    "content": "Use `mimikatz.exe \"privilege::debug\" \"sekurlsa::logonpasswords\"` to dump LSASS.",
                },
            ],
        }
    ]
    right_records = [
        {
            "id": "test_ans_01",
            "messages": [
                {"role": "user", "content": "Different user question 2."},
                {
                    "role": "assistant",
                    "content": "USE `MIMIKATZ.EXE \"PRIVILEGE::DEBUG\" \"SEKURLSA::LOGONPASSWORDS\"` TO DUMP LSASS.",
                },
            ],
        }
    ]
    matches = run_tier2_audit(left_records, right_records, "train", "test")
    ans_matches = [m for m in matches if m.field_matched == FIELD_ASSISTANT_ANSWER]
    assert len(ans_matches) == 1
    assert ans_matches[0].left_id == "train_ans_01"
    assert ans_matches[0].right_id == "test_ans_01"


def test_tier2_tool_call_target_exact_match_normalized():
    left_records = [
        {
            "id": "train_tc_01",
            "messages": [
                {"role": "user", "content": "Scan network."},
                {
                    "role": "assistant",
                    "content": "Scanning...",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "execute_command",
                                "arguments": '{"command": "nmap -sV -sC 192.168.1.100"}',
                            },
                        }
                    ],
                },
            ],
        }
    ]
    right_records = [
        {
            "id": "test_tc_01",
            "messages": [
                {"role": "user", "content": "Perform network discovery."},
                {
                    "role": "assistant",
                    "content": "Starting scan...",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "execute_command",
                                "arguments": '{"command": "nmap -sV -sC 192.168.1.100"}',
                            },
                        }
                    ],
                },
            ],
        }
    ]
    matches = run_tier2_audit(left_records, right_records, "train", "test")
    tc_matches = [m for m in matches if m.field_matched == FIELD_TOOL_TARGET]
    assert len(tc_matches) == 1
    assert tc_matches[0].left_id == "train_tc_01"
    assert tc_matches[0].right_id == "test_tc_01"


def test_tier2_tool_observation_exact_match_normalized():
    obs_text = "PORT 445/tcp open microsoft-ds Windows Server 2019 Standard 17763"
    left_records = [
        {
            "id": "train_obs_01",
            "messages": [
                {"role": "user", "content": "Check SMB port."},
                {"role": "tool", "content": obs_text},
            ],
        }
    ]
    right_records = [
        {
            "id": "test_obs_01",
            "messages": [
                {"role": "user", "content": "Analyze target ports."},
                {"role": "tool", "content": f"  {obs_text.lower()}  \n"},
            ],
        }
    ]
    matches = run_tier2_audit(left_records, right_records, "train", "test")
    obs_matches = [m for m in matches if m.field_matched == FIELD_TOOL_OBSERVATION]
    assert len(obs_matches) == 1
    assert obs_matches[0].left_id == "train_obs_01"
    assert obs_matches[0].right_id == "test_obs_01"


def test_tier2_full_conversation_exact_match():
    # Both dialogues have identical meaningful turns (user, assistant, tool, response)
    # but different system prompt scaffolding
    dialogue_messages = [
        {"role": "user", "content": "Enumerate web services."},
        {"role": "assistant", "content": "I will send a curl GET request."},
        {"role": "tool", "content": "HTTP/1.1 200 OK\nServer: Apache/2.4.41"},
        {"role": "assistant", "content": "Target is Apache 2.4.41."},
    ]
    left_records = [
        {
            "id": "train_full_01",
            "messages": [
                {"role": "system", "content": "System prompt variant Alpha"},
                *dialogue_messages,
            ],
        }
    ]
    right_records = [
        {
            "id": "test_full_01",
            "messages": [
                {"role": "system", "content": "System prompt variant Beta (different!)"},
                *dialogue_messages,
            ],
        }
    ]
    matches = run_tier2_audit(left_records, right_records, "train", "test")
    full_matches = [m for m in matches if m.field_matched == FIELD_FULL_CONVERSATION]
    assert len(full_matches) == 1
    assert full_matches[0].left_id == "train_full_01"
    assert full_matches[0].right_id == "test_full_01"


def test_tier2_empty_fields_do_not_collide():
    # Both records have NO tool calls and NO tool observations (e.g. pure reasoning track)
    left_records = [
        {
            "id": "train_reasoning_01",
            "messages": [
                {"role": "user", "content": "What is the principle of least privilege?"},
                {"role": "assistant", "content": "Users should have minimum necessary access."},
            ],
        }
    ]
    right_records = [
        {
            "id": "test_reasoning_01",
            "messages": [
                {"role": "user", "content": "Define defense in depth."},
                {"role": "assistant", "content": "Layered security defenses across multiple tiers."},
            ],
        }
    ]
    matches = run_tier2_audit(left_records, right_records, "train", "test")
    # Empty tool_call_target and tool_observation MUST NOT collide
    tc_matches = [m for m in matches if m.field_matched == FIELD_TOOL_TARGET]
    obs_matches = [m for m in matches if m.field_matched == FIELD_TOOL_OBSERVATION]
    assert len(tc_matches) == 0
    assert len(obs_matches) == 0
    assert len(matches) == 0


def test_tier2_within_split_deduplication():
    # Inside train split: rec_1, rec_2 share identical user intent
    rec_1 = {
        "id": "rec_001",
        "messages": [
            {"role": "user", "content": "Identical question across two train records."},
            {"role": "assistant", "content": "Unique answer 1."},
        ],
    }
    rec_2 = {
        "id": "rec_002",
        "messages": [
            {"role": "user", "content": "identical question across two train records."},
            {"role": "assistant", "content": "Unique answer 2."},
        ],
    }
    records = [rec_1, rec_2]
    matches = run_tier2_audit(records, records, "train", "train")

    user_matches = [m for m in matches if m.field_matched == FIELD_USER_INTENT]
    assert len(user_matches) == 1
    assert user_matches[0].left_id == "rec_001"
    assert user_matches[0].right_id == "rec_002"

    # Enforce strictly left_id < right_id
    for m in matches:
        assert m.left_id < m.right_id


def test_tier2_preview_and_redaction():
    long_intent = "A" * 300
    left_records = [
        {
            "id": "train_long",
            "messages": [{"role": "user", "content": long_intent}],
        }
    ]
    right_records = [
        {
            "id": "test_long",
            "messages": [{"role": "user", "content": long_intent}],
        }
    ]
    matches = run_tier2_audit(left_records, right_records, "train", "test")
    assert len(matches) >= 1
    user_match = [m for m in matches if m.field_matched == FIELD_USER_INTENT][0]
    preview = user_match.details["preview"]
    assert len(preview) <= 200
    assert preview.endswith("...")
    assert user_match.details["matched_length"] == 300


def test_tier2_empty_records():
    matches = run_tier2_audit([], [], "train", "test")
    assert matches == []
