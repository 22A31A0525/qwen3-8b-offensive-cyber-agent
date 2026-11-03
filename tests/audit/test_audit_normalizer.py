"""Unit tests for src.audit.normalizer."""
import json
import unicodedata
import pytest
from src.audit.normalizer import (
    normalize_text,
    canonical_record_hash,
    extract_audit_fields,
)
from src.audit.config import (
    FIELD_USER_INTENT,
    FIELD_ASSISTANT_ANSWER,
    FIELD_TOOL_TARGET,
    FIELD_TOOL_OBSERVATION,
    FIELD_FULL_CONVERSATION,
    FIELD_RAW_RECORD,
)


def test_normalize_text_unicode_nfc():
    # Composed vs decomposed 'é'
    decomposed = "cafe\u0301"
    composed = "café"
    assert decomposed != composed
    assert normalize_text(decomposed) == normalize_text(composed)
    assert unicodedata.is_normalized("NFC", normalize_text(decomposed))


def test_normalize_text_whitespace_and_stripping():
    raw = "   \t\n  SELECT   * \n FROM users \t WHERE id=1;   \n\n "
    expected = "select * from users where id=1;"
    assert normalize_text(raw) == expected


def test_normalize_text_case_handling():
    sample = "Token: c2VjcmV0X3Rva2Vu_123; Flag: CTF{Exploit_Succeed}; Path: /etc/Shadow"
    # preserve_case = False (default)
    lowercased = normalize_text(sample, preserve_case=False)
    assert "token: c2vjcmv0x3rva2vu_123; flag: ctf{exploit_succeed}; path: /etc/shadow" == lowercased

    # preserve_case = True
    preserved = normalize_text(sample, preserve_case=True)
    assert "Token: c2VjcmV0X3Rva2Vu_123; Flag: CTF{Exploit_Succeed}; Path: /etc/Shadow" == preserved


def test_normalize_text_empty_and_none():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""
    assert normalize_text("   \n\t  ") == ""


def test_canonical_record_hash_determinism_and_volatile_stripping():
    rec1 = {
        "id": "rec_001",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"category": "test"},
        "_timestamp": "2026-10-31T12:00:00Z",
        "index": 42,
    }
    rec2 = {
        "metadata": {"category": "test"},
        "id": "rec_001",
        "messages": [{"role": "user", "content": "hello"}],
        "_timestamp": "2026-10-31T14:30:00Z",  # different timestamp
        "index": 999,  # different index
    }
    rec3 = {
        "id": "rec_001",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"category": "test"},
    }
    rec_different = {
        "id": "rec_002",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"category": "test"},
    }

    hash1 = canonical_record_hash(rec1)
    hash2 = canonical_record_hash(rec2)
    hash3 = canonical_record_hash(rec3)
    hash_diff = canonical_record_hash(rec_different)

    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 hex string
    assert hash1 == hash2 == hash3
    assert hash1 != hash_diff


def test_extract_audit_fields_user_intent():
    record = {
        "id": "test_01",
        "messages": [
            {"role": "system", "content": "You are a red team bot."},
            {"role": "user", "content": "First scan the network."},
            {"role": "assistant", "content": "Running nmap."},
            {"role": "user", "content": "Now exploit port 80."},
        ],
    }
    fields = extract_audit_fields(record)
    assert FIELD_USER_INTENT in fields
    assert "First scan the network." in fields[FIELD_USER_INTENT]
    assert "Now exploit port 80." in fields[FIELD_USER_INTENT]


def test_extract_audit_fields_assistant_answer_strips_tool_call_tags():
    record = {
        "id": "test_02",
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "I will check open ports.\n"
                    "<tool_call>{\"name\": \"nmap_scan\", \"arguments\": {\"target\": \"10.0.0.1\"}}</tool_call>\n"
                    "Let's see what is listening."
                ),
            }
        ],
    }
    fields = extract_audit_fields(record)
    answer = fields[FIELD_ASSISTANT_ANSWER]
    assert "<tool_call>" not in answer
    assert "</tool_call>" not in answer
    assert "nmap_scan" not in answer
    assert "I will check open ports." in answer
    assert "Let's see what is listening." in answer


def test_extract_audit_fields_tool_call_target_from_structured():
    record = {
        "id": "test_03",
        "messages": [
            {
                "role": "assistant",
                "content": "Writing payload to disk.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "file_write",
                            "arguments": json.dumps({"path": "/tmp/evil.sh", "content": "echo pwned"}),
                        },
                    }
                ],
            }
        ],
    }
    fields = extract_audit_fields(record)
    targets = fields[FIELD_TOOL_TARGET]
    assert "file_write" in targets
    assert "/tmp/evil.sh" in targets
    assert "echo pwned" in targets


def test_extract_audit_fields_tool_call_target_from_tags():
    record = {
        "id": "test_04",
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Triggering exploit:\n"
                    "<tool_call>{\"name\": \"sqli_test\", \"arguments\": {\"param\": \"' OR 1=1--\"}}</tool_call>"
                ),
            }
        ],
    }
    fields = extract_audit_fields(record)
    targets = fields[FIELD_TOOL_TARGET]
    assert "sqli_test" in targets
    assert "' OR 1=1--" in targets


def test_extract_audit_fields_tool_observation():
    record = {
        "id": "test_05",
        "messages": [
            {"role": "user", "content": "Run whoami"},
            {
                "role": "tool",
                "name": "exec_cmd",
                "content": "uid=0(root) gid=0(root) groups=0(root) FLAG{R00T_P0WN3D}",
            },
        ],
    }
    fields = extract_audit_fields(record)
    obs = fields[FIELD_TOOL_OBSERVATION]
    assert "FLAG{R00T_P0WN3D}" in obs
    assert "uid=0(root)" in obs


def test_extract_audit_fields_full_meaningful_conversation_excludes_boilerplate():
    record = {
        "id": "test_06",
        "tools": [{"type": "function", "function": {"name": "nmap", "description": "scanner"}}],
        "messages": [
            {"role": "system", "content": "SYSTEM PROMPT BOILERPLATE THAT MUST BE STRIPPED"},
            {"role": "user", "content": "Scan the server."},
            {
                "role": "assistant",
                "content": "I am scanning now.",
                "tool_calls": [
                    {"function": {"name": "nmap", "arguments": "{\"target\": \"192.168.1.1\"}"}}
                ],
            },
            {"role": "tool", "content": "Port 80 open"},
            {"role": "assistant", "content": "Port 80 is open and vulnerable."},
        ],
    }
    fields = extract_audit_fields(record)
    full = fields[FIELD_FULL_CONVERSATION]

    # Must contain meaningful turns
    assert "Scan the server." in full
    assert "I am scanning now." in full
    assert "Port 80 open" in full
    assert "Port 80 is open and vulnerable." in full

    # Must NOT contain system prompt or tool declarations
    assert "SYSTEM PROMPT BOILERPLATE THAT MUST BE STRIPPED" not in full
    assert "description" not in full


def test_extract_audit_fields_raw_full_record():
    record = {
        "id": "test_07",
        "messages": [{"role": "user", "content": "test"}],
        "metadata": {"val": 123},
    }
    fields = extract_audit_fields(record)
    raw = fields[FIELD_RAW_RECORD]
    parsed = json.loads(raw)
    assert parsed["id"] == "test_07"
    assert parsed["metadata"]["val"] == 123


def test_extract_audit_fields_edge_cases():
    # Empty record
    empty_fields = extract_audit_fields({})
    assert empty_fields[FIELD_USER_INTENT] == ""
    assert empty_fields[FIELD_ASSISTANT_ANSWER] == ""
    assert empty_fields[FIELD_TOOL_TARGET] == ""
    assert empty_fields[FIELD_TOOL_OBSERVATION] == ""
    assert empty_fields[FIELD_FULL_CONVERSATION] == ""
    assert isinstance(empty_fields[FIELD_RAW_RECORD], str)

    # None contents in messages
    record = {
        "id": "test_08",
        "messages": [
            {"role": "user", "content": None},
            {"role": "assistant", "content": None},
            {"role": "tool", "content": None},
        ],
    }
    fields = extract_audit_fields(record)
    assert fields[FIELD_USER_INTENT] == ""
    assert fields[FIELD_ASSISTANT_ANSWER] == ""
    assert fields[FIELD_TOOL_OBSERVATION] == ""
