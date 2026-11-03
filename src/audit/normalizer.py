"""Field Normalizer and Boilerplate Stripper Module.

Extracts meaningful fields (user intent, assistant answer, tool target,
tool observation, full conversation) from dataset records, strips boilerplate
and system schemas, and performs deterministic Unicode and case normalization.
"""
from typing import Any, Dict, List, Optional
import hashlib
import json
import re
import unicodedata

from src.audit.config import (
    FIELD_USER_INTENT,
    FIELD_ASSISTANT_ANSWER,
    FIELD_TOOL_TARGET,
    FIELD_TOOL_OBSERVATION,
    FIELD_FULL_CONVERSATION,
    FIELD_RAW_RECORD,
)


def normalize_text(text: Optional[str], preserve_case: bool = False) -> str:
    """Applies Unicode NFC normalization, collapses contiguous whitespace,

    strips leading/trailing whitespace, and optionally converts to lowercase.

    Args:
        text: Input string to normalize. If None or non-string, handled gracefully.
        preserve_case: If True, preserves original case (crucial for tokens,
            hashes, Base64 strings, flags). If False, converts to lowercase.

    Returns:
        Clean normalized string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""

    # 1. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2. Case handling
    if not preserve_case:
        normalized = normalized.lower()

    # 3. Contiguous whitespace collapsing & trimming
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def canonical_record_hash(record: Dict[str, Any]) -> str:
    """Serializes record to JSON with sort_keys=True, removing volatile fields

    (_timestamp, index) if present, and returns the SHA-256 hex digest.

    Args:
        record: Dataset record dictionary.

    Returns:
        Hex-encoded SHA-256 string (64 characters).
    """
    if not isinstance(record, dict):
        record = {}
    volatile_fields = {"_timestamp", "index"}
    cleaned = {k: v for k, v in record.items() if k not in volatile_fields}
    serialized = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _format_tool_args(args: Any) -> str:
    """Helper to convert tool call arguments into a deterministic string representation."""
    if isinstance(args, str):
        args_str = args.strip()
        try:
            parsed = json.loads(args_str)
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
            return args_str
        except Exception:
            return args_str
    elif isinstance(args, (dict, list)):
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    elif args is not None:
        return str(args).strip()
    return ""


def _extract_single_tool_call(tc: Dict[str, Any]) -> str:
    """Helper to extract function name and argument keys/values from a tool call dictionary."""
    if not isinstance(tc, dict):
        return str(tc).strip()

    name = ""
    args: Any = ""
    if "function" in tc and isinstance(tc["function"], dict):
        fn_dict = tc["function"]
        name = fn_dict.get("name", "") or ""
        args = fn_dict.get("arguments", "") or ""
    else:
        name = tc.get("name", "") or ""
        args = tc.get("arguments", tc.get("parameters", "")) or ""

    formatted_args = _format_tool_args(args)
    if name and formatted_args:
        return f"{name}: {formatted_args}"
    elif name:
        return name
    elif formatted_args:
        return formatted_args
    return ""


def extract_audit_fields(record: Dict[str, Any]) -> Dict[str, str]:
    """Extracts targeted audit fields from a dataset record.

    Extracts:
    - user_intent: Combines text from all user turns.
    - assistant_answer: Combines text from assistant turns with <tool_call> tags stripped.
    - tool_call_target: Combines tool names and arguments from structured calls or tags.
    - tool_observation: Combines tool output texts (terminal output, flags, responses).
    - full_meaningful_conversation: Combines all above fields (excluding system boilerplate
      and tool declarations).
    - raw_full_record: JSON serialization of the full raw record.

    Args:
        record: Dataset record dictionary.

    Returns:
        Dict mapping field names to extracted content strings.
    """
    if not isinstance(record, dict):
        record = {}

    messages: List[Dict[str, Any]] = record.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    user_intents: List[str] = []
    assistant_answers: List[str] = []
    tool_targets: List[str] = []
    tool_observations: List[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content")
        raw_text = str(content) if content is not None else ""

        if role == "user":
            trimmed = raw_text.strip()
            if trimmed:
                user_intents.append(trimmed)

        elif role == "assistant":
            # 1. Extract tool calls from structured tool_calls list if present
            structured_tool_calls = msg.get("tool_calls", [])
            if isinstance(structured_tool_calls, list):
                for tc in structured_tool_calls:
                    target = _extract_single_tool_call(tc)
                    if target:
                        tool_targets.append(target)

            # 2. Extract tool calls embedded in <tool_call> tags in content
            if raw_text and "<tool_call>" in raw_text:
                tag_matches = re.findall(r"<tool_call>([\s\S]*?)</tool_call>", raw_text)
                for body in tag_matches:
                    body_clean = body.strip()
                    if not body_clean:
                        continue
                    try:
                        parsed = json.loads(body_clean)
                        if isinstance(parsed, dict):
                            target = _extract_single_tool_call(parsed)
                            if target:
                                tool_targets.append(target)
                                continue
                    except Exception:
                        pass
                    tool_targets.append(body_clean)

            # 3. Strip <tool_call> tags and content from assistant answer
            cleaned_answer = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", raw_text)
            cleaned_answer = re.sub(r"</?tool_call>", "", cleaned_answer)
            cleaned_answer = cleaned_answer.strip()
            if cleaned_answer:
                assistant_answers.append(cleaned_answer)

        elif role in ("tool", "function"):
            trimmed = raw_text.strip()
            if trimmed:
                tool_observations.append(trimmed)

        # Check for embedded tool output blocks in any message
        if raw_text and ("<tool_output>" in raw_text or "<tool_response>" in raw_text):
            out_matches = re.findall(
                r"<(?:tool_output|tool_response)>([\s\S]*?)</(?:tool_output|tool_response)>",
                raw_text,
            )
            for out in out_matches:
                trimmed_out = out.strip()
                if trimmed_out:
                    tool_observations.append(trimmed_out)

    user_intent_str = "\n".join(user_intents)
    assistant_answer_str = "\n".join(assistant_answers)
    tool_target_str = "\n".join(tool_targets)
    tool_observation_str = "\n".join(tool_observations)

    meaningful_components = [
        user_intent_str,
        assistant_answer_str,
        tool_target_str,
        tool_observation_str,
    ]
    full_conversation_str = "\n".join(c for c in meaningful_components if c)

    raw_record_str = json.dumps(record, sort_keys=True, ensure_ascii=False)

    return {
        FIELD_USER_INTENT: user_intent_str,
        FIELD_ASSISTANT_ANSWER: assistant_answer_str,
        FIELD_TOOL_TARGET: tool_target_str,
        FIELD_TOOL_OBSERVATION: tool_observation_str,
        FIELD_FULL_CONVERSATION: full_conversation_str,
        FIELD_RAW_RECORD: raw_record_str,
    }
