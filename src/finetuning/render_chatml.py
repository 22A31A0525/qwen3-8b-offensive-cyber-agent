"""render_chatml.py
Canonical Qwen3 ChatML Dialogue and Tool-Schema Renderer with Assistant-Only Masking.

Ensures:
1. Guaranteed <tools> XML schema injection into the system prompt for 100% of tool-calling dialogues.
2. Complete assistant-only completion loss masking (labels = -100 on system prompts, tools schemas,
   user prompts, and tool return observations).
"""

import json
import re
from typing import Any, Dict, List, Optional, Union


SYSTEM_TOOLS_TEMPLATE = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
{tools_xml}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def format_tools_xml(tools: List[Dict[str, Any]]) -> str:
    """Formats JSON tool definitions into canonical Qwen3 <tools> XML tags."""
    tools_str = json.dumps(tools, indent=2, ensure_ascii=False)
    return f"<tools>\n{tools_str}\n</tools>"


def format_message_content(msg: Dict[str, Any]) -> str:
    """Extracts text content and canonical Qwen3 <tool_call> tags from a message."""
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")
    if tool_calls and isinstance(tool_calls, list):
        tc_blocks: List[str] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            fn_args = fn.get("arguments", "")
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except Exception:
                    pass
            args_str = json.dumps(fn_args, ensure_ascii=False) if isinstance(fn_args, dict) else str(fn_args)
            tc_blocks.append(f'<tool_call>\n{{"name": "{fn_name}", "arguments": {args_str}}}\n</tool_call>')
        tc_text = "\n".join(tc_blocks)
        content = f"{content}\n{tc_text}".strip() if content else tc_text
    return content


def render_dialogue_to_chatml(record: Dict[str, Any]) -> str:
    """Renders a raw universal dialogue record into canonical Qwen3 ChatML text."""
    tools = record.get("tools")
    messages = list(record.get("messages", []))

    if not messages:
        return ""

    # Check if tools are present
    has_tools = bool(tools and isinstance(tools, list) and len(tools) > 0)
    tools_block = ""
    if has_tools:
        tools_xml = format_tools_xml(tools)
        tools_block = SYSTEM_TOOLS_TEMPLATE.format(tools_xml=tools_xml)

    # Copy messages to avoid mutating original
    rendered_messages: List[Dict[str, str]] = []
    
    first_role = messages[0].get("role", "")
    if has_tools:
        if first_role == "system":
            # Append tools block to existing system prompt
            orig_content = format_message_content(messages[0]).strip()
            combined_sys = f"{orig_content}\n\n{tools_block}" if orig_content else tools_block
            rendered_messages.append({"role": "system", "content": combined_sys})
            rendered_messages.extend(messages[1:])
        else:
            # Inject new system prompt turn
            rendered_messages.append({"role": "system", "content": tools_block})
            rendered_messages.extend(messages)
    else:
        rendered_messages.extend(messages)

    # Format into ChatML string
    chunks: List[str] = []
    for msg in rendered_messages:
        role = msg.get("role", "user")
        content = format_message_content(msg)
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    return "".join(chunks)


def mask_labels_for_assistant_only(tokens: List[str], full_text: str) -> List[int]:
    """Applies assistant-only loss masking over token strings.
    
    Tokens belonging to system, user, or tool turns receive label -100.
    Only tokens emitted by the assistant receive active training loss.
    """
    labels: List[int] = [-100] * len(tokens)
    
    # Track whether currently inside an assistant turn
    in_assistant = False
    
    for idx, tok in enumerate(tokens):
        if "<|im_start|>assistant" in tok:
            in_assistant = True
            # Mask the header itself
            labels[idx] = -100
            continue
        elif in_assistant:
            if "<|im_end|>" in tok or "<|im_start|>" in tok:
                in_assistant = False
                labels[idx] = -100
            else:
                # Active assistant token
                labels[idx] = 1  # Active gradient marker
        else:
            labels[idx] = -100

    return labels


def mask_tensor_labels_for_assistant(
    input_ids: List[int],
    tokenizer: Any,
    response_template: str = "<|im_start|>assistant\n",
    instruction_template: str = "<|im_end|>\n",
) -> List[int]:
    """Tensor/integer token ID masking for PyTorch DataCollator.
    
    Labels equal input_ids strictly inside assistant turns; -100 elsewhere.
    """
    labels = [-100] * len(input_ids)
    
    resp_ids = tokenizer.encode(response_template, add_special_tokens=False)
    inst_ids = tokenizer.encode(instruction_template, add_special_tokens=False)
    
    n_resp = len(resp_ids)
    n_inst = len(inst_ids)
    n_tokens = len(input_ids)
    
    i = 0
    while i < n_tokens:
        # Check if response_template starts at i
        if input_ids[i : i + n_resp] == resp_ids:
            start_idx = i + n_resp
            # Find the next instruction_template
            end_idx = n_tokens
            for j in range(start_idx, n_tokens):
                if input_ids[j : j + n_inst] == inst_ids:
                    end_idx = j
                    break
            # Assign active labels for assistant tokens
            for k in range(start_idx, min(end_idx + 1, n_tokens)):
                labels[k] = input_ids[k]
            i = end_idx + n_inst
        else:
            i += 1
            
    return labels

