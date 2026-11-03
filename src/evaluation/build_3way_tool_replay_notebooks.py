#!/usr/bin/env python3
"""
Build 3-Way Multi-Turn Tool Replay Evaluation Notebooks for Kaggle Dual-GPU Execution
=====================================================================================
Target: 130 Holdout Dialogues (479 assistant turns, 241 tool calls) from data/clean/test/tool_replay_20pct.jsonl
Divided into 3 shards across 3 auxiliary Kaggle accounts:
  - Shard 1 (44 dialogues, 148 turns)
  - Shard 2 (43 dialogues, 172 turns)
  - Shard 3 (43 dialogues, 159 turns)

Each shard directory contains:
  1. qwen3_8b_v3_tool_replay_eval_shard_{i}.ipynb (Self-contained notebook with embedded dataset payload)
  2. tool_replay_shard_{i}.jsonl (Raw split dataset file)
  3. kernel-metadata.json (Ready for Kaggle CLI push)
"""

import os
import sys
import json
import zlib
import base64
import math
import shutil
from typing import List, Dict, Any

_u1 = os.getenv("KAGGLE_USER_SHARD1", "shard1_user")
_k1 = os.getenv("KAGGLE_KEY_SHARD1", "")
_u2 = os.getenv("KAGGLE_USER_SHARD2", "shard2_user")
_k2 = os.getenv("KAGGLE_KEY_SHARD2", "")
_u3 = os.getenv("KAGGLE_USER_SHARD3", "shard3_user")
_k3 = os.getenv("KAGGLE_KEY_SHARD3", "")

ACCOUNTS = [
    {
        "shard_idx": 1,
        "username": _u1,
        "key": _k1,
        "dir": "v3_pipeline/kaggle_eval_multiturn/shard_1",
        "slug": f"{_u1}/qwen3-8b-v3-tool-replay-eval-shard-1",
        "title": "Qwen3 8B v3 Tool Replay Eval Shard 1"
    },
    {
        "shard_idx": 2,
        "username": _u2,
        "key": _k2,
        "dir": "v3_pipeline/kaggle_eval_multiturn/shard_2",
        "slug": f"{_u2}/qwen3-8b-v3-tool-replay-eval-shard-2",
        "title": "Qwen3 8B v3 Tool Replay Eval Shard 2"
    },
    {
        "shard_idx": 3,
        "username": _u3,
        "key": _k3,
        "dir": "v3_pipeline/kaggle_eval_multiturn/shard_3",
        "slug": f"{_u3}/qwen3-8b-v3-tool-replay-eval-shard-3",
        "title": "Qwen3 8B v3 Tool Replay Eval Shard 3"
    }
]

def create_cell(cell_type: str, source: str) -> Dict[str, Any]:
    lines = [s + "\n" for s in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines.pop()
    cell = {"cell_type": cell_type, "metadata": {}, "source": lines}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def discover_hf_token() -> str:
    t = os.environ.get("HF_TOKEN", "")
    if t:
        return t.strip()
    cache_token = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cache_token):
        try:
            with open(cache_token, "r", encoding="utf-8") as f:
                c = f.read().strip()
                if c:
                    return c
        except Exception:
            pass
    return ""

def generate_shard_notebook(
    shard_idx: int,
    total_shards: int,
    records: List[Dict[str, Any]],
    account_username: str
) -> Dict[str, Any]:
    compressed = zlib.compress(json.dumps(records).encode("utf-8"), level=9)
    b64_payload = base64.b64encode(compressed).decode("ascii")
    injected_token = discover_hf_token()

    cells = []

    # Cell 0: Markdown Header
    cells.append(create_cell("markdown", f"""# Qwen3-8B v3 Multi-Turn Tool Call Replay Evaluation — Shard {shard_idx}/{total_shards}

> **Target**: Shard {shard_idx} of {total_shards} ({len(records)} Dialogues, Teacher-Forced Multi-Turn Replay)  
> **Base Model**: `Qwen/Qwen3-8B`  
> **Production Adapter**: `{os.getenv("HF_REPO_ID", "Qwen/Qwen3-8B-offensive-cyber-agent")}` (1.0 Full Epoch, Step 1262)  
> **Hardware**: Kaggle Dual Tesla T4 GPUs (~{math.ceil(len(records)/2)} dialogues per GPU)  
> **Thinking Mode**: End-to-End Deep Reasoning (`enable_thinking=True`, `<think>...</think>`)  
> **Decoding**: Temperature Sampling (`do_sample=True`, `temperature=0.6`, `top_p=0.95`, `repetition_penalty=1.05`, `max_new_tokens=1024`)  
> **Worker Account**: `{account_username}`  
"""))

    # Cell 1: Environment & Config
    cells.append(create_cell("code", f"""# ==============================================================================
# CELL 1: ENVIRONMENT & CONFIGURATION
# ==============================================================================
import os
import sys

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    HF_TOKEN = "{injected_token}"
    print("[AUTH] Loaded authorized pipeline Hugging Face token [OK].", flush=True)

HF_REPO_ID = os.getenv("HF_REPO_ID", "Qwen/Qwen3-8B-offensive-cyber-agent")
BASE_MODEL_REPO = "Qwen/Qwen3-8B"
SHARD_INDEX = {shard_idx}
PUSH_TO_HUB = True  # Explicitly enabled: upload evaluation results to Hugging Face Hub

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OUTPUT_DIR = f"/kaggle/working/replay_shard_{{SHARD_INDEX}}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[CONFIG] Shard Index:     {{SHARD_INDEX}}", flush=True)
print(f"[CONFIG] Base Model:      {{BASE_MODEL_REPO}}", flush=True)
print(f"[CONFIG] Adapter Repo:    {{HF_REPO_ID}}", flush=True)
print(f"[CONFIG] Push To Hub:     {{PUSH_TO_HUB}}", flush=True)
print(f"[CONFIG] Output Dir:      {{OUTPUT_DIR}}", flush=True)
"""))

    # Cell 2: Dependencies & GPU Setup
    cells.append(create_cell("code", """# ==============================================================================
# CELL 2: GPU ENVIRONMENT & DEPENDENCY SETUP
# ==============================================================================
!nvidia-smi
!pip install -q -U "bitsandbytes>=0.46.1" peft accelerate huggingface_hub

import torch
device_count = torch.cuda.device_count()
print(f"[INIT] PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | GPU Count: {device_count}")
for i in range(device_count):
    p = torch.cuda.get_device_properties(i)
    print(f"[INIT] GPU {i}: {p.name} | Total VRAM: {p.total_memory / (1024**3):.2f} GB")
"""))

    # Cell 3: Unpack Embedded Records
    cells.append(create_cell("code", f"""# ==============================================================================
# CELL 3: EXTRACT EMBEDDED DIALOGUES FOR SHARD {shard_idx}
# ==============================================================================
import json
import zlib
import base64

B64_PAYLOAD = "{b64_payload}"
decompressed = zlib.decompress(base64.b64decode(B64_PAYLOAD.encode("ascii"))).decode("utf-8")
shard_records = json.loads(decompressed)

print(f"[DATA] Successfully unpacked exactly {{len(shard_records)}} dialogues for Shard {shard_idx}!", flush=True)
total_asst_turns = sum(sum(1 for m in r.get("messages", []) if m.get("role") == "assistant") for r in shard_records)
total_tool_calls = sum(sum(len(m.get("tool_calls", [])) for m in r.get("messages", []) if m.get("role") == "assistant") for r in shard_records)
print(f"[DATA] Total Assistant Turns: {{total_asst_turns}} | Total Tool Calls: {{total_tool_calls}}", flush=True)
for i, r in enumerate(shard_records[:3]):
    asst_turns = sum(1 for m in r.get("messages", []) if m.get("role") == "assistant")
    print(f"  [{{i+1:2d}}] ID: {{r['id']}} | Turns: {{asst_turns}}", flush=True)
"""))

    # Cell 4: Tokenizer Verification
    cells.append(create_cell("code", """# ==============================================================================
# CELL 4: TOKENIZER VERIFICATION FROM BASE MODEL
# ==============================================================================
from transformers import AutoTokenizer

print(f"[INIT] Loading tokenizer from base model '{BASE_MODEL_REPO}'...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_REPO, token=HF_TOKEN, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[INIT] Tokenizer vocab size: {len(tokenizer)}", flush=True)
print(f"[INIT] EOS Token: {tokenizer.eos_token} ({tokenizer.eos_token_id})", flush=True)
"""))

    # Cell 5: Worker Script Creation
    worker_script = chr(10).join([
        "import os",
        "import sys",
        "import json",
        "import re",
        "import time",
        "import argparse",
        "import datetime",
        "import torch",
        "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig",
        "from peft import PeftModel",
        "",
        "MAX_CTX = 10240",
        "MIN_NEW_TOKENS = 64",
        "",
        "def iter_teacher_forced_turns(messages):",
        "    context = []",
        "    turn_idx = 0",
        "    for msg in messages:",
        "        if msg.get('role') == 'assistant':",
        "            yield list(context), dict(msg), turn_idx",
        "            turn_idx += 1",
        "        context.append(msg)",
        "",
        "def extract_replay_tool_calls(text):",
        "    tool_calls = []",
        "    pattern = re.compile(r'<tool_call>\\s*(.*?)\\s*</tool_call>', re.DOTALL)",
        "    for m in pattern.findall(text):",
        "        m_clean = m.strip()",
        "        if m_clean.startswith('```json'): m_clean = m_clean[7:]",
        "        if m_clean.startswith('```'): m_clean = m_clean[3:]",
        "        if m_clean.endswith('```'): m_clean = m_clean[:-3]",
        "        m_clean = m_clean.strip()",
        "        try:",
        "            parsed = json.loads(m_clean)",
        "            if isinstance(parsed, dict) and 'name' in parsed:",
        "                args = parsed.get('arguments', {})",
        "                if isinstance(args, str):",
        "                    try: args = json.loads(args)",
        "                    except Exception: pass",
        "                parsed['arguments'] = args",
        "                tool_calls.append(parsed)",
        "        except Exception:",
        "            continue",
        "    return tool_calls",
        "",
        "def compute_token_f1(prediction, reference):",
        "    p_toks = re.findall(r'\\w+|[^\\w\\s]', prediction.lower())",
        "    r_toks = re.findall(r'\\w+|[^\\w\\s]', reference.lower())",
        "    if not p_toks or not r_toks:",
        "        score = 1.0 if p_toks == r_toks else 0.0",
        "        return {'precision': score, 'recall': score, 'f1': score}",
        "    p_cnt = {}",
        "    for t in p_toks: p_cnt[t] = p_cnt.get(t, 0) + 1",
        "    r_cnt = {}",
        "    for t in r_toks: r_cnt[t] = r_cnt.get(t, 0) + 1",
        "    comm = sum(min(c, r_cnt.get(t, 0)) for t, c in p_cnt.items())",
        "    prec = comm / max(1, len(p_toks))",
        "    rec = comm / max(1, len(r_toks))",
        "    f1 = (2 * prec * rec) / max(1e-8, prec + rec)",
        "    return {'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4)}",
        "",
        "def evaluate_replay_turn(pred_text, target_message):",
        "    target_tcs = target_message.get('tool_calls', [])",
        "    target_has_tool = len(target_tcs) > 0",
        "    is_no_tool_turn = not target_has_tool",
        "",
        "    if pred_text is None:",
        "        return {",
        "            'confusion': 'ERROR',",
        "            'is_no_tool_turn': is_no_tool_turn,",
        "            'tool_action_match': False,",
        "            'tool_count_target': len(target_tcs),",
        "            'tool_count_pred': 0,",
        "            'tool_name_match': False if target_has_tool else None,",
        "            'tool_call_precision': 0.0 if target_has_tool else 1.0,",
        "            'tool_call_recall': 0.0 if target_has_tool else 1.0,",
        "            'tool_call_f1': 0.0 if target_has_tool else 1.0,",
        "            'args_valid_json': None,",
        "            'arg_key_precision': None,",
        "            'arg_key_recall': None,",
        "            'arg_key_f1': None,",
        "            'arg_value_accuracy': None,",
        "            'full_tool_call_accuracy': 0.0 if target_has_tool else 1.0,",
        "            'content_f1': 0.0,",
        "            'content_precision': 0.0,",
        "            'content_recall': 0.0",
        "        }",
        "",
        "    pred_tcs = extract_replay_tool_calls(pred_text)",
        "    pred_has_tool = len(pred_tcs) > 0",
        "",
        "    tool_action_match = (target_has_tool == pred_has_tool)",
        "",
        "    if target_has_tool and pred_has_tool:",
        "        confusion = 'TP'",
        "    elif not target_has_tool and pred_has_tool:",
        "        confusion = 'FP'",
        "    elif target_has_tool and not pred_has_tool:",
        "        confusion = 'FN'",
        "    else:",
        "        confusion = 'TN'",
        "",
        "    # Normalize targets",
        "    norm_targets = []",
        "    for tc in target_tcs:",
        "        fn = tc.get('function', {}) if 'function' in tc else tc",
        "        name = fn.get('name', '').strip().lower()",
        "        args = fn.get('arguments', {})",
        "        if isinstance(args, str):",
        "            try: args = json.loads(args)",
        "            except Exception: args = {}",
        "        norm_targets.append((name, args if isinstance(args, dict) else {}))",
        "",
        "    # Normalize predictions",
        "    norm_preds = []",
        "    args_all_valid_json = True",
        "    for ptc in pred_tcs:",
        "        name = ptc.get('name', '').strip().lower()",
        "        args = ptc.get('arguments', {})",
        "        if isinstance(args, str):",
        "            try: args = json.loads(args)",
        "            except Exception:",
        "                args_all_valid_json = False",
        "                args = {}",
        "        elif not isinstance(args, dict):",
        "            args_all_valid_json = False",
        "            args = {}",
        "        norm_preds.append((name, args))",
        "",
        "    # Evaluate multiple tool calls via greedy bipartite matching",
        "    matched_targets = set()",
        "    call_matches = 0",
        "    exact_matches = 0",
        "    key_prec_list, key_rec_list, val_acc_list = [], [], []",
        "",
        "    for p_name, p_args in norm_preds:",
        "        best_t_idx = None",
        "        for t_idx, (t_name, t_args) in enumerate(norm_targets):",
        "            if t_idx in matched_targets: continue",
        "            if p_name == t_name:",
        "                best_t_idx = t_idx",
        "                break",
        "        if best_t_idx is not None:",
        "            matched_targets.add(best_t_idx)",
        "            call_matches += 1",
        "            t_name, t_args = norm_targets[best_t_idx]",
        "",
        "            # Compare argument keys",
        "            p_keys = set(p_args.keys())",
        "            t_keys = set(t_args.keys())",
        "            if p_keys and t_keys:",
        "                comm_k = p_keys & t_keys",
        "                k_p = len(comm_k) / len(p_keys)",
        "                k_r = len(comm_k) / len(t_keys)",
        "                key_prec_list.append(k_p)",
        "                key_rec_list.append(k_r)",
        "            elif not p_keys and not t_keys:",
        "                key_prec_list.append(1.0)",
        "                key_rec_list.append(1.0)",
        "            else:",
        "                key_prec_list.append(0.0)",
        "                key_rec_list.append(0.0)",
        "",
        "            # Compare argument values",
        "            if t_keys:",
        "                matching_values = 0",
        "                for k in t_keys:",
        "                    p_val = p_args.get(k)",
        "                    t_val = t_args.get(k)",
        "                    if p_val == t_val or str(p_val).strip().lower() == str(t_val).strip().lower():",
        "                        matching_values += 1",
        "                v_acc = matching_values / len(t_keys)",
        "                val_acc_list.append(v_acc)",
        "                if p_keys == t_keys and v_acc == 1.0:",
        "                    exact_matches += 1",
        "            else:",
        "                if not p_keys:",
        "                    val_acc_list.append(1.0)",
        "                    exact_matches += 1",
        "                else:",
        "                    val_acc_list.append(0.0)",
        "",
        "    if target_has_tool:",
        "        tc_prec = call_matches / len(norm_preds) if norm_preds else 0.0",
        "        tc_rec = call_matches / len(norm_targets)",
        "        tc_f1 = (2 * tc_prec * tc_rec) / (tc_prec + tc_rec) if (tc_prec + tc_rec) > 0 else 0.0",
        "        tool_name_match = (call_matches == len(norm_targets) and len(norm_preds) == len(norm_targets))",
        "        avg_k_p = sum(key_prec_list) / len(norm_targets) if norm_targets else 0.0",
        "        avg_k_r = sum(key_rec_list) / len(norm_targets) if norm_targets else 0.0",
        "        avg_k_f1 = (2 * avg_k_p * avg_k_r) / (avg_k_p + avg_k_r) if (avg_k_p + avg_k_r) > 0 else 0.0",
        "        avg_v_acc = sum(val_acc_list) / len(norm_targets) if norm_targets else 0.0",
        "        full_accuracy = exact_matches / len(norm_targets)",
        "    else:",
        "        if pred_has_tool:",
        "            tc_prec, tc_rec, tc_f1 = 0.0, 0.0, 0.0",
        "            tool_name_match = False",
        "            avg_k_p, avg_k_r, avg_k_f1, avg_v_acc, full_accuracy = 0.0, 0.0, 0.0, 0.0, 0.0",
        "        else:",
        "            tc_prec, tc_rec, tc_f1 = 1.0, 1.0, 1.0",
        "            tool_name_match = None",
        "            avg_k_p, avg_k_r, avg_k_f1, avg_v_acc, full_accuracy = None, None, None, None, 1.0",
        "",
        "    ref_content = target_message.get('content') or ''",
        "    clean_pred = re.sub(r'<think>.*?</think>', '', pred_text, flags=re.DOTALL)",
        "    clean_pred = re.sub(r'<tool_call>.*?</tool_call>', '', clean_pred, flags=re.DOTALL).strip()",
        "    f1_scores = compute_token_f1(clean_pred, ref_content)",
        "",
        "    return {",
        "        'confusion': confusion,",
        "        'is_no_tool_turn': is_no_tool_turn,",
        "        'tool_action_match': tool_action_match,",
        "        'tool_count_target': len(target_tcs),",
        "        'tool_count_pred': len(pred_tcs),",
        "        'tool_name_match': tool_name_match,",
        "        'tool_call_precision': round(tc_prec, 4),",
        "        'tool_call_recall': round(tc_rec, 4),",
        "        'tool_call_f1': round(tc_f1, 4),",
        "        'args_valid_json': args_all_valid_json if pred_has_tool else None,",
        "        'arg_key_precision': round(avg_k_p, 4) if avg_k_p is not None else None,",
        "        'arg_key_recall': round(avg_k_r, 4) if avg_k_r is not None else None,",
        "        'arg_key_f1': round(avg_k_f1, 4) if avg_k_f1 is not None else None,",
        "        'arg_value_accuracy': round(avg_v_acc, 4) if avg_v_acc is not None else None,",
        "        'full_tool_call_accuracy': round(full_accuracy, 4),",
        "        'content_f1': f1_scores['f1'],",
        "        'content_precision': f1_scores['precision'],",
        "        'content_recall': f1_scores['recall']",
        "    }",
        "",
        "def group_into_transaction_blocks(messages):",
        "    blocks = []",
        "    curr_block = []",
        "    for msg in messages:",
        "        role = msg.get('role')",
        "        if role in ['user', 'system']:",
        "            if curr_block:",
        "                blocks.append(curr_block)",
        "                curr_block = []",
        "            curr_block.append(msg)",
        "        elif role == 'assistant':",
        "            if curr_block and curr_block[0].get('role') != 'assistant':",
        "                blocks.append(curr_block)",
        "                curr_block = []",
        "            curr_block.append(msg)",
        "        elif role in ['tool', 'tool_response', 'function']:",
        "            curr_block.append(msg)",
        "        else:",
        "            if curr_block:",
        "                blocks.append(curr_block)",
        "                curr_block = []",
        "            curr_block.append(msg)",
        "    if curr_block:",
        "        blocks.append(curr_block)",
        "    return blocks",
        "",
        "def prune_context_at_message_boundaries(context, tools, tokenizer, max_ctx=10240, min_reserve=64):",
        "    try:",
        "        raw_prompt = tokenizer.apply_chat_template(context, tools=tools, add_generation_prompt=True, enable_thinking=True, tokenize=False)",
        "        raw_toks = tokenizer.encode(raw_prompt, truncation=False)",
        "        raw_count = len(raw_toks)",
        "    except Exception:",
        "        raw_count = sum(len(m.get('content', '').split()) for m in context)",
        "",
        "    budget = max_ctx - min_reserve",
        "    if raw_count <= budget:",
        "        return context, {",
        "            'was_truncated': False,",
        "            'raw_prompt_tokens': raw_count,",
        "            'truncated_prompt_tokens': raw_count,",
        "            'messages_dropped': 0,",
        "            'original_message_count': len(context),",
        "            'final_message_count': len(context)",
        "        }",
        "",
        "    has_system = len(context) > 0 and context[0].get('role') == 'system'",
        "    system_part = [context[0]] if has_system else []",
        "    conv_part = list(context[1:] if has_system else context)",
        "",
        "    # Drop oldest transaction blocks atomically so tool responses are never separated from assistant calls",
        "    blocks = group_into_transaction_blocks(conv_part)",
        "",
        "    while len(blocks) > 1:",
        "        blocks.pop(0)",
        "        cand_context = list(system_part)",
        "        for b in blocks: cand_context.extend(b)",
        "",
        "        # Sanity check: Ensure first message after system is not a tool response",
        "        check_idx = 1 if has_system and len(cand_context) > 1 else 0",
        "        while len(cand_context) > check_idx and cand_context[check_idx].get('role') in ['tool', 'tool_response']:",
        "            cand_context.pop(check_idx)",
        "",
        "        try:",
        "            cand_prompt = tokenizer.apply_chat_template(cand_context, tools=tools, add_generation_prompt=True, enable_thinking=True, tokenize=False)",
        "            cand_toks = tokenizer.encode(cand_prompt, truncation=False)",
        "            if len(cand_toks) <= budget:",
        "                return cand_context, {",
        "                    'was_truncated': True,",
        "                    'raw_prompt_tokens': raw_count,",
        "                    'truncated_prompt_tokens': len(cand_toks),",
        "                    'messages_dropped': len(context) - len(cand_context),",
        "                    'original_message_count': len(context),",
        "                    'final_message_count': len(cand_context)",
        "                }",
        "        except Exception:",
        "            break",
        "",
        "    cand_context = list(system_part)",
        "    for b in blocks: cand_context.extend(b)",
        "    check_idx = 1 if has_system and len(cand_context) > 1 else 0",
        "    while len(cand_context) > check_idx and cand_context[check_idx].get('role') in ['tool', 'tool_response']:",
        "        cand_context.pop(check_idx)",
        "",
        "    try:",
        "        cand_prompt = tokenizer.apply_chat_template(cand_context, tools=tools, add_generation_prompt=True, enable_thinking=True, tokenize=False)",
        "        cand_toks = tokenizer.encode(cand_prompt, truncation=False)",
        "        tok_len = len(cand_toks)",
        "    except Exception:",
        "        tok_len = raw_count",
        "",
        "    return cand_context, {",
        "        'was_truncated': True,",
        "        'raw_prompt_tokens': raw_count,",
        "        'truncated_prompt_tokens': tok_len,",
        "        'messages_dropped': len(context) - len(cand_context),",
        "        'original_message_count': len(context),",
        "        'final_message_count': len(cand_context)",
        "    }",
        "",
        "def main():",
        "    parser = argparse.ArgumentParser()",
        "    parser.add_argument('--gpu_id', type=int, required=True)",
        "    parser.add_argument('--shard_file', type=str, required=True)",
        "    parser.add_argument('--output_file', type=str, required=True)",
        "    parser.add_argument('--base_model', type=str, required=True)",
        "    parser.add_argument('--adapter_repo', type=str, required=True)",
        "    parser.add_argument('--hf_token', type=str, required=True)",
        "    args = parser.parse_args()",
        "",
        "    gpu_id = args.gpu_id",
        "    target_device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'",
        "    device_map = {'': target_device} if torch.cuda.is_available() else 'auto'",
        "    if torch.cuda.is_available():",
        "        torch.cuda.set_device(gpu_id)",
        "        torch.cuda.reset_peak_memory_stats(gpu_id)",
        "",
        "    print(f'[WORKER-GPU{gpu_id}] Initializing on {target_device}...', flush=True)",
        "    with open(args.shard_file, 'r', encoding='utf-8') as f:",
        "        records = [json.loads(line) for line in f if line.strip()]",
        "    print(f'[WORKER-GPU{gpu_id}] Shard size: {len(records)} dialogues', flush=True)",
        "",
        "    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=args.hf_token, trust_remote_code=True)",
        "    if tokenizer.pad_token is None:",
        "        tokenizer.pad_token = tokenizer.eos_token",
        "",
        "    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)",
        "    base_model = AutoModelForCausalLM.from_pretrained(",
        "        args.base_model,",
        "        quantization_config=bnb_cfg,",
        "        device_map=device_map,",
        "        torch_dtype=torch.float16,",
        "        token=args.hf_token,",
        "        trust_remote_code=True",
        "    )",
        "    model = PeftModel.from_pretrained(base_model, args.adapter_repo, token=args.hf_token, device_map=device_map)",
        "    model.eval()",
        "    print(f'[WORKER-GPU{gpu_id}] Model loaded successfully!', flush=True)",
        "",
        "    start_time = time.time()",
        "    total_turns_evaluated = 0",
        "    eval_failures = 0",
        "    eos_ids = [tokenizer.eos_token_id, 151645]",
        "",
        "    with open(args.output_file, 'w', encoding='utf-8') as out_f:",
        "        for d_idx, rec in enumerate(records):",
        "            conv_id = rec.get('id', f'conv_gpu{gpu_id}_{d_idx+1}')",
        "            messages = rec.get('messages', [])",
        "            tools = rec.get('tools') or None",
        "",
        "            for context, target, turn_idx in iter_teacher_forced_turns(messages):",
        "                pruned_ctx, trunc_meta = prune_context_at_message_boundaries(",
        "                    context=context,",
        "                    tools=tools,",
        "                    tokenizer=tokenizer,",
        "                    max_ctx=MAX_CTX,",
        "                    min_reserve=MIN_NEW_TOKENS",
        "                )",
        "",
        "                try:",
        "                    prompt_text = tokenizer.apply_chat_template(",
        "                        pruned_ctx,",
        "                        tools=tools,",
        "                        add_generation_prompt=True,",
        "                        enable_thinking=True,",
        "                        tokenize=False",
        "                    )",
        "                except TypeError:",
        "                    try:",
        "                        prompt_text = tokenizer.apply_chat_template(pruned_ctx, tools=tools, add_generation_prompt=True, tokenize=False)",
        "                    except Exception:",
        "                        prompt_text = ''",
        "                        for m in pruned_ctx:",
        "                            role = m.get('role', 'user')",
        "                            content = m.get('content', '')",
        "                            prompt_text += f'<|im_start|>{role}\\n{content}<|im_end|>\\n'",
        "                        prompt_text += '<|im_start|>assistant\\n'",
        "                except Exception as e:",
        "                    prompt_text = ''",
        "                    for m in pruned_ctx:",
        "                        role = m.get('role', 'user')",
        "                        content = m.get('content', '')",
        "                        prompt_text += f'<|im_start|>{role}\\n{content}<|im_end|>\\n'",
        "                    prompt_text += '<|im_start|>assistant\\n'",
        "",
        "                input_ids = tokenizer.encode(prompt_text, return_tensors='pt', truncation=False).to(target_device)",
        "                prompt_len = input_ids.shape[1]",
        "                available_budget = MAX_CTX - prompt_len",
        "                if available_budget < MIN_NEW_TOKENS:",
        "                    status = 'EVALUATION_FAILURE'",
        "                    err_msg = f'Prompt length {prompt_len} exceeds context budget ({MAX_CTX} - {MIN_NEW_TOKENS})'",
        "                    eval_failures += 1",
        "                    metrics = evaluate_replay_turn(None, target)",
        "                    gen_text = None",
        "                    gen_ids = []",
        "                else:",
        "                    max_new = min(1024, available_budget)",
        "                    try:",
        "                        with torch.no_grad():",
        "                            out = model.generate(",
        "                                input_ids=input_ids,",
        "                                max_new_tokens=max_new,",
        "                                temperature=0.6,",
        "                                top_p=0.95,",
        "                                repetition_penalty=1.05,",
        "                                do_sample=True,",
        "                                pad_token_id=tokenizer.pad_token_id,",
        "                                eos_token_id=eos_ids",
        "                            )",
        "                        gen_ids = out[0][prompt_len:]",
        "                        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()",
        "                        if '<|im_end|>' in gen_text: gen_text = gen_text.split('<|im_end|>')[0].strip()",
        "                        if '<|endoftext|>' in gen_text: gen_text = gen_text.split('<|endoftext|>')[0].strip()",
        "                        status = 'SUCCESS'",
        "                        err_msg = None",
        "                        metrics = evaluate_replay_turn(gen_text, target)",
        "                    except Exception as gen_err:",
        "                        print(f'[WORKER-GPU{gpu_id}] EVALUATION FAILURE on Conv {conv_id} Turn {turn_idx}: {gen_err}', flush=True)",
        "                        gen_text = None",
        "                        gen_ids = []",
        "                        status = 'EVALUATION_FAILURE'",
        "                        err_msg = str(gen_err)",
        "                        eval_failures += 1",
        "                        metrics = evaluate_replay_turn(None, target)",
        "",
        "                total_turns_evaluated += 1",
        "                peak_vram_mb = round(torch.cuda.max_memory_allocated(gpu_id) / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0",
        "",
        "                entry = {",
        "                    'conv_id': conv_id,",
        "                    'turn_index': turn_idx,",
        "                    'gpu_id': gpu_id,",
        "                    'status': status,",
        "                    'error': err_msg,",
        "                    'generated_text': gen_text,",
        "                    'target_message': target,",
        "                    'prompt_tokens': prompt_len,",
        "                    'generated_tokens': len(gen_ids),",
        "                    'truncation_info': trunc_meta,",
        "                    'peak_vram_mb': peak_vram_mb,",
        "                    'metrics': metrics,",
        "                    'timestamp': datetime.datetime.now().isoformat()",
        "                }",
        "                out_f.write(json.dumps(entry) + '\\n')",
        "                out_f.flush()",
        "",
        "                c_tag = metrics.get('confusion', 'UNKNOWN')",
        "                tool_str = f'[{c_tag}] ' + ('TOOL_CALL' if metrics.get('tool_name_match') else ('CORRECT_TEXT' if c_tag == 'TN' else ('ERROR' if c_tag == 'ERROR' else 'MISMATCH')))",
        "                print(f'[WORKER-GPU{gpu_id}] Dia {d_idx+1}/{len(records)} ({conv_id}) Turn {turn_idx+1} -> {tool_str} ({len(gen_ids)} toks, prompt: {prompt_len})', flush=True)",
        "",
        "            elapsed_dia = round(time.time() - start_time, 1)",
        "            print(f'[WORKER-GPU{gpu_id}] >>> Completed Dialogue {d_idx+1}/{len(records)} ({conv_id}) | Turns: {len(messages)} | Elapsed: {elapsed_dia}s', flush=True)",
        "",
        "    peak_mb = round(torch.cuda.max_memory_allocated(gpu_id) / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0",
        "    print(f'[WORKER-GPU{gpu_id}] FINISHED {len(records)} dialogues ({total_turns_evaluated} turns, {eval_failures} failures)! Peak VRAM: {peak_mb} MB', flush=True)",
        "",
        "if __name__ == '__main__':",
        "    main()"
    ])

    cells.append(create_cell("code", f"""# ==============================================================================
# CELL 5: WRITE DUAL-GPU REPLAY WORKER SCRIPT
# ==============================================================================
worker_code = {repr(worker_script)}

worker_path = "/kaggle/working/dual_gpu_replay_worker.py"
with open(worker_path, "w", encoding="utf-8") as f:
    f.write(worker_code)

print(f"[SETUP] Safeguarded worker script written to: {{worker_path}}", flush=True)
"""))

    # Cell 6: Plain string execution orchestrator
    cell_6_code = """# ==============================================================================
# CELL 6: DUAL-GPU PARALLEL EXECUTION & METRICS COMPILATION
# ==============================================================================
import subprocess
import time
import math
import json
import os
import sys

total_dia = len(shard_records)
mid = math.ceil(total_dia / 2)
shard0_records = shard_records[:mid]
shard1_records = shard_records[mid:]

shard0_in = "/kaggle/working/shard0_in.jsonl"
shard1_in = "/kaggle/working/shard1_in.jsonl"
shard0_out = "/kaggle/working/shard0_out.jsonl"
shard1_out = "/kaggle/working/shard1_out.jsonl"

with open(shard0_in, "w", encoding="utf-8") as f:
    for r in shard0_records: f.write(json.dumps(r) + "\\n")

with open(shard1_in, "w", encoding="utf-8") as f:
    for r in shard1_records: f.write(json.dumps(r) + "\\n")

print(f"[EVAL] Distributing {total_dia} dialogues: GPU 0 ({len(shard0_records)}), GPU 1 ({len(shard1_records)})...", flush=True)

cmd0 = [
    sys.executable, "-u", "/kaggle/working/dual_gpu_replay_worker.py",
    "--gpu_id", "0",
    "--shard_file", shard0_in,
    "--output_file", shard0_out,
    "--base_model", BASE_MODEL_REPO,
    "--adapter_repo", HF_REPO_ID,
    "--hf_token", HF_TOKEN
]

cmd1 = [
    sys.executable, "-u", "/kaggle/working/dual_gpu_replay_worker.py",
    "--gpu_id", "1",
    "--shard_file", shard1_in,
    "--output_file", shard1_out,
    "--base_model", BASE_MODEL_REPO,
    "--adapter_repo", HF_REPO_ID,
    "--hf_token", HF_TOKEN
]

import threading

def stream_worker_output(pipe, label):
    for line in iter(pipe.readline, ''):
        if not line: break
        print(f"[{label}] {line.rstrip()}", flush=True)

t0 = time.time()
p0 = subprocess.Popen(cmd0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

th0 = threading.Thread(target=stream_worker_output, args=(p0.stdout, "GPU 0"))
th1 = threading.Thread(target=stream_worker_output, args=(p1.stdout, "GPU 1"))
th0.start()
th1.start()

ret0 = p0.wait()
ret1 = p1.wait()
th0.join()
th1.join()
elapsed = time.time() - t0

print(f"[EVAL] Subprocesses finished: GPU 0 exit={ret0}, GPU 1 exit={ret1}. Total elapsed: {elapsed:.1f}s", flush=True)

# Combine shard outputs
final_pred_file = f"{OUTPUT_DIR}/replay_predictions.jsonl"
final_summary_file = f"{OUTPUT_DIR}/replay_summary.json"

all_results = []
for out_path in [shard0_out, shard1_out]:
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

with open(final_pred_file, "w", encoding="utf-8") as f:
    for r in all_results:
        f.write(json.dumps(r) + "\\n")

def compute_spectrum_metrics(records):
    cnt = max(1, len(records))
    tp = sum(1 for r in records if r.get("metrics", {}).get("confusion") == "TP")
    fp = sum(1 for r in records if r.get("metrics", {}).get("confusion") == "FP")
    fn = sum(1 for r in records if r.get("metrics", {}).get("confusion") == "FN")
    tn = sum(1 for r in records if r.get("metrics", {}).get("confusion") == "TN")
    errs = sum(1 for r in records if r.get("metrics", {}).get("confusion") == "ERROR")

    # True ground truth tool turns (must retain failed turns in recall denominator!)
    tool_req = sum(1 for r in records if len(r.get("target_message", {}).get("tool_calls", [])) > 0)
    no_tool = cnt - tool_req
    emitted = tp + fp

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / tool_req if tool_req > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    action_concordance = (tp + tn) / cnt
    no_tool_adherence = tn / no_tool if no_tool > 0 else 1.0

    tool_name_matches = sum(1 for r in records if r.get("metrics", {}).get("tool_name_match") is True)
    tool_name_accuracy = tool_name_matches / tool_req if tool_req > 0 else 0.0

    if emitted > 0:
        valid_schema_count = sum(1 for r in records if r.get("metrics", {}).get("args_valid_json") is True)
        schema_validity = round(valid_schema_count / emitted, 4)
    else:
        schema_validity = None

    tool_req_data = [r for r in records if len(r.get("target_message", {}).get("tool_calls", [])) > 0]
    mean_call_recall = sum(r.get("metrics", {}).get("tool_call_recall", 0.0) or 0.0 for r in tool_req_data) / max(1, len(tool_req_data))
    mean_arg_key_f1 = sum(r.get("metrics", {}).get("arg_key_f1", 0.0) or 0.0 for r in tool_req_data) / max(1, len(tool_req_data))
    mean_arg_val_acc = sum(r.get("metrics", {}).get("arg_value_accuracy", 0.0) or 0.0 for r in tool_req_data) / max(1, len(tool_req_data))
    mean_full_accuracy = sum(r.get("metrics", {}).get("full_tool_call_accuracy", 0.0) or 0.0 for r in tool_req_data) / max(1, len(tool_req_data))
    mean_content_f1 = sum(r.get("metrics", {}).get("content_f1", 0.0) or 0.0 for r in records) / cnt

    return {
        "turns_count": len(records),
        "confusion_matrix": {"true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn, "errors": errs},
        "tool_required_turns": tool_req,
        "no_tool_turns": no_tool,
        "emitted_tool_calls": emitted,
        "tool_decision_precision": round(prec, 4),
        "tool_decision_recall": round(rec, 4),
        "tool_decision_f1": round(f1, 4),
        "action_concordance_rate": round(action_concordance, 4),
        "no_tool_adherence_rate": round(no_tool_adherence, 4),
        "tool_name_accuracy_on_tool_required_turns": round(tool_name_accuracy, 4),
        "tool_call_level_recall": round(mean_call_recall, 4),
        "schema_validity_on_emitted_calls": schema_validity,
        "mean_argument_key_f1": round(mean_arg_key_f1, 4),
        "mean_argument_value_accuracy": round(mean_arg_val_acc, 4),
        "full_tool_call_exact_accuracy": round(mean_full_accuracy, 4),
        "mean_content_f1": round(mean_content_f1, 4),
        "evaluation_errors": errs
    }

overall = compute_spectrum_metrics(all_results)
standard_records = [r for r in all_results if r.get("prompt_tokens", 0) <= 4096]
long_records = [r for r in all_results if r.get("prompt_tokens", 0) > 4096]

standard_summary = compute_spectrum_metrics(standard_records)
long_summary = compute_spectrum_metrics(long_records)

peak_vram = max((r.get("peak_vram_mb", 0.0) for r in all_results), default=0.0)
truncated_turns = sum(1 for r in all_results if r.get("truncation_info", {}).get("was_truncated"))
total_errors = sum(1 for r in all_results if r.get("status") == "EVALUATION_FAILURE")

summary = {
    "shard_index": SHARD_INDEX,
    "model_version": "v3_full_1.0_epoch_step1262",
    "base_model": BASE_MODEL_REPO,
    "adapter_repo": HF_REPO_ID,
    "total_dialogues": total_dia,
    "total_turns_evaluated": len(all_results),
    "total_evaluation_failures": total_errors,
    "total_truncated_turns": truncated_turns,
    "peak_vram_mb": peak_vram,
    "total_evaluation_seconds": round(elapsed, 2),
    "overall_metrics": overall,
    "standard_context_metrics_le_4096": standard_summary,
    "long_context_metrics_gt_4096": long_summary,
    **overall
}

with open(final_summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"[SUMMARY] Shard {SHARD_INDEX} complete! Evaluated: {len(all_results)} turns | Failures: {total_errors} | Peak VRAM: {peak_vram} MB", flush=True)
print(f"  Overall: TP={overall['confusion_matrix']['true_positives']}, FP={overall['confusion_matrix']['false_positives']}, FN={overall['confusion_matrix']['false_negatives']}, TN={overall['confusion_matrix']['true_negatives']}", flush=True)
print(f"  Standard (<=4096): {standard_summary['turns_count']} turns | Decision F1: {standard_summary['tool_decision_f1']*100:.2f}%", flush=True)
print(f"  Long (>4096):     {long_summary['turns_count']} turns | Decision F1: {long_summary['tool_decision_f1']*100:.2f}%", flush=True)
"""
    cells.append(create_cell("code", cell_6_code))

    # Cell 7: Push to Hugging Face (Optional Guarded)
    cell_7_code = """# ==============================================================================
# CELL 7: OPTIONAL PUSH TO HUGGING FACE REPOSITORY
# ==============================================================================
if not PUSH_TO_HUB:
    print(f"[DEPLOY] PUSH_TO_HUB is disabled ({PUSH_TO_HUB}). All evaluation results preserved safely in {OUTPUT_DIR}.", flush=True)
    print(f"[DEPLOY] No commits pushed to Hugging Face. Ready for local inspection.", flush=True)
else:
    from huggingface_hub import HfApi
    print(f"[DEPLOY] Pushing Shard {SHARD_INDEX} outputs to Hugging Face repo: {HF_REPO_ID}...", flush=True)
    try:
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=final_pred_file,
            path_in_repo=f"evaluations/replay_v3/shard_{SHARD_INDEX}/replay_predictions.jsonl",
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        api.upload_file(
            path_or_fileobj=final_summary_file,
            path_in_repo=f"evaluations/replay_v3/shard_{SHARD_INDEX}/replay_summary.json",
            repo_id=HF_REPO_ID,
            repo_type="model"
        )
        print(f"[DEPLOY] SUCCESS: Shard {SHARD_INDEX} pushed to '{HF_REPO_ID}/evaluations/replay_v3/shard_{SHARD_INDEX}'!", flush=True)
    except Exception as e:
        print(f"[DEPLOY] ERROR: Failed to upload to Hugging Face: {e}", flush=True)
        raise RuntimeError(f"FATAL: Hugging Face upload failed for Shard {SHARD_INDEX}: {e}")
"""
    cells.append(create_cell("code", cell_7_code))

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return nb

def main():
    root_dir = os.getenv("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    shards_dir = os.getenv("SHARDS_DIR", os.path.join(root_dir, "v3_pipeline", "eval_shards"))
    deploy_root = os.getenv("OUTPUT_DIR", os.path.join(root_dir, "v3_pipeline", "kaggle_eval_multiturn"))

    print("[BUILD] Generating 3-Way Tool Replay Evaluation Notebooks and Packages...")
    for cfg in ACCOUNTS:
        s_idx = cfg["shard_idx"]
        u = cfg["username"]
        s_dir = os.path.join(deploy_root, f"shard_{s_idx}")
        os.makedirs(s_dir, exist_ok=True)

        shard_data_path = os.path.join(shards_dir, f"tool_replay_shard_{s_idx}.jsonl")
        assert os.path.exists(shard_data_path), f"Shard data missing: {shard_data_path}"

        with open(shard_data_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        # 1. Copy raw split data file into the shard folder
        dest_data_path = os.path.join(s_dir, f"tool_replay_shard_{s_idx}.jsonl")
        shutil.copy2(shard_data_path, dest_data_path)

        # 2. Build Jupyter Notebook
        nb_dict = generate_shard_notebook(
            shard_idx=s_idx,
            total_shards=3,
            records=records,
            account_username=u
        )
        nb_filename = f"qwen3_8b_v3_tool_replay_eval_shard_{s_idx}.ipynb"
        nb_path = os.path.join(s_dir, nb_filename)
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb_dict, f, indent=2)

        # 3. Write kernel-metadata.json
        meta = {
            "id": cfg["slug"],
            "title": cfg["title"],
            "code_file": nb_filename,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True
        }
        meta_path = os.path.join(s_dir, "kernel-metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        asst_turns = sum(sum(1 for m in r.get("messages", []) if m.get("role") == "assistant") for r in records)
        print(f"[SHARD {s_idx}] Account: {u:20s} | Dialogues: {len(records):2d} | Turns: {asst_turns:3d} | Saved: {nb_path}")

    print("\n[SUCCESS] All 3 deployment packages built with self-contained notebooks and standalone split files!")

if __name__ == "__main__":
    main()

