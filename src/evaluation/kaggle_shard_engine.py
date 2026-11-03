#!/usr/bin/env python3
"""
Dynamic Kaggle Sharding Engine
==============================
Centralized orchestration engine for partitioning evaluation holdouts and
generating self-contained, dual-GPU Kaggle deployment packages.

Supported Tracks:
  - exploit:      Exploit Trajectories (Tool calling, schema validity, execution safety)
  - tool_replay:  Multi-Turn Tool Replay (Teacher-forced multi-turn trajectory continuity)
  - reasoning:    Cybersecurity Reasoning (Direct answering vs deep reasoning <think> modes)
  - all:          Unified dispatch across all three evaluation tracks

Capabilities:
  - Dynamic Shard Count: Specify any integer N, or derive from comma-separated KAGGLE_ACCOUNTS.
  - Zero Dialogue Fragmentation: Strict dialogue-level preservation across shard partitions.
  - Programmatic Notebook Generation: Standalone .ipynb notebooks with embedded zlib/base64
    compressed datasets, dual-GPU parallel execution, and automated metric reconciliation.
  - Metadata Generation: Kaggle-compliant kernel-metadata.json configured for Dual T4 GPUs.
  - CLI & Library Interface: Usable via command line or imported as a Python module.
"""

import os
import sys
import json
import zlib
import base64
import math
import shutil
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_ACCOUNTS = [
    os.getenv("KAGGLE_USER_SHARD1", "shard1_user"),
    os.getenv("KAGGLE_USER_SHARD2", "shard2_user"),
    os.getenv("KAGGLE_USER_SHARD3", "shard3_user"),
]

DEFAULT_INPUT_PATHS = {
    "exploit": os.path.join("data", "clean", "test", "exploit_60pct.jsonl"),
    "tool_replay": os.path.join("data", "clean", "test", "tool_replay_20pct.jsonl"),
    "reasoning": os.path.join("data", "clean", "test", "reasoning_20pct.jsonl"),
}

DEFAULT_MODEL_ID = os.getenv("HF_REPO_ID", "Qwen/Qwen3-8B-offensive-cyber-agent")
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"


@dataclass
class KaggleShardConfig:
    track: str
    num_shards: Optional[int] = None
    accounts: Optional[List[str]] = None
    from_env: bool = False
    input_data: Optional[str] = None
    output_dir: str = "kaggle_deploys"
    model_id: str = DEFAULT_MODEL_ID
    base_model: str = DEFAULT_BASE_MODEL
    dry_run: bool = False


def resolve_accounts(
    num_shards: Optional[int] = None,
    accounts: Optional[List[str]] = None,
    from_env: bool = False
) -> Tuple[int, List[str]]:
    """
    Resolve number of shards and Kaggle account handles.
    
    Priority:
    1. If from_env=True or (KAGGLE_ACCOUNTS is in env and accounts is None and not num_shards):
       derive accounts from KAGGLE_ACCOUNTS env var.
    2. If explicit accounts list is provided, use it.
    3. If neither, fallback to DEFAULT_ACCOUNTS.
    4. If num_shards is explicitly given, align account list length to num_shards.
    """
    env_accs_raw = os.environ.get("KAGGLE_ACCOUNTS", "").strip()
    resolved_accounts: List[str] = []

    if from_env or (env_accs_raw and accounts is None and num_shards is None):
        if not env_accs_raw:
            raise ValueError("KAGGLE_ACCOUNTS environment variable is empty or not set.")
        resolved_accounts = [a.strip() for a in env_accs_raw.split(",") if a.strip()]
    elif accounts:
        resolved_accounts = [a.strip() for a in accounts if a.strip()]
    else:
        resolved_accounts = list(DEFAULT_ACCOUNTS)

    if not resolved_accounts:
        resolved_accounts = list(DEFAULT_ACCOUNTS)

    # Determine final shard count
    if num_shards is None:
        final_num_shards = len(resolved_accounts)
    else:
        if num_shards <= 0:
            raise ValueError(f"num_shards must be a positive integer, got {num_shards}")
        final_num_shards = num_shards

    # Align account list with shard count
    if len(resolved_accounts) < final_num_shards:
        needed = final_num_shards - len(resolved_accounts)
        start_idx = len(resolved_accounts) + 1
        for i in range(needed):
            resolved_accounts.append(f"kaggle_worker_{start_idx + i}")
    elif len(resolved_accounts) > final_num_shards:
        resolved_accounts = resolved_accounts[:final_num_shards]

    return final_num_shards, resolved_accounts


def partition_dialogues(
    dialogues: List[Dict[str, Any]],
    num_shards: int
) -> List[List[Dict[str, Any]]]:
    """
    Split dialogues into num_shards balanced partitions.
    Guarantees:
      - Dialogue-level preservation (zero cross-shard dialogue fragmentation).
      - Lossless roundtrip (all dialogues accounted for, zero duplicates).
      - Balanced distribution across shards.
    """
    if num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    n = len(dialogues)
    if n == 0:
        return [[] for _ in range(num_shards)]

    base_size = n // num_shards
    remainder = n % num_shards

    shards: List[List[Dict[str, Any]]] = []
    start = 0
    for i in range(num_shards):
        size = base_size + (1 if i < remainder else 0)
        shard = dialogues[start : start + size]
        shards.append(shard)
        start += size

    # Integrity verification
    total_partitioned = sum(len(s) for s in shards)
    assert total_partitioned == n, f"Partition count mismatch: {total_partitioned} != {n}"

    id_seen = set()
    for s_idx, shard in enumerate(shards, 1):
        for d in shard:
            d_id = d.get("id")
            if d_id is not None:
                if d_id in id_seen:
                    raise ValueError(f"Duplicate dialogue ID '{d_id}' found across shards (shard {s_idx})")
                id_seen.add(d_id)

    return shards


def create_cell(cell_type: str, source: str) -> Dict[str, Any]:
    """Constructs a Jupyter Notebook cell structure."""
    lines = [s + "\n" for s in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines.pop()
    cell = {"cell_type": cell_type, "metadata": {}, "source": lines}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def compress_payload(records: List[Dict[str, Any]]) -> str:
    """Serializes and zlib-compresses records into a base64 ASCII string."""
    raw_bytes = json.dumps(records).encode("utf-8")
    compressed = zlib.compress(raw_bytes, level=9)
    return base64.b64encode(compressed).decode("ascii")


def _generate_exploit_worker_code() -> str:
    """Generates the dual-GPU worker script content for exploit evaluation."""
    return '''import os
import sys
import json
import re
import shlex
import ast
import time
import argparse
import traceback
from typing import Dict, Any, List, Optional

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MAX_CTX = 4096
MIN_RESERVE = 512
SAFE_PROMPT_BUDGET = MAX_CTX - MIN_RESERVE

def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls = []
    pattern = re.compile(r'<tool_call>\\s*(.*?)\\s*</tool_call>', re.DOTALL)
    for m in pattern.findall(text):
        c = m.strip()
        if c.startswith('```json'): c = c[7:]
        if c.startswith('```'): c = c[3:]
        if c.endswith('```'): c = c[:-3]
        c = c.strip()
        try:
            p = json.loads(c)
            if isinstance(p, dict) and 'name' in p:
                args = p.get('arguments', {})
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception: pass
                p['arguments'] = args
                calls.append(p)
        except Exception:
            pass
    return calls

def validate_static_command(cmd: str, cmd_type: str = 'shell'):
    if not cmd or not cmd.strip():
        return False, 'empty_command'
    if cmd_type == 'python':
        try:
            ast.parse(cmd)
            return True, None
        except SyntaxError as e:
            return False, f'python_syntax: {e}'
        except Exception as e:
            return False, f'python_err: {e}'
    try:
        toks = shlex.split(cmd, posix=True)
        if not toks:
            return False, 'empty_tokens'
        return True, None
    except Exception as e:
        return False, f'shell_err: {e}'

def compute_token_f1(pred: str, ref: str) -> Dict[str, float]:
    p_toks = re.findall(r'\\w+', pred.lower())
    r_toks = re.findall(r'\\w+', ref.lower())
    if not p_toks or not r_toks:
        s = 1.0 if p_toks == r_toks else 0.0
        return {'precision': s, 'recall': s, 'f1': s}
    common = set(p_toks) & set(r_toks)
    if not common:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    prec = len(common) / len(p_toks)
    rec = len(common) / len(r_toks)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {'precision': prec, 'recall': rec, 'f1': f1}

def prune_context_safely(context: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], tokenizer):
    prompt_str = tokenizer.apply_chat_template(context, tools=tools, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(prompt_str, return_tensors='pt')['input_ids'][0]
    if len(tokens) <= SAFE_PROMPT_BUDGET:
        return prompt_str, len(tokens), False

    sys_msg = context[0] if (context and context[0].get('role') == 'system') else None
    remaining = context[1:] if sys_msg else list(context)
    was_pruned = False

    while remaining and len(tokens) > SAFE_PROMPT_BUDGET:
        was_pruned = True
        remaining.pop(0)
        temp_ctx = ([sys_msg] if sys_msg else []) + remaining
        prompt_str = tokenizer.apply_chat_template(temp_ctx, tools=tools, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(prompt_str, return_tensors='pt')['input_ids'][0]

    return prompt_str, len(tokens), was_pruned

def evaluate_turn(pred_text: str, target_message: Dict[str, Any], prev_tool_content: Optional[str] = None):
    pred_tools = extract_tool_calls(pred_text)
    target_tools = target_message.get('tool_calls') or []

    pred_has_tool = len(pred_tools) > 0
    target_has_tool = len(target_tools) > 0

    if target_has_tool and pred_has_tool: confusion = 'TP'
    elif not target_has_tool and pred_has_tool: confusion = 'FP'
    elif target_has_tool and not pred_has_tool: confusion = 'FN'
    else: confusion = 'TN'

    tool_action_match = (pred_has_tool == target_has_tool)
    args_valid_json = True
    syntax_valid = True
    syntax_error = None

    if pred_has_tool:
        for pt in pred_tools:
            args = pt.get('arguments', {})
            if isinstance(args, str):
                try: json.loads(args)
                except Exception: args_valid_json = False
            for k in ['command', 'cmd', 'script', 'code']:
                if k in args:
                    ctype = 'python' if ('python' in k or 'script' in k) else 'shell'
                    v, err = validate_static_command(str(args[k]), ctype)
                    if not v:
                        syntax_valid = False
                        syntax_error = err

    if target_has_tool and pred_has_tool:
        pred_names = {t.get('name') for t in pred_tools}
        target_names = {t.get('function', {}).get('name') if 'function' in t else t.get('name') for t in target_tools}
        tool_name_match = (pred_names == target_names)
    elif target_has_tool and not pred_has_tool:
        tool_name_match = False
    elif not target_has_tool and pred_has_tool:
        tool_name_match = False
    else:
        tool_name_match = True

    is_pivot_turn = False
    pivot_success = None
    if prev_tool_content:
        lower_prev = prev_tool_content.lower()
        if any(term in lower_prev for term in ['blocked', 'denied', 'malicious', 'signature detected', 'prevented']):
            is_pivot_turn = True
            pivot_success = bool(pred_has_tool or (pred_text and len(pred_text) > 20))

    ref_content = target_message.get('content') or ''
    pred_reasoning = re.sub(r'<tool_call>.*?</tool_call>', '', pred_text, flags=re.DOTALL).strip()
    f1_res = compute_token_f1(pred_reasoning, ref_content)

    return {
        'confusion': confusion,
        'tool_action_match': tool_action_match,
        'tool_name_match': tool_name_match,
        'args_valid_json': args_valid_json,
        'syntax_valid': syntax_valid,
        'syntax_error': syntax_error,
        'is_pivot_turn': is_pivot_turn,
        'pivot_success': pivot_success,
        'content_f1': f1_res['f1'],
        'content_precision': f1_res['precision'],
        'content_recall': f1_res['recall']
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_file', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--adapter_id', type=str, default=os.getenv('HF_REPO_ID', 'Qwen/Qwen3-8B'))
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--hf_token', type=str, default='')
    args = parser.parse_args()

    device = f'cuda:{args.gpu_id}'
    print(f'[WORKER-GPU{args.gpu_id}] Loading base model {args.base_model} on {device}...')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=args.hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={'': device},
        token=args.hf_token,
        trust_remote_code=True
    )

    print(f'[WORKER-GPU{args.gpu_id}] Applying adapter {args.adapter_id}...')
    model = PeftModel.from_pretrained(base_model, args.adapter_id, token=args.hf_token)
    model.eval()

    with open(args.shard_file, 'r', encoding='utf-8') as f:
        dialogues = [json.loads(line) for line in f if line.strip()]

    print(f'[WORKER-GPU{args.gpu_id}] Beginning evaluation on {len(dialogues)} dialogues...')
    start_time = time.time()
    turns_done = 0

    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        for d_idx, d in enumerate(dialogues, 1):
            conv_id = d.get('id', f'conv_{d_idx}')
            tools = d.get('tools')
            messages = d.get('messages', [])

            for t_idx, msg in enumerate(messages):
                if msg.get('role') != 'assistant':
                    continue

                context = messages[:t_idx]
                target = msg

                prev_tool_content = None
                for prev in reversed(context):
                    if prev.get('role') == 'tool':
                        prev_tool_content = prev.get('content')
                        break

                prompt_str, prompt_len, was_pruned = prune_context_safely(context, tools, tokenizer)

                try:
                    inputs = tokenizer(prompt_str, return_tensors='pt').to(device)
                    budget = min(MIN_RESERVE, MAX_CTX - inputs['input_ids'].shape[1])
                    if budget <= 16:
                        budget = 64

                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=budget,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.95,
                            repetition_penalty=1.05,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )

                    gen_ids = outputs[0][inputs['input_ids'].shape[1]:]
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                    metrics = evaluate_turn(gen_text, target, prev_tool_content=prev_tool_content)
                    rec = {
                        'conv_id': conv_id,
                        'turn_index': t_idx,
                        'gpu_id': args.gpu_id,
                        'status': 'SUCCESS',
                        'prompt_tokens': prompt_len,
                        'was_pruned': was_pruned,
                        'generated_text': gen_text,
                        'metrics': metrics,
                        'timestamp': time.time()
                    }
                except Exception as exc:
                    rec = {
                        'conv_id': conv_id,
                        'turn_index': t_idx,
                        'gpu_id': args.gpu_id,
                        'status': 'EVALUATION_FAILURE',
                        'prompt_tokens': prompt_len,
                        'was_pruned': was_pruned,
                        'error': str(exc),
                        'timestamp': time.time()
                    }

                out_f.write(json.dumps(rec) + '\\n')
                out_f.flush()
                turns_done += 1

            if d_idx % 5 == 0 or d_idx == len(dialogues):
                elapsed = (time.time() - start_time) / 60.0
                print(f'[WORKER-GPU{args.gpu_id}] Progress: [{d_idx}/{len(dialogues)} dialogues] | Turns: {turns_done} | Elapsed: {elapsed:.1f}m')

    print(f'[WORKER-GPU{args.gpu_id}] Shard evaluation complete in {(time.time() - start_time)/60.0:.2f} mins.')

if __name__ == '__main__':
    main()
'''


def _generate_tool_replay_worker_code() -> str:
    """Generates the dual-GPU worker script content for multi-turn tool replay."""
    return '''import os
import sys
import json
import re
import time
import argparse
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MAX_CTX = 10240
MIN_NEW_TOKENS = 64

def iter_teacher_forced_turns(messages):
    context = []
    turn_idx = 0
    for msg in messages:
        if msg.get('role') == 'assistant':
            yield list(context), dict(msg), turn_idx
            turn_idx += 1
        context.append(msg)

def extract_replay_tool_calls(text):
    tool_calls = []
    pattern = re.compile(r'<tool_call>\\s*(.*?)\\s*</tool_call>', re.DOTALL)
    for m in pattern.findall(text):
        m_clean = m.strip()
        if m_clean.startswith('```json'): m_clean = m_clean[7:]
        if m_clean.startswith('```'): m_clean = m_clean[3:]
        if m_clean.endswith('```'): m_clean = m_clean[:-3]
        m_clean = m_clean.strip()
        try:
            parsed = json.loads(m_clean)
            if isinstance(parsed, dict) and 'name' in parsed:
                args = parsed.get('arguments', {})
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception: pass
                parsed['arguments'] = args
                tool_calls.append(parsed)
        except Exception:
            continue
    return tool_calls

def compute_token_f1(prediction, reference):
    p_toks = re.findall(r'\\w+|[^\\w\\s]', prediction.lower())
    r_toks = re.findall(r'\\w+|[^\\w\\s]', reference.lower())
    if not p_toks or not r_toks:
        score = 1.0 if p_toks == r_toks else 0.0
        return {'precision': score, 'recall': score, 'f1': score}
    p_cnt = {}
    for t in p_toks: p_cnt[t] = p_cnt.get(t, 0) + 1
    r_cnt = {}
    for t in r_toks: r_cnt[t] = r_cnt.get(t, 0) + 1
    comm = sum(min(c, r_cnt.get(t, 0)) for t, c in p_cnt.items())
    prec = comm / max(1, len(p_toks))
    rec = comm / max(1, len(r_toks))
    f1 = (2 * prec * rec) / max(1e-8, prec + rec)
    return {'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4)}

def compare_tool_calls(pred_calls, target_calls):
    if not target_calls and not pred_calls:
        return {'turn_correct': True, 'tool_name_match': True, 'arg_exact_match': True, 'arg_subset_match': True, 'emitted_when_none_expected': False, 'missed_expected_call': False}
    if target_calls and not pred_calls:
        return {'turn_correct': False, 'tool_name_match': False, 'arg_exact_match': False, 'arg_subset_match': False, 'emitted_when_none_expected': False, 'missed_expected_call': True}
    if not target_calls and pred_calls:
        return {'turn_correct': False, 'tool_name_match': False, 'arg_exact_match': False, 'arg_subset_match': False, 'emitted_when_none_expected': True, 'missed_expected_call': False}

    p = pred_calls[0]
    t = target_calls[0]
    p_name = p.get('name', '')
    t_name = t.get('function', {}).get('name') if 'function' in t else t.get('name', '')
    name_match = (p_name == t_name)

    p_args = p.get('arguments', {})
    t_args = t.get('function', {}).get('arguments') if 'function' in t else t.get('arguments', {})
    if isinstance(t_args, str):
        try: t_args = json.loads(t_args)
        except Exception: pass
    if not isinstance(p_args, dict): p_args = {}
    if not isinstance(t_args, dict): t_args = {}

    exact_match = (name_match and p_args == t_args)
    subset_match = False
    if name_match:
        subset_match = all(p_args.get(k) == v for k, v in t_args.items())

    return {
        'turn_correct': exact_match,
        'tool_name_match': name_match,
        'arg_exact_match': exact_match,
        'arg_subset_match': subset_match,
        'emitted_when_none_expected': False,
        'missed_expected_call': False
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--shard_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--adapter_id', type=str, required=True)
    parser.add_argument('--hf_token', type=str, default='')
    args = parser.parse_args()

    gpu_id = args.gpu_id
    torch.cuda.set_device(gpu_id)

    with open(args.shard_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f'[WORKER-GPU{gpu_id}] Loading base model {args.base_model}...')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=args.hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={'': gpu_id},
        torch_dtype=torch.float16,
        token=args.hf_token,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_id, token=args.hf_token)
    model.eval()

    total_turns = sum(sum(1 for m in r.get('messages', []) if m.get('role') == 'assistant') for r in records)
    print(f'[WORKER-GPU{gpu_id}] Evaluating {len(records)} dialogues ({total_turns} turns)...')

    start_bench = time.time()
    turns_evaluated = 0

    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        for d_idx, rec in enumerate(records, 1):
            dialogue_id = rec.get('id', f'diag_{d_idx}')
            tools = rec.get('tools')
            messages = rec.get('messages', [])

            for ctx, target_msg, t_idx in iter_teacher_forced_turns(messages):
                turns_evaluated += 1
                ref_text = target_msg.get('content') or ''
                target_tool_calls = target_msg.get('tool_calls') or []

                prompt_str = tokenizer.apply_chat_template(ctx, tools=tools, tokenize=False, add_generation_prompt=True)
                encoded = tokenizer(prompt_str, return_tensors='pt')
                input_ids = encoded['input_ids'].to(f'cuda:{gpu_id}')
                ctx_len = input_ids.shape[1]

                max_gen = max(MIN_NEW_TOKENS, min(1024, MAX_CTX - ctx_len))
                t0 = time.time()

                try:
                    with torch.inference_mode():
                        outputs = model.generate(
                            input_ids,
                            max_new_tokens=max_gen,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.95,
                            repetition_penalty=1.05,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    gen_ids = outputs[0][ctx_len:]
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    lat = time.time() - t0

                    pred_tool_calls = extract_replay_tool_calls(gen_text)
                    tc_metrics = compare_tool_calls(pred_tool_calls, target_tool_calls)
                    f1_metrics = compute_token_f1(gen_text, ref_text)

                    out_rec = {
                        'dialogue_id': dialogue_id,
                        'turn_index': t_idx,
                        'gpu_id': gpu_id,
                        'prompt_tokens': ctx_len,
                        'gen_latency_sec': round(lat, 3),
                        'gen_tokens': len(gen_ids),
                        'generated_text': gen_text,
                        'pred_tool_calls': pred_tool_calls,
                        'target_tool_calls': target_tool_calls,
                        'tool_metrics': tc_metrics,
                        'content_metrics': f1_metrics,
                        'status': 'SUCCESS'
                    }
                except Exception as exc:
                    out_rec = {
                        'dialogue_id': dialogue_id,
                        'turn_index': t_idx,
                        'gpu_id': gpu_id,
                        'prompt_tokens': ctx_len,
                        'error': str(exc),
                        'status': 'ERROR'
                    }

                out_f.write(json.dumps(out_rec) + '\\n')
                out_f.flush()

            if d_idx % 5 == 0 or d_idx == len(records):
                print(f'[WORKER-GPU{gpu_id}] Progress: {d_idx}/{len(records)} dialogues | Turns: {turns_evaluated} | Elapsed: {(time.time()-start_bench)/60:.1f}m')

    print(f'[WORKER-GPU{gpu_id}] Done in {(time.time()-start_bench)/60:.2f} mins.')

if __name__ == '__main__':
    main()
'''


def _generate_reasoning_worker_code() -> str:
    """Generates the dual-GPU worker script content for reasoning evaluation."""
    return '''import os
import sys
import json
import re
import time
import argparse
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MAX_CONTEXT = 4096

def parse_thinking_and_answer(text):
    if not text or not text.strip():
        return None, "", False
    t = text.strip()
    if "<think>" in t and "</think>" in t:
        thinking_matches = re.findall(r"<think>(.*?)</think>", t, re.DOTALL)
        thinking = "\\n".join(m.strip() for m in thinking_matches if m.strip())
        final_answer = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
        return (thinking if thinking else None), final_answer, False
    if "</think>" in t and "<think>" not in t:
        parts = t.split("</think>", 1)
        return (parts[0].strip() or None), parts[1].strip(), False
    if "<think>" in t and "</think>" not in t:
        thinking = t.split("<think>", 1)[1].strip()
        return (thinking if thinking else None), "", True
    return None, t, False

def compute_token_f1(prediction, reference):
    thinking, final_answer, is_unfinished = parse_thinking_and_answer(prediction)
    eval_text = final_answer
    def tokenize(text):
        text = re.sub(r"[^\\w\\s]", " ", text.lower())
        return [t for t in text.split() if t]
    pred_tokens = tokenize(eval_text)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "brevity": 1.0, "composite_score": 1.0,
                "zero_tool_violation": False, "has_thinking": thinking is not None,
                "thinking_char_count": len(thinking) if thinking else 0, "is_unfinished_thinking": is_unfinished}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "brevity": 0.0, "composite_score": 0.0,
                "zero_tool_violation": False, "has_thinking": thinking is not None,
                "thinking_char_count": len(thinking) if thinking else 0, "is_unfinished_thinking": is_unfinished}
    pred_counts = {}
    for t in pred_tokens: pred_counts[t] = pred_counts.get(t, 0) + 1
    ref_counts = {}
    for t in ref_tokens: ref_counts[t] = ref_counts.get(t, 0) + 1
    common_tokens = sum(min(count, ref_counts.get(token, 0)) for token, count in pred_counts.items())
    precision = common_tokens / len(pred_tokens) if pred_tokens else 0.0
    recall = common_tokens / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    brevity = min(1.0, len(eval_text) / max(1, len(reference)))
    composite = f1 * brevity
    zero_tool_violation = ("<tool_call>" in prediction)
    if zero_tool_violation:
        composite = 0.0
    return {
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "brevity": round(brevity, 4), "composite_score": round(composite, 4),
        "zero_tool_violation": zero_tool_violation, "has_thinking": thinking is not None,
        "thinking_char_count": len(thinking) if thinking else 0, "is_unfinished_thinking": is_unfinished
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--shard_file", type=str, required=True)
    parser.add_argument("--output_shard_base", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_id", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--modes", type=str, default="no_thinking,thinking")
    args = parser.parse_args()

    gpu_id = args.gpu_id
    eval_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    torch.cuda.set_device(gpu_id)

    with open(args.shard_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[WORKER-GPU{gpu_id}] Initializing reasoning worker for {len(records)} samples (modes={eval_modes})...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=args.hf_token, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and isinstance(im_end_id, int) and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map={"": gpu_id}, torch_dtype=torch.float16, attn_implementation="sdpa", token=args.hf_token, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_id, device_map={"": gpu_id}, token=args.hf_token)
    model.eval()
    torch.cuda.empty_cache()

    for mode in eval_modes:
        is_thinking = (mode == "thinking")
        mode_shard_path = f"{args.output_shard_base}_{mode}.jsonl"
        print(f"[WORKER-GPU{gpu_id}] Starting pass for mode='{mode}' (enable_thinking={is_thinking})...", flush=True)

        with open(mode_shard_path, "w", encoding="utf-8") as out_f:
            for idx, rec in enumerate(records):
                sample_id = rec.get("id", f"sample_{idx}")
                messages = rec.get("messages", [])

                target_idx = None
                for m_i in reversed(range(len(messages))):
                    if messages[m_i].get("role") == "assistant":
                        target_idx = m_i
                        break
                if target_idx is None:
                    continue

                prompt_msgs = messages[:target_idx]
                reference_answer = messages[target_idx].get("content") or ""

                try:
                    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=is_thinking)
                except TypeError:
                    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
                    if not is_thinking and prompt_text.endswith("<|im_start|>assistant\\n"):
                        prompt_text += "<think>\\n\\n</think>\\n\\n"
                except Exception:
                    prompt_text = ""
                    for pm in prompt_msgs:
                        prompt_text += f"<|im_start|>{pm['role']}\\n{pm['content']}<|im_end|>\\n"
                    prompt_text += "<|im_start|>assistant\\n"
                    if not is_thinking:
                        prompt_text += "<think>\\n\\n</think>\\n\\n"

                encoded = tokenizer(prompt_text, return_tensors="pt")
                input_ids = encoded["input_ids"].to(f"cuda:{gpu_id}")
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(f"cuda:{gpu_id}")

                prompt_tokens = input_ids.shape[1]
                max_new = min(1024, MAX_CONTEXT - prompt_tokens)

                gen_kwargs = {
                    "input_ids": input_ids, "attention_mask": attention_mask, "max_new_tokens": max_new,
                    "repetition_penalty": 1.05, "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                    "eos_token_id": eos_ids,
                }
                if is_thinking:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = 0.6
                    gen_kwargs["top_p"] = 0.95
                else:
                    gen_kwargs["do_sample"] = False
                    gen_kwargs["temperature"] = 0.0

                gen_start = time.time()
                try:
                    with torch.inference_mode():
                        out_ids = model.generate(**gen_kwargs)
                    gen_latency = time.time() - gen_start
                    new_tokens = out_ids[0][prompt_tokens:]
                    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    thinking_extracted, final_answer, is_unfin = parse_thinking_and_answer(generated_text)
                    metrics = compute_token_f1(generated_text, reference_answer)

                    row = {
                        "id": sample_id,
                        "mode": mode,
                        "gpu_id": gpu_id,
                        "user_prompt": prompt_msgs[-1]["content"] if prompt_msgs else "",
                        "reference_answer": reference_answer,
                        "generated_text": generated_text,
                        "generated_reasoning": thinking_extracted,
                        "generated_final_answer": final_answer,
                        "metrics": metrics,
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": len(new_tokens),
                        "generation_latency_sec": round(gen_latency, 3),
                        "status": "SUCCESS"
                    }
                except Exception as exc:
                    row = {
                        "id": sample_id,
                        "mode": mode,
                        "gpu_id": gpu_id,
                        "error": str(exc),
                        "status": "ERROR"
                    }

                out_f.write(json.dumps(row) + "\\n")
                out_f.flush()

    print(f"[WORKER-GPU{gpu_id}] Shard completed all modes successfully.", flush=True)

if __name__ == "__main__":
    main()
'''


def build_shard_notebook(
    track: str,
    shard_idx: int,
    total_shards: int,
    records: List[Dict[str, Any]],
    account: str,
    model_id: str = DEFAULT_MODEL_ID,
    base_model: str = DEFAULT_BASE_MODEL
) -> Dict[str, Any]:
    """
    Constructs a complete, valid Jupyter Notebook (nbformat v4) for a given track shard.
    """
    b64_payload = compress_payload(records)
    track_title = track.replace("_", " ").title()

    cells = []

    # Cell 0: Header Markdown
    header_md = f"""# Qwen3-8B v3 {track_title} Benchmark Evaluation — Shard {shard_idx}/{total_shards}

> **Track**: `{track}` ({track_title})  
> **Shard**: {shard_idx} of {total_shards} ({len(records)} Dialogues)  
> **Base Model**: `{base_model}`  
> **Adapter Under Test**: `{model_id}`  
> **Target Hardware**: Kaggle Dual Tesla T4 GPUs  
> **Kaggle Account**: `{account}`  
"""
    cells.append(create_cell("markdown", header_md))

    # Cell 1: Environment & Token Setup
    cell1_source = """import os
import sys
import subprocess
import torch

print(f"[SETUP] Python version: {sys.version}")
print(f"[SETUP] PyTorch version: {torch.__version__}")
print(f"[SETUP] CUDA available: {torch.cuda.is_available()}")
device_count = torch.cuda.device_count()
print(f"[HARDWARE] Detected Available GPUs: {device_count}")
for i in range(device_count):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} | Total VRAM: {props.total_memory / (1024**3):.2f} GB")

# Resolve HF Token safely
hf_token = ""
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    print("[AUTH] Retrieved HF_TOKEN from Kaggle Secrets.")
except Exception:
    pass

token_paths = [
    "/kaggle/input/hf-auth-token/hf_token.txt",
    f"/kaggle/input/datasets/{os.getenv('KAGGLE_AUTH_DATASET', 'hf-auth-token')}/hf_token.txt"
]
for tp in token_paths:
    if not hf_token and os.path.exists(tp):
        try:
            with open(tp, "r", encoding="utf-8") as f:
                hf_token = f.read().strip()
                print(f"[AUTH] Loaded HF_TOKEN from dataset: {tp}")
        except Exception:
            pass

if not hf_token:
    for env_k in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"]:
        val = os.environ.get(env_k, "").strip()
        if val:
            hf_token = val
            break

os.environ["HF_TOKEN"] = hf_token
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure runtime packages
print("[SETUP] Verifying packages bitsandbytes peft accelerate...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "bitsandbytes>=0.46.1", "peft", "accelerate"], check=True)
print("[SETUP] Environment initialization complete [OK].")
"""
    cells.append(create_cell("code", cell1_source))

    # Cell 2: Ingest and Unpack Payload
    cell2_source = """import zlib
import base64
import json

SHARD_INDEX = __SHARD_INDEX__
TOTAL_SHARDS = __TOTAL_SHARDS__
EXPECTED_DIALOGUES = __EXPECTED_COUNT__
B64_PAYLOAD = "__B64_PAYLOAD__"

print(f"[DATA] Decompressing Shard {SHARD_INDEX}/{TOTAL_SHARDS} payload...")
raw_bytes = zlib.decompress(base64.b64decode(B64_PAYLOAD.encode("ascii")))
dialogues = json.loads(raw_bytes.decode("utf-8"))

print(f"[DATA] Ingested {len(dialogues)} dialogues.")
assert len(dialogues) == EXPECTED_DIALOGUES, f"Expected {EXPECTED_DIALOGUES}, got {len(dialogues)}"
""".replace("__SHARD_INDEX__", str(shard_idx))\
   .replace("__TOTAL_SHARDS__", str(total_shards))\
   .replace("__EXPECTED_COUNT__", str(len(records)))\
   .replace("__B64_PAYLOAD__", b64_payload)

    cells.append(create_cell("code", cell2_source))

    # Select Worker and Dispatch Code by Track
    if track == "exploit":
        worker_code = _generate_exploit_worker_code()
        worker_filename = "dual_gpu_exploit_worker.py"
        
        # Cell 3: Write Worker Script
        cell3_source = """# Write Worker Script
worker_code = __WORKER_CODE__
worker_path = "/kaggle/working/__WORKER_FILENAME__"
with open(worker_path, "w", encoding="utf-8") as f:
    f.write(worker_code)
print(f"[SETUP] Worker script written to: {worker_path}")
""".replace("__WORKER_CODE__", repr(worker_code))\
   .replace("__WORKER_FILENAME__", worker_filename)
        cells.append(create_cell("code", cell3_source))

        # Cell 4: Launch Dual GPU Subprocesses
        cell4_source = """import subprocess
import threading
import time
import math

half = math.ceil(len(dialogues) / 2)
shard0_dialogues = dialogues[:half]
shard1_dialogues = dialogues[half:]

shard0_in = "/kaggle/working/shard0_in.jsonl"
shard1_in = "/kaggle/working/shard1_in.jsonl"
shard0_out = "/kaggle/working/shard0_out.jsonl"
shard1_out = "/kaggle/working/shard1_out.jsonl"

with open(shard0_in, "w", encoding="utf-8") as f:
    for d in shard0_dialogues:
        f.write(json.dumps(d) + "\\n")

with open(shard1_in, "w", encoding="utf-8") as f:
    for d in shard1_dialogues:
        f.write(json.dumps(d) + "\\n")

print(f"[PARALLEL] Dispatched {len(shard0_dialogues)} dialogues to GPU 0 and {len(shard1_dialogues)} dialogues to GPU 1.")

cmd_gpu0 = [
    sys.executable, "-u", "/kaggle/working/__WORKER_FILENAME__",
    "--shard_file", shard0_in,
    "--gpu_id", "0",
    "--base_model", "__BASE_MODEL__",
    "--adapter_id", "__MODEL_ID__",
    "--output_file", shard0_out,
    "--hf_token", os.environ.get("HF_TOKEN", "")
]

cmd_gpu1 = [
    sys.executable, "-u", "/kaggle/working/__WORKER_FILENAME__",
    "--shard_file", shard1_in,
    "--gpu_id", "1",
    "--base_model", "__BASE_MODEL__",
    "--adapter_id", "__MODEL_ID__",
    "--output_file", shard1_out,
    "--hf_token", os.environ.get("HF_TOKEN", "")
]

def stream_logs(pipe, prefix):
    for line in iter(pipe.readline, ''):
        print(f"[{prefix}] {line.strip()}", flush=True)
    pipe.close()

start_time = time.time()
p0 = subprocess.Popen(cmd_gpu0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
p1 = subprocess.Popen(cmd_gpu1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

t0 = threading.Thread(target=stream_logs, args=(p0.stdout, "GPU-0"))
t1 = threading.Thread(target=stream_logs, args=(p1.stdout, "GPU-1"))
t0.start()
t1.start()

ret0 = p0.wait()
ret1 = p1.wait()
t0.join()
t1.join()

assert ret0 == 0, f"Worker GPU 0 failed with exit code {ret0}"
assert ret1 == 0, f"Worker GPU 1 failed with exit code {ret1}"
print(f"[PARALLEL] Both GPU workers finished cleanly in {(time.time() - start_time) / 60.0:.2f} minutes!")
""".replace("__WORKER_FILENAME__", worker_filename)\
   .replace("__BASE_MODEL__", base_model)\
   .replace("__MODEL_ID__", model_id)
        cells.append(create_cell("code", cell4_source))

        # Cell 5: Reconciliation & Summary
        cell5_source = """import os
import json

SHARD_INDEX = __SHARD_INDEX__
OUTPUT_DIR = f"/kaggle/working/exploit_shard_{SHARD_INDEX}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

final_pred_file = f"{OUTPUT_DIR}/exploit_predictions.jsonl"
final_summary_file = f"{OUTPUT_DIR}/exploit_summary.json"

all_results = []
for cand in ["/kaggle/working/shard0_out.jsonl", "/kaggle/working/shard1_out.jsonl"]:
    if os.path.exists(cand):
        with open(cand, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

with open(final_pred_file, "w", encoding="utf-8") as f:
    for r in all_results:
        f.write(json.dumps(r) + "\\n")

def compute_exploit_metrics(records):
    valid = [r for r in records if r.get("status") == "SUCCESS"]
    errs = sum(1 for r in records if r.get("status") == "EVALUATION_FAILURE")
    tp = sum(1 for r in valid if r.get("metrics", {}).get("confusion") == "TP")
    fp = sum(1 for r in valid if r.get("metrics", {}).get("confusion") == "FP")
    fn = sum(1 for r in valid if r.get("metrics", {}).get("confusion") == "FN")
    tn = sum(1 for r in valid if r.get("metrics", {}).get("confusion") == "TN")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    action_match = sum(1 for r in valid if r.get("metrics", {}).get("tool_action_match") is True)
    action_match_rate = action_match / len(valid) if valid else 0.0
    tool_req = [r for r in valid if r.get("metrics", {}).get("confusion") in ["TP", "FN"]]
    name_match = sum(1 for r in tool_req if r.get("metrics", {}).get("tool_name_match") is True)
    name_match_rate = name_match / len(tool_req) if tool_req else 0.0
    emitted = [r for r in valid if r.get("metrics", {}).get("confusion") in ["TP", "FP"]]
    valid_schema = sum(1 for r in emitted if r.get("metrics", {}).get("args_valid_json") is True)
    schema_rate = valid_schema / len(emitted) if emitted else 1.0
    syntax_valid = sum(1 for r in emitted if r.get("metrics", {}).get("syntax_valid") is True)
    syntax_rate = syntax_valid / len(emitted) if emitted else 1.0
    mean_cf1 = sum(r.get("metrics", {}).get("content_f1", 0.0) for r in valid) / len(valid) if valid else 0.0
    return {
        "turns_count": len(records),
        "valid_turns_count": len(valid),
        "evaluation_errors": errs,
        "tool_decision_precision": round(prec, 4),
        "tool_decision_recall": round(rec, 4),
        "tool_decision_f1": round(f1, 4),
        "action_concordance_rate": round(action_match_rate, 4),
        "tool_name_accuracy": round(name_match_rate, 4),
        "schema_validity_rate": round(schema_rate, 4),
        "static_command_syntax_validity_rate": round(syntax_rate, 4),
        "mean_content_f1": round(mean_cf1, 4)
    }

overall = compute_exploit_metrics(all_results)
summary = {
    "shard_index": SHARD_INDEX,
    "total_shards": __TOTAL_SHARDS__,
    "base_model": "__BASE_MODEL__",
    "adapter_repo": "__MODEL_ID__",
    "total_dialogues": len(dialogues),
    "total_turns_evaluated": len(all_results),
    "overall_metrics": overall
}

with open(final_summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"[SUMMARY] Shard {SHARD_INDEX} completed: {overall['valid_turns_count']} valid turns evaluated.")
print(f"  Tool Decision F1: {overall['tool_decision_f1']} | Tool Name Accuracy: {overall['tool_name_accuracy']}")
print(f"  Command Syntax Validity: {overall['static_command_syntax_validity_rate']} | Schema Validity: {overall['schema_validity_rate']}")
""".replace("__SHARD_INDEX__", str(shard_idx))\
   .replace("__TOTAL_SHARDS__", str(total_shards))\
   .replace("__BASE_MODEL__", base_model)\
   .replace("__MODEL_ID__", model_id)
        cells.append(create_cell("code", cell5_source))

    elif track == "tool_replay":
        worker_code = _generate_tool_replay_worker_code()
        worker_filename = "dual_gpu_replay_worker.py"

        # Cell 3: Write Worker Script
        cell3_source = """# Write Worker Script
worker_code = __WORKER_CODE__
worker_path = "/kaggle/working/__WORKER_FILENAME__"
with open(worker_path, "w", encoding="utf-8") as f:
    f.write(worker_code)
print(f"[SETUP] Worker script written to: {worker_path}")
""".replace("__WORKER_CODE__", repr(worker_code))\
   .replace("__WORKER_FILENAME__", worker_filename)
        cells.append(create_cell("code", cell3_source))

        # Cell 4: Launch Dual GPU Subprocesses
        cell4_source = """import subprocess
import threading
import time
import math

half = math.ceil(len(dialogues) / 2)
shard0_dialogues = dialogues[:half]
shard1_dialogues = dialogues[half:]

shard0_in = "/kaggle/working/shard0_in.jsonl"
shard1_in = "/kaggle/working/shard1_in.jsonl"
shard0_out = "/kaggle/working/shard0_out.jsonl"
shard1_out = "/kaggle/working/shard1_out.jsonl"

with open(shard0_in, "w", encoding="utf-8") as f:
    for d in shard0_dialogues:
        f.write(json.dumps(d) + "\\n")

with open(shard1_in, "w", encoding="utf-8") as f:
    for d in shard1_dialogues:
        f.write(json.dumps(d) + "\\n")

print(f"[PARALLEL] Dispatched {len(shard0_dialogues)} dialogues to GPU 0 and {len(shard1_dialogues)} dialogues to GPU 1.")

cmd_gpu0 = [
    sys.executable, "-u", "/kaggle/working/__WORKER_FILENAME__",
    "--shard_file", shard0_in,
    "--gpu_id", "0",
    "--base_model", "__BASE_MODEL__",
    "--adapter_id", "__MODEL_ID__",
    "--output_file", shard0_out,
    "--hf_token", os.environ.get("HF_TOKEN", "")
]

cmd_gpu1 = [
    sys.executable, "-u", "/kaggle/working/__WORKER_FILENAME__",
    "--shard_file", shard1_in,
    "--gpu_id", "1",
    "--base_model", "__BASE_MODEL__",
    "--adapter_id", "__MODEL_ID__",
    "--output_file", shard1_out,
    "--hf_token", os.environ.get("HF_TOKEN", "")
]

def stream_logs(pipe, prefix):
    for line in iter(pipe.readline, ''):
        print(f"[{prefix}] {line.strip()}", flush=True)
    pipe.close()

start_time = time.time()
p0 = subprocess.Popen(cmd_gpu0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
p1 = subprocess.Popen(cmd_gpu1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

t0 = threading.Thread(target=stream_logs, args=(p0.stdout, "GPU-0"))
t1 = threading.Thread(target=stream_logs, args=(p1.stdout, "GPU-1"))
t0.start()
t1.start()

ret0 = p0.wait()
ret1 = p1.wait()
t0.join()
t1.join()

assert ret0 == 0, f"Worker GPU 0 failed with exit code {ret0}"
assert ret1 == 0, f"Worker GPU 1 failed with exit code {ret1}"
print(f"[PARALLEL] Both GPU workers finished cleanly in {(time.time() - start_time) / 60.0:.2f} minutes!")
""".replace("__WORKER_FILENAME__", worker_filename)\
   .replace("__BASE_MODEL__", base_model)\
   .replace("__MODEL_ID__", model_id)
        cells.append(create_cell("code", cell4_source))

        # Cell 5: Reconciliation & Summary
        cell5_source = """import os
import json

SHARD_INDEX = __SHARD_INDEX__
OUTPUT_DIR = f"/kaggle/working/tool_replay_shard_{SHARD_INDEX}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

final_pred_file = f"{OUTPUT_DIR}/tool_replay_predictions.jsonl"
final_summary_file = f"{OUTPUT_DIR}/tool_replay_summary.json"

all_results = []
for cand in ["/kaggle/working/shard0_out.jsonl", "/kaggle/working/shard1_out.jsonl"]:
    if os.path.exists(cand):
        with open(cand, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

with open(final_pred_file, "w", encoding="utf-8") as f:
    for r in all_results:
        f.write(json.dumps(r) + "\\n")

def compute_replay_metrics(records):
    valid = [r for r in records if r.get("status") == "SUCCESS"]
    errs = sum(1 for r in records if r.get("status") != "SUCCESS")
    total_turns = len(valid)
    if total_turns == 0:
        return {"total_turns": 0, "evaluation_errors": errs}

    tool_req_turns = [r for r in valid if r.get("target_tool_calls")]
    req_cnt = len(tool_req_turns)

    name_matches = sum(1 for r in tool_req_turns if r.get("tool_metrics", {}).get("tool_name_match"))
    arg_exact = sum(1 for r in tool_req_turns if r.get("tool_metrics", {}).get("arg_exact_match"))
    arg_subset = sum(1 for r in tool_req_turns if r.get("tool_metrics", {}).get("arg_subset_match"))

    mean_f1 = sum(r.get("content_metrics", {}).get("f1", 0.0) for r in valid) / total_turns

    return {
        "total_turns": total_turns,
        "evaluation_errors": errs,
        "tool_calling_turns": req_cnt,
        "tool_name_accuracy": round(name_matches / max(1, req_cnt), 4),
        "arg_exact_match_accuracy": round(arg_exact / max(1, req_cnt), 4),
        "arg_subset_match_accuracy": round(arg_subset / max(1, req_cnt), 4),
        "mean_content_f1": round(mean_f1, 4)
    }

overall = compute_replay_metrics(all_results)
summary = {
    "shard_index": SHARD_INDEX,
    "total_shards": __TOTAL_SHARDS__,
    "base_model": "__BASE_MODEL__",
    "adapter_repo": "__MODEL_ID__",
    "total_dialogues": len(dialogues),
    "total_turns_evaluated": len(all_results),
    "overall_metrics": overall
}

with open(final_summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"[SUMMARY] Tool Replay Shard {SHARD_INDEX} completed: {overall.get('total_turns', 0)} turns.")
print(f"  Tool Name Accuracy: {overall.get('tool_name_accuracy', 0.0)} | Arg Exact Match: {overall.get('arg_exact_match_accuracy', 0.0)}")
print(f"  Mean Content F1: {overall.get('mean_content_f1', 0.0)}")
""".replace("__SHARD_INDEX__", str(shard_idx))\
   .replace("__TOTAL_SHARDS__", str(total_shards))\
   .replace("__BASE_MODEL__", base_model)\
   .replace("__MODEL_ID__", model_id)
        cells.append(create_cell("code", cell5_source))

    elif track == "reasoning":
        worker_code = _generate_reasoning_worker_code()
        worker_filename = "dual_gpu_reasoning_worker.py"

        # Cell 3: Write Worker Script
        cell3_source = """# Write Worker Script
worker_code = __WORKER_CODE__
worker_path = "/kaggle/working/__WORKER_FILENAME__"
with open(worker_path, "w", encoding="utf-8") as f:
    f.write(worker_code)
print(f"[SETUP] Worker script written to: {worker_path}")
""".replace("__WORKER_CODE__", repr(worker_code))\
   .replace("__WORKER_FILENAME__", worker_filename)
        cells.append(create_cell("code", cell3_source))

        # Cell 4: Launch Dual GPU Subprocesses
        cell4_source = """import subprocess
import threading
import time
import math

half = math.ceil(len(dialogues) / 2)
shard0_dialogues = dialogues[:half]
shard1_dialogues = dialogues[half:]

shard0_in = "/kaggle/working/shard0_in.jsonl"
shard1_in = "/kaggle/working/shard1_in.jsonl"
shard0_base = "/kaggle/working/shard0_out"
shard1_base = "/kaggle/working/shard1_out"

with open(shard0_in, "w", encoding="utf-8") as f:
    for d in shard0_dialogues:
        f.write(json.dumps(d) + "\\n")

with open(shard1_in, "w", encoding="utf-8") as f:
    for d in shard1_dialogues:
        f.write(json.dumps(d) + "\\n")

print(f"[PARALLEL] Dispatched {len(shard0_dialogues)} dialogues to GPU 0 and {len(shard1_dialogues)} dialogues to GPU 1.")

cmd_gpu0 = [
    sys.executable, "-u", "/kaggle/working/__WORKER_FILENAME__",
    "--shard_file", shard0_in,
    "--gpu_id", "0",
    "--base_model", "__BASE_MODEL__",
    "--adapter_id", "__MODEL_ID__",
    "--output_shard_base", shard0_base,
    "--hf_token", os.environ.get("HF_TOKEN", "")
]

cmd_gpu1 = [
    sys.executable, "-u", "/kaggle/working/__WORKER_FILENAME__",
    "--shard_file", shard1_in,
    "--gpu_id", "1",
    "--base_model", "__BASE_MODEL__",
    "--adapter_id", "__MODEL_ID__",
    "--output_shard_base", shard1_base,
    "--hf_token", os.environ.get("HF_TOKEN", "")
]

def stream_logs(pipe, prefix):
    for line in iter(pipe.readline, ''):
        print(f"[{prefix}] {line.strip()}", flush=True)
    pipe.close()

start_time = time.time()
p0 = subprocess.Popen(cmd_gpu0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
p1 = subprocess.Popen(cmd_gpu1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

t0 = threading.Thread(target=stream_logs, args=(p0.stdout, "GPU-0"))
t1 = threading.Thread(target=stream_logs, args=(p1.stdout, "GPU-1"))
t0.start()
t1.start()

ret0 = p0.wait()
ret1 = p1.wait()
t0.join()
t1.join()

assert ret0 == 0, f"Worker GPU 0 failed with exit code {ret0}"
assert ret1 == 0, f"Worker GPU 1 failed with exit code {ret1}"
print(f"[PARALLEL] Both GPU workers finished cleanly in {(time.time() - start_time) / 60.0:.2f} minutes!")
""".replace("__WORKER_FILENAME__", worker_filename)\
   .replace("__BASE_MODEL__", base_model)\
   .replace("__MODEL_ID__", model_id)
        cells.append(create_cell("code", cell4_source))

        # Cell 5: Reconciliation & Summary
        cell5_source = """import os
import json
import datetime

SHARD_INDEX = __SHARD_INDEX__
OUTPUT_DIR = f"/kaggle/working/reasoning_shard_{SHARD_INDEX}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EVAL_MODES = ["no_thinking", "thinking"]
mode_summaries = {}
mode_results = {}

for mode in EVAL_MODES:
    results = []
    for cand in [f"/kaggle/working/shard0_out_{mode}.jsonl", f"/kaggle/working/shard1_out_{mode}.jsonl"]:
        if os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
    mode_results[mode] = results

    pred_out_path = os.path.join(OUTPUT_DIR, f"reasoning_predictions_{mode}.jsonl")
    with open(pred_out_path, "w", encoding="utf-8") as out_f:
        for r in results:
            out_f.write(json.dumps(r) + "\\n")

    valid = [r for r in results if "metrics" in r]
    if valid:
        m_f1 = sum(r["metrics"]["f1"] for r in valid) / len(valid)
        m_p = sum(r["metrics"]["precision"] for r in valid) / len(valid)
        m_r = sum(r["metrics"]["recall"] for r in valid) / len(valid)
        m_comp = sum(r["metrics"]["composite_score"] for r in valid) / len(valid)
        violations = sum(1 for r in valid if r["metrics"]["zero_tool_violation"] is True)
        compliance = ((len(valid) - violations) / len(valid)) * 100.0
        avg_th = sum(r["metrics"]["thinking_char_count"] for r in valid) / len(valid)
        avg_lat = sum(r.get("generation_latency_sec", 0.0) for r in valid) / len(valid)
    else:
        m_f1 = m_p = m_r = m_comp = compliance = avg_th = avg_lat = 0.0
        violations = 0

    mode_summaries[mode] = {
        "mean_token_f1": round(m_f1, 4),
        "mean_token_precision": round(m_p, 4),
        "mean_token_recall": round(m_r, 4),
        "composite_reasoning_score": round(m_comp, 4),
        "zero_tool_compliance_rate_percent": round(compliance, 2),
        "zero_tool_violations_count": violations,
        "avg_thinking_chars": round(avg_th, 1),
        "avg_latency_sec": round(avg_lat, 2),
    }

nt = mode_summaries["no_thinking"]
th = mode_summaries["thinking"]
delta = {
    "f1_delta": round(th["mean_token_f1"] - nt["mean_token_f1"], 4),
    "precision_delta": round(th["mean_token_precision"] - nt["mean_token_precision"], 4),
    "recall_delta": round(th["mean_token_recall"] - nt["mean_token_recall"], 4),
    "composite_delta": round(th["composite_reasoning_score"] - nt["composite_reasoning_score"], 4),
    "latency_delta_sec": round(th["avg_latency_sec"] - nt["avg_latency_sec"], 2),
}

summary = {
    "shard_index": SHARD_INDEX,
    "total_shards": __TOTAL_SHARDS__,
    "model_id": "__MODEL_ID__",
    "base_model": "__BASE_MODEL__",
    "total_dialogues": len(dialogues),
    "no_thinking": nt,
    "thinking": th,
    "delta": delta,
    "timestamp": datetime.datetime.now().isoformat()
}

with open(os.path.join(OUTPUT_DIR, "reasoning_comparative_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "reasoning_summary_v3.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"[SUMMARY] Reasoning Shard {SHARD_INDEX} completed.")
print(f"  Direct Composite: {nt['composite_reasoning_score']} | Deep Reasoning Composite: {th['composite_reasoning_score']}")
print(f"  Delta: {delta['composite_delta']:+.4f}")
""".replace("__SHARD_INDEX__", str(shard_idx))\
   .replace("__TOTAL_SHARDS__", str(total_shards))\
   .replace("__BASE_MODEL__", base_model)\
   .replace("__MODEL_ID__", model_id)
        cells.append(create_cell("code", cell5_source))

    else:
        raise ValueError(f"Unsupported track: '{track}'")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return notebook


def build_kernel_metadata(
    shard_idx: int,
    total_shards: int,
    track: str,
    account: str,
    notebook_filename: str
) -> Dict[str, Any]:
    """Generates a valid Kaggle kernel-metadata.json configuration."""
    track_slug_part = track.replace("_", "-")
    track_title = track.replace("_", " ").title()

    slug = f"{account}/qwen3-8b-v3-{track_slug_part}-eval-shard-{shard_idx}"
    title = f"Qwen3 8B v3 {track_title} Eval Shard {shard_idx}"

    metadata = {
        "id": slug,
        "title": title,
        "code_file": notebook_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_gpu": True,
        "enable_gpu": True,
        "enable_internet": True,
        "accelerator": "nvidiaTeslaT4Dual",
        "dataset_sources": [os.getenv("KAGGLE_AUTH_DATASET")] if os.getenv("KAGGLE_AUTH_DATASET") else [],
        "competition_sources": [],
        "kernel_sources": []
    }
    return metadata


class KaggleShardEngine:
    """
    Centralized coordinator for partitioning test holdouts and packaging Kaggle deployment bundles.
    """

    def __init__(self, config: Optional[KaggleShardConfig] = None):
        self.config = config or KaggleShardConfig(track="exploit")

    def load_dialogues(self, track: str, custom_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Loads and validates dialogues for the requested track."""
        input_path = custom_path or DEFAULT_INPUT_PATHS.get(track)
        if not input_path:
            raise ValueError(f"No default input path configured for track: '{track}'")
        
        if not os.path.isabs(input_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            full_path = os.path.join(base_dir, input_path)
            if not os.path.exists(full_path):
                # Try relative to cwd
                full_path = os.path.abspath(input_path)
        else:
            full_path = input_path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Holdout dataset not found at: {full_path}")

        dialogues = []
        with open(full_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    dialogues.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"Corrupt JSON line at {full_path}:{line_num}: {e}")

        print(f"[DATA] Loaded {len(dialogues)} dialogues for track '{track}' from {full_path}")
        return dialogues

    def generate_track_package(
        self,
        track: str,
        dialogues: List[Dict[str, Any]],
        num_shards: int,
        accounts: List[str],
        output_dir: str,
        model_id: str = DEFAULT_MODEL_ID,
        base_model: str = DEFAULT_BASE_MODEL,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Executes partition and creates shard packages for a specific track.
        """
        print(f"\n[ENGINE] Generating {num_shards}-way sharded package for track '{track}'...")
        shards = partition_dialogues(dialogues, num_shards)

        shard_summaries = []

        for idx, (shard_dialogues, account) in enumerate(zip(shards, accounts), 1):
            shard_folder_name = f"shard_{idx}"
            shard_dir = os.path.join(output_dir, shard_folder_name)
            nb_filename = f"qwen3_8b_v3_{track}_eval_shard_{idx}.ipynb"
            nb_path = os.path.join(shard_dir, nb_filename)
            meta_path = os.path.join(shard_dir, "kernel-metadata.json")
            raw_shard_filename = f"{track}_shard_{idx}.jsonl"
            raw_shard_path = os.path.join(shard_dir, raw_shard_filename)

            notebook_obj = build_shard_notebook(
                track=track,
                shard_idx=idx,
                total_shards=num_shards,
                records=shard_dialogues,
                account=account,
                model_id=model_id,
                base_model=base_model
            )

            metadata_obj = build_kernel_metadata(
                shard_idx=idx,
                total_shards=num_shards,
                track=track,
                account=account,
                notebook_filename=nb_filename
            )

            if not dry_run:
                os.makedirs(shard_dir, exist_ok=True)

                # 1. Write notebook
                with open(nb_path, "w", encoding="utf-8") as f:
                    json.dump(notebook_obj, f, indent=2)

                # 2. Write metadata
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata_obj, f, indent=2)

                # 3. Write raw split file
                with open(raw_shard_path, "w", encoding="utf-8") as f:
                    for d in shard_dialogues:
                        f.write(json.dumps(d) + "\n")

            turns_count = sum(sum(1 for m in d.get("messages", []) if m.get("role") == "assistant") for d in shard_dialogues)

            shard_summary = {
                "shard_index": idx,
                "account": account,
                "dialogue_count": len(shard_dialogues),
                "assistant_turns_count": turns_count,
                "notebook_path": nb_path,
                "metadata_path": meta_path,
                "raw_shard_path": raw_shard_path,
                "kaggle_slug": metadata_obj["id"]
            }
            shard_summaries.append(shard_summary)
            print(f"  Shard {idx}/{num_shards}: Account='{account}' | {len(shard_dialogues)} dialogues | {turns_count} turns -> {shard_dir}")

        manifest = {
            "track": track,
            "total_dialogues": len(dialogues),
            "num_shards": num_shards,
            "model_id": model_id,
            "base_model": base_model,
            "output_dir": output_dir,
            "dry_run": dry_run,
            "shards": shard_summaries
        }

        if not dry_run:
            manifest_path = os.path.join(output_dir, "sharding_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"[ENGINE] Sharding manifest saved to: {manifest_path}")

        return manifest

    def run(self, config: Optional[KaggleShardConfig] = None) -> Dict[str, Any]:
        """
        Main execution method for the sharding engine.
        """
        cfg = config or self.config

        num_shards, accounts = resolve_accounts(
            num_shards=cfg.num_shards,
            accounts=cfg.accounts,
            from_env=cfg.from_env
        )

        tracks_to_process = (
            ["exploit", "tool_replay", "reasoning"]
            if cfg.track == "all"
            else [cfg.track]
        )

        results = {}

        for track in tracks_to_process:
            dialogues = self.load_dialogues(track, cfg.input_data)
            
            if cfg.track == "all":
                track_out_dir = os.path.join(cfg.output_dir, track)
            else:
                track_out_dir = cfg.output_dir

            manifest = self.generate_track_package(
                track=track,
                dialogues=dialogues,
                num_shards=num_shards,
                accounts=accounts,
                output_dir=track_out_dir,
                model_id=cfg.model_id,
                base_model=cfg.base_model,
                dry_run=cfg.dry_run
            )
            results[track] = manifest

        print("\n[SUCCESS] Kaggle Sharding Engine execution completed successfully!")
        return results


def parse_args(args: Optional[List[str]] = None) -> KaggleShardConfig:
    parser = argparse.ArgumentParser(
        description="Dynamic Kaggle Sharding Engine for Qwen3-8B 3-Track Evaluation Suite."
    )
    parser.add_argument(
        "--track",
        type=str,
        choices=["exploit", "tool_replay", "reasoning", "all"],
        default="exploit",
        help="Target evaluation track to partition and generate notebooks for."
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of distributed shards (e.g. 3, 5). Auto-derived if omitted or if --from-env is set."
    )
    parser.add_argument(
        "--accounts",
        type=str,
        default=None,
        help="Comma-separated Kaggle account usernames (e.g. 'acc1,acc2,acc3')."
    )
    parser.add_argument(
        "--from-env",
        action="store_true",
        help="Auto-derive shard count and accounts from KAGGLE_ACCOUNTS environment variable."
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default=None,
        help="Custom path to clean test dialogue JSONL file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="kaggle_deploys",
        help="Output root directory where generated shard folders will be saved."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face repo ID of fine-tuned adapter."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base foundation model repository."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform partitioning and preview without writing files to disk."
    )

    parsed = parser.parse_args(args)
    acc_list = [a.strip() for a in parsed.accounts.split(",") if a.strip()] if parsed.accounts else None

    return KaggleShardConfig(
        track=parsed.track,
        num_shards=parsed.num_shards,
        accounts=acc_list,
        from_env=parsed.from_env,
        input_data=parsed.input_data,
        output_dir=parsed.output_dir,
        model_id=parsed.model_id,
        base_model=parsed.base_model,
        dry_run=parsed.dry_run
    )


def main():
    cfg = parse_args()
    engine = KaggleShardEngine(cfg)
    engine.run()


if __name__ == "__main__":
    main()
