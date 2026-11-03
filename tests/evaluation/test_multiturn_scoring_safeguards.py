"""
Unit tests for Multi-Turn Tool Replay Safeguards & Correctness Audits
====================================================================
Tests the 3 critical audit findings:
1. Multi-tool & argument-value matching:
   - When target has 2 tool calls and prediction has 1 with wrong args, evaluator must NOT return 1.0.
   - Evaluates all calls and checks argument values, not just keys.
2. Error accounting in metrics:
   - Evaluated failures on tool-required turns must NOT silently disappear from recall denominators.
3. Thinking mode & atomic context pruning:
   - Context pruning must NEVER separate a tool response from its preceding assistant call.
   - No orphaned tool messages at the boundary.
"""

import json
import re
import pytest
from typing import Dict, Any, List

# Define the functions as they will be implemented in the multi-turn replay harness

def extract_replay_tool_calls(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    tool_calls = []
    pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
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
                parsed['arguments'] = args if isinstance(args, dict) else {}
                tool_calls.append(parsed)
        except Exception:
            continue
    return tool_calls

def compute_token_f1(prediction: str, reference: str) -> Dict[str, float]:
    p_toks = re.findall(r'\w+|[^\w\s]', prediction.lower())
    r_toks = re.findall(r'\w+|[^\w\s]', reference.lower())
    if not p_toks and not r_toks:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    if not p_toks or not r_toks:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    p_cnt = {}
    for t in p_toks: p_cnt[t] = p_cnt.get(t, 0) + 1
    r_cnt = {}
    for t in r_toks: r_cnt[t] = r_cnt.get(t, 0) + 1
    comm = sum(min(c, r_cnt.get(t, 0)) for t, c in p_cnt.items())
    prec = comm / max(1, len(p_toks))
    rec = comm / max(1, len(r_toks))
    f1 = (2 * prec * rec) / max(1e-8, prec + rec)
    return {'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4)}

def evaluate_replay_turn(pred_text: str, target_message: Dict[str, Any]) -> Dict[str, Any]:
    target_tcs = target_message.get('tool_calls', [])
    target_has_tool = len(target_tcs) > 0
    is_no_tool_turn = not target_has_tool

    if pred_text is None:
        return {
            'confusion': 'ERROR',
            'is_no_tool_turn': is_no_tool_turn,
            'tool_action_match': False,
            'tool_count_target': len(target_tcs),
            'tool_count_pred': 0,
            'tool_name_match': False if target_has_tool else None,
            'tool_call_precision': 0.0 if target_has_tool else 1.0,
            'tool_call_recall': 0.0 if target_has_tool else 1.0,
            'tool_call_f1': 0.0 if target_has_tool else 1.0,
            'args_valid_json': None,
            'arg_key_precision': None,
            'arg_key_recall': None,
            'arg_key_f1': None,
            'arg_value_accuracy': None,
            'full_tool_call_accuracy': 0.0 if target_has_tool else 1.0,
            'content_f1': 0.0,
            'content_precision': 0.0,
            'content_recall': 0.0
        }

    pred_tcs = extract_replay_tool_calls(pred_text)
    pred_has_tool = len(pred_tcs) > 0

    tool_action_match = (target_has_tool == pred_has_tool)

    if target_has_tool and pred_has_tool:
        confusion = 'TP'
    elif not target_has_tool and pred_has_tool:
        confusion = 'FP'
    elif target_has_tool and not pred_has_tool:
        confusion = 'FN'
    else:
        confusion = 'TN'

    # Normalize targets
    norm_targets = []
    for tc in target_tcs:
        fn = tc.get('function', {}) if 'function' in tc else tc
        name = fn.get('name', '').strip().lower()
        args = fn.get('arguments', {})
        if isinstance(args, str):
            try: args = json.loads(args)
            except Exception: args = {}
        norm_targets.append((name, args if isinstance(args, dict) else {}))

    # Normalize predictions
    norm_preds = []
    args_all_valid_json = True
    for ptc in pred_tcs:
        name = ptc.get('name', '').strip().lower()
        args = ptc.get('arguments', {})
        if isinstance(args, str):
            try: args = json.loads(args)
            except Exception:
                args_all_valid_json = False
                args = {}
        elif not isinstance(args, dict):
            args_all_valid_json = False
            args = {}
        norm_preds.append((name, args))

    # Evaluate multiple tool calls via greedy bipartite matching
    matched_targets = set()
    call_matches = 0
    exact_matches = 0
    key_prec_list, key_rec_list, val_acc_list = [], [], []

    for p_name, p_args in norm_preds:
        best_t_idx = None
        for t_idx, (t_name, t_args) in enumerate(norm_targets):
            if t_idx in matched_targets:
                continue
            if p_name == t_name:
                best_t_idx = t_idx
                break
        if best_t_idx is not None:
            matched_targets.add(best_t_idx)
            call_matches += 1
            t_name, t_args = norm_targets[best_t_idx]

            # Compare argument keys
            p_keys = set(p_args.keys())
            t_keys = set(t_args.keys())
            if p_keys and t_keys:
                comm_k = p_keys & t_keys
                k_p = len(comm_k) / len(p_keys)
                k_r = len(comm_k) / len(t_keys)
                key_prec_list.append(k_p)
                key_rec_list.append(k_r)
            elif not p_keys and not t_keys:
                key_prec_list.append(1.0)
                key_rec_list.append(1.0)
            else:
                key_prec_list.append(0.0)
                key_rec_list.append(0.0)

            # Compare argument values
            if t_keys:
                matching_values = 0
                for k in t_keys:
                    p_val = p_args.get(k)
                    t_val = t_args.get(k)
                    if p_val == t_val or str(p_val).strip().lower() == str(t_val).strip().lower():
                        matching_values += 1
                v_acc = matching_values / len(t_keys)
                val_acc_list.append(v_acc)
                if p_keys == t_keys and v_acc == 1.0:
                    exact_matches += 1
            else:
                if not p_keys:
                    val_acc_list.append(1.0)
                    exact_matches += 1
                else:
                    val_acc_list.append(0.0)

    if target_has_tool:
        tc_prec = call_matches / len(norm_preds) if norm_preds else 0.0
        tc_rec = call_matches / len(norm_targets)
        tc_f1 = (2 * tc_prec * tc_rec) / (tc_prec + tc_rec) if (tc_prec + tc_rec) > 0 else 0.0
        tool_name_match = (call_matches == len(norm_targets) and len(norm_preds) == len(norm_targets))

        avg_k_p = sum(key_prec_list) / len(norm_targets) if norm_targets else 0.0
        avg_k_r = sum(key_rec_list) / len(norm_targets) if norm_targets else 0.0
        avg_k_f1 = (2 * avg_k_p * avg_k_r) / (avg_k_p + avg_k_r) if (avg_k_p + avg_k_r) > 0 else 0.0
        avg_v_acc = sum(val_acc_list) / len(norm_targets) if norm_targets else 0.0
        full_accuracy = exact_matches / len(norm_targets)
    else:
        if pred_has_tool:
            tc_prec, tc_rec, tc_f1 = 0.0, 0.0, 0.0
            tool_name_match = False
            avg_k_p, avg_k_r, avg_k_f1, avg_v_acc, full_accuracy = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            tc_prec, tc_rec, tc_f1 = 1.0, 1.0, 1.0
            tool_name_match = None
            avg_k_p, avg_k_r, avg_k_f1, avg_v_acc, full_accuracy = None, None, None, None, 1.0

    # Reasoning / text content evaluation
    ref_content = target_message.get('content') or ''
    clean_pred = re.sub(r'<think>.*?</think>', '', pred_text, flags=re.DOTALL)
    clean_pred = re.sub(r'<tool_call>.*?</tool_call>', '', clean_pred, flags=re.DOTALL).strip()
    f1_scores = compute_token_f1(clean_pred, ref_content)

    return {
        'confusion': confusion,
        'is_no_tool_turn': is_no_tool_turn,
        'tool_action_match': tool_action_match,
        'tool_count_target': len(target_tcs),
        'tool_count_pred': len(pred_tcs),
        'tool_name_match': tool_name_match,
        'tool_call_precision': round(tc_prec, 4),
        'tool_call_recall': round(tc_rec, 4),
        'tool_call_f1': round(tc_f1, 4),
        'args_valid_json': args_all_valid_json if pred_has_tool else None,
        'arg_key_precision': round(avg_k_p, 4) if avg_k_p is not None else None,
        'arg_key_recall': round(avg_k_r, 4) if avg_k_r is not None else None,
        'arg_key_f1': round(avg_k_f1, 4) if avg_k_f1 is not None else None,
        'arg_value_accuracy': round(avg_v_acc, 4) if avg_v_acc is not None else None,
        'full_tool_call_accuracy': round(full_accuracy, 4),
        'content_f1': f1_scores['f1'],
        'content_precision': f1_scores['precision'],
        'content_recall': f1_scores['recall']
    }

def group_into_transaction_blocks(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Groups messages into atomic transaction blocks so tool responses are never separated from assistant calls."""
    blocks = []
    curr_block = []
    for msg in messages:
        role = msg.get('role')
        if role in ['user', 'system']:
            if curr_block:
                blocks.append(curr_block)
                curr_block = []
            curr_block.append(msg)
        elif role == 'assistant':
            if curr_block and curr_block[0].get('role') != 'assistant':
                blocks.append(curr_block)
                curr_block = []
            curr_block.append(msg)
        elif role in ['tool', 'tool_response', 'function']:
            # Tool responses MUST belong to the preceding assistant call block
            curr_block.append(msg)
        else:
            if curr_block:
                blocks.append(curr_block)
                curr_block = []
            curr_block.append(msg)
    if curr_block:
        blocks.append(curr_block)
    return blocks

def prune_context_atomic(context: List[Dict[str, Any]], max_messages: int = 4) -> List[Dict[str, Any]]:
    """Prunes context by dropping whole transaction blocks, ensuring tool messages are never orphaned."""
    if len(context) <= max_messages:
        return list(context)

    has_system = len(context) > 0 and context[0].get('role') == 'system'
    system_part = [context[0]] if has_system else []
    conv_part = context[1:] if has_system else context

    blocks = group_into_transaction_blocks(conv_part)
    while len(blocks) > 1 and sum(len(b) for b in blocks) + len(system_part) > max_messages:
        blocks.pop(0)

    result = list(system_part)
    for b in blocks:
        result.extend(b)

    # Sanity check: Ensure the first message after system is NOT a tool response
    check_idx = 1 if has_system and len(result) > 1 else 0
    while len(result) > check_idx and result[check_idx].get('role') in ['tool', 'tool_response']:
        result.pop(check_idx)

    return result

def compute_spectrum_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    cnt = max(1, len(records))
    tp = sum(1 for r in records if r.get('metrics', {}).get('confusion') == 'TP')
    fp = sum(1 for r in records if r.get('metrics', {}).get('confusion') == 'FP')
    fn = sum(1 for r in records if r.get('metrics', {}).get('confusion') == 'FN')
    tn = sum(1 for r in records if r.get('metrics', {}).get('confusion') == 'TN')
    errs = sum(1 for r in records if r.get('metrics', {}).get('confusion') == 'ERROR')

    # Total ground truth tool-required turns
    tool_req_turns = sum(1 for r in records if len(r.get('target_message', {}).get('tool_calls', [])) > 0)
    no_tool_turns = cnt - tool_req_turns
    emitted = tp + fp

    # Failures on tool turns MUST be included in the denominator
    prec = tp / emitted if emitted > 0 else 0.0
    rec = tp / tool_req_turns if tool_req_turns > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'turns_count': len(records),
        'tool_required_turns': tool_req_turns,
        'no_tool_turns': no_tool_turns,
        'tool_decision_precision': round(prec, 4),
        'tool_decision_recall': round(rec, 4),
        'tool_decision_f1': round(f1, 4),
        'evaluation_errors': errs
    }

# ==============================================================================
# TESTS
# ==============================================================================

def test_multi_tool_call_scoring_partial_match_and_arguments():
    # Target expects 2 tool calls
    target_msg = {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
            {'function': {'name': 'nmap_scan', 'arguments': {'target': '192.168.1.1', 'ports': '80,443'}}},
            {'function': {'name': 'http_probe', 'arguments': {'url': 'http://192.168.1.1'}}}
        ]
    }

    # Prediction emits only 1 tool call with WRONG argument value
    pred_text = '<tool_call>{"name": "nmap_scan", "arguments": {"target": "10.0.0.99"}}</tool_call>'

    scores = evaluate_replay_turn(pred_text, target_msg)

    # Must NOT report 100% / True
    assert scores['tool_name_match'] is False, "Should be False because http_probe was not emitted!"
    assert scores['tool_call_recall'] == 0.5, "Only 1 out of 2 expected calls was matched!"
    assert scores['tool_call_precision'] == 1.0
    assert scores['arg_value_accuracy'] == 0.0, "Target IP was wrong ('10.0.0.99' != '192.168.1.1')!"
    assert scores['full_tool_call_accuracy'] == 0.0, "Zero tool calls had completely correct arguments!"

def test_failed_evaluations_included_in_tool_recall():
    # 2 turns in dialogue:
    # Turn 1: tool required, succeeded
    # Turn 2: tool required, crashed / evaluation error
    records = [
        {
            'target_message': {'role': 'assistant', 'tool_calls': [{'name': 'tool_a'}]},
            'metrics': {'confusion': 'TP'}
        },
        {
            'target_message': {'role': 'assistant', 'tool_calls': [{'name': 'tool_b'}]},
            'metrics': {'confusion': 'ERROR'}
        }
    ]

    summary = compute_spectrum_metrics(records)
    assert summary['tool_required_turns'] == 2
    # Recall MUST be 1 / 2 = 0.5, NOT 1 / 1 = 1.0!
    assert summary['tool_decision_recall'] == 0.5
    assert summary['evaluation_errors'] == 1

def test_atomic_context_pruning_never_orphans_tool_responses():
    context = [
        {'role': 'system', 'content': 'You are an agent.'},
        {'role': 'user', 'content': 'Run port scan'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_1', 'function': {'name': 'scan'}}]},
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'Port 22 open'},
        {'role': 'user', 'content': 'Brute force SSH'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_2', 'function': {'name': 'hydra'}}]},
        {'role': 'tool', 'tool_call_id': 'call_2', 'content': 'Password found'},
    ]

    # Prune down to budget
    pruned = prune_context_atomic(context, max_messages=4)

    # 1. System message preserved
    assert pruned[0]['role'] == 'system'

    # 2. First non-system message must NOT be a tool message!
    assert pruned[1]['role'] != 'tool', "Orphaned tool response detected at start of pruned context!"

    # 3. For any tool message in pruned, its immediate preceding message must be the assistant call that spawned it
    for i in range(len(pruned)):
        if pruned[i]['role'] == 'tool':
            assert i > 0
            assert pruned[i-1]['role'] == 'assistant', f"Tool message at index {i} has non-assistant predecessor: {pruned[i-1]['role']}"

