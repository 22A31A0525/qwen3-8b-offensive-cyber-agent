import os
import json
from collections import defaultdict

BASE_DIR = os.getenv("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
REASONING_DIR = os.path.join(BASE_DIR, "v3_pipeline", "results", "reasoning")
TOOL_REPLAY_DIR = os.path.join(BASE_DIR, "v3_pipeline", "results", "tool_replay")
TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "clean", "test")

# Target folders: both in results/predictions and results/final_dataset/predictions
TARGET_DIRS = [
    os.path.join(BASE_DIR, "v3_pipeline", "results", "predictions"),
    os.path.join(BASE_DIR, "v3_pipeline", "results", "final_dataset", "predictions")
]

for base_target in TARGET_DIRS:
    os.makedirs(os.path.join(base_target, "reasoning"), exist_ok=True)
    os.makedirs(os.path.join(base_target, "multi_turn"), exist_ok=True)

print("[PREDICTIONS] Generating clean Predictions folder structure...")

# -------------------------------------------------------------------------
# 1. REASONING (Thinking Mode Only)
# -------------------------------------------------------------------------
src_reasoning_pred = os.path.join(REASONING_DIR, "reasoning_predictions_thinking.jsonl")
assert os.path.exists(src_reasoning_pred), f"Missing reasoning predictions: {src_reasoning_pred}"

with open(src_reasoning_pred, "r", encoding="utf-8") as f:
    reasoning_records = [json.loads(line) for line in f if line.strip()]

assert len(reasoning_records) == 130, f"Expected 130 reasoning records, got {len(reasoning_records)}"

# Load reasoning metrics and judge summaries
src_reasoning_sum = os.path.join(REASONING_DIR, "reasoning_comparative_summary.json")
src_judge_sum = os.path.join(REASONING_DIR, "reasoning_judge_summary_thinking.json")
src_judge_preds = os.path.join(REASONING_DIR, "reasoning_judge_predictions_thinking.jsonl")

with open(src_reasoning_sum, "r", encoding="utf-8") as f:
    r_sum = json.load(f)["thinking"]

with open(src_judge_sum, "r", encoding="utf-8") as f:
    j_sum = json.load(f)

# Error category breakdown from judge predictions
error_breakdown = defaultdict(int)
if os.path.exists(src_judge_preds):
    with open(src_judge_preds, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                cat = item.get("judge", {}).get("error_category", "none")
                error_breakdown[cat] += 1

reasoning_analysis = {
    "evaluation_mode": "deep_reasoning_thinking",
    "model_id": os.getenv("HF_REPO_ID", "Qwen/Qwen3-8B-offensive-cyber-agent"),
    "base_model": "Qwen/Qwen3-8B",
    "total_samples_evaluated": len(reasoning_records),
    "token_level_benchmark": {
        "mean_token_f1": r_sum["mean_token_f1"],
        "mean_token_precision": r_sum["mean_token_precision"],
        "mean_token_recall": r_sum["mean_token_recall"],
        "composite_reasoning_score": r_sum["composite_reasoning_score"],
        "zero_tool_compliance_rate_percent": r_sum["zero_tool_compliance_rate_percent"],
        "zero_tool_violations_count": r_sum["zero_tool_violations_count"],
        "avg_latency_sec": r_sum["avg_latency_sec"]
    },
    "llm_as_a_judge_benchmark": {
        "evaluator_model": "qwen/qwen3.8-27b (via Groq)",
        "academic_pass_rate": j_sum["pass_rate"],
        "pass_count": j_sum["pass_count"],
        "fail_count": j_sum["total_samples"] - j_sum["pass_count"],
        "factual_correctness_rate": j_sum["factual_correctness_rate"],
        "factual_correctness_count": j_sum["factual_correctness_count"],
        "micro_concept_f1": j_sum["micro_f1"],
        "micro_concept_precision": j_sum["micro_precision"],
        "micro_concept_recall": j_sum["micro_recall"],
        "macro_concept_f1": j_sum["macro_f1"],
        "mean_composite_score": j_sum["mean_composite_score"],
        "critical_security_errors_count": j_sum["critical_security_errors_count"],
        "error_category_breakdown": dict(error_breakdown)
    }
}

for base_target in TARGET_DIRS:
    tgt_r_pred = os.path.join(base_target, "reasoning", "reasoning_predictions_thinking.jsonl")
    tgt_r_analysis = os.path.join(base_target, "reasoning", "reasoning_analysis.json")
    
    with open(tgt_r_pred, "w", encoding="utf-8") as f:
        for r in reasoning_records:
            f.write(json.dumps(r) + "\n")
            
    with open(tgt_r_analysis, "w", encoding="utf-8") as f:
        json.dump(reasoning_analysis, f, indent=2)

print("[PREDICTIONS] Reasoning thinking predictions & analysis created [OK].")

# -------------------------------------------------------------------------
# 2. MULTI-TURN (All 3 Shards Merged - Conversation-Wide & Turn-Level)
# -------------------------------------------------------------------------
src_multiturn_pred = os.path.join(TOOL_REPLAY_DIR, "tool_replay_predictions.jsonl")
src_multiturn_sum = os.path.join(TOOL_REPLAY_DIR, "tool_replay_summary.json")
orig_tool_replay = os.path.join(TEST_DATA_DIR, "tool_replay_20pct.jsonl")

assert os.path.exists(src_multiturn_pred), f"Missing multi-turn predictions: {src_multiturn_pred}"
assert os.path.exists(orig_tool_replay), f"Missing original tool replay test set: {orig_tool_replay}"

with open(src_multiturn_pred, "r", encoding="utf-8") as f:
    turn_records = [json.loads(line) for line in f if line.strip()]

assert len(turn_records) == 479, f"Expected 479 turns, got {len(turn_records)}"

with open(orig_tool_replay, "r", encoding="utf-8") as f:
    orig_dialogues = [json.loads(line) for line in f if line.strip()]

assert len(orig_dialogues) == 130, f"Expected 130 dialogues, got {len(orig_dialogues)}"

# Index turns by conv_id
turns_by_conv = defaultdict(list)
for t in turn_records:
    turns_by_conv[t["conv_id"]].append(t)

for cid in turns_by_conv:
    turns_by_conv[cid].sort(key=lambda x: x["turn_index"])

# Build conversation-wide records
conversation_wide_records = []
for d_idx, d in enumerate(orig_dialogues, 1):
    cid = d["id"]
    eval_turns = turns_by_conv.get(cid, [])
    
    tp_count = sum(1 for t in eval_turns if t.get("metrics", {}).get("confusion") == "TP")
    fp_count = sum(1 for t in eval_turns if t.get("metrics", {}).get("confusion") == "FP")
    fn_count = sum(1 for t in eval_turns if t.get("metrics", {}).get("confusion") == "FN")
    tn_count = sum(1 for t in eval_turns if t.get("metrics", {}).get("confusion") == "TN")
    err_count = sum(1 for t in eval_turns if t.get("status") == "EVALUATION_FAILURE")
    
    gt_tools = sum(1 for t in eval_turns if t.get("metrics", {}).get("confusion") in ["TP", "FN"])
    emitted_tools = sum(1 for t in eval_turns if t.get("metrics", {}).get("confusion") in ["TP", "FP"])
    all_concordant = all(t.get("metrics", {}).get("tool_action_match") is True for t in eval_turns if t.get("metrics"))
    
    conv_entry = {
        "conv_id": cid,
        "dialogue_index": d_idx,
        "tools": d.get("tools", []),
        "full_messages": d.get("messages", []),
        "evaluated_turns_count": len(eval_turns),
        "conversation_summary": {
            "all_actions_concordant": all_concordant,
            "ground_truth_tool_turns": gt_tools,
            "emitted_tool_turns": emitted_tools,
            "confusion": {"TP": tp_count, "FP": fp_count, "FN": fn_count, "TN": tn_count, "errors": err_count},
            "mean_content_f1": round(sum(t.get("metrics", {}).get("content_f1", 0.0) for t in eval_turns) / max(1, len(eval_turns)), 4)
        },
        "evaluated_turns": eval_turns
    }
    conversation_wide_records.append(conv_entry)

assert len(conversation_wide_records) == 130, f"Expected 130 conversation-wide records, got {len(conversation_wide_records)}"

with open(src_multiturn_sum, "r", encoding="utf-8") as f:
    mt_sum = json.load(f)

multi_turn_analysis = {
    "evaluation_mode": "teacher_forced_multi_turn_tool_replay",
    "model_id": os.getenv("HF_REPO_ID", "Qwen/Qwen3-8B-offensive-cyber-agent"),
    "base_model": "Qwen/Qwen3-8B",
    "total_dialogues_evaluated": len(conversation_wide_records),
    "total_turns_evaluated": len(turn_records),
    "shards_merged": 3,
    "shard_distribution": {
        "shard_1": "44 dialogues | 148 turns",
        "shard_2": "43 dialogues | 172 turns",
        "shard_3": "43 dialogues | 159 turns"
    },
    "overall_metrics": mt_sum["overall_metrics"],
    "standard_context_metrics_le_4096": mt_sum["standard_context_metrics_le_4096"],
    "long_context_metrics_gt_4096": mt_sum["long_context_metrics_gt_4096"]
}

for base_target in TARGET_DIRS:
    tgt_mt_conv = os.path.join(base_target, "multi_turn", "multi_turn_predictions_conversation_wide.jsonl")
    tgt_mt_turns = os.path.join(base_target, "multi_turn", "multi_turn_predictions_turn_level.jsonl")
    tgt_mt_analysis = os.path.join(base_target, "multi_turn", "multi_turn_analysis.json")
    
    with open(tgt_mt_conv, "w", encoding="utf-8") as f:
        for c in conversation_wide_records:
            f.write(json.dumps(c) + "\n")
            
    with open(tgt_mt_turns, "w", encoding="utf-8") as f:
        for t in turn_records:
            f.write(json.dumps(t) + "\n")
            
    with open(tgt_mt_analysis, "w", encoding="utf-8") as f:
        json.dump(multi_turn_analysis, f, indent=2)

print("[PREDICTIONS] Multi-turn conversation-wide & turn-level predictions & analysis created [OK].")

# Print summary manifest
print("\n" + "="*70)
print("PREDICTIONS DIRECTORY MANIFEST VERIFIED")
print("="*70)
for base_target in TARGET_DIRS:
    print(f"\nLocation: {base_target}")
    for root, dirs, files in os.walk(base_target):
        for file in files:
            p = os.path.join(root, file)
            sz = os.path.getsize(p)
            rel = os.path.relpath(p, base_target)
            print(f"  {rel:<55} ({sz:,} bytes)")

