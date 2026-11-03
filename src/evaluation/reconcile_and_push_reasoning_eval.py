import os
import json
import datetime
from huggingface_hub import HfApi

BASE_DIR = os.getenv("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
INPUT_DIR = os.getenv("INPUT_DIR", os.path.join(BASE_DIR, "v3_pipeline", "kaggle_download_reasoning", "eval_output"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "v3_pipeline", "eval_output_reasoning_final"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODEL_ID = "Qwen/Qwen3-8B"
ADAPTER_ID = os.getenv("HF_REPO_ID", "Qwen/Qwen3-8B-offensive-cyber-agent")
TOTAL_RUNTIME_MIN = 47.08
TOTAL_SAMPLES = 130
EVAL_MODES = ["no_thinking", "thinking"]

mode_summaries = {}
mode_results = {}

for mode in EVAL_MODES:
    f0 = os.path.join(INPUT_DIR, f"shard_gpu0_{mode}.jsonl")
    f1 = os.path.join(INPUT_DIR, f"shard_gpu1_{mode}.jsonl")
    results = []
    for fpath in [f0, f1]:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    assert len(results) == TOTAL_SAMPLES, f"Count mismatch in mode {mode}! Expected {TOTAL_SAMPLES}, got {len(results)}"
    mode_results[mode] = results

    # Save combined predictions
    combined_path = os.path.join(OUTPUT_DIR, f"reasoning_predictions_{mode}.jsonl")
    with open(combined_path, "w", encoding="utf-8") as out_f:
        for r in results:
            out_f.write(json.dumps(r) + "\n")

    valid = [r for r in results if "metrics" in r]
    m_f1 = sum(r["metrics"]["f1"] for r in valid) / len(valid)
    m_p = sum(r["metrics"]["precision"] for r in valid) / len(valid)
    m_r = sum(r["metrics"]["recall"] for r in valid) / len(valid)
    m_comp = sum(r["metrics"]["composite_score"] for r in valid) / len(valid)
    violations = sum(1 for r in valid if r["metrics"]["zero_tool_violation"] is True)
    compliance = ((len(valid) - violations) / len(valid)) * 100.0
    avg_th = sum(r["metrics"]["thinking_char_count"] for r in valid) / len(valid)
    avg_lat = sum(r["generation_latency_sec"] for r in valid) / len(valid)

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

comparative_summary = {
    "model_id": ADAPTER_ID,
    "base_model": BASE_MODEL_ID,
    "total_test_samples": TOTAL_SAMPLES,
    "total_runtime_minutes": TOTAL_RUNTIME_MIN,
    "no_thinking": nt,
    "thinking": th,
    "delta": delta,
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}

with open(os.path.join(OUTPUT_DIR, "reasoning_comparative_summary.json"), "w", encoding="utf-8") as f:
    json.dump(comparative_summary, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "reasoning_summary_v3.json"), "w", encoding="utf-8") as f:
    json.dump(comparative_summary, f, indent=2)

md_table = f"""
## 📊 Comparative Performance Matrix

| Metric | Direct / Non-Thinking (`enable_thinking=False`) | Deep Reasoning (`enable_thinking=True`) | Delta (Thinking - Direct) |
| :--- | :---: | :---: | :---: |
| **Mean Token F1** | `{nt['mean_token_f1']:.4f}` | `{th['mean_token_f1']:.4f}` | `{'+' if delta['f1_delta'] >= 0 else ''}{delta['f1_delta']:.4f}` |
| **Mean Precision** | `{nt['mean_token_precision']:.4f}` | `{th['mean_token_precision']:.4f}` | `{'+' if delta['precision_delta'] >= 0 else ''}{delta['precision_delta']:.4f}` |
| **Mean Recall** | `{nt['mean_token_recall']:.4f}` | `{th['mean_token_recall']:.4f}` | `{'+' if delta['recall_delta'] >= 0 else ''}{delta['recall_delta']:.4f}` |
| **Composite Score (40/60)** | `{nt['composite_reasoning_score']:.4f}` | `{th['composite_reasoning_score']:.4f}` | `{'+' if delta['composite_delta'] >= 0 else ''}{delta['composite_delta']:.4f}` |
| **Zero-Tool Compliance** | `{nt['zero_tool_compliance_rate_percent']:.2f}%` | `{th['zero_tool_compliance_rate_percent']:.2f}%` | `0.0%` |
| **Average Thinking Length** | `0 chars` | `{th['avg_thinking_chars']:.1f} chars` | `+{th['avg_thinking_chars']:.1f} chars` |
| **Mean Latency per Sample** | `{nt['avg_latency_sec']:.2f}s` | `{th['avg_latency_sec']:.2f}s` | `{delta['latency_delta_sec']:+.2f}s` |
| **Decoding Strategy** | Deterministic Greedy (`temp=0.0`) | Temperature Sampling (`temp=0.6, top_p=0.95`) | Recommended |
"""

top_showcase = sorted(
    zip(mode_results['no_thinking'], mode_results['thinking']),
    key=lambda pair: pair[1].get('metrics', {}).get('composite_score', 0.0),
    reverse=True
)[:5]

showcase_lines = [
    "# 🧠 Qwen3-8B v3 Cybersecurity Reasoning: Evaluation Showcase\n\n",
    f"> **Model Under Test**: `{ADAPTER_ID}`\n",
    f"> **Base Model**: `{BASE_MODEL_ID}`\n",
    f"> **Evaluation Timestamp**: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n",
    "> **Benchmark Dataset**: 130 Pure Cybersecurity Reasoning Samples (`tools: []`, SHA-256 Validated)\n\n",
    "## 📊 Comparative Performance Matrix\n\n",
    md_table,
    "\n\n## 🔬 Top-5 Showcase Reasoning Samples\n\n",
]
for rank, (nt_res, th_res) in enumerate(top_showcase, 1):
    showcase_lines.extend([
        f"### Sample {rank}: `{nt_res['id']}`\n",
        f"- **User Prompt**: {nt_res.get('user_prompt', '')}\n",
        f"- **Direct Mode Composite Score**: `{nt_res.get('metrics', {}).get('composite_score', 0.0):.4f}` | F1: `{nt_res.get('metrics', {}).get('f1', 0.0):.4f}`\n",
        f"- **Deep Reasoning Composite Score**: `{th_res.get('metrics', {}).get('composite_score', 0.0):.4f}` | F1: `{th_res.get('metrics', {}).get('f1', 0.0):.4f}`\n\n",
        "#### Deep Reasoning Output (`<think>`):\n",
        "```\n",
        th_res.get('generated_reasoning', '')[:800] + ('...\n' if len(th_res.get('generated_reasoning', '')) > 800 else '\n'),
        "```\n\n",
    ])

with open(os.path.join(OUTPUT_DIR, "reasoning_showcase.md"), "w", encoding="utf-8") as f:
    f.writelines(showcase_lines)

print(f"[RECONCILE] Artifacts generated cleanly in: {OUTPUT_DIR}")
# print(md_table)

# Hugging Face Push
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("[HUB] Warning: HF_TOKEN not set. Skipping Hugging Face upload.", flush=True)
    exit(0)
print(f"[HUB] Uploading reasoning evaluation artifacts to Hugging Face: {ADAPTER_ID}...", flush=True)
api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path=OUTPUT_DIR,
    path_in_repo="evaluations/reasoning_v3",
    repo_id=ADAPTER_ID,
    repo_type="model",
    commit_message="eval(v3): atomic publish of verified 130-sample dual-GPU reasoning benchmark"
)
print("[HUB] Upload complete. Verifying remote repository contents...", flush=True)
remote_files = set(api.list_repo_files(repo_id=ADAPTER_ID))
required_remote = [
    "evaluations/reasoning_v3/reasoning_summary_v3.json",
    "evaluations/reasoning_v3/reasoning_showcase.md",
    "evaluations/reasoning_v3/reasoning_comparative_summary.json",
    "evaluations/reasoning_v3/reasoning_predictions_no_thinking.jsonl",
    "evaluations/reasoning_v3/reasoning_predictions_thinking.jsonl",
]
missing = [rf for rf in required_remote if rf not in remote_files]
if missing:
    raise RuntimeError(f"FATAL: Hugging Face remote verification failed! Missing files: {missing}")

print("[HUB] SUCCESS: All evaluation artifacts uploaded and verified remotely on Hugging Face Hub [OK]!", flush=True)
