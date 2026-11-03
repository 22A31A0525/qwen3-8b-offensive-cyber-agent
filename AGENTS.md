# Global Antigravity Agent Guidelines & Project Architecture

## Mission & Scope
This project finetunes and evaluates **Qwen38B Latest Version (v3)** across three tracks with an exact **60% Exploit / 20% Cybersecurity Reasoning / 20% Tool Replay** curriculum:
1. **Track 1 (Reasoning)**: 130 complex cybersecurity reasoning questions.
2. **Track 2 (Exploit)**: 390 multiturn exploitation and security testing dialogues.
3. **Track 3 (Tool Calling Replay)**: 130 holdout functioncalling dialogues (479 total turns).



## NonNegotiable ProjectWide Operating Rules

### 1. Dataset Integrity
 **Immutability of Evaluation & Training Sets**: Never modify, delete, or alter original train (`train.jsonl`), validation (`val.jsonl`), or test (`test.jsonl`) records, 60/20/20 curriculum balance requirements, test labels, tool names, or expected answers to artificially improve evaluation scores.
 **Checksum & Revision Tracking**: Dataset files must have recorded SHA256 checksums and line counts. Any approved dataset evolution must be documented with an explicit version tag, Git commit, and change rationale.

### 2. Checkpoint Integrity
 **Full Provenance Verification**: Prior to launching any benchmark or evaluation, verify and log the exact model checkpoint commit hash, base model repository, adapter configuration (`adapter_config.json`), tokenizer revision, training step, and loss curves.
 **Explicit Base + Adapter Separation**: When loading PEFT models, ensure the base model and LoRA adapter weights match the target training phase.

### 3. Fair Evaluation (Strict Comparability)
 **Zero Drift Across Model Comparisons**: Benchmark comparisons between v1 baseline, v2 checkpoints, and v3 pipeline must use identical holdout examples, identical tool schemas, identical chat template prompt construction, identical decoding settings (greedy: `do_sample=False`, `temperature=0.1`, `repetition_penalty=1.1`, `max_new_tokens=1024`), and identical scoring logic.
 **Explicit Discrepancy Reporting**: If identical conditions cannot be maintained, differences must be stated explicitly upfront before any comparison metrics are presented.

### 4. Failure Handling & Exact Count Reconciliation
 **Zero Tolerate for Dropped Shards**: A failed, canceled, missing, timedout, or incomplete shard must never be counted as successfully evaluated.
 **100% Dialogue Reconciliation**: Every multiturn replay benchmark run must account for strictly **130 / 130 dialogues** and **479 total turns**. Any missing dialogues or skipped turns must immediately halt the pipeline, trigger an investigation, and prevent publication.

### 5. Security & Credential Hygiene
 **Zero Token Leakage**: Never hardcode authentication tokens (Hugging Face tokens, Kaggle API keys/tokens) in source code, committed scripts, or notebooks. Use environment variables (`HF_TOKEN`, `KAGGLE_KEY`, `KAGGLE_API_TOKEN`) or platform secret vaults.
 **Credential Rotation**: Any token inadvertently printed in execution logs or committed to version control must be revoked and rotated immediately. Never print raw token strings in unbuffered stdout/stderr logs.

### 6. Raw Output Validation & Impossible Metric Detection
 **Inspect Raw Predictions First**: Prior to computing summary metrics, print and inspect the first 5–10 raw predictions (`generated_text`). Verify whether `<tool_call>` syntax is produced or conversational text was emitted.
 **Automated Sanity Constraints**: The evaluation pipeline must automatically reject impossible metric combinations:
   If emitted tool calls ($TP + FP$) == 0, `schema_validity` MUST be `null` (N/A), never `100%` or `0.0`.
   If toolrequired turns ($TP + FN$) == 0, `tool_name_accuracy` MUST be `null` (N/A).
   True Negatives ($TN$) on notool turns must NEVER be counted as matching tool names or valid schema.
   Total turns MUST strictly equal $TP + FP + FN + TN$.

### 7. EvidenceBased Claims
 **Facts vs. Hypotheses vs. Recommendations**: Clearly categorize all statements into verified facts (backed by execution output), hypotheses (unverified conjectures), and recommendations.
 **Zero Unearned Confidence**: Never declare that finetuning "improved reasoning", "boosted tool accuracy", or "solved tool dropout" until verified benchmark execution outputs prove it.

### 8. No Silent Fixes (Strict Change Auditability)
 **Document All Pipeline Modifications**: Never alter model prompts, generation parameters, chat templates, data preprocessing, or evaluation scripts without documenting the exact change, the root cause, and the effect on backward comparability.

### 9. PrePublication Check & Automated Verification Gates
 **Automated Integrity Gate Mandatory**: Run automated validation tests (`v3_pipeline/scripts/validate_evaluation_integrity.py`) and generate an audit report before declaring any training session, evaluation, or benchmark run complete.
 **Enforceable Gate**: If the automated validator fails on any check (checksums, missing IDs, impossible metrics), the pipeline must abort and block metric publication.

### 10. Strict Legacy Isolation & Access Restriction
 **Absolute Quarantine by Default**: The `legacy/` directory is strictly quarantined. The agent, subagents, and automated workflows MUST NEVER inspect, read, grep, import, borrow logic from, or execute files inside `legacy/`. All new code and workflows must be freshly authored in the active workspace.
 **Explicit Manual Override Only**: Under NO circumstances should `legacy/` be accessed UNLESS the user manually and explicitly instructs to refer to `legacy/` in their prompt (e.g., *"refer to legacy script X"*). Without an explicit, manual user directive, `legacy/` is completely ignored and treated as nonexistent.
 **Zero CrossContamination**: Never use `legacy/` as an automatic fallback for missing files, missing logic, or dependencies. Never copy files out of `legacy/` without explicit user permission.

### 11. Structured Documentation & TableDriven Standards
 **TableFirst Formatting**: All READMEs, architecture summaries, manifests, and data specifications must be presented using clean, structured GitHub Flavored Markdown (GFM) tables.
 **No Sprawling CodeBlock Trees**: Sprawling, unformatted ASCII tree blocks with inline comments (`# ...`) are prohibited in documentation. Use structured tables for file paths, splits, counts, and descriptions.
 **Active Pipeline Focus**: Only document active production files (`v3_pipeline/`). Legacy directories are sealed under Rule 10 and do not require documentation updates.



## Pure Academic Metrics Standard (Track 3 Replay)
Multiturn tool calling must report an explicit confusion matrix:
 **$TP$**: Tool required, valid tool call emitted.
 **$FP$**: No tool required, but tool call emitted.
 **$FN$**: Tool required, but model emitted text (missed tool call).
 **$TN$**: No tool required, model correctly emitted conversational text.

Core formulas:
 **Tool Decision Precision**: $TP / (TP + FP)$ (or 0.0 if $TP + FP == 0$)
 **Tool Decision Recall**: $TP / (TP + FN)$ (or 0.0 if $TP + FN == 0$)
 **Tool Decision F1**: $2 \times P \times R / (P + R)$ (or 0.0 if $P + R == 0$)
 **Action Concordance**: $(TP + TN) / \text{Total Turns}$
 **NoTool Adherence (TNR)**: $TN / (TN + FP)$
 **Exact Tool Name Accuracy**: Evaluated strictly on toolrequired turns ($TP / (TP + FN)$). On notool turns, reported as `null`.
 **Emitted Call Schema Validity**: Evaluated strictly on emitted tool calls ($TP + FP$). If 0 emitted, reported as `null`.



## Naming & Storage Constraints
 Always refer to the model as **"Qwen38B Latest Version (v3)"** or **"v3 Pipeline"** — never use deprecated phrases.
 Consolidated result folders (e.g. `results/multi_tool_call_latest/`) must contain **exactly two files**: one `.jsonl` predictions file and one `.json` summary file.
 All entries in `.jsonl` files must carry `conv_id` and `turn_index` linking each turn back to its parent dialogue, and the summary file must include a `dialogues_index`.
