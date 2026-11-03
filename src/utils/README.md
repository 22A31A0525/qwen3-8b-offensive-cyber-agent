# Utilities & Deployment Tooling (`src/utils`)

[![Utilities](https://img.shields.io/badge/Tooling-Packaging%20%26%20Publishing-blue.svg)](#1-utilities-index)
[![Standards](https://img.shields.io/badge/Linting-Rule%2011%20Compliant-green.svg)](#5-documentation-standards-enforcement-format_all_documentationpy)
[![Publishing](https://img.shields.io/badge/Deployment-Hugging%20Face%20Hub-orange.svg)](#4-hugging-face-hub-publishing-push_adapter_to_hubpy)

This directory contains supporting utilities for artifact packaging, prediction aggregation, documentation standardization, and Hugging Face Hub publishing for **Qwen3-8B Latest Version (v3)**.

---

## 1. Utilities Index

| Module                                                                     | Primary Function                               | Input Dependencies              | Output Artifacts                                                                                           |
| :------------------------------------------------------------------------- | :--------------------------------------------- | :------------------------------ | :--------------------------------------------------------------------------------------------------------- |
| **[`package_predictions_clean.py`](package_predictions_clean.py)**         | Normalizes and packages raw evaluation outputs | Shard prediction JSONL files    | `v3_pipeline/results/predictions/` (Clean reasoning, multi-turn, and exploit predictions)                  |
| **[`package_final_dataset_results.py`](package_final_dataset_results.py)** | Compiles final dataset manifests & checksums   | `data/clean/` JSONL splits      | `v3_pipeline/results/final_dataset/` with verified SHA-256 manifest and dataset card                       |
| **[`push_adapter_to_hub.py`](push_adapter_to_hub.py)**                     | Publishes adapter weights & model cards        | Output checkpoint directory     | Hugging Face Hub adapter repo `${HF_REPO_ID:-Qwen/Qwen3-8B-offensive-cyber-agent}` (Base: `Qwen/Qwen3-8B`) |
| **[`format_all_documentation.py`](format_all_documentation.py)**           | Lints and formats markdown documentation       | Markdown docs across repository | Uniform UTF-8, pipe-aligned GFM tables conforming to Rule 11                                               |

---

## 2. Prediction Aggregation (`package_predictions_clean.py`)

Raw predictions generated across distributed Kaggle shards contain internal runtime metadata, token timestamps, and shard identifiers. `package_predictions_clean.py` extracts, structures, and canonicalizes these outputs into clean datasets:

### Output Formats:
1. **Dialogue-Level Predictions**:
   ```json
   {
     "id": "hermes_1234abcd",
     "track": "tool_replay",
     "total_turns": 4,
     "correct_turns": 4,
     "trajectory_success": true,
     "messages": [ ... ]
   }
   ```
2. **Turn-Level Predictions**:
   ```json
   {
     "turn_id": "hermes_1234abcd_t2",
     "parent_id": "hermes_1234abcd",
     "turn_index": 2,
     "prompt_tokens": 1284,
     "target_action": "<tool_call>{\"name\": \"nmap_scan\", ...}</tool_call>",
     "predicted_action": "<tool_call>{\"name\": \"nmap_scan\", ...}</tool_call>",
     "exact_match": true,
     "json_valid": true
   }
   ```

---

## 3. Final Dataset Packaging (`package_final_dataset_results.py`)

Packages the audited 6,500-dialogue clean dataset for distribution and reproducibility:
- **Checksum Calculation**: Generates SHA-256 hashes for all 9 data partitions (3 splits $\times$ 3 domains).
- **Turn & Token Accounting**: Computes total assistant turns, tool call counts, and token budget statistics.
- **Manifest Generation**: Generates `dataset_summary.json` and a publication-ready `README.md` dataset card.

---

## 4. Hugging Face Hub Publishing (`push_adapter_to_hub.py`)

Handles publishing model artifacts to the Hugging Face Hub repository:
**`${HF_REPO_ID:-Qwen/Qwen3-8B-offensive-cyber-agent}`** (Base Model: `Qwen/Qwen3-8B`)

### Published Assets:
- Fine-tuned PEFT adapter weights (`adapter_model.safetensors`, `adapter_config.json`).
- Model card containing full technical specifications, LoRA hyperparameter tables, and benchmark performance across all 3 evaluation tracks.
- Training loss curves and telemetry summaries.

### Usage:
```bash
python -m src.utils.push_adapter_to_hub \
    --adapter_dir v3_pipeline/outputs/v3_lora_run \
    --repo_id ${HF_REPO_ID:-Qwen/Qwen3-8B-offensive-cyber-agent}
```

---

## 5. Documentation Standards Enforcement (`format_all_documentation.py`)

Enforces repository-wide formatting standards:
- **Rule 11 Compliance**: Guarantees all markdown tables have aligned vertical pipes (`|`), identical column widths, and proper header separators.
- **UTF-8 Sanitization**: Eliminates non-standard character corruptions (`\ufffd`).
- **Escaping**: Ensures pipes inside table cells are escaped (`&#124;` or `or`) to prevent broken rendering in GitHub and Hugging Face markdown viewers.
