# 3-Track Evaluation Suite (`src/evaluation`)

[![Evaluation Tracks](https://img.shields.io/badge/Tracks-Reasoning%20%7C%20Replay%20%7C%20Exploit-blue.svg)](#1-3-track-evaluation-architecture)
[![Distributed Engine](https://img.shields.io/badge/Orchestration-Kaggle%20Dual--GPU%20Shards-orange.svg)](#4-centralized-dynamic-kaggle-sharding-engine-kaggle_shard_enginepy)
[![Judge System](https://img.shields.io/badge/Judge-2--Layer%20G--Eval%20Cascade-green.svg)](#3-track-1-llm-as-a-judge-reasoning-judge_reasoning_v3py)
[![Reconciliation](https://img.shields.io/badge/Reconciliation-100%25%20Verified%20(Rule%204)-purple.svg)](#6-shard-reconciliation-logic)

This directory contains the orchestration engines, notebook generators, LLM judge systems, and shard reconciliation routines for **Qwen3-8B Latest Version (v3)**.

---

## 1. 3-Track Evaluation Architecture

Evaluation is divided into three distinct, specialized tracks covering the entire capability spectrum:

```mermaid
flowchart TD
    subgraph TestHoldouts["Test Holdout Corpus (650 Dialogues)"]
        ExpTest["Track 3: Exploit Execution (390 Dialogues / 60%)"]
        ReplayTest["Track 2: Multi-Turn Tool Replay (130 Dialogues / 20%)"]
        ReasonTest["Track 1: Cyber Reasoning (130 Dialogues / 20%)"]
    end

    subgraph DistributedExec["Kaggle Multi-GPU Distributed Shards"]
        ExpTest --> ExpShards["3x Exploit Shards: 130 / 130 / 130"]
        ReplayTest --> ReplayShards["3x Replay Shards: 44 / 43 / 43"]
        ReasonTest --> ReasonRunner["Dedicated Reasoning Runner"]
    end

    subgraph Analysis["Scoring & Reconciliation"]
        ExpShards --> RecExp["reconcile_and_push_exploit_eval.py"]
        ReplayShards --> RecReplay["reconcile_and_push_multiturn_eval.py"]
        ReasonRunner --> Judge["judge_reasoning_v3.py (2-Layer Cascade)"]
        Judge --> RecReason["reconcile_and_push_reasoning_eval.py"]
    end

    subgraph Outputs["Hugging Face Hub Publication"]
        RecExp --> Hub["${HF_REPO_ID:-Qwen/Qwen3-8B-offensive-cyber-agent}"]
        RecReplay --> Hub
        RecReason --> Hub
    end
```

---

## 2. Track Breakdown & Methodologies

| Track                                | Target Domain                                                    |                Test Volume                | Evaluation Methodology                                                                     | Primary Metrics                                                           |
| :----------------------------------- | :--------------------------------------------------------------- | :---------------------------------------: | :----------------------------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| **Track 1: Cybersecurity Reasoning** | Threat analysis, reverse engineering concepts, incident triage   |               130 Dialogues               | **Primary**: LLM-as-a-Judge (Qwen3.8-27B on Groq)<br>**Auxiliary**: Zero-Tool Gate & Floor | Academic Pass Rate, Factual Correctness, Concept F1, Zero-Tool Compliance |
| **Track 2: Multi-Turn Tool Replay**  | Stateful tool interactions, error recovery, shell navigation     | 130 Dialogues (479 turns, 241 tool calls) | Ground-truth trajectory replay under context-constrained inference                         | Turn-level Tool Accuracy, Argument Exact-Match, Trajectory Continuity     |
| **Track 3: Exploit Trajectories**    | Vulnerability assessment, payload synthesis, exploitation chains |               390 Dialogues               | Autonomous exploit execution and structured function dispatch                              | Strict Tool Selection, Valid JSON Schema % , Exploit Step Completion      |

---

## 3. Track 1: LLM-as-a-Judge Reasoning (`judge_reasoning_v3.py`)

Track 1 evaluates pure cybersecurity reasoning across 130 holdout dialogues (`test.jsonl`).

### 3.1 Two Evaluation Paradigms: Heuristics vs. LLM-as-a-Judge

In reasoning evaluation, two distinct methodologies exist:

| Feature / Dimension           | Legacy Token-Level Heuristics (Token F1 / ROUGE)                             | Authoritative LLM-as-a-Judge Standard (Groq Qwen3.8-27B)                                   |
| :---------------------------- | :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- |
| **Evaluation Basis**          | Surface unigram / character overlap against ground truth                     | Semantic concept extraction, factual correctness, and safety audit                         |
| **Synonym & Parity Handling** | **Fails**: Heavily penalizes valid alternative CLI flags, cmdlets, or syntax | **Robust**: Semantically equates valid commands, flags, and mitigations                    |
| **Hallucination Detection**   | **Blind**: Cannot detect fabricated CVEs, false exploits, or insecure steps  | **Active**: Audits and penalizes hallucinated CVEs, inverted access control, syntax errors |
| **Role in Pipeline**          | **Auxiliary Diagnostic**: Used as a sanity floor / local fast heuristic      | **Primary & Authoritative Gate**: Definitive academic standard for model benchmarking      |

### 3.2 Prediction Parsing (`generated_text` / `<think>...</think>`)

Reasoning models (like Qwen3-8B in thinking mode) emit internal Chain-of-Thought (CoT) reasoning enclosed within `<think>...</think>` tags followed by final conclusions:
- **`parse_prediction_content(raw_prediction)`**:
  - **Thinking Trace Extraction**: Extracts internal step-by-step reasoning from `<think>...</think>`.
  - **Final Answer Extraction**: Strips thinking blocks to isolate the emitted answer.
  - **Unfinished Generation Handling**: If generation exceeds context or token limits and closes abruptly inside `<think>`, flags `is_unfinished: True` and passes available reasoning for judgment.
- **Judge Context Dispatch**:
  - The LLM Judge receives the user query, ground truth reference, isolated candidate answer, and the candidate's `<think>` trace in dedicated sections.
  - This allows the judge to evaluate both:
    1. The factual correctness and completeness of the final answer against reference ground truth.
    2. The technical soundness of the reasoning trajectory (detecting hallucinations or logic gaps).

### 3.3 Authoritative LLM-as-a-Judge Methodology

- **Judge Engine**: `qwen/qwen3.8-27b` executed at ultra-high throughput on Groq Cloud (with automatic fallback to Google Gemini `gemini-2.5-flash`).
- **Semantic Decomposition**:
  - Extracts 2–6 atomic technical requirements from the reference answer.
  - Matches candidate assertions against reference concepts:
    - **True Positives (TP)**: Required reference security concepts correctly recovered.
    - **False Positives (FP)**: Hallucinated, unsupported, or technically dangerous assertions.
    - **False Negatives (FN)**: Omitted critical security requirements.
- **Academic Verdict Criteria**:
  - **`PASS`**: Factual Correctness is True, zero critical security errors, and Concept Recall $\ge 0.50$.
  - **`FAIL`**: Factual error, critical security violation, or Concept Recall $< 0.50$.
- **Error Taxonomy**: Categorizes failures into `"wrong_tool_or_command"`, `"hallucinated_vulnerability"`, `"missing_core_concepts"`, `"irrelevant_answer"`, `"factual_error"`, `"incomplete_explanation"`, or `"truncated_thinking"`.

### 3.4 Auxiliary Diagnostic Floor & Legacy Metrics

To maintain longitudinal tracking against legacy checkpoints:
- **Zero-Tool Integrity Gate**: Pure reasoning queries must NOT emit tools (`<tool_call>`). Any tool leak immediately receives a disqualification score of `0.0`.
- **Token F1 Floor**: Surface unigram overlap ($2 \cdot P \cdot R / (P + R)$) combined with a brevity factor ($\min(1.0, 150 / \text{words})$).
- **Legacy Composite Score**: $0.40 \times \text{Floor Score} + 0.60 \times \text{Judge Score}$ (reported for backward comparability with v2).

### 3.5 API Key Configuration & Execution

The judge engine is highly resilient and supports single or rotated API keys:
- **Single Key**: Set `GROQ_API_KEY` in `.env` or execution environment.
- **Multiple Keys**: Comma-separated `GROQ_API_KEYS="key1,key2"` or numbered `GROQ_API_KEY_1..N`.
- **Fallback Keys**: `GEMINI_API_KEY` or `GEMINI_API_KEY_1..N`.

```bash
# Run authoritative LLM-as-a-Judge evaluation across both thinking and direct modes
python src/evaluation/judge_reasoning_v3.py

# Run on custom predictions directory with explicit Groq model
python src/evaluation/judge_reasoning_v3.py --predictions-dir v3_pipeline/results/reasoning --groq-model qwen/qwen3.8-27b

# Run single file evaluation without Hugging Face upload
python src/evaluation/judge_reasoning_v3.py --single-file v3_pipeline/results/reasoning/reasoning_predictions_thinking.jsonl --skip-hub
```

---

## 4. Centralized Dynamic Kaggle Sharding Engine (`kaggle_shard_engine.py`)

The dynamic sharding engine (`src/evaluation/kaggle_shard_engine.py`) provides a centralized, programmatic architecture for partitioning any evaluation track across an arbitrary number of Kaggle worker accounts without maintaining static, hardcoded scripts.

### 4.1 Key Capabilities
- **Dynamic Shard Count ($N$-Way)**: Generate any arbitrary number of shards $N$ (`--num-shards 3`, `--num-shards 5`), or auto-derive $N$ directly from connected accounts.
- **Environment Account Derivation (`--from-env`)**: Automatically reads comma-separated account usernames from the `KAGGLE_ACCOUNTS` environment variable (e.g. `KAGGLE_ACCOUNTS=acc1,acc2,acc3,acc4`).
- **Unified Track Dispatch**: One CLI for all three evaluation tracks:
  - `exploit`: Exploit Trajectories (`data/clean/test/exploit_60pct.jsonl` -> 390 dialogues)
  - `tool_replay`: Multi-Turn Tool Replay (`data/clean/test/tool_replay_20pct.jsonl` -> 130 dialogues)
  - `reasoning`: Cybersecurity Reasoning (`data/clean/test/reasoning_20pct.jsonl` -> 130 dialogues)
  - `all`: Unified generation across all 3 tracks into organized subdirectories.
- **Dialogue-Level Preservation**: Partitions are balanced while guaranteeing zero dialogue fragmentation (every multi-turn conversation remains intact within a single shard).
- **Self-Contained Notebook Generation**: Programmatically generates standalone `.ipynb` notebooks containing compressed zlib/base64 payloads, dual-GPU parallel execution workers, and automated post-run metrics reconciliation.
- **Automated Metadata Creation**: Programmatically creates valid `kernel-metadata.json` configured with dual Tesla T4 GPUs (`accelerator: nvidiaTeslaT4Dual`).

### 4.2 CLI Usage Examples

```bash
# 1. 3-way sharded package for Exploit evaluation
python -m src.evaluation.kaggle_shard_engine --track exploit --num-shards 3 --output-dir kaggle_deploys

# 2. Multi-turn Tool Replay sharded dynamically from KAGGLE_ACCOUNTS environment variable
python -m src.evaluation.kaggle_shard_engine --track tool_replay --from-env

# 3. 2-way reasoning benchmark partition
python -m src.evaluation.kaggle_shard_engine --track reasoning --num-shards 2 --output-dir kaggle_deploys_reasoning

# 4. Generate sharded packages for all three tracks simultaneously
python -m src.evaluation.kaggle_shard_engine --track all --num-shards 3 --output-dir kaggle_deploys_all

# 5. Dry-run mode for pre-flight verification without writing to disk
python -m src.evaluation.kaggle_shard_engine --track exploit --num-shards 5 --dry-run
```

### 4.3 Python Library API

```python
from src.evaluation.kaggle_shard_engine import KaggleShardConfig, KaggleShardEngine

# Configure and execute programmatically
config = KaggleShardConfig(
    track="exploit",
    num_shards=3,
    accounts=["shard1_user", "shard2_user", "shard3_user"],
    output_dir="kaggle_deploys/exploit"
)

engine = KaggleShardEngine(config)
manifest = engine.run()
```

---

## 5. Kaggle Dual-GPU Architecture & Static Presets

To respect Kaggle execution limits and GPU quotas, the multi-turn and exploit holdout sets are compiled into self-contained notebooks and distributed across 3 auxiliary Kaggle accounts:

| Evaluation Track          | Shard Directory                              | Assigned Kaggle Account | Shard Evaluation Scope      | Integrity Gate               |
| :------------------------ | :------------------------------------------- | :---------------------- | :-------------------------- | :--------------------------- |
| **Exploit (Track 3)**     | `v3_pipeline/kaggle_eval_exploit/shard_1/`   | Account 1               | 130 dialogues (1,032 turns) | Disjoint ID set, 0 turn cuts |
| **Exploit (Track 3)**     | `v3_pipeline/kaggle_eval_exploit/shard_2/`   | Account 2               | 130 dialogues (1,034 turns) | Disjoint ID set, 0 turn cuts |
| **Exploit (Track 3)**     | `v3_pipeline/kaggle_eval_exploit/shard_3/`   | Account 3               | 130 dialogues (1,029 turns) | Disjoint ID set, 0 turn cuts |
| **Tool Replay (Track 2)** | `v3_pipeline/kaggle_eval_multiturn/shard_1/` | Account 1               | 44 dialogues (148 turns)    | Disjoint ID set, 0 turn cuts |
| **Tool Replay (Track 2)** | `v3_pipeline/kaggle_eval_multiturn/shard_2/` | Account 2               | 43 dialogues (172 turns)    | Disjoint ID set, 0 turn cuts |
| **Tool Replay (Track 2)** | `v3_pipeline/kaggle_eval_multiturn/shard_3/` | Account 3               | 43 dialogues (159 turns)    | Disjoint ID set, 0 turn cuts |

### Self-Contained Notebook Compilation
`build_3way_exploit_notebooks.py` and `build_3way_tool_replay_notebooks.py` generate complete deployment artifacts:
- Embedded zlib/base64 payload containing shard JSONL datasets.
- Automatic installation of dependencies (`transformers`, `peft`, `bitsandbytes`, `accelerate`).
- Direct adapter loading from Hugging Face Hub (`${HF_REPO_ID:-Qwen/Qwen3-8B-offensive-cyber-agent}`) over official base model `Qwen/Qwen3-8B`.
- Generation with sampling parameters: `temperature=0.01` (greedy reproduction), `max_new_tokens=1024`.

---

## 6. Shard Reconciliation Logic

Once Kaggle kernels complete execution, the reconciliation scripts merge distributed outputs:
1. **Download & Extraction**: Downloads kernel output artifacts from all 3 accounts.
2. **Completeness & Disjointness Check**:
   - Asserts total records match expected counts exactly (390 for Exploit, 130 / 479 turns for Tool Replay).
   - Verifies zero dialogue ID duplicates across shards.
3. **Context Length Segmentation**:
   - Scores are partitioned by context tiers: $\le 1024$, $1025-2048$, $2049-3072$, and $3073-4096$ tokens to measure degradation across long contexts.
4. **Hugging Face Hub Deployment**:
   - Automatically commits aggregated JSON reports, markdown summaries, and prediction JSONL files directly to the Hugging Face repository.
