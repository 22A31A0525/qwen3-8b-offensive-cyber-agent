# Qwen3-8B Offensive Cyber Agent -- Overall Tri-Track Benchmark Report

## 1. Executive Summary

This report documents the **Weighted Overall Benchmark Performance** for **Qwen3-8B Offensive Cyber Agent (v3)** across the curriculum holdout evaluation suite.

Evaluation is weighted according to the strict curriculum ratio defined in the training architecture:
- **60% Exploit Trajectories** (Track 1 -- Multiturn cyber attack workflows)
- **20% Cybersecurity Reasoning** (Track 2 -- LLM-as-a-Judge conceptual & factual verification)
- **20% Multi-Turn Tool Replay** (Track 3 -- Function-calling replay & schema adherence)

---

## 2. Overall Weighted Metrics Summary

| Evaluation Track               | Curriculum Weight | Precision  |   Recall   |  F1 Score  | Evaluation Standard                            |
| :----------------------------- | :---------------: | :--------: | :--------: | :--------: | :--------------------------------------------- |
| **Exploit Trajectories**       |        60%        |   97.60%   |   76.52%   |   85.79%   | Teacher-forced multi-turn trajectory execution |
| **Cybersecurity Reasoning**    |        20%        |   65.50%   |   57.82%   |   61.42%   | LLM-as-a-Judge semantic concept decomposition  |
| **Multi-Turn Tool Replay**     |        20%        |   77.38%   |   88.64%   |   82.63%   | Pure confusion matrix tool-decision evaluation |
| **Weighted Overall Benchmark** |      **100%**     | **87.14%** | **75.20%** | **80.28%** | **Composite Tri-Track Curriculum Aggregate**   |

---

## 3. Mathematical Formula & Step-by-Step Breakdown

### Weighted Precision (87.14%)
$$\text{Weighted Precision} = (0.60 \times 97.60) + (0.20 \times 65.50) + (0.20 \times 77.38)$$
$$\text{Weighted Precision} = 58.56 + 13.10 + 15.476 = \mathbf{87.14\%}$$

### Weighted Recall (75.20%)
$$\text{Weighted Recall} = (0.60 \times 76.52) + (0.20 \times 57.82) + (0.20 \times 88.64)$$
$$\text{Weighted Recall} = 45.912 + 11.564 + 17.728 = \mathbf{75.20\%}$$

### Weighted F1 Score (80.28%)
$$\text{Weighted F1} = (0.60 \times 85.79) + (0.20 \times 61.42) + (0.20 \times 82.63)$$
$$\text{Weighted F1} = 51.474 + 12.284 + 16.526 = \mathbf{80.28\%}$$

---

## 4. Track Details & Holdout Counts

| Track Identifier       | Holdout Dialogues | Total Turns / Samples | Primary Metric Description                                            |
| :--------------------- | :---------------: | :-------------------: | :-------------------------------------------------------------------- |
| `results/exploit/`     |   390 dialogues   |      3,095 turns      | Precision on valid exploit tool calls; Adaptive Pivot Rate: 100%      |
| `results/reasoning/`   |   130 dialogues   |      130 samples      | Concept precision & recall against gold references; Pass Rate: 44.62% |
| `results/tool_replay/` |   130 dialogues   |       479 turns       | Tool decision precision, recall, and strict schema compliance: 100%   |
| **Total Benchmark**    | **650 dialogues** |    **3,704 turns**    | **Full curriculum coverage with zero cross-split test leakage**       |

---

## 5. Provenance & Artifact Pointers

- **Exploit Benchmark Details**: [`results/exploit/exploit_benchmark_report.md`](exploit/exploit_benchmark_report.md)
- **Reasoning Benchmark Details**: [`results/reasoning/reasoning_judge_report.md`](reasoning/reasoning_judge_report.md)
- **Tool Replay Benchmark Details**: [`results/tool_replay/tool_replay_benchmark.md`](tool_replay/tool_replay_benchmark.md)
- **Machine-Readable Summary**: [`results/overall_summary.json`](overall_summary.json)
