# 🧠 LLM-as-a-Judge Benchmark Report: Qwen3-8B Cybersecurity Reasoning

> **Impartial Judge**: `qwen/qwen3.8-27b` via Groq Cloud
> **Benchmark Test Bed**: 130 Pure Cybersecurity Reasoning Dialogues from `data/clean/test/reasoning_20pct.jsonl`
> **Target Production Model**: `Qwen3-8B Offensive Cyber Agent (v3)`

---

## 📊 Benchmark Judgment Matrix

| Metric                       | Score / Observed Value | Evaluation Standard                                                         |
| :--------------------------- | :--------------------: | :-------------------------------------------------------------------------- |
| **Academic Pass Rate**       |      **`44.62%`**      | Factual correctness = True, zero critical errors, concept recall $\ge 0.50$ |
| **Factual Correctness Rate** |      **`50.77%`**      | Technical accuracy validated by independent judge                           |
| **Micro Concept F1**         |      **`0.6142`**      | Semantic harmonic mean of concept precision and recall                      |
| **Micro Concept Precision**  |      **`0.6550`**      | Accuracy of extracted security requirements and commands                    |
| **Micro Concept Recall**     |      **`0.5782`**      | Completeness against reference ground truth requirements                    |
| **Composite Score (40/60)**  |      **`0.5392`**      | 40% deterministic floor + 60% LLM semantic judgment                         |
| **Zero-Tool Compliance**     |      **`100.0%`**      | 0 tool call tags emitted on pure reasoning prompts                          |
| **Total Test Dialogues**     |       **`130`**        | 100% evaluated with zero omitted or dropped turns                           |

---

## 🔬 Evaluation Methodology

1. **Prediction Parsing**: The model generates Chain-of-Thought reasoning within `<think>...</think>` tags followed by the final answer. The evaluation harness parses both segments to verify reasoning validity and output correctness.
2. **Semantic Concept Decomposition**: The judge model breaks reference answers into atomic cybersecurity requirements and matches model assertions conceptually (accounting for valid command flags, syntax variations, and tooling alternatives).
3. **Fail-Closed Verification**: Critical security errors (such as hallucinated CVE identifiers or inverted authorization logic) strictly trigger failure verdicts.
