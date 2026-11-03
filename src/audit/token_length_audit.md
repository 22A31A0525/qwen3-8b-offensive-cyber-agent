# Qwen3-8B (v3 Pipeline) -- Pre-Training Token Length Audit Report

## 1. Executive Summary

This token length audit measures the distribution of token lengths across the decontaminated **`v3_pipeline/data_v3_clean/`** dataset after canonical Qwen3 ChatML formatting and `<tools>` XML schema rendering.

- **Tokenizer Employed**: `Qwen/Qwen2.5-7B-Instruct`
- **Context Length Gate**: **4,096 tokens** (Kaggle Dual Tesla T4 VRAM budget limit)
- **Target Truncation Tolerance**: $\le 3.0\%$ truncation rate across training records.

---

## 2. Split-Level Token Length Distribution

| Split       | Records | Min |   Mean | Median (P50) |    P90 |    P95 |    P99 |    Max |  % > 4096 |
| :---------- | ------: | --: | -----: | -----------: | -----: | -----: | -----: | -----: | --------: |
| **`train`** |   5,200 |  36 | 1965.2 |       2064.0 | 3569.7 | 4213.6 | 5915.4 | 18,403 | **5.73%** |
| **`val`**   |     650 |  50 | 1991.6 |       2059.5 | 3743.5 | 4350.6 | 5654.1 |  9,132 | **7.23%** |
| **`test`**  |     650 |  42 | 1959.3 |       2051.5 | 3645.3 | 4277.8 | 5230.1 | 12,690 | **5.85%** |

---

## 3. Training Curriculum Breakdown by Domain (5,200 Records)

| Curriculum Domain     | Records | Mean Tokens | Median Tokens | P90 Tokens | Max Tokens | Truncated (>4096) |
| :-------------------- | ------: | ----------: | ------------: | ---------: | ---------: | ----------------: |
| `reasoning`           |   1,040 |       124.3 |         103.0 |      229.0 |        387 |         0 (0.00%) |
| `exploit`             |   3,120 |      2332.5 |        2238.0 |     3007.0 |      7,319 |        59 (1.89%) |
| `tool_call_replay`    |   1,039 |      2705.1 |        3129.0 |     4941.0 |     18,403 |      239 (23.00%) |
| `tool_calling_replay` |       1 |      1935.0 |        1935.0 |     1935.0 |      1,935 |         0 (0.00%) |

---

## 4. Hardware Gate Verdict for Dual Tesla T4 Training

| Evaluation Metric            |    Observed Value   |     Hard Constraint      | Gate Verdict |
| :--------------------------- | :-----------------: | :----------------------: | :----------: |
| **Training Truncation Rate** |      **5.73%**      |       $\le 3.00\%$       |  **REVIEW**  |
| **P90 Context Coverage**     |  **3569.7 tokens**  |     $< 4,096$ tokens     |   **PASS**   |
| **Tool Schema Injection**    | **100.0% Rendered** |     100.0% coverage      |   **PASS**   |
| **Loss Masking Collator**    |  **Assistant Only** | System/Tools/User = -100 |   **PASS**   |
