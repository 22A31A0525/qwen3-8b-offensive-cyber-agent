# 🛠️ Multi-Turn Tool Replay Benchmark (v3)

> **Model Under Test**: `qwen3-8b-offensive-cyber-agent`
> **Base Model**: `Qwen/Qwen3-8B`
> **Evaluation Mode**: Teacher-Forced Multi-Turn Tool Replay (`enable_thinking=True`)
> **Dataset**: 130 Holdout Multi-Turn Dialogues (479 Assistant Turns, 241 Tool Calls)
> **Hardware**: Distributed 3-Way Dual Tesla T4 GPUs (6x Tesla T4 Shards)

## 📊 Benchmark Results

| Metric                              | Overall (All 479 Turns) | Standard Context (≤ 4096 tok) | Long Context (> 4096 tok) |
| :---------------------------------- | :---------------------: | :---------------------------: | :-----------------------: |
| **Evaluated Turns**                 |          `479`          |             `438`             |            `41`           |
| **Tool Decision Precision**         |         `0.7738`        |            `0.7782`           |          `0.6923`         |
| **Tool Decision Recall**            |         `0.8864`        |            `0.9347`           |          `0.4286`         |
| **Tool Decision F1**                |         `0.8263`        |            `0.8493`           |          `0.5294`         |
| **Action Concordance Rate**         |         `0.8142`        |            `0.8493`           |          `0.4390`         |
| **No-Tool Adherence Rate**          |         `0.7529`        |            `0.7782`           |          `0.4500`         |
| **Tool Name Accuracy**              |         `0.7500`        |            `0.7889`           |          `0.3810`         |
| **Tool Call Recall**                |         `0.7576`        |            `0.7973`           |          `0.3810`         |
| **Schema Validity (Emitted Calls)** |        `100.00%`        |           `100.00%`           |         `100.00%`         |
| **Argument Key F1**                 |         `0.7499`        |            `0.7888`           |          `0.3810`         |
| **Argument Value Accuracy**         |         `0.5768`        |            `0.6126`           |          `0.2381`         |
| **Full Call Exact Match**           |         `0.5212`        |            `0.5611`           |          `0.1429`         |
| **Mean Content Token F1**           |         `0.6969`        |            `0.7278`           |          `0.3665`         |

## 🔍 Key Findings
1. **Flawless Schema Conformance**: `100.0%` of all emitted tool calls adhere strictly to the JSON schema without syntax or parse errors.
2. **Standard Context Dominance**: On contexts under 4096 tokens (representing 91.4% of all interaction turns), the agent achieves **93.5% Tool Decision Recall**, **85.0% Tool Decision F1**, and **85.0% Action Concordance**.
3. **No-Tool Adherence**: When tools are not needed, the model refrains from calling tools in **78.0%** of turns, preventing tool hallucination.
