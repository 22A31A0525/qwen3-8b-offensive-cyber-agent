# Fine-Tuning Engine (`src/finetuning`)

[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen%2FQwen3--8B-blue.svg)](https://huggingface.co/Qwen/Qwen3-8B)
[![Quantization](https://img.shields.io/badge/Quantization-4--bit%20NF4%20(QLoRA)-orange.svg)](#1-architectural-highlights)
[![Attention](https://img.shields.io/badge/Attention-PyTorch%20SDPA-green.svg)](#1-architectural-highlights)
[![Loss Masking](https://img.shields.io/badge/Loss%20Masking-Assistant--Only-purple.svg)](#2-completion-only-loss-collator-collatorpy)

This directory contains the production training scripts, tokenization collators, and ChatML formatting components used for fine-tuning **Qwen3-8B Latest Version (v3)** using 4-bit QLoRA and completion-only loss masking.

---

## 1. Architectural Highlights

| Dimension              | Specification                                                   | Rationale                                                                             |
| :--------------------- | :-------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| **Base Architecture**  | `Qwen/Qwen3-8B`                                                 | High baseline reasoning and native function calling capabilities                      |
| **Quantization**       | 4-bit NormalFloat (NF4), double quant                           | Reduces base model memory from ~16 GB to ~5.5 GB VRAM per GPU                         |
| **Precision**          | Native FP16 (`torch.float16`)                                   | Full compatibility with NVIDIA Turing architecture (Tesla T4)                         |
| **Attention Engine**   | PyTorch Native SDPA (`attn_implementation="sdpa"`)              | FlashAttention-grade memory efficiency without C++ compilation requirements           |
| **Context Window**     | 4,096 Tokens (`max_length=4096`, `packing=False`)               | Zero cross-sample contamination, accommodates 94.27% of all full turns                |
| **LoRA Config**        | Rank $r=16$, Alpha $\alpha=32$, Dropout $0.05$                  | Scaled rank targeting all 7 attention and MLP linear projection layers                |
| **Target Projections** | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Full expressivity adaptation across attention queries/keys/values and MLP gates       |
| **Loss Masking**       | Assistant-Only Completion Loss (`labels = -100` elsewhere)      | Prevents gradient corruption from user prompts, tool schemas, and environment outputs |

---

## 2. Completion-Only Loss Collator (`collator.py`)

In multi-turn offensive cyber and agentic interactions, the training dialogue includes system prompts, external security tool schemas, user requests, and tool execution logs (e.g., shell outputs, network packets). Computing cross-entropy loss over these prompt tokens penalizes the model for text it did not generate and harms tool-calling fidelity.

`CompletionLossDataCollator` enforces strict completion loss masking:

```
[System Prompt / Tools Schema]  -->  labels = -100 (Masked)
[User Instruction]              -->  labels = -100 (Masked)
[Assistant Turn]                -->  labels = input_ids (Active Gradients)
[Tool Observation Return]       -->  labels = -100 (Masked)
[Assistant Next Action]         -->  labels = input_ids (Active Gradients)
[Padding Tokens]                -->  labels = -100 (Masked)
```

### Collator Implementation Details
- **Response Delimiter**: Searches for token sequence representing `<|im_start|>assistant\n`.
- **Instruction Delimiter**: Searches for token sequence representing `<|im_end|>\n`.
- **Dynamic Tensor Padding**: Pads input sequences to the maximum length in the current batch (bounded by `max_length=4096`), aligned to multiples of 8 (`pad_to_multiple_of=8`) for optimal Tensor Core utilization.

---

## 3. ChatML & Tool-Schema Rendering (`render_chatml.py`)

Qwen models parse tool signatures via structured ChatML formats. `render_chatml.py` standardizes all dialogue records into canonical Qwen ChatML representations:

```
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
[
  {
    "type": "function",
    "function": {
      "name": "execute_command",
      "description": "Runs a command in the sandboxed shell.",
      "parameters": { ... }
    }
  }
]
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Scan the target network for open SSH ports.<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "execute_command", "arguments": {"cmd": "nmap -p 22 192.168.1.0/24"}}
</tool_call><|im_end|>
```

### Key Functions:
- `format_tools_xml(tools)`: Serializes tool schema definitions into `<tools>...</tools>` tags.
- `format_message_content(msg)`: Extracts content and converts structured `tool_calls` arrays into `<tool_call>` blocks.
- `render_dialogue_to_chatml(record)`: Assembles complete multi-turn conversations with system schemas.

---

## 4. Multi-GPU Distributed Trainer (`train_v3.py`)

`train_v3.py` implements production distributed fine-tuning across 2x Tesla T4 GPUs via `torchrun`:

### Hyperparameter Configuration

| Parameter                     | Value          | Description                                             |
| :---------------------------- | :------------- | :------------------------------------------------------ |
| `per_device_batch_size`       | `1`            | Batch size per GPU                                      |
| `gradient_accumulation_steps` | `4`            | Gradient accumulation steps                             |
| `world_size`                  | `2` (Dual GPU) | Number of distributed processes                         |
| `effective_batch_size`        | `8`            | Sequences per optimization step ($1 \times 4 \times 2$) |
| `learning_rate`               | `1e-4`         | Peak learning rate                                      |
| `warmup_ratio`                | `0.05`         | Linear warmup fraction (5% of total optimizer steps)    |
| `lr_scheduler`                | `cosine`       | Cosine decay schedule to zero                           |
| `weight_decay`                | `0.01`         | AdamW weight decay                                      |
| `max_grad_norm`               | `1.0`          | Gradient clipping threshold                             |
| `epochs`                      | `1`            | Curriculum training epochs                              |

### Resumable Checkpointing & Fault Tolerance
Checkpoints persist:
1. LoRA adapter weights (`adapter_model.safetensors`, `adapter_config.json`)
2. Optimizer and learning rate scheduler states
3. Random number generator states (`torch`, `numpy`, `random`)
4. Pipeline telemetry: `training_state.json` recording global step, epoch, and elapsed time.

### Execution Command:
```bash
torchrun --nproc_per_node=2 src/finetuning/train_v3.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --data_dir data/clean \
    --output_dir v3_pipeline/outputs/v3_lora_run \
    --require_ddp
```
