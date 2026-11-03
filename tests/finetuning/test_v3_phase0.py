"""Unit and Integration Tests for v3 Phase 0 Technical Gates.

Enforces:
1. Guaranteed <tools> XML schema injection into system prompt (100% of tool-calling dialogues).
2. Pure assistant-only completion loss masking (labels = -100 on system, tools, user, and tool turns).
3. Non-tool dialogues omit <tools> block.
4. Token length auditor contract.
"""
import json
import pytest
from typing import Any, Dict, List

from src.finetuning.render_chatml import (
    format_tools_xml,
    render_dialogue_to_chatml,
    mask_labels_for_assistant_only,
)
from src.finetuning.collator import CompletionLossDataCollator


class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.vocab = {
            "<|im_start|>assistant\n": [101, 102],
            "<|im_end|>\n": [201],
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if text in self.vocab:
            return self.vocab[text]
        return [ord(c) for c in text]


def test_completion_loss_data_collator():
    tok = MockTokenizer()
    collator = CompletionLossDataCollator(tokenizer=tok, max_length=64, pad_to_multiple_of=8)

    # Construct input_ids: System [1, 2, 3] + Resp [101, 102] + Assistant [50, 51, 52] + End [201] + User [4, 5]
    seq = [1, 2, 3, 101, 102, 50, 51, 52, 201, 4, 5]
    batch = collator([{"input_ids": seq}])

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    assert input_ids.shape[0] == 1
    assert input_ids.shape[1] % 8 == 0  # padded to multiple of 8
    assert attention_mask[0, : len(seq)].sum().item() == len(seq)
    assert attention_mask[0, len(seq) :].sum().item() == 0

    # Labels:
    # 0, 1, 2 (system) -> -100
    # 3, 4 (resp header 101, 102) -> -100
    # 5, 6, 7 (assistant tokens 50, 51, 52) -> 50, 51, 52
    # 8 (end token 201) -> 201
    # 9, 10 (user tokens 4, 5) -> -100
    # 11+ (padding) -> -100
    lbl_list = labels[0].tolist()
    assert lbl_list[0] == -100
    assert lbl_list[1] == -100
    assert lbl_list[2] == -100
    assert lbl_list[3] == -100
    assert lbl_list[4] == -100
    assert lbl_list[5] == 50
    assert lbl_list[6] == 51
    assert lbl_list[7] == 52
    assert lbl_list[8] == 201
    assert lbl_list[9] == -100
    assert lbl_list[10] == -100
    assert lbl_list[11] == -100


def test_stratified_curriculum_sampler_ratio():
    from src.data.sampler import StratifiedCurriculumSampler

    # Mock dataset with 300 exploit, 100 reasoning, 100 replay (500 total)
    mock_dataset = []
    for i in range(300):
        mock_dataset.append({"id": f"exp_{i}", "category": "exploit"})
    for i in range(100):
        mock_dataset.append({"id": f"rsn_{i}", "category": "reasoning"})
    for i in range(100):
        mock_dataset.append({"id": f"rpl_{i}", "category": "tool_call_replay"})

    sampler = StratifiedCurriculumSampler(mock_dataset, seed=42)
    indices = list(sampler)

    assert len(indices) == 500

    # Inspect first 50 samples: exactly 30 exploit, 10 reasoning, 10 replay
    first_50 = [mock_dataset[idx]["category"] for idx in indices[:50]]
    assert first_50.count("exploit") == 30
    assert first_50.count("reasoning") == 10
    assert first_50.count("tool_call_replay") == 10

    # In every group of 5: exactly 3 exploit, 1 reasoning, 1 replay
    for chunk_start in range(0, 50, 5):
        chunk = [mock_dataset[idx]["category"] for idx in indices[chunk_start : chunk_start + 5]]
        assert chunk.count("exploit") == 3
        assert chunk.count("reasoning") == 1
        assert chunk.count("tool_call_replay") == 1


def test_stratified_curriculum_sampler_ddp_and_resume():
    from src.data.sampler import StratifiedCurriculumSampler

    mock_dataset = []
    for i in range(300):
        mock_dataset.append({"id": f"exp_{i}", "category": "exploit"})
    for i in range(100):
        mock_dataset.append({"id": f"rsn_{i}", "category": "reasoning"})
    for i in range(100):
        mock_dataset.append({"id": f"rpl_{i}", "category": "tool_call_replay"})

    # Test DDP Rank 0 vs Rank 1 disjointness
    sampler_rank0 = StratifiedCurriculumSampler(mock_dataset, num_replicas=2, rank=0, seed=42)
    sampler_rank1 = StratifiedCurriculumSampler(mock_dataset, num_replicas=2, rank=1, seed=42)

    indices_rank0 = list(sampler_rank0)
    indices_rank1 = list(sampler_rank1)

    assert len(indices_rank0) == 250
    assert len(indices_rank1) == 250
    # Zero overlap between ranks
    intersection = set(indices_rank0).intersection(set(indices_rank1))
    assert len(intersection) == 0, f"DDP ranks overlap: {len(intersection)} indices in common!"
    # Union covers all 500
    union = set(indices_rank0).union(set(indices_rank1))
    assert len(union) == 500

    # Test dynamic step calculation
    # 500 samples, 2 replicas, batch 1, grad_accum 4 -> effective batch = 8
    # 500 / 8 = 62.5 -> 63 steps
    dyn_steps = sampler_rank0.compute_dynamic_optimizer_steps()
    assert dyn_steps == 63

    # Test resume step offset
    sampler_resumed = StratifiedCurriculumSampler(
        mock_dataset,
        num_replicas=2,
        rank=0,
        gradient_accumulation_steps=4,
        resume_step=10,
        seed=42,
    )
    resumed_indices = list(sampler_resumed)
    # 10 steps * batch 1 * grad_accum 4 = 40 samples skipped
    assert len(resumed_indices) == 250 - 40
    assert resumed_indices == indices_rank0[40:]





@pytest.fixture
def sample_tool_record() -> Dict[str, Any]:
    return {
        "id": "test_sample_tool_01",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "nmap_scan",
                    "description": "Port scanner",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string", "description": "Target IP or domain"}
                        },
                        "required": ["target"]
                    }
                }
            }
        ],
        "messages": [
            {"role": "system", "content": "You are an autonomous offensive cybersecurity penetration testing agent."},
            {"role": "user", "content": "Scan 192.168.1.100 for open ports."},
            {
                "role": "assistant",
                "content": "<tool_call>\n{\"name\": \"nmap_scan\", \"arguments\": {\"target\": \"192.168.1.100\"}}\n</tool_call>"
            },
            {
                "role": "tool",
                "content": "{\"ports\": [22, 80, 443]}"
            },
            {
                "role": "assistant",
                "content": "Ports 22, 80, and 443 are open on 192.168.1.100."
            }
        ],
        "metadata": {"category": "tool_calling_replay", "source_dataset": "hermes-function-calling-v1"}
    }


@pytest.fixture
def sample_reasoning_record() -> Dict[str, Any]:
    return {
        "id": "test_sample_reasoning_01",
        "tools": None,
        "messages": [
            {"role": "system", "content": "You are a cybersecurity reasoning expert."},
            {"role": "user", "content": "Explain what a return-to-libc attack is."},
            {"role": "assistant", "content": "A return-to-libc attack bypasses non-executable stack protections (NX/DEP)..."}
        ],
        "metadata": {"category": "cybersecurity_reasoning", "source_dataset": "cybersecurity_reasoning"}
    }


def test_format_tools_xml():
    tools = [{"type": "function", "function": {"name": "test_fn"}}]
    xml = format_tools_xml(tools)
    assert "<tools>" in xml
    assert "</tools>" in xml
    assert "test_fn" in xml


def test_tool_schema_rendering_with_existing_system_turn(sample_tool_record):
    rendered = render_dialogue_to_chatml(sample_tool_record)
    assert "<|im_start|>system" in rendered
    assert "You are an autonomous offensive cybersecurity" in rendered
    assert "<tools>" in rendered
    assert "nmap_scan" in rendered
    assert "</tools>" in rendered
    assert "<|im_end|>" in rendered


def test_tool_schema_rendering_without_system_turn():
    record = {
        "id": "test_no_sys",
        "tools": [{"type": "function", "function": {"name": "ping"}}],
        "messages": [
            {"role": "user", "content": "Ping target"},
            {"role": "assistant", "content": "<tool_call>\n{\"name\": \"ping\"}\n</tool_call>"}
        ]
    }
    rendered = render_dialogue_to_chatml(record)
    assert rendered.startswith("<|im_start|>system\n")
    assert "<tools>" in rendered
    assert "ping" in rendered
    assert "</tools>" in rendered


def test_non_tool_dialogue_omits_tools_block(sample_reasoning_record):
    rendered = render_dialogue_to_chatml(sample_reasoning_record)
    assert "<tools>" not in rendered
    assert "</tools>" not in rendered
    assert "# Tools" not in rendered
    assert "<|im_start|>system\nYou are a cybersecurity reasoning expert.<|im_end|>" in rendered


def test_assistant_only_completion_loss_masking(sample_tool_record):
    rendered = render_dialogue_to_chatml(sample_tool_record)
    # Mock tokenizer simulation using space splitting
    tokens = rendered.split(" ")
    labels = mask_labels_for_assistant_only(tokens, rendered)

    assert len(labels) == len(tokens)
    # The first token is system start, which must have label -100
    assert labels[0] == -100

    # Ensure at least some assistant tokens have active labels != -100
    active_tokens = [tok for tok, lab in zip(tokens, labels) if lab != -100]
    assert len(active_tokens) > 0
    # Assistant text must be in active tokens
    assert any("443" in tok or "open" in tok for tok in active_tokens)


def test_end_to_end_loss_masking_with_qwen3_tokenizer():
    from v3_pipeline.scripts.verify_loss_masking import run_verification
    assert run_verification() is True


def test_turn_boundary_chunking_and_schema_preservation():
    from v3_pipeline.scripts.verify_chunking import test_chunking_end_to_end
    assert test_chunking_end_to_end() is True
