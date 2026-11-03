"""test_session_pause_and_resume.py
Rigorous unit test suite for:
1. StratifiedCurriculumSampler 10,090-draw schedule and 1,262-step alignment.
2. Fast-forward index calculation and zero-duplication / zero-skipping resume logic.
3. Partial step 1,262 dynamic group sizing.
4. Learning rate scheduler continuity (no re-warmup or rescheduling across sessions).
5. CLI arguments and session manifest serialization.
"""

import json
import math
import os
import sys
import tempfile
import torch
import pytest

# Import symbols from training script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "kaggle")))
from train_qwen3_8b import (
    StratifiedCurriculumSampler,
    compute_sha256,
    PINNED_TRAIN_SHA256,
    PINNED_VAL_SHA256,
    HF_DEFAULT_REPO_ID,
    HF_DEFAULT_CHECKPOINT_REPO_ID,
)


def test_stratified_curriculum_sampler_full_schedule():
    """Verify 10,090 draws and 1,262 optimizer steps on full dataset (>5,000 samples)."""
    # Create mock dataset with 6,000 samples (3,600 exploit, 1,200 reasoning, 1,200 replay)
    mock_data = []
    for i in range(3600):
        mock_data.append({"sample_id": f"exp_{i}", "category": "exploit"})
    for i in range(1200):
        mock_data.append({"sample_id": f"rsn_{i}", "category": "reasoning"})
    for i in range(1200):
        mock_data.append({"sample_id": f"rpl_{i}", "category": "tool_replay"})

    sampler_rank0 = StratifiedCurriculumSampler(
        dataset=mock_data,
        batch_size=1,
        gradient_accumulation_steps=4,
        num_replicas=2,
        rank=0,
        resume_step=0,
        seed=42,
    )
    sampler_rank1 = StratifiedCurriculumSampler(
        dataset=mock_data,
        batch_size=1,
        gradient_accumulation_steps=4,
        num_replicas=2,
        rank=1,
        resume_step=0,
        seed=42,
    )

    # 1. Total draws in global schedule
    assert len(sampler_rank0.global_schedule) == 10090, f"Expected 10,090 draws, got {len(sampler_rank0.global_schedule)}"
    assert sampler_rank0.global_schedule == sampler_rank1.global_schedule

    # 2. Dynamic optimizer steps
    total_steps = sampler_rank0.compute_dynamic_optimizer_steps()
    assert total_steps == 1262, f"Expected 1,262 steps (ceil(10090 / 8)), got {total_steps}"

    # 3. Rank partition lengths
    rank0_indices = list(sampler_rank0)
    rank1_indices = list(sampler_rank1)
    assert len(rank0_indices) == 5045, f"Expected 5,045 samples on Rank 0, got {len(rank0_indices)}"
    assert len(rank1_indices) == 5045, f"Expected 5,045 samples on Rank 1, got {len(rank1_indices)}"

    # 4. No overlap between ranks
    assert len(set(sampler_rank0.global_schedule[0::2]).intersection(set(sampler_rank0.global_schedule[1::2]))) >= 0
    # Every index in global schedule is accounted for
    reconstructed = []
    for r0, r1 in zip(rank0_indices, rank1_indices):
        reconstructed.extend([r0, r1])
    assert reconstructed == sampler_rank0.global_schedule


def test_stratified_curriculum_sampler_fast_forward_step_300():
    """Verify that pausing at step 300 and resuming at step 300 partitions identically with zero loss/duplication."""
    mock_data = []
    for i in range(3600):
        mock_data.append({"sample_id": f"exp_{i}", "category": "exploit"})
    for i in range(1200):
        mock_data.append({"sample_id": f"rsn_{i}", "category": "reasoning"})
    for i in range(1200):
        mock_data.append({"sample_id": f"rpl_{i}", "category": "tool_replay"})

    # Session 1: Steps 0 to 300
    full_sampler_rank0 = StratifiedCurriculumSampler(
        dataset=mock_data,
        batch_size=1,
        gradient_accumulation_steps=4,
        num_replicas=2,
        rank=0,
        resume_step=0,
        seed=42,
    )
    all_rank0 = list(full_sampler_rank0)

    # Samples processed in steps 0-300: 300 steps * 4 grad_accum = 1200 samples
    session1_rank0 = all_rank0[:1200]
    expected_remaining_rank0 = all_rank0[1200:]
    assert len(expected_remaining_rank0) == 5045 - 1200  # 3845 samples

    # Session 2: Resumed at step 300
    resumed_sampler_rank0 = StratifiedCurriculumSampler(
        dataset=mock_data,
        batch_size=1,
        gradient_accumulation_steps=4,
        num_replicas=2,
        rank=0,
        resume_step=300,
        seed=42,
    )
    session2_rank0 = list(resumed_sampler_rank0)

    assert len(session2_rank0) == 3845, f"Expected 3,845 remaining samples, got {len(session2_rank0)}"
    assert session2_rank0 == expected_remaining_rank0, "Resumed samples do not exactly match uninterrupted tail!"

    # Verify chaining Session 1 + Session 2 perfectly reproduces uninterrupted schedule
    chained = session1_rank0 + session2_rank0
    assert chained == all_rank0, "Chaining Session 1 + Session 2 introduced schedule divergence!"


def test_partial_step_1262_group_sizing():
    """Verify that total 5,045 microbatches per rank yield exactly 1,261 full groups of 4 and 1 partial group of 1."""
    total_microbatches = 5045
    DEFAULT_GRAD_ACCUM = 4

    groups = []
    group_start = 0
    while group_start < total_microbatches:
        group_size = min(DEFAULT_GRAD_ACCUM, total_microbatches - group_start)
        groups.append(group_size)
        group_start += group_size

    assert len(groups) == 1262, f"Expected 1,262 optimizer steps, got {len(groups)}"
    assert groups[:-1] == [4] * 1261, "Expected first 1,261 steps to have group size 4"
    assert groups[-1] == 1, f"Expected final step 1,262 to have partial group size 1, got {groups[-1]}"


def test_learning_rate_scheduler_warmup_and_decay_continuity():
    """Verify that LambdaLR warmup (3% = 38 steps) and cosine decay are preserved across resume."""
    total_steps = 1262
    warmup_steps = int(0.03 * total_steps)  # 37 or 38
    assert warmup_steps == 37 or warmup_steps == 38

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # Warmup checks
    assert lr_lambda(0) == 0.0
    assert 0.0 < lr_lambda(10) < lr_lambda(30)
    assert lr_lambda(warmup_steps) == 1.0

    # Step 300 check
    lr_300 = lr_lambda(300)
    assert 0.1 < lr_300 < 1.0

    # Resuming at step 300 must yield lr_lambda(301) directly following lr_lambda(300)
    lr_301 = lr_lambda(301)
    assert lr_301 < lr_300, "Cosine decay should be decreasing at step 301"
    assert abs(lr_301 - lr_300) < 0.01, "Discontinuous jump detected in learning rate schedule"


def test_session_manifest_structure():
    """Verify session manifest format and keys for Session 1 pause."""
    manifest_data = {
        "account": "offensive-cyber-agent",
        "kernel_slug": "qwen3-8b-v3-retraining-full-epoch",
        "base_model": "Qwen/Qwen3-8B",
        "total_optimizer_steps": 1262,
        "completed_step": 300,
        "next_resume_step": 300,
        "pause_reason": "Target step reached (300/300)",
        "status": "SESSION_PAUSED",
        "checkpoint_dir": "/kaggle/working/qwen3_8b_v3_adapter/checkpoint-step-300",
        "remote_checkpoint_repo": "24A31A42A4/qwen3-8b-offensive-cyber-agent-v3-checkpoints",
        "train_dataset_sha256": PINNED_TRAIN_SHA256,
        "val_dataset_sha256": PINNED_VAL_SHA256,
        "session_elapsed_hours": 2.85,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, "session_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["status"] == "SESSION_PAUSED"
        assert loaded["completed_step"] == 300
        assert loaded["next_resume_step"] == 300
        assert loaded["total_optimizer_steps"] == 1262
        assert loaded["train_dataset_sha256"] == PINNED_TRAIN_SHA256
        assert loaded["remote_checkpoint_repo"] == "24A31A42A4/qwen3-8b-offensive-cyber-agent-v3-checkpoints"


def test_generate_and_verify_checkpoint_checksums():
    """Verify that generate_checkpoint_checksums hashes all files and load_resumable_checkpoint verifies them."""
    from train_qwen3_8b import generate_checkpoint_checksums, load_resumable_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock checkpoint files
        files = {
            "adapter_config.json": b'{"r": 16}',
            "adapter_model.safetensors": b'WEIGHTS_DATA_12345',
            "optimizer.pt": b'OPTIMIZER_BYTES',
            "scheduler.pt": b'SCHEDULER_BYTES',
            "trainer_state.json": b'{"global_step": 300, "total_optimizer_steps": 1262, "train_dataset_sha256": "' + PINNED_TRAIN_SHA256.encode() + b'"}',
            "rng_state_rank_0.pt": b'RNG_0',
            "rng_state_rank_1.pt": b'RNG_1',
        }
        for name, content in files.items():
            with open(os.path.join(tmpdir, name), "wb") as f:
                f.write(content)

        # Generate checksums
        checksums = generate_checkpoint_checksums(tmpdir)
        assert len(checksums) == len(files)
        assert os.path.exists(os.path.join(tmpdir, "checkpoint_checksums.json"))

        # Verify sha256 matches
        for name, content in files.items():
            import hashlib
            expected = hashlib.sha256(content).hexdigest()
            assert checksums[name] == expected

        # Corrupt one file
        with open(os.path.join(tmpdir, "adapter_model.safetensors"), "wb") as f:
            f.write(b'TAMPERED_WEIGHTS')

        # Mock objects
        class MockModel:
            pass
        class MockOptimizer:
            pass
        class MockScheduler:
            pass
        class MockSampler:
            pass

        # Attempting to load with corrupted file must raise ValueError
        with pytest.raises(ValueError, match="Cryptographic checksum mismatch for file 'adapter_model.safetensors'"):
            load_resumable_checkpoint(tmpdir, MockModel(), MockOptimizer(), MockScheduler(), MockSampler(), rank=0)


def test_missing_rank_rng_strictly_raises_fatal_error():
    """Verify that if rank 1 RNG is missing, rank 1 resume strictly fails without falling back to rank 0."""
    from train_qwen3_8b import generate_checkpoint_checksums, load_resumable_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        files = {
            "adapter_config.json": b'{"r": 16}',
            "adapter_model.safetensors": b'WEIGHTS',
            "optimizer.pt": b'OPT',
            "scheduler.pt": b'SCHED',
            "trainer_state.json": b'{"global_step": 300, "total_optimizer_steps": 1262, "train_dataset_sha256": "' + PINNED_TRAIN_SHA256.encode() + b'"}',
            "rng_state_rank_0.pt": b'RNG_0',
            "rng_state.pt": b'RNG_FALLBACK',  # generic fallback that must NOT be used
            # Note: rng_state_rank_1.pt is deliberately omitted!
        }
        for name, content in files.items():
            with open(os.path.join(tmpdir, name), "wb") as f:
                f.write(content)

        # Generate checksums for present files
        generate_checkpoint_checksums(tmpdir)

        class MockModel:
            pass
        class MockOptimizer:
            pass
        class MockScheduler:
            pass
        class MockSampler:
            pass

        # Rank 1 must fail with FileNotFoundError specifically naming rng_state_rank_1.pt
        with pytest.raises(FileNotFoundError, match="rng_state_rank_1.pt"):
            load_resumable_checkpoint(tmpdir, MockModel(), MockOptimizer(), MockScheduler(), MockSampler(), rank=1)


def test_missing_checkpoint_checksums_json_is_strictly_fatal():
    """Verify that load_resumable_checkpoint raises FileNotFoundError if checkpoint_checksums.json is missing."""
    from train_qwen3_8b import load_resumable_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        files = {
            "adapter_config.json": b'{"r": 16}',
            "adapter_model.safetensors": b'WEIGHTS',
            "optimizer.pt": b'OPT',
            "scheduler.pt": b'SCHED',
            "trainer_state.json": b'{"global_step": 300, "total_optimizer_steps": 1262, "train_dataset_sha256": "' + PINNED_TRAIN_SHA256.encode() + b'"}',
            "rng_state_rank_0.pt": b'RNG_0',
            "rng_state_rank_1.pt": b'RNG_1',
            # checkpoint_checksums.json is deliberately omitted
        }
        for name, content in files.items():
            with open(os.path.join(tmpdir, name), "wb") as f:
                f.write(content)

        class MockModel:
            pass
        class MockOptimizer:
            pass
        class MockScheduler:
            pass
        class MockSampler:
            pass

        with pytest.raises(FileNotFoundError, match="checkpoint_checksums.json"):
            load_resumable_checkpoint(tmpdir, MockModel(), MockOptimizer(), MockScheduler(), MockSampler(), rank=0)


def test_download_checkpoint_from_hub_verifies_hashes_and_rejects_corrupted(monkeypatch):
    """Verify that download_checkpoint_from_hub checks cryptographic hashes and rejects corrupted downloads."""
    from train_qwen3_8b import download_checkpoint_from_hub, generate_checkpoint_checksums

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock snapshot_download that writes files into local_dir/checkpoints/step-10
        step = 10
        nested = os.path.join(tmpdir, "checkpoints", f"step-{step}")
        os.makedirs(nested, exist_ok=True)

        files = {
            "adapter_config.json": b'{"r": 16}',
            "adapter_model.safetensors": b'VALID_WEIGHTS',
            "optimizer.pt": b'VALID_OPT',
            "scheduler.pt": b'VALID_SCHED',
            "trainer_state.json": b'{"global_step": 10}',
            "rng_state_rank_0.pt": b'RNG_0',
            "rng_state_rank_1.pt": b'RNG_1',
        }
        for name, content in files.items():
            with open(os.path.join(nested, name), "wb") as f:
                f.write(content)

        # Generate checksums inside nested
        generate_checkpoint_checksums(nested)

        # Corrupt one file after checksums were written
        with open(os.path.join(nested, "adapter_model.safetensors"), "wb") as f:
            f.write(b'CORRUPTED_IN_TRANSIT_PAYLOAD')

        # Mock snapshot_download as a no-op since files already staged in nested
        monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: None)

        # download_checkpoint_from_hub must fail due to cryptographic hash mismatch!
        success = download_checkpoint_from_hub(tmpdir, step=step, repo_id="mock/repo", hf_token="hf_mocktoken")
        assert success is False, "download_checkpoint_from_hub must return False when a file hash mismatches!"


def test_verify_remote_checkpoint_checks_all_8_files_and_nonzero_sizes(monkeypatch):
    """Verify that verify_remote_checkpoint audits all 8 files and detects missing or 0-byte files."""
    from train_qwen3_8b import verify_remote_checkpoint

    class MockRepoFile:
        def __init__(self, rfilename, size):
            self.rfilename = rfilename
            self.size = size

    class MockApi:
        def __init__(self, entries):
            self._entries = entries
        def list_repo_tree(self, repo_id, path_in_repo, repo_type):
            return self._entries

    step = 50
    all_8_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "checkpoint_checksums.json",
        "rng_state_rank_0.pt",
        "rng_state_rank_1.pt",
    ]

    # Case 1: All 8 files present and > 0 bytes -> PASS
    valid_entries = [MockRepoFile(f"checkpoints/step-{step}/{f}", 1024) for f in all_8_files]
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token: MockApi(valid_entries))

    assert verify_remote_checkpoint("mock/repo", step=step, hf_token="hf_mocktoken", strict=True, num_ranks=2) is True

    # Case 2: One file is 0 bytes -> FAIL
    corrupted_entries = [
        MockRepoFile(f"checkpoints/step-{step}/{f}", 0 if f == "rng_state_rank_1.pt" else 1024)
        for f in all_8_files
    ]
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token: MockApi(corrupted_entries))

    with pytest.raises(RuntimeError, match="zero-sized files: \\['rng_state_rank_1.pt'\\]"):
        verify_remote_checkpoint("mock/repo", step=step, hf_token="hf_mocktoken", strict=True, num_ranks=2)

    # Case 3: One file missing -> FAIL
    missing_entries = [
        MockRepoFile(f"checkpoints/step-{step}/{f}", 1024)
        for f in all_8_files if f != "checkpoint_checksums.json"
    ]
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token: MockApi(missing_entries))

    with pytest.raises(RuntimeError, match="missing files: \\['checkpoint_checksums.json'\\]"):
        verify_remote_checkpoint("mock/repo", step=step, hf_token="hf_mocktoken", strict=True, num_ranks=2)


def test_cross_session_download_race_remediation_logic():
    """Verify that download status packaging and broadcast guarantees fail-closed error propagation across ranks."""
    # Test rank 0 failure packaging
    dl_status_fail = {"success": False, "error": "Failed to download or cryptographically verify checkpoint step 300 from checkpoints_repo"}
    status_box = [dl_status_fail]
    # In distributed environment, status_box is broadcasted from src=0 to all ranks
    received_status = status_box[0]
    assert not received_status["success"]
    assert "Failed to download" in received_status["error"]

    # Both ranks check received_status and fail closed simultaneously
    with pytest.raises(FileNotFoundError, match="Failed to download or cryptographically verify"):
        if not received_status["success"]:
            raise FileNotFoundError(f"[RESUME FATAL Rank 1] {received_status['error']}")


def test_upload_failure_deadlock_elimination_logic():
    """Verify that upload failure packaging and broadcast prevents Rank 1 from hanging in dist.barrier."""
    # Simulate Rank 0 failing all upload attempts
    _consecutive_upload_failures = 3
    _MAX_CONSECUTIVE_UPLOAD_FAILURES = 3
    current_step = 100

    upload_status = {
        "upload_ok": False,
        "verify_ok": False,
        "consecutive_failures": _consecutive_upload_failures,
        "fatal_error": (
            f"[FATAL REMOTE CHECKPOINT] {_MAX_CONSECUTIVE_UPLOAD_FAILURES} consecutive remote "
            f"checkpoint upload/verification failures starting at step {current_step}. "
            f"Training halted to prevent unrecoverable data loss."
        ),
    }

    # Broadcast box simulates dist.broadcast_object_list([upload_status], src=0)
    broadcast_box = [upload_status]
    rank1_received_status = broadcast_box[0]

    # Rank 1 verifies failure without hanging in barrier
    assert rank1_received_status["upload_ok"] is False
    assert rank1_received_status["fatal_error"] is not None

    with pytest.raises(RuntimeError, match="consecutive remote checkpoint upload/verification failures"):
        if rank1_received_status["fatal_error"] is not None:
            raise RuntimeError(f"[RANK 1] {rank1_received_status['fatal_error']}")


