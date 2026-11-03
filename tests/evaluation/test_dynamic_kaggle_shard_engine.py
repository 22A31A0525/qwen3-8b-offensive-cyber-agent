"""
Unit Tests for Dynamic Kaggle Sharding Engine
=============================================
Verifies:
1. Dynamic Shard Count & Account Derivation:
   - Explicit num_shards (e.g. 3, 5)
   - Auto-derivation from KAGGLE_ACCOUNTS environment variable via --from-env
   - Custom account lists and automatic account name padding
2. Shard Splitting Logic:
   - Dialogue-level preservation (no fragmented turns/dialogues)
   - Pairwise disjointness across shards
   - Lossless dialogue ID roundtrip and exact partition totals
3. Dynamic Notebook & Metadata Generation:
   - Valid Jupyter Notebook structure (nbformat v4) for exploit, tool_replay, reasoning
   - AST syntax check passes cleanly for EVERY code cell in the generated notebooks
   - Extracted worker script parses cleanly with zero SyntaxErrors
   - Lossless payload decompression and JSON integrity check
   - Kaggle kernel-metadata.json format validation (GPU T4 dual accelerator, correct slug/title)
4. Unified Track Dispatch & CLI Interface:
   - Exploit, Tool Replay, Reasoning, and 'all' track dispatches
   - Dry-run mode and disk write mode into isolated temporary directories
"""

import os
import json
import zlib
import base64
import ast
import tempfile
import pytest

from src.evaluation.kaggle_shard_engine import (
    KaggleShardConfig,
    KaggleShardEngine,
    resolve_accounts,
    partition_dialogues,
    build_shard_notebook,
    build_kernel_metadata,
    DEFAULT_ACCOUNTS,
    DEFAULT_INPUT_PATHS,
    parse_args
)


# ==============================================================================
# 1. ACCOUNT RESOLUTION & SHARD COUNT DERIVATION TESTS
# ==============================================================================

def test_resolve_accounts_default():
    num_shards, accounts = resolve_accounts()
    assert num_shards == 3
    assert accounts == DEFAULT_ACCOUNTS


def test_resolve_accounts_explicit_num_shards_less_than_defaults():
    num_shards, accounts = resolve_accounts(num_shards=2)
    assert num_shards == 2
    assert len(accounts) == 2
    assert accounts == DEFAULT_ACCOUNTS[:2]


def test_resolve_accounts_explicit_num_shards_greater_than_defaults():
    num_shards, accounts = resolve_accounts(num_shards=5)
    assert num_shards == 5
    assert len(accounts) == 5
    assert accounts[:3] == DEFAULT_ACCOUNTS
    assert accounts[3] == "kaggle_worker_4"
    assert accounts[4] == "kaggle_worker_5"


def test_resolve_accounts_custom_accounts():
    custom = ["alice", "bob", "charlie", "david"]
    num_shards, accounts = resolve_accounts(accounts=custom)
    assert num_shards == 4
    assert accounts == custom


def test_resolve_accounts_from_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_ACCOUNTS", "acc_1, acc_2, acc_3, acc_4")
    num_shards, accounts = resolve_accounts(from_env=True)
    assert num_shards == 4
    assert accounts == ["acc_1", "acc_2", "acc_3", "acc_4"]


def test_resolve_accounts_from_env_empty_raises(monkeypatch):
    monkeypatch.delenv("KAGGLE_ACCOUNTS", raising=False)
    with pytest.raises(ValueError, match="KAGGLE_ACCOUNTS environment variable is empty"):
        resolve_accounts(from_env=True)


# ==============================================================================
# 2. DIALOGUE PARTITIONING & INTEGRITY TESTS
# ==============================================================================

def test_partition_dialogues_balanced_split():
    # Synthetic dataset of 100 dialogues
    dialogues = [{"id": f"conv_{i}", "turns": i} for i in range(100)]
    
    # 3 shards: 34, 33, 33
    shards = partition_dialogues(dialogues, 3)
    assert len(shards) == 3
    assert [len(s) for s in shards] == [34, 33, 33]

    # Verify lossless reconstruction
    merged = [d for s in shards for d in s]
    assert len(merged) == 100
    assert [d["id"] for d in merged] == [d["id"] for d in dialogues]


def test_partition_dialogues_zero_cross_shard_fragmentation():
    # Ensure all dialogue IDs across all shards are strictly disjoint
    dialogues = [{"id": f"sample_{i}", "messages": [{"role": "user", "content": "test"}]} for i in range(130)]
    shards = partition_dialogues(dialogues, 4)
    assert len(shards) == 4

    seen_ids = set()
    for shard in shards:
        shard_ids = {d["id"] for d in shard}
        assert not (seen_ids & shard_ids), "Cross-shard dialogue ID collision detected!"
        seen_ids.update(shard_ids)

    assert len(seen_ids) == 130


def test_partition_dialogues_edge_cases():
    with pytest.raises(ValueError, match="num_shards must be >= 1"):
        partition_dialogues([{"id": "1"}], 0)

    empty_shards = partition_dialogues([], 3)
    assert len(empty_shards) == 3
    assert all(len(s) == 0 for s in empty_shards)


# ==============================================================================
# 3. PROGRAMMATIC NOTEBOOK & METADATA GENERATION (AST VALIDATION)
# ==============================================================================

@pytest.mark.parametrize("track", ["exploit", "tool_replay", "reasoning"])
def test_build_shard_notebook_ast_validity(track):
    sample_dialogues = [
        {
            "id": f"test_{track}_{i}",
            "tools": [{"name": "nmap", "description": "scanner"}],
            "messages": [
                {"role": "user", "content": "Scan the target subnet."},
                {"role": "assistant", "content": "Executing nmap...", "tool_calls": [{"name": "nmap", "arguments": {"cmd": "-sS 10.0.0.1"}}]}
            ]
        }
        for i in range(5)
    ]

    nb = build_shard_notebook(
        track=track,
        shard_idx=1,
        total_shards=3,
        records=sample_dialogues,
        account="test_account"
    )

    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 5

    # Check AST validity for EVERY single code cell
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 4

    for idx, cell in enumerate(code_cells):
        code_str = "".join(cell["source"])
        try:
            tree = ast.parse(code_str)
            assert tree is not None
        except SyntaxError as e:
            pytest.fail(f"Track '{track}' code cell {idx} failed AST syntax check: {e}\nSource:\n{code_str}")


@pytest.mark.parametrize("track", ["exploit", "tool_replay", "reasoning"])
def test_extracted_worker_script_syntax_validity(track):
    sample_dialogues = [{"id": f"sample_{i}", "messages": []} for i in range(3)]
    nb = build_shard_notebook(
        track=track,
        shard_idx=1,
        total_shards=2,
        records=sample_dialogues,
        account="test_user"
    )

    # Locate cell that defines worker_code
    worker_script = None
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "worker_code =" in source:
                # Execute variable assignment in controlled local scope to extract the string
                loc = {}
                exec(compile(ast.parse(source), filename="<cell3>", mode="exec"), {}, loc)
                worker_script = loc.get("worker_code")
                break

    assert worker_script is not None, f"Track '{track}' notebook did not contain worker_code assignment."

    # Parse worker script with AST
    try:
        worker_ast = ast.parse(worker_script)
        assert worker_ast is not None
    except SyntaxError as e:
        pytest.fail(f"Track '{track}' embedded worker script failed AST check: {e}")


def test_payload_compression_roundtrip():
    original = [
        {"id": f"diag_{i}", "messages": [{"role": "user", "content": f"query {i}"}]}
        for i in range(25)
    ]
    b64 = build_shard_notebook(
        track="exploit",
        shard_idx=1,
        total_shards=1,
        records=original,
        account="test"
    )

    # Find the payload string in Cell 2
    decompressed = None
    for cell in b64["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "B64_PAYLOAD = " in source:
                for line in source.splitlines():
                    if line.strip().startswith("B64_PAYLOAD = "):
                        raw_val = line.split("=", 1)[1].strip().strip('"')
                        raw_bytes = zlib.decompress(base64.b64decode(raw_val.encode("ascii")))
                        decompressed = json.loads(raw_bytes.decode("utf-8"))
                        break

    assert decompressed is not None
    assert len(decompressed) == 25
    assert [d["id"] for d in decompressed] == [d["id"] for d in original]


def test_build_kernel_metadata_structure():
    meta = build_kernel_metadata(
        shard_idx=2,
        total_shards=5,
        track="tool_replay",
        account="test_account_shard2",
        notebook_filename="eval_shard_2.ipynb"
    )

    assert meta["id"] == "test_account_shard2/qwen3-8b-v3-tool-replay-eval-shard-2"
    assert meta["title"] == "Qwen3 8B v3 Tool Replay Eval Shard 2"
    assert meta["code_file"] == "eval_shard_2.ipynb"
    assert meta["is_gpu"] is True
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True
    assert meta["accelerator"] == "nvidiaTeslaT4Dual"


# ==============================================================================
# 4. END-TO-END ENGINE EXECUTION & CLI TESTS
# ==============================================================================

def test_engine_run_dry_run():
    config = KaggleShardConfig(
        track="tool_replay",
        num_shards=3,
        accounts=["acc_a", "acc_b", "acc_c"],
        dry_run=True
    )
    engine = KaggleShardEngine(config)
    results = engine.run()

    assert "tool_replay" in results
    manifest = results["tool_replay"]
    assert manifest["num_shards"] == 3
    assert manifest["total_dialogues"] == 130
    assert len(manifest["shards"]) == 3
    assert manifest["dry_run"] is True


def test_engine_run_generate_files_in_temp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = KaggleShardConfig(
            track="reasoning",
            num_shards=2,
            accounts=["worker1", "worker2"],
            output_dir=tmp_dir,
            dry_run=False
        )
        engine = KaggleShardEngine(config)
        results = engine.run()

        manifest_path = os.path.join(tmp_dir, "sharding_manifest.json")
        assert os.path.exists(manifest_path)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            assert manifest["total_dialogues"] == 130
            assert len(manifest["shards"]) == 2

        for shard_info in manifest["shards"]:
            s_idx = shard_info["shard_index"]
            shard_dir = os.path.join(tmp_dir, f"shard_{s_idx}")
            assert os.path.exists(shard_dir)
            assert os.path.exists(os.path.join(shard_dir, f"qwen3_8b_v3_reasoning_eval_shard_{s_idx}.ipynb"))
            assert os.path.exists(os.path.join(shard_dir, "kernel-metadata.json"))
            assert os.path.exists(os.path.join(shard_dir, f"reasoning_shard_{s_idx}.jsonl"))

            # Verify raw shard line count
            with open(os.path.join(shard_dir, f"reasoning_shard_{s_idx}.jsonl"), "r", encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
                assert len(lines) == 65


def test_engine_run_track_all_dry_run():
    config = KaggleShardConfig(
        track="all",
        num_shards=3,
        dry_run=True
    )
    engine = KaggleShardEngine(config)
    results = engine.run()

    assert set(results.keys()) == {"exploit", "tool_replay", "reasoning"}
    assert results["exploit"]["total_dialogues"] == 390
    assert results["tool_replay"]["total_dialogues"] == 130
    assert results["reasoning"]["total_dialogues"] == 130


def test_cli_parse_args():
    cfg = parse_args(["--track", "tool_replay", "--num-shards", "5", "--dry-run"])
    assert cfg.track == "tool_replay"
    assert cfg.num_shards == 5
    assert cfg.dry_run is True

    cfg_env = parse_args(["--track", "reasoning", "--from-env"])
    assert cfg_env.track == "reasoning"
    assert cfg_env.from_env is True
