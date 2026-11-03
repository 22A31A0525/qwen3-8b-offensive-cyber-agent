"""
Unit tests for 3-Way Tool Replay Kaggle Notebook Packages
=========================================================
Verifies:
1. All notebook files (.ipynb), split data files (.jsonl), and kernel-metadata.json exist.
2. Every code cell in each notebook parses with zero syntax errors via ast.parse.
3. Embedded base64 payloads decompress cleanly and match the split JSONL files 1:1.
"""

import os
import json
import ast
import zlib
import base64
import pytest

DEPLOY_ROOT = r"c:\Users\teja9\OneDrive\Desktop\projects\project_unkown\v3_pipeline\kaggle_eval_multiturn"

@pytest.mark.parametrize("shard_idx,expected_dialogues,expected_account", [
    (1, 44, "saitejavenkatavinay"),
    (2, 43, "alluriakshay"),
    (3, 43, "rohitvijayadapa2006"),
])
def test_shard_package_integrity(shard_idx, expected_dialogues, expected_account):
    s_dir = os.path.join(DEPLOY_ROOT, f"shard_{shard_idx}")
    nb_path = os.path.join(s_dir, f"qwen3_8b_v3_tool_replay_eval_shard_{shard_idx}.ipynb")
    data_path = os.path.join(s_dir, f"tool_replay_shard_{shard_idx}.jsonl")
    meta_path = os.path.join(s_dir, "kernel-metadata.json")

    assert os.path.exists(nb_path), f"Missing notebook: {nb_path}"
    assert os.path.exists(data_path), f"Missing data file: {data_path}"
    assert os.path.exists(meta_path), f"Missing metadata: {meta_path}"

    # Check metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["id"] == f"{expected_account}/qwen3-8b-v3-tool-replay-eval-shard-{shard_idx}"
    assert meta["enable_gpu"] in [True, "true"]
    assert meta["enable_internet"] in [True, "true"]

    # Check notebook cells and syntax
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert len(nb["cells"]) >= 7
    for c_idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            py_lines = [l for l in src.splitlines() if not l.strip().startswith("!")]
            py_code = "\n".join(py_lines)
            try:
                ast.parse(py_code)
            except SyntaxError as e:
                pytest.fail(f"Shard {shard_idx} Cell {c_idx} has invalid Python syntax: {e}")

    # Check data file
    with open(data_path, "r", encoding="utf-8") as f:
        disk_records = [json.loads(line) for line in f if line.strip()]
    assert len(disk_records) == expected_dialogues

    # Check embedded payload in Cell 3
    c3_src = "".join(nb["cells"][3]["source"])
    b64_str = None
    for line in c3_src.splitlines():
        if line.startswith("B64_PAYLOAD ="):
            b64_str = line.split('"')[1]
            break
    assert b64_str is not None, f"Could not find B64_PAYLOAD in Shard {shard_idx} Cell 3"

    decompressed = zlib.decompress(base64.b64decode(b64_str.encode("ascii"))).decode("utf-8")
    payload_records = json.loads(decompressed)
    assert len(payload_records) == expected_dialogues
    assert [r["id"] for r in payload_records] == [r["id"] for r in disk_records]

