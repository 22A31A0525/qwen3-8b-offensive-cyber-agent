import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath("."))
from kaggle.train_qwen3_8b import (
    HF_DEFAULT_REPO_ID,
    MODEL_CARD_TEMPLATE,
    discover_hf_token,
    push_adapter_to_hub,
)


def test_discover_hf_token_cli_arg():
    token = discover_hf_token(cli_token="hf_test_cli_token")
    assert token == "hf_test_cli_token"


def test_discover_hf_token_env_var(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_env_token")
    token = discover_hf_token(cli_token=None)
    assert token == "hf_test_env_token"


def test_discover_hf_token_kaggle_secrets(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    mock_client = MagicMock()
    mock_client.return_value.get_secret.return_value = "hf_test_kaggle_secret"
    with patch.dict("sys.modules", {"kaggle_secrets": MagicMock(UserSecretsClient=mock_client)}):
        token = discover_hf_token(cli_token=None)
        assert token == "hf_test_kaggle_secret"


def test_push_adapter_to_hub_smoke_guard():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Raising ValueError when attempting to upload smoke weights to production repository
        with pytest.raises(ValueError, match="Smoke weights cannot be uploaded to the production repository"):
            push_adapter_to_hub(
                adapter_dir=tmp_dir,
                repo_id=HF_DEFAULT_REPO_ID,
                hf_token="hf_fake_token",
                smoke_segment=True,
                allow_smoke_push=False,
            )

        # 2. Skipping upload when targeting custom non-test repository with smoke_segment=True and allow_smoke_push=False
        success, commit_sha = push_adapter_to_hub(
            adapter_dir=tmp_dir,
            repo_id="custom-user/custom-model",
            hf_token="hf_fake_token",
            smoke_segment=True,
            allow_smoke_push=False,
        )
        assert success is False, "Should skip upload when smoke_segment is True without allow_smoke_push"
        assert commit_sha is None


def test_push_adapter_to_hub_no_token():
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("kaggle.train_qwen3_8b.discover_hf_token", return_value=None):
            success, commit_sha = push_adapter_to_hub(
                adapter_dir=tmp_dir,
                repo_id=HF_DEFAULT_REPO_ID,
                hf_token=None,
                smoke_segment=False,
            )
            assert success is False, "Should safely return False when no token is available"
            assert commit_sha is None


def test_push_adapter_to_hub_mocked_success():
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
            f.write("{}")

        with patch("huggingface_hub.HfApi") as mock_hf_api, patch(
            "huggingface_hub.upload_folder"
        ) as mock_upload_folder:
            mock_api_instance = MagicMock()
            mock_hf_api.return_value = mock_api_instance
            mock_api_instance.create_repo.return_value = "https://huggingface.co/" + HF_DEFAULT_REPO_ID
            mock_upload_folder.return_value = MagicMock(oid="fake_commit_sha_1234567890")

            success, commit_sha = push_adapter_to_hub(
                adapter_dir=tmp_dir,
                repo_id=HF_DEFAULT_REPO_ID,
                hf_token="hf_mock_token_12345",
                private=True,
                smoke_segment=False,
            )

            assert success is True
            assert commit_sha == "fake_commit_sha_1234567890"
            mock_api_instance.create_repo.assert_called_once_with(
                repo_id=HF_DEFAULT_REPO_ID, repo_type="model", private=True, exist_ok=True
            )
            mock_upload_folder.assert_called_once()
            readme_path = os.path.join(tmp_dir, "README.md")
            assert os.path.exists(readme_path)
