"""CLI reporting and saved run accounting without provider or model calls."""
# ruff: noqa: PLR2004

import json

from unittest.mock import patch

import pytest
import yaml

from click.testing import CliRunner

from deepfabric.cli import _report_token_usage, cli
from deepfabric.exceptions import ConfigurationError
from deepfabric.llm.token_usage import TokenUsageTracker, record_usage, track_token_usage


def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "llm": {"provider": "gemini", "model": "test"},
                "topics": {"prompt": "test", "mode": "tree", "depth": 2, "degree": 2},
                "generation": {"system_prompt": "test"},
                "output": {
                    "save_as": str(tmp_path / "default.jsonl"),
                    "num_samples": 2,
                    "batch_size": 1,
                },
            }
        )
    )
    return path


@pytest.mark.parametrize("tui_mode", ["rich", "simple"])
@pytest.mark.parametrize("fail", [False, True])
def test_cli_reports_usage_and_writes_sidecar_on_success_or_failure(tmp_path, tui_mode, fail):
    path = config_file(tmp_path)
    output = tmp_path / "override.jsonl"

    def topic_stage(**_kwargs):
        record_usage(
            "gemini",
            "topics-model",
            {
                "prompt_token_count": 10,
                "candidates_token_count": 4,
                "total_token_count": 14,
            },
        )
        return object()

    def dataset_stage(**_kwargs):
        record_usage(
            "gemini",
            "samples-model",
            {
                "prompt_token_count": 20,
                "candidates_token_count": 5,
                "thoughts_token_count": 3,
                "total_token_count": 28,
            },
        )
        if fail:
            raise ConfigurationError("generation failed")

    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._initialize_topic_model", side_effect=topic_stage),
        patch("deepfabric.cli._run_generation", side_effect=dataset_stage),
    ):
        result = CliRunner().invoke(
            cli, ["generate", str(path), "--output-save-as", str(output), "--tui", tui_mode]
        )
    assert result.exit_code == (1 if fail else 0), result.output
    report = json.loads(output.with_suffix(".usage.json").read_text())
    assert report["tokens"]["total_tokens"] == 42
    assert report["tokens"]["output_tokens"] == 12
    assert report["requests"] == 2
    assert len(report["by_model"]) == 2
    assert "Token usage" in result.output
    assert "topics-model" in result.output
    assert "samples-model" in result.output
    assert "total=28" in result.output
    assert not (tmp_path / "default.usage.json").exists()


def test_missing_usage_is_explicit_and_write_error_does_not_mask_generation(tmp_path, capsys):
    tracker = TokenUsageTracker()
    with track_token_usage(tracker):
        record_usage("gemini", "test", None)
    # A non-existent parent causes a real file error without modifying permissions.
    _report_token_usage(tracker, str(tmp_path / "absent/result.jsonl"))
    output = capsys.readouterr().out
    assert "input=unknown" in output
    assert "incomplete" in output
    assert "Could not save token usage" in output


def test_dry_run_does_not_report_or_save_usage(tmp_path):
    path = config_file(tmp_path)
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys", side_effect=AssertionError("API validation")),
        patch(
            "deepfabric.llm.client.make_outlines_model", side_effect=AssertionError("model init")
        ),
        patch(
            "deepfabric.llm.client.make_async_outlines_model",
            side_effect=AssertionError("async model init"),
        ),
    ):
        result = CliRunner().invoke(cli, ["generate", str(path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Token usage" not in result.output
    assert not (tmp_path / "default.usage.json").exists()
