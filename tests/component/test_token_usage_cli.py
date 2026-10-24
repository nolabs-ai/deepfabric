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
def test_cli_reports_usage_and_writes_requested_report_on_success_or_failure(
    tmp_path, tui_mode, fail
):
    path = config_file(tmp_path)
    output = tmp_path / "override.jsonl"
    report_path = tmp_path / "run.report.json"

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
            cli,
            [
                "generate",
                str(path),
                "--output-save-as",
                str(output),
                "--tui",
                tui_mode,
                "--run-report",
                str(report_path),
            ],
        )
    assert result.exit_code == (1 if fail else 0), result.output
    run = json.loads(report_path.read_text())
    assert run["status"] == ("failed" if fail else "completed")
    report = run["token_usage"]
    assert report["tokens"]["total_tokens"] == 42
    assert report["tokens"]["output_tokens"] == 12
    assert report["requests"] == 2
    assert len(report["by_model"]) == 2
    assert "Token usage" in result.output
    assert "topics-model" in result.output
    assert "samples-model" in result.output
    assert "total=28" in result.output
    assert not (tmp_path / "default.usage.json").exists()


def test_missing_usage_is_explicit(capsys):
    tracker = TokenUsageTracker()
    with track_token_usage(tracker):
        record_usage("gemini", "test", None)
    _report_token_usage(tracker)
    output = capsys.readouterr().out
    assert "input=unknown" in output
    assert "incomplete" in output


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
        result = CliRunner().invoke(
            cli,
            [
                "generate",
                str(path),
                "--dry-run",
                "--run-report",
                str(tmp_path / "preview.report.json"),
            ],
        )
    assert result.exit_code == 0, result.output
    assert "Token usage" not in result.output
    assert not (tmp_path / "preview.report.json").exists()
    assert not (tmp_path / "default.usage.json").exists()


def test_no_report_or_automatic_usage_file_without_flag(tmp_path):
    path = config_file(tmp_path)
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._initialize_topic_model", return_value=object()),
        patch(
            "deepfabric.cli._run_generation",
            side_effect=lambda **_kw: record_usage(
                "gemini", "test", {"prompt_token_count": 1234567, "candidates_token_count": 5}
            ),
        ),
    ):
        result = CliRunner().invoke(cli, ["generate", str(path)])
    assert result.exit_code == 0, result.output
    assert "1,234,567" in result.output
    assert not list(tmp_path.glob("*.json"))


def test_report_write_failure_does_not_mask_generation_failure(tmp_path):
    path = config_file(tmp_path)
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._initialize_topic_model", return_value=object()),
        patch(
            "deepfabric.cli._run_generation",
            side_effect=ConfigurationError("original generation failure"),
        ),
        patch("deepfabric.run_report.RunReport.write", side_effect=OSError("report failure")),
    ):
        result = CliRunner().invoke(
            cli, ["generate", str(path), "--run-report", str(tmp_path / "report.json")]
        )
    assert result.exit_code == 1
    assert "original generation failure" in result.output
    assert "Could not save run report" in result.output


@pytest.mark.parametrize("target", ["config", "output"])
def test_report_cannot_overwrite_input_or_dataset(tmp_path, target):
    path = config_file(tmp_path)
    report_path = path if target == "config" else tmp_path / "default.jsonl"
    original = path.read_bytes()
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch(
            "deepfabric.cli._initialize_topic_model",
            side_effect=AssertionError("provider must not start"),
        ),
    ):
        result = CliRunner().invoke(cli, ["generate", str(path), "--run-report", str(report_path)])
    assert result.exit_code == 1
    assert "must differ" in result.output
    assert path.read_bytes() == original
    if target == "output":
        assert not report_path.exists()


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_bare_report_flag_uses_effective_dataset_path(tmp_path, override, fail):
    path = config_file(tmp_path)
    output = tmp_path / "nested" / "samples.train.jsonl" if override else tmp_path / "default.jsonl"
    args = ["generate", str(path), "--run-report", "--tui", "simple"]
    if override:
        args += ["--output-save-as", str(output)]

    def dataset_stage(**_kwargs):
        record_usage("gemini", "test", {"prompt_token_count": 10, "candidates_token_count": 5})
        if fail:
            raise ConfigurationError("generation failed")

    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._initialize_topic_model", return_value=object()),
        patch("deepfabric.cli._run_generation", side_effect=dataset_stage),
    ):
        result = CliRunner().invoke(cli, args)
    assert result.exit_code == (1 if fail else 0), result.output
    report_path = output.with_suffix(".report.json")
    report = json.loads(report_path.read_text())
    assert report["output_path"] == str(output.resolve())
    assert report["token_usage"]["tokens"]["total_tokens"] == 15
    assert report["status"] == ("failed" if fail else "completed")
    assert report_path.name in result.output.replace("\n", "")
    assert not output.with_suffix(".usage.json").exists()


def test_bare_report_flag_dry_run_creates_no_report(tmp_path):
    path = config_file(tmp_path)
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys", side_effect=AssertionError("API validation")),
    ):
        result = CliRunner().invoke(cli, ["generate", str(path), "--dry-run", "--run-report"])
    assert result.exit_code == 0, result.output
    assert not list(tmp_path.glob("*.report.json"))
