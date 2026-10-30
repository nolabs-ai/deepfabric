"""Checkpoint cleanup after a completed run keeps failed slots retryable."""
# ruff: noqa: PLR2004

import json

from unittest.mock import patch

import pytest
import yaml

from click.testing import CliRunner

from deepfabric.cli import cli
from deepfabric.llm.token_usage import record_usage
from deepfabric.run_report import RunReport
from tests.component.test_run_report import make_engine, topics


class FakeEngine:
    def __init__(self, failures):
        self.stop_requested = False
        self.failures = [{"error": "quota exceeded"}] * failures
        self.cleared = False

    def has_checkpoint(self):
        return False

    def get_all_failures(self):
        return list(self.failures)

    def clear_checkpoint(self):
        self.cleared = True


def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "llm": {"provider": "gemini", "model": "test"},
                "topics": {"prompt": "test", "mode": "tree", "depth": 2, "degree": 2},
                "generation": {"system_prompt": "test"},
                "output": {
                    "save_as": str(tmp_path / "dataset.jsonl"),
                    "num_samples": 2,
                    "batch_size": 1,
                    "checkpoint": {"interval": 1, "path": str(tmp_path / "checkpoints")},
                },
            }
        )
    )
    return path


@pytest.mark.parametrize("failures", [0, 3])
def test_checkpoint_cleared_only_when_no_samples_failed(tmp_path, failures):
    engine = FakeEngine(failures)
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._initialize_topic_model", return_value=object()),
        patch("deepfabric.cli.DataSetGenerator", return_value=engine),
        patch("deepfabric.cli.create_dataset", return_value=[{"messages": []}] * 2),
        patch("deepfabric.cli.save_dataset"),
        patch("deepfabric.cli.trace"),
        patch("deepfabric.cli._trigger_cloud_upload"),
    ):
        result = CliRunner().invoke(cli, ["generate", str(config_file(tmp_path))])

    assert result.exit_code == 0, result.output
    assert engine.cleared is (failures == 0)


@pytest.mark.parametrize("report_mode", ["disabled", "saved", "failure"])
def test_real_checkpoint_accounting_report_handoff(tmp_path, report_mode):
    engine, _ = make_engine(tmp_path / "checkpoints", "independent", checkpoint_interval=1)
    tracker_snapshots = []
    report_path = tmp_path / "report.json"

    def initialize(**_kwargs):
        record_usage("gemini", "topics", {"prompt_token_count": 20, "candidates_token_count": 5})
        return topics()

    def create(**_kwargs):
        return engine.create_data(num_samples=4, batch_size=2, topic_model=topics())

    original_write = RunReport.write

    def write(report, path, usage, status):
        assert engine._checkpoint_metadata_path.exists()  # Handoff before cleanup.
        tracker_snapshots.append(usage)
        if report_mode == "failure":
            raise OSError("report disk error")
        original_write(report, path, usage, status)

    args = ["generate", str(config_file(tmp_path))]
    if report_mode != "disabled":
        args.extend(["--run-report", str(report_path)])
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._initialize_topic_model", side_effect=initialize),
        patch("deepfabric.cli.DataSetGenerator", return_value=engine),
        patch("deepfabric.cli.create_dataset", side_effect=create),
        patch("deepfabric.cli.save_dataset"),
        patch("deepfabric.cli.trace"),
        patch("deepfabric.cli._trigger_cloud_upload"),
        patch.object(RunReport, "write", write),
    ):
        result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "Cumulative recorded token usage" in result.output
    assert "total=48" in result.output
    assert "total=25" in result.output
    assert engine._checkpoint_metadata_path.exists() is (report_mode == "failure")
    if report_mode == "saved":
        usage = json.loads(report_path.read_text())["token_usage"]
        assert usage["requests"] == usage["cumulative"]["requests"] == 5
        assert usage["cumulative"]["tokens"]["total_tokens"] == 73
    elif report_mode == "failure":
        assert "Keeping checkpoint" in result.output
        assert tracker_snapshots[0]["cumulative"]["requests"] == 5
    else:
        assert not report_path.exists()
