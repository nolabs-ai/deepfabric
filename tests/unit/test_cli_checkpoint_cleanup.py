"""Checkpoint cleanup after a completed run keeps failed slots retryable."""

from unittest.mock import patch

import pytest
import yaml

from click.testing import CliRunner

from deepfabric.cli import cli


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
