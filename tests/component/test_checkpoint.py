"""Component tests for checkpoint."""

import json
import logging

from unittest.mock import patch

import pytest
import yaml

from click.testing import CliRunner

from deepfabric.cli import cli
from deepfabric.constants import (
    CHECKPOINT_VERSION,
)
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator
from deepfabric.sampling import topic_plan
from tests.support import checkpoint as _support

base_generator_params = _support.base_generator_params
generator_with_checkpoint = _support.generator_with_checkpoint
temp_checkpoint_dir = _support.temp_checkpoint_dir


def seed_checkpoint(generator, *, failure=True):
    generator._generation_plan = topic_plan(
        10,
        [
            {"uuid": "uuid-1", "topic": "A", "path": ["Root", "A"]},
            {"uuid": "uuid-2", "topic": "B", "path": ["Root", "B"]},
        ],
    )
    generator._initialize_checkpoint_paths()
    failures = (
        [{"topic_id": "uuid-1", "sample_ordinal": 1, "error": "Rate limit exceeded"}]
        if failure
        else []
    )
    completed = [("uuid-1", 0)] + ([("uuid-1", 1)] if failure else [])
    generator._save_checkpoint(
        [{"question": "Q1", "answer": "A1"}], failures, completed, sample_slots=[("uuid-1", 0)]
    )


class TestCheckpointSaveLoad:
    def test_save_checkpoint_creates_files(self, generator_with_checkpoint):
        seed_checkpoint(generator_with_checkpoint)
        for path in generator_with_checkpoint._get_checkpoint_paths():
            assert path.exists()

    def test_save_checkpoint_appends_samples(self, generator_with_checkpoint):
        generator = generator_with_checkpoint
        seed_checkpoint(generator, failure=False)
        generator._save_checkpoint(
            [{"question": "Q2"}], [], [("uuid-2", 0)], sample_slots=[("uuid-2", 0)]
        )
        assert [s["question"] for s in generator._load_all_samples_from_checkpoint()] == [
            "Q1",
            "Q2",
        ]
        assert generator._sample_slots == [("uuid-1", 0), ("uuid-2", 0)]

    def test_load_checkpoint_restores_state(self, generator_with_checkpoint, base_generator_params):
        generator = generator_with_checkpoint
        seed_checkpoint(generator)
        with patch("deepfabric.generator.LLMClient"):
            restored = DataSetGenerator(
                **{**base_generator_params, "checkpoint_path": generator.config.checkpoint_path}
            )
        assert restored.load_checkpoint()
        assert restored._flushed_samples_count == 1
        assert restored._flushed_failures_count == 1
        assert restored._samples == []
        assert restored._completed == {("uuid-1", 0), ("uuid-1", 1)}
        assert restored._generation_plan == generator._generation_plan
        assert restored._load_all_samples_from_checkpoint()[0]["question"] == "Q1"

    def test_load_checkpoint_with_retry_failed(self, generator_with_checkpoint):
        generator = generator_with_checkpoint
        seed_checkpoint(generator)
        assert generator.load_checkpoint(retry_failed=False)
        assert generator._completed == {("uuid-1", 0), ("uuid-1", 1)}
        assert generator.load_checkpoint(retry_failed=True)
        assert generator._completed == {("uuid-1", 0)}
        assert generator._flushed_failures_count == 0
        assert generator._checkpoint_failures_path.read_text() == ""
        assert generator.load_checkpoint()  # persisted state remains self-consistent

    def test_load_checkpoint_returns_false_when_no_checkpoint(self, generator_with_checkpoint):
        assert generator_with_checkpoint.load_checkpoint() is False

    def test_load_checkpoint_returns_false_when_disabled(self, generator_with_checkpoint):
        generator_with_checkpoint.config.checkpoint_interval = None
        assert generator_with_checkpoint.load_checkpoint() is False

    def test_load_checkpoint_warns_on_config_mismatch(self, generator_with_checkpoint, caplog):
        seed_checkpoint(generator_with_checkpoint)
        generator_with_checkpoint.model_name = "different-model"
        with caplog.at_level(logging.WARNING):
            assert generator_with_checkpoint.load_checkpoint()
        assert "Config mismatch" in caplog.text

    @pytest.mark.parametrize("version", [4, 3, 6, None])
    def test_incompatible_version_raises_without_mutation(self, generator_with_checkpoint, version):
        generator = generator_with_checkpoint
        seed_checkpoint(generator)
        metadata = json.loads(generator._checkpoint_metadata_path.read_text())
        metadata["version"] = version
        generator._checkpoint_metadata_path.write_text(json.dumps(metadata))
        before = {p: p.read_bytes() for p in generator._get_checkpoint_paths()}
        with pytest.raises(DataSetGeneratorError, match="version") as error:
            generator.load_checkpoint()
        if version == 4:  # noqa: PLR2004
            assert "tools/migrate_checkpoint.py" in str(error.value)
            assert "may fail" in str(error.value)
            assert "unambiguously" in str(error.value)
        assert all(p.read_bytes() == data for p, data in before.items())

    def test_corrupt_slot_accounting_rejected(self, generator_with_checkpoint):
        generator = generator_with_checkpoint
        seed_checkpoint(generator)
        metadata = json.loads(generator._checkpoint_metadata_path.read_text())
        metadata["sample_slots"] = []
        generator._checkpoint_metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(DataSetGeneratorError, match="inconsistent"):
            generator.load_checkpoint()


class TestCheckpointClear:
    def test_clear_checkpoint_removes_files(self, generator_with_checkpoint):
        seed_checkpoint(generator_with_checkpoint)
        generator_with_checkpoint.clear_checkpoint()
        assert not any(p.exists() for p in generator_with_checkpoint._get_checkpoint_paths())
        assert not generator_with_checkpoint._completed
        assert generator_with_checkpoint._generation_plan is None


class TestCheckpointMetadata:
    def test_metadata_contains_expected_fields(self, generator_with_checkpoint):
        generator = generator_with_checkpoint
        seed_checkpoint(generator)
        metadata = json.loads(generator._checkpoint_metadata_path.read_text())
        assert metadata["version"] == CHECKPOINT_VERSION == 5  # noqa: PLR2004
        assert metadata["plan"]["target"] == 10  # noqa: PLR2004
        assert metadata["plan"]["generation_mode"] == "topic_allocation"
        assert metadata["sample_slots"] == [["uuid-1", 0]]
        assert set(map(tuple, metadata["completed"])) == {("uuid-1", 0), ("uuid-1", 1)}
        assert "cycles_needed" not in metadata


class TestCheckpointStatusCommand:
    def config_path(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "topics": {"prompt": "Root", "mode": "tree"},
                    "generation": {"system_prompt": "Generate"},
                    "output": {
                        "num_samples": 10,
                        "save_as": "test_dataset.jsonl",
                        "checkpoint": {"interval": 1, "path": str(tmp_path)},
                    },
                }
            )
        )
        return path

    def test_checkpoint_status_no_checkpoint(self, tmp_path):
        result = CliRunner().invoke(cli, ["checkpoint-status", str(self.config_path(tmp_path))])
        assert result.exit_code == 0
        assert "No checkpoint found" in result.output

    def test_checkpoint_status_with_checkpoint(self, tmp_path, generator_with_checkpoint):
        generator = generator_with_checkpoint
        generator.config.checkpoint_path = str(tmp_path)
        seed_checkpoint(generator)
        result = CliRunner().invoke(cli, ["checkpoint-status", str(self.config_path(tmp_path))])
        assert result.exit_code == 0, result.output
        assert "1/10" in result.output
        assert "Allocation:" in result.output
        assert "Resume with:" in result.output

    def test_checkpoint_status_with_failures(self, tmp_path, generator_with_checkpoint):
        generator_with_checkpoint.config.checkpoint_path = str(tmp_path)
        seed_checkpoint(generator_with_checkpoint)
        result = CliRunner().invoke(cli, ["checkpoint-status", str(self.config_path(tmp_path))])
        assert result.exit_code == 0
        assert "Rate limit exceeded" in result.output
        assert "--retry-failed" in result.output

    def test_checkpoint_status_v4_guidance(self, tmp_path, generator_with_checkpoint):
        generator = generator_with_checkpoint
        generator.config.checkpoint_path = str(tmp_path)
        seed_checkpoint(generator)
        metadata = json.loads(generator._checkpoint_metadata_path.read_text())
        metadata["version"] = 4
        generator._checkpoint_metadata_path.write_text(json.dumps(metadata))
        result = CliRunner().invoke(cli, ["checkpoint-status", str(self.config_path(tmp_path))])
        assert result.exit_code == 0
        assert "migrate_checkpoint.py" in result.output
        assert "Resume with:" not in result.output
