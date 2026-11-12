"""Resume locates checkpoints written to output.checkpoint.path."""

import json

from unittest.mock import patch

import yaml

from click.testing import CliRunner

from deepfabric.cli import cli
from deepfabric.constants import CHECKPOINT_METADATA_SUFFIX, CHECKPOINT_VERSION
from deepfabric.generator import DataSetGenerator
from deepfabric.sampling import topic_plan
from deepfabric.utils import build_checkpoint_file_paths


def config_file(tmp_path, checkpoint_dir):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "llm": {"provider": "gemini", "model": "test"},
                "topics": {
                    "prompt": "test",
                    "mode": "tree",
                    "depth": 2,
                    "degree": 2,
                    "save_as": str(tmp_path / "topics.jsonl"),
                },
                "generation": {"system_prompt": "test"},
                "output": {
                    "save_as": str(tmp_path / "dataset.jsonl"),
                    "num_samples": 2,
                    "batch_size": 1,
                    "checkpoint": {"interval": 1, "path": str(checkpoint_dir)},
                },
            }
        )
    )
    return path


def write_checkpoint_metadata(checkpoint_dir, topics_file):
    checkpoint_dir.mkdir(exist_ok=True)
    metadata = checkpoint_dir / f"dataset{CHECKPOINT_METADATA_SUFFIX}"
    metadata.write_text(
        json.dumps(
            {
                "version": CHECKPOINT_VERSION,
                "plan": topic_plan(
                    2, [{"uuid": "leaf", "topic": "leaf", "path": ["root", "leaf"]}]
                ),
                "completed": [],
                "sample_slots": [],
                "total_samples": 0,
                "total_failures": 0,
                "topics_file": str(topics_file),
            }
        )
    )


def invoke_generate(args):
    seen = {}

    def topic_stage(*, preparation, options):  # noqa: ARG001
        seen["topics_load"] = options.topics_load
        return object()

    def dataset_stage(*, preparation, topic_model, options, checkpoint_dir):  # noqa: ARG001
        seen["checkpoint_dir"] = checkpoint_dir
        params = preparation.config.get_generation_params(checkpoint_path=checkpoint_dir)
        with patch("deepfabric.generator.LLMClient"):
            engine = DataSetGenerator(**params)
        assert engine._get_checkpoint_paths() == build_checkpoint_file_paths(
            preparation.config.output.save_as, checkpoint_dir
        )

    with (
        patch("deepfabric.generator.LLMClient"),
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._initialize_topic_model", side_effect=topic_stage),
        patch("deepfabric.cli._run_generation", side_effect=dataset_stage),
    ):
        result = CliRunner().invoke(cli, ["generate", *args])
    assert result.exit_code == 0, result.output
    return seen


def test_resume_reuses_topics_recorded_in_configured_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    topics_file = tmp_path / "saved-topics.jsonl"
    topics_file.write_text('{"path": ["root", "leaf"]}\n')
    write_checkpoint_metadata(checkpoint_dir, topics_file)

    seen = invoke_generate([str(config_file(tmp_path, checkpoint_dir)), "--resume"])

    assert seen == {"topics_load": str(topics_file), "checkpoint_dir": str(checkpoint_dir)}


def test_checkpoint_path_option_overrides_configured_path(tmp_path):
    configured_dir = tmp_path / "configured"
    override_dir = tmp_path / "override"
    topics_file = tmp_path / "saved-topics.jsonl"
    topics_file.write_text('{"path": ["root", "leaf"]}\n')
    write_checkpoint_metadata(override_dir, topics_file)

    seen = invoke_generate(
        [
            str(config_file(tmp_path, configured_dir)),
            "--resume",
            "--checkpoint-path",
            str(override_dir),
        ]
    )

    assert seen == {"topics_load": str(topics_file), "checkpoint_dir": str(override_dir)}


def test_resume_with_output_override_finds_checkpoint_named_after_config_output(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    topics_file = tmp_path / "saved-topics.jsonl"
    topics_file.write_text('{"path": ["root", "leaf"]}\n')
    # The engine names checkpoint files after output.save_as from the config ("dataset")
    write_checkpoint_metadata(checkpoint_dir, topics_file)

    seen = invoke_generate(
        [
            str(config_file(tmp_path, checkpoint_dir)),
            "--resume",
            "--output-save-as",
            str(tmp_path / "other.jsonl"),
        ]
    )

    assert seen == {"topics_load": str(topics_file), "checkpoint_dir": str(checkpoint_dir)}


def test_resume_uses_config_hashed_directory_when_no_path_is_configured(tmp_path):
    config = config_file(tmp_path, tmp_path / "unused")
    data = yaml.safe_load(config.read_text())
    del data["output"]["checkpoint"]["path"]
    config.write_text(yaml.safe_dump(data))
    topics_file = tmp_path / "saved-topics.jsonl"
    topics_file.write_text('{"path": ["root", "leaf"]}\n')

    with patch("deepfabric.utils._get_deepfabric_data_dir", return_value=tmp_path):
        metadata, _, _ = build_checkpoint_file_paths("dataset.jsonl", config_path=str(config))
        metadata.parent.mkdir(parents=True)
        write_checkpoint_metadata(metadata.parent, topics_file)
        seen = invoke_generate([str(config), "--resume"])

    assert seen == {"topics_load": str(topics_file), "checkpoint_dir": str(metadata.parent)}
