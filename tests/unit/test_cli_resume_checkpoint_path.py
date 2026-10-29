"""Resume locates checkpoints written to output.checkpoint.path."""

import json

from unittest.mock import patch

import yaml

from click.testing import CliRunner

from deepfabric.cli import cli
from deepfabric.constants import CHECKPOINT_METADATA_SUFFIX


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
    metadata.write_text(json.dumps({"topics_file": str(topics_file)}))


def invoke_generate(args):
    seen = {}

    def topic_stage(*, preparation, options):  # noqa: ARG001
        seen["topics_load"] = options.topics_load
        return object()

    def dataset_stage(*, preparation, topic_model, options, checkpoint_dir):  # noqa: ARG001
        seen["checkpoint_dir"] = checkpoint_dir

    with (
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
