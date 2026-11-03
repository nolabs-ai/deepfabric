"""Resume locates checkpoints written to output.checkpoint.path."""

import json

from unittest.mock import Mock, patch

import pytest
import yaml

from click.testing import CliRunner

from deepfabric.cli import _windows_prompt_with_timeout, cli
from deepfabric.constants import CHECKPOINT_METADATA_SUFFIX, CHECKPOINT_VERSION
from deepfabric.sampling import topic_plan


@pytest.fixture(autouse=True)
def mock_generator_client():
    with patch("deepfabric.generator.LLMClient"):
        yield


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


def test_implicit_resume_selects_saved_topics_before_topic_stage(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    topics_file = tmp_path / "saved-topics.jsonl"
    topics_file.write_text('{"path": ["root", "leaf"]}\n')
    write_checkpoint_metadata(checkpoint_dir, topics_file)
    with patch("deepfabric.cli._prompt_with_timeout", return_value="1") as prompt:
        seen = invoke_generate([str(config_file(tmp_path, checkpoint_dir))])
    assert seen["topics_load"] == str(topics_file)
    prompt.assert_called_once()


def test_abort_checkpoint_prompt_does_not_generate_topics(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    write_checkpoint_metadata(checkpoint_dir, tmp_path / "missing.jsonl")
    config = config_file(tmp_path, checkpoint_dir)
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._prompt_with_timeout", return_value="3"),
        patch("deepfabric.cli._initialize_topic_model") as topics,
    ):
        result = CliRunner().invoke(cli, ["generate", str(config)])
    assert result.exit_code == 0, result.output
    topics.assert_not_called()


def test_fresh_collision_can_load_existing_topics(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    config = config_file(tmp_path, checkpoint_dir)
    topics = tmp_path / "topics.jsonl"
    topics.write_text('{"path": ["root", "leaf"]}\n')
    with patch("deepfabric.cli._prompt_with_timeout", return_value="4"):
        seen = invoke_generate([str(config)])
    assert seen["topics_load"] == str(topics)
    assert topics.read_text() == '{"path": ["root", "leaf"]}\n'


def test_startup_does_not_probe_or_create_destination_parents(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    config = config_file(tmp_path, checkpoint_dir)
    destination = tmp_path / "not-created" / "dataset.jsonl"
    invoke_generate([str(config), "--output-save-as", str(destination)])
    assert not destination.parent.exists()
    assert not checkpoint_dir.exists()


def test_clear_checkpoint_cannot_delete_the_configuration_input(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    original = config_file(tmp_path, checkpoint_dir)
    aliased_config = checkpoint_dir / f"dataset{CHECKPOINT_METADATA_SUFFIX}"
    original.rename(aliased_config)
    before = aliased_config.read_text()
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._prompt_with_timeout", return_value="1"),
        patch("deepfabric.cli._initialize_topic_model") as topics,
    ):
        result = CliRunner().invoke(cli, ["generate", str(aliased_config)])
    assert result.exit_code == 1, result.output
    assert "aliases an input" in result.output
    assert aliased_config.read_text() == before
    topics.assert_not_called()


def test_windows_collision_prompt_times_out_without_console_input():
    console = Mock()
    console.kbhit.return_value = False
    clock = iter([0, 0, 0, 21])
    with (
        patch("deepfabric.cli.msvcrt", console),
        patch("deepfabric.cli.time", Mock(monotonic=lambda: next(clock))),
    ):
        assert _windows_prompt_with_timeout(["1", "2", "3"], "1", 20) == "1"


def test_windows_collision_prompt_accepts_explicit_choice():
    console = Mock()
    console.kbhit.return_value = True
    console.getwch.side_effect = ["2", "\r"]
    with (
        patch("deepfabric.cli.msvcrt", console),
        patch("deepfabric.cli.time", Mock(monotonic=lambda: 0)),
    ):
        assert _windows_prompt_with_timeout(["1", "2", "3"], "1", 20) == "2"
