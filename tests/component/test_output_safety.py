"""Actual local exporters and CLI orchestration with generation/provider I/O mocked."""

import json

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from click.testing import CliRunner
from datasets import Dataset

from deepfabric.cli import (
    GenerateOptions,
    _initialize_topic_model,
    _prepare_output_destinations,
    _run_generation,
    cli,
)
from deepfabric.config import DeepFabricConfig
from deepfabric.dataset_manager import save_dataset
from deepfabric.exceptions import ConfigurationError
from deepfabric.llm.token_usage import TokenUsageTracker, track_token_usage
from deepfabric.topic_manager import save_topic_model
from deepfabric.tree import Tree


@pytest.fixture
def config(tmp_path):
    return DeepFabricConfig.model_validate(
        {
            "llm": {"provider": "openai", "model": "test"},
            "topics": {"prompt": "test", "mode": "tree", "save_as": str(tmp_path / "topics.jsonl")},
            "generation": {"system_prompt": "test"},
            "output": {
                "save_as": str(tmp_path / "dataset.jsonl"),
                "num_samples": 1,
                "batch_size": 1,
            },
        }
    )


def fake_tree():
    tree = Mock(spec=Tree)
    tree.tree_paths = [["root", "leaf"]]
    tree.failed_generations = []
    return tree


def test_actual_topic_path_propagates_to_options(config, tmp_path):
    original = Path(config.topics.save_as)
    original.write_text("existing")
    options = GenerateOptions()
    with patch("deepfabric.cli.load_or_build_topic_model", return_value=fake_tree()):
        _initialize_topic_model(
            preparation=Mock(config=config, topics_overrides={}), options=options
        )
    assert options.topics_save_as == str(tmp_path / "topics_1.jsonl")
    assert original.read_text() == "existing"
    assert json.loads(Path(options.topics_save_as).read_text())["path"] == ["root", "leaf"]


def test_dataset_actual_path_reaches_upload(config, tmp_path):
    target = Path(config.output.save_as)
    target.write_text("existing")
    config.huggingface = Mock()
    with patch("deepfabric.dataset_manager._upload_to_huggingface") as upload:
        saved = save_dataset(Dataset.from_list([{"answer": "yes"}]), str(target), config)
        actual = saved.dataset_path
    assert upload.call_args.args[0] == actual
    assert actual == str(tmp_path / "dataset_1.jsonl")
    assert target.read_text() == "existing"


def test_topic_sidecar_is_distinct_even_without_expected_extension(config, tmp_path):
    tree = fake_tree()
    tree.failed_generations = ["failed"]
    target = tmp_path / "topics.custom"
    actual = save_topic_model(tree, config, str(target))
    assert json.loads(Path(actual).read_text())["path"] == ["root", "leaf"]
    assert (
        json.loads((tmp_path / "topics_failed.jsonl").read_text())["failed_generation"] == "failed"
    )


def test_failure_export_error_blocks_upload_and_checkpoint_cleanup(config, tmp_path):
    engine = Mock()
    engine.stop_requested = False
    engine.get_all_failures.return_value = [{"error": "bad"}]
    preparation = Mock(config=config, generation_overrides={}, num_samples=1, batch_size=1)
    tracker = TokenUsageTracker()
    with (
        track_token_usage(tracker),
        patch("deepfabric.cli.DataSetGenerator", return_value=engine),
        patch("deepfabric.cli.create_dataset", return_value=Dataset.from_list([{"answer": "yes"}])),
        patch(
            "deepfabric.dataset_manager._save_failed_samples", side_effect=PermissionError("denied")
        ),
        pytest.raises(ConfigurationError, match="denied"),
    ):
        _run_generation(
            preparation=preparation,
            topic_model=fake_tree(),
            options=GenerateOptions(),
            checkpoint_dir=str(tmp_path),
        )
    assert getattr(tracker, "checkpoint_cleanup", None) is None
    engine.clear_checkpoint.assert_not_called()


def test_python_caller_receives_actual_failure_export_path(tmp_path):
    failures = tmp_path / "failures.jsonl"
    failures.write_text("old failures")
    engine = Mock()
    engine.get_all_failures.return_value = [{"error": "bad"}]
    saved = save_dataset(
        Dataset.from_list([{"answer": "yes"}]),
        str(tmp_path / "data.jsonl"),
        engine=engine,
        failures_path=str(failures),
    )
    assert saved.failures_path == str(tmp_path / "failures_1.jsonl")
    assert failures.read_text() == "old failures"


@pytest.mark.parametrize("report_failure", [False, True])
def test_cli_real_exports_and_report_recovery(config, tmp_path, monkeypatch, report_failure):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(exclude_unset=True)))
    target = Path(config.output.save_as)
    target.write_text("old dataset")
    engine = Mock()
    engine.stop_requested = False
    engine.get_all_failures.return_value = []
    recovery = tmp_path / "app"
    monkeypatch.setattr("deepfabric.output_safety._get_deepfabric_data_dir", lambda: recovery)
    args = ["generate", str(config_file), "--run-report"]
    if report_failure:
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory")
        args.append(str(blocked / "report.json"))
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli.load_or_build_topic_model", return_value=fake_tree()),
        patch("deepfabric.cli.DataSetGenerator", return_value=engine),
        patch("deepfabric.cli.create_dataset", return_value=Dataset.from_list([{"answer": "yes"}])),
        patch("deepfabric.cli._prompt_with_timeout", return_value="1"),
        patch("deepfabric.cli._trigger_cloud_upload") as upload,
    ):
        result = CliRunner().invoke(cli, args)
    assert result.exit_code == (1 if report_failure else 0), result.output
    actual = str(tmp_path / "dataset_1.jsonl")
    assert upload.call_args.kwargs["dataset_path"] == actual
    assert target.read_text() == "old dataset"
    assert json.loads(Path(actual).read_text())["answer"] == "yes"
    report = (
        recovery / "recovery" / "report.json"
        if report_failure
        else tmp_path / "dataset_1.report.json"
    )
    assert json.loads(report.read_text())["output_path"] == actual
    if report_failure:
        assert "Recovery saved to:" in result.output


@pytest.mark.parametrize("kind", ["dataset", "topics"])
def test_clear_deletes_now_and_save_preserves_a_later_competing_file(config, tmp_path, kind):
    target = Path(config.output.save_as if kind == "dataset" else config.topics.save_as)
    target.write_text("old output")
    options = GenerateOptions(config_file=str(tmp_path / "config.yaml"))
    with patch("deepfabric.cli._prompt_with_timeout", return_value="2"):
        _prepare_output_destinations(Mock(config=config), options)
    assert not target.exists()  # Clearing occurs before generation or serialization.
    target.write_text("another process's output")
    if kind == "dataset":
        saved = save_dataset(Dataset.from_list([{"answer": "fresh"}]), str(target), config)
        actual = saved.dataset_path
    else:
        actual = save_topic_model(fake_tree(), config, str(target))
    assert actual == str(target.with_name(f"{target.stem}_1{target.suffix}"))
    assert target.read_text() == "another process's output"


def test_clear_failure_aborts_before_topic_generation(config, tmp_path):
    target = Path(config.output.save_as)
    target.write_text("old output")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(exclude_unset=True)))
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._prompt_with_timeout", return_value="2"),
        patch("deepfabric.cli.clear_output", side_effect=PermissionError("cannot clear output")),
        patch("deepfabric.cli._initialize_topic_model") as initialize,
    ):
        result = CliRunner().invoke(cli, ["generate", str(config_file)])
    assert result.exit_code == 1
    assert "cannot clear output" in result.output
    assert target.read_text() == "old output"
    initialize.assert_not_called()


def test_report_alias_is_rejected_before_clearing_dataset(config, tmp_path):
    target = Path(config.output.save_as)
    target.write_text("keep this output")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(exclude_unset=True)))
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._prompt_with_timeout", return_value="2") as prompt,
        patch("deepfabric.cli._initialize_topic_model") as initialize,
    ):
        result = CliRunner().invoke(
            cli, ["generate", str(config_file), "--run-report", str(target)]
        )
    assert result.exit_code == 1
    assert "aliases" in result.output
    assert target.read_text() == "keep this output"
    prompt.assert_not_called()
    initialize.assert_not_called()
