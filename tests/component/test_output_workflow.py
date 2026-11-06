"""Startup decisions are read-only until commitment; retry uses real checkpoint state."""
# ruff: noqa: PLR2004

import json

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml

from click.testing import CliRunner
from rich.text import Text

from deepfabric.cli import GenerateOptions, _prepare_output_destinations, cli
from deepfabric.config import DeepFabricConfig
from deepfabric.constants import (
    CHECKPOINT_FAILURES_SUFFIX,
    CHECKPOINT_METADATA_SUFFIX,
    CHECKPOINT_SAMPLES_SUFFIX,
)
from deepfabric.dataset_manager import create_dataset
from deepfabric.exceptions import ConfigurationError
from deepfabric.llm.prompt_trace import prompt_tracing
from deepfabric.output_safety import output_session
from deepfabric.run_report import RunReport
from deepfabric.tree import Tree
from tests.component.test_run_report import SCHEMA, make_engine


@pytest.fixture
def workflow(tmp_path):
    config = DeepFabricConfig.model_validate(
        {
            "llm": {"provider": "openai", "model": "test"},
            "topics": {"prompt": "test", "mode": "tree", "save_as": str(tmp_path / "topics.jsonl")},
            "generation": {"system_prompt": "test", "output_schema": SCHEMA},
            "output": {
                "save_as": str(tmp_path / "dataset.jsonl"),
                "num_samples": 2,
                "batch_size": 1,
                "checkpoint": {"interval": 1, "path": str(tmp_path / "checkpoints")},
            },
        }
    )
    return SimpleNamespace(
        config=config, generation_overrides={}, topics_overrides={}, num_samples=2
    )


def existing_files(workflow, tmp_path, *, checkpoints=True):
    paths = [
        Path(workflow.config.topics.save_as),
        Path(workflow.config.output.save_as),
        tmp_path / "report.json",
        tmp_path / "trace.jsonl",
    ]
    if checkpoints:
        directory = tmp_path / "checkpoints"
        directory.mkdir(exist_ok=True)
        paths.extend(
            directory / f"dataset{suffix}"
            for suffix in (
                CHECKPOINT_METADATA_SUFFIX,
                CHECKPOINT_SAMPLES_SUFFIX,
                CHECKPOINT_FAILURES_SUFFIX,
            )
        )
    for path in paths:
        path.write_text("old")
    return paths


def test_summary_precedes_prompts_and_final_abort_preserves_every_file(workflow, tmp_path):
    paths = existing_files(workflow, tmp_path)
    before = {path: path.read_bytes() for path in paths}
    options = GenerateOptions(run_report_path=str(tmp_path / "report.json"))
    tui = Mock()
    decisions = iter(["2", "2", "3"])

    def choose(*_args, **_kwargs):
        assert {path: path.read_bytes() for path in paths} == before
        texts = [
            call.args[0]
            for call in tui.console.print.call_args_list
            if isinstance(call.args[0], Text)
        ]
        # All enabled destinations have already been displayed before the first choice.
        for path in paths:
            assert any(str(path) in text.plain for text in texts)
        return next(decisions)

    with (
        output_session(),
        prompt_tracing(enabled=True, path=str(tmp_path / "trace.jsonl")),
        patch("deepfabric.cli.get_tui", return_value=tui),
        patch("deepfabric.cli._prompt_with_timeout", side_effect=choose),
        pytest.raises(SystemExit) as stopped,
    ):
        _prepare_output_destinations(workflow, options)
    assert stopped.value.code == 0
    assert {path: path.read_bytes() for path in paths} == before
    prompts = [
        call.args[0] for call in tui.console.print.call_args_list if isinstance(call.args[0], str)
    ]
    assert prompts.index("  1) Resume from checkpoint") < prompts.index(
        "  4) Use existing topics (resume progress if requested)"
    )
    topic_notice = next(
        i
        for i, call in enumerate(tui.console.print.call_args_list)
        if isinstance(call.args[0], Text) and "topic output:" in call.args[0].plain
    )
    dataset_notice = next(
        i
        for i, call in enumerate(tui.console.print.call_args_list)
        if isinstance(call.args[0], Text)
        and "dataset output:" in call.args[0].plain
        and i > topic_notice
    )
    assert topic_notice < dataset_notice


def test_summary_styles_identity_but_never_full_path(workflow, tmp_path):
    existing_files(workflow, tmp_path, checkpoints=False)
    tui = Mock()
    with (
        patch("deepfabric.cli.get_tui", return_value=tui),
        patch("deepfabric.cli._prompt_with_timeout", return_value="1"),
    ):
        _prepare_output_destinations(workflow, GenerateOptions())
    texts = [
        call.args[0] for call in tui.console.print.call_args_list if isinstance(call.args[0], Text)
    ]
    assert texts
    for text in texts:
        path_start = text.plain.index(" | ") + len(" | ")
        assert text.style == ""
        assert text.spans and all(span.end <= path_start for span in text.spans)
        assert all(
            call.kwargs.get("highlight") is False
            for call in tui.console.print.call_args_list
            if call.args[0] is text
        )


@pytest.mark.parametrize("choices", [["2", "1"], ["1", "2"], ["2", "2"]])
def test_fresh_output_clear_preserves_enabled_diagnostics(workflow, tmp_path, choices):
    paths = existing_files(workflow, tmp_path, checkpoints=False)
    suffixes = [tmp_path / "report_1.json", tmp_path / "trace_1.jsonl", tmp_path / "unrelated.json"]
    for path in suffixes:
        path.write_text("keep")
    options = GenerateOptions(run_report_path=str(tmp_path / "report.json"))
    with (
        output_session(),
        prompt_tracing(enabled=True, path=str(tmp_path / "trace.jsonl")) as trace,
        patch("deepfabric.cli._prompt_with_timeout", side_effect=choices),
    ):
        _prepare_output_destinations(workflow, options)
        actual_report = RunReport().write(options.run_report_path, {}, "completed")
        trace.record("test", "model", "prompt", {})
        assert actual_report == str(tmp_path / "report_2.json")
        assert trace.path == tmp_path / "trace_2.jsonl"
    for path, choice in zip(paths[:2], choices, strict=True):
        if choice == "2":
            assert not path.exists()
        else:
            assert path.read_text() == "old"
    assert all(path.read_text() == "old" for path in paths[2:])
    assert all(path.read_text() == "keep" for path in suffixes)


def test_fresh_clear_preserves_disabled_diagnostics(workflow, tmp_path):
    existing_files(workflow, tmp_path, checkpoints=False)
    with output_session(), patch("deepfabric.cli._prompt_with_timeout", return_value="2"):
        _prepare_output_destinations(workflow, GenerateOptions())
    assert (tmp_path / "report.json").read_text() == "old"
    assert (tmp_path / "trace.jsonl").read_text() == "old"


def test_checkpoint_clear_alone_preserves_existing_outputs_and_diagnostics(workflow, tmp_path):
    paths = existing_files(workflow, tmp_path)
    options = GenerateOptions(run_report_path=str(tmp_path / "report.json"))
    with (
        output_session(),
        prompt_tracing(enabled=True, path=str(tmp_path / "trace.jsonl")),
        patch("deepfabric.cli._prompt_with_timeout", side_effect=["2", "1", "1"]),
    ):
        _prepare_output_destinations(workflow, options)
    assert all(path.read_text() == "old" for path in paths[:4])
    assert all(not path.exists() for path in paths[4:])


@pytest.mark.parametrize("resume", [False, True])
def test_continue_preserves_loaded_topics_and_old_diagnostics(workflow, tmp_path, resume):
    paths = existing_files(workflow, tmp_path, checkpoints=resume)
    if resume:
        metadata = tmp_path / "checkpoints" / "dataset.checkpoint.json"
        metadata.write_text(json.dumps({"topics_file": workflow.config.topics.save_as}))
    options = GenerateOptions(resume=resume, run_report_path=str(tmp_path / "report.json"))
    choices = ["2"] if resume else ["4", "2"]
    with (
        output_session(),
        prompt_tracing(enabled=True, path=str(tmp_path / "trace.jsonl")),
        patch("deepfabric.cli._prompt_with_timeout", side_effect=choices),
    ):
        _prepare_output_destinations(workflow, options)
    assert not Path(workflow.config.output.save_as).exists()
    assert all(path.exists() for path in paths if path != Path(workflow.config.output.save_as))
    assert options.topics_load == workflow.config.topics.save_as


@pytest.mark.parametrize("kind", ["missing", "disabled", "clear"])
def test_retry_without_loaded_checkpoint_preserves_outputs(workflow, tmp_path, kind):
    paths = existing_files(workflow, tmp_path, checkpoints=kind != "missing")
    before = {path: path.read_bytes() for path in paths}
    if kind == "disabled":
        workflow.config.output.checkpoint = None
    options = GenerateOptions(resume=kind != "clear", retry_failed=True)
    with (
        patch("deepfabric.cli._prompt_with_timeout", return_value="2"),
        pytest.raises(ConfigurationError, match="actually loaded"),
    ):
        _prepare_output_destinations(workflow, options)
    assert {path: path.read_bytes() for path in paths} == before


def test_retry_dry_run_is_rejected_without_preview_or_file_changes(workflow, tmp_path):
    paths = existing_files(workflow, tmp_path)
    before = {path: path.read_bytes() for path in paths}
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli.preview_generation") as preview,
    ):
        result = CliRunner().invoke(cli, ["generate", "--retry-failed", "--dry-run"])
    assert result.exit_code == 2
    assert "retry requires loading a checkpoint" in result.output
    assert {path: path.read_bytes() for path in paths} == before
    preview.assert_not_called()


def build_retry_state(workflow, tmp_path):
    topics_path = Path(workflow.config.topics.save_as)
    engine, calls = make_engine(
        tmp_path / "checkpoints",
        "independent",
        invalid_first=True,
        sample_retries=0,
        checkpoint_interval=1,
        output_save_as=workflow.config.output.save_as,
        topics_file=str(topics_path),
    )
    with patch("deepfabric.tree.LLMClient", return_value=engine.llm_client):
        tree = Tree(topic_prompt="test", provider="openai", model_name="test")
    tree.tree_paths = [["root", "first"], ["root", "second"]]
    topics_path.write_text("".join(json.dumps({"path": path}) + "\n" for path in tree.tree_paths))
    dataset = engine.create_data(num_samples=2, batch_size=1, topic_model=tree)
    assert len(engine.get_all_failures()) == 1
    return engine, calls, dataset


@pytest.mark.parametrize("explicit_resume", [False, True])
def test_real_retry_replaces_selected_override_without_dataset_prompt(
    workflow, tmp_path, explicit_resume
):
    engine, calls, _dataset = build_retry_state(workflow, tmp_path)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(workflow.config.model_dump(exclude_unset=True)))
    target = tmp_path / "override.jsonl"
    target.write_text("old dataset")
    report = tmp_path / "report.json"
    report.write_text("old report")
    trace = tmp_path / "trace.jsonl"
    trace.write_text("old trace")
    args = [
        "generate",
        str(config_file),
        "--retry-failed",
        "--output-save-as",
        str(target),
        "--run-report",
        str(report),
        "--trace-prompt",
        "--trace-prompt-file",
        str(trace),
    ]
    if explicit_resume:
        args.append("--resume")

    def generate_with_old_output_intact(**kwargs):
        assert target.read_text() == "old dataset"
        return create_dataset(**kwargs)

    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli.DataSetGenerator", return_value=engine) as construct,
        patch("deepfabric.tree.LLMClient", return_value=engine.llm_client),
        patch("deepfabric.cli._prompt_with_timeout", return_value="1") as prompt,
        patch.object(engine, "load_checkpoint", wraps=engine.load_checkpoint) as load,
        patch("deepfabric.cli.create_dataset", side_effect=generate_with_old_output_intact),
    ):
        result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert len(target.read_text().splitlines()) == 2
    assert not target.with_name("override_1.jsonl").exists()
    assert "will be replaced only after a successful save" in " ".join(result.output.split())
    assert prompt.call_count == (0 if explicit_resume else 1)
    assert construct.call_count == load.call_count == 1
    assert len(calls) == 3  # Two original requests, one reopened failed slot.
    assert report.read_text() == "old report"
    assert (tmp_path / "report_1.json").exists()
    assert trace.read_text() == "old trace"
    assert (tmp_path / "trace_1.jsonl").exists()


@pytest.mark.parametrize("rejection", ["version", "integrity", "topics"])
def test_rejected_retry_checkpoint_preserves_all_files(workflow, tmp_path, rejection):
    engine, _calls, _dataset = build_retry_state(workflow, tmp_path)
    metadata = tmp_path / "checkpoints" / "dataset.checkpoint.json"
    document = json.loads(metadata.read_text())
    if rejection == "version":
        document["version"] = -1
    elif rejection == "integrity":
        document["total_samples"] += 1
    else:
        Path(workflow.config.topics.save_as).write_text('{"path": ["different"]}\n')
    metadata.write_text(json.dumps(document))
    target = Path(workflow.config.output.save_as)
    target.write_text("old dataset")
    before = {
        path: path.read_bytes() for path in (metadata, target, engine._checkpoint_failures_path)
    }
    with (
        patch("deepfabric.cli.DataSetGenerator", return_value=engine),
        pytest.raises(Exception, match="Unsupported checkpoint version|inconsistent|plan differs"),
    ):
        _prepare_output_destinations(workflow, GenerateOptions(resume=True, retry_failed=True))
    assert {path: path.read_bytes() for path in before} == before
