"""Offline CLI acceptance checks for tracing and prompt previews."""
# ruff: noqa: PLR2004, ARG001

import json

from unittest.mock import patch

import pytest
import yaml

from click.testing import CliRunner

from deepfabric.cli import cli
from deepfabric.llm.prompt_trace import get_prompt_trace


@pytest.fixture
def config_data():
    return {
        "llm": {"provider": "openai", "model": "preview-model"},
        "topics": {"prompt": "Ocean science", "mode": "tree", "depth": 2, "degree": 2},
        "generation": {"system_prompt": "Write a useful lesson.", "instructions": "Be precise."},
        "output": {"save_as": "dataset.jsonl", "num_samples": 4, "batch_size": 1},
    }


@pytest.fixture
def offline_guard():
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
        patch(
            "deepfabric.generator.load_tools_from_endpoint",
            side_effect=AssertionError("tool discovery"),
        ),
        patch(
            "deepfabric.cli._initialize_topic_model", side_effect=AssertionError("topic build/save")
        ),
        patch("deepfabric.cli._run_generation", side_effect=AssertionError("dataset output")),
    ):
        yield


def run_preview(tmp_path, config, *flags):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    return CliRunner().invoke(cli, ["generate", str(config_file), "--dry-run", *flags])


@pytest.mark.parametrize("mode", ["tree", "graph"])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("sample_mode", ["independent", "bulk", "sequential"])
def test_preview_uses_production_composition_without_provider_calls(
    tmp_path, config_data, offline_guard, mode, custom, sample_mode
):
    config_data["topics"]["mode"] = mode
    config_data["generation"]["topic_sample_mode"] = sample_mode
    if custom:
        config_data["generation"]["output_schema"] = {
            "type": "object",
            "properties": {"seed": {"type": "string"}, "answer": {"type": "string"}},
            "required": ["seed", "answer"],
            "additionalProperties": False,
        }
        config_data["generation"]["seed_field"] = "seed"
    trace_file = tmp_path / "preview.jsonl"
    result = run_preview(
        tmp_path, config_data, "--trace-prompt", "--trace-prompt-file", str(trace_file)
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    assert records[0]["schema_name"] == ("TopicList" if mode == "tree" else "GraphSubtopics")
    assert "Ocean science" in records[0]["prompt"]
    assert "{{{topic path}}}" in result.output
    assert "Be precise." in result.output
    assert all(record["dry_run"] for record in records)
    assert len({record["schema_name"] for record in records}) == len(records)
    assert get_prompt_trace() is None
    assert not (tmp_path / "dataset.jsonl").exists()


def test_agent_preview_includes_all_stages_and_remote_placeholders(
    tmp_path, config_data, offline_guard
):
    config_data["generation"]["conversation"] = {"type": "cot", "reasoning_style": "agent"}
    config_data["generation"]["tools"] = {
        "components": {"remote": ["lookup"]},
        "tools_endpoint": "http://localhost:9999/tools",
        "spin_endpoint": "http://localhost:9999",
    }
    result = run_preview(tmp_path, config_data)
    assert result.exit_code == 0, result.output
    for stage in ["UserQuestion", "AgentStep", "AgentResponse"]:
        assert f"--- {stage} " in result.output
    assert "{{{remote tool description and parameters}}}" in result.output
    assert "{{{tool result}}}" in result.output
    assert "{{{user request}}}" in result.output


@pytest.mark.parametrize("verbose, expected", [(False, 1), (True, 2)])
def test_verbose_preview_shows_first_and_dependent_cases(
    tmp_path, config_data, offline_guard, verbose, expected
):
    config_data["generation"]["topic_sample_mode"] = "sequential"
    result = run_preview(tmp_path, config_data, *(["--verbose"] if verbose else []))
    assert result.exit_code == 0, result.output
    assert result.output.count("--- TopicList ") == expected
    assert result.output.count("--- Conversation ") == expected
    assert ("{{{previous successful samples}}}" in result.output) == verbose


def test_topic_only_preview_skips_samples(tmp_path, config_data, offline_guard):
    result = run_preview(tmp_path, config_data, "--topic-only")
    assert result.exit_code == 0, result.output
    assert "--- TopicList " in result.output
    assert "--- Conversation " not in result.output


def test_cli_rejects_path_without_enabling_trace():
    result = CliRunner().invoke(cli, ["generate", "--trace-prompt-file", "unused.jsonl"])
    assert result.exit_code == 2
    assert "requires --trace-prompt" in result.output


@pytest.mark.parametrize(
    "trace_enabled, verbose, expected", [(False, True, 0), (True, False, 2), (True, True, 4)]
)
def test_normal_cli_flag_gates_file_and_selects_verbosity(
    tmp_path, config_data, offline_guard, monkeypatch, trace_enabled, verbose, expected
):
    config_file = tmp_path / "normal.yaml"
    config_file.write_text(yaml.safe_dump(config_data))
    trace_file = tmp_path / "normal-trace.jsonl"
    monkeypatch.setenv("DEEPFABRIC_PROMPT_TRACE_FILE", str(trace_file))

    def request_pair(schema):
        session = get_prompt_trace()
        if session is not None:
            for prompt in ["first", "later"]:
                session.record("openai", "preview-model", prompt, schema)

    def topic_stage(**_kwargs):
        request_pair({"title": "topics"})
        return object()

    def sample_stage(**_kwargs):
        request_pair({"title": "samples"})

    with (
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._initialize_topic_model", side_effect=topic_stage),
        patch("deepfabric.cli._run_generation", side_effect=sample_stage),
    ):
        args = ["generate", str(config_file), "--tui", "simple"]
        if trace_enabled:
            args.append("--trace-prompt")
        if verbose:
            args.append("--verbose")
        result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    records = trace_file.read_text().splitlines() if trace_file.exists() else []
    assert len(records) == expected
    assert "first" not in result.output
    assert "later" not in result.output
    assert get_prompt_trace() is None


@pytest.mark.parametrize("style", ["default", "anchored", "isolated"])
def test_graph_preview_respects_prompt_style(tmp_path, config_data, offline_guard, style):
    config_data["topics"].update(mode="graph", prompt_style=style)
    result = run_preview(
        tmp_path, config_data, "--verbose", "--provider", "gemini", "--model", "preview-override"
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("--- GraphSubtopics (gemini/preview-override) ---") == 2
    assert "{{{topic path}}}" in result.output


def test_loading_topics_skips_expansion_preview(tmp_path, config_data, offline_guard):
    topics = tmp_path / "existing.jsonl"
    topics.write_text("unused: dry runs use placeholders")
    result = run_preview(tmp_path, config_data, "--topics-load", str(topics))
    assert result.exit_code == 0, result.output
    assert "--- TopicList " not in result.output
    assert "--- Conversation " in result.output
    assert "{{{topic path}}}" in result.output


def test_seed_only_bulk_preview_has_no_completion_request(tmp_path, config_data, offline_guard):
    config_data["generation"].update(
        topic_sample_mode="bulk",
        seed_field="seed",
        output_schema={
            "type": "object",
            "properties": {"seed": {"type": "integer"}},
            "required": ["seed"],
            "additionalProperties": False,
        },
    )
    result = run_preview(tmp_path, config_data)
    assert result.exit_code == 0, result.output
    assert "--- CollectionModel " in result.output
    assert "--- DynamicModel " not in result.output
