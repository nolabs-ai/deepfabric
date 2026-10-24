"""Topic template selection and final requests, without provider calls."""
# ruff: noqa: PLR2004

import json

from unittest.mock import AsyncMock, patch

import pytest
import yaml

from click.testing import CliRunner
from pydantic import ValidationError

from deepfabric.cli import cli
from deepfabric.config import DeepFabricConfig, TopicsConfig
from deepfabric.exceptions import TreeError
from deepfabric.graph import Graph
from deepfabric.llm.prompt_trace import PromptPreviewCaptured, prompt_tracing
from deepfabric.prompt_templates import TopicPromptTemplates, template_manifest
from deepfabric.schemas import TopicList
from deepfabric.tree import Tree


@pytest.mark.parametrize("constructor,module", [(Tree, "tree"), (Graph, "graph")])
@pytest.mark.parametrize(
    "templates",
    [
        {"initial": "{{ typo }}"},
        {"expansion": "{{ history }}"},
        {"seed": "wrong stage"},
        {"initial": {"file": "missing-topic-template.txt"}},
    ],
)
def test_invalid_topic_templates_fail_before_client(constructor, module, templates):
    with (
        patch(f"deepfabric.{module}.LLMClient") as client,
        pytest.raises((ValueError, TreeError)),
    ):
        constructor(topic_prompt="Root", prompt_templates=templates)
    client.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["default", "isolated", "anchored"])
async def test_graph_initial_and_expansion_use_full_first_parent_path(style):
    with patch("deepfabric.graph.LLMClient"):
        graph = Graph(
            topic_prompt="Root",
            topic_system_prompt="Persona",
            prompt_style=style,
            prompt_templates={
                "initial": "FIRST {{ current_topic }} {{ requested }} {{ prompt_style }}",
                "expansion": "{{ system_prompt }}|{{ topic_path }}|{{ current_topic }}|{{ num_subtopics }}|{{ mode }}",
            },
        )
    first = graph.add_node("First")
    second = graph.add_node("Second")
    internal = graph.add_node("Internal {{ requested }}")
    graph.add_edge(0, first.id)
    graph.add_edge(0, second.id)
    graph.add_edge(first.id, internal.id)
    graph.add_edge(second.id, internal.id)
    graph._generate_subtopics_with_retry = AsyncMock(return_value=None)
    await graph.get_subtopics_and_connections(graph.root, 3)
    await graph.get_subtopics_and_connections(internal, 2)
    calls = graph._generate_subtopics_with_retry.call_args_list
    assert calls[0].args[0] == f"FIRST Root 3 {style}"
    assert (
        calls[1].args[0]
        == "Persona|Root -> First -> Internal {{ requested }}|Internal {{ requested }}|2|graph"
    )


@pytest.mark.asyncio
async def test_tree_initial_and_expansion_keep_schema_and_temperature():
    with patch("deepfabric.tree.LLMClient"):
        tree = Tree(
            topic_prompt="Root",
            temperature=0.7,
            prompt_templates={
                "initial": "ROOT {{ topic_prompt }} {{ requested }}",
                "expansion": "{{ system_prompt }} {{ topic_path }} {{ requested }} {{ current_graph_summary }}",
            },
        )
    tree.llm_client.generate_async = AsyncMock(return_value=TopicList(subtopics=["Child"]))
    assert await tree.get_subtopics("Persona", ["Root"], 2) == ["Child"]
    await tree.get_subtopics("Persona", ["Root", "Child", "Internal"], 3)
    calls = tree.llm_client.generate_async.call_args_list
    assert calls[0].kwargs["prompt"] == "ROOT Root 2"
    assert calls[1].kwargs["prompt"] == "Persona Root -> Child -> Internal 3 "
    assert all(call.kwargs["schema"] is TopicList for call in calls)
    assert all(call.kwargs["temperature"] == 0.7 for call in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,style",
    [("tree", "tree"), ("graph", "default"), ("graph", "isolated"), ("graph", "anchored")],
)
async def test_default_prompt_wrapper_and_omitted_stage_preserve_defaults(mode, style):
    constructor = Tree if mode == "tree" else Graph
    with patch(f"deepfabric.{mode}.LLMClient"):
        model = constructor(topic_prompt="Root", prompt_style=style)

    async def capture(path):
        if mode == "tree":
            model.llm_client.generate_async = AsyncMock(
                return_value=TopicList(subtopics=["Generated"])
            )
            await model.get_subtopics("Persona", path, 2)
            return model.llm_client.generate_async.call_args.kwargs["prompt"]
        node = model.root if len(path) == 1 else model.add_node(path[-1])
        if node is not model.root:
            model.add_edge(0, node.id)
        model._generate_subtopics_with_retry = AsyncMock(return_value=None)
        await model.get_subtopics_and_connections(node, 2)
        return model._generate_subtopics_with_retry.call_args.args[0]

    original = await capture(["Root"])
    model.config.prompt_templates = TopicPromptTemplates(
        initial="BEFORE\n{{ default_prompt }}\nAFTER"
    )
    assert await capture(["Root"]) == f"BEFORE\n{original}\nAFTER"
    model.config.prompt_templates = TopicPromptTemplates(initial="unused")
    expansion = await capture(["Root", "Child"])
    assert "unused" not in expansion
    assert "Child" in expansion


def test_yaml_relative_files_are_snapshotted_and_reported(tmp_path):
    file = tmp_path / "initial.txt"
    file.write_text("First {{ topic_prompt }}")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "topics": {
                    "prompt": "Root",
                    "prompt_templates": {"initial": {"file": "initial.txt"}},
                },
                "generation": {"system_prompt": "Samples"},
                "output": {"save_as": "out.jsonl", "num_samples": 1},
            }
        )
    )
    config = DeepFabricConfig.from_yaml(str(config_file))
    file.write_text("Changed {{ topic_prompt }}")
    params = config.get_topics_params()
    source = params["prompt_templates"].initial
    assert source.content == "First {{ topic_prompt }}"
    assert source.file == str(file)
    manifest = template_manifest(params["prompt_templates"])
    assert manifest["overrides"]["initial"]["text"] == source.content
    with pytest.raises(ValidationError, match="Unknown or unavailable"):
        TopicsConfig(prompt="Root", prompt_templates={"expansion": "{{ seed }}"})


@pytest.mark.parametrize("mode", ["tree", "graph"])
def test_verbose_preview_traces_both_final_topic_requests_without_models(tmp_path, mode):
    config_file = tmp_path / "config.yaml"
    trace = tmp_path / "trace.jsonl"
    config_file.write_text(
        yaml.safe_dump(
            {
                "llm": {"provider": "openai", "model": "test"},
                "topics": {
                    "prompt": "Root",
                    "mode": mode,
                    "depth": 2,
                    "degree": 2,
                    "prompt_templates": {
                        "initial": "FIRST {{ topic_path }}",
                        "expansion": "LATER {{ topic_path }}",
                    },
                },
                "generation": {"system_prompt": "Samples"},
                "output": {"save_as": "out.jsonl", "num_samples": 1},
            }
        )
    )
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys", side_effect=AssertionError("API validation")),
        patch(
            "deepfabric.llm.client.make_outlines_model", side_effect=AssertionError("provider init")
        ),
        patch(
            "deepfabric.llm.client.make_async_outlines_model",
            side_effect=AssertionError("provider init"),
        ),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "generate",
                str(config_file),
                "--topic-only",
                "--dry-run",
                "--verbose",
                "--trace-prompt",
                "--trace-prompt-file",
                str(trace),
            ],
        )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert records[0]["prompt"] == "FIRST Root"
    assert records[1]["prompt"].startswith("LATER Root -> ")
    assert all(record["dry_run"] for record in records)
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["tree", "graph"])
async def test_live_trace_captures_custom_topic_request_before_provider_call(tmp_path, mode):
    constructor = Tree if mode == "tree" else Graph
    trace = tmp_path / "trace.jsonl"
    with prompt_tracing(enabled=True, path=str(trace), dry_run=True):
        model = constructor(
            topic_prompt="Root",
            provider="openai",
            model_name="test",
            prompt_templates={"initial": "FINAL {{ topic_path }}"},
        )
        with pytest.raises(PromptPreviewCaptured):
            if mode == "tree":
                await model.get_subtopics("Persona", ["Root"], 2)
            else:
                await model.get_subtopics_and_connections(model.root, 2)
    assert json.loads(trace.read_text().splitlines()[0])["prompt"] == "FINAL Root"


def test_run_report_saves_resolved_topic_templates_before_midrun_edits(tmp_path):
    source = tmp_path / "topic.txt"
    source.write_text("ORIGINAL {{ topic_path }}")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "topics": {
                    "prompt": "Root",
                    "prompt_templates": {
                        "initial": {"file": "topic.txt"},
                        "expansion": {"file": "topic.txt"},
                    },
                },
                "generation": {"system_prompt": "Samples"},
                "output": {"save_as": str(tmp_path / "samples.jsonl")},
            }
        )
    )
    report = tmp_path / "report.json"
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli._initialize_topic_model", return_value=object()),
        patch(
            "deepfabric.cli._run_generation",
            side_effect=lambda **_kwargs: source.write_text("EDITED"),
        ),
    ):
        result = CliRunner().invoke(cli, ["generate", str(config), "--run-report", str(report)])
    assert result.exit_code == 0, result.output
    overrides = json.loads(report.read_text())["topic_prompt_templates"]["overrides"]
    assert overrides["initial"]["text"] == "ORIGINAL {{ topic_path }}"
    assert overrides["expansion"]["sha256"] == overrides["initial"]["sha256"]
    assert overrides["initial"]["file"] == str(source)
