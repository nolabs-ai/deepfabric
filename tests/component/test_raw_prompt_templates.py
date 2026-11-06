"""Raw prompt values share source data with framed fields, without reparsing."""
# ruff: noqa: PLR2004

import json

from unittest.mock import AsyncMock, patch

import pytest

from datasets import Dataset as HFDataset

from deepfabric.bulk import completion_schema, make_completion_model, record_validator
from deepfabric.generator import DataSetGenerator
from deepfabric.graph import Graph
from deepfabric.prompt_templates import GenerationPromptTemplates, render_template
from deepfabric.prompts import GraphPromptBuilder, TreePromptBuilder
from deepfabric.schemas import ToolDefinition, ToolParameter, ToolRegistry, TopicList
from deepfabric.tree import Tree

SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}, "rationale": {"type": "string"}},
    "required": ["command", "rationale"],
    "additionalProperties": False,
}
ALLOCATION = {"uuid": "target", "path": ["Root", "Parent", "内部 {{ num_items }}"], "quota": 3}
RAW_FIELDS = (
    "raw_instructions",
    "raw_examples",
    "raw_tools",
    "raw_seed_field",
    "raw_retry_feedback",
    "topic_path_json",
)


def make_engine(**options):
    params = {
        "generation_system_prompt": "Generate",
        "provider": "openai",
        "model_name": "test",
        "output_schema": SCHEMA,
    }
    params.update(options)
    with patch("deepfabric.generator.LLMClient"):
        return DataSetGenerator(**params)


@pytest.mark.parametrize("step", ["full_sample", "sequential_later", "seed", "completion"])
@pytest.mark.parametrize("instructions", ["", " \t\n", "Café {{ num_items }}"])
def test_raw_sample_fields_available_in_every_stage_and_preserve_text(step, instructions):
    text = "|".join("{{ " + name + " }}" for name in RAW_FIELDS)
    engine = make_engine(instructions=instructions, prompt_templates={step: text})
    prompt = engine._apply_prompt_override(step, "unused", ALLOCATION["path"], 0)
    assert prompt.split("|") == [
        instructions,
        "",
        "",
        "",
        "",
        json.dumps(ALLOCATION["path"], ensure_ascii=False, indent=2),
    ]
    assert "内部" in prompt and "{{ num_items }}" in prompt


@pytest.mark.parametrize("path,expected", [(None, ""), ([], "[]")])
def test_raw_path_distinguishes_missing_and_empty_when_completion_ancestry_disabled(path, expected):
    engine = make_engine(prompt_templates={"completion": "{{ subtopics }}|{{ topic_path_json }}"})
    assert engine._apply_prompt_override("completion", "unused", path, 0) == "|" + expected


@pytest.mark.parametrize("data,count,raw", [(None, 1, ""), ([], 0, ""), ([], 1, "[]")])
def test_raw_examples_distinguish_missing_disabled_and_explicitly_empty(data, count, raw):
    engine = make_engine(prompt_templates={"full_sample": "{{ raw_examples }}|{{ examples }}"})
    engine.config.example_data = None if data is None else HFDataset.from_list(data)
    prompt = engine.build_prompt(engine._get_prompt_template(), count, ALLOCATION["path"])
    actual, framed = prompt.split("|", 1)
    assert actual == raw
    assert ("<examples>" in framed) == (data == [] and count > 0)


@pytest.mark.parametrize(
    "mode", ["independent", "sequential", "bulk_seed", "bulk_full", "cot_seed", "completion"]
)
def test_examples_selected_once_and_shared_across_views(mode):
    frame = "{{ raw_examples }}\n--FRAMED--\n{{ examples }}"
    options = {
        "prompt_templates": dict.fromkeys(
            ("full_sample", "sequential_later", "seed", "completion"), frame
        )
    }
    if mode == "sequential":
        options.update(topic_sample_mode="sequential", shared_scope="full")
    elif mode != "independent":
        options.update(
            topic_sample_mode="bulk", shared_scope="full" if mode == "bulk_full" else "seed"
        )
        if mode in ("cot_seed", "completion"):
            options.update(output_schema=None, conversation_type="cot")
        else:
            options["seed_field"] = "command"
    engine = make_engine(**options)
    selected = [{"command": "café {{ seed }}", "rationale": "selected"}]
    engine.config.example_data = HFDataset.from_list(
        selected + [{"command": "other", "rationale": "unused"}]
    )
    with (
        patch("deepfabric.generator.random.sample", return_value=selected) as sample,
        patch("deepfabric.generator.render_template", wraps=render_template) as render,
    ):
        if mode == "independent":
            prompt = engine.build_prompt(engine._get_prompt_template(), 1, ALLOCATION["path"])
        elif mode == "sequential":
            prompt = engine._build_sequential_prompt(ALLOCATION, ["earlier"], 1)
        elif mode == "completion":
            prompt = engine._conversation_completion_prompt(ALLOCATION, "Question", 1)
        else:
            prompt = engine._build_bulk_collection_prompt(ALLOCATION, 3, [], 1)
    sample.assert_called_once()
    assert render.call_args.args[1]["raw_examples"] is selected
    raw, framed = prompt.split("\n--FRAMED--\n")
    assert json.loads(raw) == selected
    assert repr(selected[0]) in framed and "other" not in framed
    assert "{{ seed }}" in raw and "café" in raw


def test_raw_tools_uses_resolved_tool_definition_shape_and_preserves_nested_values():
    tool = ToolDefinition(
        name="lookup",
        description="Café {{ num_items }}",
        returns="Found data",
        parameters=[
            ToolParameter(
                name="flag", type="bool", description="Toggle", required=False, default="False"
            )
        ],
        category="search",
        component="lookup-component",
    )
    engine = make_engine(
        prompt_templates={"full_sample": "{{ raw_tools }}|{{ tools }}|{{ tools_section }}"}
    )
    engine.tool_registry = ToolRegistry(tools=[tool])
    raw, tools, section = engine.build_prompt(engine._get_prompt_template(), 0).split("|")
    assert json.loads(raw) == [tool.model_dump(mode="json")]
    assert "function" not in json.loads(raw)[0]
    assert "Parameters:" in tools and "(optional)" in tools
    assert section == f"<tools>\n{tools}\n</tools>"
    assert json.loads(raw)[0]["parameters"][0]["required"] is False
    engine.tool_registry = ToolRegistry(tools=[])
    assert engine.build_prompt(engine._get_prompt_template(), 0) == "||"


async def test_custom_completion_raw_error_is_original_source_and_examples_selected_once_per_attempt():
    frame = "{{ raw_retry_feedback }}|{{ retry_feedback }}|{{ raw_seed_field }}|{{ seed_field }}|{{ raw_examples }}|{{ examples }}"
    engine = make_engine(
        topic_sample_mode="bulk",
        seed_field="command",
        sample_retries=1,
        prompt_templates={"completion": frame},
    )
    selected = [{"command": "café", "rationale": "reason"}]
    engine.config.example_data = HFDataset.from_list(selected)
    remainder = completion_schema(SCHEMA, "command")
    model = make_completion_model(remainder)
    error = ValueError("RETRY: validation error {{ num_items }}")
    engine.llm_client.generate_async = AsyncMock(
        side_effect=[error, model.model_validate_json('{"rationale": "done"}')]
    )
    with patch("deepfabric.generator.random.sample", return_value=selected) as sample:
        row, failure = await engine._complete_custom_seed_outcome(
            ALLOCATION,
            0,
            0,
            "seed",
            model,
            record_validator(remainder),
            record_validator(SCHEMA),
            1,
        )
    assert failure is None and row["command"] == "seed"
    assert sample.call_count == 2
    first, second = [
        call.args[0].split("|") for call in engine.llm_client.generate_async.call_args_list
    ]
    assert first[:4] == ["", "", "command", '"command"']
    assert second[:4] == [str(error), "RETRY: " + str(error), "command", '"command"']
    assert json.loads(first[4]) == selected and repr(selected[0]) in first[5]


@pytest.mark.parametrize("step", ["initial", "expansion"])
@pytest.mark.parametrize("style", ["default", "isolated", "anchored"])
async def test_raw_graph_is_uncapped_and_keeps_canonical_ancestry(step, style):
    frame = "{{ raw_graph }}|{{ raw_tree }}|{{ topic_path_json }}|{{ raw_examples }}|{{ examples }}"
    with patch("deepfabric.graph.LLMClient"):
        graph = Graph(topic_prompt="Root", prompt_style=style, prompt_templates={step: frame})
    first = graph.add_node("First")
    second = graph.add_node("Second")
    graph.add_edge(graph.root.id, first.id)
    graph.add_edge(graph.root.id, second.id)
    target = graph.add_node("内部 {{ num_items }}")
    graph.add_edge(first.id, target.id)
    graph.add_edge(second.id, target.id)
    for index in range(25):
        graph.add_node(f"Extra {index}")
    node = graph.root if step == "initial" else target
    graph._generate_subtopics_with_retry = AsyncMock(return_value=None)
    await graph.get_subtopics_and_connections(node, 3)
    raw, tree, path, examples, framed = graph._generate_subtopics_with_retry.call_args.args[
        0
    ].split("|")
    assert json.loads(raw) == json.loads(graph.to_json())
    assert len(json.loads(raw)["nodes"]) > 20 and tree == ""
    assert json.loads(path) == (["Root"] if step == "initial" else ["Root", "First", target.topic])
    domain = GraphPromptBuilder.detect_domain(graph.model_system_prompt, json.loads(path))
    assert (
        json.loads(examples)
        == GraphPromptBuilder.EXAMPLES[domain][: GraphPromptBuilder.MAX_PROMPT_EXAMPLES]
    )
    assert framed == GraphPromptBuilder._format_examples(json.loads(examples))


@pytest.mark.parametrize("path", [["Root"], ["Root", "Parent", "内部 {{ num_items }}"]])
@pytest.mark.parametrize("recorded_paths", [[], [["Root", "Completed"]]])
async def test_raw_tree_snapshot_contains_recorded_paths_and_is_blank_for_graph(
    path, recorded_paths
):
    frame = "{{ raw_tree }}|{{ raw_graph }}|{{ topic_path_json }}|{{ raw_examples }}|{{ examples }}"
    with patch("deepfabric.tree.LLMClient"):
        tree = Tree(topic_prompt="Root", prompt_templates={"initial": frame, "expansion": frame})
    tree.tree_paths = recorded_paths
    tree.llm_client.generate_async = AsyncMock(return_value=TopicList(subtopics=["Child"]))
    await tree.get_subtopics("Persona", path, 2)
    raw, graph, ancestry, examples, framed = tree.llm_client.generate_async.call_args.kwargs[
        "prompt"
    ].split("|")
    assert json.loads(raw) == tree.to_dict() and graph == ""
    assert json.loads(ancestry) == path
    domain = tree._detect_domain("Persona", path)
    assert (
        json.loads(examples)
        == TreePromptBuilder.EXAMPLES.get(domain, TreePromptBuilder.EXAMPLES["general"])[:3]
    )
    assert framed == TreePromptBuilder._format_examples(json.loads(examples))


async def test_graph_snapshot_not_built_when_raw_graph_is_unused():
    with patch("deepfabric.graph.LLMClient"):
        graph = Graph(
            topic_prompt="Root",
            prompt_style="isolated",
            prompt_templates={"initial": "{{ current_topic }}"},
        )
    graph._generate_subtopics_with_retry = AsyncMock(return_value=None)
    for index in range(21):
        graph.add_node(f"Extra {index}")
    with patch.object(graph, "to_pydantic", side_effect=AssertionError("unused snapshot")):
        await graph.get_subtopics_and_connections(graph.root, 2)


@pytest.mark.parametrize("mode", ["tree", "graph"])
async def test_empty_topic_examples_are_json_array_without_framing(mode):
    constructor = Tree if mode == "tree" else Graph
    builder = TreePromptBuilder if mode == "tree" else GraphPromptBuilder
    with patch(f"deepfabric.{mode}.LLMClient"):
        model = constructor(
            topic_prompt="Root", prompt_templates={"initial": "{{ raw_examples }}|{{ examples }}"}
        )
    with patch.dict(builder.EXAMPLES, dict.fromkeys(builder.EXAMPLES, [])):
        if mode == "tree":
            model.llm_client.generate_async = AsyncMock(return_value=TopicList(subtopics=["Child"]))
            await model.get_subtopics("", ["Root"], 2)
            prompt = model.llm_client.generate_async.call_args.kwargs["prompt"]
        else:
            model._generate_subtopics_with_retry = AsyncMock(return_value=None)
            await model.get_subtopics_and_connections(model.root, 2)
            prompt = model._generate_subtopics_with_retry.call_args.args[0]
    assert prompt == "[]|"


def test_json_renderer_preserves_nested_empty_and_false_values_without_reparsing():
    value = {"array": [], "zero": 0, "false": False, "null": None, "text": "café {{ seed }}"}
    assert json.loads(render_template("{{ raw_graph }}", {"raw_graph": value})) == value
    assert render_template("{{ raw_graph }}", {"raw_graph": None}) == ""
    assert render_template("{{ num_items }}", {"num_items": 0}) == "0"
    for step in ("full_sample", "sequential_later", "seed", "completion"):
        GenerationPromptTemplates(**{step: " ".join("{{ " + field + " }}" for field in RAW_FIELDS)})
