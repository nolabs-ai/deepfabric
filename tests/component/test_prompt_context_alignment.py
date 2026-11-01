"""Generation modes share context framing and preview the actual collection quota."""
# ruff: noqa: PLR2004

import json

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from click.testing import CliRunner
from datasets import Dataset as HFDataset

from deepfabric.bulk import completion_schema, make_completion_model, record_validator
from deepfabric.cli import cli
from deepfabric.generator import DataSetGenerator

SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}, "rationale": {"type": "string"}},
    "required": ["command", "rationale"],
    "additionalProperties": False,
}
ALLOCATION = {"uuid": "target", "path": ["Root", "Parent", "Internal target"], "quota": 5}


@pytest.fixture
def engine():
    with patch("deepfabric.generator.LLMClient"):
        return DataSetGenerator(
            generation_system_prompt="GENERATION_PERSONA",
            dataset_system_prompt="DATASET_PERSONA",
            instructions="Follow the supplied topic context.",
            provider="openai",
            model_name="test",
            output_schema=SCHEMA,
            topic_sample_mode="bulk",
            shared_scope="seed",
            seed_field="command",
        )


def test_collection_uses_independent_context_and_omits_empty_optional_sections(engine):
    independent = engine.build_prompt(engine._get_prompt_template(), 0, ALLOCATION["path"])
    bulk = engine._build_bulk_collection_prompt(ALLOCATION, 5, [], 0)
    for context in [
        engine.generation_prompt,
        engine.build_custom_instructions_text(),
        engine.build_subtopics_text(ALLOCATION["path"]),
    ]:
        assert context in independent and context in bulk
    assert bulk.startswith(engine.generation_prompt)
    assert 'Generate 5 diverse values for the seed field "command".' in bulk
    assert "<topic_path>" not in bulk
    assert "<accepted_siblings>" not in bulk
    assert "<tools>" not in bulk
    assert "No tools available" not in bulk


def test_retry_uses_singular_and_only_actual_accepted_siblings(engine):
    prompt = engine._build_bulk_collection_prompt(ALLOCATION, 1, ["previous command"], 0)
    assert 'Generate 1 value for the seed field "command".' in prompt
    assert "diverse" not in prompt
    assert '<accepted_siblings>\n["previous command"]\n</accepted_siblings>' in prompt


def test_configured_examples_use_the_same_framing_in_independent_and_bulk(engine):
    engine.config.example_data = HFDataset.from_list(
        [{"command": "docker ps", "rationale": "Read-only listing"}]
    )
    independent = engine.build_prompt(engine._get_prompt_template(), 1, ALLOCATION["path"])
    bulk = engine._build_bulk_collection_prompt(ALLOCATION, 5, [], 1)
    context = engine.build_examples_text(1)
    assert context in independent and context in bulk


def test_sequential_context_matches_independent_and_history_names_seed_field(engine):
    engine.config.topic_sample_mode = "sequential"
    independent = engine.build_prompt(engine._get_prompt_template(), 0, ALLOCATION["path"])
    first = engine._build_sequential_prompt(ALLOCATION, [], 0)
    common = "\n".join(
        [
            engine.generation_prompt,
            engine.build_custom_instructions_text(),
            engine.build_examples_text(0),
            engine.build_subtopics_text(ALLOCATION["path"]),
        ]
    )
    assert independent.startswith(common) and first.startswith(common)
    assert "DATASET_PERSONA" not in first
    assert "prior_successes" not in first and "Earlier" not in first
    assert "Return one complete record as a single JSON object" in first
    later = engine._build_sequential_prompt(ALLOCATION, ["docker ps"], 0)
    assert 'Here are earlier values of "command".' in later
    assert '<prior_successes scope="seed">["docker ps"]</prior_successes>' in later
    assert (
        later.index("</prior_successes>")
        < later.index("Use them to avoid")
        < later.index("Return one complete record")
    )
    engine.config.shared_scope = "full"
    full = engine._build_sequential_prompt(ALLOCATION, [{"command": "docker ps"}], 0)
    assert "Here are earlier complete records." in full
    engine.config.output_schema = None
    engine.config.shared_scope = "seed"
    assert "Here are earlier user requests." in engine._build_sequential_prompt(
        ALLOCATION, ["hello"], 0
    )


def test_tools_are_only_included_when_configured(engine):
    engine.tool_registry = SimpleNamespace(
        tools=[SimpleNamespace(name="lookup", description="Find facts", parameters=[])]
    )
    prompt = engine._build_bulk_collection_prompt(ALLOCATION, 5, [], 0)
    assert "<tools>" in prompt and "lookup" in prompt
    assert "No tools available" not in prompt


@pytest.mark.parametrize("include_path", [False, True])
async def test_completion_retains_instructions_with_explicit_context_scope(engine, include_path):
    engine.config.completion_includes_topic_path = include_path
    engine.llm_client.generate_async = AsyncMock(
        return_value=SimpleNamespace(model_dump=lambda: {"rationale": "read-only"})
    )
    schema = completion_schema(SCHEMA, "command")
    sample, failure = await engine._complete_custom_seed_outcome(
        ALLOCATION,
        0,
        0,
        "docker ps",
        make_completion_model(schema),
        record_validator(schema),
        record_validator(SCHEMA),
        0,
    )
    assert failure is None and sample["command"] == "docker ps"
    prompt = engine.llm_client.generate_async.call_args.args[0]
    assert "Follow the supplied topic context." in prompt
    assert "DATASET_PERSONA" in prompt
    assert (engine.build_subtopics_text(ALLOCATION["path"]) in prompt) is include_path
    assert (
        "Interpret instructions about supplied topic context using that seed." in prompt
    ) is not include_path
    assert "<tools>" not in prompt
    assert "<topic_path>" not in prompt


def preview_config(tmp_path, *, target_count=15):
    graph_path = tmp_path / "graph.json"
    nodes = {
        "0": {
            "id": 0,
            "topic": "Root",
            "children": list(range(1, target_count + 1)),
            "metadata": {"uuid": "root"},
        }
    }
    nodes.update(
        {
            str(i): {
                "id": i,
                "topic": f"Target {i}",
                "children": [],
                "parents": [0],
                "metadata": {"uuid": f"uuid-{i}"},
            }
            for i in range(1, target_count + 1)
        }
    )
    graph_path.write_text(json.dumps({"root_id": 0, "nodes": nodes}))
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "llm": {"provider": "openai", "model": "test"},
                "topics": {"prompt": "Root", "mode": "graph"},
                "generation": {
                    "system_prompt": "GENERATION_PERSONA",
                    "instructions": "Follow the supplied topic context.",
                    "topic_sample_mode": "bulk",
                    "shared_scope": "seed",
                    "seed_field": "command",
                    "output_schema": SCHEMA,
                },
                "output": {
                    "save_as": str(tmp_path / "output.jsonl"),
                    "num_samples": 75,
                    "batch_size": 1,
                },
            }
        )
    )
    return config, graph_path


@pytest.mark.parametrize("num_samples,expected", [(None, 5), (30, 2), (1, 1)])
def test_bulk_preview_uses_real_graph_quotas_and_cli_override(tmp_path, num_samples, expected):
    config, graph = preview_config(tmp_path)
    trace = tmp_path / "prompts.jsonl"
    args = [
        "generate",
        str(config),
        "--topics-load",
        str(graph),
        "--dry-run",
        "--trace-prompt",
        "--trace-prompt-file",
        str(trace),
        "--verbose",
    ]
    if num_samples:
        args += ["--num-samples", str(num_samples)]
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys", side_effect=AssertionError("API validation")),
        patch(
            "deepfabric.llm.client.make_outlines_model", side_effect=AssertionError("provider init")
        ),
        patch(
            "deepfabric.llm.client.make_async_outlines_model",
            side_effect=AssertionError("async provider init"),
        ),
    ):
        result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    prompts = [json.loads(line)["prompt"] for line in trace.read_text().splitlines()]
    phrase = f"Generate {expected} " + ("diverse values" if expected != 1 else "value")
    assert phrase in prompts[0]
    assert "GENERATION_PERSONA" in prompts[0]
    assert "<accepted_siblings>" not in prompts[0]
    assert 'Generate 1 value for the seed field "command".' in prompts[1]
    assert ("<accepted_siblings>" in prompts[1]) is (expected > 1)
    if expected > 1:
        accepted = json.loads(
            prompts[1].split("<accepted_siblings>\n", 1)[1].split("\n</accepted_siblings>", 1)[0]
        )
        assert len(accepted) == expected - 1
    assert "{{{topic path}}}" in prompts[0]
    assert all("<tools>" not in prompt for prompt in prompts)
    assert not (tmp_path / "output.jsonl").exists()
