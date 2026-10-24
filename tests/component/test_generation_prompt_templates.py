"""User templates select framing without changing response contracts or roles."""
# ruff: noqa: PLR2004, ARG001

import hashlib
import json

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from click.testing import CliRunner

from deepfabric.bulk import completion_schema, make_completion_model, record_validator
from deepfabric.cli import cli
from deepfabric.config import DeepFabricConfig, GenerationConfig
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator
from deepfabric.prompt_templates import (
    GenerationPromptTemplates,
    default_template,
    render_template,
    template_manifest,
)

SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}, "rationale": {"type": "string"}},
    "required": ["command", "rationale"],
    "additionalProperties": False,
}
ALLOCATION = {"uuid": "internal", "path": ["Root", "Parent", "Internal target"], "quota": 5}


def make_engine(**options):
    params = {
        "generation_system_prompt": "GENERATION_PERSONA",
        "dataset_system_prompt": "DATASET_PERSONA",
        "provider": "openai",
        "model_name": "test",
        "output_schema": SCHEMA,
    }
    params.update(options)
    with patch("deepfabric.generator.LLMClient"):
        return DataSetGenerator(**params)


@pytest.mark.parametrize(
    "templates,error",
    [
        ({"unknown_step": "text"}, "Extra inputs"),
        ({"seed": "{{ typo }}"}, "Unknown or unavailable seed"),
        ({"seed": "{{ seed }}"}, "Unknown or unavailable seed"),
        ({"completion": "{{ history }}"}, "Unknown or unavailable completion"),
        ({"sequential_later": "{{ seed }}"}, "Unknown or unavailable sequential_later"),
        ({"full_sample": "{{ topic_path.upper() }}"}, "Invalid prompt placeholder"),
        ({"full_sample": "{{ topic_path"}, "Unclosed prompt placeholder"),
        ({"seed": "  "}, "must not be empty"),
        ({"seed": {"file": "absent-template.txt"}}, "Cannot read UTF-8"),
        ({"seed": {"text": "x", "file": "x"}}, "exactly one"),
        ({"seed": {"text": "x", "unexpected": True}}, "Extra inputs"),
    ],
)
def test_invalid_templates_fail_before_client_initialization(templates, error):
    with (
        patch("deepfabric.generator.LLMClient") as client,
        pytest.raises(DataSetGeneratorError, match=error),
    ):
        DataSetGenerator(
            generation_system_prompt="test",
            provider="openai",
            model_name="test",
            prompt_templates=templates,
        )
    client.assert_not_called()


def test_legacy_scalar_is_active_and_conflicting_override_is_rejected():
    engine = make_engine(prompt_template="{{{{system_prompt}}}}\n{{ topic_path }}")
    assert (
        engine.build_prompt(engine._get_prompt_template(), 0, ALLOCATION["path"])
        == "GENERATION_PERSONA\nRoot -> Parent -> Internal target"
    )
    with pytest.raises(DataSetGeneratorError, match="not both"):
        make_engine(prompt_template="legacy", prompt_templates={"full_sample": "new"})


def test_rendering_is_literal_single_pass_and_preserves_json_and_go_fields():
    text = '{{ instructions }} {"key": 1} {{.ID}}'
    assert (
        render_template(text, {"instructions": "literal {{ seed }}"})
        == 'literal {{ seed }} {"key": 1} {{.ID}}'
    )


def test_agent_completion_override_can_explicitly_include_tools():
    engine = make_engine(
        output_schema=None,
        topic_sample_mode="bulk",
        prompt_templates={"completion": "{{ tools_section }} {{ seed }}"},
    )
    engine.config.custom_tools = [{"name": "lookup"}]
    engine.tool_registry = SimpleNamespace(
        tools=[SimpleNamespace(name="lookup", description="Find facts", parameters=[])]
    )
    prompt = engine._conversation_completion_prompt(ALLOCATION, "Question", 0)
    assert "lookup" in prompt and "<tools>" in prompt and '"Question"' in prompt


def test_files_resolve_relative_to_yaml_and_are_snapshotted(tmp_path, monkeypatch):
    directory = tmp_path / "config"
    directory.mkdir()
    source = directory / "prompt.txt"
    source.write_text("FIRST {{ system_prompt }}")
    path = directory / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "topics": {"prompt": "Topic"},
                "generation": {
                    "system_prompt": "PERSONA",
                    "prompt_templates": {"full_sample": {"file": "prompt.txt"}},
                },
                "output": {"save_as": "samples.jsonl"},
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    config = DeepFabricConfig.from_yaml(str(path))
    source.write_text("SECOND {{ system_prompt }}")
    params = config.get_generation_params()
    with patch("deepfabric.generator.LLMClient"):
        engine = DataSetGenerator(**params)
    assert engine.build_prompt(engine._get_prompt_template(), 0) == "FIRST PERSONA"
    manifest = template_manifest(engine.config.prompt_templates)
    assert manifest["overrides"]["full_sample"]["text"] == "FIRST {{ system_prompt }}"
    assert (
        manifest["overrides"]["full_sample"]["sha256"]
        == hashlib.sha256(b"FIRST {{ system_prompt }}").hexdigest()
    )
    assert manifest["overrides"]["full_sample"]["file"] == str(source)


@pytest.mark.parametrize(
    "mode,scope",
    [("independent", None), ("sequential", "seed"), ("sequential", "full"), ("bulk", "full")],
)
def test_full_sample_override_selects_all_full_sample_modes(mode, scope):
    options = {
        "topic_sample_mode": mode,
        "prompt_templates": {
            "full_sample": "FULL {{ requested }} {{ system_prompt }} {{ history }} {{ topic_path }}"
        },
    }
    if scope:
        options.update(shared_scope=scope)
    if scope == "seed":
        options["seed_field"] = "command"
    engine = make_engine(**options)
    if mode == "independent":
        prompt = engine.build_prompt(engine._get_prompt_template(), 0, ALLOCATION["path"])
    elif mode == "sequential":
        prompt = engine._build_sequential_prompt(ALLOCATION, ["earlier command"], 0)
        assert '["earlier command"]' in prompt
    else:
        prompt = engine._build_bulk_collection_prompt(ALLOCATION, 5, [], 0)
    assert prompt.startswith("FULL 5" if mode == "bulk" else "FULL 1")
    assert "GENERATION_PERSONA" in prompt and "DATASET_PERSONA" not in prompt
    assert "Root -> Parent -> Internal target" in prompt


async def test_seed_and_completion_overrides_preserve_pin_and_schema():
    engine = make_engine(
        topic_sample_mode="bulk",
        seed_field="command",
        prompt_templates={
            "full_sample": "UNUSED FULL",
            "seed": "SEEDS {{ requested }} {{ seed_field }} {{ history }} {{ topic_path }}",
            "completion": "COMPLETE {{ seed }} {{ seed_section }} {{ system_prompt }} PATH={{ topic_path }} FRAMING={{ subtopics }} {{ retry_feedback }}",
        },
    )
    prompt = engine._build_bulk_collection_prompt(ALLOCATION, 2, ["docker ps"], 0)
    assert prompt == 'SEEDS 2 "command" ["docker ps"] Root -> Parent -> Internal target'
    remainder = completion_schema(SCHEMA, "command")
    model = make_completion_model(remainder)
    engine.llm_client.generate_async = AsyncMock(
        return_value=model.model_validate_json('{"rationale": "read-only"}')
    )
    sample, failure = await engine._complete_custom_seed_outcome(
        ALLOCATION,
        0,
        0,
        "docker ps",
        model,
        record_validator(remainder),
        record_validator(SCHEMA),
        0,
    )
    assert failure is None and sample == {"command": "docker ps", "rationale": "read-only"}
    call = engine.llm_client.generate_async.call_args
    assert 'COMPLETE "docker ps"' in call.args[0] and "DATASET_PERSONA" in call.args[0]
    assert "PATH=Root -> Parent -> Internal target FRAMING= " in call.args[0]
    assert "command" not in call.args[1].model_json_schema()["properties"]
    assert "system_prompt" not in call.kwargs  # Templates do not add API system-role delivery.


async def test_output_guidance_is_specific_to_seed_and_completion_steps():
    engine = make_engine(
        topic_sample_mode="bulk",
        seed_field="command",
        prompt_templates={
            "seed": "{{ output_instructions }}",
            "completion": "{{ output_instructions }}",
        },
    )
    assert engine._build_bulk_collection_prompt(ALLOCATION, 5, [], 0) == default_template(
        "output_seed_custom"
    )
    remainder = completion_schema(SCHEMA, "command")
    model = make_completion_model(remainder)
    engine.llm_client.generate_async = AsyncMock(
        return_value=model.model_validate_json('{"rationale":"read-only"}')
    )
    _, failure = await engine._complete_custom_seed_outcome(
        ALLOCATION,
        0,
        0,
        "docker ps",
        model,
        record_validator(remainder),
        record_validator(SCHEMA),
        0,
    )
    assert failure is None
    assert engine.llm_client.generate_async.call_args.args[0] == default_template(
        "output_completion_custom"
    )


@pytest.mark.parametrize("include_system", [False, True])
async def test_conversation_override_preserves_exported_roles_and_system_prompt(include_system):
    engine = make_engine(
        output_schema=None,
        topic_sample_mode="sequential",
        shared_scope="full",
        prompt_templates={
            "full_sample": "CUSTOM {{ system_prompt }} {{ dataset_system_prompt }} {{ topic_path }}"
        },
    )

    async def generate(prompt, schema, **kwargs):
        assert isinstance(prompt, str) and "CUSTOM GENERATION_PERSONA DATASET_PERSONA" in prompt
        assert "system_prompt" not in kwargs
        return schema.model_validate(
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        )

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    prompt = engine._build_sequential_prompt(ALLOCATION, [], 0)
    sample, failure = await engine._generate_sequential_sample_outcome(
        prompt, ALLOCATION, 0, 0, include_system
    )
    assert failure is None
    assert [message["role"] for message in sample["messages"]] == (
        ["system"] if include_system else []
    ) + ["user", "assistant"]
    if include_system:
        assert sample["messages"][0]["content"] == "DATASET_PERSONA"


def test_custom_preview_and_real_composition_match_without_provider_calls(tmp_path):
    source = tmp_path / "frame.txt"
    source.write_text("FRAME {{ requested }} {{ system_prompt }} {{ topic_path }} {{ history }}")
    config = {
        "topics": {"prompt": "Topic", "mode": "graph"},
        "generation": {
            "system_prompt": "PERSONA",
            "topic_sample_mode": "bulk",
            "shared_scope": "full",
            "output_schema": SCHEMA,
            "prompt_templates": {"full_sample": {"file": "frame.txt"}},
        },
        "output": {"save_as": "samples.jsonl", "num_samples": 5},
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    trace = tmp_path / "trace.jsonl"
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch(
            "deepfabric.llm.client.make_outlines_model", side_effect=AssertionError("provider call")
        ),
        patch(
            "deepfabric.llm.client.make_async_outlines_model",
            side_effect=AssertionError("provider call"),
        ),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "generate",
                str(path),
                "--dry-run",
                "--trace-prompt",
                "--trace-prompt-file",
                str(trace),
            ],
        )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    record = records[-1]
    assert record["prompt"].startswith("FRAME ") and record["dry_run"]
    params = DeepFabricConfig.from_yaml(str(path)).get_generation_params()
    with patch("deepfabric.generator.LLMClient"):
        engine = DataSetGenerator(**params)
    allocation = {"path": ["{{{topic path}}}"]}
    prompt = engine._build_bulk_collection_prompt(
        allocation, "{{{samples requested for this topic}}}", [], 0
    )
    assert record["prompt"] == prompt


def test_yaml_accepts_legacy_scalar():
    config = GenerationConfig(system_prompt="test", prompt_template="{{ system_prompt }}")
    assert config.prompt_template == "{{ system_prompt }}"
    assert GenerationPromptTemplates().full_sample is None


@pytest.mark.parametrize(
    "step,name",
    [
        ("full_sample", "bulk_full_sample"),
        ("full_sample", "sequential_sample"),
        ("full_sample", "custom_schema"),
        ("full_sample", "conversation_generation"),
        ("seed", "bulk_seed_sample"),
        ("seed", "bulk_conversation_seed"),
        ("completion", "bulk_completion"),
        ("completion", "bulk_conversation_completion"),
    ],
)
def test_packaged_step_templates_can_be_copied_as_overrides(step, name):
    GenerationPromptTemplates(**{step: default_template(name)})


def test_diverse_distinct_ablation_changes_only_the_requested_word(tmp_path):
    baseline = make_engine(topic_sample_mode="bulk", shared_scope="full")
    original = baseline._build_bulk_collection_prompt(ALLOCATION, 5, [], 0)
    template = default_template("bulk_full_sample").replace(
        "{{ request_instruction }}", "Generate {{ requested }} diverse, complete JSON records."
    )
    prompts = {}
    for word in ("diverse", "distinct"):
        path = tmp_path / f"{word}.txt"
        path.write_text(template.replace("diverse,", f"{word},"))
        engine = make_engine(
            topic_sample_mode="bulk",
            shared_scope="full",
            prompt_templates={"full_sample": {"file": str(path)}},
        )
        prompts[word] = engine._build_bulk_collection_prompt(ALLOCATION, 5, [], 0)
    assert prompts["diverse"] == original
    assert prompts["distinct"] == original.replace("diverse,", "distinct,", 1)


def test_report_captures_loaded_template_even_if_file_changes_mid_run(tmp_path):
    source = tmp_path / "frame.txt"
    source.write_text("FIRST {{ system_prompt }}")
    config = tmp_path / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "topics": {"prompt": "Topic"},
                "generation": {
                    "system_prompt": "PERSONA",
                    "prompt_templates": {
                        "full_sample": {"file": "frame.txt"},
                        "sequential_later": {"file": "frame.txt"},
                    },
                },
                "output": {"save_as": str(tmp_path / "samples.jsonl")},
            }
        )
    )
    reports = []
    for index in range(2):
        report = tmp_path / f"report-{index}.json"
        with (
            patch("deepfabric.cli.check_for_updates"),
            patch("deepfabric.cli._validate_api_keys"),
            patch("deepfabric.cli._initialize_topic_model", return_value=object()),
            patch(
                "deepfabric.cli._run_generation",
                side_effect=lambda **_kwargs: source.write_text("SECOND {{ system_prompt }}"),
            ),
        ):
            result = CliRunner().invoke(cli, ["generate", str(config), "--run-report", str(report)])
        assert result.exit_code == 0, result.output
        reports.append(json.loads(report.read_text()))
    assert reports[0]["config"]["sha256"] == reports[1]["config"]["sha256"]
    first, second = [report["prompt_templates"]["overrides"]["full_sample"] for report in reports]
    assert first["text"] == "FIRST {{ system_prompt }}"
    assert second["text"] == "SECOND {{ system_prompt }}" and first["sha256"] != second["sha256"]
    assert reports[0]["prompt_templates"]["builtins"]["bulk_full_sample.txt"]
    assert (
        reports[0]["prompt_templates"]["overrides"]["sequential_later"]["sha256"] == first["sha256"]
    )


@pytest.mark.parametrize("scope", ["seed", "full"])
def test_sequential_later_can_move_history_before_context(scope):
    engine = make_engine(
        topic_sample_mode="sequential",
        shared_scope=scope,
        seed_field="command",
        prompt_templates={
            "full_sample": "FIRST {{ topic_path }}",
            "sequential_later": "{{ history_section }}LATER {{ topic_path }} {{ output_instructions }}",
        },
    )
    first = engine._build_sequential_prompt(ALLOCATION, [], 0, later=False)
    history = (
        ["docker ps"] if scope == "seed" else [{"command": "docker ps", "rationale": "read-only"}]
    )
    later = engine._build_sequential_prompt(ALLOCATION, history, 0, later=True)
    assert first == "FIRST Root -> Parent -> Internal target"
    assert later.index("<prior_successes") < later.index("LATER Root -> Parent -> Internal target")
    assert 'scope="' + scope + '"' in later
    assert "FIRST" not in later and "Return one complete record" in later


@pytest.mark.parametrize("legacy", [False, True])
def test_missing_sequential_later_falls_back_to_full_sample(legacy):
    options = (
        {"prompt_template": "FALLBACK {{ history }}"}
        if legacy
        else {"prompt_templates": {"full_sample": "FALLBACK {{ history }}"}}
    )
    engine = make_engine(topic_sample_mode="sequential", shared_scope="full", **options)
    assert (
        engine._build_sequential_prompt(ALLOCATION, ["earlier"], 0, later=True)
        == 'FALLBACK ["earlier"]'
    )


async def test_later_selection_after_failed_first_slot_uses_ordinal_not_history():
    engine = make_engine(
        topic_sample_mode="sequential",
        shared_scope="full",
        sample_retries=0,
        prompt_templates={
            "full_sample": "FIRST {{ history }}",
            "sequential_later": "LATER {{ history }}",
        },
    )
    prompts = []

    async def generate(prompt, schema, **kwargs):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise ValueError("validation error: failed first slot")
        return schema.model_validate_json('{"command":"docker ps","rationale":"read-only"}')

    engine.llm_client.generate_async = AsyncMock(side_effect=generate)
    topics = SimpleNamespace(
        get_unique_topics=lambda: [SimpleNamespace(uuid="internal", topic="Internal target")],
        get_path_by_id=lambda _uuid: ALLOCATION["path"],
    )
    events = [
        event
        async for event in engine.create_data_with_events_async(num_samples=2, topic_model=topics)
    ]
    assert len(events[-1]) == 1
    assert prompts == ["FIRST []", "LATER []"]


def test_verbose_preview_selects_separate_later_template(tmp_path):
    template = tmp_path / "later.txt"
    template.write_text("LATER {{ history }} THEN {{ topic_path }}")
    config = tmp_path / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "topics": {"prompt": "Topic"},
                "generation": {
                    "system_prompt": "PERSONA",
                    "topic_sample_mode": "sequential",
                    "shared_scope": "full",
                    "output_schema": SCHEMA,
                    "prompt_templates": {
                        "full_sample": "FIRST {{ topic_path }}",
                        "sequential_later": {"file": "later.txt"},
                    },
                },
                "output": {"save_as": str(tmp_path / "samples.jsonl")},
            }
        )
    )
    trace = tmp_path / "preview.jsonl"
    with (
        patch("deepfabric.cli.check_for_updates"),
        patch(
            "deepfabric.llm.client.make_outlines_model", side_effect=AssertionError("provider call")
        ),
        patch(
            "deepfabric.llm.client.make_async_outlines_model",
            side_effect=AssertionError("provider call"),
        ),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "generate",
                str(config),
                "--dry-run",
                "--verbose",
                "--trace-prompt",
                "--trace-prompt-file",
                str(trace),
            ],
        )
    assert result.exit_code == 0, result.output
    prompts = [json.loads(line)["prompt"] for line in trace.read_text().splitlines()]
    samples = [prompt for prompt in prompts if prompt.startswith(("FIRST", "LATER"))]
    assert len(samples) == 2 and samples[0].startswith("FIRST") and samples[1].startswith("LATER")
    assert "previous successful samples" in samples[1]
