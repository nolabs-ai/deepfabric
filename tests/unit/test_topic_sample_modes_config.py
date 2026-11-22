"""Topic sample mode contracts shared by YAML and direct Python callers."""

import warnings

import pytest
import yaml

from deepfabric.builders import determine_builder_type
from deepfabric.config import DeepFabricConfig, GenerationConfig
from deepfabric.generator import DataSetGeneratorConfig
from deepfabric.topic_sampling import TopicSamplingConfig

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "price": {"type": "number"}},
    "required": ["name", "price"],
}


@pytest.fixture(params=["yaml", "python"])
def make_config(request):
    def make(**options):
        if request.param == "yaml":
            return GenerationConfig(system_prompt="Persona", **options)
        return DataSetGeneratorConfig(
            generation_system_prompt="Persona", provider="openai", model_name="test", **options
        )

    return make


def test_defaults_preserve_independent_custom_schema(make_config):
    config = make_config(output_schema=SCHEMA)
    assert config.topic_sample_mode == "independent"
    assert config.shared_scope == "seed"
    assert config.seed_field is None
    assert config.completion_includes_topic_path is False


@pytest.mark.parametrize("mode", ["sequential", "bulk"])
def test_coordinated_seed_schema(make_config, mode):
    config = make_config(topic_sample_mode=mode, output_schema=SCHEMA, seed_field="name")
    assert config.shared_scope == "seed"


@pytest.mark.parametrize(
    "options,error",
    [
        ({"seed_field": "name"}, "requires generation.output_schema"),
        ({"output_schema": SCHEMA, "seed_field": "missing"}, "top-level"),
        (
            {"output_schema": {"properties": {"name": {}}, "required": []}, "seed_field": "name"},
            "must be required",
        ),
        ({"topic_sample_mode": "bulk", "output_schema": SCHEMA}, "requires seed_field"),
        ({"topic_sample_mode": "sequential", "output_schema": SCHEMA}, "requires seed_field"),
        ({"shared_scope": "seed"}, "explicitly set"),
        ({"shared_scope": "full"}, "explicitly set"),
        ({"completion_includes_topic_path": True}, "requires bulk"),
        ({"completion_includes_topic_path": False}, "requires bulk"),
        (
            {"topic_sample_mode": "sequential", "completion_includes_topic_path": True},
            "requires bulk",
        ),
        ({"topic_sample_mode": "unknown"}, "Input should be"),
        ({"topic_sample_mode": "bulk", "shared_scope": "unknown"}, "Input should be"),
    ],
)
def test_invalid_combinations(make_config, options, error):
    with pytest.raises(ValueError, match=error):
        make_config(**options)


@pytest.mark.parametrize("mode", ["bulk", "sequential"])
@pytest.mark.parametrize("schema", [None, SCHEMA])
def test_full_scope_needs_no_seed_field(make_config, mode, schema):
    config = make_config(topic_sample_mode=mode, shared_scope="full", output_schema=schema)
    assert config.shared_scope == "full" and config.seed_field is None


@pytest.mark.parametrize(
    "tools,python_tools",
    [
        (
            {"components": {"builtin": ["read_file"]}},
            {"tool_components": {"builtin": ["read_file"]}},
        ),
        (
            {"custom": [{"name": "lookup", "description": "Look up a value", "parameters": []}]},
            {
                "custom_tools": [
                    {"name": "lookup", "description": "Look up a value", "parameters": []}
                ]
            },
        ),
    ],
)
def test_tools_full_collection_has_actionable_error(tools, python_tools):
    with pytest.raises(ValueError, match="agent builder cannot produce a collection"):
        GenerationConfig(
            system_prompt="Persona", tools=tools, topic_sample_mode="bulk", shared_scope="full"
        )
    with pytest.raises(ValueError, match="agent builder cannot produce a collection"):
        DataSetGeneratorConfig(
            generation_system_prompt="Persona",
            provider="openai",
            model_name="test",
            **python_tools,
            topic_sample_mode="bulk",
            shared_scope="full",
        )


@pytest.mark.parametrize(
    "tools",
    [
        None,
        {},
        {"components": {}, "custom": []},
        {"spin_endpoint": "http://localhost:3000"},
        {"tools_endpoint": "http://localhost:3000/tools"},
    ],
)
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"topic_sample_mode": "sequential", "shared_scope": "seed"},
        {"topic_sample_mode": "sequential", "shared_scope": "full"},
        {"topic_sample_mode": "bulk", "shared_scope": "seed"},
        {"topic_sample_mode": "bulk", "shared_scope": "full"},
    ],
)
def test_empty_tools_yaml_and_python_select_non_agent(tmp_path, tools, options):
    path = tmp_path / "empty-tools.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "topics": {"prompt": "Topics"},
                "generation": {"system_prompt": "Persona", "tools": tools, **options},
                "output": {"save_as": "samples.jsonl"},
            }
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = DeepFabricConfig.from_yaml(str(path))
        direct = DataSetGeneratorConfig(**loaded.get_generation_params())
    assert direct.tool_components == {}
    assert direct.custom_tools == []
    assert determine_builder_type(direct).name == "single_shot"
    assert direct.topic_sample_mode == loaded.generation.topic_sample_mode


@pytest.mark.parametrize("include_path", [False, True])
def test_completion_path_allowed_for_bulk_seed(make_config, include_path):
    assert (
        make_config(
            topic_sample_mode="bulk", completion_includes_topic_path=include_path
        ).completion_includes_topic_path
        is include_path
    )


def yaml_config(**generation):
    return DeepFabricConfig(
        topics={"prompt": "Topics"},
        generation={"system_prompt": "Persona", **generation},
        output={"save_as": "test.jsonl"},
    )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"topic_sample_mode": "bulk"},
        {"topic_sample_mode": "bulk", "shared_scope": "full"},
        {"topic_sample_mode": "sequential", "shared_scope": "full"},
    ],
)
def test_parameter_forwarding_preserves_explicit_settings(options):
    config = yaml_config(**options)
    params = config.get_generation_params()
    direct = DataSetGeneratorConfig(**params)
    assert direct.topic_sample_mode == config.generation.topic_sample_mode
    assert direct.shared_scope == config.generation.shared_scope
    assert ("shared_scope" in direct.model_fields_set) == ("shared_scope" in options)
    assert "completion_includes_topic_path" not in direct.model_fields_set


def test_all_fields_forwarded():
    config = yaml_config(
        topic_sample_mode="bulk",
        shared_scope="seed",
        output_schema=SCHEMA,
        seed_field="name",
        completion_includes_topic_path=True,
    )
    direct = DataSetGeneratorConfig(**config.get_generation_params())
    assert direct.seed_field == "name"
    assert direct.completion_includes_topic_path is True


@pytest.mark.parametrize("output_prompt", [None, "Same"])
def test_resolved_duplicate_warning(output_prompt):
    with pytest.warns(UserWarning, match="same text twice"):
        DeepFabricConfig(
            topics={"prompt": "Topics"},
            generation={"system_prompt": "Same", "instructions": "Same"},
            output={"save_as": "test.jsonl", "system_prompt": output_prompt},
        )
    with pytest.warns(UserWarning, match="same text twice"):
        DataSetGeneratorConfig(
            generation_system_prompt="Same",
            instructions="Same",
            dataset_system_prompt=output_prompt,
            provider="openai",
            model_name="test",
        )


def test_override_and_matching_personas_do_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DeepFabricConfig(
            topics={"prompt": "Topics"},
            generation={"system_prompt": "Persona", "instructions": "Other"},
            output={"save_as": "test.jsonl", "system_prompt": "Persona"},
        )
        DeepFabricConfig(
            topics={"prompt": "Topics"},
            generation={"system_prompt": "Same", "instructions": "Same"},
            output={"save_as": "test.jsonl", "system_prompt": "Different"},
        )
    assert not caught


def test_yaml_load_forwards_mode_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "topics:\n  prompt: Topics\n"
        "generation:\n  system_prompt: Persona\n  topic_sample_mode: bulk\n"
        "  shared_scope: seed\n  completion_includes_topic_path: true\n"
        "output:\n  save_as: samples.jsonl\n"
    )
    loaded = DeepFabricConfig.from_yaml(str(path))
    direct = DataSetGeneratorConfig(**loaded.get_generation_params())
    assert direct.topic_sample_mode == "bulk"
    assert direct.shared_scope == "seed"
    assert direct.completion_includes_topic_path is True


def test_duplicate_warning_uses_output_override():
    with pytest.warns(UserWarning, match="same text twice"):
        DeepFabricConfig(
            topics={"prompt": "Topics"},
            generation={"system_prompt": "Persona", "instructions": "Guidance"},
            output={"save_as": "test.jsonl", "system_prompt": "Guidance"},
        )


@pytest.mark.parametrize("override", [None, 1, 9000])
def test_bulk_token_override_is_forwarded(make_config, override):
    config = make_config(topic_sample_mode="bulk", bulk={"max_tokens_per_call": override})
    assert config.bulk.max_tokens_per_call == override
    loaded = yaml_config(topic_sample_mode="bulk", bulk={"max_tokens_per_call": override})
    direct = DataSetGeneratorConfig(**loaded.get_generation_params())
    assert direct.bulk.max_tokens_per_call == override


@pytest.mark.parametrize("override", [0, -1])
def test_bulk_token_override_must_be_positive(make_config, override):
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        make_config(topic_sample_mode="bulk", bulk={"max_tokens_per_call": override})


@pytest.mark.parametrize("mode", ["independent", "sequential"])
def test_bulk_token_override_rejects_nonbulk_modes(make_config, mode):
    with pytest.raises(ValueError, match="requires topic_sample_mode: bulk"):
        make_config(topic_sample_mode=mode, bulk={"max_tokens_per_call": 1000})


def test_yaml_bulk_token_override(tmp_path):
    path = tmp_path / "bulk.yaml"
    path.write_text(
        "topics:\n  prompt: Topics\n"
        "generation:\n  system_prompt: Persona\n  topic_sample_mode: bulk\n"
        "  max_tokens: 400\n  bulk:\n    max_tokens_per_call: 1200\n"
        "output:\n  save_as: samples.jsonl\n"
    )
    config = DeepFabricConfig.from_yaml(str(path))
    direct = DataSetGeneratorConfig(**config.get_generation_params())
    assert direct.max_tokens == 400  # noqa: PLR2004
    assert direct.bulk.max_tokens_per_call == 1200  # noqa: PLR2004


AGENT = ["agent_step", "agent_response"]
EVERY_STEP = ("full_sample", "sequential_later", "seed", "completion", *AGENT)
# A placeholder every stage accepts, so only usage (not validation) differs.
ANY_STAGE = "{{ dataset_system_prompt }}"


@pytest.mark.parametrize(
    ("options", "unused"),
    [
        ({}, ["sequential_later", "seed", "completion", *AGENT]),
        ({"topic_sample_mode": "sequential"}, ["seed", "completion", *AGENT]),
        ({"topic_sample_mode": "bulk"}, ["full_sample", "sequential_later", *AGENT]),
        (
            {"topic_sample_mode": "bulk", "shared_scope": "full"},
            ["sequential_later", "seed", "completion", *AGENT],
        ),
    ],
)
def test_unused_prompt_templates_warn(make_config, options, unused):
    with pytest.warns(UserWarning, match="not used by this configuration") as caught:
        make_config(prompt_templates=dict.fromkeys(EVERY_STEP, ANY_STAGE), **options)
    listed = str(caught[0].message).split("ignored: ")[1].split(" (")[0].split(", ")
    assert listed == [f"prompt_templates.{step}" for step in unused]


def test_used_prompt_templates_do_not_warn(make_config):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_config(
            topic_sample_mode="sequential",
            prompt_templates={"full_sample": ANY_STAGE, "sequential_later": ANY_STAGE},
        )
        make_config(prompt_template=ANY_STAGE)


def test_prompt_template_shorthand_is_unused_in_bulk_seed(make_config):
    with pytest.warns(UserWarning, match=r"ignored: prompt_template \("):
        make_config(topic_sample_mode="bulk", prompt_template=ANY_STAGE)


def test_tool_runs_use_agent_stages_not_completion():
    config = TopicSamplingConfig(
        topic_sample_mode="bulk",
        prompt_templates=dict.fromkeys(("seed", "completion", "agent_step"), ANY_STAGE),
    )
    assert config._unused_prompt_templates(has_tools=True) == ["prompt_templates.completion"]
    assert config._unused_prompt_templates(has_tools=False) == ["prompt_templates.agent_step"]


@pytest.mark.parametrize("has_tools", [False, True])
def test_custom_schema_completes_seeds_and_bypasses_the_agent(has_tools):
    """With output_schema, bulk/seed renders completion and no mode runs the tool agent."""
    config = TopicSamplingConfig(
        topic_sample_mode="bulk",
        seed_field="name",
        prompt_templates=dict.fromkeys(("seed", "completion", *AGENT), ANY_STAGE),
    )
    assert config._unused_prompt_templates(has_tools=has_tools, output_schema=SCHEMA) == [
        f"prompt_templates.{step}" for step in AGENT
    ]


def test_tools_block_without_tools_counts_as_no_tools():
    """Matches the generator: spin_endpoint alone configures no tools."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        GenerationConfig(
            system_prompt="Persona",
            topic_sample_mode="bulk",
            tools={"spin_endpoint": "http://localhost:3000"},
            prompt_templates={"seed": ANY_STAGE, "completion": ANY_STAGE},
        )


def test_seed_only_schema_never_renders_completion():
    """The seed is the whole record, so seed completion is skipped."""
    config = TopicSamplingConfig(
        topic_sample_mode="bulk",
        seed_field="name",
        prompt_templates={"seed": ANY_STAGE, "completion": ANY_STAGE},
    )
    seed_only = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    assert config._unused_prompt_templates(has_tools=False, output_schema=seed_only) == [
        "prompt_templates.completion"
    ]
    assert config._unused_prompt_templates(has_tools=False, output_schema=SCHEMA) == []


@pytest.mark.parametrize(
    "tools",
    [
        None,
        {},
        {"components": {}, "custom": []},
        {"spin_endpoint": "http://localhost:3000"},
        {"tools_endpoint": "http://x"},
        {"components": {"builtin": ["read_file"]}},
        {"custom": [{"name": "lookup", "description": "Look up a value", "parameters": []}]},
    ],
)
def test_both_config_layers_report_the_same_tools_value(tools):
    """Identical messages let Python show the CLI warning once."""
    options = {
        "topic_sample_mode": "bulk",
        "prompt_templates": {"full_sample": ANY_STAGE},
    }
    with pytest.warns(UserWarning) as yaml_layer:
        config = GenerationConfig(system_prompt="Persona", tools=tools, **options)
    with pytest.warns(UserWarning) as generator_layer:
        DataSetGeneratorConfig(
            generation_system_prompt="Persona",
            provider="openai",
            model_name="test",
            tool_components=config.tools.components if config.tools else {},
            custom_tools=config.tools.custom if config.tools else [],
            **options,
        )
    assert str(yaml_layer[0].message) == str(generator_layer[0].message)
    has_tools = config.tools is not None and bool(config.tools.components or config.tools.custom)
    assert f"tools={'yes' if has_tools else 'no'}" in str(yaml_layer[0].message)
