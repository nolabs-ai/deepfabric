"""Topic sample mode contracts shared by YAML and direct Python callers."""

import warnings

import pytest

from deepfabric.config import DeepFabricConfig, GenerationConfig
from deepfabric.generator import DataSetGeneratorConfig

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
        ({"topic_sample_mode": "bulk", "shared_scope": "full"}, "truncation fix"),
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


def test_tools_full_collection_has_actionable_error():
    with pytest.raises(ValueError, match="agent builder cannot produce a collection"):
        GenerationConfig(
            system_prompt="Persona", tools={}, topic_sample_mode="bulk", shared_scope="full"
        )
    with pytest.raises(ValueError, match="agent builder cannot produce a collection"):
        DataSetGeneratorConfig(
            generation_system_prompt="Persona",
            provider="openai",
            model_name="test",
            tool_components={"builtin": ["read_file"]},
            topic_sample_mode="bulk",
            shared_scope="full",
        )


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
