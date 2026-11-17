"""Tests for the opt-in Verbalized Sampling strategy (``sampling: verbalized``)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic import ValidationError

from deepfabric.builders import SingleShotBuilder
from deepfabric.builders_agent import SingleTurnAgentBuilder, UserQuestion
from deepfabric.config import DeepFabricConfig
from deepfabric.generator import DataSetGeneratorConfig
from deepfabric.graph import Graph
from deepfabric.prompts import VerbalizedSamplingPromptBuilder
from deepfabric.schemas import TopicList
from deepfabric.tree import Tree

METHOD_HEADER = "METHOD -- Verbalized Sampling"
EXISTING_HEADER = "Topics already generated elsewhere"


def _config_dict(topics_sampling=None, generation_sampling=None):
    config = {
        "topics": {"prompt": "Root", "mode": "graph", "llm": {"provider": "test", "model": "m"}},
        "generation": {"system_prompt": "Gen", "llm": {"provider": "test", "model": "m"}},
        "output": {"save_as": "out.jsonl"},
    }
    if topics_sampling:
        config["topics"]["sampling"] = topics_sampling
    if generation_sampling:
        config["generation"]["sampling"] = generation_sampling
    return config


def _graph(sampling):
    with patch("deepfabric.graph.LLMClient"):
        return Graph(
            topic_prompt="Root",
            provider="openai",
            model_name="test-model",
            prompt_style="isolated",
            sampling=sampling,
        )


def _tree(sampling):
    with patch("deepfabric.tree.LLMClient"):
        return Tree(
            topic_prompt="Root", provider="openai", model_name="test-model", sampling=sampling
        )


def _generator_config(sampling):
    return DataSetGeneratorConfig(
        generation_system_prompt="Gen", provider="openai", model_name="m", sampling=sampling
    )


class TestPromptBuilder:
    def test_subtopic_instructions_scale_candidate_pool(self):
        assert "about 12 candidate subtopics" in (
            VerbalizedSamplingPromptBuilder.subtopic_instructions(4)
        )
        # Small requests still get a pool larger than the request.
        assert "about 6 candidate subtopics" in (
            VerbalizedSamplingPromptBuilder.subtopic_instructions(1)
        )

    def test_existing_topics_keeps_most_recent(self):
        block = VerbalizedSamplingPromptBuilder.existing_topics_block(
            ["old", "", "mid", "new"], max_topics=2
        )
        assert "- mid\n- new" in block
        assert "old" not in block

    def test_existing_topics_empty(self):
        assert VerbalizedSamplingPromptBuilder.existing_topics_block([]) == ""
        assert VerbalizedSamplingPromptBuilder.existing_topics_block(["a"], max_topics=0) == ""

    def test_augment_keeps_original_prompt_first(self):
        prompt = VerbalizedSamplingPromptBuilder.augment_subtopic_prompt("Base prompt\n", 3, ["x"])
        assert prompt.startswith("Base prompt")
        assert prompt.index(EXISTING_HEADER) < prompt.index(METHOD_HEADER)


class TestConfig:
    def test_defaults_to_direct(self):
        config = DeepFabricConfig(**_config_dict())
        assert config.get_topics_params()["sampling"] == "direct"
        assert config.get_generation_params()["sampling"] == "direct"

    def test_passes_verbalized_through(self):
        config = DeepFabricConfig(**_config_dict("verbalized", "verbalized"))
        assert config.get_topics_params()["sampling"] == "verbalized"
        assert config.get_generation_params()["sampling"] == "verbalized"

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValidationError):
            DeepFabricConfig(**_config_dict(topics_sampling="nucleus"))


class TestGraph:
    async def _expansion_prompt(self, sampling):
        graph = _graph(sampling)
        graph.add_node("Existing topic")
        graph._generate_subtopics_with_retry = AsyncMock(return_value=None)
        await graph.get_subtopics_and_connections(graph.root, 3)
        return graph._generate_subtopics_with_retry.await_args.args[0]

    async def test_direct_prompt_unchanged(self):
        prompt = await self._expansion_prompt("direct")
        assert METHOD_HEADER not in prompt
        assert EXISTING_HEADER not in prompt

    async def test_verbalized_prompt_lists_existing_topics(self):
        prompt = await self._expansion_prompt("verbalized")
        assert METHOD_HEADER in prompt
        assert "- Existing topic" in prompt
        assert "- Root" not in prompt


class TestTree:
    async def test_verbalized_prompt_includes_earlier_subtopics(self):
        tree = _tree("verbalized")
        tree.llm_client.generate_async = AsyncMock(
            side_effect=[TopicList(subtopics=["alpha", "beta"]), TopicList(subtopics=["gamma"])]
        )

        assert await tree.get_subtopics("", ["Root"], 2) == ["alpha", "beta"]
        await tree.get_subtopics("", ["Root", "alpha"], 2)

        first, second = (c.kwargs["prompt"] for c in tree.llm_client.generate_async.await_args_list)
        assert METHOD_HEADER in first
        assert EXISTING_HEADER not in first
        assert "- alpha\n- beta" in second
        assert tree.generated_topics == ["alpha", "beta", "gamma"]

    async def test_direct_prompt_unchanged(self):
        tree = _tree("direct")
        tree.llm_client.generate_async = AsyncMock(return_value=TopicList(subtopics=["alpha"]))
        await tree.get_subtopics("", ["Root"], 2)
        assert METHOD_HEADER not in tree.llm_client.generate_async.await_args.kwargs["prompt"]


class TestBuilders:
    @pytest.mark.parametrize(("sampling", "expected"), [("direct", False), ("verbalized", True)])
    def test_single_shot_prompt(self, sampling, expected):
        builder = SingleShotBuilder(MagicMock(), _generator_config(sampling))
        assert (METHOD_HEADER in builder._build_prompt("A topic")) is expected

    @pytest.mark.parametrize(("sampling", "expected"), [("direct", False), ("verbalized", True)])
    async def test_agent_user_question_prompt(self, sampling, expected):
        llm = MagicMock()
        llm.generate_async = AsyncMock(return_value=UserQuestion(content="What is the weather?"))
        builder = SingleTurnAgentBuilder(llm, _generator_config(sampling), MagicMock())

        await builder._generate_user_question("Weather in Paris")

        prompt = llm.generate_async.await_args.kwargs["prompt"]
        assert (METHOD_HEADER in prompt) is expected
        assert "Weather in Paris" in prompt
