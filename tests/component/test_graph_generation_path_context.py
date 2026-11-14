"""Verify saved graph ancestry reaches prompt construction during generation."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from deepfabric.generator import DataSetGenerator
from deepfabric.graph import Graph
from deepfabric.schemas import ChatMessage, Conversation


@pytest.mark.parametrize("target", [3, 8])
def test_generation_uses_full_paths_from_saved_graph(target):
    graph_path = Path(__file__).resolve().parents[1] / "fixtures" / "graph_path_context.json"
    with patch("deepfabric.graph.LLMClient"), patch("deepfabric.generator.LLMClient"):
        graph = Graph.from_json(
            str(graph_path),
            {
                "topic_prompt": "Root",
                "provider": "openai",
                "model_name": "test-model",
                "degree": 1,
                "depth": 3,
            },
        )
        generator = DataSetGenerator(
            instructions="Generate a question and answer about the topic.",
            generation_system_prompt="Test generation system prompt",
            dataset_system_prompt="Test dataset system prompt",
            provider="openai",
            model_name="test-model",
        )

    def mock_response(*_args, **_kwargs):
        # Generation can mutate responses, so each call needs its own instance.
        return Conversation(
            messages=[
                ChatMessage(role="user", content="Example question"),
                ChatMessage(role="assistant", content="Example answer"),
            ]
        )

    generator.llm_client.generate_async = AsyncMock(side_effect=mock_response)
    expected_paths = [
        ["Root", "Branch"],
        ["Root", "Branch", "Interior"],
        ["Root", "Branch", "Interior", "Leaf"],
    ]

    # Keep graph lookup and prompt construction real; spy on the handoff.
    with patch.object(generator, "build_prompt", wraps=generator.build_prompt) as build_prompt:
        dataset = generator.create_data(
            num_samples=target,
            batch_size=len(expected_paths),
            topic_model=graph,
        )

    received_paths = [call.kwargs["subtopics_list"] for call in build_prompt.call_args_list]
    base, remainder = divmod(target, len(expected_paths))
    assert sorted(received_paths) == sorted(
        path for i, path in enumerate(expected_paths) for _ in range(base + (i < remainder))
    )
    assert len(dataset) == target
    assert generator.llm_client.generate_async.await_count == target
    assert generator.failed_samples == []
