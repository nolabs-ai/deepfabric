from unittest.mock import AsyncMock, Mock, patch

import pytest

from deepfabric.dataset_manager import create_dataset_async, resolve_num_samples
from deepfabric.exceptions import ConfigurationError
from deepfabric.graph import Graph
from deepfabric.tree import Tree


@pytest.fixture(params=["tree", "interior", "cross_connected", "cyclic"])
def topic_model(request):
    """Real topic models whose route counts differ from their topic counts."""
    if request.param == "tree":
        with patch("deepfabric.tree.LLMClient"):
            model = Tree(topic_prompt="Root")
        model.tree_paths = [["Root", f"Topic {i}"] for i in range(6)]
        assert len(model.get_all_paths()) == len(model.get_unique_topics())
        return model

    with patch("deepfabric.graph.LLMClient"):
        model = Graph(topic_prompt="Root")
    parents = [model.add_node(f"Parent {i}") for i in range(2)]
    for parent in parents:
        model.add_edge(model.root.id, parent.id)
        for i in range(2):
            child = model.add_node(f"Child {parent.id}-{i}")
            model.add_edge(parent.id, child.id)

    if request.param == "cross_connected":
        for parent, other in [(parents[0], parents[1]), (parents[1], parents[0])]:
            for child in list(parent.children):
                model.add_edge(other.id, child.id)
        assert len(model.get_all_paths()) > len(model.get_unique_topics())
    elif request.param == "cyclic":
        for parent in parents:
            for child in parent.children:
                model.add_edge(child.id, parent.id)
        assert model.get_all_paths() == []
    else:
        assert len(model.get_all_paths()) < len(model.get_unique_topics())
    assert len(model.get_unique_topics()) == len(model.nodes) - 1
    return model


@pytest.mark.parametrize("raw_count, expected", [("auto", 6), ("50%", 3), ("200%", 12), (5, 5)])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("batch_size", [1, 4, 20])
async def test_sample_counts_use_unique_topics(
    topic_model, raw_count, expected, override, batch_size
):
    config = Mock()
    config.get_output_config.return_value = {
        "num_samples": 100 if override else raw_count,
        "batch_size": batch_size,
    }
    config.get_generation_params.return_value = {"model_name": "test-model"}
    engine = Mock()
    dataset = Mock()

    # Resolution must never enumerate routes, including for cyclic graphs.
    with (
        patch.object(topic_model, "get_all_paths", side_effect=AssertionError("enumerated routes")),
        patch("deepfabric.dataset_manager.get_dataset_tui"),
        patch(
            "deepfabric.dataset_manager.handle_dataset_events_async",
            new_callable=AsyncMock,
            return_value=dataset,
        ),
    ):
        result = await create_dataset_async(
            engine, topic_model, config, num_samples=raw_count if override else None
        )

    assert result is dataset
    call = engine.create_data_with_events_async.call_args.kwargs
    assert call["num_samples"] == expected
    assert call["batch_size"] == batch_size
    assert call["topic_model"] is topic_model


@pytest.mark.parametrize("raw_count", ["auto", "50%", "200%"])
def test_dynamic_sample_counts_reject_empty_topics(raw_count):
    with pytest.raises(ConfigurationError, match="Ensure topic generation produced topics"):
        resolve_num_samples(raw_count, 0)


def test_integer_sample_count_does_not_require_topics():
    count = 5
    assert resolve_num_samples(count, 0) == count


@pytest.mark.parametrize("raw_count, expected", [("50%", 2), ("1%", 1)])
def test_percentage_sample_count_rounds_down_with_minimum_one(raw_count, expected):
    assert resolve_num_samples(raw_count, 5) == expected
