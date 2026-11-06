"""Resolved sample temperatures reach the native adapter, including retries."""
# ruff: noqa: PLR2004

import json

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from deepfabric.config import DeepFabricConfig
from deepfabric.constants import TOPIC_TREE_DEFAULT_TEMPERATURE
from deepfabric.generator import DataSetGenerator
from deepfabric.llm.client import GeminiModel, LLMClient
from deepfabric.run_report import RunReport, reporting
from deepfabric.topic_manager import load_or_build_topic_model

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


def response(text):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))],
    )


@pytest.mark.parametrize(
    "shared,section,expected",
    [
        (0.91, 0.23, 0.23),
        (0.91, 0.0, 0.0),
        (0.23, None, 0.23),
        (0.0, None, 0.0),
        (None, None, TOPIC_TREE_DEFAULT_TEMPERATURE),
    ],
)
@pytest.mark.parametrize("retry", [False, True])
async def test_yaml_temperature_reaches_generation_after_graph_load(
    tmp_path, shared, section, expected, retry
):
    shared_llm = {"provider": "gemini", "model": "test"}
    if shared is not None:
        shared_llm["temperature"] = shared
    generation = {
        "system_prompt": "Generate a record",
        "output_schema": SCHEMA,
        "topic_sample_mode": "independent",
        "sample_retries": 1,
        "max_tokens": 400,
    }
    if section is not None:
        generation["llm"] = {"temperature": section}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "llm": shared_llm,
                "topics": {"prompt": "Root", "mode": "graph", "llm": {"temperature": 1.7}},
                "generation": generation,
                "output": {"save_as": str(tmp_path / "samples.jsonl"), "num_samples": 1},
            }
        )
    )
    config = DeepFabricConfig.from_yaml(str(config_path))
    graph_path = tmp_path / "topics.json"
    graph_path.write_text(
        json.dumps(
            {
                "root_id": 0,
                "nodes": {
                    "0": {"id": 0, "topic": "Root", "children": [1]},
                    "1": {"id": 1, "topic": "Child", "parents": [0]},
                },
                "metadata": {
                    "provider": "gemini",
                    "model": "graph-model",
                    "temperature": 1.9,
                    "created_at": "2026-10-30T00:00:00+00:00",
                },
            }
        )
    )
    with patch("deepfabric.graph.LLMClient") as graph_client:
        graph = load_or_build_topic_model(config, topics_load=str(graph_path))
    graph_client.return_value.generate_async.assert_not_called()
    assert graph.temperature == 1.7

    replies = [response('{"name": "accepted"}')]
    if retry:
        replies.insert(0, ValueError("validation error: simulated sample failure"))
    generate = AsyncMock(side_effect=replies)
    native = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    model = GeminiModel(native, "test")
    with (
        patch("deepfabric.llm.client.make_outlines_model", return_value=model),
        patch("deepfabric.llm.client.make_async_outlines_model", return_value=model),
    ):
        client = LLMClient("gemini", "test", rate_limit_config={"max_retries": 0})
        assert client.async_model is model
    with patch("deepfabric.generator.LLMClient", return_value=client):
        engine = DataSetGenerator(**config.get_generation_params())
    report = RunReport()
    with reporting(report):
        dataset = await engine.create_data_async(num_samples=1, batch_size=1, topic_model=graph)
    assert len(dataset) == 1
    assert not engine.get_all_failures()
    assert generate.await_count == (2 if retry else 1)
    for call in generate.await_args_list:
        assert call.kwargs["config"].temperature == expected
        assert call.kwargs["config"].max_output_tokens == 400
        assert "Root -> Child" in call.kwargs["contents"]
    result = report.snapshot(client.token_usage.snapshot(), "completed")
    requests = [event for event in result["events"] if event["event"] == "request"]
    assert len(requests) == generate.await_count
    assert all(event["parameters"]["temperature"] == expected for event in requests)
    assert result["summary"]["retries_by_category"] == ({"sample_validation": 1} if retry else {})
