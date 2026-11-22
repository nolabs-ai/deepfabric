"""Unit tests for generator."""

from unittest.mock import MagicMock, patch

import pytest  # type: ignore

from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator
from tests.support import generator as _support

data_engine = _support.data_engine
engine_args = _support.engine_args


def test_engine_initialization(engine_args):
    with patch("deepfabric.generator.LLMClient"):
        engine = DataSetGenerator(**engine_args)
    assert engine.config.instructions == engine_args["instructions"]
    assert engine.config.generation_system_prompt == engine_args["generation_system_prompt"]
    assert engine.config.model_name == engine_args["model_name"]
    assert isinstance(engine._samples, list)
    assert engine.failed_samples == []


def test_create_data_no_steps(data_engine):
    with pytest.raises(DataSetGeneratorError, match="positive"):
        data_engine.create_data(num_steps=0)


def test_build_prompt(data_engine):
    prompt = data_engine.build_prompt("Test prompt", 3, ["subtopic1", "subtopic2"])
    assert "{{generation_system_prompt}}" not in prompt
    assert "{{instructions}}" not in prompt
    assert "{{examples}}" not in prompt
    assert "{{subtopics}}" not in prompt


def test_build_system_prompt(data_engine):
    system_prompt = data_engine.build_system_prompt()
    assert system_prompt == data_engine.config.dataset_system_prompt


def test_build_custom_instructions_text(data_engine):
    instructions_text = data_engine.build_custom_instructions_text()
    assert "<instructions>" in instructions_text
    assert data_engine.config.instructions in instructions_text


def test_build_examples_text_no_examples(data_engine):
    examples_text = data_engine.build_examples_text(3)
    assert examples_text == ""


def test_build_subtopics_text(data_engine):
    subtopics_text = data_engine.build_subtopics_text(["subtopic1", "subtopic2"])
    assert "subtopic1 -> subtopic2" in subtopics_text


def test_save_dataset(data_engine):
    # Add some samples to save
    data_engine._samples = [{"messages": [{"role": "user", "content": "test"}]}]

    mock_open = MagicMock()
    mock_file = MagicMock()
    mock_open.return_value.__enter__ = MagicMock(return_value=mock_file)
    mock_open.return_value.__exit__ = MagicMock(return_value=False)

    with patch("builtins.open", mock_open):
        data_engine.save_dataset("test_path.jsonl")

    mock_open.assert_called_once_with("test_path.jsonl", "w", encoding="utf-8")
    mock_file.write.assert_called()
