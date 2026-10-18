"""Tests for the CLI module."""

import os
import tempfile

from unittest.mock import Mock, patch

import pytest  # type: ignore

from click.testing import CliRunner

from deepfabric.cli import cli


async def _async_iter(items):
    for item in items:
        yield item


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def sample_yaml_content():
    """Sample YAML content for testing (new format)."""
    return """
topics:
  prompt: "Test root prompt"
  mode: tree
  system_prompt: "Test topic system prompt"
  depth: 2
  degree: 3
  save_as: "test_tree.jsonl"
  llm:
    provider: "test"
    model: "model"
    temperature: 0.7

generation:
  system_prompt: "Test generation system prompt"
  instructions: "Test instructions"
  conversation:
    type: basic
  max_retries: 2
  llm:
    provider: "test"
    model: "model"
    temperature: 0.9

output:
  system_prompt: "Test output system prompt"
  include_system_message: true
  num_samples: 5
  batch_size: 1
  save_as: "test_dataset.jsonl"
"""


@pytest.fixture
def sample_yaml_content_no_sys_msg():
    """Sample YAML content without include_system_message setting."""
    return """
topics:
  prompt: "Test root prompt"
  mode: tree
  system_prompt: "Test topic system prompt"
  depth: 2
  degree: 3
  save_as: "test_tree.jsonl"
  llm:
    provider: "test"
    model: "model"
    temperature: 0.7

generation:
  system_prompt: "Test generation system prompt"
  instructions: "Test instructions"
  conversation:
    type: basic
  max_retries: 2
  llm:
    provider: "test"
    model: "model"
    temperature: 0.9

output:
  num_samples: 5
  batch_size: 1
  save_as: "test_dataset.jsonl"
"""


@pytest.fixture
def sample_config_file(sample_yaml_content):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(sample_yaml_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_config_file_no_sys_msg(sample_yaml_content_no_sys_msg):
    """Create a temporary config file without include_system_message setting."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(sample_yaml_content_no_sys_msg)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_cli_help(cli_runner):
    """Test CLI help command."""
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "DeepFabric CLI" in result.output


def test_generate_help(cli_runner):
    """Test generate command help."""
    result = cli_runner.invoke(cli, ["generate", "--help"])
    assert result.exit_code == 0
    assert "Generate training data from a YAML configuration file" in result.output
    assert "--include-system-message" in result.output


@patch("deepfabric.cli.save_dataset")
@patch("deepfabric.topic_manager.create_topic_generator")
@patch("deepfabric.cli.DataSetGenerator")
def test_generate_command_basic(
    mock_data_engine, mock_create_topic_generator, mock_save_dataset, cli_runner, sample_config_file
):
    """Test basic start command execution."""
    # Setup mocks
    from datasets import Dataset as HFDataset  # noqa: PLC0415

    from deepfabric.tree import Tree  # noqa: PLC0415

    mock_tree_instance = Mock(spec=Tree)
    mock_tree_instance.build_async.return_value = _async_iter(
        [{"event": "build_complete", "total_paths": 9, "failed_generations": 0}]
    )  # Make build return proper event
    mock_tree_instance.tree_paths = [
        ["root", "child1"],
        ["root", "child2"],
    ]  # Add tree_paths attribute
    # Add get_all_paths for dataset_manager.resolve_num_samples
    mock_tree_instance.get_all_paths.return_value = [
        ["root", "child1"],
        ["root", "child2"],
        ["root", "child3"],
        ["root", "child4"],
        ["root", "child5"],
    ]
    # Add get_unique_topics for cycle-based generation info
    from deepfabric.topic_model import Topic  # noqa: PLC0415

    mock_tree_instance.get_unique_topics.return_value = [
        Topic(uuid=f"uuid-{i}", topic=f"child{i}") for i in range(1, 6)
    ]
    mock_engine_instance = Mock()
    mock_engine_instance.has_checkpoint.return_value = False  # No checkpoint conflicts in tests
    mock_engine_instance.stop_requested = False  # Not gracefully stopped
    mock_dataset = Mock(spec=HFDataset)  # Make dataset a proper HFDataset mock
    mock_dataset.__len__ = Mock(return_value=5)

    mock_create_topic_generator.return_value = mock_tree_instance
    mock_data_engine.return_value = mock_engine_instance
    mock_engine_instance.create_data_with_events_async.return_value = _async_iter(
        [
            {"event": "generation_complete", "total_samples": 5, "failed_samples": 0},
            mock_dataset,  # Yield the dataset as final result
        ]
    )  # Make create_data_with_events return proper events and dataset

    # Run command
    result = cli_runner.invoke(cli, ["generate", sample_config_file])

    # Verify command executed successfully
    assert result.exit_code == 0

    # Verify mocks were called correctly
    mock_create_topic_generator.assert_called_once()
    mock_tree_instance.build_async.assert_called_once()
    mock_tree_instance.save.assert_called_once()
    mock_data_engine.assert_called_once()
    mock_engine_instance.create_data_with_events_async.assert_called_once()
    mock_save_dataset.assert_called_once()


@patch("deepfabric.cli.save_dataset")
@patch("deepfabric.topic_manager.create_topic_generator")
@patch("deepfabric.cli.DataSetGenerator")
def test_generate_command_with_sys_msg_override(
    mock_data_engine,
    mock_create_topic_generator,
    mock_save_dataset,  # noqa: ARG001
    cli_runner,
    sample_config_file,  # noqa: ARG001
):
    """Test start command with include_system_message override."""
    # Setup mocks
    from datasets import Dataset as HFDataset  # noqa: PLC0415

    from deepfabric.tree import Tree  # noqa: PLC0415

    mock_tree_instance = Mock(spec=Tree)
    mock_tree_instance.build_async.return_value = _async_iter(
        [{"event": "build_complete", "total_paths": 9, "failed_generations": 0}]
    )
    mock_tree_instance.tree_paths = [["root", "child1"], ["root", "child2"]]
    # Add get_all_paths for dataset_manager.resolve_num_samples
    mock_tree_instance.get_all_paths.return_value = [
        ["root", "child1"],
        ["root", "child2"],
        ["root", "child3"],
        ["root", "child4"],
        ["root", "child5"],
    ]
    # Add get_unique_topics for cycle-based generation info
    from deepfabric.topic_model import Topic  # noqa: PLC0415

    mock_tree_instance.get_unique_topics.return_value = [
        Topic(uuid=f"uuid-{i}", topic=f"child{i}") for i in range(1, 6)
    ]
    mock_engine_instance = Mock()
    mock_engine_instance.has_checkpoint.return_value = False  # No checkpoint conflicts in tests
    mock_engine_instance.stop_requested = False  # Not gracefully stopped
    mock_dataset = Mock(spec=HFDataset)
    mock_dataset.__len__ = Mock(return_value=5)

    mock_create_topic_generator.return_value = mock_tree_instance
    mock_data_engine.return_value = mock_engine_instance
    mock_engine_instance.create_data_with_events_async.return_value = _async_iter(
        [{"event": "generation_complete", "total_samples": 5, "failed_samples": 0}, mock_dataset]
    )

    # Run command with include_system_message override (using --no-system-message)
    result = cli_runner.invoke(
        cli,
        [
            "generate",
            sample_config_file,
            "--no-system-message",
        ],
    )

    # Verify command executed successfully
    assert result.exit_code == 0

    # Verify create_data was called with sys_msg=False
    args, kwargs = mock_engine_instance.create_data_with_events_async.call_args
    assert kwargs["sys_msg"] is False


@patch("deepfabric.cli.save_dataset")
@patch("deepfabric.topic_manager.create_topic_generator")
@patch("deepfabric.cli.DataSetGenerator")
def test_generate_command_default_sys_msg(
    mock_data_engine,
    mock_create_topic_generator,
    mock_save_dataset,  # noqa: ARG001
    cli_runner,
    sample_config_file_no_sys_msg,
):
    """Test start command with default include_system_message behavior."""
    # Setup mocks
    from datasets import Dataset as HFDataset  # noqa: PLC0415

    from deepfabric.tree import Tree  # noqa: PLC0415

    mock_tree_instance = Mock(spec=Tree)
    mock_tree_instance.build_async.return_value = _async_iter(
        [{"event": "build_complete", "total_paths": 9, "failed_generations": 0}]
    )
    mock_tree_instance.tree_paths = [["root", "child1"], ["root", "child2"]]
    # Add get_all_paths for dataset_manager.resolve_num_samples
    mock_tree_instance.get_all_paths.return_value = [
        ["root", "child1"],
        ["root", "child2"],
        ["root", "child3"],
        ["root", "child4"],
        ["root", "child5"],
    ]
    # Add get_unique_topics for cycle-based generation info
    from deepfabric.topic_model import Topic  # noqa: PLC0415

    mock_tree_instance.get_unique_topics.return_value = [
        Topic(uuid=f"uuid-{i}", topic=f"child{i}") for i in range(1, 6)
    ]
    mock_engine_instance = Mock()
    mock_engine_instance.has_checkpoint.return_value = False  # No checkpoint conflicts in tests
    mock_engine_instance.stop_requested = False  # Not gracefully stopped
    mock_dataset = Mock(spec=HFDataset)
    mock_dataset.__len__ = Mock(return_value=5)

    mock_create_topic_generator.return_value = mock_tree_instance
    mock_data_engine.return_value = mock_engine_instance
    mock_engine_instance.create_data_with_events_async.return_value = _async_iter(
        [{"event": "generation_complete", "total_samples": 5, "failed_samples": 0}, mock_dataset]
    )

    # Run command without include_system_message override
    result = cli_runner.invoke(cli, ["generate", sample_config_file_no_sys_msg])

    # Verify command executed successfully
    assert result.exit_code == 0

    # Verify create_data was called with default sys_msg (should be None to use config default)
    args, kwargs = mock_engine_instance.create_data_with_events_async.call_args
    assert "sys_msg" not in kwargs or kwargs["sys_msg"] is None


@patch("deepfabric.cli.save_dataset")
@patch("deepfabric.topic_manager.create_topic_generator")
@patch("deepfabric.cli.DataSetGenerator")
def test_generate_command_with_overrides(
    mock_data_engine, mock_create_topic_generator, mock_save_dataset, cli_runner, sample_config_file
):
    """Test start command with parameter overrides."""
    # Setup mocks
    from datasets import Dataset as HFDataset  # noqa: PLC0415

    from deepfabric.tree import Tree  # noqa: PLC0415

    mock_tree_instance = Mock(spec=Tree)
    mock_tree_instance.build_async.return_value = _async_iter(
        [{"event": "build_complete", "total_paths": 9, "failed_generations": 0}]
    )
    mock_tree_instance.tree_paths = [["root", "child1"], ["root", "child2"]]
    # Add get_all_paths for dataset_manager.resolve_num_samples (need enough for 10 samples)
    mock_tree_instance.get_all_paths.return_value = [["root", f"child{i}"] for i in range(10)]
    # Add get_unique_topics for cycle-based generation info
    from deepfabric.topic_model import Topic  # noqa: PLC0415

    mock_tree_instance.get_unique_topics.return_value = [
        Topic(uuid=f"uuid-{i}", topic=f"child{i}") for i in range(10)
    ]
    mock_engine_instance = Mock()
    mock_engine_instance.has_checkpoint.return_value = False  # No checkpoint conflicts in tests
    mock_engine_instance.stop_requested = False  # Not gracefully stopped
    mock_dataset = Mock(spec=HFDataset)
    mock_dataset.__len__ = Mock(return_value=5)

    mock_create_topic_generator.return_value = mock_tree_instance
    mock_data_engine.return_value = mock_engine_instance
    mock_engine_instance.create_data_with_events_async.return_value = _async_iter(
        [{"event": "generation_complete", "total_samples": 5, "failed_samples": 0}, mock_dataset]
    )

    # Run command with overrides (using new CLI flags)
    result = cli_runner.invoke(
        cli,
        [
            "generate",
            sample_config_file,
            "--topics-save-as",
            "override_tree.jsonl",
            "--output-save-as",
            "override_dataset.jsonl",
            "--provider",
            "override",
            "--model",
            "model",
            "--temperature",
            "0.5",
            "--degree",
            "4",
            "--depth",
            "3",
            "--num-samples",
            "10",
            "--batch-size",
            "2",
            "--no-system-message",
        ],
    )

    # Verify command executed successfully
    assert result.exit_code == 0

    # Verify mocks were called
    mock_create_topic_generator.assert_called_once()

    mock_tree_instance.save.assert_called_once_with("override_tree.jsonl")
    mock_save_dataset.assert_called_once()

    args, kwargs = mock_engine_instance.create_data_with_events_async.call_args
    assert kwargs["num_samples"] == 10  # noqa: PLR2004
    assert kwargs["batch_size"] == 2  # noqa: PLR2004
    assert kwargs["model_name"] == "model"  # Updated expectation based on current CLI behavior
    assert kwargs["sys_msg"] is False


@patch("deepfabric.cli.save_dataset")
@patch("deepfabric.topic_manager.read_topic_tree_from_jsonl")
@patch("deepfabric.topic_manager.Tree")
@patch("deepfabric.cli.DataSetGenerator")
def test_generate_command_with_jsonl(
    mock_data_engine,
    mock_topic_tree,
    mock_read_topic_tree_from_jsonl,
    mock_save_dataset,  # noqa: ARG001
    cli_runner,
    sample_config_file,
):
    """Test start command with JSONL file."""
    from datasets import Dataset as HFDataset  # noqa: PLC0415

    mock_tree_instance = Mock()
    mock_tree_instance.build = Mock()  # Add build method
    # Add get_all_paths for dataset_manager.resolve_num_samples
    mock_tree_instance.get_all_paths.return_value = [
        ["root", "child1"],
        ["root", "child2"],
        ["root", "child3"],
        ["root", "child4"],
        ["root", "child5"],
    ]
    # Add get_unique_topics for cycle-based generation info
    from deepfabric.topic_model import Topic  # noqa: PLC0415

    mock_tree_instance.get_unique_topics.return_value = [
        Topic(uuid=f"uuid-{i}", topic=f"child{i}") for i in range(1, 6)
    ]
    mock_topic_tree.return_value = mock_tree_instance
    mock_read_topic_tree_from_jsonl.return_value = [{"path": ["root", "child"]}]

    mock_engine_instance = Mock()
    mock_engine_instance.has_checkpoint.return_value = False  # No checkpoint conflicts in tests
    mock_engine_instance.stop_requested = False  # Not gracefully stopped
    mock_data_engine.return_value = mock_engine_instance
    mock_dataset = Mock(spec=HFDataset)
    mock_dataset.__len__ = Mock(return_value=5)
    mock_engine_instance.create_data_with_events_async.return_value = _async_iter(
        [{"event": "generation_complete", "total_samples": 5, "failed_samples": 0}, mock_dataset]
    )
    # Create a temporary JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"path": ["root", "child"]}\n')
        temp_jsonl_path = f.name

    try:
        # Run command with JSONL file (using new flag name)
        result = cli_runner.invoke(
            cli,
            ["generate", sample_config_file, "--topics-load", temp_jsonl_path],
        )

        # Print output if command fails
        if result.exit_code != 0:
            print(result.output)

        # Verify command executed successfully
        assert result.exit_code == 0

        # Verify JSONL read function was called
        mock_read_topic_tree_from_jsonl.assert_called_once_with(temp_jsonl_path)

        # Verify from_dict_list was called with the correct data
        mock_tree_instance.from_dict_list.assert_called_once_with([{"path": ["root", "child"]}])

        # Verify save was not called since JSONL file was provided
        mock_tree_instance.save.assert_not_called()

    finally:
        # Cleanup the temporary JSONL file
        if os.path.exists(temp_jsonl_path):
            os.unlink(temp_jsonl_path)


def test_generate_command_missing_config(cli_runner):
    """Test generate command with missing config file."""
    result = cli_runner.invoke(cli, ["generate", "nonexistent.yaml"])
    assert result.exit_code != 0
    assert "Error" in result.output


def test_generate_command_invalid_yaml(cli_runner):
    """Test generate command with invalid YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content:")
        temp_path = f.name

    try:
        result = cli_runner.invoke(cli, ["generate", temp_path])
        assert result.exit_code != 0
        assert "Error" in result.output
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@patch("deepfabric.topic_manager.create_topic_generator")
@patch("deepfabric.cli.DataSetGenerator")
def test_generate_command_error_handling(
    _mock_data_engine,
    mock_create_topic_generator,
    cli_runner,
    sample_config_file,  # noqa: ARG001
):
    """Test error handling in start command."""
    # Setup mock to raise an exception
    mock_create_topic_generator.side_effect = Exception("Test error")

    # Run command
    result = cli_runner.invoke(cli, ["generate", sample_config_file])

    # Verify command failed with error
    assert result.exit_code != 0
    assert "error" in result.output.lower()  # Case-insensitive check
    assert "Test error" in result.output


def test_validate_command(cli_runner, sample_config_file):
    """Test validate command."""
    result = cli_runner.invoke(cli, ["validate", sample_config_file])
    assert result.exit_code == 0
    assert "Configuration is valid" in result.output


def test_info_command(cli_runner):
    """Test info command."""
    result = cli_runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "DeepFabric" in result.output
    assert "Available Commands" in result.output


@patch("deepfabric.hf_hub.HFUploader")
def test_upload_command(mock_uploader, cli_runner):
    """Test upload command."""
    # Create a temporary JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"test": "data"}\n')
        temp_path = f.name

    # Setup mock
    mock_uploader_instance = Mock()
    mock_uploader.return_value = mock_uploader_instance
    mock_uploader_instance.push_to_hub.return_value = {
        "status": "success",
        "message": "Dataset uploaded successfully",
    }

    try:
        # Set HF_TOKEN environment variable for test
        os.environ["HF_TOKEN"] = "test_token"  # noqa: S105 # nosec

        result = cli_runner.invoke(cli, ["upload-hf", temp_path, "--repo", "test/repo"])

        assert result.exit_code == 0
        assert "Dataset uploaded successfully" in result.output
        mock_uploader_instance.push_to_hub.assert_called_once()

    finally:
        # Cleanup
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@patch("deepfabric.topic_manager.Graph.from_json")
def test_visualize_command(mock_from_json, cli_runner):
    """Test visualize command."""
    # Create a temporary graph JSON file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"nodes": {}, "edges": [], "degree": 3, "depth": 2}')
        temp_path = f.name

    # Setup mock
    mock_graph_instance = Mock()
    mock_from_json.return_value = mock_graph_instance

    try:
        result = cli_runner.invoke(cli, ["visualize", temp_path, "--output", "test_graph"])

        # Print output for debugging
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")

        assert result.exit_code == 0
        assert "Graph visualization saved to" in result.output
        mock_graph_instance.visualize.assert_called_once_with("test_graph")

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.fixture
def sample_yaml_content_tree_save():
    """Sample YAML with topics.save_as for tree mode."""
    return """
topics:
  prompt: "Test root prompt"
  mode: tree
  system_prompt: "Test topic system prompt"
  depth: 2
  degree: 3
  save_as: "test_tree.jsonl"
  llm:
    provider: "test"
    model: "model"
    temperature: 0.7

generation:
  system_prompt: "Test generation system prompt"
  instructions: "Test instructions"
  conversation:
    type: basic
  max_retries: 2
  llm:
    provider: "test"
    model: "model"
    temperature: 0.9

output:
  system_prompt: "Test output system prompt"
  include_system_message: true
  num_samples: 5
  batch_size: 1
  save_as: "test_dataset.jsonl"
"""


@pytest.fixture
def sample_config_file_tree_save(sample_yaml_content_tree_save):
    """Create a temporary config file with tree save path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(sample_yaml_content_tree_save)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@patch("deepfabric.topic_manager.create_topic_generator")
def test_topic_only_flag_tree(
    mock_create_topic_generator, cli_runner, sample_config_file_tree_save
):
    """Test --topic-only flag with tree mode."""
    from deepfabric.tree import Tree  # noqa: PLC0415

    # Setup mock tree
    mock_tree_instance = Mock(spec=Tree)
    mock_tree_instance.build_async.return_value = _async_iter(
        [{"event": "build_complete", "total_paths": 9, "failed_generations": 0}]
    )
    mock_tree_instance.tree_paths = [["root", "child1"], ["root", "child2"]]
    mock_create_topic_generator.return_value = mock_tree_instance

    # Run command with --topic-only
    result = cli_runner.invoke(cli, ["generate", sample_config_file_tree_save, "--topic-only"])

    # Verify command executed successfully
    assert result.exit_code == 0

    # Verify tree was built and saved
    mock_create_topic_generator.assert_called_once()
    mock_tree_instance.build_async.assert_called_once()
    mock_tree_instance.save.assert_called_once_with("test_tree.jsonl")

    # Verify success message about topic save
    assert "Topic tree saved to" in result.output


@patch("deepfabric.topic_manager.create_topic_generator")
def test_topic_only_flag_graph(mock_create_topic_generator, cli_runner):
    """Test --topic-only flag with graph mode."""
    from deepfabric.graph import Graph  # noqa: PLC0415

    # Setup mock graph
    mock_graph_instance = Mock(spec=Graph)
    mock_graph_instance.build_async.return_value = _async_iter(
        [{"event": "build_complete", "failed_generations": 0}]
    )
    mock_create_topic_generator.return_value = mock_graph_instance

    # Create sample graph config with new format
    graph_yaml = """
topics:
  prompt: "Test root prompt"
  mode: graph
  prompt_style: anchored
  system_prompt: "Test topic system prompt"
  depth: 2
  degree: 3
  save_as: "test_graph.json"
  llm:
    provider: "test"
    model: "model"
    temperature: 0.7

generation:
  system_prompt: "Test generation system prompt"
  instructions: "Test instructions"
  conversation:
    type: basic
  max_retries: 2
  llm:
    provider: "test"
    model: "model"
    temperature: 0.9

output:
  system_prompt: "Test output system prompt"
  include_system_message: true
  num_samples: 5
  batch_size: 1
  save_as: "test_dataset.jsonl"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(graph_yaml)
        temp_config = f.name

    try:
        # Run command with --topic-only and --mode graph
        result = cli_runner.invoke(
            cli, ["generate", temp_config, "--mode", "graph", "--topic-only"]
        )

        # Verify command executed successfully
        assert result.exit_code == 0

        # Verify graph was built and saved
        mock_create_topic_generator.assert_called_once()
        mock_graph_instance.build_async.assert_called_once()
        mock_graph_instance.save.assert_called_once_with("test_graph.json")

        # Verify success message about topic save
        assert "Topic graph saved to" in result.output

    finally:
        if os.path.exists(temp_config):
            os.unlink(temp_config)


def test_topic_only_with_topics_load_fails(cli_runner, sample_config_file):
    """Test that --topic-only fails when used with --topics-load."""
    # Create a temporary JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"path": ["root", "child"]}\n')
        temp_jsonl_path = f.name

    try:
        result = cli_runner.invoke(
            cli,
            ["generate", sample_config_file, "--topic-only", "--topics-load", temp_jsonl_path],
        )

        # Should fail validation
        assert result.exit_code != 0
        assert "--topic-only cannot be used with --topics-load" in result.output

    finally:
        if os.path.exists(temp_jsonl_path):
            os.unlink(temp_jsonl_path)


@patch("deepfabric.topic_manager.create_topic_generator")
@patch("deepfabric.cli.DataSetGenerator")
def test_generate_resume_v4_error_does_not_start_generation(
    mock_engine_class, mock_topics_factory, cli_runner, sample_config_file
):
    from deepfabric.checkpoint import version_error  # noqa: PLC0415
    from deepfabric.exceptions import DataSetGeneratorError  # noqa: PLC0415
    from deepfabric.tree import Tree  # noqa: PLC0415

    topics = Mock(spec=Tree)
    topics.build_async.return_value = _async_iter(
        [{"event": "build_complete", "total_paths": 1, "failed_generations": 0}]
    )
    topics.tree_paths = [["Root", "A"]]
    mock_topics_factory.return_value = topics
    engine = mock_engine_class.return_value
    engine.load_checkpoint.side_effect = DataSetGeneratorError(
        version_error(4, "old.checkpoint.json")
    )
    result = cli_runner.invoke(
        cli, ["generate", sample_config_file, "--resume", "--checkpoint-interval", "1"]
    )
    assert result.exit_code != 0
    assert "migrate_checkpoint.py" in result.output
    assert "starting fresh" not in result.output
    engine.create_data_with_events_async.assert_not_called()
