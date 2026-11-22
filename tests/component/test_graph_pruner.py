"""Component tests for graph pruner."""

import json

from pathlib import Path

import pytest

from click.testing import CliRunner

from deepfabric.cli import cli
from deepfabric.graph import Graph
from deepfabric.graph_pruner import (
    load_graph_for_pruning,
    prune_graph_at_level,
    prune_graph_by_uuid,
)
from tests.support import graph_pruner as _support

deep_graph_json_file = _support.deep_graph_json_file


@pytest.fixture
def cli_runner():
    return CliRunner()


class TestLoadGraphForPruning:
    def test_load_valid(self, deep_graph_json_file):
        graph = load_graph_for_pruning(deep_graph_json_file)
        assert len(graph.nodes) == 9  # noqa: PLR2004

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_graph_for_pruning("/nonexistent/file.json")

    def test_non_json_file(self, tmp_path):
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"path": ["a", "b"]}')
        with pytest.raises(ValueError, match="JSON graph file"):
            load_graph_for_pruning(str(jsonl_file))


class TestPruneGraphAtLevel:
    def test_prune_creates_output(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        result = prune_graph_at_level(deep_graph_json_file, 1, output)
        assert Path(output).exists()
        assert result.operation == "level"
        assert result.removed_count == 6  # noqa: PLR2004
        assert result.remaining_nodes == 3  # noqa: PLR2004
        assert result.output_path == output

    def test_prune_output_is_valid_graph(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        prune_graph_at_level(deep_graph_json_file, 1, output)
        # Should be loadable
        reloaded = Graph.load(output)
        assert len(reloaded.nodes) == 3  # noqa: PLR2004

    def test_auto_derived_filename(self, deep_graph_json_file):
        result = prune_graph_at_level(deep_graph_json_file, 2)
        assert "pruned_level2" in result.output_path
        # Clean up auto-generated file
        Path(result.output_path).unlink(missing_ok=True)

    def test_preserves_metadata(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        prune_graph_at_level(deep_graph_json_file, 1, output)
        with open(output) as f:
            data = json.load(f)
        assert data["metadata"]["provider"] == "openai"
        assert data["metadata"]["model"] == "gpt-4"


class TestPruneGraphByUuid:
    def test_prune_leaf(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        result = prune_graph_by_uuid(deep_graph_json_file, "uuid-b1", output)
        assert result.removed_count == 1
        assert result.remaining_nodes == 8  # noqa: PLR2004

    def test_prune_subtree(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        result = prune_graph_by_uuid(deep_graph_json_file, "uuid-a", output)
        # A + A1 + A2 + A1a + A1b = 5 removed
        assert result.removed_count == 5  # noqa: PLR2004
        assert result.remaining_nodes == 4  # noqa: PLR2004

    def test_uuid_not_found(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        with pytest.raises(ValueError, match="No node found"):
            prune_graph_by_uuid(deep_graph_json_file, "no-such-uuid", output)

    def test_root_uuid_raises(self, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "pruned.json")
        with pytest.raises(ValueError, match="Cannot remove the root"):
            prune_graph_by_uuid(deep_graph_json_file, "uuid-root", output)


class TestTopicPruneCLI:
    def test_prune_level(self, cli_runner, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "out.json")
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--level", "1", "-o", output]
        )
        assert result.exit_code == 0
        assert "pruned successfully" in result.output
        assert Path(output).exists()

    def test_prune_uuid(self, cli_runner, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "out.json")
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--uuid", "uuid-b", "-o", output]
        )
        assert result.exit_code == 0
        assert "pruned successfully" in result.output

    def test_dry_run_level(self, cli_runner, deep_graph_json_file):
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--level", "1", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "9 unique nodes" in result.output
        assert "Would remove" in result.output
        assert "Would keep" in result.output

    def test_dry_run_uuid(self, cli_runner, deep_graph_json_file):
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--uuid", "uuid-a", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "9 unique nodes" in result.output
        assert "5 nodes" in result.output  # A subtree = 5 nodes

    def test_no_mode_error(self, cli_runner, deep_graph_json_file):
        result = cli_runner.invoke(cli, ["topic", "prune", deep_graph_json_file])
        assert result.exit_code == 1

    def test_both_modes_error(self, cli_runner, deep_graph_json_file):
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--level", "1", "--uuid", "uuid-a"]
        )
        assert result.exit_code == 1

    def test_uuid_not_found_error(self, cli_runner, deep_graph_json_file, tmp_path):
        output = str(tmp_path / "out.json")
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--uuid", "nonexistent", "-o", output]
        )
        assert result.exit_code == 1

    def test_force_overwrites(self, cli_runner, deep_graph_json_file):
        result = cli_runner.invoke(
            cli, ["topic", "prune", deep_graph_json_file, "--level", "1", "--force"]
        )
        assert result.exit_code == 0
        # Original file should now have only 3 nodes
        reloaded = Graph.load(deep_graph_json_file)
        assert len(reloaded.nodes) == 3  # noqa: PLR2004
