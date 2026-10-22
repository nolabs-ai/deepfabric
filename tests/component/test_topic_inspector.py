"""Component tests for topic inspector."""

import json

import pytest

from click.testing import CliRunner

from deepfabric.cli import cli
from tests.support import topic_inspector as _support

graph_json_file = _support.graph_json_file
tree_jsonl_file = _support.tree_jsonl_file


class TestTopicInspectCLI:
    """Tests for the CLI command."""

    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    def test_inspect_basic(self, cli_runner, tree_jsonl_file):
        result = cli_runner.invoke(cli, ["topic", "inspect", tree_jsonl_file])
        assert result.exit_code == 0
        assert "Total Paths" in result.output

    def test_inspect_with_level(self, cli_runner, tree_jsonl_file):
        result = cli_runner.invoke(cli, ["topic", "inspect", tree_jsonl_file, "--level", "2"])
        assert result.exit_code == 0
        assert "Level 2" in result.output

    def test_inspect_show_all(self, cli_runner, tree_jsonl_file):
        result = cli_runner.invoke(cli, ["topic", "inspect", tree_jsonl_file, "--all"])
        assert result.exit_code == 0
        assert "Full Tree Structure" in result.output

    def test_inspect_json_format(self, cli_runner, tree_jsonl_file):
        result = cli_runner.invoke(cli, ["topic", "inspect", tree_jsonl_file, "--format", "json"])
        assert result.exit_code == 0
        # Extract JSON from output (may have update notice before it)
        json_start = result.output.find("{")
        json_output = result.output[json_start:]
        output = json.loads(json_output)
        assert output["format"] == "tree"
        assert output["total_paths"] == 3  # noqa: PLR2004

    def test_inspect_table_format(self, cli_runner, tree_jsonl_file):
        result = cli_runner.invoke(
            cli, ["topic", "inspect", tree_jsonl_file, "--all", "--format", "table"]
        )
        assert result.exit_code == 0
        # Table output should have depth column
        assert "Depth" in result.output

    def test_inspect_nonexistent_file(self, cli_runner):
        result = cli_runner.invoke(cli, ["topic", "inspect", "/nonexistent/file.jsonl"])
        # Click will fail with exit code 2 for invalid path
        assert result.exit_code != 0

    def test_inspect_graph_file(self, cli_runner, graph_json_file):
        result = cli_runner.invoke(cli, ["topic", "inspect", graph_json_file])
        assert result.exit_code == 0
        assert "Graph (JSON)" in result.output
        assert "Total Nodes" in result.output

    def test_inspect_graph_json_format(self, cli_runner, graph_json_file):
        result = cli_runner.invoke(cli, ["topic", "inspect", graph_json_file, "--format", "json"])
        assert result.exit_code == 0
        # Extract JSON from output (may have update notice before it)
        json_start = result.output.find("{")
        json_output = result.output[json_start:]
        output = json.loads(json_output)
        assert output["format"] == "graph"
        assert output["metadata"]["total_nodes"] == 3  # noqa: PLR2004

    def test_inspect_level_shows_simple_list(self, cli_runner, tree_jsonl_file):
        """--level without --expand shows simple bullet list."""
        result = cli_runner.invoke(cli, ["topic", "inspect", tree_jsonl_file, "--level", "1"])
        assert result.exit_code == 0
        assert "Topics at Level 1" in result.output
        # Should show bullet points, not tree structure
        assert "•" in result.output

    def test_inspect_level_with_expand_all(self, cli_runner, tree_jsonl_file):
        """--level with --expand shows tree format with all sublevels."""
        result = cli_runner.invoke(
            cli, ["topic", "inspect", tree_jsonl_file, "--level", "1", "--expand"]
        )
        assert result.exit_code == 0
        assert "Subtree from Level 1" in result.output
        assert "all sublevels" in result.output

    def test_inspect_level_with_expand_depth(self, cli_runner, tree_jsonl_file):
        """--level with --expand N shows tree format with N sublevels."""
        result = cli_runner.invoke(
            cli, ["topic", "inspect", tree_jsonl_file, "--level", "1", "--expand", "1"]
        )
        assert result.exit_code == 0
        assert "Subtree from Level 1" in result.output
        assert "1 sublevel(s)" in result.output

    def test_inspect_with_uuid_flag(self, cli_runner, tree_jsonl_file):
        """--uuid flag shows UUIDs for leaf nodes."""
        result = cli_runner.invoke(cli, ["topic", "inspect", tree_jsonl_file, "--all", "--uuid"])
        assert result.exit_code == 0
        assert "UUID:" in result.output

    def test_inspect_level_with_uuid(self, cli_runner, tree_jsonl_file):
        """--level with --uuid shows UUIDs for leaf topics at that level."""
        # The test fixture has Child2 as a leaf at level 1
        result = cli_runner.invoke(
            cli, ["topic", "inspect", tree_jsonl_file, "--level", "1", "--uuid"]
        )
        assert result.exit_code == 0
        assert "UUID:" in result.output

    def test_inspect_graph_with_uuid_at_level(self, cli_runner, graph_json_file):
        """--uuid shows node UUIDs for graph format at any level."""
        result = cli_runner.invoke(
            cli, ["topic", "inspect", graph_json_file, "--level", "1", "--uuid"]
        )
        assert result.exit_code == 0
        # Graph nodes should show their UUIDs at level 1
        assert "UUID:" in result.output
        assert "test-uuid-" in result.output

    def test_inspect_graph_with_uuid_all(self, cli_runner, graph_json_file):
        """--uuid shows node UUIDs for all nodes in graph format."""
        result = cli_runner.invoke(cli, ["topic", "inspect", graph_json_file, "--all", "--uuid"])
        assert result.exit_code == 0
        # Should show UUIDs for all nodes (Root, Child1, Child2)
        assert "test-uuid-0" in result.output
        assert "test-uuid-1" in result.output
        assert "test-uuid-2" in result.output
