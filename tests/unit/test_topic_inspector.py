"""Unit tests for topic inspector."""

import tempfile

from pathlib import Path

import pytest

from deepfabric.topic_inspector import (
    detect_format,
    inspect_topic_file,
)
from tests.support import topic_inspector as _support

graph_json_file = _support.graph_json_file
tree_jsonl_file = _support.tree_jsonl_file


@pytest.fixture
def empty_file():
    """Create an empty file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


class TestDetectFormat:
    """Tests for format detection."""

    def test_detect_tree_format(self, tree_jsonl_file):
        assert detect_format(tree_jsonl_file) == "tree"

    def test_detect_graph_format(self, graph_json_file):
        assert detect_format(graph_json_file) == "graph"

    def test_detect_format_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            detect_format("/nonexistent/file.jsonl")

    def test_detect_format_empty_file(self, empty_file):
        with pytest.raises(ValueError, match="Empty file"):
            detect_format(empty_file)


class TestInspectTopicFile:
    """Tests for topic file inspection."""

    def test_inspect_tree_file(self, tree_jsonl_file):
        result = inspect_topic_file(tree_jsonl_file)

        assert result.format == "tree"
        assert result.total_paths == 3  # noqa: PLR2004
        assert result.max_depth == 3  # noqa: PLR2004
        assert result.paths_at_level is None
        assert result.all_paths is None
        assert result.metadata.get("root_topic") == "Root"

    def test_inspect_tree_file_with_level(self, tree_jsonl_file):
        result = inspect_topic_file(tree_jsonl_file, level=2)

        assert result.paths_at_level is not None
        assert len(result.paths_at_level) == 2  # noqa: PLR2004  # Grandchild1, Grandchild2

    def test_inspect_tree_file_with_level_0(self, tree_jsonl_file):
        result = inspect_topic_file(tree_jsonl_file, level=0)

        # Level 0 shows the root topic
        assert result.paths_at_level is not None
        assert len(result.paths_at_level) == 1
        assert result.paths_at_level[0] == ["Root"]

    def test_inspect_tree_file_with_level_1(self, tree_jsonl_file):
        result = inspect_topic_file(tree_jsonl_file, level=1)

        assert result.paths_at_level is not None
        # Level 1 shows unique topics at depth 1: Child1, Child2
        assert len(result.paths_at_level) == 2  # noqa: PLR2004

    def test_inspect_tree_file_with_expand_all(self, tree_jsonl_file):
        """Test expand_depth=-1 returns all paths from level onwards."""
        result = inspect_topic_file(tree_jsonl_file, level=1, expand_depth=-1)

        assert result.expanded_paths is not None
        # From level 1: Child1 > Grandchild1, Child1 > Grandchild2, Child2
        assert len(result.expanded_paths) == 3  # noqa: PLR2004
        # Paths should start from level 1 (not include Root)
        for path in result.expanded_paths:
            assert path[0] in ("Child1", "Child2")

    def test_inspect_tree_file_with_expand_limited(self, tree_jsonl_file):
        """Test expand_depth=1 limits to 1 sublevel."""
        result = inspect_topic_file(tree_jsonl_file, level=0, expand_depth=1)

        assert result.expanded_paths is not None
        # With expand_depth=1, max path length from level 0 is 2
        for path in result.expanded_paths:
            assert len(path) <= 2  # noqa: PLR2004

    def test_inspect_tree_file_show_all(self, tree_jsonl_file):
        result = inspect_topic_file(tree_jsonl_file, show_all=True)

        assert result.all_paths is not None
        assert len(result.all_paths) == 3  # noqa: PLR2004

    def test_inspect_graph_file(self, graph_json_file):
        result = inspect_topic_file(graph_json_file)

        assert result.format == "graph"
        assert result.total_paths == 2  # noqa: PLR2004  # Two leaf paths
        assert "total_nodes" in result.metadata
        assert result.metadata["total_nodes"] == 3  # noqa: PLR2004
        assert result.metadata["root_topic"] == "Root"
        assert result.metadata["has_cycles"] is False
