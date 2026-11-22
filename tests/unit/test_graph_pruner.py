"""Unit tests for graph pruner."""

import pytest

from deepfabric.graph import Graph
from deepfabric.graph_pruner import (
    _derive_output_path,
)
from tests.support import graph_pruner as _support

deep_graph_json_file = _support.deep_graph_json_file


class TestFindNodeByUuid:
    def test_find_existing(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        node = graph.find_node_by_uuid("uuid-a1")
        assert node is not None
        assert node.topic == "A1"
        assert node.id == 3  # noqa: PLR2004

    def test_find_root(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        node = graph.find_node_by_uuid("uuid-root")
        assert node is not None
        assert node.topic == "Root"

    def test_find_nonexistent(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        assert graph.find_node_by_uuid("no-such-uuid") is None


class TestRemoveNode:
    def test_remove_leaf(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        graph.remove_node(7)  # A1a
        assert 7 not in graph.nodes  # noqa: PLR2004
        # Parent A1 should no longer list A1a as child
        a1 = graph.nodes[3]
        assert all(c.id != 7 for c in a1.children)  # noqa: PLR2004

    def test_remove_internal_node(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        graph.remove_node(3)  # A1 (has children A1a, A1b)
        assert 3 not in graph.nodes  # noqa: PLR2004
        # Parent A should no longer list A1
        a = graph.nodes[1]
        assert all(c.id != 3 for c in a.children)  # noqa: PLR2004
        # Children A1a, A1b should no longer list A1 as parent
        assert all(p.id != 3 for p in graph.nodes[7].parents)  # noqa: PLR2004
        assert all(p.id != 3 for p in graph.nodes[8].parents)  # noqa: PLR2004

    def test_remove_root_raises(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        with pytest.raises(ValueError, match="Cannot remove the root node"):
            graph.remove_node(0)

    def test_remove_nonexistent_raises(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        with pytest.raises(ValueError, match="not found"):
            graph.remove_node(999)


class TestRemoveSubtree:
    def test_remove_leaf_subtree(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        removed = graph.remove_subtree(7)  # A1a (leaf)
        assert removed == [7]
        assert 7 not in graph.nodes  # noqa: PLR2004
        assert len(graph.nodes) == 8  # noqa: PLR2004

    def test_remove_internal_subtree(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        removed = graph.remove_subtree(1)  # A -> A1 -> A1a, A1b + A2
        assert set(removed) == {1, 3, 4, 7, 8}
        assert len(graph.nodes) == 4  # noqa: PLR2004  # Root, B, B1, B2
        # Root should no longer have A as child
        assert all(c.id != 1 for c in graph.root.children)

    def test_remove_root_raises(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        with pytest.raises(ValueError, match="Cannot remove the root node"):
            graph.remove_subtree(0)

    def test_remove_subtree_parent_becomes_leaf(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        graph.remove_subtree(3)  # A1 and its children
        # A should now have only A2 as child
        a = graph.nodes[1]
        assert len(a.children) == 1
        assert a.children[0].id == 4  # noqa: PLR2004


class TestPruneAtLevel:
    def test_prune_level_0(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        removed = graph.prune_at_level(0)
        assert len(removed) == 8  # noqa: PLR2004  # Everything except root
        assert len(graph.nodes) == 1
        assert graph.root.children == []

    def test_prune_level_1(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        removed = graph.prune_at_level(1)
        # Should keep Root, A, B (3 nodes), remove the rest (6 nodes)
        assert len(removed) == 6  # noqa: PLR2004
        assert len(graph.nodes) == 3  # noqa: PLR2004
        assert set(graph.nodes.keys()) == {0, 1, 2}
        # A and B should now be leaves
        assert graph.nodes[1].children == []
        assert graph.nodes[2].children == []

    def test_prune_level_2(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        removed = graph.prune_at_level(2)
        # Should keep Root, A, B, A1, A2, B1, B2 (7 nodes), remove A1a, A1b (2 nodes)
        assert len(removed) == 2  # noqa: PLR2004
        assert len(graph.nodes) == 7  # noqa: PLR2004
        # A1 should now be a leaf
        assert graph.nodes[3].children == []

    def test_prune_level_beyond_max(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        removed = graph.prune_at_level(10)
        assert len(removed) == 0
        assert len(graph.nodes) == 9  # noqa: PLR2004

    def test_prune_negative_raises(self, deep_graph_json_file):
        graph = Graph.load(deep_graph_json_file)
        with pytest.raises(ValueError, match="non-negative"):
            graph.prune_at_level(-1)


class TestDeriveOutputPath:
    def test_level_suffix(self):
        assert (
            _derive_output_path("/tmp/graph.json", "pruned_level2")  # noqa: S108
            == "/tmp/graph_pruned_level2.json"  # noqa: S108
        )

    def test_uuid_suffix(self):
        assert _derive_output_path("/tmp/graph.json", "pruned") == "/tmp/graph_pruned.json"  # noqa: S108
