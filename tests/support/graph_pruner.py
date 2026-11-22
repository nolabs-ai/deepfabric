"""Shared fixtures and helpers for graph pruner tests."""

import json
import tempfile

from pathlib import Path

import pytest


@pytest.fixture
def deep_graph_json_file():
    """Create a 9-node graph with depth 3 for pruning tests.

    Structure:
        Root (0)
        ├── A (1)
        │   ├── A1 (3)
        │   │   ├── A1a (7)
        │   │   └── A1b (8)
        │   └── A2 (4)
        └── B (2)
            ├── B1 (5)
            └── B2 (6)
    """
    content = {
        "nodes": {
            "0": {
                "id": 0,
                "topic": "Root",
                "children": [1, 2],
                "parents": [],
                "metadata": {"uuid": "uuid-root"},
            },
            "1": {
                "id": 1,
                "topic": "A",
                "children": [3, 4],
                "parents": [0],
                "metadata": {"uuid": "uuid-a"},
            },
            "2": {
                "id": 2,
                "topic": "B",
                "children": [5, 6],
                "parents": [0],
                "metadata": {"uuid": "uuid-b"},
            },
            "3": {
                "id": 3,
                "topic": "A1",
                "children": [7, 8],
                "parents": [1],
                "metadata": {"uuid": "uuid-a1"},
            },
            "4": {
                "id": 4,
                "topic": "A2",
                "children": [],
                "parents": [1],
                "metadata": {"uuid": "uuid-a2"},
            },
            "5": {
                "id": 5,
                "topic": "B1",
                "children": [],
                "parents": [2],
                "metadata": {"uuid": "uuid-b1"},
            },
            "6": {
                "id": 6,
                "topic": "B2",
                "children": [],
                "parents": [2],
                "metadata": {"uuid": "uuid-b2"},
            },
            "7": {
                "id": 7,
                "topic": "A1a",
                "children": [],
                "parents": [3],
                "metadata": {"uuid": "uuid-a1a"},
            },
            "8": {
                "id": 8,
                "topic": "A1b",
                "children": [],
                "parents": [3],
                "metadata": {"uuid": "uuid-a1b"},
            },
        },
        "root_id": 0,
        "metadata": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "created_at": "2024-01-01T00:00:00+00:00",
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(content, f)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)
