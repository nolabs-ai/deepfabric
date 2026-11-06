"""Shared fixtures and helpers for topic inspector tests."""

import json
import tempfile

from pathlib import Path

import pytest


@pytest.fixture
def tree_jsonl_file():
    """Create a temporary tree JSONL file."""
    content = [
        {"path": ["Root", "Child1", "Grandchild1"]},
        {"path": ["Root", "Child1", "Grandchild2"]},
        {"path": ["Root", "Child2"]},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in content:
            f.write(json.dumps(item) + "\n")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def graph_json_file():
    """Create a temporary graph JSON file."""
    content = {
        "nodes": {
            "0": {
                "id": 0,
                "topic": "Root",
                "children": [1, 2],
                "parents": [],
                "metadata": {"uuid": "test-uuid-0"},
            },
            "1": {
                "id": 1,
                "topic": "Child1",
                "children": [],
                "parents": [0],
                "metadata": {"uuid": "test-uuid-1"},
            },
            "2": {
                "id": 2,
                "topic": "Child2",
                "children": [],
                "parents": [0],
                "metadata": {"uuid": "test-uuid-2"},
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
