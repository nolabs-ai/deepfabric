"""Unit tests for loader."""

import tempfile

from pathlib import Path

import pytest

from deepfabric.loader import (
    _detect_source,
    _normalize_data_files,
    _read_text_file,
)
from tests.support.loader import EXPECTED_PARAGRAPH_COUNT

EXPECTED_LINE_COUNT = 5


class TestSourceDetection:
    """Test source detection logic."""

    def test_detect_text_format(self):
        """Test 'text' format detected as local."""
        assert _detect_source("text") == "local"
        assert _detect_source("TEXT") == "local"

    def test_detect_jsonl_format(self):
        """Test 'jsonl' format detected as local."""
        assert _detect_source("jsonl") == "local"
        assert _detect_source("json") == "local"

    def test_detect_cloud_path(self):
        """Test namespace/slug pattern detected as cloud."""
        assert _detect_source("username/dataset-name") == "cloud"
        assert _detect_source("org_name/my-dataset") == "cloud"

    def test_detect_cloud_path_with_underscores(self):
        """Test namespace/slug with underscores."""
        assert _detect_source("my_org/my_dataset") == "cloud"

    def test_detect_local_file_path(self):
        """Test file path detected as local."""
        # Non-existent paths that don't match cloud pattern
        assert _detect_source("/path/to/file.jsonl") == "local"
        assert _detect_source("./data/train.txt") == "local"


class TestNormalizeDataFiles:
    """Test data_files normalization."""

    def test_normalize_string(self):
        """Test normalizing string input."""
        result = _normalize_data_files("train.txt", None)
        assert result == {"train": ["train.txt"]}

    def test_normalize_list(self):
        """Test normalizing list input."""
        result = _normalize_data_files(["a.txt", "b.txt"], None)
        assert result == {"train": ["a.txt", "b.txt"]}

    def test_normalize_dict(self):
        """Test normalizing dict input."""
        result = _normalize_data_files({"train": "train.txt", "test": "test.txt"}, None)
        assert result == {"train": ["train.txt"], "test": ["test.txt"]}

    def test_normalize_dict_with_lists(self):
        """Test normalizing dict with list values."""
        result = _normalize_data_files({"train": ["a.txt", "b.txt"]}, None)
        assert result == {"train": ["a.txt", "b.txt"]}

    def test_normalize_with_data_dir(self):
        """Test normalizing with data_dir prefix."""
        result = _normalize_data_files("train.txt", "/data")
        assert result == {"train": ["/data/train.txt"]}

    def test_normalize_none(self):
        """Test normalizing None returns empty dict."""
        result = _normalize_data_files(None, None)
        assert result == {}


class TestReadTextFile:
    """Test text file reading with different sampling strategies."""

    @pytest.fixture
    def text_file(self):
        """Create a temporary text file."""
        content = """Line one
Line two

Paragraph two line one
Paragraph two line two

Paragraph three"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name
        yield path
        Path(path).unlink()

    def test_read_by_line(self, text_file):
        """Test reading by line."""
        lines = _read_text_file(text_file, "line")
        assert len(lines) == EXPECTED_LINE_COUNT
        assert lines[0] == "Line one"
        assert lines[1] == "Line two"

    def test_read_by_paragraph(self, text_file):
        """Test reading by paragraph."""
        paragraphs = _read_text_file(text_file, "paragraph")
        assert len(paragraphs) == EXPECTED_PARAGRAPH_COUNT
        assert "Line one" in paragraphs[0]
        assert "Paragraph two" in paragraphs[1]

    def test_read_by_document(self, text_file):
        """Test reading entire document."""
        docs = _read_text_file(text_file, "document")
        assert len(docs) == 1
        assert "Line one" in docs[0]
        assert "Paragraph three" in docs[0]
