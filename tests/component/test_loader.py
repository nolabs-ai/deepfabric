"""Component tests for loader."""

import json
import shutil
import tempfile

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from deepfabric.dataset import Dataset, DatasetDict
from deepfabric.exceptions import LoaderError
from deepfabric.loader import (
    load_dataset,
)
from tests.support.loader import EXPECTED_PARAGRAPH_COUNT

EXPECTED_TRAIN_LINES = 2

EXPECTED_SAMPLE_COUNT = 2

HTTP_NOT_FOUND = 404

HTTP_UNAUTHORIZED = 401


class TestLoadTextFiles:
    """Test loading text files."""

    @pytest.fixture
    def text_files(self):
        """Create temporary text files."""
        files = {}
        for name, content in [
            ("train.txt", "train line 1\ntrain line 2"),
            ("test.txt", "test line 1"),
        ]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(content)
                files[name] = f.name
        yield files
        for path in files.values():
            Path(path).unlink()

    def test_load_single_text_file(self, text_files):
        """Test loading a single text file."""
        ds = load_dataset("text", data_files=text_files["train.txt"])
        assert isinstance(ds, Dataset)
        assert len(ds) == EXPECTED_TRAIN_LINES
        assert ds["text"] == ["train line 1", "train line 2"]

    def test_load_text_with_splits(self, text_files):
        """Test loading text files with train/test splits."""
        ds = load_dataset(
            "text",
            data_files={"train": text_files["train.txt"], "test": text_files["test.txt"]},
        )
        assert isinstance(ds, DatasetDict)
        assert len(ds["train"]) == EXPECTED_TRAIN_LINES
        assert len(ds["test"]) == 1

    def test_load_text_sample_by_paragraph(self):
        """Test loading text with paragraph sampling."""
        content = "Para 1\n\nPara 2\n\nPara 3"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name

        try:
            ds = load_dataset("text", data_files=path, sample_by="paragraph")
            assert len(ds) == EXPECTED_PARAGRAPH_COUNT
        finally:
            Path(path).unlink()

    def test_load_text_sample_by_document(self):
        """Test loading text with document sampling."""
        content = "Line 1\nLine 2\nLine 3"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name

        try:
            ds = load_dataset("text", data_files=path, sample_by="document")
            assert isinstance(ds, Dataset)
            assert len(ds) == 1
            assert "Line 1" in ds[0]["text"]
        finally:
            Path(path).unlink()

    def test_load_text_missing_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(LoaderError, match="File not found"):
            load_dataset("text", data_files="/nonexistent/file.txt")

    def test_load_text_no_data_files(self):
        """Test loading text without data_files raises error."""
        with pytest.raises(LoaderError, match="requires data_files or data_dir"):
            load_dataset("text")


class TestLoadJsonlFiles:
    """Test loading JSONL files."""

    @pytest.fixture
    def jsonl_file(self):
        """Create a temporary JSONL file."""
        data = [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "World"}]},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        yield path
        Path(path).unlink()

    def test_load_jsonl_file(self, jsonl_file):
        """Test loading a JSONL file."""
        ds = load_dataset("jsonl", data_files=jsonl_file)
        assert isinstance(ds, Dataset)
        assert len(ds) == EXPECTED_SAMPLE_COUNT
        assert ds[0]["messages"][0]["content"] == "Hello"

    def test_load_jsonl_direct_path(self, jsonl_file):
        """Test loading JSONL by direct path."""
        ds = load_dataset(jsonl_file)
        assert len(ds) == EXPECTED_SAMPLE_COUNT

    def test_load_jsonl_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"valid": true}\n')
            f.write("not valid json\n")
            path = f.name

        try:
            with pytest.raises(LoaderError, match="Invalid JSON"):
                load_dataset("jsonl", data_files=path)
        finally:
            Path(path).unlink()


class TestLoadJsonFiles:
    """Test loading standard JSON files (array of objects)."""

    @pytest.fixture
    def json_array_file(self):
        """Create a temporary JSON file with array of objects."""
        data = [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "World"}]},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        yield path
        Path(path).unlink()

    @pytest.fixture
    def json_single_object_file(self):
        """Create a temporary JSON file with single object."""
        data = {"messages": [{"role": "user", "content": "Single"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        yield path
        Path(path).unlink()

    def test_load_json_array_file(self, json_array_file):
        """Test loading a JSON file with array of objects."""
        ds = load_dataset("json", data_files=json_array_file)
        assert isinstance(ds, Dataset)
        assert len(ds) == EXPECTED_SAMPLE_COUNT
        assert ds[0]["messages"][0]["content"] == "Hello"

    def test_load_json_direct_path(self, json_array_file):
        """Test loading JSON by direct path."""
        ds = load_dataset(json_array_file)
        assert len(ds) == EXPECTED_SAMPLE_COUNT

    def test_load_json_single_object(self, json_single_object_file):
        """Test loading JSON file with single object wraps in list."""
        ds = load_dataset(json_single_object_file)
        assert isinstance(ds, Dataset)
        assert len(ds) == 1
        assert ds[0]["messages"][0]["content"] == "Single"

    def test_load_json_invalid_content(self):
        """Test loading JSON with non-object array items raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["string1", "string2"], f)
            path = f.name

        try:
            with pytest.raises(LoaderError, match="Expected array of objects"):
                load_dataset(path)
        finally:
            Path(path).unlink()

    def test_load_json_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            path = f.name

        try:
            with pytest.raises(LoaderError, match="Invalid JSON"):
                load_dataset(path)
        finally:
            Path(path).unlink()


class TestLoadFromDirectory:
    """Test loading from directory."""

    @pytest.fixture
    def data_dir(self):
        """Create a temporary directory with data files."""
        dir_path = tempfile.mkdtemp()
        # Create text file
        with open(Path(dir_path) / "text.txt", "w") as f:
            f.write("text line 1\ntext line 2")
        # Create jsonl file
        with open(Path(dir_path) / "data.jsonl", "w") as f:
            f.write('{"key": "value"}\n')
        yield dir_path
        # Cleanup
        shutil.rmtree(dir_path)

    @pytest.fixture
    def data_dir_with_json(self):
        """Create a temporary directory with JSON file."""
        dir_path = tempfile.mkdtemp()
        # Create standard JSON file (array)
        with open(Path(dir_path) / "data.json", "w") as f:
            json.dump([{"text": "from json"}], f)
        yield dir_path
        shutil.rmtree(dir_path)

    def test_load_from_directory_text(self, data_dir):
        """Test loading text files from directory."""
        ds = load_dataset("text", data_dir=data_dir)
        # Should load both txt and jsonl files
        assert len(ds) > 0

    def test_load_directory_as_path(self, data_dir):
        """Test loading directory by direct path."""
        ds = load_dataset(data_dir)
        assert len(ds) > 0

    def test_load_directory_with_json_file(self, data_dir_with_json):
        """Test loading directory with standard JSON file (array format)."""
        ds = load_dataset(data_dir_with_json)
        assert isinstance(ds, Dataset)
        assert len(ds) == 1
        assert ds[0]["text"] == "from json"


class TestLoadFromCloud:
    """Test loading from DeepFabric Cloud."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock API response."""
        return {
            "id": "test-id",
            "name": "Test Dataset",
            "samples": [
                {"messages": [{"role": "user", "content": "Hello"}]},
                {"messages": [{"role": "user", "content": "World"}]},
            ],
        }

    def test_load_cloud_dataset(self, mock_response):
        """Test loading from cloud with mocked API."""
        with patch("deepfabric.loader.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = MagicMock()
            mock_instance.get.return_value = mock_resp

            # Disable cache for this test
            ds = load_dataset("testuser/test-dataset", use_cache=False)

            assert isinstance(ds, Dataset)
            assert len(ds) == EXPECTED_SAMPLE_COUNT

    def test_load_cloud_dataset_with_token(self, mock_response):
        """Test loading from cloud with explicit token."""
        with patch("deepfabric.loader.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = MagicMock()
            mock_instance.get.return_value = mock_resp

            load_dataset("testuser/test-dataset", token="my-token", use_cache=False)  # noqa: S106

            # Verify token was passed in headers
            call_args = mock_instance.get.call_args
            assert "Authorization" in call_args.kwargs["headers"]
            assert "Bearer my-token" in call_args.kwargs["headers"]["Authorization"]

    def test_load_cloud_dataset_not_found(self):
        """Test loading non-existent cloud dataset."""
        with patch("deepfabric.loader.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.status_code = HTTP_NOT_FOUND
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Not found", request=MagicMock(), response=mock_resp
            )
            mock_instance.get.return_value = mock_resp

            with pytest.raises(LoaderError, match="Dataset not found"):
                load_dataset("testuser/nonexistent", use_cache=False)

    def test_load_cloud_dataset_auth_required(self):
        """Test loading private dataset without auth."""
        with patch("deepfabric.loader.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.status_code = HTTP_UNAUTHORIZED
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=mock_resp
            )
            mock_instance.get.return_value = mock_resp

            with pytest.raises(LoaderError, match="Authentication required"):
                load_dataset("testuser/private-ds", use_cache=False)


class TestCaching:
    """Test caching functionality."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock API response."""
        return {
            "samples": [{"text": "cached data"}],
        }

    def test_cache_saves_dataset(self, mock_response):
        """Test that dataset is cached after loading."""
        with patch("deepfabric.loader.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = MagicMock()
            mock_instance.get.return_value = mock_resp

            # Use a temp cache dir
            with (
                tempfile.TemporaryDirectory() as cache_dir,
                patch("deepfabric.loader.DEFAULT_CACHE_DIR", Path(cache_dir)),
            ):
                load_dataset("testuser/test-ds", use_cache=True)

                # Check cache file exists
                cache_file = Path(cache_dir) / "testuser_test-ds.jsonl"
                assert cache_file.exists()

    def test_cache_loads_from_disk(self):
        """Test that cached dataset is loaded from disk."""
        with tempfile.TemporaryDirectory() as cache_dir:
            # Pre-populate cache
            cache_file = Path(cache_dir) / "testuser_cached-ds.jsonl"
            cache_file.write_text('{"text": "from cache"}\n')

            with (
                patch("deepfabric.loader.DEFAULT_CACHE_DIR", Path(cache_dir)),
                patch("deepfabric.loader.httpx.Client") as mock_client,
            ):
                ds = load_dataset("testuser/cached-ds", use_cache=True)
                assert isinstance(ds, Dataset)

                # Should not have called API
                mock_client.return_value.__enter__.return_value.get.assert_not_called()
                assert ds[0]["text"] == "from cache"

    def test_cache_disabled(self, mock_response):
        """Test that caching can be disabled."""
        with tempfile.TemporaryDirectory() as cache_dir:
            # Pre-populate cache
            cache_file = Path(cache_dir) / "testuser_test-ds.jsonl"
            cache_file.write_text('{"text": "from cache"}\n')

            with (
                patch("deepfabric.loader.DEFAULT_CACHE_DIR", Path(cache_dir)),
                patch("deepfabric.loader.httpx.Client") as mock_client,
            ):
                mock_instance = MagicMock()
                mock_client.return_value.__enter__.return_value = mock_instance

                mock_resp = MagicMock()
                mock_resp.json.return_value = mock_response
                mock_resp.raise_for_status = MagicMock()
                mock_instance.get.return_value = mock_resp

                load_dataset("testuser/test-ds", use_cache=False)

                # Should have called API despite cache existing
                mock_instance.get.assert_called_once()


class TestSplitParameter:
    """Test split parameter handling."""

    def test_split_selects_from_datasetdict(self):
        """Test split parameter selects from DatasetDict."""
        with tempfile.TemporaryDirectory() as dir_path:
            # Create train and test files
            Path(dir_path, "train.txt").write_text("train data")
            Path(dir_path, "test.txt").write_text("test data")

            ds = load_dataset(
                "text",
                data_files={
                    "train": str(Path(dir_path) / "train.txt"),
                    "test": str(Path(dir_path) / "test.txt"),
                },
                split="train",
            )

            assert isinstance(ds, Dataset)
            assert ds[0]["text"] == "train data"

    def test_split_invalid_raises_error(self):
        """Test invalid split raises error."""
        with tempfile.TemporaryDirectory() as dir_path:
            # Create train and test files to get a DatasetDict
            Path(dir_path, "train.txt").write_text("train data")
            Path(dir_path, "test.txt").write_text("test data")

            # This creates train and test splits, so asking for validation should fail
            with pytest.raises(LoaderError, match="Split .* not found"):
                load_dataset(
                    "text",
                    data_files={
                        "train": str(Path(dir_path) / "train.txt"),
                        "test": str(Path(dir_path) / "test.txt"),
                    },
                    split="validation",
                )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_load_empty_file(self):
        """Test loading empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name

        try:
            ds = load_dataset("text", data_files=path)
            assert len(ds) == 0
        finally:
            Path(path).unlink()

    def test_load_invalid_cloud_path_format(self):
        """Test invalid cloud path format raises error."""
        # This would be detected as cloud but has wrong format
        with (
            patch("deepfabric.loader._detect_source", return_value="cloud"),
            pytest.raises(LoaderError, match="Invalid cloud path"),
        ):
            load_dataset("invalid/path/too/many/slashes", use_cache=False)

    def test_load_nonexistent_path(self):
        """Test loading non-existent path raises error."""
        with pytest.raises(LoaderError, match="Path not found"):
            load_dataset("/this/path/does/not/exist.txt")
