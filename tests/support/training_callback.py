"""Offline HTTP fixture for training callback and metrics sender tests."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_metrics_http():
    """Keep real sender threads while replacing outbound HTTP requests."""
    with patch(
        "deepfabric.training.metrics_sender.requests.post",
        return_value=MagicMock(ok=True),
    ) as post:
        yield post
