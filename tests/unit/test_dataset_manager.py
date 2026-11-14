"""Unit tests for dataset manager."""

import pytest

from deepfabric.dataset_manager import resolve_num_samples
from deepfabric.exceptions import ConfigurationError


@pytest.mark.parametrize("raw_count", ["auto", "50%", "200%"])
def test_dynamic_sample_counts_reject_empty_topics(raw_count):
    with pytest.raises(ConfigurationError, match="Ensure topic generation produced topics"):
        resolve_num_samples(raw_count, 0)


def test_integer_sample_count_does_not_require_topics():
    count = 5
    assert resolve_num_samples(count, 0) == count


@pytest.mark.parametrize("raw_count, expected", [("50%", 2), ("1%", 1)])
def test_percentage_sample_count_rounds_down_with_minimum_one(raw_count, expected):
    assert resolve_num_samples(raw_count, 5) == expected
