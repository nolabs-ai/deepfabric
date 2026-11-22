"""Unit tests for topic sample allocation."""
# ruff: noqa: PLR2004, ARG001

import pytest

from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.sampling import resolve_target, topic_plan


@pytest.mark.parametrize(
    "target,count,quotas",
    [
        (10, 4, [3, 3, 2, 2]),
        (12, 4, [3, 3, 3, 3]),
        (4, 4, [1, 1, 1, 1]),
        (2, 4, [1, 1, 0, 0]),
        (7, 1, [7]),
        (5000, 1875, [3] * 1250 + [2] * 625),
    ],
)
def test_ordered_allocations(target, count, quotas):
    topics = [{"uuid": str(i), "topic": str(i), "path": [str(i)]} for i in range(count)]
    plan = topic_plan(target, topics)
    assert [a["quota"] for a in plan["allocations"]] == quotas
    assert sum(quotas) == target
    assert max(quotas) - min(quotas) <= 1


@pytest.mark.parametrize("batch", [0, -1, True, 1.5])
def test_invalid_batch(batch):
    with pytest.raises(DataSetGeneratorError, match="batch_size"):
        resolve_target(1, 0, batch)
