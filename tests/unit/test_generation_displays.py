"""Unit tests for generation displays."""

from unittest.mock import patch

import pytest

from deepfabric.validation import show_validation_success
from tests.support.generation_displays import display


def test_topic_panel_uses_quotas_not_cycles():
    tui, stream = display()
    _, panel = tui.build_generation_panels(
        "test",
        1,
        4,
        total_samples=10,
        is_topic_based=True,
        unique_topics=4,
        selected_topics=4,
        base_quota=2,
        remainder=2,
    )
    tui.console.print(panel)
    output = stream.getvalue()
    assert "10" in output and "Concurrency:" in output
    assert "One pass:" in output and "base quota 2" in output
    assert "Cycle" not in output


@pytest.mark.parametrize("mode", ["tree", "graph"])
def test_validation_labels_estimates_and_single_pass(mode):
    tui, stream = display()
    with (
        patch("deepfabric.validation.get_tui", return_value=tui.tui),
        patch("deepfabric.validation.time.sleep"),
    ):
        show_validation_success(mode, 2, 2, 10, 4)
    output = stream.getvalue()
    assert "Estimated one-pass allocation" in output
    assert "cycling" not in output.lower()
    if mode == "graph":
        assert "actual non-root nodes" in output
