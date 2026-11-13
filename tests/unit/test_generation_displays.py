"""Allocation reporting in native event consumers, TUI panels, and validation output."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from datasets import Dataset
from rich.console import Console

from deepfabric.dataset_manager import handle_dataset_events_async
from deepfabric.tui import DatasetGenerationTUI, DeepFabricTUI
from deepfabric.validation import show_validation_success


def display():
    stream = StringIO()
    tui = DatasetGenerationTUI(DeepFabricTUI(Console(file=stream, width=140, force_terminal=False)))
    return tui, stream


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


async def test_headless_progress_counts_new_slots_without_double_counting_history():
    tui, stream = display()
    result = Dataset.from_list([{"question": "saved"}])

    async def events():
        yield {
            "event": "generation_start",
            "generation_mode": "topic_allocation",
            "model_name": "test",
            "total_samples": 6,
            "concurrency": 2,
            "unique_topics": 4,
            "selected_topics": 4,
            "base_quota": 1,
            "remainder": 2,
            "resumed_samples": 3,
            "resumed_failures": 1,
        }
        yield {"event": "batch_complete", "samples_generated": 1, "samples_failed": 0}
        yield {"event": "batch_complete", "samples_generated": 0, "samples_failed": 1}
        yield {
            "event": "generation_complete",
            "total_samples": 4,
            "failed_samples": 2,
            "expected_samples": 6,
            "unaccounted": 0,
        }
        yield result

    with (
        patch("deepfabric.dataset_manager.get_dataset_tui", return_value=tui),
        patch(
            "deepfabric.dataset_manager._get_tui_settings",
            return_value=SimpleNamespace(mode="simple"),
        ),
    ):
        assert await handle_dataset_events_async(events()) is result
    output = stream.getvalue()
    assert "Processed 6/6" in output
    assert "Cycles" not in output
    assert "base quota 1" in output


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
