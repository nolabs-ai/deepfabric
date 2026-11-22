"""Component tests for generation displays."""

from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset

from deepfabric.dataset_manager import handle_dataset_events_async
from tests.support.generation_displays import display


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
