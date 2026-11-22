"""Unit tests for TUI components."""

from deepfabric.tui import DatasetGenerationTUI, DeepFabricTUI


class TestDatasetGenerationTUI:
    """Tests for DatasetGenerationTUI."""

    def test_on_llm_retry_does_not_crash(self):
        """Ensure on_llm_retry works without _refresh_left (which only exists on TopicGenerationTUI)."""
        tui = DeepFabricTUI()
        dataset_tui = DatasetGenerationTUI(tui)

        # Should not raise AttributeError
        dataset_tui.on_llm_retry(
            provider="openai",
            attempt=1,
            wait=2.0,
            error_summary="rate limit",
            metadata={},
        )

        assert "↻ openai retry (attempt 1), backoff 2.0s" in dataset_tui.events_log

    def test_resolved_bulk_token_budget_is_visible_in_events(self):
        dataset_tui = DatasetGenerationTUI(DeepFabricTUI())
        label = "Seed token cap 600, calculated as 200 x 3; completions separately capped at 200"
        dataset_tui.on_step_start(label, {"max_tokens_per_call": 600})
        assert label in dataset_tui.events_log
        dataset_tui.on_step_start("Ordinary sample", {})
        assert "Ordinary sample" not in dataset_tui.events_log
