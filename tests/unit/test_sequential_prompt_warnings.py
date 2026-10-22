"""Unit tests for sequential prompt warnings."""
# ruff: noqa: PLR2004, ARG001

from io import StringIO

import pytest

from rich.console import Console
from rich.layout import Layout

from deepfabric.constants import SEQUENTIAL_PROMPT_WARNING_CHARS
from deepfabric.tui import DatasetGenerationTUI, DeepFabricTUI
from tests.support import sequential_prompt_warnings as _support

make_engine = _support.make_engine


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_character_threshold_and_exact_message_use_slot_ordinal(make_engine, delta):
    assert SEQUENTIAL_PROMPT_WARNING_CHARS == 50_000
    engine = make_engine()
    prompt = "x" * (SEQUENTIAL_PROMPT_WARNING_CHARS + delta)
    event = engine._check_sequential_prompt_size(prompt, {"uuid": "test-topic", "quota": 30}, 19)
    if delta < 0:
        assert event is None
        assert engine._large_sequential_prompt_topics == set()
    else:
        assert event["message"] == (
            f"Sequential prompt for topic test-topic (sample 20 of 30) has reached {len(prompt)} characters. "
            "Large prompts increase input cost and may exceed the model's context window. "
            "All history is retained."
        )
        assert event["characters"] == len(prompt)
        assert event["affected_topics"] == 1 and event["warnings_suppressed"] == 0


def test_rich_counter_updates_one_status_panel_and_resets():
    output = StringIO()
    tui = DatasetGenerationTUI(DeepFabricTUI(Console(file=output, width=140)))
    layout = Layout(name="root")
    layout.split(Layout(name="main"))
    layout["main"].split_row(Layout(name="left"), Layout(name="right"))
    layout["main"]["right"].split(Layout(name="status", size=8))
    tui.live_layout = layout
    tui.set_large_prompt_topics(1)
    tui.set_large_prompt_topics(6)
    tui.console.print(tui._status_panel())
    assert "6 topics (5 warnings suppressed)" in output.getvalue()
    assert "1 topics" not in output.getvalue()
    assert layout["main"]["right"]["status"].size == 10
    tui.set_large_prompt_topics(0)
    assert layout["main"]["right"]["status"].size == 8
