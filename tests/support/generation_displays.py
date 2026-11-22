"""Shared fixtures and helpers for generation displays tests."""

from io import StringIO

from rich.console import Console

from deepfabric.tui import DatasetGenerationTUI, DeepFabricTUI


def display():
    stream = StringIO()
    tui = DatasetGenerationTUI(DeepFabricTUI(Console(file=stream, width=140, force_terminal=False)))
    return tui, stream
