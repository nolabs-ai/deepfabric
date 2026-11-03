"""Filesystem publication invariants with isolated destinations, no provider calls."""

import errno
import json
import os

from pathlib import Path
from unittest.mock import Mock

import pytest

from deepfabric.dataset_manager import _save_failed_samples
from deepfabric.output_safety import (
    OutputSaveError,
    clear_output,
    open_unique,
    output_session,
    protect_paths,
    save_output,
)
from deepfabric.run_report import RunReport


def write_text(path):
    Path(path).write_text("new", encoding="utf-8")


def test_numbered_candidates_preserve_existing_outputs(tmp_path):
    target = tmp_path / "dataset.jsonl"
    target.write_text("old")
    target.with_name("dataset_1.jsonl").write_text("older")
    actual = save_output(target, write_text)
    assert actual == str(tmp_path / "dataset_2.jsonl")
    assert target.read_text() == "old"
    assert Path(actual).read_text() == "new"


def test_concurrent_collision_is_not_overwritten(tmp_path, monkeypatch):
    target = tmp_path / "dataset.jsonl"
    path_open = Path.open

    def racing_open(path, mode="r", *args, **kwargs):
        if path == target and mode == "xb":
            target.write_text("racing writer")
        return path_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    actual = save_output(target, write_text)
    assert target.read_text() == "racing writer"
    assert actual == str(tmp_path / "dataset_1.jsonl")


def test_unique_export_does_not_require_hard_links(tmp_path, monkeypatch):
    def unsupported(*_args, **_kwargs):
        raise OSError(errno.EPERM, "hard links unsupported")

    monkeypatch.setattr(os, "link", unsupported)
    target = tmp_path / "dataset.jsonl"
    assert save_output(target, write_text) == str(target)
    assert target.read_text() == "new"


@pytest.mark.parametrize("competing_replacement", [False, True])
def test_copy_failure_cleans_only_owned_partial_and_recovers(
    tmp_path, monkeypatch, competing_replacement
):
    target = tmp_path / "dataset.jsonl"
    app = tmp_path / "app"
    monkeypatch.setattr("deepfabric.output_safety._get_deepfabric_data_dir", lambda: app)
    attempts = 0

    def fail_first_copy(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            destination.write(b"partial")
            destination.flush()
            if competing_replacement:
                target.unlink()
                target.write_text("another process")
            raise OSError(errno.ENOSPC, "full during copy")
        destination.write(source.read())

    monkeypatch.setattr("deepfabric.output_safety.copyfileobj", fail_first_copy)
    with pytest.raises(OutputSaveError) as raised:
        save_output(target, write_text)
    assert raised.value.original_error.errno == errno.ENOSPC
    assert Path(raised.value.recovery).read_text() == "new"
    expected_attempts = 2  # Requested destination and one recovery export.
    assert attempts == expected_attempts
    if competing_replacement:
        assert target.read_text() == "another process"
    else:
        assert not target.exists()


def test_replace_preserves_old_contents_on_write_failure(tmp_path, monkeypatch):
    target = tmp_path / "dataset.jsonl"
    target.write_text("old")
    monkeypatch.setattr(
        "deepfabric.output_safety._get_deepfabric_data_dir", lambda: tmp_path / "app"
    )

    def fail(path):
        Path(path).write_text("partial")
        raise OSError(errno.ENOSPC, "full")

    with pytest.raises(OutputSaveError, match="Recovery also failed"):
        save_output(target, fail, replace=True)
    assert target.read_text() == "old"


def test_failed_publication_recovers_once_but_reports_failure(tmp_path, monkeypatch):
    target = tmp_path / "blocked" / "dataset.jsonl"
    target.parent.write_text("not a directory")
    app = tmp_path / "app"
    monkeypatch.setattr("deepfabric.output_safety._get_deepfabric_data_dir", lambda: app)
    with pytest.raises(OutputSaveError) as raised:
        save_output(target, write_text)
    assert Path(raised.value.recovery).read_text() == "new"
    assert raised.value.path == str(target)


def test_explicit_python_failure_path_is_unique_and_returned(tmp_path):
    path = tmp_path / "failures.jsonl"
    path.write_text("existing")
    actual = _save_failed_samples(str(path), [{"error": "bad"}], Mock(), use_path_directly=True)
    assert actual == str(tmp_path / "failures_1.jsonl")
    assert json.loads(Path(actual).read_text())["error"] == "bad"
    assert path.read_text() == "existing"


def test_report_separates_invocations(tmp_path):
    path = tmp_path / "run.json"
    report = RunReport()
    first = report.write(str(path), {}, "completed")
    second = RunReport().write(str(path), {}, "completed")
    assert first != second
    assert (
        json.loads(Path(first).read_text())["run_id"]
        != json.loads(Path(second).read_text())["run_id"]
    )


def test_exclusive_creation_only_retries_collisions(tmp_path, monkeypatch):
    def denied(*_args, **_kwargs):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(PermissionError):
        open_unique(tmp_path / "trace.jsonl")


def test_replacement_cannot_destroy_a_trace_allocated_during_generation(tmp_path, monkeypatch):
    target = tmp_path / "trace.jsonl"
    target.write_text("old trace")
    monkeypatch.setattr(
        "deepfabric.output_safety._get_deepfabric_data_dir", lambda: tmp_path / "app"
    )
    with output_session():
        actual, stream = open_unique(target)
        with stream:
            stream.write("current trace")
        with pytest.raises(OutputSaveError, match="aliases"):
            save_output(actual, write_text, replace=True)
    assert actual.read_text() == "current trace"


def test_clear_cannot_delete_a_protected_input(tmp_path):
    path = tmp_path / "topics.jsonl"
    path.write_text("input")
    with output_session():
        protect_paths(path)
        with pytest.raises(OSError, match="aliases an input"):
            clear_output(path)
    assert path.read_text() == "input"
