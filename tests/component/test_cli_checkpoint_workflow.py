"""Checkpoint decisions precede topic work and retry only after plan validation."""
# ruff: noqa: PLR2004

import json

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from click.testing import CliRunner

from deepfabric.cli import _prompt_with_timeout, cli
from deepfabric.generator import DataSetGenerator
from deepfabric.schemas import TopicList
from deepfabric.utils import build_checkpoint_file_paths


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "llm": {"provider": "openai", "model": "test"},
                "topics": {
                    "prompt": "Root",
                    "mode": "tree",
                    "depth": 1,
                    "degree": 2,
                    "save_as": str(tmp_path / "topics.jsonl"),
                },
                "generation": {"system_prompt": "Generate"},
                "output": {
                    "save_as": str(tmp_path / "dataset.jsonl"),
                    "num_samples": 4,
                    "batch_size": 1,
                    "checkpoint": {"interval": 1, "path": str(tmp_path / "checkpoints")},
                },
            }
        )
    )
    state = SimpleNamespace(config=config, calls=[], stop_after=None, fail_again=False)

    async def outcome(engine, prompt, include_sys_msg, sample_idx, topic_path, identity):  # noqa: ARG001
        state.calls.append(identity)
        if state.stop_after == len(state.calls):
            engine.stop_requested = True
        if len(state.calls) == 1 or state.fail_again:
            return None, {
                "topic_id": identity[0],
                "sample_ordinal": identity[1],
                "error": "test failure",
            }
        return {"messages": [{"role": "assistant", "content": f"sample-{len(state.calls)}"}]}, None

    with (
        patch("deepfabric.cli.check_for_updates"),
        patch("deepfabric.cli._validate_api_keys"),
        patch("deepfabric.cli.trace"),
        patch("deepfabric.cli._trigger_cloud_upload"),
        patch("deepfabric.cli.platform.system", return_value="Linux"),
        patch("deepfabric.llm.client.make_outlines_model", return_value=Mock()),
        patch("deepfabric.llm.client.make_async_outlines_model", return_value=Mock()),
        patch(
            "deepfabric.llm.client.LLMClient.generate_async",
            new_callable=AsyncMock,
            return_value=TopicList(subtopics=["A", "B"]),
        ) as topics,
        patch.object(
            DataSetGenerator, "_generate_sample_outcome", autospec=True, side_effect=outcome
        ),
        patch("deepfabric.cli.select.select", return_value=([object()], [], [])),
    ):
        state.topics = topics
        state.runner = CliRunner()
        state.run = lambda *flags: state.runner.invoke(
            cli, ["generate", str(config), *flags], input=""
        )
        state.paths = build_checkpoint_file_paths("dataset.jsonl", str(tmp_path / "checkpoints"))
        state.dataset = tmp_path / "dataset.jsonl"
        state.snapshot = lambda: {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        yield state


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize(
    "flags", [(), ("--resume",), ("--retry-failed",), ("--resume", "--retry-failed")]
)
def test_completed_failure_rerun_retries_instead_of_reexporting(workflow, flags):
    first = workflow.run()
    assert first.exit_code == 0, first.output
    original_rows = rows(workflow.dataset)
    assert len(original_rows) == 3
    metadata = json.loads(workflow.paths[0].read_text())
    assert len(metadata["completed"]) == 4
    assert metadata["total_failures"] == 1
    exports = list(workflow.dataset.parent.glob("dataset_failures_*.jsonl"))
    topic_calls = workflow.topics.await_count

    second = workflow.run(*flags)

    assert second.exit_code == 0, second.output
    assert len(workflow.calls) == 5  # Only the failed slot is reopened.
    assert workflow.calls[-1] == workflow.calls[0]
    assert workflow.topics.await_count == topic_calls
    assert len(rows(workflow.dataset)) == 4
    assert all(row in rows(workflow.dataset) for row in original_rows)
    assert not any(path.exists() for path in workflow.paths)
    assert list(workflow.dataset.parent.glob("dataset_failures_*.jsonl")) == exports
    if "--retry-failed" not in flags:
        assert "Plain resume would generate no new samples" in second.output
        assert "Retry failed samples [default]" in second.output
    else:
        assert "Choose [" not in second.output


@pytest.mark.parametrize("retry", [False, True])
def test_mixed_checkpoint_resume_and_retry_are_distinct(workflow, retry):
    settings = yaml.safe_load(workflow.config.read_text())
    settings["output"]["num_samples"] = 5
    workflow.config.write_text(yaml.safe_dump(settings))
    workflow.stop_after = 4
    first = workflow.run()
    assert first.exit_code == 0, first.output
    assert len(workflow.calls) == 4
    assert len(json.loads(workflow.paths[0].read_text())["completed"]) == 4
    workflow.stop_after = None

    if retry:
        with patch("deepfabric.cli._prompt_with_timeout", return_value="2"):
            result = workflow.run()
    else:
        result = workflow.run()

    assert result.exit_code == 0, result.output
    assert "Plain resume would generate no new samples" not in result.output
    assert "1 outstanding" in result.output
    assert "Previously failed slots will be skipped" not in result.output
    assert len(workflow.calls) == (6 if retry else 5)
    assert len(rows(workflow.dataset)) == (5 if retry else 4)
    assert workflow.paths[0].exists() is not retry


@pytest.mark.parametrize("choice", ["2", "3"])
def test_completed_failure_clear_or_abort_happens_before_topic_generation(workflow, choice):
    assert workflow.run().exit_code == 0
    before = workflow.snapshot()
    topic_calls = workflow.topics.await_count
    with patch("deepfabric.cli._prompt_with_timeout", return_value=choice):
        result = workflow.run()
    assert result.exit_code == 0, result.output
    if choice == "3":
        assert len(workflow.calls) == 4
        assert workflow.topics.await_count == topic_calls
        assert workflow.snapshot() == before
    else:
        assert len(workflow.calls) == 8
        assert len(rows(workflow.dataset)) == 4
        assert workflow.topics.await_count > topic_calls
        assert not workflow.paths[0].exists()


@pytest.mark.parametrize("flags", [("--retry-failed",), ("--resume",)])
def test_retry_plan_mismatch_preserves_checkpoint_and_topics(workflow, flags):
    assert workflow.run().exit_code == 0
    settings = yaml.safe_load(workflow.config.read_text())
    settings["output"]["num_samples"] = 5
    workflow.config.write_text(yaml.safe_dump(settings))
    before = workflow.snapshot()
    result = workflow.run(*flags)
    assert result.exit_code == 1, result.output
    assert "Checkpoint target/topic plan differs" in result.output
    assert len(workflow.calls) == 4
    assert workflow.snapshot() == before


@pytest.mark.parametrize(
    "flags", [("--resume",), ("--retry-failed",), ("--resume", "--retry-failed")]
)
def test_explicit_continuation_with_outstanding_slots_bypasses_prompt(workflow, flags):
    settings = yaml.safe_load(workflow.config.read_text())
    settings["output"]["num_samples"] = 5
    workflow.config.write_text(yaml.safe_dump(settings))
    workflow.stop_after = 4
    assert workflow.run().exit_code == 0
    workflow.stop_after = None

    with patch(
        "deepfabric.cli._prompt_with_timeout", side_effect=AssertionError("unexpected prompt")
    ):
        result = workflow.run(*flags)

    assert result.exit_code == 0, result.output
    retry = "--retry-failed" in flags
    guidance = (
        "Previously failed slots will be skipped. Use --retry-failed to continue "
        "outstanding work and retry failed slots."
    )
    assert (guidance in " ".join(result.output.split())) is not retry
    assert len(workflow.calls) == (6 if retry else 5)
    assert len(rows(workflow.dataset)) == (5 if retry else 4)


def test_missing_saved_topics_do_not_regenerate_or_modify_checkpoint(workflow):
    assert workflow.run().exit_code == 0
    workflow.dataset.with_name("topics.jsonl").unlink()
    before = workflow.snapshot()
    topic_calls = workflow.topics.await_count

    result = workflow.run("--retry-failed")

    assert result.exit_code == 1, result.output
    assert "provide the original saved topics with --topics-load" in " ".join(result.output.split())
    assert len(workflow.calls) == 4
    assert workflow.topics.await_count == topic_calls
    assert workflow.snapshot() == before


@pytest.mark.parametrize(
    "flags", [("--resume",), ("--retry-failed",), ("--resume", "--retry-failed")]
)
def test_missing_checkpoint_preserves_fresh_run_fallback(workflow, flags):
    result = workflow.run(*flags)
    assert result.exit_code == 0, result.output
    assert "No checkpoint found, starting fresh generation" in result.output
    assert len(workflow.calls) == 4


def test_timeout_uses_displayed_retry_default(workflow):
    assert workflow.run().exit_code == 0

    def timed_prompt(*args, **kwargs):
        with patch("deepfabric.cli.sys.stdin", Mock(fileno=lambda: 0)):
            return _prompt_with_timeout(*args, **kwargs)

    with (
        patch("deepfabric.cli.select.select", return_value=([], [], [])),
        patch("deepfabric.cli._prompt_with_timeout", side_effect=timed_prompt),
    ):
        result = workflow.run()
    assert result.exit_code == 0, result.output
    assert "auto-1 in" in result.output
    assert len(workflow.calls) == 5


def test_retries_that_fail_again_are_retained_without_looping(workflow):
    assert workflow.run().exit_code == 0
    workflow.fail_again = True
    result = workflow.run("--retry-failed")
    assert result.exit_code == 0, result.output
    assert len(workflow.calls) == 5
    assert len(rows(workflow.dataset)) == 3
    assert json.loads(workflow.paths[0].read_text())["total_failures"] == 1


def test_configured_retry_setting_requests_retry_without_a_prompt(workflow):
    assert workflow.run().exit_code == 0
    settings = yaml.safe_load(workflow.config.read_text())
    settings["output"]["checkpoint"]["retry_failed"] = True
    workflow.config.write_text(yaml.safe_dump(settings))
    with patch(
        "deepfabric.cli._prompt_with_timeout", side_effect=AssertionError("unexpected prompt")
    ):
        result = workflow.run()
    assert result.exit_code == 0, result.output
    assert len(workflow.calls) == 5
    assert not workflow.paths[0].exists()


def test_retry_without_checkpointing_preserves_fresh_run_fallback(workflow):
    settings = yaml.safe_load(workflow.config.read_text())
    settings["output"].pop("checkpoint")
    workflow.config.write_text(yaml.safe_dump(settings))
    result = workflow.run("--retry-failed")
    assert result.exit_code == 0, result.output
    assert "No checkpoint found, starting fresh generation" in result.output
    assert len(workflow.calls) == 4
    assert not workflow.paths[0].exists()


def completed_success_checkpoint(workflow):
    """Leave the final successful checkpoint as an interrupted cleanup would."""
    assert workflow.run().exit_code == 0
    with patch.object(DataSetGenerator, "clear_checkpoint"):
        result = workflow.run("--retry-failed")
    assert result.exit_code == 0, result.output
    assert json.loads(workflow.paths[0].read_text())["total_failures"] == 0
    workflow.dataset.unlink()


@pytest.mark.parametrize(
    "flags", [("--resume",), ("--retry-failed",), ("--resume", "--retry-failed")]
)
def test_complete_success_explicit_continuation_saves_without_generation(workflow, flags):
    completed_success_checkpoint(workflow)
    calls = len(workflow.calls)
    topic_calls = workflow.topics.await_count
    result = workflow.run(*flags)
    assert result.exit_code == 0, result.output
    assert "No generation remains" in result.output
    assert "Save samples to output and remove checkpoint [default]" in result.output
    assert len(rows(workflow.dataset)) == 4
    assert len(workflow.calls) == calls
    assert workflow.topics.await_count == topic_calls
    assert not any(path.exists() for path in workflow.paths)


def test_complete_success_plain_invocation_defaults_to_abort(workflow):
    completed_success_checkpoint(workflow)
    before = workflow.snapshot()
    calls = len(workflow.calls)
    result = workflow.run()
    assert result.exit_code == 0, result.output
    assert "Abort [default]" in result.output
    assert workflow.snapshot() == before
    assert len(workflow.calls) == calls


@pytest.mark.parametrize("failure", ["plan", "topics", "save", "cleanup"])
def test_complete_success_recovery_failures_preserve_checkpoint(workflow, failure):
    completed_success_checkpoint(workflow)
    if failure == "plan":
        settings = yaml.safe_load(workflow.config.read_text())
        settings["output"]["num_samples"] = 5
        workflow.config.write_text(yaml.safe_dump(settings))
    elif failure == "topics":
        workflow.dataset.with_name("topics.jsonl").unlink()
    before = {p: p.read_bytes() for p in workflow.paths if p.exists()}
    calls = len(workflow.calls)
    if failure == "save":
        with patch("deepfabric.cli.save_dataset", side_effect=OSError("test save failure")):
            result = workflow.run("--resume")
    elif failure == "cleanup":
        with patch.object(
            DataSetGenerator, "clear_checkpoint", side_effect=OSError("test cleanup failure")
        ):
            result = workflow.run("--resume")
    else:
        result = workflow.run("--resume")
    assert result.exit_code == (0 if failure == "cleanup" else 1), result.output
    if failure in ("save", "cleanup"):
        # Accounting records the recovery invocation; saved slot outcomes remain intact.
        for path, content in before.items():
            if path == workflow.paths[0]:
                original = json.loads(content)
                current = json.loads(path.read_bytes())
                for metadata in (original, current):
                    metadata.pop("created_at", None)
                    metadata.pop("token_accounting", None)
                assert current == original
            else:
                assert path.read_bytes() == content
    else:
        assert {p: p.read_bytes() for p in before} == before
    assert len(workflow.calls) == calls
    if failure == "cleanup":
        assert len(rows(workflow.dataset)) == 4
        assert "Failed to clean up checkpoint files" in result.output
    else:
        assert not workflow.dataset.exists()


@pytest.mark.parametrize("choice", ["1", "2", "3"])
def test_complete_success_explicit_choices(workflow, choice):
    completed_success_checkpoint(workflow)
    before = workflow.snapshot()
    calls = len(workflow.calls)
    with patch("deepfabric.cli._prompt_with_timeout", return_value=choice):
        result = workflow.run()
    assert result.exit_code == 0, result.output
    if choice == "1":
        assert len(rows(workflow.dataset)) == 4
        assert len(workflow.calls) == calls
        assert not any(path.exists() for path in workflow.paths)
    elif choice == "2":
        assert len(workflow.calls) == calls + 4
        assert len(rows(workflow.dataset)) == 4
    else:
        assert workflow.snapshot() == before
        assert len(workflow.calls) == calls


def break_checkpoint(workflow, kind):
    assert workflow.run().exit_code == 0
    metadata = workflow.paths[0]
    if kind == "torn":
        metadata.write_text(metadata.read_text()[:40])
    else:
        metadata.write_text(json.dumps({"version": 4, "completed": [], "total_samples": 0}))


@pytest.mark.parametrize("kind", ["torn", "v4"])
@pytest.mark.parametrize("flags", [(), ("--resume",), ("--retry-failed",)])
def test_unloadable_checkpoint_defaults_to_abort_without_changes(workflow, kind, flags):
    break_checkpoint(workflow, kind)
    before = workflow.snapshot()
    calls = len(workflow.calls)
    topic_calls = workflow.topics.await_count

    result = workflow.run(*flags)

    assert result.exit_code == 1, result.output
    assert "Existing checkpoint cannot be loaded" in result.output
    assert "Abort [default]" in result.output
    assert workflow.snapshot() == before
    assert len(workflow.calls) == calls
    assert workflow.topics.await_count == topic_calls


@pytest.mark.parametrize("kind", ["torn", "v4"])
def test_unloadable_checkpoint_can_be_cleared_for_a_fresh_run(workflow, kind):
    break_checkpoint(workflow, kind)
    calls = len(workflow.calls)
    original_rows = rows(workflow.dataset)

    with patch("deepfabric.cli._prompt_with_timeout", return_value="1"):
        result = workflow.run()

    assert result.exit_code == 0, result.output
    assert "Checkpoint cleared, starting fresh generation" in result.output
    assert len(workflow.calls) == calls + 4
    assert rows(workflow.dataset) == original_rows
    assert len(rows(workflow.dataset.with_name("dataset_1.jsonl"))) == 4
    assert not any(path.exists() for path in workflow.paths)
