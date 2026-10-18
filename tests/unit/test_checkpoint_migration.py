"""Offline v4-to-v5 migration, refusal paths, and real script entry-point coverage."""
# ruff: noqa: PLR2004, ARG001

import json
import os
import subprocess
import sys

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from deepfabric.constants import (
    CHECKPOINT_FAILURES_SUFFIX,
    CHECKPOINT_METADATA_SUFFIX,
    CHECKPOINT_SAMPLES_SUFFIX,
)
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.generator import DataSetGenerator
from tools.migrate_checkpoint import (
    discover_checkpoint,
    load_migration_topics,
    main,
    migrate_checkpoint,
    migrate_v4_to_v5,
)


@pytest.fixture
def original(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "topics": {"prompt": "Root", "mode": "tree"},
                "generation": {"system_prompt": "Generate"},
                "output": {"num_samples": 5, "batch_size": 4, "save_as": "samples.jsonl"},
            }
        )
    )
    topics = tmp_path / "tree.jsonl"
    topics.write_text(' {"path":["Root","A"]}\n{"path":["Root","B"]}\n')
    uuids = [topic.uuid for topic in load_migration_topics(topics, "tree").get_unique_topics()]
    a, b = uuids
    metadata = tmp_path / ("samples" + CHECKPOINT_METADATA_SUFFIX)
    metadata.write_text(
        json.dumps(
            {
                "version": 4,
                "total_samples": 2,
                "total_failures": 1,
                "completed": [[a, 0], [b, 0], [a, 1]],
                "unique_topics": 0,
                "cycles_needed": 0,
                "provider": "openai",
                "model_name": "test",
                "checkpoint_interval": 1,
            }
        )
    )
    samples = tmp_path / ("samples" + CHECKPOINT_SAMPLES_SUFFIX)
    samples.write_text(
        "".join(
            json.dumps({"question": uuid, "metadata": {"topic_id": uuid}}) + "\n" for uuid in uuids
        )
    )
    failures = tmp_path / ("samples" + CHECKPOINT_FAILURES_SUFFIX)
    failures.write_text(json.dumps({"topic_id": a, "cycle": 1, "error": "failed"}) + "\n")
    return metadata, config, topics, tmp_path / "migrated"


async def test_partial_migration_and_resume_preserves_rows(original):
    metadata, config, topics, destination = original
    source_files = list(metadata.parent.iterdir())
    before = {p: p.read_bytes() for p in source_files}
    output, requested, effective = migrate_v4_to_v5(*original)
    assert (requested, effective) == (5, 8)
    assert all(p.read_bytes() == content for p, content in before.items())
    converted = json.loads(output.read_text())
    assert converted["version"] == 5
    assert converted["plan"]["target"] == 8
    assert "cycles_needed" not in converted
    assert len(converted["sample_slots"]) == 2
    model = load_migration_topics(topics, "tree")
    with patch("deepfabric.generator.LLMClient"):
        engine = DataSetGenerator(
            generation_system_prompt="Generate",
            provider="openai",
            model_name="test",
            checkpoint_interval=1,
            checkpoint_path=str(destination),
            output_save_as="samples.jsonl",
        )
    assert engine.load_checkpoint()

    async def outcome(prompt, include_sys_msg, sample_idx, topic_path, identity):
        return {"question": str(identity), "metadata": {"topic_id": identity[0]}}, None

    engine._generate_sample_outcome = AsyncMock(side_effect=outcome)
    with pytest.raises(DataSetGeneratorError, match="plan differs"):
        await engine.create_data_async(num_samples=5, topic_model=model)
    result = await engine.create_data_async(num_samples=8, topic_model=model)
    assert len(result) == 7  # old terminal failure remains skipped
    assert engine._generate_sample_outcome.await_count == 5
    engine.load_checkpoint(retry_failed=True)
    engine._generate_sample_outcome.reset_mock()
    result = await engine.create_data_async(num_samples=8, topic_model=model)
    assert len(result) == 8
    assert engine._generate_sample_outcome.await_count == 1


@pytest.mark.parametrize(
    "override,expected", [({"num_samples": 9}, 12), ({"batch_size": 3}, 6), ({"num_steps": 3}, 12)]
)
def test_original_overrides(original, override, expected):
    output, _, target = migrate_v4_to_v5(*original, **override)
    assert target == expected
    assert json.loads(output.read_text())["plan"]["target"] == expected


@pytest.mark.parametrize(
    "problem",
    [
        "version",
        "bad_uuid",
        "bad_ordinal",
        "bad_count",
        "duplicate",
        "missing_cycle",
        "ambiguous_sample",
        "no_topic",
        "destination",
    ],
)
def test_refuses_inconsistent_or_unsupported_input(original, problem):
    metadata, _, _, destination = original
    data = json.loads(metadata.read_text())
    if problem == "version":
        data["version"] = 3
    elif problem == "bad_uuid":
        data["completed"][0][0] = "unknown"
    elif problem == "bad_ordinal":
        data["completed"][0][1] = 100
    elif problem == "bad_count":
        data["total_samples"] = 10
    elif problem == "duplicate":
        data["completed"].append(data["completed"][0])
    elif problem == "no_topic":
        data["completed"] = [["__step_based__", 0]]
    elif problem == "destination":
        destination.mkdir()
        (destination / "keep").write_text("existing")
    elif problem == "missing_cycle":
        path = metadata.with_name("samples" + CHECKPOINT_FAILURES_SUFFIX)
        failure = json.loads(path.read_text())
        del failure["cycle"]
        path.write_text(json.dumps(failure) + "\n")
    elif problem == "ambiguous_sample":
        path = metadata.with_name("samples" + CHECKPOINT_SAMPLES_SUFFIX)
        path.write_text('{"question":"no identity"}\n{"question":"no identity"}\n')
    metadata.write_text(json.dumps(data))
    before = {p: p.read_bytes() for p in metadata.parent.iterdir() if p.is_file()}
    with pytest.raises((DataSetGeneratorError, ValueError, KeyError, TypeError)):
        migrate_v4_to_v5(*original)
    assert all(p.read_bytes() == content for p, content in before.items())
    if problem != "destination":
        assert not destination.exists()
    else:
        assert (destination / "keep").read_text() == "existing"


@pytest.mark.parametrize("directory_input", [False, True])
def test_real_script_entry_point(original, directory_input):
    metadata, config, topics, destination = original
    script = Path(__file__).resolve().parents[2] / "tools" / "migrate_checkpoint.py"
    result = subprocess.run(  # noqa: S603 - fixed local tool and temporary test paths
        [
            sys.executable,
            str(script),
            "--checkpoint-dir" if directory_input else "--checkpoint",
            str(metadata.parent if directory_input else metadata),
            "--config",
            str(config),
            "--topics",
            str(topics),
            "--output-dir",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "preserved effective target: 8" in result.stdout
    assert "--num-samples 8" in result.stdout
    assert "Supported migrations: v4 -> v5" in script.read_text()


def test_graph_topology_loaded_offline():
    path = Path(__file__).resolve().parents[1] / "fixtures" / "graph_path_context.json"
    with patch("deepfabric.graph.LLMClient", side_effect=AssertionError("must be offline")):
        model = load_migration_topics(path, "graph")
    assert len(model.get_unique_topics()) == 3
    leaf = model.get_unique_topics()[-1]
    assert model.get_path_by_id(leaf.uuid) == ["Root", "Branch", "Interior", "Leaf"]


def test_version_dispatch_rejects_unsupported_target(original):
    with pytest.raises(DataSetGeneratorError, match="Unsupported migration path 4 -> 6"):
        migrate_checkpoint(*original, target_version=6)
    assert not original[-1].exists()


@pytest.mark.parametrize("problem", ["missing", "file", "empty", "multiple"])
def test_checkpoint_directory_discovery_errors(original, tmp_path, problem):
    metadata, _, _, _ = original
    directory = tmp_path / "input"
    if problem == "missing":
        match = "does not exist or is not a directory"
    elif problem == "file":
        directory.write_text("not a directory")
        match = "does not exist or is not a directory"
    elif problem == "empty":
        directory.mkdir()
        # Nested checkpoints and unfinished metadata are not discovery candidates.
        (directory / (metadata.name + ".tmp")).write_text("unfinished")
        (directory / ("nested" + CHECKPOINT_METADATA_SUFFIX)).mkdir()
        match = "No checkpoint metadata"
    else:
        directory = metadata.parent
        (directory / ("other" + CHECKPOINT_METADATA_SUFFIX)).write_text("{}")
        match = "Multiple checkpoints.*--checkpoint"
    with pytest.raises(DataSetGeneratorError, match=match):
        discover_checkpoint(directory)


@pytest.mark.parametrize("both", [False, True])
def test_cli_requires_exactly_one_checkpoint_source(original, both, capsys):
    metadata, config, topics, destination = original
    args = ["--config", str(config), "--topics", str(topics), "--output-dir", str(destination)]
    if both:
        args += ["--checkpoint", str(metadata), "--checkpoint-dir", str(metadata.parent)]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "--checkpoint" in capsys.readouterr().err
    assert not destination.exists()


def test_cli_reports_directory_discovery_error(original, capsys):
    _, config, topics, destination = original
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--checkpoint-dir",
                str(destination),
                "--config",
                str(config),
                "--topics",
                str(topics),
                "--output-dir",
                str(destination),
            ]
        )
    assert error.value.code == 1
    assert "Migration failed: Checkpoint directory" in capsys.readouterr().err
    assert not destination.exists()


def test_temporary_files_validated_then_published_with_metadata_last(original):
    metadata, _, _, destination = original
    rename = os.replace
    published = []

    def publish(source, target):
        assert source.parent == target.parent == destination
        assert source.name == target.name + ".tmp"
        assert not (destination / metadata.name).exists()
        assert not list(destination.parent.glob(".checkpoint-migration-*"))
        published.append(target.name)
        rename(source, target)

    with patch("tools.migrate_checkpoint.os.replace", side_effect=publish):
        migrate_v4_to_v5(*original)
    assert published == [
        "samples" + CHECKPOINT_SAMPLES_SUFFIX,
        "samples" + CHECKPOINT_FAILURES_SUFFIX,
        metadata.name,
    ]
    assert sorted(p.name for p in destination.iterdir()) == sorted(published)


@pytest.mark.parametrize("phase", ["validation", "samples", "failures", "metadata"])
def test_failed_validation_or_publication_removes_new_destination(original, phase):
    metadata, _, _, destination = original
    before = {p: p.read_bytes() for p in metadata.parent.iterdir() if p.is_file()}
    rename = os.replace

    def publish(source, target):
        suffix = {
            "samples": CHECKPOINT_SAMPLES_SUFFIX,
            "failures": CHECKPOINT_FAILURES_SUFFIX,
            "metadata": CHECKPOINT_METADATA_SUFFIX,
        }[phase]
        if target.name.endswith(suffix):
            raise OSError("publication failed")
        rename(source, target)

    target = "validate_checkpoint" if phase == "validation" else "os.replace"
    failure = ValueError("validation failed") if phase == "validation" else publish
    with (
        patch(f"tools.migrate_checkpoint.{target}", side_effect=failure),
        pytest.raises((ValueError, OSError), match="failed"),
    ):
        migrate_v4_to_v5(*original)
    assert not destination.exists()
    assert not list(destination.parent.glob(".checkpoint-migration-*"))
    assert all(p.read_bytes() == content for p, content in before.items())
