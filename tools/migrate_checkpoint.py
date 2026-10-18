"""Migrate saved generation checkpoints without modifying the originals.

Supported migrations: v4 -> v5
Scope: validated topic-model checkpoints with the original config and saved topics.
No-topic checkpoints and ambiguous/inconsistent original plans are not supported.
Select metadata with --checkpoint, or use --checkpoint-dir for a directory containing
exactly one checkpoint. Migration writes .tmp files into a new destination directory,
validates them, and removes their temporary suffixes, publishing metadata last.
A hard termination may leave an incomplete destination; remove it before retrying.
"""
# ruff: noqa: E402 - source-checkout bootstrap must precede DeepFabric imports

import argparse
import json
import os
import shlex
import shutil
import sys

from collections import defaultdict, deque
from pathlib import Path

# Support direct execution from a source checkout without changing import paths in tests.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepfabric.checkpoint import read_jsonl, validate_checkpoint
from deepfabric.config import DeepFabricConfig
from deepfabric.constants import (
    CHECKPOINT_FAILURES_SUFFIX,
    CHECKPOINT_METADATA_SUFFIX,
    CHECKPOINT_SAMPLES_SUFFIX,
    CHECKPOINT_VERSION,
)
from deepfabric.dataset_manager import resolve_num_samples
from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.graph import Graph, GraphModel, Node
from deepfabric.sampling import resolve_target, slot_limits, topic_plan, valid_slot
from deepfabric.tree import Tree
from deepfabric.utils import parse_num_samples


def load_migration_topics(path: Path, mode: str):
    """Restore only saved topology, without constructing provider clients or making requests."""
    if mode == "tree":
        model = Tree.__new__(Tree)
        model.from_dict_list(read_jsonl(path))
        return model
    with path.open(encoding="utf-8") as stream:
        saved = GraphModel(**json.load(stream))
    model = Graph.__new__(Graph)
    model.nodes = {
        node.id: Node(node.topic, node.id, node.metadata) for node in saved.nodes.values()
    }
    model.root = model.nodes[saved.root_id]
    model._clear_uuid_index()
    for node in saved.nodes.values():
        for child in node.children:
            if child not in model.nodes:
                raise DataSetGeneratorError("Saved topic graph references a missing node")
            model.add_edge(node.id, child)
    return model


def migrate_v4_to_v5(
    checkpoint: Path,
    config_path: Path,
    topics_path: Path,
    output_dir: Path,
    *,
    num_samples: str | int | None = None,
    batch_size: int | None = None,
    num_steps: int | None = None,
) -> tuple[Path, int, int]:
    """Validate original inputs and publish a separate, fully validated migrated checkpoint set."""
    with checkpoint.open(encoding="utf-8") as stream:
        original = json.load(stream)
    if original.get("version") != 4 or CHECKPOINT_VERSION != 5:  # noqa: PLR2004
        raise DataSetGeneratorError("Unsupported migration path; supported migrations: v4 -> v5")
    if not checkpoint.name.endswith(CHECKPOINT_METADATA_SUFFIX):
        raise DataSetGeneratorError("--checkpoint must name a checkpoint metadata file")
    stem = checkpoint.name.removesuffix(CHECKPOINT_METADATA_SUFFIX)
    config = DeepFabricConfig.from_yaml(str(config_path))
    model = load_migration_topics(topics_path, config.topics.mode)
    topics = [
        {"uuid": t.uuid, "topic": t.topic, "path": model.get_path_by_id(t.uuid) or [t.topic]}
        for t in model.get_unique_topics()
    ]
    if not topics:
        raise DataSetGeneratorError("Cannot migrate an empty topic model or no-topic checkpoint")
    batch = config.output.batch_size if batch_size is None else batch_size
    requested = resolve_num_samples(
        parse_num_samples(num_samples) if num_samples is not None else config.output.num_samples,
        len(topics),
    )
    if num_steps is not None:
        target = resolve_target(None, num_steps, batch)
        requested = target
    else:
        resolve_target(requested, None, batch)
        target = ((requested + batch - 1) // batch) * batch
    plan = topic_plan(target, topics)
    limits = slot_limits(plan)
    completed_list = original.get("completed")
    if not isinstance(completed_list, list):
        raise DataSetGeneratorError("Missing original completed identities")
    if not all(valid_slot(item, limits) for item in completed_list):
        raise DataSetGeneratorError("Original completion identities do not match the supplied plan")
    completed = {tuple(item) for item in completed_list}
    if len(completed) != len(completed_list):
        raise DataSetGeneratorError(
            "Original completed identities do not match the supplied topic plan/target"
        )
    samples_path = checkpoint.with_name(stem + CHECKPOINT_SAMPLES_SUFFIX)
    failures_path = checkpoint.with_name(stem + CHECKPOINT_FAILURES_SUFFIX)
    samples, failures = read_jsonl(samples_path), read_jsonl(failures_path)
    if len(samples) != original.get("total_samples") or len(failures) != original.get(
        "total_failures", 0
    ):
        raise DataSetGeneratorError("Original checkpoint file counts do not match metadata")
    migrated_failures, failed = [], set()
    for failure in failures:
        if "topic_id" not in failure or type(failure.get("cycle")) is not int:
            raise DataSetGeneratorError(
                "Failure lacks a trustworthy topic_id/cycle; cannot infer its slot"
            )
        slot = (failure["topic_id"], failure["cycle"])
        if slot in failed or slot not in completed:
            raise DataSetGeneratorError(
                "Failure identities are duplicate or not in original completion state"
            )
        failed.add(slot)
        migrated = dict(failure)
        migrated["sample_ordinal"] = migrated.pop("cycle")
        migrated_failures.append(migrated)
    success_by_topic = defaultdict(deque)
    for uuid, ordinal in sorted(completed - failed):
        success_by_topic[uuid].append(ordinal)
    sample_slots = []
    for sample in samples:
        uuid = (sample.get("metadata") or {}).get("topic_id")
        if uuid is None:
            candidates = [key for key, slots in success_by_topic.items() if slots]
            if len(candidates) != 1:
                raise DataSetGeneratorError(
                    "Sample lacks topic identity; original row-to-slot mapping is ambiguous"
                )
            uuid = candidates[0]
        if not success_by_topic[uuid]:
            raise DataSetGeneratorError("Sample topic does not match original successful slots")
        sample_slots.append([uuid, success_by_topic[uuid].popleft()])
    if any(success_by_topic.values()):
        raise DataSetGeneratorError("Original successful slots do not match persisted rows")
    # Reject positive legacy plan metadata that contradicts the supplied original inputs.
    if original.get("unique_topics", 0) not in (0, len(topics)):
        raise DataSetGeneratorError(
            "Original unique topic count differs from the supplied saved topics"
        )
    if original.get("cycles_needed", 0) not in (0, (target + len(topics) - 1) // len(topics)):
        raise DataSetGeneratorError(
            "Original cycle count differs from the supplied effective target"
        )
    metadata = {k: v for k, v in original.items() if k not in {"cycles_needed", "unique_topics"}}
    metadata.update(
        version=5,
        plan=plan,
        sample_slots=sample_slots,
        topics_file=str(topics_path.resolve()),
        total_samples=len(samples),
        total_failures=len(migrated_failures),
    )
    if output_dir.exists():
        raise DataSetGeneratorError("Migration destination already exists; choose a new directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the destination without overwriting another checkpoint set.
    output_dir.mkdir()
    try:
        out_metadata = output_dir / (checkpoint.name + ".tmp")
        out_samples = output_dir / (samples_path.name + ".tmp")
        out_failures = output_dir / (failures_path.name + ".tmp")
        out_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        out_samples.write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
        out_failures.write_text(
            "".join(json.dumps(row) + "\n" for row in migrated_failures), encoding="utf-8"
        )
        validate_checkpoint(metadata, out_samples, out_failures)
        # Publish metadata last so a consumer never sees a resumable partial set.
        for source in (out_samples, out_failures, out_metadata):
            os.replace(source, source.with_suffix(""))
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return output_dir / checkpoint.name, requested, target


MIGRATIONS = {(4, 5): migrate_v4_to_v5}


def migrate_checkpoint(
    checkpoint: Path,
    config_path: Path,
    topics_path: Path,
    output_dir: Path,
    *,
    target_version: int = CHECKPOINT_VERSION,
    **overrides,
) -> tuple[Path, int, int]:
    """Dispatch supported source/target versions; the default target is the current format."""
    with checkpoint.open(encoding="utf-8") as stream:
        source_version = json.load(stream).get("version")
    migration = MIGRATIONS.get((source_version, target_version))
    if migration is None:
        raise DataSetGeneratorError(
            f"Unsupported migration path {source_version!r} -> {target_version}; "
            "supported migrations: v4 -> v5 (topic-model checkpoints)"
        )
    return migration(checkpoint, config_path, topics_path, output_dir, **overrides)


def discover_checkpoint(directory: Path) -> Path:
    """Select the sole checkpoint metadata file directly inside a directory."""
    if not directory.is_dir():
        raise DataSetGeneratorError(
            f"Checkpoint directory does not exist or is not a directory: {directory}"
        )
    candidates = sorted(
        path for path in directory.glob(f"*{CHECKPOINT_METADATA_SUFFIX}") if path.is_file()
    )
    if not candidates:
        raise DataSetGeneratorError(f"No checkpoint metadata found in {directory}")
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates)
        raise DataSetGeneratorError(
            f"Multiple checkpoints found in {directory}: {names}; select one with --checkpoint"
        )
    return candidates[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Checkpoint metadata JSON file")
    source.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory containing exactly one checkpoint metadata file",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--topics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument(
        "--num-samples", help="Original run's overridden count (integer, auto, or percentage)"
    )
    targets.add_argument("--num-steps", type=int, help="Original Python run's legacy step count")
    parser.add_argument("--batch-size", type=int, help="Original run's overridden batch size")
    args = parser.parse_args(argv)
    try:
        checkpoint = (
            args.checkpoint
            if args.checkpoint is not None
            else discover_checkpoint(args.checkpoint_dir)
        )
        path, requested, target = migrate_checkpoint(
            checkpoint,
            args.config,
            args.topics,
            args.output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
        )
    except Exception as exc:  # noqa: BLE001
        parser.exit(1, f"Migration failed: {exc}\n")
    print(f"Migrated checkpoint: {path}")
    print(f"Original requested target: {requested}; preserved effective target: {target}")
    print("Resume with the original saved topics and the exact effective target:")
    print(
        f"deepfabric generate {shlex.quote(str(args.config))} --topics-load {shlex.quote(str(args.topics))} "
        f"--checkpoint-path {shlex.quote(str(args.output_dir))} --checkpoint-interval 10 --resume --num-samples {target}"
    )


if __name__ == "__main__":
    main()
