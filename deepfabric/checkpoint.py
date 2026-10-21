"""Validation for versioned per-slot generation checkpoints."""

import json
import shlex

from pathlib import Path

from .constants import CHECKPOINT_VERSION
from .exceptions import DataSetGeneratorError
from .sampling import slot_limits, topic_plan, valid_slot


def iter_jsonl(path: Path):
    """Validate rows as a stream; resuming need not load saved samples into memory."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise DataSetGeneratorError(f"Invalid checkpoint row in {path}")
            yield row


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def version_error(version, path) -> str:
    if version == 4:  # noqa: PLR2004 - the supported migration source version
        return (
            f"Cannot resume version 4 checkpoint {path}; this generator requires version 5. "
            "Start a new generation with a separate checkpoint location, or run "
            f"python tools/migrate_checkpoint.py --checkpoint {shlex.quote(str(path))} "
            "--config <original-config> --topics <original-topics> --output-dir <new-directory>. "
            "Migration requires the original config and saved topics, and may fail if the original "
            "allocation plan or sample identities cannot be recovered unambiguously. "
            "The migration tool leaves source checkpoint files unchanged."
        )
    return f"Unsupported checkpoint version {version!r} at {path}; requires {CHECKPOINT_VERSION}"


def validate_checkpoint(metadata: dict, samples_path: Path, failures_path: Path) -> None:
    """Reject incomplete or inconsistent plans and outcome identities before resume."""
    if metadata.get("version") != CHECKPOINT_VERSION:
        raise DataSetGeneratorError(version_error(metadata.get("version"), samples_path))
    plan = metadata.get("plan")
    if not isinstance(plan, dict) or type(plan.get("target")) is not int or plan["target"] <= 0:
        raise DataSetGeneratorError("Checkpoint has no valid generation plan/target")
    if plan.get("generation_mode") == "topic_allocation":
        allocations = plan.get("allocations", [])
        if any(
            not isinstance(a.get("uuid"), str)
            or not isinstance(a.get("topic"), str)
            or not isinstance(a.get("path"), list)
            or not all(isinstance(p, str) for p in a["path"])
            or type(a.get("quota")) is not int
            for a in allocations
        ):
            raise DataSetGeneratorError("Invalid checkpoint allocation records")
        topics = [{k: a[k] for k in ("uuid", "topic", "path")} for a in allocations]
        if len({a["uuid"] for a in topics}) != len(topics):
            raise DataSetGeneratorError("Checkpoint topic UUIDs are not unique")
        if topic_plan(plan["target"], topics) != plan:
            raise DataSetGeneratorError("Checkpoint allocation does not match its exact target")
    elif plan.get("generation_mode") != "step_based":
        raise DataSetGeneratorError("Unknown checkpoint generation mode")
    limits = slot_limits(plan)
    seeds = metadata.get("seeds", {})
    attempts = metadata.get("collection_attempts", {})
    if not isinstance(seeds, dict) or not isinstance(attempts, dict):
        raise DataSetGeneratorError("Invalid checkpoint topic collection state")
    for topic_id, topic_seeds in seeds.items():
        if topic_id not in limits or not isinstance(topic_seeds, dict):
            raise DataSetGeneratorError("Invalid checkpoint seed topic")
        for ordinal in topic_seeds:
            if (
                not isinstance(ordinal, str)
                or not ordinal.isascii()
                or not ordinal.isdecimal()
                or str(int(ordinal)) != ordinal
                or not valid_slot((topic_id, int(ordinal)), limits)
            ):
                raise DataSetGeneratorError("Invalid checkpoint seed identity")
    if any(
        topic_id not in limits or type(count) is not int or count < 0
        for topic_id, count in attempts.items()
    ):
        raise DataSetGeneratorError("Invalid checkpoint collection attempt count")
    completed_list = metadata.get("completed", [])
    success_list = metadata.get("sample_slots", [])
    if not all(valid_slot(item, limits) for item in completed_list + success_list):
        raise DataSetGeneratorError("Checkpoint contains invalid sample identities")
    completed = {tuple(item) for item in completed_list}
    successes = {tuple(item) for item in success_list}
    sample_count = sum(1 for _ in iter_jsonl(samples_path))
    failures = read_jsonl(failures_path)
    failed_list = [(row["topic_id"], row["sample_ordinal"]) for row in failures]
    if not all(valid_slot(item, limits) for item in failed_list):
        raise DataSetGeneratorError("Checkpoint contains invalid failed identities")
    failed = set(failed_list)
    if (
        len(completed) != len(completed_list)
        or len(successes) != len(success_list)
        or len(failed) != len(failed_list)
        or successes & failed
        or completed != successes | failed
        or sample_count != len(success_list)
        or sample_count != metadata.get("total_samples")
        or len(failures) != metadata.get("total_failures")
    ):
        raise DataSetGeneratorError(
            "Checkpoint counts or completed sample identities are inconsistent"
        )
