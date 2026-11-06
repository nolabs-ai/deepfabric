"""Exact, ordered sample allocation shared by generation and checkpoint migration."""

from .exceptions import DataSetGeneratorError

STEP_TOPIC_ID = "__step_based__"


def resolve_target(num_samples: int | None, num_steps: int | None, batch_size: int) -> int:
    """Explicit targets override legacy steps; batch size never rounds the target."""
    if type(batch_size) is not int or batch_size <= 0:
        raise DataSetGeneratorError("batch_size must be a positive integer")
    if num_samples is not None:
        if type(num_samples) is not int or num_samples <= 0:
            raise DataSetGeneratorError("num_samples must be a positive integer")
        return num_samples
    steps = 1 if num_steps is None else num_steps
    if type(steps) is not int or steps <= 0:
        raise DataSetGeneratorError("num_steps must be a positive integer")
    return steps * batch_size


def topic_plan(total_samples: int, topics: list[dict]) -> dict:
    """Allocate quotas in the existing topic order, without route multiplication."""
    if not topics:
        raise DataSetGeneratorError("Topic model has no unique topics")
    resolve_target(total_samples, None, 1)
    if len({topic["uuid"] for topic in topics}) != len(topics):
        raise DataSetGeneratorError("Topic allocation requires unique UUIDs")
    base, remainder = divmod(total_samples, len(topics))
    allocations = [
        dict(topic, path=list(topic["path"]), quota=base + (i < remainder))
        for i, topic in enumerate(topics)
    ]
    return {
        "generation_mode": "topic_allocation",
        "target": total_samples,
        "allocations": allocations,
        "unique_topics": len(topics),
        "selected_topics": min(total_samples, len(topics)),
        "base_quota": base,
        "remainder": remainder,
    }


def slot_limits(plan: dict) -> dict[str, int]:
    """Represent slot bounds in O(topic count), even for very large quotas."""
    if plan["generation_mode"] == "step_based":
        return {STEP_TOPIC_ID: plan["target"]}
    return {a["uuid"]: a["quota"] for a in plan["allocations"]}


def valid_slot(slot, limits: dict[str, int]) -> bool:
    """Validate identity types and ordinal bounds without expanding the entire plan."""
    return (
        isinstance(slot, tuple | list)
        and len(slot) == 2  # noqa: PLR2004 - identity is exactly (uuid, ordinal)
        and isinstance(slot[0], str)
        and type(slot[1]) is int
        and 0 <= slot[1] < limits.get(slot[0], 0)
    )
