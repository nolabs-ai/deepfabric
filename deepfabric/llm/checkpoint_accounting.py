"""Versioned observed usage snapshots; sample durability is owned by the generator."""

from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from deepfabric.exceptions import DataSetGeneratorError

from .token_usage import _FIELDS


def _now():
    return datetime.now(timezone.utc).isoformat()


class CheckpointAccounting:
    """Replace a session snapshot, never add repeated snapshots of that session."""

    def __init__(self, tracker, previous=None, *, legacy=False):
        if previous is not None:
            validate_accounting(previous)
        self.tracker = tracker
        self.baseline = getattr(tracker, "_checkpoint_usage_baseline", None)
        self.state = (
            deepcopy(previous)
            if previous
            else {
                "version": 1,
                "generation_id": str(uuid4()),
                "prior_usage_unknown": legacy,
                "invocations": {},
            }
        )
        self.invocation_id = str(uuid4())
        self.state["current_invocation_id"] = self.invocation_id
        self.state["invocations"][self.invocation_id] = {
            "started_at": _now(),
            "status": "running",
            "sequence": 0,
        }
        self.refresh()

    def refresh(self, status=None):
        session = self.state["invocations"][self.invocation_id]
        snapshot = self.tracker.snapshot(include_accounting=False, include_records=False)
        if self.baseline is not None:
            snapshot = _difference(snapshot, self.baseline)
        session.update(
            usage={key: value for key, value in snapshot.items() if key != "records"},
            sequence=session["sequence"] + 1,
            updated_at=_now(),
        )
        if status is not None:
            session["status"] = status

    def snapshot(self):
        return deepcopy(self.state)

    def cumulative(self):
        sessions = list(self.state["invocations"].values())
        usages = [session["usage"] for session in sessions]
        groups = {}
        for usage in usages:
            for group in usage["by_model"]:
                groups.setdefault((group["provider"], group["model"]), []).append(group)
        result = _combine(usages)
        result["by_model"] = [
            {"provider": provider, "model": model, **_combine(rows)}
            for (provider, model), rows in groups.items()
        ]
        result["persistence_failures"] = [
            identity
            for identity, session in self.state["invocations"].items()
            if session.get("persistence_failed", False)
        ]
        result["prior_usage_unknown"] = self.state["prior_usage_unknown"]
        result["unfinished_invocations"] = [
            identity
            for identity, session in self.state["invocations"].items()
            if session["status"] != "completed"
            and (identity != self.invocation_id or session["status"] == "unfinished")
        ]
        result["complete"] = (
            result["complete"]
            and not result["prior_usage_unknown"]
            and not result["unfinished_invocations"]
            and not result["persistence_failures"]
        )
        return result


def _combine(usages):
    coverage = {field: sum(u["field_coverage"][field] for u in usages) for field in _FIELDS}
    requests = sum(u["requests"] for u in usages)
    return {
        "requests": requests,
        "requests_with_usage": sum(u["requests_with_usage"] for u in usages),
        "requests_without_usage": sum(u["requests_without_usage"] for u in usages),
        "field_coverage": coverage,
        "tokens": {
            field: sum(u["tokens"][field] or 0 for u in usages) if coverage[field] else None
            for field in _FIELDS
        },
        "complete": bool(requests)
        and all(
            coverage[field] == requests
            for field in ("input_tokens", "output_tokens", "total_tokens")
        ),
    }


def _difference(current, previous):
    result = _subtract_summary(current, previous)
    earlier = {(g["provider"], g["model"]): g for g in previous["by_model"]}
    result["by_model"] = []
    for group in current["by_model"]:
        identity = (group["provider"], group["model"])
        delta = _subtract_summary(group, earlier[identity]) if identity in earlier else group
        if delta["requests"]:
            result["by_model"].append({"provider": identity[0], "model": identity[1], **delta})
    return result


def _subtract_summary(current, previous):
    result = deepcopy(current)
    for key in ("requests", "requests_with_usage", "requests_without_usage"):
        result[key] -= previous[key]
    for field in _FIELDS:
        coverage = current["field_coverage"][field] - previous["field_coverage"][field]
        result["field_coverage"][field] = coverage
        result["tokens"][field] = (
            (current["tokens"][field] or 0) - (previous["tokens"][field] or 0) if coverage else None
        )
    result["complete"] = bool(result["requests"]) and all(
        result["field_coverage"][field] == result["requests"]
        for field in ("input_tokens", "output_tokens", "total_tokens")
    )
    return result


def validate_accounting(value):
    """Reject malformed optional history before resuming or rewriting its checkpoint."""
    try:
        if (
            not isinstance(value, dict)
            or value["version"] != 1
            or not isinstance(value["generation_id"], str)
            or not value["generation_id"]
            or type(value["prior_usage_unknown"]) is not bool
            or not isinstance(value["invocations"], dict)
        ):
            raise ValueError
        if "current_invocation_id" in value and (
            not isinstance(value["current_invocation_id"], str)
            or value["current_invocation_id"] not in value["invocations"]
        ):
            raise ValueError
        for identity, session in value["invocations"].items():
            if (
                not isinstance(identity, str)
                or not identity
                or not isinstance(session, dict)
                or session["status"] not in {"running", "completed", "unfinished"}
                or type(session["sequence"]) is not int
                or session["sequence"] < 1
                or not isinstance(session["started_at"], str)
                or not isinstance(session["updated_at"], str)
            ):
                raise ValueError
            usage = session["usage"]
            _validate_summary(usage)
            if not isinstance(usage["by_model"], list):
                raise TypeError
            identities = set()
            for group in usage["by_model"]:
                _validate_summary(group)
                key = (group["provider"], group["model"])
                if not all(isinstance(part, str) for part in key) or key in identities:
                    raise ValueError
                identities.add(key)
            combined = _combine(usage["by_model"])
            if any(combined[key] != usage[key] for key in combined):
                raise ValueError
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise DataSetGeneratorError(
            "Invalid checkpoint token_accounting metadata; restore a valid checkpoint copy "
            "or use a separate checkpoint location."
        ) from error


def _validate_summary(usage):
    if not isinstance(usage, dict) or type(usage["complete"]) is not bool:
        raise ValueError
    for key in ("requests", "requests_with_usage", "requests_without_usage"):
        if type(usage[key]) is not int or usage[key] < 0:
            raise ValueError
    if usage["requests_with_usage"] + usage["requests_without_usage"] != usage["requests"]:
        raise ValueError
    for field in _FIELDS:
        count, coverage = usage["tokens"][field], usage["field_coverage"][field]
        if (
            type(coverage) is not int
            or not 0 <= coverage <= usage["requests"]
            or (count is not None and (type(count) is not int or count < 0))
            or (count is None) != (coverage == 0)
        ):
            raise ValueError
    expected = bool(usage["requests"]) and all(
        usage["field_coverage"][field] == usage["requests"]
        for field in ("input_tokens", "output_tokens", "total_tokens")
    )
    if usage["complete"] != expected:
        raise ValueError
