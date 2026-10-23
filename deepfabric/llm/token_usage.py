"""Provider-reported token accounting, separate from generation results."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Any

_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "reasoning_tokens",
)


def _mapping(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    if callable(getattr(value, "model_dump", None)):
        result = value.model_dump(mode="json", exclude_none=True)
        return result if isinstance(result, dict) else {}
    return {}


def _count(data: dict, key: str) -> int | None:
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def normalize_usage(provider: str, usage: Any) -> dict:
    """Normalize accounting fields while retaining their original provider values."""
    raw = _mapping(usage)
    tokens = dict.fromkeys(_FIELDS)
    if provider == "gemini":
        prompt = _count(raw, "prompt_token_count")
        tokens["input_tokens"] = (
            prompt + (_count(raw, "tool_use_prompt_token_count") or 0)
            if prompt is not None
            else None
        )
        candidates = _count(raw, "candidates_token_count")
        thoughts = _count(raw, "thoughts_token_count")
        tokens["output_tokens"] = candidates + (thoughts or 0) if candidates is not None else None
        tokens["total_tokens"] = _count(raw, "total_token_count")
        tokens["cached_input_tokens"] = _count(raw, "cached_content_token_count")
        tokens["reasoning_tokens"] = thoughts
    elif provider == "anthropic":
        regular = _count(raw, "input_tokens")
        cached = _count(raw, "cache_read_input_tokens")
        created = _count(raw, "cache_creation_input_tokens")
        tokens["input_tokens"] = (
            regular + (cached or 0) + (created or 0) if regular is not None else None
        )
        tokens["output_tokens"] = _count(raw, "output_tokens")
        tokens["cached_input_tokens"] = cached
        tokens["cache_creation_input_tokens"] = created
    else:
        tokens["input_tokens"] = _count(raw, "prompt_tokens")
        tokens["output_tokens"] = _count(raw, "completion_tokens")
        tokens["total_tokens"] = _count(raw, "total_tokens")
        tokens["cached_input_tokens"] = _count(
            _mapping(raw.get("prompt_tokens_details")), "cached_tokens"
        )
        tokens["reasoning_tokens"] = _count(
            _mapping(raw.get("completion_tokens_details")), "reasoning_tokens"
        )
    if tokens["total_tokens"] is None and all(
        tokens[k] is not None for k in ("input_tokens", "output_tokens")
    ):
        tokens["total_tokens"] = tokens["input_tokens"] + tokens["output_tokens"]
    return {"tokens": tokens, "raw_usage": raw}


class TokenUsageTracker:
    """Thread-safe observed-call ledger; missing counts remain unknown."""

    def __init__(self):
        self._records: list[dict] = []
        self._lock = Lock()

    def record(self, record: dict) -> None:
        with self._lock:
            self._records.append(record)

    def snapshot(self) -> dict:
        """Return totals and provider/model groups plus each observed response."""
        with self._lock:
            records = list(self._records)
        groups: dict[tuple[str, str], list[dict]] = {}
        for record in records:
            groups.setdefault((record["provider"], record["model"]), []).append(record)
        return {
            **_summarize(records),
            "by_model": [
                {"provider": provider, "model": model, **_summarize(rows)}
                for (provider, model), rows in groups.items()
            ],
            "records": records,
        }


def _summarize(records: list[dict]) -> dict:
    coverage = {
        field: sum(row["tokens"][field] is not None for row in records) for field in _FIELDS
    }
    return {
        "requests": len(records),
        "requests_with_usage": sum(bool(row["raw_usage"]) for row in records),
        "requests_without_usage": sum(not row["raw_usage"] for row in records),
        "complete": bool(records)
        and all(
            coverage[k] == len(records) for k in ("input_tokens", "output_tokens", "total_tokens")
        ),
        "tokens": {
            field: sum(row["tokens"][field] or 0 for row in records) if coverage[field] else None
            for field in _FIELDS
        },
        "field_coverage": coverage,
    }


@dataclass
class _Request:
    provider: str
    model: str
    stage: str | None
    reported: bool = False


_trackers: ContextVar[tuple[TokenUsageTracker, ...]] = ContextVar(
    "token_usage_trackers", default=()
)
_request: ContextVar[_Request | None] = ContextVar("token_usage_request", default=None)


@contextmanager
def track_token_usage(tracker: TokenUsageTracker):
    """Accumulate this task's usage (including inherited async tasks/worker threads)."""
    existing = _trackers.get()
    token = _trackers.set(existing if tracker in existing else (*existing, tracker))
    try:
        yield tracker
    finally:
        _trackers.reset(token)


@contextmanager
def request_token_usage(tracker: TokenUsageTracker, provider: str, model: str, schema: Any):
    """Attribute provider responses to a client and stage without changing its return type."""
    request = _Request(provider, model, getattr(schema, "__name__", None))
    token = _request.set(request)
    with track_token_usage(tracker):
        try:
            yield
        finally:
            if not request.reported:
                record_usage(provider, model, None, response_received=False)
            _request.reset(token)


def record_usage(provider: str, model: str, usage: Any, *, response_received: bool = True) -> None:
    """Record once per provider API call, before response validation can fail."""
    request = _request.get()
    if request is not None:
        request.reported = True
    record = {
        "provider": provider,
        "model": model,
        "stage": request.stage if request is not None else None,
        "response_received": response_received,
        **normalize_usage(provider, usage),
    }
    for tracker in _trackers.get():
        tracker.record(record)
