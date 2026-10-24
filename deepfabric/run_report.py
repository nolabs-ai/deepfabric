"""Opt-in run metrics, kept separate from training data and prompt tracing."""

import hashlib
import json
import os

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from time import perf_counter
from uuid import uuid4

_active: ContextVar = ContextVar("deepfabric_run_report", default=None)
_context: ContextVar[dict | None] = ContextVar("deepfabric_metric_context", default=None)


def file_reference(path):
    """Identify an input without copying its contents or environment secrets."""
    if path is None:
        return None
    source = Path(path)
    return {
        "path": str(source.resolve()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


class RunReport:
    """Collect invocation-local events across async tasks and worker threads."""

    def __init__(self):
        self.run_id = str(uuid4())
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started = perf_counter()
        self.metadata = {}
        self.events = []
        self._lock = Lock()

    def record(self, kind, **fields):
        with self._lock:
            self.events.append(
                {
                    **(_context.get() or {}),
                    **fields,
                    "event": kind,
                    "elapsed_seconds": perf_counter() - self.started,
                }
            )

    def snapshot(self, usage, status):
        with self._lock:
            events = [dict(event) for event in self.events]
        try:
            package_version = version("deepfabric")
        except PackageNotFoundError:
            package_version = None
        # Distinguish edited checkouts even when the installed version is unchanged.
        source_hash = hashlib.sha256()
        root = Path(__file__).parent
        for source in sorted(root.rglob("*.py")):
            source_hash.update(str(source.relative_to(root)).encode())
            source_hash.update(b"\0")
            source_hash.update(source.read_bytes())
        plans = [e for e in events if e["event"] == "generation_plan"]
        paths = {a["uuid"]: a["path"] for e in plans for a in e["plan"].get("allocations", [])}
        outcomes = [e for e in events if e["event"] == "sample_outcome"]
        for outcome in outcomes:
            outcome["topic_path"] = paths.get(outcome["topic_id"])
        if any(e["event"] == "generation_result" and e.get("stopped") for e in events):
            status = "stopped"
        responses = {e.get("request_id"): e for e in events if e["event"] == "response"}
        requests = [e for e in events if e["event"] == "request"]
        for request in requests:
            response = responses.get(request.get("request_id"), {})
            request.update(
                finish_reason=response.get("finish_reason"), truncated=response.get("truncated")
            )
        retries = [e for e in events if e["event"] == "retry"]
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": perf_counter() - self.started,
            "status": status,
            "scope": "Current invocation; resumed samples are identified separately from new outcomes.",
            "deepfabric_version": package_version,
            "source_sha256": source_hash.hexdigest(),
            **self.metadata,
            "output_row_indexing": "One-based rows in the generated dataset; a row is only exported if dataset saving succeeds.",
            "summary": {
                "requests": len(requests),
                "retries": len(retries),
                "retries_by_category": {
                    category: sum(e["category"] == category for e in retries)
                    for category in sorted({e["category"] for e in retries})
                },
                "truncated_requests": sum(e.get("truncated") is True for e in requests),
                "requests_without_finish_reason": sum(
                    e.get("finish_reason") is None for e in requests
                ),
                "accepted_samples": sum(e["status"] == "accepted" for e in outcomes),
                "failed_samples": sum(e["status"] == "failed" for e in outcomes),
            },
            "token_usage": usage,
            "events": events,
            "limitations": [
                "Provider SDK-internal retries may be unobservable.",
                "Missing token usage and finish reasons remain unknown; no cost is estimated.",
                "Request duration includes SDK work; retry backoff is recorded separately.",
                "Direct streaming accounting is not covered by this report.",
                "Prompts are recorded only through --trace-prompt, separately from run metrics.",
            ],
        }

    def prepare_path(self, path):
        """Check the requested destination before starting paid generation."""
        target = Path(path)
        if target.is_dir():
            raise OSError("Run report path is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent):
            pass

    def write(self, path, usage, status):
        """Atomically replace the explicitly requested report."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=target.parent, delete=False
            ) as stream:
                temporary = stream.name
                json.dump(self.snapshot(usage, status), stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            os.replace(temporary, target)
        finally:
            if temporary and Path(temporary).exists():
                Path(temporary).unlink()


@contextmanager
def reporting(report):
    token = _active.set(report)
    try:
        yield report
    finally:
        _active.reset(token)


def metric_fields():
    """Return the current request/slot identity for joining token records."""
    return dict(_context.get() or {}) if _active.get() is not None else {}


def record_event(kind, **fields):
    report = _active.get()
    if report is not None:
        report.record(kind, **fields)


@contextmanager
def metric_context(**fields):
    token = _context.set({**(_context.get() or {}), **fields})
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def request_metrics(provider, model, schema, parameters):
    """Measure one observable model attempt, including failed attempts."""
    if _active.get() is None:
        yield
        return
    started = perf_counter()
    request_id = str(uuid4())
    error = None
    # Only sampling settings are retained, never arbitrary SDK kwargs or prompts.
    settings = {
        k: v
        for k, v in parameters.items()
        if k in ("temperature", "max_tokens", "max_output_tokens", "top_p", "seed")
    }
    with metric_context(request_id=request_id):
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            record_event(
                "request",
                provider=provider,
                model=model,
                stage=getattr(schema, "__name__", None),
                duration_seconds=perf_counter() - started,
                parameters=settings,
                error_type=error,
            )


def response_metrics(provider, model, finish_reason):
    """Record the provider's termination reason, without interpreting response text."""
    reason = getattr(finish_reason, "value", finish_reason)
    reason = str(reason) if reason is not None else None
    record_event(
        "response",
        provider=provider,
        model=model,
        finish_reason=reason,
        truncated=reason.lower() in ("length", "max_tokens", "max_output_tokens")
        if reason
        else None,
    )


def generation_scope(function):
    """Carry allocation/slot identity into nested requests without altering scheduling."""
    sig = signature(function)

    @wraps(function)
    async def wrapped(*args, **kwargs):
        if _active.get() is None:
            return await function(*args, **kwargs)
        bound = sig.bind(*args, **kwargs).arguments
        fields = {"operation": function.__name__.removeprefix("_")}
        allocation = bound.get("allocation")
        if allocation:
            fields.update(topic_id=allocation["uuid"], topic_path=allocation["path"])
        if "ordinal" in bound:
            fields["sample_ordinal"] = bound["ordinal"]
        if "identity" in bound:
            fields.update(topic_id=bound["identity"][0], sample_ordinal=bound["identity"][1])
            topic_path = bound.get("topic_path")
            fields["topic_path"] = topic_path.path if topic_path else None
        with metric_context(**fields):
            return await function(*args, **kwargs)

    return wrapped
