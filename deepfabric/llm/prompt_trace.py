"""Run-scoped, opt-in prompt tracing shared by all LLM clients."""

import json

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, TextIO

from ..output_safety import open_unique


class PromptPreviewCaptured(BaseException):
    """Stop a preview at the request boundary without entering retry/error handling."""


@dataclass
class PromptTrace:
    """Trace the first request for each schema, or every request in verbose mode."""

    path: Path | None = None
    stream: TextIO | None = field(default=None, repr=False)
    verbose: bool = False
    dry_run: bool = False
    seen: set[str] = field(default_factory=set)
    lock: Any = field(default_factory=Lock)

    def record(self, provider: str, model: str, prompt: Any, schema: Any) -> None:
        definition = (
            schema.model_json_schema()
            if callable(getattr(schema, "model_json_schema", None))
            else schema
        )
        name = getattr(schema, "__name__", None)
        key = json.dumps([provider, model, name, definition], sort_keys=True)
        record = {
            "provider": provider,
            "model": model,
            "schema_name": name,
            "schema": definition,
            "prompt": prompt,
            "dry_run": self.dry_run,
        }
        with self.lock:
            if not self.verbose and key in self.seen:
                return
            if self.path is not None:
                if self.stream is None:
                    self.path, self.stream = open_unique(self.path)
                    print(f"Prompt trace saved to: {self.path}")
                self.stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.stream.flush()
            if self.dry_run:
                # Preview runs never start a live TUI. Print literal text without markup.
                print(f"\n--- {name or 'custom schema'} ({provider}/{model}) ---")
                print(prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False))
            self.seen.add(key)


_active_trace: ContextVar[PromptTrace | None] = ContextVar("prompt_trace", default=None)


def get_prompt_trace() -> PromptTrace | None:
    """Return the trace session active for this run/task (also inherited by workers)."""
    return _active_trace.get()


@contextmanager
def prompt_tracing(
    *, enabled: bool = False, path: str | None = None, verbose: bool = False, dry_run: bool = False
):
    """Explicitly enable tracing with a CLI path or the per-invocation default."""
    destination = Path(path or "./prompt-trace.jsonl") if enabled else None
    session = (
        PromptTrace(path=destination, verbose=verbose, dry_run=dry_run)
        if enabled or dry_run
        else None
    )
    token = _active_trace.set(session)
    try:
        yield session
    finally:
        try:
            if session is not None and session.stream is not None:
                session.stream.close()
        finally:
            _active_trace.reset(token)
