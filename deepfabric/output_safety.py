"""Publication and recovery for generation outputs, without startup write probes."""

import errno
import os

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import TextIO

from .utils import _get_deepfabric_data_dir

_protected: ContextVar[list[Path] | None] = ContextVar("output_protected_paths", default=None)


@contextmanager
def output_session():
    """Track input and published paths within one CLI invocation."""
    token = _protected.set([])
    try:
        yield
    finally:
        _protected.reset(token)


def protect_paths(*paths):
    """Prevent later replacements from overwriting inputs or earlier outputs."""
    registry = _protected.get()
    if registry is not None:
        registry.extend(Path(path).expanduser().absolute() for path in paths if path)


def _check_replacement(path):
    for protected in _protected.get() or []:
        if path.resolve() == protected.resolve() or (
            path.exists() and protected.exists() and path.samefile(protected)
        ):
            raise OSError(
                errno.EINVAL, f"Replacement aliases an input or published output: {protected}"
            )


def clear_output(path: str | Path) -> None:
    """Delete an explicitly selected output now, preserving protected inputs."""
    target = Path(path).expanduser()
    _check_replacement(target)
    target.unlink(missing_ok=True)


class OutputSaveError(OSError):
    """Requested publication failed, possibly with a successful recovery export."""

    def __init__(self, path, error, recovery=None, recovery_error=None):
        self.path = str(path)
        self.original_error = error
        self.recovery = str(recovery) if recovery else None
        message = f"Could not save {path}: {error}."
        if recovery:
            message += f" Recovery saved to: {recovery}"
        else:
            message += f" Recovery also failed: {recovery_error}"
        super().__init__(message)


def candidates(path: Path):
    """Yield the original name, then increasing numeric filename suffixes."""
    yield path
    number = 1
    while True:
        yield path.with_name(f"{path.stem}_{number}{path.suffix}")
        number += 1


def open_unique(path: str | Path) -> tuple[Path, TextIO]:
    """Exclusively create a file; only collisions advance the numeric suffix."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    for candidate in candidates(target):
        try:
            stream = candidate.open("x", encoding="utf-8")
        except FileExistsError:
            continue
        else:
            protect_paths(candidate)
            return candidate, stream
    raise AssertionError("Unreachable")


def _publish(path: Path, writer: Callable[[str], None], replace: bool) -> str:
    if replace:
        _check_replacement(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".deepfabric-", dir=path.parent) as directory:
        staged = Path(directory) / path.name
        writer(str(staged))
        if replace:
            os.replace(staged, path)
            return str(path)
        # Keep path-based serializers unchanged, but copy through the handle
        # created exclusively. Never reopen a claimed destination with "w".
        for candidate in candidates(path):
            try:
                stream = candidate.open("xb")
            except FileExistsError:
                continue
            identity = None
            try:
                with stream:
                    identity = os.fstat(stream.fileno())
                    with staged.open("rb") as source:
                        copyfileobj(source, stream)
            except BaseException:
                if identity is not None:
                    _remove_owned_partial(candidate, identity)
                raise
            return str(candidate)
    raise AssertionError("Unreachable")


def _remove_owned_partial(path, identity):
    """Best-effort cleanup without deleting a detected replacement or symlink."""
    try:
        if os.path.samestat(path.lstat(), identity):
            path.unlink()
    except OSError:
        # Preserve the write error, including when cleanup itself is denied.
        pass


def save_output(path: str | Path, writer: Callable[[str], None], *, replace: bool = False) -> str:
    """Publish once; on filesystem failure try one application-data recovery save.

    Recovery success still raises OutputSaveError: it is not success at the
    requested destination. Serialization exceptions are not filesystem failures.
    """
    target = Path(path).expanduser()
    try:
        actual = _publish(target, writer, replace)
    except OSError as error:
        try:
            recovery_dir = _get_deepfabric_data_dir() / "recovery"
            recovery_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            recovery = _publish(recovery_dir / target.name, writer, False)
        except OSError as recovery_error:
            raise OutputSaveError(target, error, recovery_error=recovery_error) from error
        raise OutputSaveError(target, error, recovery=recovery) from error
    else:
        protect_paths(actual)
        return actual
