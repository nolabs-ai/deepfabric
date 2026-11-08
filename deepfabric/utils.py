import ast
import asyncio
import hashlib
import importlib
import json
import os
import re
import sys

from pathlib import Path
from typing import Any

from .constants import (
    CHECKPOINT_FAILURES_SUFFIX,
    CHECKPOINT_METADATA_SUFFIX,
    CHECKPOINT_SAMPLES_SUFFIX,
)

VALIDATION_ERROR_INDICATORS = [
    "validation error",
    "value error",
    "is null",
    "is empty string",
    "must provide actual value",
    "invalid schema",
    "pydantic",
    "string should have at least",
    "field required",
]


def is_validation_error(error: Exception) -> bool:
    """Check if an error is a validation/schema error that can be retried."""
    error_str = str(error).lower()
    return any(indicator in error_str for indicator in VALIDATION_ERROR_INDICATORS)


def ensure_not_running_loop(method_name: str) -> None:
    """Raise when invoked inside an active asyncio event loop."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    if loop.is_running():
        msg = (
            f"{method_name} cannot be called while an event loop is running. "
            "Use the async variant instead."
        )
        raise RuntimeError(msg)


def extract_list(input_string: str):
    """
    Extracts a Python list from a given input string.

    This function attempts to parse the input string as JSON. If that fails,
    it searches for the first Python list within the string by identifying
    the opening and closing brackets. If a list is found, it is evaluated
    safely to ensure it is a valid Python list.

    Args:
        input_string (str): The input string potentially containing a Python list.

    Returns:
        list: The extracted Python list if found and valid, otherwise an empty list.

    Raises:
        None: This function handles its own exceptions and does not raise any.
    """
    try:
        return json.loads(input_string)
    except json.JSONDecodeError:
        print("Failed to parse the input string as JSON.")

    start = input_string.find("[")
    if start == -1:
        print("No Python list found in the input string.")
        return []

    count = 0
    for i, char in enumerate(input_string[start:]):
        if char == "[":
            count += 1
        elif char == "]":
            count -= 1
        if count == 0:
            end = i + start + 1
            break
    else:
        print("No matching closing bracket found.")
        return []

    found_list_str = input_string[start:end]
    found_list = safe_literal_eval(found_list_str)
    if found_list is None:
        print("Failed to parse the list due to syntax issues.")
        return []

    return found_list


def remove_linebreaks_and_spaces(input_string):
    """
    Remove line breaks and extra spaces from the input string.

    This function replaces all whitespace characters (including line breaks)
    with a single space and then ensures that there are no consecutive spaces
    in the resulting string.

    Args:
        input_string (str): The string from which to remove line breaks and extra spaces.

    Returns:
        str: The processed string with line breaks and extra spaces removed.
    """
    no_linebreaks = re.sub(r"\s+", " ", input_string)
    return " ".join(no_linebreaks.split())


def safe_literal_eval(list_string: str):
    """
    Safely evaluate a string containing a Python literal expression.

    This function attempts to evaluate a string containing a Python literal
    expression using `ast.literal_eval`. If a `SyntaxError` or `ValueError`
    occurs, it tries to sanitize the string by replacing problematic apostrophes
    with the actual right single quote character and attempts the evaluation again.

    Args:
        list_string (str): The string to be evaluated.

    Returns:
        The result of the evaluated string if successful, otherwise `None`.
    """
    try:
        return ast.literal_eval(list_string)
    except (SyntaxError, ValueError):
        # Replace problematic apostrophes with the actual right single quote character
        sanitized_string = re.sub(r"(\w)'(\w)", r"\1’\2", list_string)
        try:
            return ast.literal_eval(sanitized_string)
        except (SyntaxError, ValueError):
            print("Failed to parse the list due to syntax issues.")
            return None


def read_topic_tree_from_jsonl(file_path: str) -> list[dict]:
    """
    Read the topic tree from a JSONL file.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        list[dict]: The topic tree.
    """
    topic_tree = []
    with open(file_path) as file:
        for line in file:
            topic_tree.append(json.loads(line.strip()))

    return topic_tree


def parse_num_samples(value: int | str | None) -> int | str | None:
    """Parse and validate num_samples: integer, 'auto', or percentage like '50%'.

    This is a shared utility used by both CLI argument parsing and config validation.

    Args:
        value: Raw value - can be int, string, or None

    Returns:
        Parsed value: int, "auto", percentage string like "50%", or None

    Raises:
        ValueError: If the value is invalid
    """
    if value is None:
        return None
    if isinstance(value, int):
        if value < 1:
            raise ValueError("num_samples must be at least 1")
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        if normalized.endswith("%"):
            try:
                pct = float(normalized[:-1])
            except ValueError as e:
                raise ValueError(f"Invalid percentage format: {value}") from e
            if pct <= 0:
                raise ValueError("Percentage must be greater than 0")
            return normalized
        # Try to parse as integer string
        try:
            parsed = int(normalized)
        except ValueError as e:
            raise ValueError(
                f"Invalid num_samples value: {value}. Use integer, 'auto', or percentage like '50%'"
            ) from e
        if parsed < 1:
            raise ValueError("num_samples must be at least 1")
        return parsed
    raise ValueError(f"num_samples must be int or string, got {type(value).__name__}")


def get_bool_env(key: str, default: bool = False) -> bool:
    """Get a boolean environment variable.

    Supports: '1', 'true', 'yes', 'on' (case-insensitive) as True.
    Everything else is False unless default is True and key is missing.
    """
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def import_optional_dependency(
    module_name: str,
    extra: str | None = None,
) -> Any:
    """
    Import an optional dependency at runtime.

    Args:
        module_name (str): The name of the module to import.
        extra (str | None): The optional dependency group providing this module.

    Returns:
        Any: The imported module.

    Raises:
        ModuleNotFoundError: If the module is not installed.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        if extra:
            msg = (
                f"The '{module_name}' library is required for the '{extra}' features. "
                f"Please install it using: pip install 'deepfabric[{extra}]'"
            )
        else:
            msg = f"The '{module_name}' library is required but is not installed."
        raise ModuleNotFoundError(msg) from None


def check_path_writable(path: str, path_description: str) -> tuple[bool, str | None]:
    """Check if a path is writable.

    Checks whether the specified file path can be written to by verifying:
    1. If the file exists, whether it's writable
    2. If the file doesn't exist, whether the parent directory exists and is writable

    Args:
        path: The file path to check
        path_description: Human-readable description for error messages

    Returns:
        Tuple of (is_writable, error_message). error_message is None if writable.
    """
    file_path = Path(path)
    parent_dir = file_path.parent
    error_msg: str | None = None

    # If the file exists, check if it's writable
    if file_path.exists():
        if not os.access(file_path, os.W_OK):
            error_msg = f"{path_description} exists but is not writable: {path}"
    elif not parent_dir.exists():
        # File doesn't exist and parent doesn't exist
        # Walk up to find the first existing ancestor
        ancestor = parent_dir
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent

        if not ancestor.exists():
            error_msg = (
                f"{path_description} parent directory does not exist "
                f"and cannot be created: {parent_dir}"
            )
        elif not os.access(ancestor, os.W_OK):
            error_msg = (
                f"{path_description} cannot create parent directory "
                f"(no write access to {ancestor}): {parent_dir}"
            )
    elif not os.access(parent_dir, os.W_OK):
        # Parent exists but is not writable
        error_msg = f"{path_description} parent directory is not writable: {parent_dir}"

    return (error_msg is None, error_msg)


def check_dir_writable(path: str, path_description: str) -> tuple[bool, str | None]:
    """Check if a directory path is writable.

    Checks whether files can be created in the specified directory by verifying:
    1. If the directory exists, whether it's writable
    2. If the directory doesn't exist, whether we can create it

    Args:
        path: The directory path to check
        path_description: Human-readable description for error messages

    Returns:
        Tuple of (is_writable, error_message). error_message is None if writable.
    """
    dir_path = Path(path)

    # If the directory exists, check if it's writable
    if dir_path.exists():
        if not dir_path.is_dir():
            return False, f"{path_description} exists but is not a directory: {path}"
        if not os.access(dir_path, os.W_OK):
            return False, f"{path_description} directory is not writable: {path}"
        return True, None

    # Directory doesn't exist - check if we can create it
    ancestor = dir_path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent

    if not ancestor.exists():
        return False, f"{path_description} cannot be created (root does not exist): {path}"

    if not os.access(ancestor, os.W_OK):
        return (
            False,
            f"{path_description} cannot be created (no write access to {ancestor}): {path}",
        )

    return True, None


# Checkpoint directory resolution
APP_NAME = "deepfabric"


def _get_deepfabric_data_dir() -> Path:
    """Get the DeepFabric data directory using platformdirs or fallback."""
    try:
        from platformdirs import user_data_dir  # noqa: PLC0415

        return Path(user_data_dir(APP_NAME))
    except ImportError:
        # Fallback if platformdirs not available
        if os.name == "nt":
            # Windows: APPDATA
            base = os.environ.get("APPDATA") or os.path.expanduser(r"~\AppData\Roaming")
        elif sys.platform == "darwin":
            # macOS: ~/Library/Application Support
            base = os.path.expanduser("~/Library/Application Support")
        else:
            # Linux and other Unix: XDG_DATA_HOME
            base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        return Path(base) / APP_NAME


def get_checkpoint_dir(config_path: str | None = None) -> str:
    """
    Get the checkpoint directory for a given config file.

    Uses ~/.deepfabric/checkpoints/{hash}/ where hash is derived from
    the absolute path of the config file. This ensures:
    - Consistent location regardless of current working directory
    - No conflicts between different projects with same output filename

    Args:
        config_path: Path to the config file. If None, uses a default subdirectory.

    Returns:
        Path to the checkpoint directory (not created, just resolved)
    """
    base_dir = _get_deepfabric_data_dir() / "checkpoints"

    if config_path is None:
        # No config file - use a "default" subdirectory
        return str(base_dir / "default")

    # Create a short hash from the absolute path of the config file
    abs_path = str(Path(config_path).resolve())
    path_hash = hashlib.sha256(abs_path.encode()).hexdigest()[:12]

    return str(base_dir / path_hash)


def build_checkpoint_file_paths(
    output_save_as: str,
    checkpoint_dir: str | None = None,
    *,
    config_path: str | None = None,
) -> tuple[Path, Path, Path]:
    """Build metadata, samples, and failures paths without creating directories.

    The caller selects the effective directory (for the CLI, its option overrides
    the configured directory). A nonempty checkpoint_dir is used even if it does
    not exist. Only None or an empty string selects get_checkpoint_dir(config_path).
    This function does not check existence or read checkpoint contents.

    Args:
        output_save_as: Configured output filename, which determines the checkpoint stem.
        checkpoint_dir: Effective directory after caller-specific option precedence.
        config_path: Source path for automatic directory hashing; None uses the default.
    """
    directory = Path(checkpoint_dir or get_checkpoint_dir(config_path))
    stem = Path(output_save_as).stem
    return (
        directory / f"{stem}{CHECKPOINT_METADATA_SUFFIX}",
        directory / f"{stem}{CHECKPOINT_SAMPLES_SUFFIX}",
        directory / f"{stem}{CHECKPOINT_FAILURES_SUFFIX}",
    )
