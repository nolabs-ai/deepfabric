"""Shared fixtures and helpers for bulk custom seeds tests."""
# ruff: noqa: PLR2004, ARG001

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "price": {"type": "integer", "minimum": 0},
    },
    "required": ["name", "price"],
    "additionalProperties": False,
}
