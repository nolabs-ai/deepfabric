"""Structured collection responses with independent validation of each item."""

import json

from copy import deepcopy
from typing import Any

from jsonschema.validators import validator_for
from pydantic import BaseModel

from .llm.client import make_dynamic_model


def _rebase_local_refs(value, prefix: str) -> bool:
    """Rebase schema-local JSON Pointers without changing literal const/enum data."""
    found = False
    if isinstance(value, dict):
        # An explicit resource ID keeps its own reference scope.
        if "$id" in value:
            return False
        ref = value.get("$ref")
        if isinstance(ref, str) and (ref == "#" or ref.startswith("#/")):
            value["$ref"] = prefix + ref[1:]
            found = True
        # Descend only through schema keywords, preserving literal const/enum data.
        for keyword in (
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependentSchemas",
            "dependencies",
        ):
            children = value.get(keyword, {})
            if isinstance(children, dict):
                for child in children.values():
                    if isinstance(child, dict):
                        found = _rebase_local_refs(child, prefix) or found
        for keyword in (
            "additionalProperties",
            "additionalItems",
            "items",
            "not",
            "if",
            "then",
            "else",
            "contains",
            "propertyNames",
            "unevaluatedProperties",
            "unevaluatedItems",
            "contentSchema",
            "allOf",
            "anyOf",
            "oneOf",
            "prefixItems",
        ):
            found = _rebase_local_refs(value.get(keyword), prefix) or found
    elif isinstance(value, list):
        for child in value:
            found = _rebase_local_refs(child, prefix) or found
    return found


def make_collection_model(item_schema: dict) -> type:
    """Keep a parseable collection intact so valid siblings can survive invalid items.

    The provider receives the full item schema. Local parsing validates the envelope
    only; the caller validates each item independently before assigning slot identities.
    Local JSON Pointer references are rebased when embedding the item schema.
    """
    embedded = deepcopy(item_schema)

    _rebase_local_refs(embedded, "#/properties/items/items")
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": embedded}},
        "required": ["items"],
        "additionalProperties": False,
    }

    class CollectionModel(BaseModel):
        items: list[Any]

        @classmethod
        def model_json_schema(cls, **_kwargs: Any) -> dict:
            return deepcopy(schema)

        @classmethod
        def model_validate_json(cls, json_data: str) -> "CollectionModel":
            try:
                data = json.loads(json_data)
            except (ValueError, TypeError) as exc:
                raise ValueError("Collection validation error: malformed JSON") from exc
            if not isinstance(data, dict) or not isinstance(data.get("items"), list):
                raise TypeError(
                    "Collection validation error: expected an object with an items list"
                )
            return cls(items=data["items"])

    return CollectionModel


def record_validator(schema: dict):
    """Check the configured schema once, then validate individual collection records."""
    validator_type = validator_for(schema)
    validator_type.check_schema(schema)
    return validator_type(schema)


def estimate_record_tokens(item_schema: dict | None) -> int:
    """Coarse warning heuristic, not a tokenizer or an automatic budget adjustment.

    Allow 1000 tokens per full conversation. For custom records, use the larger
    of 128 tokens and a quarter of the serialized schema's character count as a
    proxy for record width. Values, constraints, and provider tokenization vary.
    """
    if item_schema is None:
        return 1000
    return max(128, len(json.dumps(item_schema, ensure_ascii=False)) // 4)


def seed_value_schema(schema: dict, seed_field: str) -> dict:
    """Extract a field while preserving references to its original schema root."""
    field = deepcopy(schema["properties"][seed_field])
    if isinstance(field, bool):
        field = {} if field else {"not": {}}
    definitions = field.get("$defs", {})
    namespace = "__deepfabric_source"
    while namespace in definitions:
        namespace += "_"
    prefix = f"#/$defs/{namespace}"
    if _rebase_local_refs(field, prefix):
        source = deepcopy(schema)
        _rebase_local_refs(source, prefix)
        # Keep definitions off a $ref object: strict provider transforms remove
        # $ref siblings, which would otherwise discard the reference's target.
        field = {"anyOf": [field], "$defs": {namespace: source}}
    if "$schema" in schema:
        field.setdefault("$schema", schema["$schema"])
    return field


def completion_schema(schema: dict, seed_field: str) -> dict:
    """Derive the remainder without mutating the configured output schema."""
    result = deepcopy(schema)
    result["properties"].pop(seed_field)
    if "required" in result:
        result["required"] = [field for field in result["required"] if field != seed_field]
    return result


def make_completion_model(schema: dict) -> type:
    """Adapt make_dynamic_model to Outlines while keeping flat record serialization."""
    dynamic = make_dynamic_model(deepcopy(schema))

    class CompletionModel(BaseModel):
        record: dict[str, Any]

        @classmethod
        def model_json_schema(cls, **_kwargs: Any) -> dict:
            return deepcopy(dynamic.model_json_schema())

        @classmethod
        def model_validate_json(cls, json_data: str) -> "CompletionModel":
            try:
                if not isinstance(json.loads(json_data), dict):
                    raise TypeError("Expected a record object")
                result = dynamic.model_validate_json(json_data).model_dump()
            except (ValueError, TypeError) as exc:
                raise ValueError("Completion validation error: malformed record JSON") from exc
            if not isinstance(result, dict):
                raise TypeError("Completion validation error: expected a record object")
            return cls(record=result)

        def model_dump(self, **_kwargs: Any) -> dict:
            # A permitted null is data, including when the pinned seed itself is null.
            return deepcopy(self.record)

    return CompletionModel
