"""Structured collection responses with independent validation of each item."""

import json

from copy import deepcopy
from typing import Any

from jsonschema.validators import validator_for
from pydantic import BaseModel


def make_collection_model(item_schema: dict) -> type:
    """Keep a parseable collection intact so valid siblings can survive invalid items.

    The provider receives the full item schema. Local parsing validates the envelope
    only; the caller validates each item independently before assigning slot identities.
    Local JSON Pointer references are rebased when embedding the item schema.
    """
    embedded = deepcopy(item_schema)

    def rebase(value):
        if isinstance(value, dict):
            # An explicit resource ID keeps its own reference scope.
            if "$id" in value:
                return
            ref = value.get("$ref")
            if isinstance(ref, str) and (ref == "#" or ref.startswith("#/")):
                value["$ref"] = "#/properties/items/items" + ref[1:]
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
                            rebase(child)
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
                rebase(value.get(keyword))
        elif isinstance(value, list):
            for child in value:
                rebase(child)

    rebase(embedded)
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
