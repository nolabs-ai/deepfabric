"""Unit tests for bulk custom seeds."""
# ruff: noqa: PLR2004, ARG001

from copy import deepcopy

import pytest

from jsonschema import validate

from deepfabric.bulk import (
    completion_schema,
    make_collection_model,
    make_completion_model,
    seed_value_schema,
)
from tests.support.bulk_custom_seeds import SCHEMA


@pytest.mark.parametrize("reference", ["#/$defs/Value", "#/properties/template"])
def test_seed_schema_preserves_original_root_references(reference):
    schema = deepcopy(SCHEMA)
    schema["$defs"] = {"Value": {"type": "string", "enum": ["A", "B"]}}
    schema["properties"]["template"] = {"type": "string", "enum": ["A", "B"]}
    schema["properties"]["name"] = {"$ref": reference}
    original = deepcopy(schema)
    extracted = seed_value_schema(schema, "name")
    validate("A", extracted)
    with pytest.raises(Exception, match="not one of"):
        validate("Invalid", extracted)
    validate({"items": ["A"]}, make_collection_model(extracted).model_json_schema())
    assert schema == original


def test_completion_schema_copy_and_parser_preserve_null_and_reject_arrays():
    schema = deepcopy(SCHEMA)
    original = deepcopy(schema)
    narrow = completion_schema(schema, "name")
    assert schema == original
    assert "name" not in narrow["properties"] and narrow["required"] == ["price"]
    model = make_completion_model(narrow)
    assert model.model_validate_json('{"price": null}').model_dump(exclude_none=True) == {
        "price": None
    }
    with pytest.raises(ValueError, match="validation error"):
        model.model_validate_json('[["price", 1]]')
