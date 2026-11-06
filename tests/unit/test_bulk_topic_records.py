"""Unit tests for bulk topic records."""
# ruff: noqa: PLR2004, ARG001

from copy import deepcopy

import pytest

from jsonschema import validate

from deepfabric.bulk import make_collection_model


def test_collection_embedding_preserves_local_refs_and_original_schema():
    item_schema = {
        "type": "object",
        "$defs": {"Name": {"type": "string", "minLength": 2}},
        "properties": {"name": {"$ref": "#/$defs/Name"}},
        "required": ["name"],
    }
    original = deepcopy(item_schema)
    collection = make_collection_model(item_schema)
    validate({"items": [{"name": "Valid"}]}, collection.model_json_schema())
    assert item_schema == original
    with pytest.raises(ValueError, match="malformed JSON"):
        collection.model_validate_json('{"items": [{"name": "Valid"}')


def test_collection_embedding_preserves_literal_reference_data():
    literal = {"$ref": "#/$defs/Example"}
    schema = {
        "type": "object",
        "properties": {"literal": {"const": literal}},
        "required": ["literal"],
    }
    collection = make_collection_model(schema)
    validate({"items": [{"literal": literal}]}, collection.model_json_schema())
    assert (
        collection.model_json_schema()["properties"]["items"]["items"]["properties"]["literal"][
            "const"
        ]
        == literal
    )
