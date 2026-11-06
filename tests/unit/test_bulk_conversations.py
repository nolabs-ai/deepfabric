"""Unit tests for bulk conversations."""
# ruff: noqa: PLR2004, ARG001

from deepfabric.bulk import estimate_record_tokens
from deepfabric.schemas import Conversation
from tests.support.bulk_conversations import conversation


def test_token_estimate_accounts_for_wide_custom_schema():
    wide = {"type": "object", "properties": {f"field_{i}": {"type": "string"} for i in range(100)}}
    assert estimate_record_tokens(wide) > estimate_record_tokens({"type": "object"})


def test_embedded_conversation_schema_preserves_nested_references():
    from jsonschema import validate  # noqa: PLC0415

    from deepfabric.bulk import make_collection_model  # noqa: PLC0415

    model = make_collection_model(Conversation.model_json_schema())
    validate({"items": [conversation(cot=True)]}, model.model_json_schema())
