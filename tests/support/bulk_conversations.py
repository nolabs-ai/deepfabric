"""Shared fixtures and helpers for bulk conversations tests."""
# ruff: noqa: PLR2004, ARG001


def conversation(question="Question", answer="Answer", *, cot=False):
    result = {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }
    if cot:
        result["reasoning"] = {"style": "freetext", "content": "Reasoning with concrete steps"}
    return result
