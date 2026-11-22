"""Shared fixtures and markers for integration tests."""

import logging
import os

import pytest

# Suppress httpx async client cleanup noise.
# LLMClient doesn't expose aclose(), so httpx connections linger after
# pytest-asyncio tears down the event loop, producing harmless
# "Event loop is closed" ERROR logs.
logging.getLogger("asyncio").addFilter(
    lambda record: "Event loop is closed" not in record.getMessage()
)

# Skip markers for conditional test execution based on API key availability
requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping OpenAI integration test",
)

requires_gemini = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping Gemini integration test",
)

requires_huggingface = pytest.mark.skipif(
    not os.getenv("HF_TOKEN"),
    reason="HF_TOKEN not set; skipping HuggingFace Hub integration test",
)


# Shared fixtures for provider configurations
@pytest.fixture
def openai_config():
    """Common OpenAI provider configuration for integration tests."""
    return {
        "provider": "openai",
        "model_name": os.getenv("OPENAI_TEST_MODEL", "gpt-4o-mini"),
        "temperature": 0.2,
    }


@pytest.fixture
def gemini_config():
    """Common Gemini provider configuration for integration tests."""
    return {
        "provider": "gemini",
        "model_name": os.getenv("GEMINI_TEST_MODEL", "gemini-3.8-flash"),
        "temperature": 0.2,
    }


def assert_generation_result(result, generator, min_samples=1, context=""):
    """Assert generation produced enough samples, with diagnostics on failure.

    Args:
        result: HuggingFace Dataset returned by create_data.
        generator: DataSetGenerator instance for reading failure diagnostics.
        min_samples: Minimum expected sample count.
        context: Test context for the failure message.
    """
    if len(result) >= min_samples:
        return

    lines = [
        f"Generation produced {len(result)} samples, expected >= {min_samples}.",
    ]
    if context:
        lines.append(f"Context: {context}")

    lines.append(f"Total failures: {len(generator.failed_samples)}")

    failure_counts = {k: len(v) for k, v in generator.failure_analysis.items() if v}
    if failure_counts:
        lines.append("Failure breakdown:")
        for ftype, count in sorted(failure_counts.items()):
            lines.append(f"  {ftype}: {count}")

    for i, f in enumerate(generator.failed_samples[:5], 1):
        error_msg = f.get("error", str(f)) if isinstance(f, dict) else str(f)
        lines.append(f"  [{i}] {error_msg[:300]}")

    pytest.fail("\n".join(lines))


def assert_topic_build_result(paths, model, min_paths=1, context=""):
    """Assert topic model build produced enough paths, with diagnostics on failure.

    Args:
        paths: List of paths from get_all_paths().
        model: Tree or Graph instance for reading failed_generations.
        min_paths: Minimum expected path count.
        context: Test context for the failure message.
    """
    if len(paths) >= min_paths:
        return

    lines = [
        f"Topic build produced {len(paths)} paths, expected >= {min_paths}.",
    ]
    if context:
        lines.append(f"Context: {context}")

    failed_gens = getattr(model, "failed_generations", [])
    if failed_gens:
        lines.append(f"Failed generations ({len(failed_gens)}):")
        for i, fg in enumerate(failed_gens[:5], 1):
            lines.append(f"  [{i}] {fg}")

    pytest.fail("\n".join(lines))
