"""Component tests for rate limiting."""
# ruff: noqa: PLR2004, PLC0415

import asyncio

from unittest.mock import Mock, patch

import openai


class TestRetryIntegration:
    """Component tests of provider clients and retry handling with mocked I/O."""

    def test_retry_succeeds_after_transient_failure(self):
        """Test that retry succeeds after a transient failure."""
        from deepfabric.llm.client import LLMClient

        # Mock the model to fail once then succeed
        attempt_count = [0]

        def mock_model_call(prompt, schema, **kwargs):  # noqa: ARG001
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                # First attempt: rate limit error
                raise openai.RateLimitError(
                    "Rate limit exceeded",
                    response=Mock(headers={"retry-after": "0.1"}),
                    body=None,
                )
            # Second attempt: success
            return '{"test": "value"}'

        # Create client with fast retry config
        config = {"max_retries": 2, "base_delay": 0.1, "max_delay": 1.0}

        with patch("deepfabric.llm.client.make_outlines_model") as mock_make_model:
            mock_model = Mock(side_effect=mock_model_call)
            mock_make_model.return_value = mock_model

            with patch("deepfabric.llm.client.make_async_outlines_model") as mock_make_async:
                mock_make_async.return_value = None  # No async model needed for this test

                client = LLMClient("openai", "gpt-4", rate_limit_config=config)

                # Mock schema validation
                from pydantic import BaseModel

                class TestSchema(BaseModel):
                    test: str

                # Call should succeed on second attempt
                with patch.object(
                    TestSchema, "model_validate_json", return_value=TestSchema(test="value")
                ):
                    result = client.generate("test prompt", TestSchema)
                    assert result.test == "value"
                    assert attempt_count[0] == 2  # Failed once, succeeded on retry

    def test_async_retry_succeeds_after_transient_failure(self):
        """Test that async retry succeeds after a transient failure."""
        from deepfabric.llm.client import LLMClient

        # Mock the async model to fail once then succeed
        attempt_count = [0]

        async def mock_async_model_call(prompt, schema, **kwargs):  # noqa: ARG001
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                # First attempt: rate limit error
                raise openai.RateLimitError(
                    "Rate limit exceeded",
                    response=Mock(headers={"retry-after": "0.1"}),
                    body=None,
                )
            # Second attempt: success
            await asyncio.sleep(0.01)
            return '{"test": "value"}'

        async def run_test():
            # Create client with fast retry config
            config = {"max_retries": 2, "base_delay": 0.1, "max_delay": 1.0}

            with patch("deepfabric.llm.client.make_async_outlines_model") as mock_make_async:
                mock_async_model = Mock(side_effect=mock_async_model_call)
                mock_make_async.return_value = mock_async_model

                with patch("deepfabric.llm.client.make_outlines_model") as mock_make_model:
                    mock_make_model.return_value = Mock()
                    client = LLMClient("openai", "gpt-4", rate_limit_config=config)

                    # Mock schema validation
                    from pydantic import BaseModel

                    class TestSchema(BaseModel):
                        test: str

                    # Call should succeed on second attempt
                    with patch.object(
                        TestSchema, "model_validate_json", return_value=TestSchema(test="value")
                    ):
                        result = await client.generate_async("test prompt", TestSchema)
                        assert result.test == "value"
                        assert attempt_count[0] == 2  # Failed once, succeeded on retry

        # Run the async test
        asyncio.run(run_test())
