"""Retry handler with intelligent backoff for LLM API calls."""

import asyncio
import logging
import random
import time

from collections.abc import Callable, Coroutine
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from deepfabric.run_report import record_event

from .rate_limit_config import BackoffStrategy, RateLimitConfig
from .rate_limit_detector import RateLimitDetector

if TYPE_CHECKING:
    from deepfabric.progress import ProgressReporter

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Max chars for error summaries emitted through progress reporter
_ERROR_SUMMARY_MAX_LENGTH = 200


class RetryHandler:
    """Intelligent retry handler for LLM API calls with provider-aware backoff."""

    def __init__(self, config: RateLimitConfig, provider: str):
        """Initialize retry handler.

        Args:
            config: Rate limit configuration
            provider: Provider name (openai, anthropic, gemini, ollama)
        """
        self.config = config
        self.provider = provider
        self.detector = RateLimitDetector()
        self.progress_reporter: ProgressReporter | None = None

    def should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry.

        Args:
            exception: The exception that occurred

        Returns:
            True if the error is retryable
        """
        # Check if it's a retryable error (rate limit, timeout, server error)
        if not self.detector.is_retryable_error(exception, self.provider):
            return False

        # If it's a rate limit error, check if we should fail fast
        if self.detector.is_rate_limit_error(exception, self.provider):
            quota_info = self.detector.extract_quota_info(exception, self.provider)

            # Don't retry if we should fail fast (e.g., daily quota exhausted)
            if self.detector.should_fail_fast(quota_info):
                logger.warning(
                    "Failing fast for %s: %s (quota_info: %s)",
                    self.provider,
                    exception,
                    quota_info,
                )
                return False

        return True

    def calculate_delay(
        self,
        attempt: int,
        exception: Exception | None = None,
    ) -> float:
        """Calculate the delay before the next retry attempt.

        Args:
            attempt: Current attempt number (0-indexed)
            exception: Optional exception to extract retry-after from

        Returns:
            Delay in seconds
        """
        # Extract retry-after from exception if available
        retry_after = None
        if exception and self.config.respect_retry_after:
            quota_info = self.detector.extract_quota_info(exception, self.provider)
            retry_after = quota_info.retry_after

        # If provider specifies retry-after, use it (with max_delay cap)
        if retry_after is not None:
            delay = min(retry_after, self.config.max_delay)
            logger.debug("Using retry-after header: %.2fs (capped at %.2fs)", retry_after, delay)
            return delay

        # Otherwise calculate delay based on backoff strategy
        if self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.exponential_base**attempt)
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            base_delay = self.config.base_delay * (self.config.exponential_base**attempt)
            delay = base_delay
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        elif self.config.backoff_strategy == BackoffStrategy.CONSTANT:
            delay = self.config.base_delay
        else:
            # Default to exponential
            delay = self.config.base_delay * (self.config.exponential_base**attempt)

        # Apply jitter if enabled
        if self.config.jitter:
            # Add random jitter of ±25% to prevent thundering herd
            jitter_range = delay * 0.25
            delay = delay + random.uniform(-jitter_range, jitter_range)  # noqa: S311 # nosec

        # Ensure delay is within bounds and return
        return max(self.config.base_delay, min(delay, self.config.max_delay))

    def on_backoff_handler(self, details: dict[str, Any]) -> None:
        """Callback for backoff retry attempts.

        Args:
            details: Backoff details including attempt number, wait time, exception
        """
        exception = details.get("exception")
        wait = details.get("wait", 0)
        tries = details.get("tries", 0)

        record_event(
            "retry",
            category="api",
            provider=self.provider,
            attempt=tries,
            backoff_seconds=wait,
            error_type=type(exception).__name__ if exception else None,
        )

        # Extract quota information if it's a rate limit error
        quota_info_str = ""
        if exception and self.detector.is_rate_limit_error(exception, self.provider):
            quota_info = self.detector.extract_quota_info(exception, self.provider)
            if quota_info.quota_type:
                quota_info_str = f" (quota_type: {quota_info.quota_type})"

        if self.progress_reporter:
            error_summary = str(exception)
            if len(error_summary) > _ERROR_SUMMARY_MAX_LENGTH:
                error_summary = error_summary[:_ERROR_SUMMARY_MAX_LENGTH] + "..."
            self.progress_reporter.emit_llm_retry(
                provider=self.provider,
                attempt=tries,
                wait=wait,
                error_summary=error_summary,
                quota_type=quota_info_str.strip(" ()") if quota_info_str else "",
            )
        else:
            logger.warning(
                "Rate limit/transient error for %s on attempt %d, backing off %.2fs%s: %s",
                self.provider,
                tries,
                wait,
                quota_info_str,
                exception,
            )

    def on_giveup_handler(self, details: dict[str, Any]) -> None:
        """Callback when giving up after max retries.

        Args:
            details: Backoff details including final exception
        """
        exception = details.get("exception")
        tries = details.get("tries", 0)

        logger.error(
            "Giving up after %d attempts for %s: %s",
            tries,
            self.provider,
            exception,
        )


def retry_with_backoff(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to add retry logic with backoff to synchronous functions.

    This decorator is applied to LLMClient methods to handle rate limits
    and transient errors automatically.

    Args:
        func: Function to wrap with retry logic

    Returns:
        Wrapped function with retry capability
    """

    @wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any) -> T:
        # Get retry handler from self (assumes LLMClient instance)
        retry_handler = getattr(self, "retry_handler", None)
        if not retry_handler or not isinstance(retry_handler, RetryHandler):
            # No retry handler configured, call function directly
            return func(self, *args, **kwargs)

        config = retry_handler.config
        attempt = 0

        while attempt <= config.max_retries:
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                # Check if we should retry
                if not retry_handler.should_retry(e):
                    # Not retryable, raise immediately
                    raise

                # Check if we've exhausted retries
                if attempt >= config.max_retries:
                    retry_handler.on_giveup_handler({"exception": e, "tries": attempt + 1})
                    raise

                # Calculate delay and wait
                delay = retry_handler.calculate_delay(attempt, e)
                retry_handler.on_backoff_handler(
                    {
                        "exception": e,
                        "wait": delay,
                        "tries": attempt + 1,
                    }
                )

                time.sleep(delay)
                attempt += 1

        # Should never reach here, but for type safety
        msg = "Unexpected state in retry logic"
        raise RuntimeError(msg)

    return wrapper


def retry_with_backoff_async(
    func: Callable[..., Coroutine[Any, Any, T]],
) -> Callable[..., Coroutine[Any, Any, T]]:
    """Decorator to add retry logic with backoff to async functions.

    This decorator is applied to async LLMClient methods to handle rate limits
    and transient errors automatically.

    Args:
        func: Async function to wrap with retry logic

    Returns:
        Wrapped async function with retry capability
    """

    @wraps(func)
    async def wrapper(self, *args: Any, **kwargs: Any) -> T:
        # Get retry handler from self (assumes LLMClient instance)
        retry_handler = getattr(self, "retry_handler", None)
        if not retry_handler or not isinstance(retry_handler, RetryHandler):
            # No retry handler configured, call function directly
            return await func(self, *args, **kwargs)

        config = retry_handler.config
        attempt = 0

        while attempt <= config.max_retries:
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                # Check if we should retry
                if not retry_handler.should_retry(e):
                    # Not retryable, raise immediately
                    raise

                # Check if we've exhausted retries
                if attempt >= config.max_retries:
                    retry_handler.on_giveup_handler({"exception": e, "tries": attempt + 1})
                    raise

                # Calculate delay and wait
                delay = retry_handler.calculate_delay(attempt, e)
                retry_handler.on_backoff_handler(
                    {
                        "exception": e,
                        "wait": delay,
                        "tries": attempt + 1,
                    }
                )

                await asyncio.sleep(delay)
                attempt += 1

        # Should never reach here, but for type safety
        msg = "Unexpected state in retry logic"
        raise RuntimeError(msg)

    return wrapper
