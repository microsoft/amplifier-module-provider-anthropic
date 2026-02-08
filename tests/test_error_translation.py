"""Tests for Phase 2 error translation: SDK errors → kernel error types.

Verifies that every Anthropic SDK error is translated to the correct
kernel error type with proper attributes (provider, status_code, retryable,
__cause__ preserved).
"""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest

from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import (
    AuthenticationError as KernelAuthenticationError,
    ContentFilterError as KernelContentFilterError,
    ContextLengthError as KernelContextLengthError,
    InvalidRequestError as KernelInvalidRequestError,
    LLMError as KernelLLMError,
    LLMTimeoutError as KernelLLMTimeoutError,
    ProviderUnavailableError as KernelProviderUnavailableError,
    RateLimitError as KernelRateLimitError,
)
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_provider() -> AnthropicProvider:
    """Create a provider with streaming disabled and max_retries=0 for isolation."""
    provider = AnthropicProvider(
        api_key="test-key",
        config={"use_streaming": False, "max_retries": 0},
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _make_anthropic_error(cls, message="error", status_code=400):
    """Construct an Anthropic SDK error with the expected shape."""
    # Anthropic SDK errors take (message, response, body)
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    return cls(message, response=mock_response, body=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRateLimitErrorTranslation:
    def test_translates_to_kernel_rate_limit_error(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.RateLimitError, "rate limited", status_code=429
        )
        sdk_error.response.headers = {"retry-after": "5.0"}
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelRateLimitError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.provider == "anthropic"
        assert e.status_code == 429
        assert e.retryable is True
        assert e.retry_after == 5.0
        assert e.__cause__ is sdk_error

    def test_retry_after_none_when_header_missing(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.RateLimitError, "rate limited", status_code=429
        )
        sdk_error.response.headers = {}
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelRateLimitError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.retry_after is None


class TestAuthenticationErrorTranslation:
    def test_translates_to_kernel_auth_error(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.AuthenticationError, "invalid key", status_code=401
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelAuthenticationError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.provider == "anthropic"
        assert e.retryable is False
        assert e.__cause__ is sdk_error


class TestBadRequestErrorTranslation:
    def test_context_length_message(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "prompt is too long: context length exceeded",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContextLengthError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.__cause__ is sdk_error

    def test_too_many_tokens_message(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "too many tokens in request",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContextLengthError):
            asyncio.run(provider.complete(_simple_request()))

    def test_content_filter_message(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "content blocked by safety filter",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContentFilterError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.__cause__ is sdk_error

    def test_safety_message(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "output blocked by safety system",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContentFilterError):
            asyncio.run(provider.complete(_simple_request()))

    def test_blocked_message(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "request blocked",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContentFilterError):
            asyncio.run(provider.complete(_simple_request()))

    def test_generic_bad_request(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "invalid model name",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelInvalidRequestError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.__cause__ is sdk_error


class TestAPIStatusErrorTranslation:
    def test_5xx_translates_to_provider_unavailable(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.InternalServerError,
            "internal server error",
            status_code=500,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.provider == "anthropic"
        assert e.status_code == 500
        assert e.retryable is True
        assert e.__cause__ is sdk_error

    def test_503_translates_to_provider_unavailable(self):
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.headers = {}
        sdk_error = anthropic.APIStatusError(
            "service unavailable", response=mock_response, body=None
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.status_code == 503


class TestTimeoutErrorTranslation:
    def test_asyncio_timeout_translates(self):
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        with pytest.raises(KernelLLMTimeoutError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.provider == "anthropic"
        assert e.retryable is True


class TestGenericExceptionTranslation:
    def test_unknown_exception_translates_to_llm_error(self):
        provider = _make_provider()
        original = RuntimeError("something unexpected")
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=original
        )

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.provider == "anthropic"
        assert e.retryable is True
        assert e.__cause__ is original


class TestCauseChainPreservation:
    """Verify __cause__ is set on all translated errors (raise X from e)."""

    def test_all_error_types_preserve_cause(self):
        """Smoke test: each SDK error type preserves __cause__."""
        test_cases = [
            (anthropic.RateLimitError, KernelRateLimitError, 429),
            (anthropic.AuthenticationError, KernelAuthenticationError, 401),
            (anthropic.BadRequestError, KernelInvalidRequestError, 400),
            (anthropic.InternalServerError, KernelProviderUnavailableError, 500),
        ]

        for sdk_cls, kernel_cls, status in test_cases:
            provider = _make_provider()
            sdk_error = _make_anthropic_error(sdk_cls, "test", status_code=status)
            provider.client.messages.with_raw_response.create = AsyncMock(
                side_effect=sdk_error
            )

            with pytest.raises(kernel_cls) as exc_info:
                asyncio.run(provider.complete(_simple_request()))

            assert exc_info.value.__cause__ is sdk_error, (
                f"{sdk_cls.__name__} → {kernel_cls.__name__}: __cause__ not preserved"
            )
