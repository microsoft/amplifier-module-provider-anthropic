"""Tests for HTTP 200 with error body retry behavior.

Anthropic can return HTTP 200 with an "event: error" SSE during streaming.
The SDK raises APIStatusError(status_code=200) in this case — a base class
instance because _make_status_error has no mapping for 200.

This MUST map to KernelProviderUnavailableError(retryable=True), NOT
KernelLLMError(retryable=False).

Regression test for session 897eb0d4: HTTP 200 error was marked
retryable=False and killed the session instead of retrying.
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import (
    LLMError as KernelLLMError,
    ProviderUnavailableError as KernelProviderUnavailableError,
)
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider
from anthropic import APIStatusError as AnthropicAPIStatusError


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


class DummyResponse:
    """Minimal response stub for success cases."""

    def __init__(self, content=None):
        self.content = content or []
        self.usage = SimpleNamespace(input_tokens=0, output_tokens=0)
        self.stop_reason = "end_turn"
        self.model = "claude-sonnet-4-5-20250929"


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


def _make_http200_error(
    body: dict | None = None,
) -> AnthropicAPIStatusError:
    """Create an APIStatusError with status_code=200.

    This is what the SDK raises when the streaming response sends an
    "event: error" SSE despite HTTP 200 status.  _make_status_error()
    has no mapping for 200, so it returns a base APIStatusError.
    """
    if body is None:
        body = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Internal server error",
            },
        }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    # Build the error the same way the SDK does
    error = AnthropicAPIStatusError.__new__(AnthropicAPIStatusError)
    error.status_code = 200
    error.body = body
    error.response = mock_response
    error.message = str(body)
    error.args = (error.message,)
    return error


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHTTP200ErrorRetry:
    """HTTP 200 with error body must be retryable."""

    def test_http200_error_raises_retryable_provider_unavailable(self):
        """Core regression test: HTTP 200 error must be retryable=True."""
        provider = _make_provider()
        sdk_error = _make_http200_error()
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.retryable is True, (
            f"HTTP 200 error must be retryable=True, got retryable={e.retryable}. "
            "The AnthropicAPIStatusError handler must catch status < 400 before "
            "the retryable=False fallthrough."
        )
        assert e.status_code == 200
        assert e.provider == "anthropic"
        assert e.__cause__ is sdk_error

    def test_http200_error_is_not_kernel_llm_error(self):
        """HTTP 200 error must NOT raise KernelLLMError (the old broken path)."""
        provider = _make_provider()
        sdk_error = _make_http200_error()
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelProviderUnavailableError):
            asyncio.run(provider.complete(_simple_request()))

        # If this raises KernelLLMError instead of KernelProviderUnavailableError,
        # the except above won't catch it and the test fails — which is correct.

    def test_http200_error_preserves_body_in_message(self):
        """Error message should contain the API error body for diagnostics."""
        provider = _make_provider()
        body = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Internal server error",
            },
        }
        sdk_error = _make_http200_error(body=body)
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert "api_error" in str(exc_info.value) or "Internal server error" in str(
            exc_info.value
        )

    def test_http200_error_includes_model(self):
        """Model name should be propagated to the kernel error."""
        provider = _make_provider()
        sdk_error = _make_http200_error()
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.model == "claude-sonnet-4-5"

    def test_http200_error_retried_then_succeeds(self):
        """HTTP 200 error followed by success should work via retry loop."""
        provider = AnthropicProvider(
            api_key="test-key",
            config={"use_streaming": False, "max_retries": 2},
        )
        fake_coordinator = FakeCoordinator()
        provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

        sdk_error = _make_http200_error()

        raw_ok = MagicMock()
        raw_ok.parse.return_value = DummyResponse()
        raw_ok.headers = {}

        call_count = 0

        async def flaky_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise sdk_error
            return raw_ok

        provider.client.messages.with_raw_response.create = flaky_create  # type: ignore[method-assign]

        response = asyncio.run(provider.complete(_simple_request()))

        assert response is not None
        assert call_count == 2, "Should have retried once after HTTP 200 error"


class TestOther2xxAnd3xxErrors:
    """Other non-4xx, non-5xx status codes should also be retryable.

    If the SDK ever raises APIStatusError with another 2xx or 3xx status,
    it's likely a similar transient issue and should be retried.
    """

    @pytest.mark.parametrize("status_code", [201, 202, 204, 301, 302, 307])
    def test_non_error_status_codes_are_retryable(self, status_code):
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.headers = {}
        sdk_error = AnthropicAPIStatusError.__new__(AnthropicAPIStatusError)
        sdk_error.status_code = status_code
        sdk_error.body = None
        sdk_error.response = mock_response
        sdk_error.message = f"Unexpected status {status_code}"
        sdk_error.args = (sdk_error.message,)

        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == status_code


class TestExistingBehaviorUnchanged:
    """Verify the fix doesn't break existing 4xx non-retryable behavior."""

    @pytest.mark.parametrize("status_code", [405, 406, 408, 409, 410, 418, 422])
    def test_4xx_errors_remain_non_retryable(self, status_code):
        """4xx errors (not handled by specific branches) stay retryable=False."""
        provider = _make_provider()
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.headers = {}
        sdk_error = AnthropicAPIStatusError.__new__(AnthropicAPIStatusError)
        sdk_error.status_code = status_code
        sdk_error.body = None
        sdk_error.response = mock_response
        sdk_error.message = f"Error {status_code}"
        sdk_error.args = (sdk_error.message,)

        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.retryable is False
        assert exc_info.value.status_code == status_code
