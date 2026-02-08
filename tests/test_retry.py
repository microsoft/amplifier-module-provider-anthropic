"""Tests for Phase 2 retry refactor: catch LLMError with retryable check.

Verifies:
- Retryable errors (RateLimitError, ProviderUnavailableError, LLMTimeoutError) are retried
- Non-retryable errors (AuthenticationError) raise immediately
- Fail-fast when retry_after > max_retry_delay
- Event name is provider:retry (not anthropic:rate_limit_retry)
- Final failure raises kernel error type (not RuntimeError)
- Existing backoff/jitter behavior preserved
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import (
    AuthenticationError as KernelAuthenticationError,
    ProviderUnavailableError as KernelProviderUnavailableError,
    RateLimitError as KernelRateLimitError,
)
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider


# ---------------------------------------------------------------------------
# Helpers (same as test_error_translation.py)
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
    """Minimal Anthropic API response stub."""

    def __init__(self):
        self.content = [SimpleNamespace(type="text", text="ok")]
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.stop_reason = "end_turn"
        self.model = "claude-sonnet-4-5-20250929"


def _make_provider(
    max_retries: int = 3, max_retry_delay: float = 60.0
) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": max_retries,
            "min_retry_delay": 0.01,  # Fast for tests
            "max_retry_delay": max_retry_delay,
            "retry_jitter": False,  # Deterministic for tests
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _make_sdk_rate_limit_error(
    retry_after: float | None = None,
) -> anthropic.RateLimitError:
    mock_response = MagicMock()
    mock_response.status_code = 429
    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["retry-after"] = str(retry_after)
    mock_response.headers = headers
    return anthropic.RateLimitError("rate limited", response=mock_response, body=None)


def _make_sdk_server_error() -> anthropic.InternalServerError:
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.headers = {}
    return anthropic.InternalServerError(
        "server error", response=mock_response, body=None
    )


def _make_sdk_auth_error() -> anthropic.AuthenticationError:
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}
    return anthropic.AuthenticationError("bad key", response=mock_response, body=None)


# ---------------------------------------------------------------------------
# Retry behaviour tests
# ---------------------------------------------------------------------------


class TestRetryOnTransientErrors:
    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_retries_rate_limit_then_succeeds(self, mock_sleep):
        """RateLimitError on first 2 calls → success on 3rd → response returned."""
        provider = _make_provider(max_retries=3)
        dummy = DummyResponse()

        raw_mock = MagicMock()
        raw_mock.parse.return_value = dummy
        raw_mock.headers = {}

        # Fail twice, succeed third time
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_rate_limit_error(retry_after=1.0),
                _make_sdk_rate_limit_error(retry_after=1.0),
                raw_mock,
            ]
        )

        result = asyncio.run(provider.complete(_simple_request()))
        assert result is not None
        assert provider.client.messages.with_raw_response.create.await_count == 3

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_retries_provider_unavailable(self, mock_sleep):
        """ProviderUnavailableError (5xx) should be retried."""
        provider = _make_provider(max_retries=3)
        dummy = DummyResponse()

        raw_mock = MagicMock()
        raw_mock.parse.return_value = dummy
        raw_mock.headers = {}

        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_server_error(),
                raw_mock,
            ]
        )

        result = asyncio.run(provider.complete(_simple_request()))
        assert result is not None
        assert provider.client.messages.with_raw_response.create.await_count == 2

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_retries_timeout_error(self, mock_sleep):
        """LLMTimeoutError should be retried."""
        provider = _make_provider(max_retries=3)
        dummy = DummyResponse()

        raw_mock = MagicMock()
        raw_mock.parse.return_value = dummy
        raw_mock.headers = {}

        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                asyncio.TimeoutError(),
                raw_mock,
            ]
        )

        result = asyncio.run(provider.complete(_simple_request()))
        assert result is not None
        assert provider.client.messages.with_raw_response.create.await_count == 2


class TestNoRetryOnNonRetryable:
    def test_auth_error_not_retried(self):
        """AuthenticationError (retryable=False) should raise immediately, no retry."""
        provider = _make_provider(max_retries=5)
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=_make_sdk_auth_error()
        )

        with pytest.raises(KernelAuthenticationError):
            asyncio.run(provider.complete(_simple_request()))

        # Should only be called once — no retries
        assert provider.client.messages.with_raw_response.create.await_count == 1


class TestFailFastRetryAfterExceedsMax:
    def test_retry_after_exceeds_max_raises_immediately(self):
        """If retry_after > max_retry_delay, raise immediately instead of waiting."""
        provider = _make_provider(max_retries=5, max_retry_delay=60.0)
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=_make_sdk_rate_limit_error(retry_after=120.0)
        )

        with pytest.raises(KernelRateLimitError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        # Only called once — fail-fast, no retry
        assert provider.client.messages.with_raw_response.create.await_count == 1
        assert exc_info.value.retry_after == 120.0


class TestRetryEventEmission:
    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_emits_provider_retry_event(self, mock_sleep):
        """Retry should emit provider:retry (not anthropic:rate_limit_retry)."""
        provider = _make_provider(max_retries=3)
        fake_coord = cast(FakeCoordinator, provider.coordinator)
        dummy = DummyResponse()

        raw_mock = MagicMock()
        raw_mock.parse.return_value = dummy
        raw_mock.headers = {}

        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_rate_limit_error(retry_after=1.0),
                raw_mock,
            ]
        )

        asyncio.run(provider.complete(_simple_request()))

        retry_events = [e for e in fake_coord.hooks.events if e[0] == "provider:retry"]
        assert len(retry_events) == 1

        payload = retry_events[0][1]
        assert payload["provider"] == "anthropic"
        assert payload["attempt"] == 1
        assert payload["error_type"] == "RateLimitError"
        assert "delay" in payload

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_no_old_event_name(self, mock_sleep):
        """Old event name anthropic:rate_limit_retry should NOT be emitted."""
        provider = _make_provider(max_retries=3)
        fake_coord = cast(FakeCoordinator, provider.coordinator)
        dummy = DummyResponse()

        raw_mock = MagicMock()
        raw_mock.parse.return_value = dummy
        raw_mock.headers = {}

        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_rate_limit_error(retry_after=1.0),
                raw_mock,
            ]
        )

        asyncio.run(provider.complete(_simple_request()))

        old_events = [
            e for e in fake_coord.hooks.events if e[0] == "anthropic:rate_limit_retry"
        ]
        assert len(old_events) == 0, "Old event name should not be emitted"


class TestFinalFailureRaisesKernelError:
    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_exhausted_retries_raises_kernel_error_not_runtime_error(self, mock_sleep):
        """After exhausting retries, the kernel error propagates (not RuntimeError)."""
        provider = _make_provider(max_retries=2)
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=_make_sdk_rate_limit_error(retry_after=1.0)
        )

        with pytest.raises(KernelRateLimitError):
            asyncio.run(provider.complete(_simple_request()))

        # 1 initial + 2 retries = 3 total calls
        assert provider.client.messages.with_raw_response.create.await_count == 3

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_exhausted_retries_does_not_raise_runtime_error(self, mock_sleep):
        """RuntimeError should NOT be raised anymore (old behavior removed)."""
        provider = _make_provider(max_retries=1)
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=_make_sdk_rate_limit_error(retry_after=1.0)
        )

        # Should NOT raise RuntimeError
        with pytest.raises(KernelRateLimitError):
            asyncio.run(provider.complete(_simple_request()))

        # Verify it's NOT RuntimeError
        try:
            asyncio.run(provider.complete(_simple_request()))
        except RuntimeError:
            pytest.fail(
                "Should not raise RuntimeError — should raise KernelRateLimitError"
            )
        except KernelRateLimitError:
            pass  # Expected


class TestBackoffPattern:
    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_exponential_backoff_delays(self, mock_sleep):
        """Delays should follow exponential backoff when no retry-after header."""
        provider = _make_provider(max_retries=3)
        # Use errors without retry-after to trigger exponential backoff
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=_make_sdk_server_error()
        )

        with pytest.raises(KernelProviderUnavailableError):
            asyncio.run(provider.complete(_simple_request()))

        # With min_retry_delay=0.01 and no jitter:
        # attempt 1: 0.01 * 2^0 = 0.01
        # attempt 2: 0.01 * 2^1 = 0.02
        # attempt 3: 0.01 * 2^2 = 0.04
        assert mock_sleep.await_count == 3
        delays = [call.args[0] for call in mock_sleep.await_args_list]
        assert abs(delays[0] - 0.01) < 0.001
        assert abs(delays[1] - 0.02) < 0.001
        assert abs(delays[2] - 0.04) < 0.001
