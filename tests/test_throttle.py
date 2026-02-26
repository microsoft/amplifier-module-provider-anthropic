"""Tests for pre-emptive rate limit throttling.

Verifies:
- _RateLimitState starts empty (all None)
- State is updated after a successful API call
- Throttle fires when remaining < threshold (10%)
- No throttle when remaining > threshold
- No throttle when throttle_threshold=0 (disabled)
- Throttle picks the most constrained dimension
- provider:throttle event is emitted with correct payload when throttling
- No provider:throttle event when above threshold
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_retry.py)
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
    throttle_threshold: float = 0.1,
    throttle_delay: float = 5.0,
) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": 0,
            "min_retry_delay": 0.01,
            "max_retry_delay": 60.0,
            "retry_jitter": False,
            "throttle_threshold": throttle_threshold,
            "throttle_delay": throttle_delay,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _make_raw_mock(headers: dict | None = None) -> MagicMock:
    """Create a mock raw response with optional rate limit headers."""
    dummy = DummyResponse()
    raw_mock = MagicMock()
    raw_mock.parse.return_value = dummy
    raw_mock.headers = headers or {}
    return raw_mock


# ---------------------------------------------------------------------------
# _RateLimitState unit tests
# ---------------------------------------------------------------------------


class TestRateLimitStateDataclass:
    def test_starts_empty(self):
        """All fields should be None on creation."""
        from amplifier_module_provider_anthropic import _RateLimitState

        state = _RateLimitState()
        assert state.requests_remaining is None
        assert state.requests_limit is None
        assert state.requests_reset is None
        assert state.input_tokens_remaining is None
        assert state.input_tokens_limit is None
        assert state.input_tokens_reset is None
        assert state.output_tokens_remaining is None
        assert state.output_tokens_limit is None
        assert state.output_tokens_reset is None

    def test_update_from_headers(self):
        """update_from_headers should set known fields from dict."""
        from amplifier_module_provider_anthropic import _RateLimitState

        state = _RateLimitState()
        state.update_from_headers(
            {
                "requests_remaining": 5,
                "requests_limit": 100,
                "requests_reset": "2026-02-24T10:30:00Z",
                "input_tokens_remaining": 50000,
                "input_tokens_limit": 1000000,
            }
        )
        assert state.requests_remaining == 5
        assert state.requests_limit == 100
        assert state.requests_reset == "2026-02-24T10:30:00Z"
        assert state.input_tokens_remaining == 50000
        assert state.input_tokens_limit == 1000000
        # Unset fields remain None
        assert state.output_tokens_remaining is None

    def test_update_from_empty_dict(self):
        """Empty dict should not change any state."""
        from amplifier_module_provider_anthropic import _RateLimitState

        state = _RateLimitState()
        state.update_from_headers({})
        assert state.requests_remaining is None

    def test_update_from_none(self):
        """None input should not crash."""
        from amplifier_module_provider_anthropic import _RateLimitState

        state = _RateLimitState()
        state.update_from_headers(None)
        assert state.requests_remaining is None

    def test_ignores_unknown_keys(self):
        """Unknown keys in the dict should be silently ignored."""
        from amplifier_module_provider_anthropic import _RateLimitState

        state = _RateLimitState()
        state.update_from_headers({"unknown_key": 42, "requests_remaining": 5})
        assert state.requests_remaining == 5


class TestMostConstrainedRatio:
    def test_no_data_returns_1(self):
        """No data means no constraint known -> ratio 1.0."""
        from amplifier_module_provider_anthropic import _RateLimitState

        state = _RateLimitState()
        ratio, dimension, remaining, limit, reset = state.most_constrained_ratio()
        assert ratio == 1.0
        assert dimension == "unknown"
        assert remaining is None
        assert limit is None
        assert reset is None

    def test_single_dimension(self):
        """Single dimension data should be returned."""
        from amplifier_module_provider_anthropic import _RateLimitState

        state = _RateLimitState()
        state.requests_remaining = 5
        state.requests_limit = 100
        state.requests_reset = "2026-02-24T10:30:00Z"
        ratio, dimension, remaining, limit, reset = state.most_constrained_ratio()
        assert ratio == 0.05
        assert dimension == "requests"
        assert remaining == 5
        assert limit == 100
        assert reset == "2026-02-24T10:30:00Z"

    def test_picks_most_constrained(self):
        """Should return the dimension with the lowest ratio."""
        from amplifier_module_provider_anthropic import _RateLimitState

        state = _RateLimitState()
        # requests: 50/100 = 0.5
        state.requests_remaining = 50
        state.requests_limit = 100
        # input_tokens: 1000/1000000 = 0.001 (most constrained)
        state.input_tokens_remaining = 1000
        state.input_tokens_limit = 1000000
        state.input_tokens_reset = "2026-02-24T10:31:00Z"
        # output_tokens: 50000/100000 = 0.5
        state.output_tokens_remaining = 50000
        state.output_tokens_limit = 100000

        ratio, dimension, remaining, limit, reset = state.most_constrained_ratio()
        assert ratio == pytest.approx(0.001)
        assert dimension == "input_tokens"
        assert remaining == 1000
        assert limit == 1000000
        assert reset == "2026-02-24T10:31:00Z"

    def test_zero_limit_ignored(self):
        """A dimension with limit=0 should be skipped (avoid division by zero)."""
        from amplifier_module_provider_anthropic import _RateLimitState

        state = _RateLimitState()
        state.requests_remaining = 0
        state.requests_limit = 0
        ratio, dimension, _, _, _ = state.most_constrained_ratio()
        assert ratio == 1.0
        assert dimension == "unknown"


# ---------------------------------------------------------------------------
# Provider integration: throttle state initialization
# ---------------------------------------------------------------------------


class TestProviderThrottleInit:
    def test_provider_has_rate_limit_state(self):
        """Provider should have a _rate_limit_state after init."""
        provider = _make_provider()
        assert hasattr(provider, "_rate_limit_state")
        assert provider._rate_limit_state.requests_remaining is None

    def test_provider_has_throttle_config(self):
        """Provider should read throttle config from config dict."""
        provider = _make_provider(throttle_threshold=0.15, throttle_delay=10.0)
        assert provider._throttle_threshold == 0.15
        assert provider._throttle_delay == 10.0

    def test_default_throttle_config(self):
        """Without explicit config, defaults should be used."""
        provider = AnthropicProvider(
            api_key="test-key",
            config={"use_streaming": False, "max_retries": 0},
        )
        assert provider._throttle_threshold == 0.02
        assert provider._throttle_delay == 1.0


# ---------------------------------------------------------------------------
# Provider integration: state updated after successful API call
# ---------------------------------------------------------------------------


class TestThrottleStateUpdatedAfterCall:
    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_state_updated_from_response_headers(self, mock_sleep):
        """After a successful API call, _rate_limit_state should be updated."""
        provider = _make_provider()

        raw_mock = _make_raw_mock(
            headers={
                "anthropic-ratelimit-requests-remaining": "90",
                "anthropic-ratelimit-requests-limit": "100",
                "anthropic-ratelimit-input-tokens-remaining": "800000",
                "anthropic-ratelimit-input-tokens-limit": "1000000",
                "anthropic-ratelimit-output-tokens-remaining": "9000",
                "anthropic-ratelimit-output-tokens-limit": "100000",
            }
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=raw_mock
        )

        asyncio.run(provider.complete(_simple_request()))

        state = provider._rate_limit_state
        assert state.requests_remaining == 90
        assert state.requests_limit == 100
        assert state.input_tokens_remaining == 800000
        assert state.input_tokens_limit == 1000000
        assert state.output_tokens_remaining == 9000
        assert state.output_tokens_limit == 100000


# ---------------------------------------------------------------------------
# Provider integration: throttle fires / doesn't fire
# ---------------------------------------------------------------------------


class TestThrottleFires:
    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_throttle_fires_when_below_threshold(self, mock_sleep):
        """When remaining < threshold, asyncio.sleep should be called for the delay."""
        provider = _make_provider(throttle_threshold=0.1, throttle_delay=5.0)

        # Seed state: requests at 5/100 = 0.05, below 0.1 threshold
        provider._rate_limit_state.requests_remaining = 5
        provider._rate_limit_state.requests_limit = 100

        raw_mock = _make_raw_mock(
            headers={
                "anthropic-ratelimit-requests-remaining": "5",
                "anthropic-ratelimit-requests-limit": "100",
            }
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=raw_mock
        )

        asyncio.run(provider.complete(_simple_request()))

        # asyncio.sleep should have been called with the throttle delay
        sleep_calls = [c.args[0] for c in mock_sleep.await_args_list]
        assert any(abs(d - 5.0) < 0.01 for d in sleep_calls), (
            f"Expected a ~5.0s throttle sleep, got calls: {sleep_calls}"
        )

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_no_throttle_when_above_threshold(self, mock_sleep):
        """When remaining > threshold, no throttle delay should be injected."""
        provider = _make_provider(throttle_threshold=0.1, throttle_delay=5.0)

        # Seed state: requests at 50/100 = 0.5, well above 0.1 threshold
        provider._rate_limit_state.requests_remaining = 50
        provider._rate_limit_state.requests_limit = 100

        raw_mock = _make_raw_mock(
            headers={
                "anthropic-ratelimit-requests-remaining": "50",
                "anthropic-ratelimit-requests-limit": "100",
            }
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=raw_mock
        )

        asyncio.run(provider.complete(_simple_request()))

        # No sleep should have been called (max_retries=0, no throttle)
        assert mock_sleep.await_count == 0

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_no_throttle_when_disabled(self, mock_sleep):
        """When throttle_threshold=0, throttling should be disabled."""
        provider = _make_provider(throttle_threshold=0.0, throttle_delay=5.0)

        # Seed state: requests at 1/100 = 0.01, would be below threshold if enabled
        provider._rate_limit_state.requests_remaining = 1
        provider._rate_limit_state.requests_limit = 100

        raw_mock = _make_raw_mock(
            headers={
                "anthropic-ratelimit-requests-remaining": "1",
                "anthropic-ratelimit-requests-limit": "100",
            }
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=raw_mock
        )

        asyncio.run(provider.complete(_simple_request()))

        # No throttle sleep when disabled
        assert mock_sleep.await_count == 0


class TestThrottleMostConstrainedDimension:
    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_throttle_picks_most_constrained(self, mock_sleep):
        """Throttle should fire based on the most constrained dimension."""
        provider = _make_provider(throttle_threshold=0.1, throttle_delay=5.0)
        fake_coord = cast(FakeCoordinator, provider.coordinator)

        # requests: 50/100 = 0.5 (above threshold)
        provider._rate_limit_state.requests_remaining = 50
        provider._rate_limit_state.requests_limit = 100
        # output_tokens: 500/100000 = 0.005 (below threshold, most constrained)
        provider._rate_limit_state.output_tokens_remaining = 500
        provider._rate_limit_state.output_tokens_limit = 100000

        raw_mock = _make_raw_mock()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=raw_mock
        )

        asyncio.run(provider.complete(_simple_request()))

        # Should fire throttle because output_tokens ratio < threshold
        throttle_events = [
            e for e in fake_coord.hooks.events if e[0] == "provider:throttle"
        ]
        assert len(throttle_events) == 1
        payload = throttle_events[0][1]
        assert payload["dimension"] == "output_tokens"
        assert payload["remaining"] == 500
        assert payload["limit"] == 100000


# ---------------------------------------------------------------------------
# Provider integration: event emission
# ---------------------------------------------------------------------------


class TestThrottleEventEmission:
    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_emits_provider_throttle_event(self, mock_sleep):
        """provider:throttle event should be emitted with correct payload."""
        provider = _make_provider(throttle_threshold=0.1, throttle_delay=5.0)
        fake_coord = cast(FakeCoordinator, provider.coordinator)

        # Seed state below threshold
        provider._rate_limit_state.requests_remaining = 5
        provider._rate_limit_state.requests_limit = 100

        raw_mock = _make_raw_mock()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=raw_mock
        )

        asyncio.run(provider.complete(_simple_request()))

        throttle_events = [
            e for e in fake_coord.hooks.events if e[0] == "provider:throttle"
        ]
        assert len(throttle_events) == 1

        payload = throttle_events[0][1]
        assert payload["provider"] == "anthropic"
        assert payload["reason"] == "requests_low"
        assert payload["dimension"] == "requests"
        assert payload["remaining"] == 5
        assert payload["limit"] == 100
        assert payload["delay"] == 5.0
        assert payload["ratio"] == 0.05  # 5/100
        assert "reset_timestamp" in payload  # present (may be None)
        assert "model" in payload

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_no_throttle_event_when_above_threshold(self, mock_sleep):
        """No provider:throttle event should be emitted when above threshold."""
        provider = _make_provider(throttle_threshold=0.1, throttle_delay=5.0)
        fake_coord = cast(FakeCoordinator, provider.coordinator)

        # Seed state well above threshold
        provider._rate_limit_state.requests_remaining = 90
        provider._rate_limit_state.requests_limit = 100

        raw_mock = _make_raw_mock()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=raw_mock
        )

        asyncio.run(provider.complete(_simple_request()))

        throttle_events = [
            e for e in fake_coord.hooks.events if e[0] == "provider:throttle"
        ]
        assert len(throttle_events) == 0
