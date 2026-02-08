"""Tests for Phase 2 Usage named fields: cache_read_tokens, cache_write_tokens.

Verifies:
- cache_read_tokens is set from cache_read_input_tokens
- cache_write_tokens is set from cache_creation_input_tokens
- Provider-native extras still present for backward compat
- reasoning_tokens is None (Anthropic doesn't provide separate count)
- Existing input/output/total tokens still correct
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_core import ModuleCoordinator
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
    provider = AnthropicProvider(
        api_key="test-key",
        config={"use_streaming": False, "max_retries": 0},
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _make_raw_response(
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
):
    """Create a mock raw API response with usage data."""
    usage_attrs = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cache_creation_input_tokens is not None:
        usage_attrs["cache_creation_input_tokens"] = cache_creation_input_tokens
    if cache_read_input_tokens is not None:
        usage_attrs["cache_read_input_tokens"] = cache_read_input_tokens

    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="response text")],
        usage=SimpleNamespace(**usage_attrs),
        stop_reason="end_turn",
        model="claude-sonnet-4-5-20250929",
    )

    raw = MagicMock()
    raw.parse.return_value = response
    raw.headers = {}
    return raw


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUsageNamedFields:
    def test_cache_read_tokens_from_response(self):
        """cache_read_tokens should be set from cache_read_input_tokens."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_response(cache_read_input_tokens=500)
        )

        result = asyncio.run(provider.complete(_simple_request()))
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 500

    def test_cache_write_tokens_from_response(self):
        """cache_write_tokens should be set from cache_creation_input_tokens."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_response(cache_creation_input_tokens=200)
        )

        result = asyncio.run(provider.complete(_simple_request()))
        assert result.usage is not None
        assert result.usage.cache_write_tokens == 200

    def test_both_cache_fields_set(self):
        """Both cache fields should be set when both are in the response."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_response(
                cache_creation_input_tokens=200,
                cache_read_input_tokens=500,
            )
        )

        result = asyncio.run(provider.complete(_simple_request()))
        assert result.usage is not None
        assert result.usage.cache_read_tokens == 500
        assert result.usage.cache_write_tokens == 200

    def test_cache_fields_none_when_not_in_response(self):
        """Cache fields should be None when not in the API response."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_response()
        )

        result = asyncio.run(provider.complete(_simple_request()))
        assert result.usage is not None
        assert result.usage.cache_read_tokens is None
        assert result.usage.cache_write_tokens is None


class TestUsageBackwardCompat:
    def test_extras_still_present(self):
        """Provider-native extras (cache_creation_input_tokens, cache_read_input_tokens)
        should still be present via extra='allow' for backward compatibility."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_response(
                cache_creation_input_tokens=200,
                cache_read_input_tokens=500,
            )
        )

        result = asyncio.run(provider.complete(_simple_request()))
        assert result.usage is not None

        # Access extras via model_dump or direct attribute
        usage_dict = result.usage.model_dump()
        assert usage_dict.get("cache_creation_input_tokens") == 200
        assert usage_dict.get("cache_read_input_tokens") == 500

    def test_extras_not_present_when_values_absent(self):
        """Extras should NOT be set when the API response doesn't have cache data."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_response()
        )

        result = asyncio.run(provider.complete(_simple_request()))
        assert result.usage is not None

        usage_dict = result.usage.model_dump()
        # These should NOT be in the dict when values are absent
        assert "cache_creation_input_tokens" not in usage_dict
        assert "cache_read_input_tokens" not in usage_dict


class TestUsageReasoningTokens:
    def test_reasoning_tokens_is_none(self):
        """reasoning_tokens should always be None for Anthropic.

        Anthropic does not provide a separate reasoning token count —
        thinking tokens are included in output_tokens.
        """
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_response()
        )

        result = asyncio.run(provider.complete(_simple_request()))
        assert result.usage is not None
        assert result.usage.reasoning_tokens is None


class TestUsageBaseFields:
    def test_input_output_total_tokens(self):
        """Standard token fields should be correctly populated."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_response(input_tokens=100, output_tokens=50)
        )

        result = asyncio.run(provider.complete(_simple_request()))
        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150

    def test_zero_cache_values_treated_as_none(self):
        """Zero values for cache fields should be treated as None (falsy → None)."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_response(
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
        )

        result = asyncio.run(provider.complete(_simple_request()))
        assert result.usage is not None
        # 0 is falsy, so `or None` converts to None
        assert result.usage.cache_read_tokens is None
        assert result.usage.cache_write_tokens is None
