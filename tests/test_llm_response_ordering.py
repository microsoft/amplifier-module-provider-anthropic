"""Tests for llm:response event canonical usage keys + input_tokens gross total.

Verifies:
- input_tokens in ChatResponse.usage is the gross total
  (fresh input + cache_read_input_tokens), not just the fresh input alone.
- llm:response event uses canonical usage keys:
  input_tokens, output_tokens, cache_read_tokens (not input/output/cache_read).
- llm:response event input_tokens reflects the gross total.
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import FakeCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
):
    """Create a mock raw API response with usage data."""
    usage_attrs = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cache_read_input_tokens is not None:
        usage_attrs["cache_read_input_tokens"] = cache_read_input_tokens
    if cache_creation_input_tokens is not None:
        usage_attrs["cache_creation_input_tokens"] = cache_creation_input_tokens

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
# input_tokens gross total in ChatResponse.usage
# ---------------------------------------------------------------------------


def test_input_tokens_includes_cache_read():
    """input_tokens in usage should be fresh input + cache_read_input_tokens."""
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response(input_tokens=100, cache_read_input_tokens=500)
    )
    result = asyncio.run(provider.complete(_simple_request()))
    assert result.usage is not None
    assert result.usage.input_tokens == 600, (
        f"Expected gross total 600 (100 + 500), got {result.usage.input_tokens}"
    )


def test_input_tokens_no_cache_unchanged():
    """input_tokens should be unchanged when there is no cache_read."""
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response(input_tokens=200, output_tokens=75)
    )
    result = asyncio.run(provider.complete(_simple_request()))
    assert result.usage is not None
    assert result.usage.input_tokens == 200, (
        f"Expected 200 with no cache, got {result.usage.input_tokens}"
    )


def test_total_tokens_reflects_gross_input():
    """total_tokens should include the gross input (fresh + cache_read) + output."""
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response(
            input_tokens=100, output_tokens=50, cache_read_input_tokens=500
        )
    )
    result = asyncio.run(provider.complete(_simple_request()))
    assert result.usage is not None
    assert result.usage.total_tokens == 650, (
        f"Expected total_tokens 650, got {result.usage.total_tokens}"
    )


# ---------------------------------------------------------------------------
# llm:response event canonical usage keys
# ---------------------------------------------------------------------------


def test_event_usage_uses_input_tokens_key():
    """llm:response event usage should have 'input_tokens' key (not 'input')."""
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response()
    )
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None
    usage = payload.get("usage", {})
    assert "input_tokens" in usage, f"Expected 'input_tokens', got keys: {list(usage.keys())}"


def test_event_usage_uses_output_tokens_key():
    """llm:response event usage should have 'output_tokens' key (not 'output')."""
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response()
    )
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None
    usage = payload.get("usage", {})
    assert "output_tokens" in usage, f"Expected 'output_tokens', got keys: {list(usage.keys())}"


def test_event_usage_uses_cache_read_tokens_key():
    """llm:response event usage should have 'cache_read_tokens' key when cache data present."""
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response(cache_read_input_tokens=500)
    )
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None
    usage = payload.get("usage", {})
    assert "cache_read_tokens" in usage, f"Expected 'cache_read_tokens', got keys: {list(usage.keys())}"


def test_event_usage_contains_cache_write_tokens():
    """Regression guard: cache_write_tokens must appear in emitted event_usage.

    ddcbb29 restored this field after a silent regression removed it.
    This test ensures it never silently disappears again.
    """
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response(cache_creation_input_tokens=300)
    )
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None
    usage = payload.get("usage", {})
    assert "cache_write_tokens" in usage, (
        f"Expected 'cache_write_tokens' in event_usage (regression guard for ddcbb29), "
        f"got keys: {list(usage.keys())}"
    )
    assert usage["cache_write_tokens"] == 300, (
        f"Expected cache_write_tokens=300, got {usage['cache_write_tokens']}"
    )


def test_event_usage_input_tokens_is_gross_total():
    """llm:response event's input_tokens should be the gross total (fresh + cache_read)."""
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response(input_tokens=100, cache_read_input_tokens=500)
    )
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None
    usage = payload.get("usage", {})
    assert usage.get("input_tokens") == 600, (
        f"Expected gross total 600 in event usage, got {usage.get('input_tokens')}"
    )


def test_event_does_not_use_old_input_key():
    """llm:response event usage should NOT have the old 'input' key."""
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response()
    )
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None
    usage = payload.get("usage", {})
    assert "input" not in usage, f"Old 'input' key should be gone, found: {usage}"


def test_event_does_not_use_old_output_key():
    """llm:response event usage should NOT have the old 'output' key."""
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response()
    )
    asyncio.run(provider.complete(_simple_request()))
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    payload = hooks.payload_for("llm:response")
    assert payload is not None
    usage = payload.get("usage", {})
    assert "output" not in usage, f"Old 'output' key should be gone, found: {usage}"
