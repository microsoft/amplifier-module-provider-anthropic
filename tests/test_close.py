"""Tests for AnthropicProvider.close() method."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_module_provider_anthropic import AnthropicProvider


@pytest.mark.asyncio
async def test_close_calls_client_close_when_initialized():
    """close() should await the underlying client's close() when _client is set."""
    provider = AnthropicProvider(api_key="fake-key")
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    provider._client = mock_client

    await provider.close()

    mock_client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_is_safe_when_client_is_none():
    """close() should be safe to call when _client is None (never initialized)."""
    provider = AnthropicProvider(api_key="fake-key")
    assert provider._client is None

    await provider.close()  # Should not raise


@pytest.mark.asyncio
async def test_close_handles_cancelled_error():
    """close() should swallow CancelledError from the underlying client."""
    provider = AnthropicProvider(api_key="fake-key")
    mock_client = MagicMock()
    mock_client.close = AsyncMock(side_effect=asyncio.CancelledError)
    provider._client = mock_client

    await provider.close()  # Should not raise


@pytest.mark.asyncio
async def test_close_can_be_called_twice():
    """close() should be safe to call multiple times.

    The second call is a no-op: close() resets `_client` to None (see
    test_close_resets_client_to_none), so the second call's
    `if self._client is not None` guard is False and the mock's close()
    is only awaited once, not twice.
    """
    provider = AnthropicProvider(api_key="fake-key")
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    provider._client = mock_client

    await provider.close()
    await provider.close()

    mock_client.close.assert_awaited_once()
    assert provider._client is None


@pytest.mark.asyncio
async def test_close_resets_client_to_none():
    """close() must reset `_client` to None so a provider reused after
    teardown (e.g. a background task still in flight when session cleanup
    closes providers -- see hooks-session-naming) lazily rebuilds a fresh
    client via the `client` property instead of permanently failing every
    subsequent call with "Cannot send a request, as the client has been
    closed." (RuntimeError raised by httpx.AsyncClient.send() against a
    closed client).
    """
    provider = AnthropicProvider(api_key="fake-key")
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    provider._client = mock_client

    await provider.close()

    assert provider._client is None
    # Reusing the provider after close() must lazily rebuild a fresh client
    # rather than returning the (closed) mock.
    rebuilt = provider.client
    assert rebuilt is not mock_client
