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
    """close() should be safe to call multiple times."""
    provider = AnthropicProvider(api_key="fake-key")
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    provider._client = mock_client

    await provider.close()
    await provider.close()

    assert mock_client.close.await_count == 2
    assert provider._client is not None  # close() does not clear the reference
