"""Tests for AnthropicProvider.close() method."""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_module_provider_anthropic import _DEFAULT_CLOSE_TIMEOUT
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


# ---------------------------------------------------------------------------
# recipes-7nj -- close() must be HARD BOUNDED.
#
# `AsyncAnthropic.close()` awaits `httpx.AsyncClient.aclose()`, which has no
# deadline of its own: on a half-closed (CLOSE-WAIT) connection it blocks
# indefinitely. Session cleanup runs inside the `finally` that PRECEDES a CLI
# command's return, so an unbounded close swallows the result of a completed
# run (recipes-8sr: measured 28-minute hang).
#
# Every test below drives the real bound, not a mocked-out one: the fake
# client's close() genuinely never returns.
# ---------------------------------------------------------------------------


def _never_returns_client() -> MagicMock:
    """A client whose close() never completes -- the CLOSE-WAIT stand-in."""
    client = MagicMock()

    async def _hang():
        await asyncio.Event().wait()  # never set

    client.close = MagicMock(side_effect=lambda: _hang())
    return client


@pytest.mark.asyncio
async def test_close_returns_within_bound_when_client_close_hangs(caplog):
    """A close() that never returns must not hold the caller past the bound."""
    provider = AnthropicProvider(api_key="k", config={"close_timeout": 0.25})
    provider._client = _never_returns_client()

    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(provider.close(), timeout=5.0)
    elapsed = time.monotonic() - started

    # Bounded: returns at ~the bound, nowhere near the outer 5s backstop.
    assert 0.25 <= elapsed < 2.0, f"close() took {elapsed:.2f}s, expected ~0.25s"

    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    abandoned = [m for m in warnings if "abandoning the httpx client" in m]
    assert len(abandoned) == 1, f"expected one abandonment warning, got: {warnings}"
    # The warning must name the provider instance so a hang in one of several
    # mounted instances is attributable.
    assert f"id=0x{id(provider):x}" in abandoned[0]
    assert "anthropic" in abandoned[0]
    # And the client slot is cleared even though the close never finished,
    # so the lazy-init contract still holds.
    assert provider._client is None


@pytest.mark.asyncio
async def test_close_default_bound_is_five_seconds():
    """The documented default (README `close_timeout`) is what ships."""
    assert _DEFAULT_CLOSE_TIMEOUT == 5.0
    provider = AnthropicProvider(api_key="k")
    assert provider._close_timeout == 5.0


@pytest.mark.asyncio
async def test_close_timeout_config_key_is_recognized(caplog):
    """`close_timeout` must be in the consumed-key allowlist -- otherwise
    setting it earns a spurious 'Unknown config key' warning."""
    with caplog.at_level(logging.WARNING):
        provider = AnthropicProvider(api_key="k", config={"close_timeout": "1.5"})
    assert provider._close_timeout == 1.5  # settings.yaml strings coerce
    assert not any("Unknown config key" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_close_is_prompt_and_silent_for_a_healthy_client(caplog):
    """The normal path must not pay the bound, and must not warn."""
    provider = AnthropicProvider(api_key="k")
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    provider._client = mock_client

    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        await provider.close()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, f"healthy close took {elapsed:.2f}s"
    mock_client.close.assert_awaited_once()
    assert not [r for r in caplog.records if "abandoning" in r.message]


@pytest.mark.asyncio
async def test_close_cancellation_of_caller_leaves_shielded_close_running():
    """CancelledError path unchanged: cancelling the CALLER does not cancel a
    close already in flight, and close() swallows the CancelledError."""
    provider = AnthropicProvider(api_key="k", config={"close_timeout": 30.0})

    started = asyncio.Event()
    inner_cancelled = False

    async def _slow_close():
        nonlocal inner_cancelled
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            inner_cancelled = True
            raise

    client = MagicMock()
    client.close = MagicMock(side_effect=lambda: _slow_close())
    provider._client = client

    task = asyncio.create_task(provider.close())
    await started.wait()
    task.cancel()
    # close() swallows the CancelledError rather than propagating it, exactly
    # as it did before the bound was added.
    await task

    await asyncio.sleep(0)
    assert not inner_cancelled, "shield must protect the in-flight close"


@pytest.mark.asyncio
async def test_close_does_not_swallow_a_real_close_error():
    """A genuine failure from client.close() still surfaces -- the bound must
    not turn every teardown error into silence."""
    provider = AnthropicProvider(api_key="k")
    client = MagicMock()
    client.close = AsyncMock(side_effect=RuntimeError("transport exploded"))
    provider._client = client

    with pytest.raises(RuntimeError, match="transport exploded"):
        await provider.close()
    assert provider._client is None
