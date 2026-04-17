"""Tests for process-wide concurrency semaphore and diagnostic logging.

Two features under test:

  Feature 1 — Semaphore
    A configurable ``max_concurrent_requests`` semaphore (default 5) limits how
    many API calls a single process can have in-flight simultaneously.  The
    semaphore is process-wide (shared across all AnthropicProvider instances) so
    that parent + delegated child sessions in the same process share the gate.
    Setting ``max_concurrent_requests=0`` disables the semaphore entirely.

  Feature 2 — Diagnostic logging
    Structured ``provider:concurrency`` events are emitted before each API call
    attempt, and ``provider:cloudflare_challenge`` events are emitted whenever a
    Cloudflare bot-challenge 403 is detected.  Both events carry the current
    active/waiting request counts, the configured limit, and os.getpid(), so
    that post-mortem analysis of events.jsonl can prove (or disprove) that
    concurrent request volume was responsible for a given Cloudflare block.
"""

import asyncio
import os
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

import amplifier_module_provider_anthropic as _mod
from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import AccessDeniedError as KernelAccessDeniedError
from amplifier_core.llm_errors import (
    ProviderUnavailableError as KernelProviderUnavailableError,
)
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider
from anthropic import APIStatusError as AnthropicAPIStatusError

from tests._helpers import DummyResponse, FakeCoordinator


# ---------------------------------------------------------------------------
# Shared test helpers  (mirror subset of test_cloudflare_retry.py)
# ---------------------------------------------------------------------------

CLOUDFLARE_HTML = """\
<!DOCTYPE html>
<html><head><title>Just a moment...</title></head>
<body>
<div id="cf-browser-verification">
  Checking if the site connection is secure
</div>
</body></html>
"""


def _make_api_status_error(
    status_code: int,
    body,
    content_type: str = "application/json",
    response_text: str = "",
) -> AnthropicAPIStatusError:
    """Build a fake AnthropicAPIStatusError with controllable attributes."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-type": content_type}
    response.text = response_text
    error = AnthropicAPIStatusError.__new__(AnthropicAPIStatusError)
    error.status_code = status_code
    error.body = body
    error.response = response
    error.message = f"Error code: {status_code}"
    error.args = (error.message,)
    return error


def _ok_raw_response():
    """Return a fake raw HTTP response that parse()s to DummyResponse."""
    raw = MagicMock()
    raw.parse.return_value = DummyResponse()
    raw.headers = {}
    return raw


# ---------------------------------------------------------------------------
# Fixture: isolate module-level globals between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_concurrency_globals():
    """Reset the process-wide semaphore and request counters before every test.

    asyncio.Semaphore objects are tied to the event loop that was running when
    they were created.  Each ``asyncio.run()`` call spins a new loop, so we
    must invalidate the cached semaphore to prevent "Future attached to a
    different loop" errors when tests run back-to-back.
    """
    _mod._process_semaphore = None
    _mod._process_semaphore_loop = None
    _mod._process_semaphore_max = 0
    _mod._active_requests = 0
    _mod._waiting_requests = 0
    yield
    # Clean up after the test too (defensive, avoids leaking into next fixture)
    _mod._process_semaphore = None
    _mod._process_semaphore_loop = None
    _mod._process_semaphore_max = 0
    _mod._active_requests = 0
    _mod._waiting_requests = 0


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _make_provider(
    max_concurrent: int = 5, **extra_config
) -> tuple[AnthropicProvider, FakeCoordinator]:
    config = {
        "use_streaming": False,
        "max_retries": 0,
        "max_concurrent_requests": max_concurrent,
        **extra_config,
    }
    provider = AnthropicProvider(api_key="test-key", config=config)
    coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, coordinator)
    return provider, coordinator


# ============================================================================
# Feature 1: Semaphore — configuration
# ============================================================================


class TestSemaphoreConfig:
    """Unit tests for max_concurrent_requests config parsing."""

    def test_default_is_5(self):
        """Default max_concurrent_requests should be 5."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider._max_concurrent_requests == 5

    def test_config_overrides_default(self):
        provider = AnthropicProvider(
            api_key="test-key", config={"max_concurrent_requests": 3}
        )
        assert provider._max_concurrent_requests == 3

    def test_zero_disables_semaphore(self):
        provider = AnthropicProvider(
            api_key="test-key", config={"max_concurrent_requests": 0}
        )
        assert provider._max_concurrent_requests == 0

    def test_get_process_semaphore_returns_none_for_zero(self):
        async def _run():
            sem = await _mod._get_process_semaphore(0)
            assert sem is None

        asyncio.run(_run())

    def test_get_process_semaphore_returns_semaphore_for_positive(self):
        async def _run():
            sem = await _mod._get_process_semaphore(3)
            assert sem is not None
            assert isinstance(sem, asyncio.Semaphore)

        asyncio.run(_run())

    def test_get_process_semaphore_is_idempotent_within_same_loop(self):
        """Same semaphore instance should be reused within one event loop."""

        async def _run():
            sem1 = await _mod._get_process_semaphore(5)
            sem2 = await _mod._get_process_semaphore(5)
            assert sem1 is sem2

        asyncio.run(_run())

    def test_get_process_semaphore_refreshes_across_loops(self):
        """Semaphore created in loop A must not be reused in loop B."""
        sem_from_loop_a: asyncio.Semaphore | None = None

        async def _loop_a():
            nonlocal sem_from_loop_a
            sem_from_loop_a = await _mod._get_process_semaphore(5)

        asyncio.run(_loop_a())
        # Reset only the loop reference to simulate a new run (globals fixture
        # already ensures a clean slate, but we want a specific mid-test reset)
        _mod._process_semaphore_loop = None  # trigger recreation

        async def _loop_b():
            sem = await _mod._get_process_semaphore(5)
            # Different object — must have been recreated for this loop
            assert sem is not sem_from_loop_a

        asyncio.run(_loop_b())


# ============================================================================
# Feature 1: Semaphore — concurrency enforcement
# ============================================================================


class TestSemaphoreLimitsConcurrency:
    """Verify that at most max_concurrent API calls are in-flight at once."""

    def test_semaphore_limits_concurrent_calls(self):
        """With limit=2 and 5 concurrent tasks, peak in-flight must be ≤ 2."""
        max_concurrent = 2
        provider, _ = _make_provider(max_concurrent=max_concurrent)

        in_flight = 0
        max_in_flight_seen = 0

        async def slow_api(**kwargs):
            nonlocal in_flight, max_in_flight_seen
            in_flight += 1
            max_in_flight_seen = max(max_in_flight_seen, in_flight)
            await asyncio.sleep(0.02)  # enough for all 5 coroutines to be created
            in_flight -= 1
            return _ok_raw_response()

        provider.client.messages.with_raw_response.create = slow_api

        async def _run():
            request = ChatRequest(messages=[Message(role="user", content="Hello")])
            await asyncio.gather(*[provider.complete(request) for _ in range(5)])

        asyncio.run(_run())
        assert max_in_flight_seen <= max_concurrent, (
            f"Expected ≤{max_concurrent} concurrent calls, saw {max_in_flight_seen}"
        )

    def test_semaphore_limit_of_1_serializes_calls(self):
        """Limit of 1 must fully serialize all API calls."""
        provider, _ = _make_provider(max_concurrent=1)

        order: list[int] = []
        n = 4

        async def serialized_api(**kwargs):
            order.append(len(order))
            await asyncio.sleep(0.01)
            return _ok_raw_response()

        provider.client.messages.with_raw_response.create = serialized_api

        async def _run():
            request = ChatRequest(messages=[Message(role="user", content="Hello")])
            await asyncio.gather(*[provider.complete(request) for _ in range(n)])

        asyncio.run(_run())
        # All n calls must have completed
        assert len(order) == n

    def test_disabled_semaphore_allows_full_concurrency(self):
        """With max_concurrent=0, all calls run without a gate."""
        provider, _ = _make_provider(max_concurrent=0)

        in_flight = 0
        max_in_flight_seen = 0

        async def concurrent_api(**kwargs):
            nonlocal in_flight, max_in_flight_seen
            in_flight += 1
            max_in_flight_seen = max(max_in_flight_seen, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return _ok_raw_response()

        provider.client.messages.with_raw_response.create = concurrent_api

        async def _run():
            request = ChatRequest(messages=[Message(role="user", content="Hello")])
            await asyncio.gather(*[provider.complete(request) for _ in range(5)])

        asyncio.run(_run())
        # Without gate, multiple calls overlap
        assert max_in_flight_seen > 1

    def test_all_requests_complete_with_semaphore(self):
        """Semaphore must not prevent any request from completing."""
        provider, _ = _make_provider(max_concurrent=2)

        call_count = 0

        async def counting_api(**kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.005)
            return _ok_raw_response()

        provider.client.messages.with_raw_response.create = counting_api

        async def _run():
            request = ChatRequest(messages=[Message(role="user", content="Hello")])
            results = await asyncio.gather(
                *[provider.complete(request) for _ in range(6)]
            )
            return results

        results = asyncio.run(_run())
        assert call_count == 6
        assert len(results) == 6
        assert all(r is not None for r in results)


# ============================================================================
# Feature 2: provider:concurrency event emission
# ============================================================================


class TestConcurrencyEventEmission:
    """Verify provider:concurrency events are emitted with correct payload."""

    def test_event_emitted_on_success(self):
        """A provider:concurrency event should be emitted for each API call."""
        provider, coordinator = _make_provider(max_concurrent=5)
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_ok_raw_response()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert len(concurrency_events) >= 1

    def test_event_has_all_required_fields(self):
        """provider:concurrency payload must contain every documented field."""
        provider, coordinator = _make_provider(max_concurrent=3)
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_ok_raw_response()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert len(concurrency_events) >= 1
        payload = concurrency_events[0][1]

        for field in (
            "provider",
            "model",
            "active_requests",
            "waiting_requests",
            "max_concurrent",
            "process_id",
        ):
            assert field in payload, (
                f"Missing field '{field}' in provider:concurrency event"
            )

    def test_event_max_concurrent_matches_config(self):
        """max_concurrent in event must equal the configured value."""
        provider, coordinator = _make_provider(max_concurrent=7)
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_ok_raw_response()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert concurrency_events[0][1]["max_concurrent"] == 7

    def test_event_process_id_matches_current_process(self):
        """process_id in event must equal os.getpid()."""
        provider, coordinator = _make_provider(max_concurrent=5)
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_ok_raw_response()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert concurrency_events[0][1]["process_id"] == os.getpid()

    def test_event_emitted_when_semaphore_disabled(self):
        """provider:concurrency is emitted even when max_concurrent=0."""
        provider, coordinator = _make_provider(max_concurrent=0)
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_ok_raw_response()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert len(concurrency_events) >= 1
        # max_concurrent should be 0 (disabled) in the payload
        assert concurrency_events[0][1]["max_concurrent"] == 0

    def test_active_requests_at_least_1_during_call(self):
        """active_requests in the event should be ≥ 1 (the call itself)."""
        provider, coordinator = _make_provider(max_concurrent=5)
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_ok_raw_response()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        concurrency_events = [
            e for e in coordinator.hooks.events if e[0] == "provider:concurrency"
        ]
        assert concurrency_events[0][1]["active_requests"] >= 1

    def test_no_event_emitted_without_coordinator(self):
        """When no coordinator is attached, provider:concurrency is silently skipped."""
        provider = AnthropicProvider(
            api_key="test-key",
            config={
                "use_streaming": False,
                "max_retries": 0,
                "max_concurrent_requests": 5,
            },
        )
        provider.coordinator = None
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_ok_raw_response()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        # Should not raise even without a coordinator
        result = asyncio.run(provider.complete(request))
        assert result is not None


# ============================================================================
# Feature 2: provider:cloudflare_challenge event
# ============================================================================


class TestCloudflargeChallengeEvent:
    """Verify provider:cloudflare_challenge events carry concurrency diagnostics."""

    def _cf_error(self) -> AnthropicAPIStatusError:
        return _make_api_status_error(
            status_code=403,
            body=None,
            content_type="text/html",
            response_text=CLOUDFLARE_HTML,
        )

    def test_cloudflare_challenge_event_is_emitted(self):
        """provider:cloudflare_challenge emitted when CF 403 is detected."""
        provider, coordinator = _make_provider(max_concurrent=5)
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=self._cf_error()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        with pytest.raises(KernelProviderUnavailableError):
            asyncio.run(provider.complete(request))

        cf_events = [
            e
            for e in coordinator.hooks.events
            if e[0] == "provider:cloudflare_challenge"
        ]
        assert len(cf_events) >= 1

    def test_cloudflare_challenge_event_has_all_required_fields(self):
        """provider:cloudflare_challenge payload must contain all diagnostic fields."""
        provider, coordinator = _make_provider(max_concurrent=5)
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=self._cf_error()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        with pytest.raises(KernelProviderUnavailableError):
            asyncio.run(provider.complete(request))

        cf_events = [
            e
            for e in coordinator.hooks.events
            if e[0] == "provider:cloudflare_challenge"
        ]
        assert len(cf_events) >= 1
        payload = cf_events[0][1]

        for field in (
            "provider",
            "model",
            "active_requests",
            "waiting_requests",
            "max_concurrent",
            "process_id",
            "timestamp",
        ):
            assert field in payload, (
                f"Missing field '{field}' in provider:cloudflare_challenge event"
            )

    def test_cloudflare_challenge_event_process_id_is_current(self):
        """process_id must equal os.getpid() so cross-process events can be distinguished."""
        provider, coordinator = _make_provider(max_concurrent=5)
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=self._cf_error()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        with pytest.raises(KernelProviderUnavailableError):
            asyncio.run(provider.complete(request))

        cf_events = [
            e
            for e in coordinator.hooks.events
            if e[0] == "provider:cloudflare_challenge"
        ]
        assert cf_events[0][1]["process_id"] == os.getpid()

    def test_cloudflare_challenge_event_emitted_on_each_retry(self):
        """One cloudflare_challenge event emitted per attempt (initial + retries)."""
        # max_retries=2 → 1 initial attempt + 2 retries = 3 total attempts
        provider, coordinator = _make_provider(
            max_concurrent=5, max_retries=2, min_retry_delay=0.005
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=self._cf_error()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        with pytest.raises(KernelProviderUnavailableError):
            asyncio.run(provider.complete(request))

        cf_events = [
            e
            for e in coordinator.hooks.events
            if e[0] == "provider:cloudflare_challenge"
        ]
        # 1 initial + 2 retries = 3 cloudflare_challenge events
        assert len(cf_events) == 3

    def test_no_cloudflare_event_for_real_api_403(self):
        """A genuine API 403 (JSON body) must NOT emit cloudflare_challenge."""
        provider, coordinator = _make_provider(max_concurrent=5)
        api_error = _make_api_status_error(
            status_code=403,
            body={
                "type": "error",
                "error": {"type": "forbidden", "message": "Forbidden"},
            },
            content_type="application/json",
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=api_error
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        with pytest.raises(KernelAccessDeniedError):
            asyncio.run(provider.complete(request))

        cf_events = [
            e
            for e in coordinator.hooks.events
            if e[0] == "provider:cloudflare_challenge"
        ]
        assert len(cf_events) == 0, (
            "Real API 403 must not be misidentified as Cloudflare"
        )

    def test_cloudflare_challenge_event_has_concurrency_info(self):
        """active_requests and max_concurrent in the CF event should be well-formed."""
        provider, coordinator = _make_provider(max_concurrent=5)
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=self._cf_error()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        with pytest.raises(KernelProviderUnavailableError):
            asyncio.run(provider.complete(request))

        cf_events = [
            e
            for e in coordinator.hooks.events
            if e[0] == "provider:cloudflare_challenge"
        ]
        payload = cf_events[0][1]
        assert isinstance(payload["active_requests"], int)
        assert isinstance(payload["waiting_requests"], int)
        assert payload["max_concurrent"] == 5
        assert isinstance(payload["timestamp"], float)
