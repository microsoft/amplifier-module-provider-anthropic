"""Focused classification, optional hooks, throttling, and concurrent attribution."""

import asyncio
import contextvars
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest
from anthropic import Timeout, _base_client
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._request_observation import (
    RequestObservation, effective_limits,
)
from amplifier_module_provider_anthropic._request_safety import RequestOutcomeUnknownError
from tests._helpers import DummyResponse


transport = getattr(_base_client, "httpx2", None) or _base_client.httpx


@pytest.mark.parametrize("phase", ["ConnectError", "ConnectTimeout", "PoolTimeout"])
@pytest.mark.asyncio
async def test_real_sdk_typed_pre_send_chain_can_retry(phase):
    attempts = 0
    async def handle(request):
        nonlocal attempts
        if request.method != "POST":
            return transport.Response(404, json={"type": "error", "error": {"type": "not_found_error"}})
        if request.url.path.endswith("/count_tokens"):
            return transport.Response(200, json={"input_tokens": 1})
        assert request.url.path == "/v1/messages"
        attempts += 1
        if attempts == 1:
            raise getattr(transport, phase)("private-phase-sentinel", request=request)
        return transport.Response(200, json={
            "id": "msg_fixture", "type": "message", "role": "assistant",
            "model": "claude-sonnet-5-5", "content": [], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    provider = AnthropicProvider("offline-placeholder", {
        "use_streaming": False, "min_retry_delay": 0.001, "retry_jitter": False,
    })
    provider._client = anthropic.AsyncAnthropic(
        api_key="offline-placeholder", max_retries=0,
        http_client=transport.AsyncClient(
            transport=transport.MockTransport(handle), trust_env=False,
        ),
    )
    try:
        await provider.complete(ChatRequest(messages=[Message(role="user", content="fixture")]))
        assert attempts == 2
    finally:
        await provider.close()


@pytest.mark.parametrize("phase", ["ReadError", "ReadTimeout", "WriteTimeout", "RemoteProtocolError"])
@pytest.mark.asyncio
async def test_real_sdk_other_phase_never_retries(phase):
    attempts = 0
    async def handle(request):
        nonlocal attempts
        if request.method != "POST":
            return transport.Response(404, json={"type": "error", "error": {"type": "not_found_error"}})
        if request.url.path.endswith("/count_tokens"):
            return transport.Response(200, json={"input_tokens": 1})
        assert request.url.path == "/v1/messages"
        attempts += 1
        raise getattr(transport, phase)("private-phase-sentinel", request=request)

    provider = AnthropicProvider("offline-placeholder", {"use_streaming": False})
    provider._client = anthropic.AsyncAnthropic(
        api_key="offline-placeholder", max_retries=0,
        http_client=transport.AsyncClient(
            transport=transport.MockTransport(handle), trust_env=False,
        ),
    )
    try:
        from amplifier_core.llm_errors import LLMTimeoutError
        with pytest.raises((RequestOutcomeUnknownError, LLMTimeoutError)) as caught:
            await provider.complete(ChatRequest(messages=[Message(role="user", content="fixture")]))
        assert attempts == 1
        assert not caught.value.retryable
        assert "private-phase-sentinel" not in str(caught.value)
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_request_local_throttle_and_receipt_context(monkeypatch):
    call = contextvars.ContextVar("call")
    events = []
    class Hooks:
        async def emit(self, event, payload):
            events.append((call.get(), event, payload))

    clock = [0.0]
    monkeypatch.setattr(
        "amplifier_module_provider_anthropic._request_observation.time.monotonic",
        lambda: clock[0],
    )
    async def observe(label):
        call.set(label)
        observation = RequestObservation(Hooks(), effective_limits(None, Timeout(None, connect=5, pool=5)))
        await observation.started()
        await observation.activity()
        for _ in range(100):
            await observation.activity()
        clock[0] += 1.1
        await observation.activity()
        await observation.activity()
        await observation.flush()
        await observation.flush()

    await asyncio.gather(observe("root"), observe("delegate"))
    for label in ("root", "delegate"):
        local = [p for owner, _, p in events if owner == label]
        assert [p["observation"] for p in local] == [
            "attempt_started", "response_activity", "response_activity", "response_activity",
        ]
        assert all(p["attempt"] == 1 for p in local)
        assert all(set(p) == {"version", "observation", "attempt", "limits"} for p in local)


@pytest.mark.parametrize("hooks", [None, SimpleNamespace(), SimpleNamespace(emit=AsyncMock(side_effect=ValueError("private-hook")))])
@pytest.mark.asyncio
async def test_optional_hook_absence_or_failure_does_not_fail_generation(hooks):
    provider = AnthropicProvider("offline-placeholder", {"use_streaming": False},
                                 coordinator=SimpleNamespace(hooks=hooks) if hooks else None)
    # Other existing channels require a hook; test optional progress in isolation
    # and the genuine hook-absent generation path separately.
    observation = RequestObservation(hooks, effective_limits(None, None))
    await observation.started()
    await observation.activity()
    await observation.flush()
    provider.coordinator = None
    raw = SimpleNamespace(parse=AsyncMock(return_value=DummyResponse()), headers={})
    provider.client.messages.with_raw_response.create = AsyncMock(return_value=raw)
    await provider.complete(ChatRequest(messages=[Message(role="user", content="fixture")]))
    await provider.close()


def test_limits_are_finite_and_match_actual_phase_override():
    assert effective_limits(None, Timeout(None, connect=5, pool=5)) == {
        "mode": "none", "elapsed_seconds": None, "connect_seconds": 5.0,
        "pool_seconds": 5.0, "read_seconds": None, "write_seconds": None,
    }
    limits = effective_limits(20, Timeout(7, connect=2, pool=3))
    assert limits["mode"] == "elapsed"
    assert limits["elapsed_seconds"] == 20
    assert limits["read_seconds"] == limits["write_seconds"] == 7
    assert limits["connect_seconds"] == 2 and limits["pool_seconds"] == 3
    assert effective_limits(None, Timeout(7))["mode"] == "phase"
    bad = effective_limits(float("nan"), Timeout(float("inf")))
    assert json.dumps(bad, allow_nan=False)
    assert bad["elapsed_seconds"] is None and bad["read_seconds"] is None


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_elapsed_deadline_is_not_published_as_unlimited(value):
    with pytest.raises(ValueError, match="finite positive seconds or null"):
        AnthropicProvider("offline-placeholder", {"timeout": value})


@pytest.mark.asyncio
async def test_supplied_sdk_retries_are_rejected_before_generation_dispatch():
    from amplifier_core.llm_errors import InvalidRequestError
    attempts = []
    async def handle(request):
        attempts.append(request.url.path)
        return transport.Response(500, json={"type": "error", "error": {"type": "api_error"}})
    provider = AnthropicProvider("offline-placeholder", {"use_streaming": False})
    provider._runtime_model_info_cache[provider.default_model] = None
    provider._client = anthropic.AsyncAnthropic(
        api_key="offline-placeholder", max_retries=1,
        http_client=transport.AsyncClient(
            transport=transport.MockTransport(handle), trust_env=False,
        ),
    )
    try:
        with pytest.raises(InvalidRequestError) as caught:
            await provider.complete(ChatRequest(messages=[Message(role="user", content="fixture")]))
        assert caught.value.retryable is False
        assert attempts == []
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_cancel_before_dispatch_has_no_attempt():
    provider = AnthropicProvider("offline-placeholder")
    create = provider.client.messages.stream = MagicMock()
    task = asyncio.create_task(provider.complete(
        ChatRequest(messages=[Message(role="user", content="fixture")])
    ))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert create.call_count == 0
    await provider.close()