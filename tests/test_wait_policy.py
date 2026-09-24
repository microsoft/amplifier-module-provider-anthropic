"""Default model waits survive virtual elapsed time; cancellation/errors remain live."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest
from amplifier_core.llm_errors import LLMError, LLMTimeoutError
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import DummyResponse, FakeCoordinator


def setup_call(streaming, config=None, failure=None):
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": streaming,
            "max_retries": 0,
            "default_model": "claude-sonnet-4-5-20250929",
            **(config or {}),
        },
        coordinator=FakeCoordinator(),
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    closed = []

    async def wait():
        entered.set()
        await release.wait()
        if failure:
            raise failure

    class Stream:
        response = SimpleNamespace(headers={})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)

        def __aiter__(self):
            return self

        async def __anext__(self):
            await wait()
            raise StopAsyncIteration

        async def get_final_message(self):
            return DummyResponse()

    client = MagicMock()
    provider._client = client
    if streaming:
        call = client.messages.stream = MagicMock(return_value=Stream())
    else:

        async def create(**kwargs):
            await wait()
            return SimpleNamespace(
                parse=AsyncMock(return_value=DummyResponse()), headers={}
            )

        call = client.messages.with_raw_response.create = AsyncMock(side_effect=create)

    request = ChatRequest(messages=[Message(role="user", content="hello")])
    return provider, request, entered, release, closed, call


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_default_wait_survives_an_hour_then_completes(monkeypatch, streaming):
    provider, request, entered, release, closed, call = setup_call(streaming)
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    loop = asyncio.get_running_loop()
    clock = loop.time
    # No real sleeping: move beyond every former model-work deadline while
    # the mock provider remains healthy but silent.
    with monkeypatch.context() as patch:
        patch.setattr(loop, "time", lambda: clock() + 3600)
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done()
    release.set()
    await task
    timeout = call.call_args.kwargs["timeout"]
    assert timeout.read is None
    assert timeout.connect == 5.0
    assert timeout.pool == 5.0
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_explicit_cancellation_propagates_without_retry(streaming):
    provider, request, entered, _release, closed, call = setup_call(streaming)
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert call.call_count == 1
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_explicit_deadline_still_stops_model_work(monkeypatch, streaming):
    provider, request, entered, _release, closed, call = setup_call(
        streaming, {"timeout": 10}
    )
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    loop = asyncio.get_running_loop()
    clock = loop.time
    with monkeypatch.context() as patch:
        patch.setattr(loop, "time", lambda: clock() + 11)
        with pytest.raises(LLMTimeoutError):
            await task
    assert call.call_args.kwargs["timeout"].read == 10
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_transport_failure_is_still_reported(streaming):
    provider, request, entered, release, closed, call = setup_call(
        streaming, failure=anthropic.APIConnectionError(request=MagicMock())
    )
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    release.set()
    with pytest.raises(LLMError):
        await task
    assert call.call_count == 1
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_actual_sdk_request_disables_hidden_read_deadline(streaming):
    import json

    from anthropic import _base_client

    sdk_module = anthropic
    body = {
        "id": "msg_wait",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    events = [
        {"type": "message_start", "message": body},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    stream_body = "".join(
        "event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n"
        for event in events
    )
    provider = AnthropicProvider(
        api_key="fixture",
        config={
            "default_model": body["model"],
            "use_streaming": streaming,
            "max_retries": 0,
        },
    )
    provider.coordinator = FakeCoordinator()
    client_type = sdk_module.AsyncAnthropic

    transport = getattr(_base_client, "httpx2", None) or _base_client.httpx
    seen = []

    async def handle(request):
        seen.append(request)
        if streaming:
            return transport.Response(
                200, text=stream_body, headers={"content-type": "text/event-stream"}
            )
        return transport.Response(200, json=body)

    provider._client = client_type(
        api_key="fixture",
        timeout=0.001,
        http_client=transport.AsyncClient(transport=transport.MockTransport(handle)),
    )
    try:
        await provider.complete(
            ChatRequest(messages=[Message(role="user", content="hello")])
        )
        seen = [request for request in seen if request.url.path.endswith("/messages")]
        assert len(seen) == 1
        assert seen[0].extensions["timeout"] == {
            "read": None,
            "write": None,
            "connect": 5.0,
            "pool": 5.0,
        }
        assert "timeout" not in json.loads(seen[0].content)
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("override", [None, 30])
async def test_explicit_extra_request_timeout_is_preserved(streaming, override):
    provider, request, entered, release, _, call = setup_call(
        streaming, {"extra_request_params": {"timeout": override}}
    )
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    release.set()
    await task
    assert "timeout" in call.call_args.kwargs
    assert call.call_args.kwargs["timeout"] == override
