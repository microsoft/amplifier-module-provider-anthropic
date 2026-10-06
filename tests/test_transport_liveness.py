"""Real SDK, loopback HTTP/SSE receiving-boundary checks; no accounts/inference.

ANTHROPIC_WAIT_SOAK_SECONDS=2100 opts into actual long silence (outside CI).
Run in a loopback-only network namespace for kernel-enforced egress isolation.
Fixtures are in-process and close every connection/task on exit.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import anthropic
import pytest
from anthropic import _base_client
from amplifier_core.llm_errors import LLMTimeoutError
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._request_safety import (
    UNKNOWN_MESSAGE, RequestOutcomeUnknownError,
)
from tests._helpers import FakeCoordinator


transport = getattr(_base_client, "httpx2", None) or _base_client.httpx
MODEL = "claude-sonnet-5-5"
PRIVATE = "private-fixture-sentinel"


def message():
    return {
        "id": "msg_fixture", "type": "message", "role": "assistant",
        "model": MODEL, "content": [], "stop_reason": "end_turn",
        "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def sse(event):
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def start_event():
    body = message()
    body["stop_reason"] = None
    return sse({"type": "message_start", "message": body})


def finish_events():
    return (
        sse({"type": "message_delta",
             "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": 1}})
        + sse({"type": "message_stop"})
    )


class Wire:
    def __init__(self, mode, streaming):
        self.mode = mode
        self.streaming = streaming
        self.posts = 0
        self.accepted = 0
        self.count_reads = 0
        self.remote_finished = 0
        self.entered = asyncio.Event()
        self.sent = asyncio.Event()
        self.release = asyncio.Event()
        self.tasks = set()
        self.extensions = []

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            path = headers.split(b" ")[1].split(b"?")[0]
            if path.startswith(b"/v1/models"):
                await self.json_reply(writer, 200, {
                    "data": [], "has_more": False, "first_id": None, "last_id": None,
                })
                return
            length = next((int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                           if line.lower().startswith(b"content-length:")), 0)
            await reader.readexactly(length)
            if path.endswith(b"/count_tokens"):
                self.count_reads += 1
                await self.json_reply(writer, 200, {"input_tokens": 1})
                return
            assert path == b"/v1/messages"
            self.posts += 1
            if self.mode.startswith("refuse") and self.posts == 1:
                status = int(self.mode[-3:])
                kind = "rate_limit_error" if status == 429 else "overloaded_error"
                self.entered.set()
                await self.json_reply(writer, status, {
                    "type": "error", "error": {"type": kind, "message": PRIVATE},
                }, extra=b"retry-after: 0.01\r\n")
                return
            self.accepted += 1
            self.entered.set()
            if self.mode == "reset_before":
                writer.transport.abort()
                return
            if self.mode == "redirect":
                await self.json_reply(
                    writer, 307, {"type": "error"},
                    extra=b"location: /v1/messages\r\n",
                )
                return
            if self.mode in {"500", "bare429", "bare529"}:
                status = 500 if self.mode == "500" else int(self.mode[-3:])
                await self.json_reply(writer, status, {
                    "type": "error", "error": {"type": "api_error", "message": PRIVATE},
                })
                return
            if self.streaming:
                writer.write(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                    b"connection: close\r\n\r\n"
                )
                if self.mode in {"controls", "reset_after", "eof_after", "malformed", "burst", "sse_refusal"}:
                    writer.write(start_event())
                if self.mode == "ping":
                    writer.write(b": private-fixture-sentinel\n\nevent: ping\ndata: {}\n\n")
                await writer.drain()
                self.sent.set()
                if self.mode == "reset_after":
                    writer.transport.abort()
                    return
                if self.mode in {"eof_before", "eof_after"}:
                    return
                if self.mode == "malformed":
                    writer.write(b"event: message_delta\ndata: {broken\n\n")
                    await writer.drain()
                    return
                if self.mode == "sse_refusal":
                    writer.write(sse({"type": "error", "error": {
                        "type": "overloaded_error", "message": PRIVATE,
                    }}))
                    await writer.drain()
                    return
                if self.mode == "burst":
                    writer.write(sse({
                        "type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }))
                    for _ in range(200):
                        writer.write(sse({
                            "type": "content_block_delta", "index": 0,
                            "delta": {"type": "text_delta", "text": PRIVATE},
                        }))
                    writer.write(sse({"type": "content_block_stop", "index": 0}))
                if self.mode in {"hold", "controls", "ping"}:
                    await self.release.wait()
                self.remote_finished += 1
                if self.mode not in {"controls", "burst"}:
                    writer.write(start_event())
                writer.write(finish_events())
                await writer.drain()
            else:
                self.sent.set()
                if self.mode in {"hold", "controls", "ping"}:
                    await self.release.wait()
                self.remote_finished += 1
                if self.mode.startswith("eof") or self.mode.startswith("reset"):
                    writer.transport.abort()
                    return
                if self.mode == "malformed":
                    await self.json_reply(writer, 200, b"{broken")
                elif self.mode == "incomplete":
                    await self.json_reply(writer, 200, {"stop_reason": "end_turn"})
                else:
                    await self.json_reply(writer, 200, message())
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self.tasks.discard(task)

    async def json_reply(self, writer, status, body, extra=b""):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        writer.write(
            f"HTTP/1.1 {status} Fixture\r\ncontent-type: application/json\r\n"
            f"content-length: {len(data)}\r\nconnection: close\r\n".encode()
            + extra + b"\r\n" + data
        )
        await writer.drain()


@asynccontextmanager
async def wire_call(mode="hold", streaming=True, **config):
    wire = Wire(mode, streaming)
    server = await asyncio.start_server(wire.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    hooks = FakeCoordinator()
    provider = AnthropicProvider(
        "offline-fixture-placeholder",
        {"default_model": MODEL, "use_streaming": streaming, "max_retries": 3,
         "min_retry_delay": 0.01, "retry_jitter": False, "max_concurrent_requests": 0,
         **config},
        coordinator=hooks,
    )
    class RecordingTransport(transport.AsyncBaseTransport):
        def __init__(self):
            self.inner = transport.AsyncHTTPTransport(retries=0)

        async def handle_async_request(self, request):
            if request.url.path.endswith("/messages"):
                wire.extensions.append(dict(request.extensions["timeout"]))
            return await self.inner.handle_async_request(request)

        async def aclose(self):
            await self.inner.aclose()

    provider._client = anthropic.AsyncAnthropic(
        api_key="offline-fixture-placeholder",
        base_url=f"http://127.0.0.1:{port}", max_retries=0,
        http_client=transport.AsyncClient(
            transport=RecordingTransport(), trust_env=False,
        ),
    )
    request = ChatRequest(messages=[Message(role="user", content=PRIVATE)])
    try:
        yield wire, provider, request, hooks.hooks
    finally:
        wire.release.set()
        await provider.close()
        server.close()
        await server.wait_closed()
        if wire.tasks:
            await asyncio.wait_for(asyncio.gather(*tuple(wire.tasks)), 2)
        print(json.dumps({
            "fixture": mode, "streaming": streaming,
            "generation_posts": wire.posts, "accepted_markers": wire.accepted,
            "count_reads": wire.count_reads, "remote_finished": wire.remote_finished,
            "sdk_phases": wire.extensions,
        }))


def progress(hooks):
    return [data for event, data in hooks.events if event == "llm:progress"]


def assert_unknown(error):
    assert error.retryable is False
    assert error.request_outcome == "unknown"
    assert error.effects == "may_have_occurred"
    assert str(error) == UNKNOWN_MESSAGE
    assert error.__cause__ is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_healthy_silence_then_success(streaming):
    async with wire_call(streaming=streaming) as (wire, provider, request, hooks):
        task = asyncio.create_task(provider.complete(request))
        await asyncio.wait_for(wire.entered.wait(), 2)
        await asyncio.sleep(0.03)
        assert not task.done()
        assert len(progress(hooks)) == 1
        wire.release.set()
        result = await asyncio.wait_for(task, 2)
        assert result.finish_reason == "end_turn"
        assert wire.posts == wire.accepted == 1
        assert wire.extensions[-1] == {
            "connect": 5.0, "pool": 5.0, "read": None, "write": None,
        }
        assert progress(hooks)[0]["limits"]["mode"] == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_accepted_deadline_never_replays(streaming):
    async with wire_call(streaming=streaming, timeout=0.08) as (wire, provider, request, hooks):
        with pytest.raises(LLMTimeoutError) as caught:
            await provider.complete(request)
        assert_unknown(caught.value)
        assert wire.posts == wire.accepted == 1
        assert progress(hooks)[0]["limits"]["elapsed_seconds"] == 0.08
        assert wire.extensions[-1]["read"] == 0.08
        assert wire.extensions[-1]["connect"] == 5
        wire.release.set()  # Remote fixture may finish after local timeout.


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mode", [
    "reset_before", "reset_after", "eof_before", "eof_after", "malformed", "500",
    "bare429", "bare529", "redirect",
])
async def test_accepted_fault_never_replays(mode, streaming):
    async with wire_call(mode, streaming) as (wire, provider, request, hooks):
        with pytest.raises(RequestOutcomeUnknownError) as caught:
            await asyncio.wait_for(provider.complete(request), 2)
        assert_unknown(caught.value)
        assert wire.posts == wire.accepted == 1
        assert PRIVATE not in json.dumps(progress(hooks))
        assert not any(name == "provider:retry" for name, _ in hooks.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 529])
@pytest.mark.parametrize("streaming", [False, True])
async def test_documented_http_refusal_retries(status, streaming):
    async with wire_call(f"refuse{status}", streaming) as (wire, provider, request, hooks):
        await asyncio.wait_for(provider.complete(request), 2)
        assert wire.posts == 2 and wire.accepted == 1
        assert [p["attempt"] for p in progress(hooks)
                if p["observation"] == "attempt_started"] == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["hold", "controls"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_cancel_locally_remote_may_finish(mode, streaming):
    async with wire_call(mode, streaming) as (wire, provider, request, hooks):
        task = asyncio.create_task(provider.complete(request))
        await asyncio.wait_for(wire.sent.wait(), 2)
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert wire.posts == wire.accepted == 1
        wire.release.set()
        await asyncio.sleep(0.02)
        assert wire.remote_finished == 1


@pytest.mark.asyncio
async def test_cancel_backoff_never_dispatches_replacement():
    async with wire_call("refuse429", min_retry_delay=5) as (wire, provider, request, hooks):
        task = asyncio.create_task(provider.complete(request))
        for _ in range(100):
            if any(name == "provider:retry" for name, _ in hooks.events):
                break
            await asyncio.sleep(0.01)
        assert any(name == "provider:retry" for name, _ in hooks.events)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert wire.posts == 1 and wire.accepted == 0


@pytest.mark.asyncio
async def test_control_event_without_text_and_invisible_ping():
    for mode in ("controls", "ping"):
        async with wire_call(mode) as (wire, provider, request, hooks):
            task = asyncio.create_task(provider.complete(request))
            await asyncio.wait_for(wire.sent.wait(), 2)
            await asyncio.sleep(0.03)
            assert not task.done()
            activity = [p for p in progress(hooks) if p["observation"] == "response_activity"]
            assert len(activity) == (1 if mode == "controls" else 0)
            assert not any(name == "llm:stream_block_delta" for name, _ in hooks.events)
            wire.release.set()
            await task
            assert PRIVATE not in json.dumps(progress(hooks))


@pytest.mark.asyncio
async def test_burst_is_throttled_and_terminal_latest_is_retained():
    async with wire_call("burst") as (wire, provider, request, hooks):
        await provider.complete(request)
        observations = progress(hooks)
        assert len(observations) == 3  # start, initial activity, terminal flush
        assert PRIVATE not in json.dumps(observations)
        for payload in observations:
            assert set(payload) == {"version", "observation", "attempt", "limits"}
            assert set(payload["limits"]) == {
                "mode", "elapsed_seconds", "connect_seconds", "pool_seconds",
                "read_seconds", "write_seconds",
            }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,streaming", [("sse_refusal", True), ("incomplete", False)])
async def test_http200_is_not_a_proved_refusal_or_complete_envelope(mode, streaming):
    async with wire_call(mode, streaming) as (wire, provider, request, hooks):
        with pytest.raises(RequestOutcomeUnknownError) as caught:
            await provider.complete(request)
        assert_unknown(caught.value)
        assert wire.posts == wire.accepted == 1
        assert PRIVATE not in json.dumps(progress(hooks))


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("override", [None, 0.08, "phases"])
async def test_actual_sdk_phase_overrides_do_not_invent_elapsed_policy(streaming, override):
    timeout = (anthropic.Timeout(None, connect=2, pool=3, write=4)
               if override == "phases" else override)
    async with wire_call(
        streaming=streaming, extra_request_params={"timeout": timeout},
    ) as (wire, provider, request, hooks):
        task = asyncio.create_task(provider.complete(request))
        await asyncio.wait_for(wire.entered.wait(), 2)
        wire.release.set()
        await task
        limits = progress(hooks)[0]["limits"]
        assert limits["elapsed_seconds"] is None
        assert limits["mode"] == ("none" if override is None else "phase")
        for phase, value in wire.extensions[-1].items():
            assert limits[f"{phase}_seconds"] == value


@pytest.mark.asyncio
async def test_concurrent_real_sdk_calls_keep_context_local_attempts():
    import contextvars
    call = contextvars.ContextVar("call")
    receipts = []
    async with wire_call("controls") as (wire, provider, request, hooks):
        original_emit = hooks.emit
        async def emit(name, data):
            if name == "llm:progress":
                receipts.append((call.get(), data))
            await original_emit(name, data)
        hooks.emit = emit
        async def complete(label):
            call.set(label)
            return await provider.complete(request.model_copy(deep=True))
        tasks = [asyncio.create_task(complete(label)) for label in ("root", "delegate")]
        for _ in range(200):
            if all(any(owner == label and p["observation"] == "response_activity"
                       for owner, p in receipts) for label in ("root", "delegate")):
                break
            await asyncio.sleep(0.01)
        assert wire.posts == wire.accepted == 2
        for label in ("root", "delegate"):
            local = [p for owner, p in receipts if owner == label]
            assert [p["observation"] for p in local] == ["attempt_started", "response_activity"]
            assert all(p["attempt"] == 1 for p in local)
        wire.release.set()
        await asyncio.gather(*tasks)
        assert PRIVATE not in json.dumps(receipts)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_optional_real_long_silence(streaming):
    seconds = float(os.environ.get("ANTHROPIC_WAIT_SOAK_SECONDS", "0"))
    if seconds <= 0:
        pytest.skip("one-off real soak is manager-owned, opt in explicitly")
    async with wire_call(streaming=streaming) as (wire, provider, request, hooks):
        task = asyncio.create_task(provider.complete(request))
        await asyncio.wait_for(wire.entered.wait(), 2)
        started = time.monotonic()
        await asyncio.sleep(seconds)
        assert not task.done() and len(progress(hooks)) == 1
        wire.release.set()
        await asyncio.wait_for(task, 5)
        assert wire.posts == 1
        print(json.dumps({"actual_silence_seconds": time.monotonic() - started}))