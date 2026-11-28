"""Connection classification with the real SDK and local-only transports."""

import asyncio
import socket
import ssl
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import httpx2 as httpx
import pytest
from amplifier_core import llm_errors as errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_anthropic import AnthropicProvider


@pytest.fixture(autouse=True)
def local_only_sockets(monkeypatch):
    connect = socket.socket.connect

    def guarded(sock, address):
        if sock.family != getattr(socket, "AF_UNIX", None):
            assert address[0] == "127.0.0.1", "non-local test connection refused"
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


class Hooks:
    def __init__(self):
        self.events = []

    async def emit(self, event, data):
        self.events.append((event, data))


@asynccontextmanager
async def endpoint(mode):
    tasks = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            await reader.read(65536)
            if mode == "no_answer":
                await asyncio.Event().wait()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    if mode == "refused":
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        # Darwin can blackhole a bound, non-listening socket. Close it before
        # connecting so this row measures refusal rather than an OS timeout.
        yield f"http://127.0.0.1:{port}"
        return
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        server.close()
        pending = list(tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await server.wait_closed()


def provider(url, streaming=False, **config):
    p = AnthropicProvider(
        api_key="not-a-real-key",
        config={
            "base_url": url, "use_streaming": streaming, "max_retries": 0,
            "max_concurrent_requests": 0, "rate_limit_state_path": "",
            "persist_fallback_state": False, "fallback_on_overload": False,
            **config,
        },
    )
    # Static fallback metadata, not a synthetic vendor discovery claim. The
    # matrix isolates the Messages send, rather than its optional Models probe.
    p._runtime_model_info_cache[p.default_model] = None
    return p


def request():
    return ChatRequest(messages=[Message(role="user", content="hello")])


def failure_transport(case):
    def send(req):
        if case == "dns":
            raise httpx.ConnectError("DNS failed", request=req) from socket.gaierror(-2, "test DNS")
        if case == "tls":
            raise httpx.ConnectError("TLS failed", request=req) from ssl.SSLError("test TLS")
        if case == "sdk_timeout":
            raise httpx.ReadTimeout("SDK read timed out", request=req)
        if case == "invalid_url":
            raise httpx.InvalidURL("invalid URL")
        status = {"5xx": 500, "overloaded": 529, "rate_limit": 429, "auth": 401}[case]
        return httpx.Response(status, json={"type": "error", "error": {
            "message": "test failure", "type": "overloaded_error" if status == 529 else "api_error",
        }}, request=req)
    return httpx.MockTransport(send)


@asynccontextmanager
async def failing_provider(case, streaming=False, **config):
    if case in {"refused", "closed", "module_timeout", "sdk_no_answer"}:
        mode = "no_answer" if case in {"module_timeout", "sdk_no_answer"} else case
        async with endpoint(mode) as url:
            p = provider(url, streaming, timeout=0.1 if case == "module_timeout" else 2, **config)
            if case == "sdk_no_answer":
                p._sdk_timeout = anthropic.Timeout(0.1)
            try:
                yield p
            finally:
                await p.close()
        return
    url = "ftp://127.0.0.1" if case == "unsupported_protocol" else "http://127.0.0.1:9"
    p = provider(url, streaming, **config)
    if case != "unsupported_protocol":
        p._client = anthropic.AsyncAnthropic(
            api_key="not-a-real-key", base_url=url, max_retries=0,
            http_client=httpx.AsyncClient(transport=failure_transport(case)),
        )
    try:
        yield p
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_error", [httpx.ReadError, httpx.ReadTimeout])
async def test_real_sdk_interrupted_answer_keeps_prior_class(transport_error):
    original = transport_error("peer lost")

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (
                b'event: message_start\ndata: {"type":"message_start","message":'
                b'{"id":"msg-local","type":"message","role":"assistant","content":[],'
                b'"model":"claude-sonnet-5-5","usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
                b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
                b'"content_block":{"type":"text","text":""}}\n\n'
                b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
                b'"delta":{"type":"text_delta","text":"partial"}}\n\n'
            )
            raise original

    def send(req):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Body(), request=req)

    p = provider("http://127.0.0.1:9", streaming=True)
    p._client = anthropic.AsyncAnthropic(
        api_key="not-a-real-key", base_url="http://127.0.0.1:9", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(send)),
    )
    hooks = Hooks()
    p.coordinator = SimpleNamespace(hooks=hooks)
    try:
        with pytest.raises(errors.LLMError) as caught:
            await p.complete(request())
        assert type(caught.value) is errors.LLMError
        assert caught.value.__cause__ is original
        assert any(event == "llm:stream_block_delta" for event, _ in hooks.events)
        assert any(event == "llm:stream_aborted" for event, _ in hooks.events)
    finally:
        await p.close()


CASES = {
    "refused": errors.NetworkError,
    "closed": errors.NetworkError,
    "dns": errors.NetworkError,
    "tls": errors.NetworkError,
    "sdk_timeout": errors.LLMTimeoutError,
    "sdk_no_answer": errors.LLMTimeoutError,
    "module_timeout": errors.LLMTimeoutError,
    "unsupported_protocol": errors.LLMError,
    "invalid_url": errors.LLMError,
    "5xx": errors.ProviderUnavailableError,
    "overloaded": errors.ProviderUnavailableError,
    "rate_limit": errors.RateLimitError,
    "auth": errors.AuthenticationError,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["complete", "stream"])
@pytest.mark.parametrize("case,expected", CASES.items())
async def test_failure_matrix(case, expected, streaming):
    async with failing_provider(case, streaming) as p:
        hooks = Hooks()
        p.coordinator = SimpleNamespace(hooks=hooks)
        with pytest.raises(Exception) as caught:
            await p.complete(request())
        error = caught.value
        assert type(error) is expected
        assert error.retryable is (case != "auth")
        assert error.provider == "anthropic"
        assert error.model == p.default_model
        assert error.__cause__ is not None
        assert error.status_code == {"5xx": 500, "overloaded": 529, "rate_limit": 429, "auth": 401}.get(case)
        events = [data for name, data in hooks.events if name == "llm:response"]
        assert len(events) == 1
        assert set(events[0]) == {"provider", "model", "status", "duration_ms", "error"}
        assert events[0]["error"] == str(error)
        if case in {"refused", "closed", "dns", "tls", "unsupported_protocol"}:
            assert isinstance(error.__cause__, anthropic.APIConnectionError)
            cause = error.__cause__.__cause__
            assert cause is not None
            assert str(error).startswith(str(error.__cause__))
            assert type(cause).__name__ in str(error)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["dns", "tls", "sdk_timeout", "unsupported_protocol", "invalid_url", "5xx", "overloaded", "rate_limit", "auth"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_retry_count_and_delays_unchanged(case, streaming, monkeypatch):
    sleeps = AsyncMock()
    monkeypatch.setattr("amplifier_core.utils.retry.asyncio.sleep", sleeps)
    async with failing_provider(case, streaming, max_retries=2, min_retry_delay=0.01, retry_jitter=False) as p:
        hooks = Hooks()
        p.coordinator = SimpleNamespace(hooks=hooks)
        with pytest.raises(errors.LLMError):
            await p.complete(request())
    retries = [data for name, data in hooks.events if name == "provider:retry"]
    assert [data["attempt"] for data in retries] == ([] if case == "auth" else [1, 2])
    # Anthropic overloads already use a 10x delay multiplier.
    expected_delays = [0.1, 0.2] if case == "overloaded" else [0.01, 0.02]
    assert [call.args[0] for call in sleeps.await_args_list] == ([] if case == "auth" else expected_delays)


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk_error", [anthropic.APIConnectionError, anthropic.APITimeoutError])
@pytest.mark.parametrize("with_hooks", [False, True])
async def test_stream_failure_after_response_keeps_prior_class(sdk_error, with_hooks):
    p = provider("http://127.0.0.1:9", streaming=True)
    original = sdk_error(request=httpx.Request("POST", "http://127.0.0.1:9/v1/messages"))
    original.__cause__ = httpx.ConnectError("peer lost")

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def __aiter__(self):
            return self.events()

        async def events(self):
            yield SimpleNamespace(type="message_start")
            raise original

    p.client.messages.stream = lambda **kwargs: Stream()
    if with_hooks:
        p.coordinator = SimpleNamespace(hooks=Hooks())
    try:
        with pytest.raises(errors.LLMError) as caught:
            await p.complete(request())
        assert type(caught.value) is errors.LLMError
        assert caught.value.retryable is True
        assert caught.value.__cause__ is original
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("transport_error", [httpx.ReadError, httpx.ReadTimeout])
async def test_failure_reading_response_body_is_not_promoted(streaming, transport_error):
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            if not streaming:
                yield b'{"id":"partial-message"'
            raise transport_error("response body lost")

    def send(req):
        return httpx.Response(200, stream=Body(), request=req)

    p = provider("http://127.0.0.1:9", streaming)
    p._client = anthropic.AsyncAnthropic(
        api_key="not-a-real-key", base_url="http://127.0.0.1:9", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(send)),
    )
    try:
        with pytest.raises(errors.LLMError) as caught:
            await p.complete(request())
        assert type(caught.value) is errors.LLMError
        assert caught.value.retryable is True
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_response_observation_is_task_and_attempt_local():
    from amplifier_module_provider_anthropic._network_errors import (
        _received_headers, observe_response, response_received,
    )
    ready, done = asyncio.Event(), asyncio.Event()

    async def received():
        with observe_response():
            await asyncio.create_task(_received_headers(None))
            ready.set()
            await done.wait()
            assert response_received()
        assert not response_received()

    async def not_received():
        await ready.wait()
        with observe_response():
            assert not response_received()
            done.set()

    await asyncio.gather(received(), not_received())