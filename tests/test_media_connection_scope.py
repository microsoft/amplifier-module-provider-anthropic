"""Finite SDK-facade scope regressions; real SDK and offline bottom transport."""
import json

import httpx2
import pytest
from anthropic import AsyncAnthropic
from amplifier_core.message_models import ChatRequest, Message
from amplifier_core.llm_errors import InvalidRequestError

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import FakeCoordinator
from tests.test_media_projection import MODEL, guarded_provider, reply, request


def sdk_capture(captured, key="account-A"):
    def handle(req):
        captured.append({"path": req.url.path, "headers": dict(req.headers),
                         "body": json.loads(req.content)})
        return httpx2.Response(200, json=(
            {"input_tokens": 20} if req.url.path.endswith("/count_tokens") else reply()
        ))
    return AsyncAnthropic(api_key=key, base_url="https://selected-a.invalid", max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)))


@pytest.mark.asyncio
async def test_new_facade_does_not_reread_constructor_routing_environment(monkeypatch):
    captured = []
    req = request()
    provider, _ = guarded_provider(req)
    provider._runtime_model_info_cache[MODEL] = None
    client = sdk_capture(captured)
    provider._client = client
    try:
        assert await provider.request_budget(req, context_estimate=100)
        monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Host: wrong.invalid\nx-route-owner: injected")
        await provider.complete(req)
        assert len(captured) == 2
        for call in captured:
            assert call["headers"]["host"] == "selected-a.invalid"
            assert "x-route-owner" not in call["headers"]
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mounted", [False, True])
@pytest.mark.parametrize("cut", ["before_count", "after_count", "request_hook"])
async def test_production_transport_execution_override_never_receives_media(
    monkeypatch, mounted, cut,
):
    physical, observers = [], []

    async def bottom(_transport, wire):
        physical.append(wire.url.path)
        return httpx2.Response(200, json=(
            {"input_tokens": 20} if wire.url.path.endswith("/count_tokens") else reply()
        ))

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", bottom)
    transport = httpx2.AsyncHTTPTransport()
    http = httpx2.AsyncClient(
        trust_env=False,
        **({"mounts": {"https://": transport}} if mounted else {"transport": transport}),
    )
    client = AsyncAnthropic(api_key="account-A", base_url="https://selected-a.invalid",
                            max_retries=0, http_client=http)
    req = request()
    provider, _ = guarded_provider(req)
    provider._runtime_model_info_cache[MODEL] = None
    provider._client = client

    async def observer(wire):
        observers.append(wire.url.path)
        wire.headers["x-api-key"] = "account-B"
        return await bottom(transport, wire)

    emit = provider.coordinator.hooks.emit
    async def hook(name, payload):
        if name == "llm:request":
            transport.handle_async_request = observer
        return await emit(name, payload)

    try:
        if cut == "before_count":
            transport.handle_async_request = observer
            with pytest.raises(InvalidRequestError, match="request not sent"):
                await provider.request_budget(req, context_estimate=100)
        else:
            assert await provider.request_budget(req, context_estimate=100)
            if cut == "after_count":
                transport.handle_async_request = observer
            else:
                provider.coordinator.hooks.emit = hook
            with pytest.raises(InvalidRequestError, match="request not sent"):
                await provider.complete(req)
        assert observers == []
        assert physical == ([] if cut == "before_count" else ["/v1/messages/count_tokens"])
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_count", [False, True])
async def test_closed_original_client_refuses_before_borrowed_transport(after_count):
    captured = []
    req = request()
    provider, _ = guarded_provider(req)
    provider._runtime_model_info_cache[MODEL] = None
    client = sdk_capture(captured)
    provider._client = client
    try:
        if after_count:
            assert await provider.request_budget(req, context_estimate=100)
        await client.close()
        with pytest.raises(InvalidRequestError, match="request not sent"):
            if after_count:
                await provider.complete(req)
            else:
                await provider.request_budget(req, context_estimate=100)
        assert len(captured) == (1 if after_count else 0)
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_count", [False, True])
async def test_http_pool_inner_retry_is_not_a_prepared_media_retry(monkeypatch, after_count):
    physical = []
    async def bottom(_transport, wire):
        physical.append(wire.url.path)
        return httpx2.Response(200, json={"input_tokens": 20})
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", bottom)
    transport = httpx2.AsyncHTTPTransport(retries=0 if after_count else 2)
    client = AsyncAnthropic(api_key="account-A", max_retries=0,
        http_client=httpx2.AsyncClient(transport=transport))
    req = request()
    provider, _ = guarded_provider(req)
    provider._runtime_model_info_cache[MODEL] = None
    provider._client = client
    try:
        if after_count:
            assert await provider.request_budget(req, context_estimate=100)
            transport._pool._retries = 2
        with pytest.raises(InvalidRequestError, match="request not sent"):
            if after_count:
                await provider.complete(req)
            else:
                await provider.request_budget(req, context_estimate=100)
        assert physical == (["/v1/messages/count_tokens"] if after_count else [])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_normal_text_open_then_media_preserves_pool_ownership(monkeypatch):
    physical, closed = [], []
    async def bottom(_transport, wire):
        physical.append(wire.url.path)
        return httpx2.Response(200, json=(
            {"input_tokens": 20} if wire.url.path.endswith("/count_tokens") else reply()
        ))
    original_close = httpx2.AsyncHTTPTransport.aclose
    async def close(transport):
        closed.append(True)
        await original_close(transport)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", bottom)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "aclose", close)
    client = AsyncAnthropic(api_key="account-A", max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.AsyncHTTPTransport()))
    req = request()
    provider, _ = guarded_provider(req)
    provider._runtime_model_info_cache[MODEL] = None
    provider._client = client
    try:
        await client.messages.create(
            model=MODEL, max_tokens=2, messages=[{"role": "user", "content": "plain"}],
        )
        assert not client.is_closed()
        assert await provider.request_budget(req, context_estimate=100)
        await provider.complete(req)
        assert physical == ["/v1/messages", "/v1/messages/count_tokens", "/v1/messages"]
        assert closed == [] and not client.is_closed()
    finally:
        await client.close()
    assert len(closed) == 1


@pytest.mark.asyncio
async def test_response_hook_drift_does_not_relabel_completed_send_as_not_sent():
    captured = []
    req = request()
    provider, _ = guarded_provider(req)
    provider._runtime_model_info_cache[MODEL] = None
    client = sdk_capture(captured)
    provider._client = client
    emit = provider.coordinator.hooks.emit
    async def hook(name, payload):
        if name == "llm:response":
            client.api_key = "account-B"
        return await emit(name, payload)
    provider.coordinator.hooks.emit = hook
    try:
        assert await provider.request_budget(req, context_estimate=100)
        result = await provider.complete(req)
        assert result is not None
        assert len(captured) == 2
        with pytest.raises(InvalidRequestError, match="request not sent"):
            await provider.complete(req)
        assert len(captured) == 2  # Next use refuses; successful send is not replayed.
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_unprotected_nested_other_provider_keeps_its_own_sdk_scope():
    captured_a, captured_b = [], []
    req = request()
    provider, _ = guarded_provider(req)
    provider._runtime_model_info_cache[MODEL] = None
    client_a = sdk_capture(captured_a)
    provider._client = client_a
    other = AnthropicProvider(api_key="account-B", config={
        "default_model": MODEL, "use_streaming": False, "max_retries": 0,
        "enable_prompt_caching": False, "throttle_threshold": 0,
    })
    other.coordinator = FakeCoordinator()
    other._runtime_model_info_cache[MODEL] = None
    client_b = sdk_capture(captured_b, "account-B")
    other._client = client_b
    emit = provider.coordinator.hooks.emit
    async def hook(name, payload):
        if name == "llm:request":
            await other.complete(ChatRequest(messages=[Message(role="user", content="plain nested")]))
        return await emit(name, payload)
    provider.coordinator.hooks.emit = hook
    try:
        assert await provider.request_budget(req, context_estimate=100)
        await provider.complete(req)
        assert len(captured_a) == 2 and len(captured_b) == 1
        assert captured_b[0]["headers"]["x-api-key"] == "account-B"
        assert captured_b[0]["body"]["messages"][0]["content"] == "plain nested"
    finally:
        await client_a.close()
        await client_b.close()


@pytest.mark.asyncio
async def test_frozen_request_effective_routing_headers_apply_to_count_and_send():
    captured = []
    req = request()
    options = {}
    provider, _ = guarded_provider(req, options=options, extra_request_params={
        "extra_headers": {"anthropic-organization-id": "explicit-org"},
    })
    provider._runtime_model_info_cache[MODEL] = None
    client = sdk_capture(captured)
    provider._client = client
    try:
        assert await provider.request_budget(req, context_estimate=100, request_options=options)
        await provider.complete(req, **options)
        assert len(captured) == 2
        assert all(call["headers"]["anthropic-organization-id"] == "explicit-org" for call in captured)
        assert captured[0]["body"]["messages"] == captured[1]["body"]["messages"]
    finally:
        await client.close()
