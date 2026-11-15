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
