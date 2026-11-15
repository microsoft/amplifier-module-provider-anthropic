"""Offline request-contract checks, not remote API validation or Mac diagnosis.

Run on each qualified SDK version. These assertions check documented shapes
without claiming that a synthetic response reproduces the historical failure.
"""
import asyncio
import json
import typing

import anthropic
import httpx2
import pytest
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider


@pytest.mark.parametrize("effort", [None, "low"])
def test_actual_sdk_count_serializes_fable_binding_with_adaptive_type(effort):
    calls = []

    def transport(request):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx2.Response(200, json={"input_tokens": 123})

    async def run():
        provider = AnthropicProvider(api_key="synthetic-not-live", config={
            "default_model": "claude-fable-5-2", "max_retries": 0})
        async with anthropic.AsyncAnthropic(api_key="synthetic-not-live", max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(transport), trust_env=False)) as sdk:
            provider._client = sdk
            request = ChatRequest(messages=[Message(role="user", content="synthetic source")],
                                  max_output_tokens=32768, reasoning_effort=effort)
            assert await provider.request_budget(request, context_estimate=1000) is not None

    asyncio.run(run())
    assert len(calls) == 1 and calls[0][0] == "/v1/messages/count_tokens"
    body = calls[0][1]
    assert body["thinking"] == {
        "type": "adaptive",
        "block_binding": {"prefix_mismatch_behavior": "drop_block"},
    }
    assert "max_tokens" not in body and "output_config" not in body
    # Required annotations, not merely field presence: TypedDict runtime
    # __required_keys__ can be empty when annotations are postponed.
    from anthropic.types import ThinkingConfigParam
    variants = typing.get_args(ThinkingConfigParam)
    assert variants and all(
        typing.get_origin(typing.get_type_hints(v, include_extras=True)["type"]) is typing.Required
        for v in variants
    )


@pytest.mark.parametrize("effort", [None, "low"])
def test_automatic_binding_count_and_send_share_explicit_adaptive_contract(effort):
    """Validate the documented discriminated shape, not historical API causation."""
    from copy import deepcopy
    from amplifier_module_provider_anthropic._thinking import BINDING_BETA

    calls = []

    def transport(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body, request.headers))
        assert body["thinking"] == {
            "type": "adaptive",
            "block_binding": {"prefix_mismatch_behavior": "drop_block"},
        }
        assert BINDING_BETA in request.headers["anthropic-beta"].split(",")
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 123})
        assert request.url.path == "/v1/messages"
        return httpx2.Response(200, json={
            "id": "msg_synthetic_binding", "type": "message", "role": "assistant",
            "model": "claude-fable-5-2",
            "content": [{"type": "text", "text": "Synthetic completion"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 123, "output_tokens": 3},
        })

    async def run():
        provider = AnthropicProvider(api_key="synthetic-not-live", config={
            "default_model": "claude-fable-5-2", "max_retries": 0,
            "use_streaming": False, "enable_prompt_caching": False,
        })
        async with anthropic.AsyncAnthropic(
            api_key="synthetic-not-live", max_retries=0,
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(transport), trust_env=False),
        ) as sdk:
            provider._client = sdk
            request = ChatRequest(messages=[
                Message(role="user", content="Synthetic unchanged source")],
                max_output_tokens=32768, reasoning_effort=effort)
            original = deepcopy(request.model_dump())
            decision = await provider.request_budget(request, context_estimate=1000)
            assert decision is not None
            assert decision["measurement"]["input_tokens"] == 123
            response = await provider.complete(request)
            assert response.finish_reason == "end_turn"
            assert request.model_dump() == original

    asyncio.run(run())
    assert [path for path, _, _ in calls] == [
        "/v1/messages/count_tokens", "/v1/messages"]
    count, sent = calls[0][1], calls[1][1]
    assert count["thinking"] == sent["thinking"]
    assert count["messages"] == sent["messages"]
    assert sent["max_tokens"] == 32768 and "max_tokens" not in count
