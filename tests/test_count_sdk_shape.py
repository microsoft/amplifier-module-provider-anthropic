"""Offline serialization observations, not remote API validation or Mac diagnosis.

Run on each qualified SDK version. These assertions deliberately retain the
current envelope, even where its server schema acceptance is not established.
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
def test_actual_sdk_count_serializes_fable_binding_without_synthesizing_type(effort):
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
    assert body["thinking"] == {"block_binding": {"prefix_mismatch_behavior": "drop_block"}}
    assert "max_tokens" not in body and "output_config" not in body
    # SDK serialization preserves this expert/beta extension; it does not
    # validate a missing discriminator before MockTransport. Server acceptance
    # cannot be inferred from our synthetic200 response.
    from anthropic.types import ThinkingConfigParam
    variants = typing.get_args(ThinkingConfigParam)
    assert variants and all("type" in typing.get_type_hints(v) for v in variants)
