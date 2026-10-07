"""Offline Haiku 5.5 contracts; synthetic signatures prove storage, not replay."""

import asyncio
import json
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from anthropic import AsyncAnthropic
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolSpec,
)

from amplifier_module_provider_anthropic import AnthropicProvider, _RuntimeModelInfo
from amplifier_module_provider_anthropic._cost import compute_cost
from tests._helpers import FakeCoordinator

MODEL = "claude-haiku-5-5"


def provider(**config):
    p = AnthropicProvider(
        api_key="offline-placeholder",
        config={
            "default_model": MODEL,
            "enable_prompt_caching": False,
            "max_retries": 0,
            "use_streaming": False,
            **config,
        },
    )
    p.coordinator = FakeCoordinator()
    return p


def request(**overrides):
    return ChatRequest(
        **{"messages": [Message(role="user", content="hello")], **overrides}
    )


def assemble(p=None, req=None, **options):
    p = p or provider()
    result = p._assemble_request_params(
        req or request(),
        request_options={"model": p.default_model, **options},
        request_caps=p._get_capabilities(options.get("model", p.default_model)),
    )
    assert result is not None
    return result.params


def tool():
    return ToolSpec(
        name="lookup",
        description="ordinary function",
        parameters={"type": "object", "properties": {}},
    )


def test_exact_capabilities_and_overlay_preserve_static_controls():
    p = provider()
    caps = p._get_capabilities(MODEL)
    assert caps.max_output_tokens == 128_000
    assert caps.supports_1m and caps.supports_adaptive_thinking
    assert not caps.supports_manual_thinking
    assert not caps.supports_sampling
    assert caps.supports_output_config
    assert caps.supported_efforts == ("low", "medium", "high", "xhigh", "max")
    assert caps.min_cacheable_tokens == 512
    assert caps.default_thinking_budget == 0
    assert not caps.requires_adaptive_thinking and not caps.thinking_always_on
    assert caps.supports_forced_tool_choice
    assert caps.computer_use_tool_type == "computer_toolset_20260801"
    overlay = p._apply_runtime_capability_overrides(
        caps, _RuntimeModelInfo(max_input_tokens=1_000_000, max_tokens=128_000)
    )
    for key in (
        "supports_manual_thinking",
        "supports_sampling",
        "supported_efforts",
        "min_cacheable_tokens",
        "supports_forced_tool_choice",
        "computer_use_tool_type",
        "default_thinking_budget",
    ):
        assert getattr(overlay, key) == getattr(caps, key)
    assert p.get_info().defaults["context_window"] == 200_000
    assert (
        provider(enable_1m_context=True).get_info().defaults["context_window"]
        == 1_000_000
    )
    old = p._get_capabilities("claude-haiku-4-5")
    assert old.max_output_tokens == 64_000 and old.base_context_window == 200_000
    assert old.supports_manual_thinking and old.supports_sampling
    assert not old.supports_1m and not old.supports_adaptive_thinking
    assert old.default_thinking_budget == 32_000 and old.min_cacheable_tokens == 4096
    assert old.computer_use_tool_type == "computer_20250124"
    assert p._budget_capabilities_for("claude-haiku-5-5-20261007") is None


@pytest.mark.parametrize("filtered", [True, False])
def test_numeric_discovery_and_saved_pins(filtered):
    p = provider(filtered=filtered)
    ids = ["claude-haiku-4-5", MODEL, "claude-haiku-5-9", "claude-haiku-5-10"]
    p.client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(id=id, display_name=id, created_at="") for id in ids]
        )
    )
    models = asyncio.run(p.list_models())
    assert {m.id for m in models} == ({"claude-haiku-5-10"} if filtered else set(ids))
    assert p._family_latest["haiku"] == "claude-haiku-5-10"
    assert p.default_model == MODEL
    assert AnthropicProvider(api_key="offline").default_model == "claude-sonnet-5-5"
    assert (
        provider(default_model="claude-haiku-4-5").default_model == "claude-haiku-4-5"
    )
    asyncio.run(p.close())


def test_default_adaptive_medium_and_no_sampling():
    params = assemble(req=request(temperature=0.1))
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "output_config" not in params  # omission retains vendor medium, not high
    assert params["max_tokens"] == 128_000
    assert not ({"temperature", "top_p", "top_k"} & params.keys())
    assert not ({"temperature", "top_p", "top_k"} & params.get("extra_body", {}).keys())


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_effort_pins_reach_wire_without_budget(effort):
    params = assemble(provider(reasoning_effort=effort), request(max_output_tokens=123))
    assert params["thinking"]["type"] == "adaptive"
    assert "budget_tokens" not in params["thinking"]
    assert params["output_config"] == {"effort": effort}
    assert params["max_tokens"] == 123


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_real_disabled_preserves_effort(effort):
    params = assemble(provider(extended_thinking=False, reasoning_effort=effort))
    assert params["thinking"] == {"type": "disabled"}
    assert params["output_config"] == {"effort": effort}


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_disabled_high_efforts_fail_locally(effort):
    with pytest.raises(
        InvalidRequestError, match=rf"{effort}.*disabled|disabled.*{effort}"
    ):
        assemble(provider(extended_thinking=False, reasoning_effort=effort))


@pytest.mark.parametrize(
    "config",
    [
        {"thinking_type": "enabled"},
        {"thinking_type": "between_tools"},
        {"thinking_budget_tokens": 4096},
        {"thinking_budget_tokens": "bad"},
        {"reasoning_effort": "unsupported"},
        {"effort": "unsupported"},
    ],
)
def test_explicit_forbidden_config_not_silently_substituted(config):
    with pytest.raises(InvalidRequestError, match="claude-haiku-5-5"):
        assemble(provider(**config))


@pytest.mark.parametrize(
    "options",
    [
        {"thinking_type": "enabled"},
        {"thinking_type": "between_tools"},
        {"thinking_budget_tokens": 4096},
        {"effort": "unsupported"},
    ],
)
def test_explicit_forbidden_call_controls(options):
    with pytest.raises(InvalidRequestError, match="claude-haiku-5-5"):
        assemble(**options)


@pytest.mark.parametrize(
    "extra",
    [
        {"thinking": {"type": "enabled", "budget_tokens": 4096}},
        {"thinking": {"type": "between_tools"}},
        {"thinking": {"type": "adaptive", "budget_tokens": 4096}},
        {"thinking": {"type": "disabled"}, "output_config": {"effort": "max"}},
        {"temperature": 0.2},
        {"top_p": 0.99},
        {"top_k": 2},
        {"messages": [{"role": "assistant", "content": "{"}]},
        {"tools": [{"type": "computer_20250124", "name": "desktop"}]},
        {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 4096}}},
        {"budget_tokens": 4096},
        {"output_config": {"effort": "unsupported"}},
        {"extra_body": {"model": MODEL, "top_k": 2}},
        {"thinking": None},
    ],
)
def test_post_override_falsifiers(extra):
    p = provider(extra_request_params=extra)
    before = deepcopy(extra)
    with pytest.raises(InvalidRequestError, match="claude-haiku-5-5"):
        assemble(p)
    assert extra == before
    assert p._client is None


def test_final_model_override_cannot_evade_guard():
    with pytest.raises(InvalidRequestError, match="claude-haiku-5-5"):
        assemble(
            provider(
                default_model="claude-haiku-4-5",
                extra_request_params={"model": MODEL, "thinking": {"type": "enabled"}},
            )
        )


def test_prefill_error_is_actionable_and_original_untouched():
    req = request(
        messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="{"),
        ]
    )
    before = req.model_dump()
    with pytest.raises(
        InvalidRequestError, match="prefill.*structured|structured.*prefill"
    ):
        assemble(req=req)
    assert req.model_dump() == before
    signed = req.model_copy(
        update={
            "messages": [
                req.messages[0],
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="", signature="signed-original"),
                        TextBlock(text="{"),
                    ],
                ),
            ]
        },
        deep=True,
    )
    original = signed.model_dump()
    with pytest.raises(InvalidRequestError, match="signed history"):
        assemble(req=signed)
    assert signed.model_dump() == original


@pytest.mark.parametrize(
    "choice", ["auto", "none", "required", {"type": "tool", "name": "lookup"}]
)
def test_forced_ordinary_tools_suppress_thinking_without_mutation(choice):
    req = request(tools=[tool()], tool_choice=choice, reasoning_effort="max")
    before = req.model_dump()
    params = assemble(req=req)
    forced = choice == "required" or isinstance(choice, dict)
    assert (
        ("thinking" not in params)
        if forced
        else params["thinking"]["type"] == "adaptive"
    )
    assert params["output_config"] == {"effort": "max"}
    assert req.model_dump() == before
    followup = request(
        tools=[tool()],
        messages=[
            Message(role="user", content="hello"),
            Message(
                role="assistant",
                content=[ToolCallBlock(id="call_1", name="lookup", input={})],
            ),
            Message.model_construct(
                role="tool", tool_call_id="call_1", content="result"
            ),
        ],
    )
    next_params = assemble(req=followup)
    assert next_params["thinking"]["type"] == "adaptive"
    assert next_params["messages"][-1]["content"][0]["type"] == "tool_result"


@pytest.mark.parametrize(
    "choice", [{"type": "any"}, {"type": "tool", "name": "lookup"}]
)
def test_forced_expert_override_is_turn_local(choice):
    params = assemble(
        provider(extra_request_params={"tool_choice": choice}), request(tools=[tool()])
    )
    assert params["tool_choice"] == choice and "thinking" not in params
    assert assemble(req=request(tools=[tool()]))["thinking"]["type"] == "adaptive"


def test_shadowed_count_fields_are_canonical_and_output_ceiling_survives():
    extra = {
        "extra_body": {
            "model": MODEL,
            "messages": [{"role": "user", "content": "override"}],
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "low"},
            "max_tokens": 900_000,
        }
    }
    p = provider(extra_request_params=extra)
    params = assemble(p)
    assert params["messages"] == extra["extra_body"]["messages"]
    assert params["thinking"] == {"type": "disabled"}
    assert params["max_tokens"] == 128_000
    assert not params.get("extra_body")
    assert p._count_tokens_params(params)["messages"] == params["messages"]
    assert extra["extra_body"]["max_tokens"] == 900_000
    assert assemble(p, request(max_output_tokens=123))["max_tokens"] == 123


@pytest.mark.parametrize("full_context", [False, True])
def test_runtime_native_ceiling_does_not_promote_catalog_policy(full_context):
    p = provider(enable_1m_context=full_context)
    info = SimpleNamespace(
        id=MODEL,
        display_name=MODEL,
        created_at="",
        max_input_tokens=1_000_000,
        max_tokens=128_000,
    )
    p.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[info]))
    models = asyncio.run(p.list_models())
    advertised = 1_000_000 if full_context else 200_000
    assert models[0].context_window == advertised
    assert p.get_info().defaults["context_window"] == advertised
    # The independent budget ceiling can still use known native runtime info.
    p._runtime_model_info_cache[MODEL] = p._extract_runtime_model_info(info)
    assert p._budget_input_limit(MODEL, p._get_capabilities(MODEL)) == 1_000_000
    asyncio.run(p.close())


@pytest.mark.parametrize("full_context", [False, True])
def test_cold_budget_uses_exact_shared_payload_and_reserve(full_context):
    p = provider(enable_1m_context=full_context)
    p.client.messages.count_tokens = AsyncMock(
        return_value=SimpleNamespace(input_tokens=321)
    )
    req = request(max_output_tokens=123, tools=[tool()])
    before = req.model_dump()
    params = assemble(p, req)
    decision = asyncio.run(p.request_budget(req, context_estimate=200_000))
    sent = p.client.messages.count_tokens.call_args.kwargs
    assert sent["model"] == MODEL and sent["messages"] == params["messages"]
    assert sent["tools"] == params["tools"] and sent["thinking"] == params["thinking"]
    assert "max_tokens" not in sent
    assert decision["measurement"]["input_tokens"] == 321
    assert decision["estimated_input_tokens"] == 321 + 4096
    assert decision["max_output_tokens"] == 123
    assert decision["input_limit_tokens"] == (1_000_000 if full_context else 200_000)
    assert req.model_dump() == before
    asyncio.run(p.close())


@pytest.mark.parametrize(
    "result", [None, SimpleNamespace(input_tokens=True), RuntimeError("offline")]
)
def test_unavailable_count_is_not_estimated(result):
    p = provider()
    p.client.messages.count_tokens = AsyncMock(
        **(
            {"side_effect": result}
            if isinstance(result, Exception)
            else {"return_value": result}
        )
    )
    assert asyncio.run(p.request_budget(request(), context_estimate=200_000)) is None
    asyncio.run(p.close())


@pytest.mark.parametrize("tokens", [0, 99_999, 100_000, 100_001, 900_000])
def test_haiku_unknown_cost_never_legacy_or_zero(tokens):
    assert (
        compute_cost(
            MODEL,
            input_tokens=tokens,
            output_tokens=10,
            cache_read_input_tokens=99,
            cache_creation_input_tokens=100,
            cache_creation_5m_input_tokens=40,
            cache_creation_1h_input_tokens=60,
        )
        is None
    )
    assert compute_cost(MODEL + "-20261007", input_tokens=tokens) is None


def test_sonnet55_cache_read_reduction_exact_id_and_ttl_split():
    assert compute_cost(
        "claude-sonnet-5-5",
        input_tokens=1000,
        output_tokens=100,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1000,
        cache_creation_5m_input_tokens=400,
        cache_creation_1h_input_tokens=600,
    ) == Decimal("0.1064")
    assert compute_cost("claude-sonnet-5-5-20261007", input_tokens=10) is None
    assert compute_cost("claude-haiku-4-5", input_tokens=1000) == Decimal("0.001")


def sdk_body(model=MODEL, content=None, stop="end_turn"):
    return {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content if content is not None else [{"type": "text", "text": "ok"}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read_input_tokens": 1_000_000,
            "cache_creation_input_tokens": 1000,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 400,
                "ephemeral_1h_input_tokens": 600,
            },
        },
    }


@pytest.mark.parametrize("model", [MODEL, "claude-sonnet-5-5"])
def test_real_sdk_create_count_models_usage_and_json(model):
    seen = []

    def handler(req):
        data = json.loads(req.content) if req.content else None
        seen.append((req.url.path, data))
        if req.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 91})
        if req.url.path.endswith("/models"):
            return httpx2.Response(
                200,
                json={
                    "data": [
                        {
                            "id": model,
                            "type": "model",
                            "display_name": model,
                            "created_at": "2026-10-07T00:00:00Z",
                        }
                    ],
                    "has_more": False,
                    "first_id": model,
                    "last_id": model,
                },
            )
        return httpx2.Response(200, json=sdk_body(model))

    async def run():
        p = provider(default_model=model)
        p._client = AsyncAnthropic(
            api_key="offline-placeholder",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        try:
            assert (await p.list_models())[0].id == model
            assert (await p.request_budget(request(), context_estimate=100))[
                "measurement"
            ]["input_tokens"] == 91
            response = await p.complete(request())
            event = p.coordinator.hooks.payload_for("llm:response")
            json.loads(response.model_dump_json())
            json.dumps(event)
            if model == MODEL:
                assert response.usage.cost_usd is None
                assert event["usage"]["cost_usd"] is None
                assert (
                    response.metadata["anthropic_cost_unavailable"]
                    == "haiku55_prompt_tier_unverified"
                )
                assert event["metadata"] == response.metadata
            else:
                assert response.usage.cost_usd == Decimal("0.1064")
                assert event["usage"]["cost_usd"] == "0.1064"
            calls = {path: data for path, data in seen}
            assert calls["/v1/messages/count_tokens"]["model"] == model
            assert calls["/v1/messages"]["model"] == model
            assert (
                calls["/v1/messages/count_tokens"]["messages"]
                == calls["/v1/messages"]["messages"]
            )
        finally:
            await p.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "stop,blocks",
    [
        (
            "end_turn",
            [
                {"type": "thinking", "thinking": "", "signature": "signed-empty"},
                {"type": "text", "text": "visible"},
                {
                    "type": "thinking",
                    "thinking": "private",
                    "signature": "signed-interleaved",
                },
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "lookup",
                    "input": {"value": 1},
                },
            ],
        ),
        (
            "max_tokens",
            [{"type": "thinking", "thinking": "", "signature": "truncated-signature"}],
        ),
        ("refusal", [{"type": "text", "text": "cannot comply"}]),
    ],
)
def test_real_sdk_stream_order_signatures_private_text_and_stop(stop, blocks):
    events = []
    initial = sdk_body(content=[], stop=None)
    events.append({"type": "message_start", "message": initial})
    for index, block in enumerate(blocks):
        start = dict(block)
        if block["type"] == "tool_use":
            start["input"] = {}
        elif block["type"] == "text":
            start["text"] = ""
        elif block["type"] == "thinking":
            start.update(thinking="", signature="")
        events.append(
            {"type": "content_block_start", "index": index, "content_block": start}
        )
        if block["type"] == "tool_use":
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"]),
            }
        elif block["type"] == "text":
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            events.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": block["thinking"]},
                }
            )
            delta = {"type": "signature_delta", "signature": block["signature"]}
        events.append({"type": "content_block_delta", "index": index, "delta": delta})
        events.append({"type": "content_block_stop", "index": index})
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": 100},
            },
            {"type": "message_stop"},
        ]
    )
    sse = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    )

    async def run():
        p = provider(use_streaming=True, refusal_fallback_enabled=False)
        p._client = AsyncAnthropic(
            api_key="offline-placeholder",
            max_retries=0,
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(
                    lambda req: httpx2.Response(
                        200,
                        headers={"content-type": "text/event-stream"},
                        content=sse.encode(),
                    )
                )
            ),
        )
        try:
            response = await p.complete(request(tools=[tool()]))
            assert response.finish_reason == stop
            assert response.text == (
                None
                if stop == "max_tokens"
                else "visible"
                if stop == "end_turn"
                else "cannot comply"
            )
            restored = Message.model_validate_json(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ).model_dump_json()
            )
            result = p._convert_messages([restored.model_dump()])[0]["content"]
            assert result == blocks
            if stop == "end_turn":
                following = request(
                    tools=[tool()],
                    messages=[
                        Message(role="user", content="hello"),
                        restored,
                        Message(
                            role="tool",
                            tool_call_id="call_1",
                            content=[
                                TextBlock(text="result"),
                                ImageBlock(
                                    source={
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "aW1hZ2U=",
                                    }
                                ),
                            ],
                        ),
                    ],
                )
                before = following.model_dump()
                wire = assemble(p, following)
                assert wire["messages"][1]["content"] == blocks
                assert (
                    wire["messages"][2]["content"][0]["content"][1]["type"] == "image"
                )
                assert following.model_dump() == before
        finally:
            await p.close()

    asyncio.run(run())
