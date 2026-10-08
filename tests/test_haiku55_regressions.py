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
from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import ContextLengthError, InvalidRequestError, ProviderUnavailableError
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    ToolSpec,
)

from amplifier_module_provider_anthropic import AnthropicProvider, _RuntimeModelInfo, mount
from amplifier_module_provider_anthropic._cost import compute_cost
from tests._helpers import FakeCoordinator
import amplifier_module_provider_anthropic as provider_module

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


def recording_client(seen, *, zero_usage=False):
    """Real SDK transport; every endpoint is synthetic and recorded."""
    def handler(req):
        body = json.loads(req.content) if req.content else {}
        seen.append((req.method, req.url.path, body))
        if req.url.path.startswith("/v1/models"):
            model = req.url.path.rsplit("/", 1)[-1]
            return httpx2.Response(200, json={
                "id": model, "type": "model", "display_name": model,
                "created_at": "2026-10-07T00:00:00Z",
                "max_input_tokens": 1_000_000, "max_tokens": 64_000,
            })
        if req.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 321})
        response = sdk_body(body.get("model", MODEL))
        if zero_usage:
            response["usage"] = {"input_tokens": 0, "output_tokens": 0}
        return httpx2.Response(200, json=response)

    return AsyncAnthropic(
        api_key="offline-placeholder", max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


@pytest.mark.parametrize("models,zero", [
    (["claude-sonnet-5-5", MODEL, "claude-sonnet-5-5"], False),
    ([MODEL, "claude-sonnet-5-5"], False),
    ([MODEL], False),
    (["claude-sonnet-5-5", MODEL], True),
])
def test_public_mount_cost_contribution_stays_unknown(models, zero):
    async def run():
        coordinator = ModuleCoordinator()
        cleanup = await mount(coordinator, {
            "api_key": "offline-placeholder", "default_model": MODEL,
            "use_streaming": False, "enable_prompt_caching": False, "max_retries": 0,
        })
        p = coordinator.get("providers", "anthropic")
        seen = []
        p._client = recording_client(seen, zero_usage=zero)
        try:
            assert await coordinator.collect_contributions("session.cost") == []
            unknown = False
            subtotal = Decimal("0")
            for model in models:
                response = await p.complete(request(), model=model)
                if model == MODEL:
                    unknown = True
                    assert response.usage.cost_usd is None
                    assert response.metadata["anthropic_cost_unavailable"] == (
                        "haiku55_prompt_tier_unverified"
                    )
                else:
                    assert response.usage.cost_usd is not None
                    subtotal += response.usage.cost_usd
                contributions = await coordinator.collect_contributions("session.cost")
                assert contributions == [{"cost_usd": None if unknown else str(subtotal)}]
                assert json.loads(json.dumps(contributions)) == contributions
        finally:
            await cleanup()
    asyncio.run(run())


def missing_history(*, prefill=False):
    rows = [
        Message(role="system", content="synthetic system", metadata={"nested": [1]}),
        Message(role="user", content="hello", metadata={"nested": {"value": [2]}}),
        Message(role="assistant", content=[
            ThinkingBlock(thinking="", signature="synthetic-empty-original"),
            TextBlock(text="checking"),
            ThinkingBlock(thinking="synthetic", signature="synthetic-interleaved"),
            ToolCallBlock(id="call_missing", name="lookup", input={"nested": [3]}),
        ]),
        Message(role="user", content="continue"),
    ]
    if prefill:
        rows.append(Message(role="assistant", content=[
            ThinkingBlock(thinking="", signature="synthetic-prefill-original"),
            TextBlock(text="{"),
        ]))
    return request(messages=rows, tools=[tool()], metadata={"nested": {"value": [4]}})


@pytest.mark.parametrize("operation", ["complete", "request_budget"])
@pytest.mark.parametrize("source", ["direct", "expert", "nested-expert"])
@pytest.mark.parametrize("config,options", [
    ({"thinking_budget_tokens": 4096}, {}),
    ({}, {"thinking_budget_tokens": "invalid"}),
    ({}, {"thinking_type": "enabled"}),
    ({}, {"thinking_type": "between_tools"}),
    ({"extended_thinking": False}, {"effort": "max"}),
    ({"extended_thinking": False}, {"effort": "xhigh"}),
    ({"extra_request_params": {"thinking": None}}, {}),
    ({"extra_request_params": {"thinking": {"type": "between_tools"}}}, {}),
    ({"extra_request_params": {"thinking": {"type": "adaptive", "budget_tokens": 1}}}, {}),
    ({"extra_request_params": {"temperature": 0.2}}, {}),
    ({"extra_request_params": {"extra_body": {"top_p": 0.9}}}, {}),
    ({"extra_request_params": {"top_k": 2}}, {}),
    ({"extra_request_params": {"budget_tokens": 1024}}, {}),
    ({"extra_request_params": {"output_config": {"effort": "unsupported"}}}, {}),
    ({"extra_request_params": {"tools": [{"type": "computer_toolset_20260801"}]}}, {}),
    ({"extra_request_params": {"extra_body": {"tools": [{"type": "computer_20250124"}]}}}, {}),
    ({"extra_request_params": {"messages": [{"role": "assistant", "content": "{"}]}}, {}),
])
def test_invalid_public_envelope_has_zero_transport_and_no_repair(
    operation, source, config, options
):
    async def run():
        config_copy = deepcopy(config)
        extra = config_copy.setdefault("extra_request_params", {})
        if source == "expert":
            extra["model"] = MODEL
        elif source == "nested-expert":
            extra.setdefault("extra_body", {})["model"] = MODEL
        p = provider(
            default_model=MODEL if source == "direct" else "claude-haiku-4-5",
            **config_copy,
        )
        req = missing_history()
        before = req.model_dump_json()
        config_before = deepcopy(p.config)
        options_before = deepcopy(options)
        prefix_before = deepcopy(p._prefix_fingerprints)
        seen = []
        p._client = recording_client(seen)
        try:
            with pytest.raises(InvalidRequestError, match="claude-haiku-5-5"):
                if operation == "complete":
                    await p.complete(req, **options)
                else:
                    await p.request_budget(req, context_estimate=100, request_options=options)
            assert seen == []  # Includes Models GET, not only generation/count.
            assert req.model_dump_json() == before
            assert p.config == config_before and options == options_before
            assert p._repaired_tool_ids == set()
            assert p._prefix_fingerprints == prefix_before
            assert p._extra_params_warned_keys == set()
            assert p._runtime_model_info_cache == {}
            assert p.coordinator.hooks.events == []
        finally:
            await p.close()
    asyncio.run(run())


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("source", ["direct", "expert", "nested-expert"])
def test_signed_prefill_refused_before_client_or_caller_mutation(
    streaming, missing, source, monkeypatch
):
    seen = []
    created = []
    def offline_factory(*args, **kwargs):
        created.append(True)
        return recording_client(seen)
    monkeypatch.setattr(provider_module, "AsyncAnthropic", offline_factory)
    extra = (
        {"model": MODEL} if source == "expert" else
        {"extra_body": {"model": MODEL}} if source == "nested-expert" else {}
    )
    p = provider(
        use_streaming=streaming,
        default_model=MODEL if source == "direct" else "claude-haiku-4-5",
        extra_request_params=extra,
    )
    req = missing_history(prefill=True)
    if not missing:
        req.messages.insert(3, Message(
            role="tool", tool_call_id="call_missing", content="synthetic result",
        ))
    before = req.model_dump_json()
    async def run():
        try:
            with pytest.raises(InvalidRequestError, match="prefill.*structured"):
                await p.complete(req)
            assert p._client is None and created == []
            assert seen == []
            assert req.model_dump_json() == before
            assert not p._repaired_tool_ids
            assert p.coordinator.hooks.events == []
        finally:
            await p.close()
    asyncio.run(run())


def test_accepted_haiku_repairs_a_copy_on_each_public_call():
    async def run():
        p = provider()
        seen = []
        p._client = recording_client(seen)
        req = missing_history()
        before = req.model_dump_json()
        try:
            for _ in range(2):
                await p.request_budget(req, context_estimate=200_000)
                await p.complete(req)
                assert req.model_dump_json() == before
                body = [body for _, path, body in seen if path == "/v1/messages"][-1]
                blocks = body["messages"][1]["content"]
                assert blocks[0] == {
                    "type": "thinking", "thinking": "", "signature": "synthetic-empty-original",
                }
                assert blocks[2]["signature"] == "synthetic-interleaved"
                result = body["messages"][2]["content"][0]
                assert result["type"] == "tool_result"
                assert result["tool_use_id"] == "call_missing"
                assert "SYSTEM ERROR" in result["content"]
                count = [body for _, path, body in seen if path.endswith("/count_tokens")][-1]
                assert count["messages"] == body["messages"]
            assert p._repaired_tool_ids == set()
            assert len([e for e, _ in p.coordinator.hooks.events
                        if e == "provider:tool_sequence_repaired"]) == 2
        finally:
            await p.close()
    asyncio.run(run())


@pytest.mark.parametrize("history", ["scalar", "canonical", "partial", "multiple_turns"])
@pytest.mark.parametrize("caching", [False, True])
def test_public_compatibility_repairs_follow_canonical_projection(history, caching):
    async def run():
        p = provider(enable_prompt_caching=caching)
        seen = []
        p._client = recording_client(seen)
        calls = [ToolCall(id="compat_1", name="lookup", arguments={"value": 1})]
        content = "legacy text"
        if history == "canonical":
            content = [
                ThinkingBlock(thinking="", signature="synthetic-canonical"),
                ToolCallBlock(id="compat_1", name="lookup", input={"canonical": True}),
                TextBlock(text="ordered text"),
            ]
            calls.append(ToolCall(id="compat_1", name="stale_name", arguments={}))
        if history == "partial":
            calls.extend([
                ToolCall(id="compat_2", name="lookup", arguments={"value": 2}),
                ToolCall(id="compat_2", name="stale_name", arguments={}),
            ])
        messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content=content, tool_calls=calls),
        ]
        if history == "partial":
            messages.append(Message(
                role="tool", tool_call_id="compat_1", content="existing result",
            ))
        messages.append(Message(role="user", content="continue"))
        if history == "multiple_turns":
            messages.extend([
                Message(role="assistant", content="next", tool_calls=[
                    ToolCall(id="compat_2", name="lookup", arguments={"value": 2}),
                ]),
                Message(role="user", content="continue again"),
            ])
        req = request(messages=messages, tools=[tool()], max_output_tokens=123)
        before = req.model_dump_json()
        try:
            if caching:
                await p.complete(request())
                assert p._prefix_fingerprints
                seen.clear()
                p.coordinator.hooks.events.clear()
            for _ in range(2):
                events_before = deepcopy(p.coordinator.hooks.events)
                prefix_before = deepcopy(p._prefix_fingerprints)
                runtime_before = deepcopy(p._runtime_model_info_cache)
                budget = await p.request_budget(req, context_estimate=200_000)
                assert budget["max_output_tokens"] == 123
                assert p.coordinator.hooks.events == events_before
                assert p._prefix_fingerprints == prefix_before
                assert p._runtime_model_info_cache == runtime_before
                assert not p._repaired_tool_ids
                await p.complete(req)
                assert req.model_dump_json() == before
                count = [b for _, path, b in seen if path.endswith("/count_tokens")][-1]
                body = [b for _, path, b in seen if path == "/v1/messages"][-1]
                assert count["messages"] == body["messages"]
                assert body["max_tokens"] == 123 and "max_tokens" not in count
                wire = body["messages"]
                for index, message in enumerate(wire):
                    if message["role"] != "assistant":
                        continue
                    uses = [b for b in message["content"] if b["type"] == "tool_use"]
                    ids = [b["id"] for b in uses]
                    assert len(ids) == len(set(ids))
                    assert wire[index + 1]["role"] == "user"
                    results = wire[index + 1]["content"]
                    assert isinstance(results, list)
                    assert {r["tool_use_id"] for r in results} == set(ids)
                    assert len(results) == len(ids)
                    for result in results:
                        assert result["type"] == "tool_result"
                        if history == "partial" and result["tool_use_id"] == "compat_1":
                            assert result["content"] == "existing result"
                        else:
                            assert "SYSTEM ERROR" in result["content"]
                            assert "Tool: lookup" in result["content"]
                if history == "canonical":
                    assert [{k: v for k, v in b.items() if k != "cache_control"}
                            for b in wire[1]["content"]] == [
                        {"type": "thinking", "thinking": "", "signature": "synthetic-canonical"},
                        {"type": "tool_use", "id": "compat_1", "name": "lookup",
                         "input": {"canonical": True}},
                        {"type": "text", "text": "ordered text"},
                    ]
                else:
                    assert wire[1]["content"][0] == {"type": "text", "text": "legacy text"}
            assert not p._repaired_tool_ids
            repairs = [payload for event, payload in p.coordinator.hooks.events
                       if event == "provider:tool_sequence_repaired"]
            assert len(repairs) == 2
            assert all(r["repair_count"] == (2 if history == "multiple_turns" else 1)
                       for r in repairs)
        finally:
            await p.close()
    asyncio.run(run())


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("explicit_model", [False, True])
@pytest.mark.parametrize("change", [
    "none", "request", "compatibility", "options", "model", "wire", "foreign", "copy",
])
def test_public_overflow_binds_caller_options_not_fallback_injection(
    fallback, explicit_model, change, monkeypatch,
):
    monkeypatch.setattr(provider_module, "_fallback_windows", {})
    async def run():
        p = provider(fallback_on_overload=fallback)
        seen = []
        def handler(req):
            body = json.loads(req.content) if req.content else {}
            seen.append((req.method, req.url.path, body))
            if req.method == "GET":
                return httpx2.Response(404, json={"error": {"type": "not_found_error"}})
            return httpx2.Response(
                400, headers={"request-id": "synthetic-input-overflow"},
                json={"type": "error", "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: 208310 tokens > 200000 maximum",
                }},
            )
        p._client = AsyncAnthropic(
            api_key="offline-placeholder", max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        req = request(messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="legacy", tool_calls=[
                ToolCall(id="compat_1", name="lookup", arguments={}),
            ]),
            Message(role="user", content="continue"),
        ], tools=[tool()])
        options = {"model": MODEL} if explicit_model else {}
        before = req.model_dump_json()
        original_messages = deepcopy(req.messages)
        other = provider()
        try:
            with pytest.raises(ContextLengthError) as raised:
                await p.complete(req, **options)
            assert req.model_dump_json() == before
            assert options == ({"model": MODEL} if explicit_model else {})
            assert len([b for _, path, b in seen if path == "/v1/messages"]) == 1
            error = raised.value
            assert error.__cause__.request_id == "synthetic-input-overflow"
            target = req
            recovering = p
            if change == "request":
                req.messages[0].content = "changed"
            elif change == "compatibility":
                req.messages[1].tool_calls[0].arguments["changed"] = True
            elif change == "options":
                # Even an identical wire model cannot erase a caller-option change.
                if explicit_model:
                    options.clear()
                else:
                    options["model"] = MODEL
            elif change == "model":
                if explicit_model:
                    options["model"] = "claude-haiku-4-5"
                else:
                    p.default_model = "claude-haiku-4-5"
            elif change == "wire":
                p.extra_request_params = {"system": [{"type": "text", "text": "changed"}]}
            elif change == "foreign":
                recovering = other
            elif change == "copy":
                target = req.model_copy(deep=True)
            decision = recovering.recover_context_overflow(
                target, error, context_estimate=200_000, request_options=options,
            )
            if change == "none":
                assert decision == {
                    "estimated_input_tokens": 208310,
                    "input_limit_tokens": 200000,
                    "context_token_budget": 188087,
                    "max_output_tokens": 128000,
                }
            else:
                assert decision is None
            assert recovering.recover_context_overflow(
                target, error, context_estimate=200_000, request_options=options,
            ) is None
            if change in {"request", "compatibility", "options", "model", "wire"}:
                # A refused consumed feedback cannot be revived by restoring the
                # original payload/options, even when that wire matches again.
                req.messages = original_messages
                p.default_model = MODEL
                p.extra_request_params = {}
                options = {"model": MODEL} if explicit_model else {}
                assert req.model_dump_json() == before
                assert p.recover_context_overflow(
                    req, error, context_estimate=200_000, request_options=options,
                ) is None
            elif change in {"foreign", "copy"}:
                # An identity mismatch refuses without stealing the owner's
                # one-shot recovery from the original request.
                assert p.recover_context_overflow(
                    req, error, context_estimate=200_000, request_options=options,
                ) is not None
                assert p.recover_context_overflow(
                    req, error, context_estimate=200_000, request_options=options,
                ) is None
            assert len([b for _, path, b in seen if path == "/v1/messages"]) == 1
        finally:
            await p.close()
            await other.close()
    asyncio.run(run())


@pytest.mark.parametrize("legacy", ["claude-haiku-4-5", "claude-sonnet-5-5"])
def test_request_local_repairs_do_not_suppress_later_legacy_repair(legacy):
    async def run():
        p = provider()
        seen = []
        p._client = recording_client(seen)
        req = missing_history()
        before = req.model_dump_json()
        try:
            await p.complete(req)
            assert req.model_dump_json() == before
            await p.complete(req, model=legacy)
            bodies = [body for _, path, body in seen if path == "/v1/messages"]
            for body in bodies:
                assert body["messages"][2]["content"][0]["tool_use_id"] == "call_missing"
        finally:
            await p.close()
    asyncio.run(run())


@pytest.mark.parametrize("mutate", [False, True])
def test_repaired_haiku_overflow_is_bound_to_unchanged_public_request(mutate):
    async def run():
        p = provider()
        def handler(req):
            if req.method == "GET":
                return httpx2.Response(404, json={"error": {"type": "not_found_error"}})
            return httpx2.Response(
                400, headers={"request-id": "synthetic-overflow"},
                json={"type": "error", "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: 208310 tokens > 200000 maximum",
                }},
            )
        p._client = AsyncAnthropic(
            api_key="offline-placeholder", max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        req = missing_history()
        before = req.model_dump_json()
        try:
            with pytest.raises(ContextLengthError) as raised:
                await p.complete(req)
            assert req.model_dump_json() == before
            if mutate:
                req.messages[1].content = "changed"
            decision = p.recover_context_overflow(req, raised.value, context_estimate=200_000)
            if mutate:
                assert decision is None
            else:
                assert decision is not None
                assert decision["estimated_input_tokens"] == 208310
                assert 0 < decision["context_token_budget"] < 200_000
            assert p.recover_context_overflow(
                req, raised.value, context_estimate=200_000,
            ) is None
        finally:
            await p.close()
    asyncio.run(run())


@pytest.mark.parametrize("nested", [False, True])
def test_valid_expert_haiku_uses_runtime_caps_and_canonical_count_dispatch(nested):
    async def run():
        wire = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "synthetic override"}],
            "system": [{"type": "text", "text": "synthetic system"}],
            "tools": [{"name": "computer", "input_schema": {"type": "object", "properties": {}}}],
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "low"},
            "max_tokens": 900_000,
        }
        extra = {"extra_body": wire} if nested else wire
        p = provider(default_model="claude-haiku-4-5", extra_request_params=extra)
        seen = []
        p._client = recording_client(seen)
        req = request(max_output_tokens=123)
        before = req.model_dump_json()
        extra_before = deepcopy(extra)
        try:
            # Cold count stays pure; dispatch still retrieves real SDK metadata.
            cold = await p.request_budget(req, context_estimate=200_000)
            assert cold["max_output_tokens"] == 123
            assert cold["estimated_input_tokens"] == 321 + 4096
            await p.complete(req)
            warm = await p.request_budget(request(), context_estimate=200_000)
            assert warm["input_limit_tokens"] == 1_000_000
            assert warm["max_output_tokens"] == 64_000
            await p.complete(request())
            assert [path for method, path, _ in seen if method == "GET"] == [
                "/v1/models/claude-haiku-5-5",
            ]
            bodies = [body for _, path, body in seen if path == "/v1/messages"]
            counts = [body for _, path, body in seen if path.endswith("/count_tokens")]
            for body, count, cap in zip(bodies, counts, [123, 64_000], strict=True):
                assert body["max_tokens"] == cap
                assert "max_tokens" not in count
                for key in ("model", "messages", "system", "tools", "thinking", "output_config"):
                    assert body[key] == count[key] == wire[key]
            assert req.model_dump_json() == before and extra == extra_before
        finally:
            await p.close()
    asyncio.run(run())


def test_public_forced_tool_then_adaptive_preserves_signed_history():
    async def run():
        p = provider()
        seen = []
        p._client = recording_client(seen)
        first = request(tools=[tool()], tool_choice="required", reasoning_effort="max")
        followup = missing_history()
        followup.messages.insert(3, Message(
            role="tool", tool_call_id="call_missing", content="synthetic result",
        ))
        originals = [req.model_dump_json() for req in (first, followup)]
        try:
            await p.complete(first)
            await p.complete(followup)
            bodies = [body for _, path, body in seen if path == "/v1/messages"]
            assert "thinking" not in bodies[0]
            assert bodies[0]["output_config"] == {"effort": "max"}
            assert bodies[1]["thinking"]["type"] == "adaptive"
            assert bodies[1]["messages"][1]["content"][0]["signature"] == "synthetic-empty-original"
            assert [req.model_dump_json() for req in (first, followup)] == originals
        finally:
            await p.close()
    asyncio.run(run())


@pytest.mark.parametrize("operation", ["complete", "request_budget"])
@pytest.mark.parametrize("config", [
    {"thinking_budget_tokens": 1},
    {"extra_request_params": {"top_k": 2}},
    {"extra_request_params": {"thinking": None}},
    {"extra_request_params": {"tools": [{"type": "computer_toolset_20260801"}]}},
])
def test_other_cold_refusals_never_construct_client(operation, config, monkeypatch):
    created = []
    seen = []
    def offline_factory(*args, **kwargs):
        created.append(True)
        return recording_client(seen)
    monkeypatch.setattr(provider_module, "AsyncAnthropic", offline_factory)
    p = provider(**config)
    async def run():
        try:
            with pytest.raises(InvalidRequestError):
                if operation == "complete":
                    await p.complete(missing_history())
                else:
                    await p.request_budget(missing_history(), context_estimate=100)
            assert p._client is None and created == [] and seen == []
            assert not p._repaired_tool_ids and not p.coordinator.hooks.events
        finally:
            await p.close()
    asyncio.run(run())


@pytest.mark.parametrize("operation", ["complete", "request_budget"])
def test_refusal_with_populated_cache_state_is_side_effect_free(operation):
    async def run():
        p = provider(enable_prompt_caching=True)
        seen = []
        p._client = recording_client(seen)
        try:
            await p.complete(request())
            assert p._prefix_fingerprints
            prefix_before = deepcopy(p._prefix_fingerprints)
            runtime_before = deepcopy(p._runtime_model_info_cache)
            seen.clear()
            p.coordinator.hooks.events.clear()
            req = missing_history(prefill=True)
            before = req.model_dump_json()
            with pytest.raises(InvalidRequestError, match="prefill"):
                if operation == "complete":
                    await p.complete(req)
                else:
                    await p.request_budget(req, context_estimate=100)
            assert req.model_dump_json() == before
            assert p._prefix_fingerprints == prefix_before
            assert p._runtime_model_info_cache == runtime_before
            assert not seen and not p.coordinator.hooks.events and not p._repaired_tool_ids
        finally:
            await p.close()
    asyncio.run(run())


@pytest.mark.parametrize("tool_type", [
    "computer_20241022", "computer_20250124", "computer_20251124", "computer_toolset_20260801",
])
def test_native_request_tools_refused_without_transport(tool_type):
    async def run():
        p = provider()
        seen = []
        p._client = recording_client(seen)
        native = tool()
        setattr(native, "type", tool_type)
        req = request(tools=[native])
        before = req.model_dump_json()
        try:
            for operation in ("complete", "request_budget"):
                with pytest.raises(InvalidRequestError, match="executor qualification"):
                    if operation == "complete":
                        await p.complete(req)
                    else:
                        await p.request_budget(req, context_estimate=100)
                assert seen == [] and req.model_dump_json() == before
        finally:
            await p.close()
    asyncio.run(run())


@pytest.mark.parametrize("model,config,expected", [
    (
        "claude-haiku-4-5", {"extended_thinking": True, "thinking_type": "enabled"},
        {"type": "enabled", "budget_tokens": 32_000},
    ),
    (
        "claude-sonnet-5-5", {"extra_request_params": {"thinking": {"type": "between_tools"}}},
        {"type": "between_tools"},
    ),
    (
        MODEL, {"extended_thinking": False}, {"type": "disabled"},
    ),
])
def test_public_sibling_thinking_modes_remain_distinct(model, config, expected):
    async def run():
        p = provider(default_model=model, **config)
        seen = []
        p._client = recording_client(seen)
        try:
            await p.complete(request(tools=[tool()]))
            body = [body for _, path, body in seen if path == "/v1/messages"][-1]
            assert body["thinking"] == expected
        finally:
            await p.close()
    asyncio.run(run())


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("outcome", ["refusal", "overload"])
@pytest.mark.parametrize("overload_fallback", [False, True])
def test_effective_expert_haiku_stays_on_terminal_fallback_rung(
    nested, outcome, overload_fallback, monkeypatch
):
    monkeypatch.setattr(provider_module, "_fallback_windows", {})
    async def run():
        extra = {"extra_body": {"model": MODEL}} if nested else {"model": MODEL}
        p = provider(
            default_model="claude-sonnet-5-5", extra_request_params=extra,
            fallback_on_overload=overload_fallback, fallback_retry_count=0,
        )
        seen = []
        def handler(req):
            body = json.loads(req.content) if req.content else {}
            seen.append((req.method, req.url.path, body))
            if req.method == "GET":
                return httpx2.Response(404, json={"error": {"type": "not_found_error"}})
            if outcome == "overload":
                return httpx2.Response(529, json={"type": "error", "error": {
                    "type": "overloaded_error", "message": "synthetic overload",
                }})
            return httpx2.Response(200, json=sdk_body(stop="refusal"))
        p._client = AsyncAnthropic(
            api_key="offline-placeholder", max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        req = missing_history()
        req.messages.insert(3, Message(
            role="tool", tool_call_id="call_missing", content="synthetic result",
        ))
        before = req.model_dump_json()
        try:
            if outcome == "overload":
                with pytest.raises(ProviderUnavailableError):
                    await p.complete(req)
            else:
                assert (await p.complete(req)).finish_reason == "refusal"
            bodies = [body for _, path, body in seen if path == "/v1/messages"]
            assert len(bodies) == 1
            assert bodies[0]["model"] == MODEL
            assert bodies[0]["messages"][1]["content"][0]["signature"] == "synthetic-empty-original"
            assert req.model_dump_json() == before
        finally:
            await p.close()
    asyncio.run(run())


@pytest.mark.parametrize("streaming", [False, True])
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
def test_real_sdk_stream_order_signatures_private_text_and_stop(stop, blocks, streaming):
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
        p = provider(use_streaming=streaming, refusal_fallback_enabled=False)
        p._client = AsyncAnthropic(
            api_key="offline-placeholder",
            max_retries=0,
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(
                    lambda req: httpx2.Response(
                        200,
                        **(
                            {
                                "headers": {"content-type": "text/event-stream"},
                                "content": sse.encode(),
                            } if streaming else {"json": sdk_body(content=blocks, stop=stop)}
                        ),
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
