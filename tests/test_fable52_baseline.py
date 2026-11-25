"""Offline exact Fable/Mythos 5.2 baseline; not live acceptance or billed costs."""

import asyncio
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_anthropic import AnthropicProvider, _RuntimeModelInfo
from amplifier_module_provider_anthropic._cost import compute_cost
from tests._helpers import FakeCoordinator

MODEL = "claude-fable-5-2"


@pytest.fixture(params=[MODEL, "claude-mythos-5-2"])
def model(request):
    return request.param


def provider(**config):
    return AnthropicProvider(
        api_key="offline-placeholder",
        config={
            "default_model": MODEL,
            "enable_prompt_caching": False,
            "max_retries": 0,
            "reasoning_effort": "high",
            **config,
        },
        coordinator=FakeCoordinator(),
    )


def request(choice=None):
    return ChatRequest(
        messages=[Message(role="user", content="Hello")],
        tools=[ToolSpec(name="lookup", description="Lookup", parameters={"type": "object"})],
        tool_choice=choice,
    )


def assemble(p, req, **options):
    return p._assemble_request_params(
        req,
        request_options=options,
        request_caps=p._get_capabilities(options.get("model", p.default_model)),
    ).params


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"input_tokens": 1_000_000}, "8"),
        ({"output_tokens": 1_000_000}, "40"),
        ({"cache_read_input_tokens": 1_000_000}, ".20"),
        ({"cache_creation_input_tokens": 1_000_000}, "10"),
        ({"cache_creation_5m_input_tokens": 1_000_000}, "10"),
        ({"cache_creation_1h_input_tokens": 1_000_000}, "16"),
        ({
            "input_tokens": 1000, "output_tokens": 200,
            "cache_read_input_tokens": 3000, "cache_creation_input_tokens": 1000,
            "cache_creation_5m_input_tokens": 400,
            "cache_creation_1h_input_tokens": 600,
        }, ".0302"),
    ],
)
def test_exact_standard_cost_categories_and_mixed_ttl(model, usage, expected):
    assert compute_cost(model, **usage) == Decimal(expected)


@pytest.mark.parametrize("model", [
    MODEL + "-20261109", MODEL + "-latest", "claude-fable-5-20",
    "claude-mythos-5-2-20261109", "claude-mythos-5-2-latest", "claude-mythos-5-20",
    "claude-mythos-preview",
])
def test_no_guessed_pricing_or_cold_budget(model):
    assert compute_cost(model, input_tokens=1000) is None
    p = provider()
    assert p._budget_capabilities_for(model) is None
    assert p._client is None


@pytest.mark.parametrize("choice", ["required", {"type": "any"}, {"type": "tool", "name": "lookup"}])
@pytest.mark.parametrize("location", ["request", "options", "expert", "extra_body"])
def test_forced_choice_rejected_including_expert_wire(model, choice, location):
    config = {}
    options = {}
    if location == "options":
        options["tool_choice"] = choice
    elif location in {"expert", "extra_body"}:
        wire = {"type": "any"} if choice == "required" else choice
        config["extra_request_params"] = (
            {"tool_choice": wire} if location == "expert"
            else {"extra_body": {"tool_choice": wire}}
        )
    p = provider(default_model=model, **config)
    req = request(choice if location == "request" else "auto")
    before = deepcopy(req.model_dump())
    config_before, options_before = deepcopy(config), deepcopy(options)
    with pytest.raises(InvalidRequestError, match=model) as caught:
        assemble(p, req, **options)
    assert caught.value.model == model
    assert caught.value.status_code == 400
    assert "claude-opus-5-5" not in str(caught.value)
    assert req.model_dump() == before
    assert config == config_before and options == options_before
    assert p._client is None


@pytest.mark.parametrize("nested", [False, True])
def test_expert_model_override_cannot_bypass_rejection(model, nested):
    extra = {"model": model, "tool_choice": {"type": "tool", "name": "lookup"}}
    if nested:
        extra = {"extra_body": extra}
    p = provider(default_model="claude-fable-5-1", extra_request_params=extra)
    with pytest.raises(InvalidRequestError, match=model):
        assemble(p, request("auto"))


@pytest.mark.parametrize("choice", [None, "auto", "none", {"type": "auto"}, {"type": "none"}])
def test_ordinary_choices_still_assemble(model, choice):
    params = assemble(provider(default_model=model), request(choice))
    assert params["model"] == model
    assert params["max_tokens"] == 128000
    if choice is None:
        assert "tool_choice" not in params
    else:
        expected = choice if isinstance(choice, dict) else {"type": choice}
        assert params["tool_choice"] == expected


@pytest.mark.parametrize("nested", [False, True])
def test_final_wire_override_precedence_and_no_mutation(model, nested):
    extra = {"tool_choice": {"type": "auto"}}
    if nested:
        extra = {"tool_choice": {"type": "any"}, "extra_body": extra}
    p = provider(default_model=model, extra_request_params=extra)
    req = request("required")
    before, req_before = deepcopy(extra), deepcopy(req.model_dump())
    params = assemble(p, req)
    assert params.get("extra_body", params)["tool_choice"] == {"type": "auto"}
    assert extra == before
    assert req.model_dump() == req_before
    # An override away from the exact model preserves existing expert policy.
    extra = {"model": "claude-mythos-5-1"}
    if nested:
        extra = {"extra_body": extra}
    p = provider(default_model=model, extra_request_params=extra)
    assert assemble(p, req)["tool_choice"] == {"type": "any"}


def test_exact_capability_and_prior_models_preserved(model):
    p = provider()
    caps = p._get_capabilities(model)
    assert not caps.supports_forced_tool_choice
    assert caps.supports_1m and caps.max_output_tokens == 128000
    assert not caps.supports_speed and not caps.supports_sampling
    assert not caps.supports_manual_thinking and caps.thinking_always_on
    overlay = p._apply_runtime_capability_overrides(
        caps, _RuntimeModelInfo(max_input_tokens=1_000_000)
    )
    assert not overlay.supports_forced_tool_choice
    for old in ["claude-fable-5", "claude-fable-5-1", "claude-mythos-5",
                "claude-mythos-5-1", "claude-mythos-preview", model + "-latest"]:
        assert p._get_capabilities(old).supports_forced_tool_choice
        assert assemble(provider(default_model=old), request("required"))["tool_choice"] == {"type": "any"}


@pytest.mark.parametrize("old", ["claude-mythos-5", "claude-mythos-5-1"])
def test_older_mythos_remains_unpriced_with_existing_cold_budget(old):
    p = provider(default_model=old)
    assert compute_cost(old, input_tokens=1000) is None
    caps = p._budget_capabilities_for(old)
    assert caps is not None and p._budget_input_limit(old, caps) == 200000
    assert p._client is None


@pytest.mark.parametrize(("enabled", "limit"), [(False, 200_000), (True, 1_000_000)])
def test_cold_budget_recognition_preserves_1m_policy(model, enabled, limit):
    p = provider(default_model=model, enable_1m_context=enabled)
    assert not p._runtime_model_info_cache
    caps = p._budget_capabilities_for(model)
    assert caps is not None
    assert p._budget_input_limit(model, caps) == limit
    assert p.get_info().defaults["context_window"] == limit
    assert p._client is None
    p.client.messages.count_tokens = AsyncMock(return_value=SimpleNamespace(input_tokens=100))
    result = asyncio.run(p.request_budget(request(), context_estimate=100))
    assert result is not None
    assert result["input_limit_tokens"] == limit
    assert result["max_output_tokens"] == 128000
    assert result["measurement"]["kind"] == "provider_count"
    assert result["measurement"]["input_tokens"] == 100


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("entry", ["complete", "request_budget"])
@pytest.mark.parametrize("choice", ["required", {"type": "any"}, {"type": "tool", "name": "lookup"}])
@pytest.mark.parametrize("location", ["request", "options", "expert", "extra_body"])
def test_public_forced_choice_refuses_without_generation(model, streaming, entry, choice, location):
    config, options = {}, {}
    if location == "options":
        options["tool_choice"] = choice
    elif location in {"expert", "extra_body"}:
        extra = {"model": model, "tool_choice": {"type": "any"} if choice == "required" else choice}
        config = {"default_model": "claude-mythos-5-1", "extra_request_params":
                  extra if location == "expert" else {"extra_body": extra}}
    p = provider(**{"default_model": model, "use_streaming": streaming, **config})
    req = request(choice if location == "request" else "auto")
    before = deepcopy(req.model_dump())
    config_before, options_before = deepcopy(config), deepcopy(options)
    assert not p._runtime_model_info_cache and p._client is None
    # A cold local refusal needs neither Models metadata nor a Messages client.
    with pytest.raises(InvalidRequestError, match=model):
        asyncio.run(p.complete(req, **options) if entry == "complete"
                    else p.request_budget(req, context_estimate=100, request_options=options))
    assert req.model_dump() == before
    assert config == config_before and options == options_before
    assert p._client is None


def test_budget_abstains_on_expert_model_mismatch(model):
    p = provider(default_model=model, extra_request_params={"model": "claude-sonnet-5"})
    assert asyncio.run(p.request_budget(request(), context_estimate=100)) is None
    assert p._client is None


def test_runtime_budget_metadata_precedence(model):
    p = provider(default_model=model)
    p._runtime_model_info_cache[model] = _RuntimeModelInfo(max_input_tokens=123456)
    caps = p._budget_capabilities_for(model)
    assert p._budget_input_limit(model, caps) == 123456
    assert p._client is None


@pytest.mark.parametrize("choice", ["required", {"type": "tool", "name": "lookup"}])
def test_explicit_none_choice_rejects_before_metadata_or_client(model, choice):
    p = provider(default_model=model)
    p._get_request_capabilities = AsyncMock(side_effect=AssertionError("metadata reached"))
    req = request(choice)
    before = deepcopy(req.model_dump())
    with pytest.raises(InvalidRequestError, match=model):
        asyncio.run(p.complete(req, tool_choice=None))
    p._get_request_capabilities.assert_not_called()
    assert p._client is None
    assert req.model_dump() == before


@pytest.mark.parametrize("choice", ["required", {"type": "tool", "name": "lookup"}])
@pytest.mark.parametrize("nested", [False, True])
def test_explicit_none_choice_preserves_public_expert_auto(model, choice, nested):
    import json
    import httpx2 as httpx
    from anthropic import AsyncAnthropic

    extra = {"tool_choice": {"type": "auto"}}
    if nested:
        extra = {"tool_choice": {"type": "any"}, "extra_body": extra}
    req = request(choice)
    before, extra_before = deepcopy(req.model_dump()), deepcopy(extra)
    wires = []

    def respond(wire):
        wires.append(json.loads(wire.content))
        return httpx.Response(200, json={
            "id": "msg_offline", "type": "message", "role": "assistant",
            "model": model, "content": [], "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, "inference_geo": "global"},
        })

    async def run():
        p = provider(default_model=model, use_streaming=False, extra_request_params=extra)
        p._get_request_capabilities = AsyncMock(return_value=p._get_capabilities(model))
        p._client = AsyncAnthropic(
            api_key="offline-placeholder", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        try:
            await p.complete(req, tool_choice=None)
        finally:
            await p.close()

    asyncio.run(run())
    assert len(wires) == 1
    assert wires[0]["tool_choice"] == {"type": "auto"}
    assert req.model_dump() == before and extra == extra_before