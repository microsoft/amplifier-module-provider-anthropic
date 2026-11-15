"""Offline Fable 5.2 baseline; not live vendor acceptance or billed costs."""

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
def test_exact_standard_cost_categories_and_mixed_ttl(usage, expected):
    assert compute_cost(MODEL, **usage) == Decimal(expected)


@pytest.mark.parametrize("model", [
    MODEL + "-20261109", MODEL + "-latest", "claude-fable-5-20", "claude-mythos-5-2",
])
def test_no_guessed_pricing_or_cold_budget(model):
    assert compute_cost(model, input_tokens=1000) is None
    p = provider()
    assert p._budget_capabilities_for(model) is None
    assert p._client is None


@pytest.mark.parametrize("choice", ["required", {"type": "any"}, {"type": "tool", "name": "lookup"}])
@pytest.mark.parametrize("location", ["request", "options", "expert", "extra_body"])
def test_forced_choice_rejected_including_expert_wire(choice, location):
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
    p = provider(**config)
    req = request(choice if location == "request" else "auto")
    before = deepcopy(req.model_dump())
    with pytest.raises(InvalidRequestError, match=MODEL) as caught:
        assemble(p, req, **options)
    assert caught.value.model == MODEL
    assert caught.value.status_code == 400
    assert "claude-opus-5-5" not in str(caught.value)
    assert req.model_dump() == before
    assert p._client is None


@pytest.mark.parametrize("extra", [
    {"model": MODEL, "tool_choice": {"type": "any"}},
    {"extra_body": {"model": MODEL, "tool_choice": {"type": "tool", "name": "lookup"}}},
])
def test_expert_model_override_cannot_bypass_rejection(extra):
    p = provider(default_model="claude-fable-5-1", extra_request_params=extra)
    with pytest.raises(InvalidRequestError, match=MODEL):
        assemble(p, request("auto"))


@pytest.mark.parametrize("choice", [None, "auto", "none", {"type": "auto"}, {"type": "none"}])
def test_ordinary_choices_still_assemble(choice):
    params = assemble(provider(), request(choice))
    assert params["model"] == MODEL
    assert params["max_tokens"] == 128000
    if choice is None:
        assert "tool_choice" not in params
    else:
        expected = choice if isinstance(choice, dict) else {"type": choice}
        assert params["tool_choice"] == expected


def test_final_wire_override_precedence_and_no_mutation():
    extra = {"extra_body": {"tool_choice": {"type": "auto"}}}
    p = provider(extra_request_params=extra)
    req = request("required")
    before = deepcopy(extra)
    params = assemble(p, req)
    assert params["extra_body"]["tool_choice"] == {"type": "auto"}
    assert extra == before
    # An override away from the exact model preserves existing expert policy.
    p = provider(extra_request_params={"model": "claude-fable-5-1"})
    assert assemble(p, req)["tool_choice"] == {"type": "any"}


def test_exact_capability_and_prior_models_preserved():
    p = provider()
    caps = p._get_capabilities(MODEL)
    assert not caps.supports_forced_tool_choice
    assert caps.supports_1m and caps.max_output_tokens == 128000
    assert not caps.supports_speed and not caps.supports_sampling
    assert not caps.supports_manual_thinking and caps.thinking_always_on
    overlay = p._apply_runtime_capability_overrides(
        caps, _RuntimeModelInfo(max_input_tokens=1_000_000)
    )
    assert not overlay.supports_forced_tool_choice
    for old in ["claude-fable-5", "claude-fable-5-1", "claude-mythos-5-2", MODEL + "-latest"]:
        assert p._get_capabilities(old).supports_forced_tool_choice
        assert assemble(provider(default_model=old), request("required"))["tool_choice"] == {"type": "any"}


@pytest.mark.parametrize(("enabled", "limit"), [(False, 200_000), (True, 1_000_000)])
def test_cold_budget_recognition_preserves_1m_policy(enabled, limit):
    p = provider(enable_1m_context=enabled)
    assert not p._runtime_model_info_cache
    caps = p._budget_capabilities_for(MODEL)
    assert caps is not None
    assert p._budget_input_limit(MODEL, caps) == limit
    assert p.get_info().defaults["context_window"] == limit
    assert p._client is None
    p.client.messages.count_tokens = AsyncMock(return_value=SimpleNamespace(input_tokens=100))
    result = asyncio.run(p.request_budget(request(), context_estimate=100))
    assert result is not None
    assert result["input_limit_tokens"] == limit
    assert result["max_output_tokens"] == 128000
    assert result["measurement"]["kind"] == "provider_count"