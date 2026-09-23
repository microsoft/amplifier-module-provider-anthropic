"""Opus 5.5 forced tool_choice gating (messages + count_tokens).

Anthropic's Opus 5.5 "What's new" migration guidance documents that forced
tool_choice (type "any" or a named "tool") is rejected with HTTP 400, on both
the Messages and count_tokens endpoints. A deterministic 400 tells the caller
nothing new, so the provider raises a local, actionable
KernelInvalidRequestError before any HTTP request -- never sending the
request in the first place. Opus 5 and earlier are unaffected (they still
send the forced choice unchanged).
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import InvalidRequestError as KernelInvalidRequestError
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import DummyResponse, FakeCoordinator


def _make_provider(
    model: str = "claude-opus-5-5", **overrides: Any
) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="x",
        config={
            "default_model": model,
            "max_retries": 0,
            "use_streaming": False,
            **overrides,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _tool_request(tool_choice: Any) -> ChatRequest:
    tool = ToolSpec(
        name="f", description="a tool", parameters={"type": "object", "properties": {}}
    )
    return ChatRequest(
        messages=[Message(role="user", content="hi")],
        tools=[tool],
        tool_choice=tool_choice,
    )


def _stub_create(provider: AnthropicProvider, model: str) -> AsyncMock:
    raw_response = MagicMock()
    raw_response.parse = AsyncMock(return_value=DummyResponse(model=model))
    raw_response.headers = {}
    create = AsyncMock(return_value=raw_response)
    provider.client.messages.with_raw_response.create = create
    return create


def _run(provider: AnthropicProvider, request: ChatRequest, **kwargs: Any):
    create = _stub_create(provider, kwargs.get("model", provider.default_model))
    response = asyncio.run(provider.complete(request, **kwargs))
    assert create.await_count == 1
    return create.call_args.kwargs, response


class TestOpus55ForcedToolChoiceRejected:
    def test_required_raises_before_dispatch(self):
        provider = _make_provider()
        create = _stub_create(provider, provider.default_model)
        with pytest.raises(KernelInvalidRequestError) as excinfo:
            asyncio.run(provider.complete(_tool_request("required")))
        assert create.await_count == 0
        assert "claude-opus-5-5" in str(excinfo.value)
        assert excinfo.value.provider == "anthropic"
        assert excinfo.value.model == "claude-opus-5-5"

    def test_type_any_raises_before_dispatch(self):
        provider = _make_provider()
        create = _stub_create(provider, provider.default_model)
        with pytest.raises(KernelInvalidRequestError):
            asyncio.run(provider.complete(_tool_request({"type": "any"})))
        assert create.await_count == 0

    def test_named_tool_choice_raises_before_dispatch(self):
        provider = _make_provider()
        create = _stub_create(provider, provider.default_model)
        with pytest.raises(KernelInvalidRequestError):
            asyncio.run(provider.complete(_tool_request({"type": "tool", "name": "f"})))
        assert create.await_count == 0

    def test_options_kwarg_tool_choice_also_gated(self):
        provider = _make_provider()
        create = _stub_create(provider, provider.default_model)
        with pytest.raises(KernelInvalidRequestError):
            asyncio.run(
                provider.complete(_tool_request(None), tool_choice={"type": "any"})
            )
        assert create.await_count == 0

    def test_options_kwarg_portable_string_required_also_gated(self):
        """A kwargs tool_choice="required" (the portable STRING form, not a
        pre-mapped wire dict) must go through the identical
        string -> {"type": "any"} mapping as request.tool_choice before the
        gate runs -- it must not bypass rejection just because it arrived
        as a bare string through kwargs instead of the request object."""
        provider = _make_provider()
        create = _stub_create(provider, provider.default_model)
        with pytest.raises(KernelInvalidRequestError):
            asyncio.run(provider.complete(_tool_request(None), tool_choice="required"))
        assert create.await_count == 0

    def test_options_kwarg_portable_string_auto_unaffected(self):
        """Regression guard for the normalization above: a kwargs
        tool_choice="auto" must still resolve to {"type": "auto"} and must
        not be rejected."""
        provider = _make_provider()
        params, _response = _run(provider, _tool_request(None), tool_choice="auto")
        assert params["tool_choice"] == {"type": "auto"}

    def test_error_message_advises_auto_or_none(self):
        provider = _make_provider()
        _stub_create(provider, provider.default_model)
        with pytest.raises(KernelInvalidRequestError) as excinfo:
            asyncio.run(provider.complete(_tool_request("required")))
        message = str(excinfo.value)
        assert "auto" in message
        assert "none" in message

    def test_none_and_auto_unchanged_on_opus_55(self):
        provider = _make_provider()
        params, _response = _run(provider, _tool_request("none"))
        assert params["tool_choice"] == {"type": "none"}

        provider = _make_provider()
        params, _response = _run(provider, _tool_request("auto"))
        assert params["tool_choice"] == {"type": "auto"}


class TestOpus5Unchanged:
    def test_opus_5_required_still_sends_any_forced_choice(self):
        provider = _make_provider(model="claude-opus-5")
        params, _response = _run(provider, _tool_request("required"))
        assert params["tool_choice"] == {"type": "any"}

    def test_opus_5_named_tool_choice_unchanged(self):
        provider = _make_provider(model="claude-opus-5")
        params, _response = _run(provider, _tool_request({"type": "tool", "name": "f"}))
        assert params["tool_choice"] == {"type": "tool", "name": "f"}


class TestOpus55ExtraRequestParamsBypass:
    """extra_request_params is documented "user wins loudly": it is applied
    AFTER assembly (including this gate), so it deliberately bypasses it."""

    def test_extra_request_params_bypasses_the_gate(self):
        provider = _make_provider(extra_request_params={"tool_choice": {"type": "any"}})
        params, _ = _run(provider, _tool_request("auto"))
        assert params["tool_choice"] == {"type": "any"}


class TestOpus55CountTokensInheritsGate:
    """The gate lives inside _assemble_request_params, so request_budget
    (which projects the same assembled params for count_tokens) raises
    identically -- no separate gate is implemented on that path, and no
    count_tokens call is ever made."""

    def test_request_budget_raises_before_any_count_tokens_call(self):
        provider = _make_provider()
        count_tokens = AsyncMock()
        provider.client.messages.count_tokens = count_tokens
        request = _tool_request("required")
        with pytest.raises(KernelInvalidRequestError):
            asyncio.run(provider.request_budget(request, context_estimate=100))
        assert count_tokens.await_count == 0
