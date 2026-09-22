"""Opus 5.5 forced tool_choice gating (messages + count_tokens).

Anthropic's Opus 5.5 "What's new" migration guidance documents that forced
tool_choice (type "any" or a named "tool") is rejected with HTTP 400. A
deterministic 400 is useless to callers, so the provider downgrades a forced
choice to {"type": "auto"} (preserving disable_parallel_tool_use) and reports
the downgrade via ChatResponse.degradation, rather than letting every such
request fail. Opus 5 and earlier are unaffected (they still send the forced
choice unchanged).
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import DummyResponse, FakeCoordinator


def _make_provider(model: str = "claude-opus-5-5", **overrides: Any) -> AnthropicProvider:
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
    tool = ToolSpec(name="f", description="a tool", parameters={"type": "object", "properties": {}})
    return ChatRequest(
        messages=[Message(role="user", content="hi")],
        tools=[tool],
        tool_choice=tool_choice,
    )


def _run(provider: AnthropicProvider, request: ChatRequest, **kwargs: Any):
    raw_response = MagicMock()
    raw_response.parse = AsyncMock(
        return_value=DummyResponse(model=kwargs.get("model", provider.default_model))
    )
    raw_response.headers = {}
    create = AsyncMock(return_value=raw_response)
    provider.client.messages.with_raw_response.create = create

    response = asyncio.run(provider.complete(request, **kwargs))
    assert create.await_count == 1
    return create.call_args.kwargs, response


class TestOpus55ForcedToolChoiceDowngraded:
    def test_required_downgraded_to_auto(self):
        provider = _make_provider()
        params, response = _run(provider, _tool_request("required"))
        assert params["tool_choice"] == {"type": "auto"}
        assert response.degradation is not None
        assert "auto" in response.degradation.actual

    def test_type_any_downgraded_to_auto(self):
        provider = _make_provider()
        params, response = _run(provider, _tool_request({"type": "any"}))
        assert params["tool_choice"] == {"type": "auto"}
        assert response.degradation is not None

    def test_named_tool_choice_downgraded_to_auto(self):
        provider = _make_provider()
        params, response = _run(
            provider, _tool_request({"type": "tool", "name": "f"})
        )
        assert params["tool_choice"] == {"type": "auto"}
        assert response.degradation is not None

    def test_options_kwarg_tool_choice_also_gated(self):
        provider = _make_provider()
        params, response = _run(
            provider, _tool_request(None), tool_choice={"type": "any"}
        )
        assert params["tool_choice"] == {"type": "auto"}
        assert response.degradation is not None

    def test_disable_parallel_tool_use_preserved_through_downgrade(self):
        provider = _make_provider()
        params, _ = _run(
            provider,
            _tool_request({"type": "any", "disable_parallel_tool_use": True}),
        )
        assert params["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }

    def test_none_and_auto_unchanged_on_opus_55(self):
        provider = _make_provider()
        params, response = _run(provider, _tool_request("none"))
        assert params["tool_choice"] == {"type": "none"}
        assert response.degradation is None

        provider = _make_provider()
        params, response = _run(provider, _tool_request("auto"))
        assert params["tool_choice"] == {"type": "auto"}
        assert response.degradation is None


class TestOpus5Unchanged:
    def test_opus_5_required_still_sends_any_forced_choice(self):
        provider = _make_provider(model="claude-opus-5")
        params, response = _run(provider, _tool_request("required"))
        assert params["tool_choice"] == {"type": "any"}
        assert response.degradation is None

    def test_opus_5_named_tool_choice_unchanged(self):
        provider = _make_provider(model="claude-opus-5")
        params, response = _run(
            provider, _tool_request({"type": "tool", "name": "f"})
        )
        assert params["tool_choice"] == {"type": "tool", "name": "f"}
        assert response.degradation is None


class TestOpus55ExtraRequestParamsBypass:
    """extra_request_params is documented "user wins loudly": it is applied
    AFTER assembly (including this gate), so it deliberately bypasses it."""

    def test_extra_request_params_bypasses_the_gate(self):
        provider = _make_provider(
            extra_request_params={"tool_choice": {"type": "any"}}
        )
        params, _ = _run(provider, _tool_request("auto"))
        assert params["tool_choice"] == {"type": "any"}


class TestOpus55CountTokensInheritsGate:
    """_count_tokens_params projects the already-assembled dispatch params, so
    the gate is inherited automatically -- no separate gate is implemented on
    the count_tokens path."""

    def test_count_tokens_body_has_gated_tool_choice(self):
        provider = _make_provider()
        request = _tool_request("required")
        caps = provider._budget_capabilities_for("claude-opus-5-5")
        assert caps is not None
        assembly = provider._assemble_request_params(
            request, request_options={}, request_caps=caps
        )
        assert assembly is not None
        assert assembly.params["tool_choice"] == {"type": "auto"}
        counted = provider._count_tokens_params(assembly.params)
        assert counted["tool_choice"] == {"type": "auto"}
