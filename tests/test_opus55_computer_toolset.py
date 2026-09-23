"""Opus 5.5 computer-use: computer_toolset_20260801 translation.

Anthropic's Opus 5.5 migration guidance ("Migrate from computer_20251124")
replaces the legacy versioned computer_* native tool type with a single
computer_toolset_20260801 declaration, and shapes member tool_use blocks
differently on the wire (name = member action, toolset_name = "computer").
This provider translates both directions so an existing Amplifier tool that
dispatches on name="computer", arguments={"action": ...} keeps working
unchanged, and never advertises the legacy type/header on Opus 5.5+.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_anthropic import _computer_toolset, AnthropicProvider

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


def _computer_tool_request() -> ChatRequest:
    tool = ToolSpec(
        name="computer",
        parameters={},
        type="computer_20251124",
        display_width_px=1024,
        display_height_px=768,
    )
    return ChatRequest(messages=[Message(role="user", content="hi")], tools=[tool])


def _run(provider: AnthropicProvider, request: ChatRequest, response_content: list | None = None):
    raw_response = MagicMock()
    raw_response.parse = AsyncMock(
        return_value=DummyResponse(content=response_content or [], model=provider.default_model)
    )
    raw_response.headers = {}
    create = AsyncMock(return_value=raw_response)
    provider.client.messages.with_raw_response.create = create
    response = asyncio.run(provider.complete(request))
    assert create.await_count == 1
    return create.call_args.kwargs, response


class TestTranslateToolsUnit:
    def test_legacy_translated_to_toolset(self):
        tools = [
            {
                "type": "computer_20251124",
                "name": "computer",
                "display_width_px": 1024,
                "display_height_px": 768,
            }
        ]
        wire, aliases = _computer_toolset.translate_tools(
            tools, _computer_toolset.TOOLSET_TYPE
        )
        assert wire == [{"type": "computer_toolset_20260801", "configs": {"zoom": {"enabled": False}}}]
        assert aliases == {"computer": "computer"}

    def test_legacy_type_unchanged_for_non_toolset_target(self):
        tools = [{"type": "computer_20251124", "name": "computer"}]
        wire, aliases = _computer_toolset.translate_tools(tools, "computer_20251124")
        assert wire == tools
        assert aliases == {}

    def test_round_trip_member_call(self):
        name, args = _computer_toolset.to_amplifier_call(
            "left_click", {"coordinate": [1, 2]}, "computer", {"computer": "computer"}
        )
        assert name == "computer"
        assert args == {"action": "left_click", "coordinate": [1, 2]}

        wire_name, wire_input, toolset_name = _computer_toolset.to_wire_tool_call(
            name, args, target_type=_computer_toolset.TOOLSET_TYPE
        )
        assert wire_name == "left_click"
        assert wire_input == {"coordinate": [1, 2]}
        assert toolset_name == "computer"


class TestOpus55ComputerToolsetIntegration:
    def test_request_sends_toolset_type_no_legacy_header(self):
        provider = _make_provider()
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"] == [
            {"type": "computer_toolset_20260801", "configs": {"zoom": {"enabled": False}}}
        ]
        beta = (params.get("extra_headers") or {}).get("anthropic-beta", "")
        assert "computer-use-2025-11-24" not in beta
        assert "computer_20251124" not in str(params["tools"])
        assert params["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }

    def test_batch_actions_opt_in_does_not_force_serial_calls(self):
        provider = _make_provider(computer_batch_actions=True)
        params, _ = _run(provider, _computer_tool_request())
        assert "tool_choice" not in params

    def test_response_member_call_translated_to_legacy_shape(self):
        provider = _make_provider()
        member_block = type(
            "Block",
            (),
            {
                "type": "tool_use",
                "id": "t1",
                "name": "left_click",
                "input": {"coordinate": [1, 2]},
                "toolset_name": "computer",
            },
        )()
        _, response = _run(provider, _computer_tool_request(), response_content=[member_block])
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert call.name == "computer"
        assert call.arguments == {"action": "left_click", "coordinate": [1, 2]}

    def test_opus_5_unchanged_sends_legacy_type_and_header(self):
        provider = _make_provider(model="claude-opus-5")
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"][0]["type"] == "computer_20251124"
        beta = (params.get("extra_headers") or {}).get("anthropic-beta", "")
        assert "computer-use-2025-11-24" in beta

    def test_bedrock_gateway_override_sends_legacy_type_on_opus_55(self):
        provider = _make_provider(computer_use_tool_type="computer_20251124")
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"][0]["type"] == "computer_20251124"
