"""Captured-client tests for portable ChatRequest.tool_choice handling."""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import DummyResponse, FakeCoordinator


def _make_provider(**config_overrides: Any) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "default_model": "claude-sonnet-5",
            "max_retries": 0,
            "use_streaming": False,
            **config_overrides,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _capture_client_params(request: ChatRequest, **kwargs: Any) -> dict[str, Any]:
    provider = _make_provider()
    raw_response = MagicMock()
    raw_response.parse = AsyncMock(return_value=DummyResponse(model="claude-sonnet-5"))
    raw_response.headers = {}
    create = AsyncMock(return_value=raw_response)
    provider.client.messages.with_raw_response.create = create

    asyncio.run(provider.complete(request, **kwargs))

    assert create.await_count == 1
    return create.call_args.kwargs


def _native_computer_request(
    tool_choice: str | dict[str, Any] | None = None,
) -> ChatRequest:
    computer = ToolSpec(
        name="computer",
        description="Control the computer",
        parameters={"type": "object", "properties": {}},
    )
    setattr(computer, "type", "computer_20251124")
    return ChatRequest(
        messages=[Message(role="user", content="take a screenshot")],
        tools=[computer],
        tool_choice=tool_choice,
    )


def test_dated_native_computer_request_none_reaches_client() -> None:
    params = _capture_client_params(_native_computer_request("none"))

    assert params["tools"] == [{"name": "computer", "type": "computer_20251124"}]
    assert params["tool_choice"] == {"type": "none"}


def test_explicit_kwargs_wire_dict_wins_over_portable_request_choice() -> None:
    request_choice = {
        "type": "tool",
        "name": "computer",
        "disable_parallel_tool_use": True,
    }
    wire_choice = {
        "type": "tool",
        "name": "other_tool",
        "disable_parallel_tool_use": False,
    }
    params = _capture_client_params(
        _native_computer_request(request_choice), tool_choice=wire_choice
    )

    assert params["tool_choice"] == wire_choice


def test_unset_tool_choice_is_omitted() -> None:
    params = _capture_client_params(_native_computer_request())

    assert "tool_choice" not in params


@pytest.mark.parametrize(
    ("portable_choice", "wire_choice"),
    [
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
    ],
)
def test_portable_tool_choice_is_normalized_at_anthropic_boundary(
    portable_choice: str, wire_choice: dict[str, str]
) -> None:
    params = _capture_client_params(_native_computer_request(portable_choice))

    assert params["tool_choice"] == wire_choice


def test_request_wire_dict_is_preserved_with_extra_fields() -> None:
    wire_choice = {
        "type": "tool",
        "name": "computer",
        "disable_parallel_tool_use": True,
    }
    params = _capture_client_params(_native_computer_request(wire_choice))

    assert params["tool_choice"] == wire_choice


def test_web_search_only_request_applies_portable_tool_choice() -> None:
    provider = _make_provider(enable_web_search=True)
    raw_response = MagicMock()
    raw_response.parse = AsyncMock(return_value=DummyResponse(model="claude-sonnet-5"))
    raw_response.headers = {}
    create = AsyncMock(return_value=raw_response)
    provider.client.messages.with_raw_response.create = create

    request = ChatRequest(
        messages=[Message(role="user", content="search for weather")],
        tool_choice="none",
    )
    asyncio.run(provider.complete(request))

    assert create.await_count == 1
    params = create.call_args.kwargs
    assert params["tools"][0]["type"] == "web_search_20250305"
    assert params["tool_choice"] == {"type": "none"}


def test_installed_sdk_supports_the_none_wire_type() -> None:
    from anthropic.types.tool_choice_none_param import ToolChoiceNoneParam

    assert ToolChoiceNoneParam.__name__ == "ToolChoiceNoneParam"
