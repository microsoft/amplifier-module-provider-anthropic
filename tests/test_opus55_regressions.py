"""Focused offline regressions for Claude Opus 5.5 and persisted histories."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any

import pytest
from anthropic.types import Message as SDKMessage
from amplifier_core.llm_errors import InvalidRequestError as KernelInvalidRequestError
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolSpec,
)

from amplifier_module_provider_anthropic import AnthropicProvider, _RuntimeModelInfo
from amplifier_module_provider_anthropic._cost import compute_cost


MODEL = "claude-opus-5-5"


def _provider() -> AnthropicProvider:
    return AnthropicProvider(
        api_key="test-key",
        config={
            "default_model": MODEL,
            "enable_prompt_caching": False,
            "max_retries": 0,
        },
    )


def _request(
    *,
    tool_choice: str | dict[str, Any] | None = None,
    tools: list[ToolSpec] | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = 123,
) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="hello")],
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        model=MODEL,
        max_output_tokens=max_output_tokens,
    )


def _assemble(
    request: ChatRequest, **options: Any
) -> dict[str, Any]:
    provider = _provider()
    assembly = provider._assemble_request_params(
        request,
        request_options={"model": MODEL, **options},
        request_caps=provider._get_capabilities(MODEL),
    )
    assert assembly is not None
    return assembly.params


def _function_tool(name: str = "computer") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="ordinary function",
        parameters={"type": "object", "properties": {}},
    )


@pytest.mark.parametrize("choice", ["auto", "none"])
def test_opus55_normalizes_safe_tool_choices_from_request(choice: str) -> None:
    params = _assemble(_request(tool_choice=choice, tools=[_function_tool()]))
    assert params["tool_choice"] == {"type": choice}


@pytest.mark.parametrize(
    "choice",
    [
        "required",
        {"type": "any"},
        {"type": "tool", "name": "computer"},
    ],
)
def test_opus55_rejects_forced_tool_choice_from_request(choice: Any) -> None:
    with pytest.raises(KernelInvalidRequestError, match="only tool_choice 'auto' or 'none'"):
        _assemble(_request(tool_choice=choice, tools=[_function_tool()]))


@pytest.mark.parametrize(
    "choice",
    [
        "required",
        {"type": "any"},
        {"type": "tool", "name": "computer"},
    ],
)
def test_opus55_rejects_forced_tool_choice_from_kwargs(choice: Any) -> None:
    with pytest.raises(KernelInvalidRequestError, match="only tool_choice 'auto' or 'none'"):
        _assemble(_request(tools=[_function_tool()]), tool_choice=choice)


def test_opus55_rejects_forced_tool_choice_without_tools() -> None:
    with pytest.raises(KernelInvalidRequestError, match="only tool_choice 'auto' or 'none'"):
        _assemble(_request(tool_choice="required"))


def test_opus55_rejects_only_native_computer_declarations() -> None:
    native = _function_tool()
    setattr(native, "type", "computer_20251124")
    with pytest.raises(KernelInvalidRequestError, match="native computer-toolset"):
        _assemble(_request(tools=[native]))

    params = _assemble(_request(tools=[_function_tool("computer")]))
    assert params["tools"][0]["name"] == "computer"
    assert "type" not in params["tools"][0]


def test_opus55_requires_adaptive_thinking_without_expanding_output_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _provider()
    request = _request(reasoning_effort="high", max_output_tokens=123)
    assembly = provider._assemble_request_params(
        request,
        request_options={"model": MODEL, "extended_thinking": False},
        request_caps=provider._get_capabilities(MODEL),
        emit_diagnostics=True,
    )
    assert assembly is not None
    assert assembly.params["thinking"] == {"type": "adaptive"}
    assert assembly.params["output_config"] == {"effort": "high"}
    assert assembly.params["max_tokens"] == 123
    assert "cannot disable model thinking" in caplog.text


def test_opus55_keeps_explicit_kwargs_effort_when_thinking_opted_out() -> None:
    params = _assemble(
        _request(max_output_tokens=123),
        extended_thinking=False,
        effort="high",
    )
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "high"}
    assert params["max_tokens"] == 123


def test_opus55_real_sdk_mock_transport_receives_serialized_safe_request() -> None:
    """Exercise the installed SDK transport, not an AsyncMock call boundary."""
    httpx = pytest.importorskip("httpx")
    from anthropic import AsyncAnthropic

    captured: list[dict[str, Any]] = []

    async def handler(request: Any) -> Any:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    async def run() -> None:
        provider = _provider()
        provider._runtime_model_info_cache[MODEL] = None
        provider._client = AsyncAnthropic(
            api_key="test-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        try:
            response = await provider.complete(
                _request(
                    tools=[_function_tool()],
                    tool_choice="auto",
                    reasoning_effort="medium",
                    max_output_tokens=123,
                ),
                extended_thinking=False,
            )
            assert response.content == [TextBlock(text="ok")]
        finally:
            await provider.close()

    asyncio.run(run())
    assert captured[0]["thinking"] == {"type": "adaptive"}
    assert captured[0]["tool_choice"] == {"type": "auto"}
    assert captured[0]["max_tokens"] == 123


def test_opus55_costs_and_fast_multiplier_are_exact() -> None:
    assert compute_cost(
        MODEL,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_5m_input_tokens=1_000_000,
        cache_creation_1h_input_tokens=1_000_000,
        speed="fast",
    ) == Decimal("74.40")
    assert compute_cost(MODEL, input_tokens=1_000_000, speed="standard") == Decimal(
        "4.00"
    )


def test_opus55_static_budget_and_overlay_keep_static_fields() -> None:
    provider = _provider()
    assert provider._budget_capabilities_for(MODEL) is not None
    base = provider._get_capabilities(MODEL)
    overlaid = provider._apply_runtime_capability_overrides(
        base,
        _RuntimeModelInfo(
            max_input_tokens=200_000,
            max_tokens=128_000,
            supports_thinking=True,
            supports_adaptive_thinking=True,
        ),
    )
    assert overlaid.min_cacheable_tokens == 512
    assert overlaid.manual_thinking_deprecated is base.manual_thinking_deprecated
    assert overlaid.requires_adaptive_thinking is True


def test_sdk_response_and_persisted_core_message_round_trip_in_order() -> None:
    provider = _provider()
    sdk_response = SDKMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "content": [
                {"type": "thinking", "thinking": "first", "signature": "signature-1"},
                {"type": "thinking", "thinking": "second", "signature": ""},
                {"type": "text", "text": "between"},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
    )
    response = provider._convert_to_chat_response(sdk_response)
    assert [block.type for block in response.content or []] == [
        "thinking",
        "thinking",
        "text",
        "redacted_thinking",
        "tool_call",
    ]
    assert isinstance(response.content[3], RedactedThinkingBlock)

    persisted = Message(role="assistant", content=response.content or []).model_dump()
    wire = provider._convert_messages([persisted])
    assert [block["type"] for block in wire[0]["content"]] == [
        "thinking",
        "thinking",
        "text",
        "redacted_thinking",
        "tool_use",
    ]
    assert wire[0]["content"][1]["signature"] == ""
    assert wire[0]["content"][3]["data"] == "opaque"


def test_structured_history_wins_and_missing_legacy_calls_are_reconstructed() -> None:
    provider = _provider()
    canonical = [
        ThinkingBlock(thinking="private", signature="sig").model_dump(),
        TextBlock(text="visible").model_dump(),
        ToolCallBlock(id="toolu_current", name="current", input={"current": True}).model_dump(),
    ]
    wire = provider._convert_messages(
        [
            {
                "role": "assistant",
                "content": canonical,
                "thinking_block": {"type": "thinking", "thinking": "stale"},
                "tool_calls": [
                    {
                        "id": "toolu_current",
                        "tool": "stale",
                        "arguments": {"stale": True},
                    },
                    {
                        "id": "toolu_missing",
                        "name": "reconstructed",
                        "input": {"missing": True},
                    },
                ],
            }
        ]
    )
    blocks = wire[0]["content"]
    assert [block["type"] for block in blocks] == [
        "thinking",
        "text",
        "tool_use",
        "tool_use",
    ]
    assert blocks[2] == {
        "type": "tool_use",
        "id": "toolu_current",
        "name": "current",
        "input": {"current": True},
    }
    assert blocks[3]["id"] == "toolu_missing"


def test_text_only_structured_history_reconstructs_separate_tool_calls() -> None:
    provider = _provider()
    wire = provider._convert_messages(
        [
            {
                "role": "assistant",
                "content": [TextBlock(text="visible").model_dump()],
                "tool_calls": [
                    {
                        "id": "toolu_reconstructed",
                        "tool": "lookup",
                        "arguments": {"query": "current"},
                    }
                ],
            }
        ]
    )
    assert wire == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "visible"},
                {
                    "type": "tool_use",
                    "id": "toolu_reconstructed",
                    "name": "lookup",
                    "input": {"query": "current"},
                },
            ],
        }
    ]


def test_pydantic_structured_tool_calls_authorize_tool_results() -> None:
    provider = _provider()
    wire = provider._convert_messages(
        [
            {
                "role": "assistant",
                "content": [
                    ToolCallBlock(
                        id="toolu_pydantic",
                        name="lookup",
                        input={"query": "current"},
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_pydantic",
                "content": "result",
            },
        ]
    )

    assert wire == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_pydantic",
                    "name": "lookup",
                    "input": {"query": "current"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_pydantic",
                    "content": "result",
                }
            ],
        },
    ]