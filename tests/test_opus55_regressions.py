"""Focused offline regressions for Claude Opus 5.5 and persisted histories."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any

import httpx2 as httpx
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
PREVIOUS_OPUS_MODEL = "claude-opus-4-7-20260416"
FABLE_MODEL = "claude-fable-5-1"
MYTHOS_MODEL = "claude-mythos-5"


def _provider(**config_overrides: Any) -> AnthropicProvider:
    config = {
        "default_model": MODEL,
        "enable_prompt_caching": False,
        "max_retries": 0,
        "use_streaming": False,
    }
    config.update(config_overrides)
    return AnthropicProvider(
        api_key="test-key",
        config=config,
    )


def _request(
    *,
    tool_choice: str | dict[str, Any] | None = None,
    tools: list[ToolSpec] | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = 123,
    model: str = MODEL,
) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="hello")],
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        model=model,
        max_output_tokens=max_output_tokens,
    )


def _assemble(
    request: ChatRequest, *, model: str = MODEL, **options: Any
) -> dict[str, Any]:
    provider = _provider(default_model=model)
    assembly = provider._assemble_request_params(
        request,
        request_options={"model": model, **options},
        request_caps=provider._get_capabilities(model),
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
    assert assembly.params["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }
    assert assembly.params["output_config"] == {"effort": "high"}
    assert assembly.params["max_tokens"] == 123
    assert "cannot disable model thinking" in caplog.text


def test_opus55_adaptive_thinking_defaults_display_without_manual_budget() -> None:
    provider = _provider()
    caps = provider._get_capabilities(MODEL)
    params = _assemble(
        _request(max_output_tokens=123),
        thinking_type="enabled",
        thinking_budget_tokens=8000,
    )

    assert caps.requires_adaptive_thinking is True
    assert caps.thinking_display_required is True
    assert caps.supports_manual_thinking is False
    assert caps.max_output_tokens == 128_000
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "budget_tokens" not in params["thinking"]
    assert params["max_tokens"] == 123


def test_opus55_adaptive_thinking_uses_configured_display() -> None:
    provider = _provider(thinking_display="omitted")
    request = _request()
    assembly = provider._assemble_request_params(
        request,
        request_options={"model": MODEL},
        request_caps=provider._get_capabilities(MODEL),
    )

    assert assembly is not None
    assert assembly.params["thinking"] == {"type": "adaptive", "display": "omitted"}


def test_opus55_adaptive_thinking_kwargs_display_overrides_config() -> None:
    provider = _provider(thinking_display="omitted")
    request = _request()
    assembly = provider._assemble_request_params(
        request,
        request_options={"model": MODEL, "thinking_display": "summarized"},
        request_caps=provider._get_capabilities(MODEL),
    )

    assert assembly is not None
    assert assembly.params["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }


def test_opus55_keeps_explicit_kwargs_effort_when_thinking_opted_out() -> None:
    params = _assemble(
        _request(max_output_tokens=123),
        extended_thinking=False,
        effort="high",
    )
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert params["output_config"] == {"effort": "high"}
    assert params["max_tokens"] == 123


def test_opus55_keeps_configured_effort_when_thinking_opted_out() -> None:
    provider = _provider(reasoning_effort="medium", extended_thinking=False)
    request = _request(max_output_tokens=123)
    assembly = provider._assemble_request_params(
        request,
        request_options={"model": MODEL},
        request_caps=provider._get_capabilities(MODEL),
    )

    assert assembly is not None
    assert assembly.params["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }
    assert assembly.params["output_config"] == {"effort": "medium"}
    assert assembly.params["max_tokens"] == 123


def test_previous_opus_keeps_explicit_kwargs_effort_when_thinking_opted_out() -> None:
    params = _assemble(
        _request(model=PREVIOUS_OPUS_MODEL, max_output_tokens=123),
        model=PREVIOUS_OPUS_MODEL,
        extended_thinking=False,
        effort="high",
    )

    assert params["output_config"] == {"effort": "high"}


@pytest.mark.parametrize("model", [FABLE_MODEL, MYTHOS_MODEL])
def test_always_on_models_keep_their_omit_thinking_parameter_behavior(
    model: str,
) -> None:
    params = _assemble(
        _request(model=model, reasoning_effort="high"),
        model=model,
    )

    assert "thinking" not in params


def test_opus55_real_sdk_mock_transport_receives_serialized_safe_request() -> None:
    """Exercise the installed SDK transport, not an AsyncMock call boundary."""
    from anthropic import AsyncAnthropic

    captured: list[dict[str, Any]] = []

    async def handler(request: Any) -> Any:
        assert request.url.path == "/v1/messages"
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
        assert provider.use_streaming is False
        provider._runtime_model_info_cache[MODEL] = None
        provider._client = AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
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
    assert len(captured) == 1
    assert captured[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
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
    canonical = [TextBlock(text="visible").model_dump()]
    original_canonical = [dict(block) for block in canonical]
    wire = provider._convert_messages(
        [
            {
                "role": "assistant",
                "content": canonical,
                "thinking_block": {
                    "type": "thinking",
                    "thinking": "legacy private",
                    "signature": "legacy-signature",
                },
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
                {
                    "type": "thinking",
                    "thinking": "legacy private",
                    "signature": "legacy-signature",
                },
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
    assert canonical == original_canonical


@pytest.mark.parametrize(
    ("content", "tool_call", "expected_blocks"),
    [
        (
            "",
            {"id": "toolu_empty", "name": "lookup", "input": {}},
            [
                {
                    "type": "tool_use",
                    "id": "toolu_empty",
                    "name": "lookup",
                    "input": {},
                }
            ],
        ),
        (
            "visible",
            {"id": "toolu_text", "tool": "lookup", "arguments": {"query": "now"}},
            [
                {"type": "text", "text": "visible"},
                {
                    "type": "tool_use",
                    "id": "toolu_text",
                    "name": "lookup",
                    "input": {"query": "now"},
                },
            ],
        ),
    ],
)
def test_legacy_assistant_text_with_tool_calls_skips_only_empty_text(
    content: str, tool_call: dict[str, Any], expected_blocks: list[dict[str, Any]]
) -> None:
    provider = _provider()
    wire = provider._convert_messages(
        [{"role": "assistant", "content": content, "tool_calls": [tool_call]}]
    )

    assert wire == [{"role": "assistant", "content": expected_blocks}]


def test_empty_legacy_text_with_thinking_block_skips_empty_text() -> None:
    provider = _provider()
    wire = provider._convert_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "thinking_block": {
                    "type": "thinking",
                    "thinking": "private",
                    "signature": "sig",
                },
            }
        ]
    )

    assert wire == [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private", "signature": "sig"}
            ],
        }
    ]


def test_ordinary_empty_legacy_assistant_content_stays_scalar() -> None:
    provider = _provider()

    assert provider._convert_messages([{"role": "assistant", "content": ""}]) == [
        {"role": "assistant", "content": ""}
    ]


def test_structured_canonical_tool_call_dedupes_legacy_alias() -> None:
    provider = _provider()
    wire = provider._convert_messages(
        [
            {
                "role": "assistant",
                "content": [
                    TextBlock(text="visible").model_dump(),
                    ToolCallBlock(
                        id="toolu_current", name="current", input={"current": True}
                    ).model_dump(),
                ],
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

    assert [block["id"] for block in wire[0]["content"] if block["type"] == "tool_use"] == [
        "toolu_current",
        "toolu_missing",
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


@pytest.mark.parametrize("content", ["visible", [{"type": "text", "text": "visible"}]])
def test_empty_legacy_thinking_field_does_not_create_empty_block(content: Any) -> None:
    wire = _provider()._convert_messages(
        [{"role": "assistant", "content": content, "thinking_block": {}}]
    )
    assert wire[0]["content"] in ("visible", [{"type": "text", "text": "visible"}])


@pytest.mark.parametrize("content", ["visible", [TextBlock(text="visible")]])
def test_refusal_strip_clears_legacy_thinking_without_mutating_history(content: Any) -> None:
    thinking = {"type": "thinking", "thinking": "private", "signature": "original"}
    request = ChatRequest(
        messages=[Message(role="assistant", content=content, thinking_block=thinking)]
    )
    stripped = AnthropicProvider._strip_thinking_blocks(request)
    assert "thinking_block" not in stripped.messages[0].model_dump()
    assert request.messages[0].model_dump()["thinking_block"] == thinking
    wire = _provider()._convert_messages([stripped.messages[0].model_dump()])
    assert wire[0]["content"] in ("visible", [{"type": "text", "text": "visible"}])


def test_pydantic_legacy_tool_calls_authorize_tool_results() -> None:
    provider = _provider()
    wire = provider._convert_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    ToolCallBlock(
                        id="toolu_legacy", name="lookup", input={"query": "current"}
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_legacy",
                "content": "result",
            },
        ]
    )

    assert wire[1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_legacy",
                "content": "result",
            }
        ],
    }