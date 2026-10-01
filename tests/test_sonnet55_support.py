"""Focused offline regressions for Claude Sonnet 5.5 support."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from amplifier_core.llm_errors import InvalidRequestError as KernelInvalidRequestError
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_anthropic import AnthropicProvider


MODEL = "claude-sonnet-5-5"


def _provider(**config_overrides: Any) -> AnthropicProvider:
    config = {
        "default_model": MODEL,
        "enable_prompt_caching": False,
        "max_retries": 0,
        "use_streaming": False,
    }
    config.update(config_overrides)
    return AnthropicProvider(api_key="test-key", config=config)


def _request(
    *,
    model: str = MODEL,
    tool_choice: str | dict[str, Any] | None = None,
    tools: list[ToolSpec] | None = None,
    reasoning_effort: str | None = None,
) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="hello")],
        model=model,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        max_output_tokens=123,
    )


def _native_computer(name: str = "desktop") -> ToolSpec:
    tool = ToolSpec(
        name=name,
        description="native computer",
        parameters={"type": "object", "properties": {}},
    )
    setattr(tool, "type", "computer_20251124")
    return tool


def _direct_native_computer(name: str = "desktop") -> ToolSpec:
    tool = _native_computer(name)
    setattr(tool, "type", "computer_toolset_20260801")
    return tool


def _assemble(request: ChatRequest, **options: Any) -> dict[str, Any]:
    provider = _provider(default_model=options.get("model", request.model or MODEL))
    model = options.pop("model", request.model or MODEL)
    assembly = provider._assemble_request_params(
        request,
        request_options={"model": model, **options},
        request_caps=provider._get_capabilities(model),
    )
    assert assembly is not None
    return assembly.params


def test_sonnet55_capabilities_and_exact_native_adapter_boundary() -> None:
    provider = _provider()
    caps = provider._get_capabilities(MODEL)

    assert caps.max_output_tokens == 128_000
    assert caps.supports_1m is True
    assert caps.min_cacheable_tokens == 512
    assert caps.supports_forced_tool_choice is False
    assert caps.computer_use_tool_type == "computer_toolset_20260801"
    assert _assemble(_request(tools=[_native_computer()]))["tools"] == [
        {"type": "computer_toolset_20260801", "configs": {"zoom": {"enabled": False}}}
    ]
    assert _assemble(
        _request(model="claude-sonnet-5-5-20260929", tools=[_native_computer()]),
        model="claude-sonnet-5-5-20260929",
    )["tools"][0]["type"] == "computer_toolset_20260801"
    with pytest.raises(KernelInvalidRequestError, match="Unrecognized Sonnet 5.5 model suffix"):
        _assemble(
            _request(model="claude-sonnet-5-5-latest", tools=[_native_computer()]),
            model="claude-sonnet-5-5-latest",
        )


def test_sonnet55_direct_native_toolset_without_configs_stays_bare() -> None:
    assert _assemble(_request(tools=[_direct_native_computer()]))["tools"] == [
        {"type": "computer_toolset_20260801"}
    ]


def test_sonnet55_native_auto_does_not_derive_parallel_restriction() -> None:
    assert _assemble(_request(tools=[_native_computer()]))["tool_choice"] == {
        "type": "auto"
    }
    caller_choice = {"type": "auto", "disable_parallel_tool_use": True}
    assert _assemble(
        _request(tools=[_native_computer()], tool_choice=caller_choice)
    )["tool_choice"] == caller_choice
    assert caller_choice == {"type": "auto", "disable_parallel_tool_use": True}


def test_sonnet55_rejects_native_adapter_at_custom_endpoint() -> None:
    provider = _provider(base_url="https://gateway.example.test")

    with pytest.raises(KernelInvalidRequestError, match="first-party"):
        provider._assemble_request_params(
            _request(tools=[_native_computer()]),
            request_options={"model": MODEL},
            request_caps=provider._get_capabilities(MODEL),
        )


@pytest.mark.parametrize(
    ("request_choice", "options"),
    [
        ("required", {}),
        (None, {"tool_choice": {"type": "tool", "name": "lookup"}}),
    ],
)
def test_sonnet55_rejects_forced_tool_choice_from_request_and_kwargs(
    request_choice: str | None, options: dict[str, Any]
) -> None:
    with pytest.raises(KernelInvalidRequestError, match="only tool_choice 'auto' or 'none'"):
        _assemble(
            _request(tool_choice=request_choice, tools=[_native_computer()]),
            **options,
        )


def test_sonnet55_rejects_forced_tool_choice_during_preflight() -> None:
    provider = _provider()

    async def run() -> None:
        with pytest.raises(KernelInvalidRequestError, match="only tool_choice 'auto' or 'none'"):
            await provider.request_budget(
                _request(tool_choice="required", tools=[_native_computer()]),
                context_estimate=10_000,
            )

    asyncio.run(run())


@pytest.mark.parametrize(
    "choice",
    [
        {"type": "any"},
        {"type": "tool", "name": "lookup"},
    ],
)
def test_sonnet55_rejects_extra_params_forced_tool_choice_during_assembly(
    choice: dict[str, str],
) -> None:
    provider = _provider(extra_request_params={"tool_choice": choice})

    with pytest.raises(KernelInvalidRequestError, match="only tool_choice 'auto' or 'none'"):
        provider._assemble_request_params(
            _request(tools=[_native_computer()]),
            request_options={"model": MODEL},
            request_caps=provider._get_capabilities(MODEL),
        )


@pytest.mark.parametrize(
    "choice",
    [
        {"type": "any"},
        {"type": "tool", "name": "lookup"},
    ],
)
def test_sonnet55_rejects_extra_params_forced_tool_choice_during_preflight(
    choice: dict[str, str],
) -> None:
    provider = _provider(extra_request_params={"tool_choice": choice})

    async def run() -> None:
        with pytest.raises(
            KernelInvalidRequestError, match="only tool_choice 'auto' or 'none'"
        ):
            await provider.request_budget(
                _request(tools=[_native_computer()]),
                context_estimate=10_000,
            )

    asyncio.run(run())


def test_sonnet55_allows_extra_params_auto_tool_choice_with_parallel_flag() -> None:
    choice = {"type": "auto", "disable_parallel_tool_use": True}
    provider = _provider(extra_request_params={"tool_choice": choice})

    assembly = provider._assemble_request_params(
        _request(tools=[_native_computer()]),
        request_options={"model": MODEL},
        request_caps=provider._get_capabilities(MODEL),
    )

    assert assembly is not None
    assert assembly.params["tool_choice"] == choice


@pytest.mark.parametrize(
    "options",
    [
        {"extended_thinking": False},
        {"thinking_type": "between_tools"},
    ],
)
def test_sonnet55_between_tools_is_the_only_thinking_payload(options: dict[str, Any]) -> None:
    params = _assemble(_request(), **options)

    assert params["thinking"] == {"type": "between_tools"}


def test_sonnet55_per_call_between_tools_suppresses_inherited_thinking_fields_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _provider(
        reasoning_effort="max",
        thinking_budget_tokens=8000,
        thinking_display="summarized",
    )
    request = _request(reasoning_effort="high")

    for _ in range(2):
        assembly = provider._assemble_request_params(
            request,
            request_options={
                "model": MODEL,
                "extended_thinking": False,
                "effort": "high",
            },
            request_caps=provider._get_capabilities(MODEL),
            emit_diagnostics=True,
        )
        assert assembly is not None
        assert assembly.params["thinking"] == {"type": "between_tools"}
        assert assembly.params["output_config"] == {"effort": "high"}
        assert assembly.params["max_tokens"] == 123

    warnings = [
        record.getMessage()
        for record in caplog.records
        if "Ignoring config" in record.getMessage()
    ]
    assert warnings == [
        "[PROVIDER] Ignoring config 'thinking_budget_tokens' for Sonnet 5.5 "
        "per-call extended_thinking=false: thinking.type='between_tools' sends "
        "no matching field.",
        "[PROVIDER] Ignoring config 'thinking_display' for Sonnet 5.5 per-call "
        "extended_thinking=false: thinking.type='between_tools' sends no "
        "matching field.",
    ]


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"thinking_budget_tokens": 1024}, "budget"),
        ({"thinking_display": "summarized"}, "display"),
    ],
)
def test_sonnet55_per_call_between_tools_rejects_explicit_thinking_fields(
    options: dict[str, Any], match: str
) -> None:
    with pytest.raises(KernelInvalidRequestError, match=match):
        _assemble(_request(), extended_thinking=False, **options)


def test_sonnet55_adaptive_request_keeps_all_thinking_config() -> None:
    provider = _provider(
        reasoning_effort="max",
        extended_thinking=True,
        thinking_type="adaptive",
        thinking_budget_tokens=8000,
        thinking_display="summarized",
    )
    assembly = provider._assemble_request_params(
        _request(),
        request_options={"model": MODEL},
        request_caps=provider._get_capabilities(MODEL),
    )

    assert assembly is not None
    assert assembly.params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert assembly.params["output_config"] == {"effort": "max"}
    assert assembly.params["max_tokens"] == 123


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"thinking_type": "between_tools", "thinking_budget_tokens": 1024}, "budget"),
        ({"thinking_type": "between_tools", "thinking_display": "summarized"}, "display"),
        ({"thinking_type": "between_tools", "effort": "xhigh"}, "xhigh or max"),
        ({"extended_thinking": False, "effort": "max"}, "xhigh or max"),
    ],
)
def test_sonnet55_rejects_between_tools_incompatible_options(
    options: dict[str, Any], match: str
) -> None:
    with pytest.raises(KernelInvalidRequestError, match=match):
        _assemble(_request(), **options)


@pytest.mark.parametrize(
    "config",
    [
        {"beta_headers": ["fine-grained-tool-streaming-2025-05-14"]},
        {
            "extra_request_params": {
                "extra_headers": {
                    "anthropic-beta": "fine-grained-tool-streaming-2025-05-14"
                }
            }
        },
    ],
)
def test_sonnet55_rejects_fine_grained_streaming_with_native_toolset(
    config: dict[str, Any],
) -> None:
    provider = _provider(**config)

    with pytest.raises(KernelInvalidRequestError, match="incompatible"):
        provider._assemble_request_params(
            _request(tools=[_native_computer()]),
            request_options={"model": MODEL},
            request_caps=provider._get_capabilities(MODEL),
        )


def test_sonnet55_preserves_native_multiple_actions_for_sequential_execution() -> None:
    provider = _provider()
    assembly = provider._assemble_request_params(
        _request(tools=[_native_computer()]),
        request_options={"model": MODEL},
        request_caps=provider._get_capabilities(MODEL),
    )
    assert assembly is not None
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="toolu_left",
                toolset_name="computer",
                name="left_click",
                input={"coordinate": [1, 2]},
            ),
            SimpleNamespace(
                type="tool_use",
                id="toolu_right",
                toolset_name="computer",
                name="right_click",
                input={"coordinate": [3, 4]},
            ),
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
        model=MODEL,
    )

    converted = provider._convert_to_chat_response(
        response, native_computer_adapter=assembly.native_computer_adapter
    )

    assert [call.id for call in converted.tool_calls] == ["toolu_left", "toolu_right"]
    assert all(
        getattr(call, "_amplifier_execution_mode") == "sequential"
        for call in converted.tool_calls
    )


def test_native_tool_result_history_preserves_is_error() -> None:
    wire = _provider()._convert_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "toolu_1",
                        "name": "lookup",
                        "input": {},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "content": "failed", "is_error": True},
        ]
    )

    assert wire[1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "failed",
            "is_error": True,
        }
    ]