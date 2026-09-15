"""Focused offline contract tests for calibrated Anthropic request budgeting."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Self, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolSpec,
)

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import FakeCoordinator

MODEL = "claude-sonnet-5"


def _provider(**config: Any) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="[REDACTED:SECRET]",
        config={
            "default_model": MODEL,
            "use_streaming": False,
            "max_retries": 0,
            **config,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _raw_response(input_tokens: int = 600) -> MagicMock:
    response = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            cache_creation_input_tokens=200,
            cache_read_input_tokens=100,
            output_tokens=1,
        ),
        stop_reason="end_turn",
        model=MODEL,
    )
    raw = MagicMock()
    raw.parse = AsyncMock(return_value=response)
    raw.headers = {}
    return raw


def _request(*, cap: int | None = None) -> ChatRequest:
    return ChatRequest(
        messages=[
            Message(role="system", content="System authority."),
            Message(role="developer", content="Developer context."),
            Message(role="user", content="Use all available detail."),
        ],
        tools=[
            ToolSpec(
                name="lookup",
                description="Find a value.",
                parameters={"type": "object", "properties": {}},
            )
        ],
        reasoning_effort="high",
        max_output_tokens=cap,
    )


class _StreamManager:
    """Minimal successful SDK stream that captures the real outbound params."""

    def __init__(self, response: object) -> None:
        self.response = SimpleNamespace(headers={})
        self._response = response

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False

    def __aiter__(self):
        async def events():
            return
            yield  # pragma: no cover - keeps this an async generator

        return events()

    async def get_final_message(self) -> object:
        return self._response


def _rich_adaptive_request() -> ChatRequest:
    """Adaptive thinking, signed history, and a realistic large tool burst."""
    return ChatRequest(
        messages=[
            Message(role="system", content="System authority."),
            Message(role="developer", content="Developer context."),
            Message(role="user", content="First question."),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="Internal reasoning.", signature="signed-history"),
                    TextBlock(text="Prior answer."),
                ],
            ),
            Message(role="user", content="Use all available detail."),
        ],
        tools=[
            ToolSpec(
                name=f"lookup_{index}",
                description=f"Find value {index}.",
                parameters={"type": "object", "properties": {"value": {"type": "string"}}},
            )
            for index in range(45)
        ],
        reasoning_effort="high",
        max_output_tokens=128_000,
    )


class TestRequestBudget:
    def test_cold_budget_is_unavailable_without_dispatch_or_mutation(self) -> None:
        provider = _provider()
        request = _request()
        before = request.model_dump()
        prefix_before = provider._prefix_fingerprints.copy()

        assert provider.request_budget(request, context_estimate=200_000) is None
        assert request.model_dump() == before
        assert provider._prefix_fingerprints == prefix_before
        assert provider._client is None

    def test_successful_raw_usage_calibrates_adaptive_tool_request(self) -> None:
        provider = _provider()
        request = _request(cap=12_345)
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_raw_response()
        )

        asyncio.run(provider.complete(request))
        decision = provider.request_budget(request, context_estimate=200_000)

        assert decision is not None
        assert set(decision) == {
            "estimated_input_tokens",
            "input_limit_tokens",
            "context_token_budget",
            "max_output_tokens",
        }
        assert all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in decision.values()
        )
        assert decision["estimated_input_tokens"] > 0
        assert decision["input_limit_tokens"] > 0
        assert decision["max_output_tokens"] == 12_345
        assert decision["context_token_budget"] == 200_000

    def test_warm_oversized_request_requests_strictly_smaller_context(self) -> None:
        provider = _provider()
        request = _request()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_raw_response(input_tokens=900_000)
        )
        asyncio.run(provider.complete(request))

        oversized = ChatRequest(
            messages=[Message(role="user", content="x" * 500_000)],
            max_output_tokens=100,
        )
        decision = provider.request_budget(oversized, context_estimate=50_000)

        assert decision is not None
        assert decision["estimated_input_tokens"] > decision["input_limit_tokens"]
        assert 0 <= decision["context_token_budget"] < 50_000

    def test_explicit_output_cap_wins_over_adaptive_sizing_and_extras(self) -> None:
        provider = _provider(extra_request_params={"max_tokens": 64_000})
        request = _request(cap=3210)
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_raw_response()
        )

        asyncio.run(provider.complete(request))

        _, sent = provider.client.messages.with_raw_response.create.call_args
        assert sent["max_tokens"] == 3210
        decision = provider.request_budget(request, context_estimate=200_000)
        assert decision is not None
        assert decision["max_output_tokens"] == 3210

    def test_budget_assembly_matches_adaptive_dispatch_for_function_tools(self) -> None:
        provider = _provider()
        request = _request(cap=12_345)
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_raw_response()
        )

        asyncio.run(provider.complete(request))
        _, sent = provider.client.messages.with_raw_response.create.call_args
        caps = provider._budget_capabilities_for(MODEL)
        assert caps is not None
        assembly = provider._assemble_request_params(
            request, request_options={}, request_caps=caps
        )
        assert assembly is not None
        assembled = assembly.params

        assert assembled == {key: value for key, value in sent.items() if key != "timeout"}

    def test_staged_prefix_state_matches_warm_dispatch_without_advancing_it(self) -> None:
        provider = _provider()
        first = ChatRequest(
            messages=[
                Message(role="user", content="opening", metadata={"persisted": True}),
                Message(role="assistant", content="answer", metadata={"persisted": True}),
            ]
        )
        second = ChatRequest(
            messages=[
                *first.messages,
                Message(role="user", content="next", metadata={"ephemeral": True}),
            ]
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[_raw_response(), _raw_response()]
        )
        asyncio.run(provider.complete(first))
        state_before = provider._prefix_fingerprints.copy()
        caps = provider._budget_capabilities_for(MODEL)
        assert caps is not None
        assembly = provider._assemble_request_params(
            second, request_options={}, request_caps=caps
        )
        assert assembly is not None
        assembled = assembly.params

        assert provider._prefix_fingerprints == state_before
        asyncio.run(provider.complete(second))
        _, sent = provider.client.messages.with_raw_response.create.call_args
        assert assembled == {key: value for key, value in sent.items() if key != "timeout"}

    def test_disabled_history_inference_keeps_prior_prefix_state_during_probe(self) -> None:
        provider = _provider(cache_infer_stability_from_history=False)
        request = _request(cap=12_345)
        provider._prefix_fingerprints["existing"] = (["prior"], 1)
        state_before = provider._prefix_fingerprints.copy()
        caps = provider._budget_capabilities_for(MODEL)
        assert caps is not None

        assembly = provider._assemble_request_params(
            request, request_options={}, request_caps=caps
        )

        assert assembly is not None
        assert provider._prefix_fingerprints == state_before
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_raw_response()
        )
        asyncio.run(provider.complete(request))
        _, sent = provider.client.messages.with_raw_response.create.call_args
        assert assembly.params == {
            key: value for key, value in sent.items() if key != "timeout"
        }
        assert provider._prefix_fingerprints == state_before

    def test_warm_adaptive_signed_history_and_45_tools_match_both_dispatches(self) -> None:
        initial = _rich_adaptive_request()
        request = initial.model_copy(
            update={
                "messages": [
                    *initial.messages,
                    Message(role="assistant", content="Previous rich-turn answer."),
                    Message(
                        role="user",
                        content="Follow-up to the same rich conversation.",
                        metadata={"ephemeral": True},
                    ),
                ]
            },
            deep=True,
        )

        nonstream = _provider()
        nonstream.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[_raw_response(), _raw_response()]
        )
        asyncio.run(nonstream.complete(initial))
        nonstream_state = deepcopy(nonstream._prefix_fingerprints)
        assert nonstream_state
        caps = nonstream._budget_capabilities_for(MODEL)
        assert caps is not None
        nonstream_assembly = nonstream._assemble_request_params(
            request, request_options={}, request_caps=caps
        )
        assert nonstream_assembly is not None
        assert nonstream._prefix_fingerprints == nonstream_state
        asyncio.run(nonstream.complete(request))
        _, sent_nonstream = nonstream.client.messages.with_raw_response.create.call_args
        assert nonstream_assembly.params == {
            key: value for key, value in sent_nonstream.items() if key != "timeout"
        }

        stream = _provider(use_streaming=True)
        streamed_response = _raw_response().parse.return_value
        stream.client.messages.stream = MagicMock(
            return_value=_StreamManager(streamed_response)
        )
        asyncio.run(stream.complete(initial))
        stream_state = deepcopy(stream._prefix_fingerprints)
        assert stream_state
        stream_caps = stream._budget_capabilities_for(MODEL)
        assert stream_caps is not None
        stream_assembly = stream._assemble_request_params(
            request, request_options={}, request_caps=stream_caps
        )
        assert stream_assembly is not None
        assert stream._prefix_fingerprints == stream_state
        asyncio.run(stream.complete(request))
        _, sent_stream = stream.client.messages.stream.call_args
        assert stream_assembly.params == sent_stream
        assert sent_stream["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert len(sent_stream["tools"]) == 45
        signed_block = sent_stream["messages"][2]["content"][0]
        assert signed_block["signature"] == "signed-history"