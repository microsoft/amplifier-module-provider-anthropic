"""Focused offline contract tests for calibrated Anthropic request budgeting."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

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
        assembled = provider._assemble_budget_params(
            request, request_options={}, request_caps=caps
        )

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
        assembled = provider._assemble_budget_params(
            second, request_options={}, request_caps=caps
        )

        assert provider._prefix_fingerprints == state_before
        asyncio.run(provider.complete(second))
        _, sent = provider.client.messages.with_raw_response.create.call_args
        assert assembled == {key: value for key, value in sent.items() if key != "timeout"}