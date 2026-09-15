"""Focused offline contract tests for exact Anthropic request budgeting."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import anthropic
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
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
        api_key="test-key",
        config={
            "default_model": MODEL,
            "use_streaming": False,
            "max_retries": 0,
            **config,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _raw_response() -> MagicMock:
    response = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(
            input_tokens=600,
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


def _rich_adaptive_request() -> ChatRequest:
    """Adaptive thinking, signed history, multimodal input, and 45 tools."""
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
            Message(
                role="user",
                content=[
                    TextBlock(text="Use all available detail."),
                    ImageBlock(
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aW1hZ2U=",
                        }
                    ),
                ],
            ),
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
        tool_choice="required",
    )


def _rate_limit_error() -> anthropic.RateLimitError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    return anthropic.RateLimitError("rate limited", response=response, body=None)


class TestRequestBudget:
    def test_unknown_budget_capability_is_unavailable_without_creating_a_client(self) -> None:
        provider = _provider()
        request = _request()
        before = request.model_dump()
        prefix_before = provider._prefix_fingerprints.copy()

        result = asyncio.run(
            provider.request_budget(
                request,
                context_estimate=200_000,
                request_options={"model": "claude-new-unknown"},
            )
        )

        assert result is None
        assert request.model_dump() == before
        assert provider._prefix_fingerprints == prefix_before
        assert provider._client is None
        assert not hasattr(provider, "_input_token_ratio_by_model")

    def test_count_result_makes_budget_decision_with_fixed_reserve(self) -> None:
        provider = _provider()
        provider.client.messages.count_tokens = AsyncMock(
            return_value=SimpleNamespace(input_tokens=250_000)
        )

        decision = asyncio.run(
            provider.request_budget(_request(cap=12_345), context_estimate=200_000)
        )

        assert decision == {
            "estimated_input_tokens": 254_096,
            "input_limit_tokens": 200_000,
            "context_token_budget": 157_419,
            "max_output_tokens": 12_345,
        }

    def test_count_projects_shared_assembly_and_preserves_rich_dispatch_shape(self) -> None:
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
        provider = _provider(beta_headers=["test-count-beta"])
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[_raw_response(), _raw_response()]
        )
        asyncio.run(provider.complete(initial))
        state_before = deepcopy(provider._prefix_fingerprints)
        caps = provider._budget_capabilities_for(MODEL)
        assert caps is not None
        assembly = provider._assemble_request_params(
            request, request_options={}, request_caps=caps
        )
        assert assembly is not None
        provider.client.messages.count_tokens = AsyncMock(
            return_value=SimpleNamespace(input_tokens=10_000)
        )

        decision = asyncio.run(provider.request_budget(request, context_estimate=200_000))

        assert decision is not None
        assert provider._prefix_fingerprints == state_before
        _, counted = provider.client.messages.count_tokens.call_args
        assert counted == {
            **provider._count_tokens_params(assembly.params),
            "timeout": 5.0,
        }
        assert "max_tokens" not in counted
        assert counted["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert len(counted["tools"]) == 45
        assert counted["tool_choice"] == {"type": "any"}
        assert counted["messages"][2]["content"][0]["signature"] == "signed-history"
        assert counted["messages"][3]["content"][1]["type"] == "image"
        assert "test-count-beta" in counted["extra_headers"]["anthropic-beta"]

        asyncio.run(provider.complete(request))
        _, dispatched = provider.client.messages.with_raw_response.create.call_args
        assert {
            key: dispatched[key]
            for key in provider._count_tokens_params(dispatched)
        } == provider._count_tokens_params(assembly.params)
        assert dispatched["max_tokens"] == 128_000

    def test_explicit_output_cap_remains_local_to_budget_decision(self) -> None:
        provider = _provider(extra_request_params={"max_tokens": 64_000})
        provider.client.messages.count_tokens = AsyncMock(
            return_value=SimpleNamespace(input_tokens=600)
        )

        decision = asyncio.run(
            provider.request_budget(_request(cap=3210), context_estimate=200_000)
        )

        assert decision is not None
        assert decision["max_output_tokens"] == 3210
        _, counted = provider.client.messages.count_tokens.call_args
        assert "max_tokens" not in counted

    def test_count_failures_are_unavailable_and_never_dispatch_or_recover(self) -> None:
        failures: tuple[object, ...] = (
            RuntimeError("count failed"),
            TimeoutError(),
            _rate_limit_error(),
            SimpleNamespace(input_tokens="not-an-int"),
            SimpleNamespace(input_tokens=True),
            SimpleNamespace(input_tokens=-1),
        )
        for outcome in failures:
            provider = _provider()
            provider.client.messages.with_raw_response.create = AsyncMock()
            if isinstance(outcome, Exception):
                provider.client.messages.count_tokens = AsyncMock(side_effect=outcome)
            else:
                provider.client.messages.count_tokens = AsyncMock(return_value=outcome)

            decision = asyncio.run(
                provider.request_budget(_request(), context_estimate=200_000)
            )

            assert decision is None
            provider.client.messages.with_raw_response.create.assert_not_called()