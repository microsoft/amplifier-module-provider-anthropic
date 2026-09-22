"""Opus 5.5: thinking can never be disabled (always adaptive + display)."""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

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


def _run(provider: AnthropicProvider, request: ChatRequest):
    raw_response = MagicMock()
    raw_response.parse = AsyncMock(return_value=DummyResponse(model=provider.default_model))
    raw_response.headers = {}
    create = AsyncMock(return_value=raw_response)
    provider.client.messages.with_raw_response.create = create
    asyncio.run(provider.complete(request))
    assert create.await_count == 1
    return create.call_args.kwargs


class TestOpus55AlwaysOnThinking:
    def test_no_effort_no_thinking_config_still_sends_adaptive(self):
        provider = _make_provider()
        params = _run(provider, ChatRequest(messages=[Message(role="user", content="hi")]))
        assert params["thinking"]["type"] == "adaptive"
        assert params["thinking"]["display"] == "summarized"

    def test_explicit_extended_thinking_false_still_sends_adaptive(self):
        provider = _make_provider()
        params = _run(
            provider,
            ChatRequest(
                messages=[Message(role="user", content="hi")], extended_thinking=False
            ),
        )
        assert params["thinking"]["type"] == "adaptive"

    def test_config_extended_thinking_false_still_sends_adaptive(self):
        provider = _make_provider(extended_thinking=False)
        params = _run(provider, ChatRequest(messages=[Message(role="user", content="hi")]))
        assert params["thinking"]["type"] == "adaptive"

    def test_opus_5_unchanged_omits_thinking_with_no_effort(self):
        provider = _make_provider(model="claude-opus-5")
        params = _run(provider, ChatRequest(messages=[Message(role="user", content="hi")]))
        assert "thinking" not in params
