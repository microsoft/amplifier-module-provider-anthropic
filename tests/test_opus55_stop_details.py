"""stop_details is captured into ChatResponse.metadata["anthropic"] (all models)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
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


def _run(provider: AnthropicProvider, response):
    raw_response = MagicMock()
    raw_response.parse = AsyncMock(return_value=response)
    raw_response.headers = {}
    create = AsyncMock(return_value=raw_response)
    provider.client.messages.with_raw_response.create = create
    request = ChatRequest(messages=[Message(role="user", content="hi")])
    return asyncio.run(provider.complete(request))


class _StopDetails:
    def __init__(self, **kwargs):
        self._data = kwargs

    def model_dump(self, mode="python", exclude_unset=False):
        return dict(self._data)


def test_stop_details_captured_in_metadata():
    provider = _make_provider()
    response = DummyResponse(model=provider.default_model)
    response.stop_details = _StopDetails(type="refusal", category="bio")
    result = _run(provider, response)
    assert result.metadata["anthropic"]["stop_details"] == {
        "type": "refusal",
        "category": "bio",
    }


def test_no_stop_details_leaves_metadata_without_the_key():
    provider = _make_provider()
    response = DummyResponse(model=provider.default_model)
    result = _run(provider, response)
    assert result.metadata is None or "stop_details" not in result.metadata.get(
        "anthropic", {}
    )


def test_pause_turn_and_context_window_exceeded_pass_through_unchanged():
    provider = _make_provider()
    for reason in ("pause_turn", "model_context_window_exceeded"):
        response = DummyResponse(model=provider.default_model)
        response.stop_reason = reason
        result = _run(provider, response)
        assert result.finish_reason == reason
