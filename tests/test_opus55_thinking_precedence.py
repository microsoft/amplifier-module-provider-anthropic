"""Opus 5.5: extended_thinking=False cannot suppress an explicit resolved
reasoning_effort.

On Opus 5.5+ thinking is never disableable (thinking_disableable=False):
an `extended_thinking: false` opt-out is already a no-op for the `thinking`
wire param (see test_opus55_thinking_display.py). This module additionally
guards the separate output_config.effort branch, which used to honor the
same opt-out independently and could suppress an explicit config- or
request-level reasoning_effort even though the server keeps thinking on
regardless -- silently falling back to the server's medium default instead
of the caller's/config's explicit choice.

Models whose thinking IS disableable (thinking_disableable=True, e.g.
Sonnet 5, Opus 4.7+) keep the pre-existing suppression behavior unchanged
-- see tests/test_reasoning_effort.py::TestExtendedThinkingFalseSuppressesOutputConfig.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import DummyResponse, FakeCoordinator


def _make_provider(
    model: str = "claude-opus-5-5", **overrides: Any
) -> AnthropicProvider:
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


def _run(provider: AnthropicProvider, request: ChatRequest, **opts: Any):
    raw_response = MagicMock()
    raw_response.parse = AsyncMock(
        return_value=DummyResponse(model=provider.default_model)
    )
    raw_response.headers = {}
    create = AsyncMock(return_value=raw_response)
    provider.client.messages.with_raw_response.create = create
    asyncio.run(provider.complete(request, **opts))
    return create.call_args.kwargs


class TestOpus55ExtendedThinkingFalseDoesNotSuppressEffort:
    def test_config_reasoning_effort_survives_extended_thinking_false(self):
        """Ambient config reasoning_effort='xhigh' + kwargs
        extended_thinking=False on claude-opus-5-5 -> output_config.effort
        is STILL sent (unlike the disableable-thinking case)."""
        provider = _make_provider(reasoning_effort="xhigh")
        request = ChatRequest(messages=[Message(role="user", content="hi")])

        params = _run(provider, request, extended_thinking=False)

        assert params["output_config"] == {"effort": "xhigh"}
        # Thinking itself is also still sent -- it can never be disabled.
        assert params["thinking"]["type"] == "adaptive"

    def test_request_reasoning_effort_survives_extended_thinking_false(self):
        """Same via request.reasoning_effort instead of provider config."""
        provider = _make_provider()
        request = ChatRequest(
            messages=[Message(role="user", content="hi")],
            reasoning_effort="high",
        )

        params = _run(provider, request, extended_thinking=False)

        assert params["output_config"] == {"effort": "high"}
        assert params["thinking"]["type"] == "adaptive"

    def test_explicit_effort_kwarg_still_wins(self):
        """A per-call kwargs['effort'] override still wins over the config
        default, exactly as on disableable-thinking models."""
        provider = _make_provider(reasoning_effort="xhigh")
        request = ChatRequest(messages=[Message(role="user", content="hi")])

        params = _run(provider, request, extended_thinking=False, effort="low")

        assert params["output_config"] == {"effort": "low"}

    def test_no_reasoning_effort_at_all_omits_output_config(self):
        """Regression guard: with no reasoning_effort resolved from any
        source, output_config.effort is still omitted (nothing to suppress
        or emit) -- the server's own medium default applies."""
        provider = _make_provider()
        request = ChatRequest(messages=[Message(role="user", content="hi")])

        params = _run(provider, request, extended_thinking=False)

        assert "output_config" not in params
        assert params["thinking"]["type"] == "adaptive"


class TestSonnet5ExtendedThinkingFalseStillSuppresses:
    """Regression guard: models with thinking_disableable=True (thinking
    actually turns off) keep the pre-existing suppression behavior --
    unaffected by the Opus 5.5 carve-out above."""

    def test_config_reasoning_effort_still_suppressed_on_sonnet_5(self):
        provider = _make_provider(model="claude-sonnet-5", reasoning_effort="xhigh")
        request = ChatRequest(messages=[Message(role="user", content="hi")])

        params = _run(provider, request, extended_thinking=False)

        assert "output_config" not in params
        assert "thinking" not in params

    def test_explicit_effort_kwarg_still_wins_on_sonnet_5(self):
        provider = _make_provider(model="claude-sonnet-5", reasoning_effort="xhigh")
        request = ChatRequest(messages=[Message(role="user", content="hi")])

        params = _run(provider, request, extended_thinking=False, effort="high")

        assert params["output_config"] == {"effort": "high"}
