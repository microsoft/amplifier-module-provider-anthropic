"""Opus 5.5: thinking can never be disabled (always adaptive + display)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_core.message_models import RedactedThinkingBlock, ThinkingBlock

from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic import (
    BETA_HEADER_THINKING_DISPLAY_UPDATES,
)

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


def _run(provider: AnthropicProvider, request: ChatRequest, *, response=None, **opts):
    raw_response = MagicMock()
    raw_response.parse = AsyncMock(
        return_value=response
        if response is not None
        else DummyResponse(model=provider.default_model)
    )
    raw_response.headers = {}
    create = AsyncMock(return_value=raw_response)
    provider.client.messages.with_raw_response.create = create
    result = asyncio.run(provider.complete(request, **opts))
    assert create.await_count == 1
    return create.call_args.kwargs, result


def _params(provider: AnthropicProvider, request: ChatRequest, **opts):
    return _run(provider, request, **opts)[0]


def _thinking_block(text: str, signature: str | None = "sig") -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=text, signature=signature)


class TestOpus55AlwaysOnThinking:
    def test_no_effort_no_thinking_config_still_sends_adaptive(self):
        provider = _make_provider()
        params = _params(provider, ChatRequest(messages=[Message(role="user", content="hi")]))
        assert params["thinking"]["type"] == "adaptive"
        assert params["thinking"]["display"] == "summarized"

    def test_explicit_extended_thinking_false_still_sends_adaptive(self):
        provider = _make_provider()
        params = _params(
            provider,
            ChatRequest(
                messages=[Message(role="user", content="hi")], extended_thinking=False
            ),
        )
        assert params["thinking"]["type"] == "adaptive"

    def test_config_extended_thinking_false_still_sends_adaptive(self):
        provider = _make_provider(extended_thinking=False)
        params = _params(provider, ChatRequest(messages=[Message(role="user", content="hi")]))
        assert params["thinking"]["type"] == "adaptive"

    def test_opus_5_unchanged_omits_thinking_with_no_effort(self):
        provider = _make_provider(model="claude-opus-5")
        params = _params(provider, ChatRequest(messages=[Message(role="user", content="hi")]))
        assert "thinking" not in params


class TestOpus55ThinkingDisplayUpdatesBeta:
    def test_updates_display_sends_header(self):
        provider = _make_provider(thinking_display="updates")
        params = _params(
            provider, ChatRequest(messages=[Message(role="user", content="hi")])
        )
        assert params["thinking"]["display"] == "updates"
        betas = params["extra_headers"]["anthropic-beta"].split(",")
        assert BETA_HEADER_THINKING_DISPLAY_UPDATES in betas

    def test_updates_display_via_option_sends_header(self):
        provider = _make_provider()
        params = _params(
            provider,
            ChatRequest(messages=[Message(role="user", content="hi")]),
            thinking_display="updates",
        )
        assert params["thinking"]["display"] == "updates"
        betas = params["extra_headers"]["anthropic-beta"].split(",")
        assert BETA_HEADER_THINKING_DISPLAY_UPDATES in betas

    def test_opus_5_downgrades_updates_to_summarized_no_header(self):
        provider = _make_provider(model="claude-opus-5", thinking_display="updates")
        params = _params(
            provider,
            ChatRequest(
                messages=[Message(role="user", content="hi")], reasoning_effort="high"
            ),
        )
        assert params["thinking"]["display"] == "summarized"
        betas = params.get("extra_headers", {}).get("anthropic-beta", "").split(",")
        assert BETA_HEADER_THINKING_DISPLAY_UPDATES not in betas

    def test_updates_via_extra_request_params_still_gets_header(self):
        provider = _make_provider(
            extra_request_params={
                "thinking": {"type": "adaptive", "display": "updates"}
            }
        )
        params = _params(
            provider, ChatRequest(messages=[Message(role="user", content="hi")])
        )
        assert params["thinking"]["display"] == "updates"
        betas = params["extra_headers"]["anthropic-beta"].split(",")
        assert BETA_HEADER_THINKING_DISPLAY_UPDATES in betas

    def test_summarized_display_has_no_updates_header(self):
        provider = _make_provider()
        params = _params(
            provider, ChatRequest(messages=[Message(role="user", content="hi")])
        )
        betas = params.get("extra_headers", {}).get("anthropic-beta", "").split(",")
        assert BETA_HEADER_THINKING_DISPLAY_UPDATES not in betas


class TestOpus55ProgressUpdateVisibility:
    def test_updates_display_tags_nonempty_thinking_as_user_visible(self):
        provider = _make_provider(thinking_display="updates")
        response = DummyResponse(
            content=[
                _thinking_block(""),
                _thinking_block("Checking the config."),
            ],
            model=provider.default_model,
        )
        _, result = _run(
            provider,
            ChatRequest(messages=[Message(role="user", content="hi")]),
            response=response,
        )
        thinking_blocks = [b for b in result.content if isinstance(b, ThinkingBlock)]
        assert thinking_blocks[0].visibility == "internal"
        assert getattr(thinking_blocks[0], "progress_update", None) is not True
        assert thinking_blocks[1].visibility == "user"
        assert thinking_blocks[1].progress_update is True

    def test_summarized_display_keeps_thinking_internal(self):
        provider = _make_provider()
        response = DummyResponse(
            content=[_thinking_block("Checking the config.")],
            model=provider.default_model,
        )
        _, result = _run(
            provider,
            ChatRequest(messages=[Message(role="user", content="hi")]),
            response=response,
        )
        thinking_blocks = [b for b in result.content if isinstance(b, ThinkingBlock)]
        assert thinking_blocks[0].visibility == "internal"

    def test_interrupted_sentinel_is_user_visible_even_under_summarized(self):
        provider = _make_provider(thinking_display="summarized")
        interrupted = "This part of the response was interrupted before it finished."
        response = DummyResponse(
            content=[_thinking_block(interrupted)], model=provider.default_model
        )
        _, result = _run(
            provider,
            ChatRequest(messages=[Message(role="user", content="hi")]),
            response=response,
        )
        thinking_blocks = [b for b in result.content if isinstance(b, ThinkingBlock)]
        assert thinking_blocks[0].visibility == "user"

    def test_max_tokens_without_answer_text_warns(self, caplog):
        provider = _make_provider()
        response = DummyResponse(
            content=[_thinking_block("Still reasoning")], model=provider.default_model
        )
        response.stop_reason = "max_tokens"
        _run(
            provider,
            ChatRequest(messages=[Message(role="user", content="hi")]),
            response=response,
        )
        assert "thinking counts toward max_tokens" in caplog.text


class TestOpus55RedactedThinking:
    def test_redacted_thinking_block_is_preserved_not_dropped(self):
        provider = _make_provider()
        response = DummyResponse(
            content=[SimpleNamespace(type="redacted_thinking", data="opaque-blob")],
            model=provider.default_model,
        )
        _, result = _run(
            provider,
            ChatRequest(messages=[Message(role="user", content="hi")]),
            response=response,
        )
        redacted = [b for b in result.content if isinstance(b, RedactedThinkingBlock)]
        assert len(redacted) == 1
        assert redacted[0].data == "opaque-blob"
        assert redacted[0].visibility == "internal"
