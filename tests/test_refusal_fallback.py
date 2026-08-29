"""Tests for the refusal-fallback feature.

When a model returns finish_reason="refusal", complete() retries exactly once
against the SAME fallback ladder overload fallback uses (owner-adjudicated:
refusal fallback is not a separate escalation path -- it downgrades one rung,
resolved via the same three-source precedence: fallback_models override ->
live list_models cache -> static backstop), with thinking/redacted_thinking
blocks stripped from assistant messages in the retried request (still
required cross-model). Non-refusal responses pass through untouched, and the
fallback is skipped entirely when disabled, or when the ladder is exhausted
(haiku, the terminal rung), or when a rung would route onto a model in the
same family that just refused (loop guard, e.g. via a pathological
fallback_models override).

The old `refusal_fallback_model` explicit-override key (a hardcoded
escalation target, always Opus) is RETIRED -- see
test_config_surface.py / _INERT_CONFIG_KEY_MESSAGES for its migration
message.

Covers:
  (a) Refusal triggers exactly one fallback call to the ladder's resolved
      target; the fallback response is returned.
  (b) Non-refusal responses are returned untouched; no fallback call made.
  (c) _refusal_fallback_target returns None when disabled via config.
  (d) _refusal_fallback_target returns None when the ladder is exhausted
      (haiku, terminal) or a pathological fallback_models override would
      route onto the same family that just refused (loop guard).
  (e) _strip_thinking_blocks does not mutate the original request, only
      removes thinking/redacted_thinking blocks from assistant messages,
      and leaves everything else untouched.
"""

import asyncio
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
)

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import FakeCoordinator


def _make_provider(default_model: str, **config_overrides) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "default_model": default_model,
            "max_retries": 0,
            **config_overrides,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _request_with_refused_turn() -> ChatRequest:
    """A conversation whose last assistant turn carries thinking blocks --
    representative of what a real refusal-then-retry conversation looks like.
    """
    return ChatRequest(
        messages=[
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="internal reasoning", signature="sig"),
                    RedactedThinkingBlock(data="opaque"),
                    TextBlock(text="partial reply"),
                ],
            ),
            Message(role="user", content="Please continue"),
        ]
    )


def _response(finish_reason: str, text: str = "ok") -> ChatResponse:
    return ChatResponse(
        content=[TextBlock(text=text)],
        finish_reason=finish_reason,
    )


def _assistant_content(message: Message) -> list:
    assert isinstance(message.content, list)
    return message.content


# ---------------------------------------------------------------------------
# (a) Refusal triggers exactly one fallback call; fallback response returned
# ---------------------------------------------------------------------------
def test_refusal_triggers_single_fallback_call_with_thinking_stripped():
    provider = _make_provider("claude-fable-5")
    request = _request_with_refused_turn()

    refusal_response = _response("refusal", text="")
    fallback_response = _response("end_turn", text="fallback answer")
    provider._complete_chat_request = AsyncMock(
        side_effect=[refusal_response, fallback_response]
    )

    result = asyncio.run(provider.complete(request))

    assert result is fallback_response
    assert provider._complete_chat_request.await_count == 2

    _, second_call = provider._complete_chat_request.await_args_list

    # Second (fallback) call goes to the ladder's resolved target for
    # fable's next-lower rung (opus) -- the static backstop, since no
    # fallback_models override or live cache entry is set.
    assert second_call.kwargs["model"] == "claude-opus-5"

    # The fallback request has thinking/redacted_thinking stripped from the
    # assistant message, other content untouched.
    fallback_request = second_call.args[0]
    assistant_msg = next(m for m in fallback_request.messages if m.role == "assistant")
    block_types = [b.type for b in _assistant_content(assistant_msg)]
    assert "thinking" not in block_types
    assert "redacted_thinking" not in block_types
    assert block_types == ["text"]

    # The original request object passed to complete() was not mutated.
    original_assistant_msg = next(m for m in request.messages if m.role == "assistant")
    original_block_types = [b.type for b in _assistant_content(original_assistant_msg)]
    assert "thinking" in original_block_types
    assert "redacted_thinking" in original_block_types


# ---------------------------------------------------------------------------
# (b) Non-refusal responses returned untouched -- no fallback call
# ---------------------------------------------------------------------------
def test_non_refusal_response_returned_untouched_no_fallback_call():
    provider = _make_provider("claude-fable-5")
    request = _request_with_refused_turn()

    normal_response = _response("end_turn", text="normal answer")
    provider._complete_chat_request = AsyncMock(return_value=normal_response)

    result = asyncio.run(provider.complete(request))

    assert result is normal_response
    assert provider._complete_chat_request.await_count == 1


# ---------------------------------------------------------------------------
# (c) _refusal_fallback_target returns None when disabled
# ---------------------------------------------------------------------------
def test_refusal_fallback_target_none_when_disabled():
    provider = _make_provider("claude-fable-5", refusal_fallback_enabled=False)
    assert provider._refusal_fallback_target("claude-fable-5") is None


# ---------------------------------------------------------------------------
# (d) _refusal_fallback_target returns None on ladder exhaustion / loop guard
# ---------------------------------------------------------------------------
def test_refusal_fallback_target_none_when_ladder_exhausted_at_haiku():
    """haiku is the ladder's terminal rung -- a haiku refusal has no
    fallback target and surfaces normally."""
    provider = _make_provider("claude-haiku-4-5-20251001")
    assert provider._refusal_fallback_target("claude-haiku-4-5-20251001") is None


def test_refusal_fallback_target_none_when_same_family():
    """A pathological fallback_models override that resolves back into the
    same family as the refusing model is rejected by the ladder's
    same-family guard, exactly like overload fallback's."""
    provider = _make_provider(
        "claude-opus-4-5",
        fallback_models={"sonnet": "claude-opus-4-1"},  # still "opus" family
    )
    # opus -> sonnet override resolves to an opus-family id -> rejected ->
    # walk continues to haiku, which IS a valid (different-family) target.
    assert provider._refusal_fallback_target("claude-opus-4-5") == "claude-haiku-4-5"


def test_refusal_fallback_uses_same_target_resolution_as_overload():
    """The refusal ladder resolves through the exact same three-source
    precedence as overload fallback: fallback_models override wins."""
    provider = _make_provider(
        "claude-opus-4-5", fallback_models={"sonnet": "claude-sonnet-4-6"}
    )
    assert provider._refusal_fallback_target("claude-opus-4-5") == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# (e) _strip_thinking_blocks: no mutation, only thinking/redacted_thinking
#     removed from assistant messages, everything else untouched
# ---------------------------------------------------------------------------
def test_strip_thinking_blocks_does_not_mutate_and_only_strips_assistant_thinking():
    request = _request_with_refused_turn()
    original_assistant_content_ids = [
        id(block)
        for msg in request.messages
        if msg.role == "assistant"
        for block in _assistant_content(msg)
    ]

    stripped = AnthropicProvider._strip_thinking_blocks(request)

    # Original untouched.
    original_assistant_msg = next(m for m in request.messages if m.role == "assistant")
    assert [b.type for b in _assistant_content(original_assistant_msg)] == [
        "thinking",
        "redacted_thinking",
        "text",
    ]

    # Stripped copy has thinking/redacted_thinking removed, text preserved.
    stripped_assistant_msg = next(m for m in stripped.messages if m.role == "assistant")
    stripped_content = _assistant_content(stripped_assistant_msg)
    assert [b.type for b in stripped_content] == ["text"]
    assert cast(TextBlock, stripped_content[0]).text == "partial reply"

    # Non-assistant messages (string content) pass through untouched.
    stripped_user_msgs = [m for m in stripped.messages if m.role == "user"]
    original_user_msgs = [m for m in request.messages if m.role == "user"]
    assert [m.content for m in stripped_user_msgs] == [
        m.content for m in original_user_msgs
    ]

    # It's a deep copy -- the returned object is not the same instance/blocks.
    assert stripped is not request
    stripped_content_ids = [
        id(block)
        for msg in stripped.messages
        if msg.role == "assistant"
        for block in _assistant_content(msg)
    ]
    assert not set(stripped_content_ids) & set(original_assistant_content_ids)
