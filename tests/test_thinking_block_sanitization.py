"""Tests for thinking-block sanitization (fixes amplifier-support#207).

Cross-provider session resume can persist thinking-shaped content blocks
that the Anthropic API rejects outright, bricking the session. Two
malformed shapes are seen in the wild:

  - Shape A (chat-completions producer): thinking block with
    ``signature: null``.
  - Shape B (OpenAI Responses producer): thinking block with the
    ``signature`` key ENTIRELY ABSENT, and ``content`` holding
    provider-internal strings (an encrypted blob + a response item id,
    e.g. ``["gAAAAAB...", "rs_abc123"]``). This is the same shape found in
    the real corrupted transcript referenced in the issue (25 occurrences).

``AnthropicProvider._sanitize_thinking_blocks`` runs in the shared
request-construction path (``_complete_chat_request``) for every request,
so subclasses (e.g. the Fable provider) inherit the protection.

Covers:
  (a) Shape A stripped.
  (b) Shape B stripped.
  (c) Valid thinking block (non-empty string signature) passes through.
  (d) redacted_thinking passes through untouched (no signature expected).
  (e) Mixed-content message keeps non-thinking blocks.
  (f) Emptied assistant message gets a minimal placeholder block.
  (g) Non-assistant / non-list-content messages are left untouched.
  (h) Stripped counts accumulate correctly across blocks and messages.
  (i) The sanitizer never mutates its input and never raises, even on
      malformed/unexpected shapes.
  (j) Integration through complete(): warning logged with count, event
      emitted on the hook bus, and the sanitized (not raw) messages are
      what actually get sent to the Anthropic API.
"""

import asyncio
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_anthropic import (
    PROVIDER_THINKING_BLOCKS_SANITIZED,
    AnthropicProvider,
)
from tests._helpers import DummyResponse, FakeCoordinator

# ---------------------------------------------------------------------------
# Fixtures: the two malformed shapes from the issue, plus valid comparisons
# ---------------------------------------------------------------------------


def _shape_a_null_signature() -> dict[str, Any]:
    """chat-completions producer: signature explicitly null."""
    return {
        "type": "thinking",
        "thinking": "internal reasoning from a chat-completions turn",
        "signature": None,
        "visibility": "internal",
    }


def _shape_b_missing_signature() -> dict[str, Any]:
    """OpenAI Responses producer: signature key entirely absent.

    Mirrors the real corrupted transcript pattern: encrypted-blob +
    response-item-id strings under "content", no "signature" key at all.
    """
    return {
        "type": "thinking",
        "thinking": "internal reasoning from an OpenAI Responses turn",
        "content": ["gAAAAAB_synthetic_encrypted_blob", "rs_synthetic_item_id"],
        "visibility": "internal",
    }


def _valid_thinking_block(
    signature: str = "valid-anthropic-signature",
) -> dict[str, Any]:
    return {
        "type": "thinking",
        "thinking": "internal reasoning from a real Anthropic turn",
        "signature": signature,
    }


def _redacted_thinking_block() -> dict[str, Any]:
    return {"type": "redacted_thinking", "data": "opaque-redacted-blob"}


def _text_block(text: str = "hello") -> dict[str, Any]:
    return {"type": "text", "text": text}


def _assistant_msg(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


# ---------------------------------------------------------------------------
# Unit tests: AnthropicProvider._sanitize_thinking_blocks (dict-level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malformed_block,label",
    [
        (_shape_a_null_signature(), "shape-a-null-signature"),
        (_shape_b_missing_signature(), "shape-b-missing-signature"),
    ],
)
def test_malformed_thinking_block_is_stripped(malformed_block, label):
    messages = [
        {"role": "user", "content": "hi"},
        _assistant_msg([malformed_block, _text_block("visible reply")]),
    ]

    sanitized, stripped_count = AnthropicProvider._sanitize_thinking_blocks(messages)

    assert stripped_count == 1, label
    assistant_out = sanitized[1]
    types_out = [b["type"] for b in assistant_out["content"]]
    assert "thinking" not in types_out, label
    assert types_out == ["text"], label
    assert assistant_out["content"][0]["text"] == "visible reply"


def test_valid_thinking_block_passes_through_unchanged():
    valid_block = _valid_thinking_block()
    messages = [_assistant_msg([valid_block, _text_block("reply")])]

    sanitized, stripped_count = AnthropicProvider._sanitize_thinking_blocks(messages)

    assert stripped_count == 0
    assert sanitized[0]["content"][0] == valid_block
    assert sanitized[0]["content"][0]["signature"] == "valid-anthropic-signature"


def test_redacted_thinking_passes_through_unchanged():
    redacted = _redacted_thinking_block()
    messages = [_assistant_msg([redacted, _text_block("reply")])]

    sanitized, stripped_count = AnthropicProvider._sanitize_thinking_blocks(messages)

    assert stripped_count == 0
    types_out = [b["type"] for b in sanitized[0]["content"]]
    assert "redacted_thinking" in types_out
    assert sanitized[0]["content"][0] == redacted


def test_mixed_content_message_keeps_non_thinking_blocks():
    messages = [
        _assistant_msg(
            [
                _shape_a_null_signature(),
                _text_block("first"),
                _redacted_thinking_block(),
                _valid_thinking_block(),
                _text_block("second"),
            ]
        )
    ]

    sanitized, stripped_count = AnthropicProvider._sanitize_thinking_blocks(messages)

    assert stripped_count == 1
    types_out = [b["type"] for b in sanitized[0]["content"]]
    assert types_out == ["text", "redacted_thinking", "thinking", "text"]


def test_emptied_message_gets_placeholder_block():
    """Stripping the only block would leave an empty content array --
    Anthropic rejects that too, so a placeholder must be inserted."""
    messages = [_assistant_msg([_shape_b_missing_signature()])]

    sanitized, stripped_count = AnthropicProvider._sanitize_thinking_blocks(messages)

    assert stripped_count == 1
    content = sanitized[0]["content"]
    assert len(content) == 1
    assert content != []
    assert content[0]["type"] == "text"
    assert content[0].get("text")


def test_multiple_malformed_blocks_in_one_message_all_stripped():
    """Mirrors the real corrupted transcript: many shape-B blocks in a row."""
    many_malformed = [_shape_b_missing_signature() for _ in range(5)]
    messages = [_assistant_msg([*many_malformed, _text_block("final reply")])]

    sanitized, stripped_count = AnthropicProvider._sanitize_thinking_blocks(messages)

    assert stripped_count == 5
    assert [b["type"] for b in sanitized[0]["content"]] == ["text"]


def test_stripped_count_accumulates_across_messages():
    messages = [
        _assistant_msg([_shape_a_null_signature(), _text_block("a")]),
        {"role": "user", "content": "continue"},
        _assistant_msg([_shape_b_missing_signature(), _valid_thinking_block()]),
    ]

    _, stripped_count = AnthropicProvider._sanitize_thinking_blocks(messages)

    assert stripped_count == 2


def test_non_assistant_messages_untouched():
    user_msg = {"role": "user", "content": [_text_block("hi")]}
    tool_msg = {"role": "tool", "content": [{"type": "tool_result", "content": "ok"}]}
    messages = [user_msg, tool_msg]

    sanitized, stripped_count = AnthropicProvider._sanitize_thinking_blocks(messages)

    assert stripped_count == 0
    assert sanitized == [user_msg, tool_msg]


def test_assistant_message_with_string_content_untouched():
    messages = [{"role": "assistant", "content": "plain string reply"}]

    sanitized, stripped_count = AnthropicProvider._sanitize_thinking_blocks(messages)

    assert stripped_count == 0
    assert sanitized[0]["content"] == "plain string reply"


def test_does_not_mutate_input_messages():
    original_block = _shape_a_null_signature()
    original_msg = _assistant_msg([original_block, _text_block("reply")])
    messages = [original_msg]

    sanitized, stripped_count = AnthropicProvider._sanitize_thinking_blocks(messages)

    assert stripped_count == 1
    # Original list/message/blocks are untouched.
    assert original_msg["content"] == [original_block, _text_block("reply")]
    assert len(original_msg["content"]) == 2
    # The returned message is a different object when content changed.
    assert sanitized[0] is not original_msg


def test_unaffected_message_returned_as_is_not_copied():
    """Messages with no malformed thinking blocks aren't needlessly copied."""
    valid_msg = _assistant_msg([_valid_thinking_block(), _text_block("reply")])
    messages = [valid_msg]

    sanitized, stripped_count = AnthropicProvider._sanitize_thinking_blocks(messages)

    assert stripped_count == 0
    assert sanitized[0] is valid_msg


@pytest.mark.parametrize(
    "weird_messages",
    [
        [{"role": "assistant", "content": [None, "not-a-dict", 42]}],
        [{"role": "assistant", "content": [{"type": "thinking"}]}],  # no signature key
        [{"role": "assistant"}],  # no content key at all
        ["not-a-dict-message"],
        [{"role": "assistant", "content": [{"signature": "sig-no-type"}]}],
    ],
)
def test_sanitizer_never_raises_on_malformed_shapes(weird_messages):
    """Defensive: malformed/unexpected input is cleaned up, not a crash."""
    sanitized, stripped_count = AnthropicProvider._sanitize_thinking_blocks(
        weird_messages
    )
    assert isinstance(sanitized, list)
    assert isinstance(stripped_count, int)


# ---------------------------------------------------------------------------
# Integration tests: through complete() -- observability + outgoing payload
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "default_model": "claude-sonnet-4-5-20250929",
            "use_streaming": False,
            "max_retries": 0,
            **config_overrides,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _make_raw_mock() -> MagicMock:
    raw = MagicMock()
    raw.parse.return_value = DummyResponse()
    raw.headers = {}
    return raw


def _get_api_params(mock_create: AsyncMock) -> dict[str, Any]:
    assert mock_create.await_count == 1
    _, kwargs = mock_create.call_args
    return kwargs


def test_complete_strips_malformed_thinking_before_sending_to_api(caplog):
    """End-to-end: a resumed session with a shape-A block in history must
    not forward it to the Anthropic API, must log a warning with the
    count, and must emit the sanitization event."""
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_mock()
    )

    # Simulate the post-_convert_messages output containing a malformed
    # thinking block, as would occur on resume of a cross-provider session.
    # (ChatRequest/Message models always normalize a missing "signature"
    # key to signature=None, so we exercise the wire-level shapes directly
    # against the method that runs after that conversion.)
    original_convert = provider._convert_messages

    def _convert_with_malformed_history(messages):
        converted = original_convert(messages)
        converted.insert(
            0,
            _assistant_msg(
                [_shape_b_missing_signature(), _text_block("earlier reply")]
            ),
        )
        return converted

    provider._convert_messages = _convert_with_malformed_history

    request = ChatRequest(messages=[Message(role="user", content="continue")])

    with caplog.at_level(logging.WARNING, logger="amplifier_module_provider_anthropic"):
        asyncio.run(provider.complete(request))

    params = _get_api_params(provider.client.messages.with_raw_response.create)

    # No malformed thinking block reached the API payload.
    for msg in params["messages"]:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    assert isinstance(block.get("signature"), str)
                    assert block["signature"].strip()

    # Warning logged with the stripped count.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Stripped" in r.message and "1" in r.message for r in warnings)

    # Event emitted on the hook bus with the correct payload shape.
    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    assert PROVIDER_THINKING_BLOCKS_SANITIZED in hooks.emitted_names()
    payload = hooks.payload_for(PROVIDER_THINKING_BLOCKS_SANITIZED)
    assert payload is not None
    assert payload["stripped_count"] == 1
    assert payload["provider"] == "anthropic"


def test_complete_emits_no_sanitization_event_when_nothing_stripped():
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_mock()
    )

    request = ChatRequest(messages=[Message(role="user", content="hello")])
    asyncio.run(provider.complete(request))

    hooks = cast(FakeCoordinator, provider.coordinator).hooks
    assert PROVIDER_THINKING_BLOCKS_SANITIZED not in hooks.emitted_names()
