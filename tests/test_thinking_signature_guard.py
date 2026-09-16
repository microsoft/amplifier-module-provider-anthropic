"""Tests for the unsigned-thinking-block guard (microsoft/amplifier#330).

Sessions that ran partly on non-Anthropic providers (e.g. OpenAI) persist
thinking blocks with signature=None or no signature field at all. Anthropic
requires a valid non-empty signature string on every thinking block sent as
input; forwarding an unsigned block yields HTTP 400
(messages.N.content.0.thinking.signature.str: Input should be a valid string)
and permanently bricks the session on resume.

Covers:
  (a) Thinking block with a valid string signature passes through unchanged
  (b) Thinking block with signature=None is dropped; sibling text/tool_use
      blocks are preserved
  (c) Thinking block with a missing signature field is dropped
  (d) Thinking block with an empty-string signature is dropped
  (e) Assistant message whose ONLY content is unsigned thinking blocks still
      produces a valid message (placeholder text, never an empty content array)
  (f) thinking_block message field: unsigned block is skipped, and a
      thinking-only message falls back to a placeholder
  (g) _convert_to_chat_response never persists an unsigned thinking block
      from an API response
"""

from types import SimpleNamespace
from typing import cast

from amplifier_core import ModuleCoordinator

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import DummyResponse, FakeCoordinator


def _make_provider() -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={"max_retries": 0},
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _thinking(signature=..., thinking: str = "some reasoning") -> dict:
    """Build a thinking block dict; pass signature=... to omit the field."""
    block = {"type": "thinking", "thinking": thinking}
    if signature is not ...:
        block["signature"] = signature
    return block


# ---------------------------------------------------------------------------
# (a) Valid signature passes through unchanged
# ---------------------------------------------------------------------------
def test_thinking_block_with_valid_signature_passes_through():
    provider = _make_provider()
    cleaned = provider._clean_content_block(
        _thinking(signature="sig-abc123", thinking="deep thought")
    )
    assert cleaned == {
        "type": "thinking",
        "thinking": "deep thought",
        "signature": "sig-abc123",
    }


# ---------------------------------------------------------------------------
# (b) signature=None is dropped; sibling blocks preserved
# ---------------------------------------------------------------------------
def test_thinking_block_with_none_signature_dropped_siblings_preserved():
    provider = _make_provider()
    assert provider._clean_content_block(_thinking(signature=None)) is None

    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": [
                _thinking(signature=None),
                {"type": "text", "text": "visible reply"},
                {"type": "tool_use", "id": "tc-1", "name": "grep", "input": {}},
            ],
        },
    ]
    converted = provider._convert_messages(messages)
    assistant = converted[-1]
    assert assistant["role"] == "assistant"
    block_types = [b["type"] for b in assistant["content"]]
    assert "thinking" not in block_types
    assert block_types == ["text", "tool_use"]
    assert assistant["content"][0]["text"] == "visible reply"
    assert assistant["content"][1]["id"] == "tc-1"


# ---------------------------------------------------------------------------
# (c) Missing signature field is dropped
# ---------------------------------------------------------------------------
def test_thinking_block_with_missing_signature_dropped():
    provider = _make_provider()
    assert provider._clean_content_block(_thinking()) is None


# ---------------------------------------------------------------------------
# (d) Empty-string signature is dropped
# ---------------------------------------------------------------------------
def test_thinking_block_with_empty_signature_dropped():
    provider = _make_provider()
    assert provider._clean_content_block(_thinking(signature="")) is None


# ---------------------------------------------------------------------------
# (e) All-unsigned-thinking assistant message never yields empty content
# ---------------------------------------------------------------------------
def test_all_unsigned_thinking_message_gets_placeholder_not_empty_content():
    provider = _make_provider()
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": [
                _thinking(signature=None, thinking="reasoning one"),
                _thinking(thinking="reasoning two"),  # missing signature
            ],
        },
        {"role": "user", "content": "Continue"},
    ]
    converted = provider._convert_messages(messages)
    assistant = next(m for m in converted if m["role"] == "assistant")
    # Never an empty content array - Anthropic rejects those
    assert assistant["content"]
    assert all(b["type"] == "text" for b in assistant["content"])
    assert assistant["content"][0]["text"]


# ---------------------------------------------------------------------------
# (f) thinking_block message field paths
# ---------------------------------------------------------------------------
def test_unsigned_thinking_block_field_skipped_with_tool_calls():
    provider = _make_provider()
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "",
            "thinking_block": _thinking(signature=None),
            "tool_calls": [{"id": "tc-1", "tool": "grep", "arguments": {}}],
        },
    ]
    converted = provider._convert_messages(messages)
    assistant = converted[-1]
    block_types = [b["type"] for b in assistant["content"]]
    assert "thinking" not in block_types
    assert "tool_use" in block_types


def test_unsigned_thinking_block_field_only_content_gets_placeholder():
    provider = _make_provider()
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "",
            "thinking_block": _thinking(signature=None),
        },
    ]
    converted = provider._convert_messages(messages)
    assistant = converted[-1]
    assert assistant["content"]
    assert all(b["type"] == "text" for b in assistant["content"])


def test_signed_thinking_block_field_preserved():
    provider = _make_provider()
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "reply text",
            "thinking_block": _thinking(signature="sig-xyz"),
        },
    ]
    converted = provider._convert_messages(messages)
    assistant = converted[-1]
    assert assistant["content"][0] == {
        "type": "thinking",
        "thinking": "some reasoning",
        "signature": "sig-xyz",
    }


# ---------------------------------------------------------------------------
# (g) _convert_to_chat_response never persists unsigned thinking blocks
# ---------------------------------------------------------------------------
def test_response_conversion_skips_unsigned_thinking_block():
    provider = _make_provider()
    response = DummyResponse(
        content=[
            SimpleNamespace(type="thinking", thinking="unsigned", signature=None),
            SimpleNamespace(type="text", text="hello"),
        ]
    )
    result = provider._convert_to_chat_response(response)
    thinking_blocks = [
        b for b in result.content if getattr(b, "type", "") == "thinking"
    ]
    assert thinking_blocks == []
    text_blocks = [b for b in result.content if getattr(b, "type", "") == "text"]
    assert len(text_blocks) == 1
    assert text_blocks[0].text == "hello"


def test_response_conversion_keeps_signed_thinking_block():
    provider = _make_provider()
    response = DummyResponse(
        content=[
            SimpleNamespace(type="thinking", thinking="signed", signature="sig-1"),
            SimpleNamespace(type="text", text="hello"),
        ]
    )
    result = provider._convert_to_chat_response(response)
    thinking_blocks = [
        b for b in result.content if getattr(b, "type", "") == "thinking"
    ]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].signature == "sig-1"
