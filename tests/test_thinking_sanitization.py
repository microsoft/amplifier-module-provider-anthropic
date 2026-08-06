"""Tests for defensive sanitization of invalid thinking-block signatures.

Fixes microsoft-amplifier/amplifier-support#207.

Background
----------
Anthropic strict-validates the ``signature`` field of every ``thinking``
content block it receives as history: it must be a non-empty string. A
session that switches providers mid-conversation (e.g. some turns handled by
provider-chat-completions or provider-openai, interleaved with anthropic
turns) can persist thinking blocks whose signature is ``null``, absent
entirely, or otherwise not something Anthropic minted -- the signing scheme
is provider-specific and cannot be retrofitted. A single such block anywhere
in history causes Anthropic to 400 the *entire* request with e.g.::

    messages.51.content[0].thinking.signature.str: Input should be a valid string

which bricks the session on every future resume attempt.

``AnthropicProvider._sanitize_thinking_blocks`` is a defensive pass over the
fully-assembled ``params["messages"]`` list (run once in
``_complete_chat_request``, so it covers the streaming transport, the
non-streaming transport, and the refusal-fallback retry -- they all consume
the same params dict) that strips any thinking block it cannot positively
verify as Anthropic-signed, replacing an emptied message's content with a
neutral placeholder rather than sending an empty array (which Anthropic also
rejects).

Two malformed shapes are covered (see issue #207 for the real-world repro):
  (a) ``{"type": "thinking", "thinking": "...", "signature": None}`` --
      round-tripped through provider-chat-completions' own history format.
  (b) ``{"type": "thinking", "content": ["<encrypted>", "rs_..."]}`` with no
      ``signature`` key at all -- provider-openai's Responses API persists
      encrypted reasoning + a reasoning-item id instead of a signature.

Both shapes reduce to the same check once evaluated with ``dict.get`` (a
missing key and an explicit ``None`` value are indistinguishable to
``.get()``), which is why one filter handles both. Note that when shape (a)
or (b) is constructed via the real ``ThinkingBlock``/``Message`` pydantic
models (as most tests below do, to stay on the same end-to-end path real
history takes), ``model_dump()`` always serializes the ``signature`` field
(defaulting to ``None``) -- so the "signature key entirely absent from the
dict" nuance can only be observed by calling ``_sanitize_thinking_blocks``
directly with a hand-built plain dict, which a couple of tests do below to
prove the implementation uses ``.get()`` (never bare indexing, which would
KeyError on truly-missing keys).
"""

import asyncio
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import DummyResponse, FakeCoordinator


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/test_reasoning_effort.py style)
# ---------------------------------------------------------------------------


def _make_provider(
    default_model: str = "claude-sonnet-4-5-20250929",
) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": 0,
            "default_model": default_model,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _make_raw_mock() -> MagicMock:
    raw = MagicMock()
    raw.parse.return_value = DummyResponse()
    raw.headers = {}
    return raw


def _wire_mock(provider: AnthropicProvider) -> AsyncMock:
    mock = AsyncMock(return_value=_make_raw_mock())
    provider.client.messages.with_raw_response.create = mock
    return mock


def _get_api_params(mock_create: AsyncMock) -> dict[str, Any]:
    """Extract the kwargs actually sent to the (mocked) Anthropic SDK call.

    This is the payload-level assertion surface: it's the same dict that
    would have been serialized onto the wire, so asserting on it here is
    the e2e check for a provider module (no live API call needed).
    """
    assert mock_create.await_count == 1
    _, kwargs = mock_create.call_args
    return kwargs


def _assistant_messages(params: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in params["messages"] if m.get("role") == "assistant"]


# ---------------------------------------------------------------------------
# (a) shape (a): signature explicitly null -> stripped
# ---------------------------------------------------------------------------


def test_null_signature_thinking_block_is_stripped_from_outgoing_payload():
    """A thinking block with signature=None must never reach the API."""
    provider = _make_provider()
    mock = _wire_mock(provider)

    request = ChatRequest(
        messages=[
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="internal reasoning", signature=None),
                    TextBlock(text="here is my answer"),
                ],
            ),
            Message(role="user", content="Continue"),
        ]
    )
    asyncio.run(provider.complete(request))

    params = _get_api_params(mock)
    assistant_msgs = _assistant_messages(params)
    assert len(assistant_msgs) == 1

    block_types = [b["type"] for b in assistant_msgs[0]["content"]]
    assert "thinking" not in block_types
    assert block_types == ["text"]  # bad thinking gone, text preserved


# ---------------------------------------------------------------------------
# (b) shape (b): OpenAI-Responses-API-shaped block (content list, no
#     meaningful signature) -> stripped
# ---------------------------------------------------------------------------


def test_reasoning_shaped_thinking_block_with_no_signature_is_stripped():
    """A thinking block carrying opaque cross-provider `content` (encrypted
    payload + reasoning-item id) instead of a real signature must be
    stripped, exactly like shape (a).
    """
    provider = _make_provider()
    mock = _wire_mock(provider)

    request = ChatRequest(
        messages=[
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(
                        thinking="",
                        content=["gAAAAAB123encryptedblob", "rs_abc123"],
                    ),
                    TextBlock(text="answer text"),
                ],
            ),
            Message(role="user", content="Continue"),
        ]
    )
    asyncio.run(provider.complete(request))

    params = _get_api_params(mock)
    assistant_msgs = _assistant_messages(params)
    block_types = [b["type"] for b in assistant_msgs[0]["content"]]
    assert "thinking" not in block_types
    assert block_types == ["text"]


def test_sanitize_directly_handles_truly_absent_signature_key():
    """Direct unit test of ``_sanitize_thinking_blocks`` with a plain dict
    where the ``signature`` key is genuinely absent (not just None) -- the
    one nuance that can't be produced via the pydantic Message/ThinkingBlock
    models, since model_dump() always serializes declared fields. Proves
    the implementation reads the field defensively (``.get``), not via
    direct indexing (which would KeyError here).
    """
    provider = _make_provider()

    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "content": ["gAAAAAB123encryptedblob", "rs_abc123"],
                    # NOTE: no "signature" key at all.
                },
                {"type": "text", "text": "answer text"},
            ],
        }
    ]

    sanitized = asyncio.run(provider._sanitize_thinking_blocks(messages))

    block_types = [b["type"] for b in sanitized[0]["content"]]
    assert "thinking" not in block_types
    assert block_types == ["text"]


# ---------------------------------------------------------------------------
# Valid signed thinking blocks are left untouched
# ---------------------------------------------------------------------------


def test_valid_signed_thinking_block_is_preserved_verbatim():
    provider = _make_provider()
    mock = _wire_mock(provider)

    request = ChatRequest(
        messages=[
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(
                        thinking="valid internal reasoning",
                        signature="a-real-anthropic-signature",
                    ),
                    TextBlock(text="answer text"),
                ],
            ),
            Message(role="user", content="Continue"),
        ]
    )
    asyncio.run(provider.complete(request))

    params = _get_api_params(mock)
    assistant_msgs = _assistant_messages(params)
    thinking_blocks = [
        b for b in assistant_msgs[0]["content"] if b["type"] == "thinking"
    ]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0]["thinking"] == "valid internal reasoning"
    assert thinking_blocks[0]["signature"] == "a-real-anthropic-signature"

    # Order preserved: thinking first, then text.
    assert [b["type"] for b in assistant_msgs[0]["content"]] == [
        "thinking",
        "text",
    ]


# ---------------------------------------------------------------------------
# Emptied message gets a placeholder, never an empty content array
# ---------------------------------------------------------------------------


def test_message_with_only_invalid_thinking_block_gets_placeholder():
    """Anthropic rejects empty content arrays just as strictly as it rejects
    unsigned thinking blocks. If stripping leaves nothing behind, a minimal
    non-empty placeholder block must be inserted instead.
    """
    provider = _make_provider()
    mock = _wire_mock(provider)

    request = ChatRequest(
        messages=[
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[ThinkingBlock(thinking="orphaned reasoning", signature=None)],
            ),
            Message(role="user", content="Continue"),
        ]
    )
    asyncio.run(provider.complete(request))

    params = _get_api_params(mock)
    assistant_msgs = _assistant_messages(params)
    content = assistant_msgs[0]["content"]

    assert content != []
    assert len(content) >= 1
    assert content[0]["type"] == "text"
    assert isinstance(content[0]["text"], str) and content[0]["text"].strip()


# ---------------------------------------------------------------------------
# Mixed message: only the bad thinking block is removed; other blocks
# (tool calls, text) are untouched and stay in order.
# ---------------------------------------------------------------------------


def test_only_invalid_thinking_block_removed_tool_calls_and_text_intact():
    provider = _make_provider()
    mock = _wire_mock(provider)

    request = ChatRequest(
        messages=[
            Message(role="user", content="Do the thing"),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="planning...", signature=None),
                    TextBlock(text="I'll call a tool"),
                    ToolCallBlock(id="call_1", name="do_something", input={"x": 1}),
                ],
            ),
            Message(
                role="tool",
                tool_call_id="call_1",
                content="tool result",
            ),
        ]
    )
    asyncio.run(provider.complete(request))

    params = _get_api_params(mock)
    assistant_msgs = _assistant_messages(params)
    assert len(assistant_msgs) == 1

    content = assistant_msgs[0]["content"]
    types = [b["type"] for b in content]

    assert "thinking" not in types
    assert types == ["text", "tool_call"]
    assert content[0]["text"] == "I'll call a tool"
    assert content[1]["id"] == "call_1"
    assert content[1]["name"] == "do_something"


def test_redacted_thinking_without_signature_key_is_untouched():
    """redacted_thinking blocks don't normally carry a signature at all --
    their `data` field is the whole payload, and Anthropic doesn't require
    signature validation on them. Only strip one if some producer attached
    an invalid `signature` key to it; leave the normal (no-signature) shape
    alone.
    """
    provider = _make_provider()
    mock = _wire_mock(provider)

    request = ChatRequest(
        messages=[
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[
                    RedactedThinkingBlock(data="opaque-encrypted-payload"),
                    TextBlock(text="answer text"),
                ],
            ),
            Message(role="user", content="Continue"),
        ]
    )
    asyncio.run(provider.complete(request))

    params = _get_api_params(mock)
    assistant_msgs = _assistant_messages(params)
    types = [b["type"] for b in assistant_msgs[0]["content"]]
    assert types == ["redacted_thinking", "text"]


# ---------------------------------------------------------------------------
# Non-dict content block present -> no crash
# ---------------------------------------------------------------------------


def test_sanitize_does_not_crash_on_non_dict_content_block():
    """A corrupted/partial transcript could in principle surface a raw
    string inside a content array. The sanitizer must tolerate this rather
    than raising (block.get(...) would AttributeError on a str).
    """
    provider = _make_provider()

    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                "gAAAAAB123encryptedblob",  # malformed: raw str, not a dict
                {"type": "text", "text": "answer text"},
            ],
        }
    ]

    sanitized = asyncio.run(provider._sanitize_thinking_blocks(messages))

    # No crash, and both entries survive untouched (neither is identifiable
    # as an invalid thinking block).
    assert sanitized[0]["content"][0] == "gAAAAAB123encryptedblob"
    assert sanitized[0]["content"][1] == {"type": "text", "text": "answer text"}


# ---------------------------------------------------------------------------
# Clean histories are completely unaffected (no behavior change)
# ---------------------------------------------------------------------------


def test_clean_history_with_no_thinking_blocks_is_unaffected():
    provider = _make_provider()
    mock = _wire_mock(provider)

    request = ChatRequest(
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="How are you?"),
        ]
    )
    asyncio.run(provider.complete(request))

    params = _get_api_params(mock)
    assistant_msgs = _assistant_messages(params)
    assert len(assistant_msgs) == 1
    # Plain string content should remain a plain string (untouched).
    assert assistant_msgs[0]["content"] == "Hi there"


# ---------------------------------------------------------------------------
# Observability: warning logged + event emitted when stripping occurs
# ---------------------------------------------------------------------------


def test_warning_logged_with_count_and_model_when_blocks_stripped(caplog):
    provider = _make_provider(default_model="claude-sonnet-4-5-20250929")
    mock = _wire_mock(provider)

    request = ChatRequest(
        messages=[
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="bad 1", signature=None),
                    TextBlock(text="ok"),
                ],
            ),
            Message(role="user", content="Continue"),
        ]
    )
    with caplog.at_level(logging.WARNING):
        asyncio.run(provider.complete(request))

    _get_api_params(mock)  # sanity: request still succeeded

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    stripped_warnings = [
        r for r in warnings if "thinking" in r.message.lower() and "strip" in r.message.lower()
    ]
    assert stripped_warnings, f"expected a strip warning, got: {[r.message for r in warnings]}"
    assert "1" in stripped_warnings[0].message  # stripped-count present


def test_event_emitted_when_blocks_stripped():
    provider = _make_provider()
    mock = _wire_mock(provider)
    fake_coordinator = cast(FakeCoordinator, provider.coordinator)

    request = ChatRequest(
        messages=[
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="bad", signature=None),
                    TextBlock(text="ok"),
                ],
            ),
            Message(role="user", content="Continue"),
        ]
    )
    asyncio.run(provider.complete(request))

    _get_api_params(mock)

    event_names = fake_coordinator.hooks.emitted_names()
    matches = [n for n in event_names if ("thinking" in n and "strip" in n) or "sanitiz" in n]
    assert matches, f"expected an observability event for stripped thinking blocks, got: {event_names}"


def test_no_event_emitted_when_nothing_stripped():
    """No behavior change for clean histories: no spurious strip event."""
    provider = _make_provider()
    mock = _wire_mock(provider)
    fake_coordinator = cast(FakeCoordinator, provider.coordinator)

    request = ChatRequest(
        messages=[
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="fine", signature="valid-sig"),
                    TextBlock(text="ok"),
                ],
            ),
            Message(role="user", content="Continue"),
        ]
    )
    asyncio.run(provider.complete(request))

    _get_api_params(mock)

    event_names = fake_coordinator.hooks.emitted_names()
    matches = [
        n for n in event_names if ("thinking" in n and "strip" in n) or "sanitiz" in n
    ]
    assert not matches, f"unexpected strip event on clean history: {matches}"
