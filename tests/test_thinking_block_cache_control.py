"""Regression test: a cache breakpoint must never land on a thinking block.

Reproduces a live 400 from the Anthropic API that broke a scheduled
automation. The provider places rolling conversation-region cache breakpoints
by stamping ``cache_control`` on the LAST content block of a chosen "stable"
message (``_stamp_last_block`` -> ``content[-1]``). The only placement guards
were "don't split a tool_use/tool_result pair" and "don't stamp an empty text
block" (``_last_safe_breakpoint_index`` / ``_stamps_empty_text_block``).

Neither guard rejects a message whose last block is a ``thinking`` (or
``redacted_thinking``) block. Once a conversation is long enough that a
breakpoint lands on such a message, the provider sends::

    messages.N.content.0.thinking.cache_control: Extra inputs are not permitted

and Anthropic rejects the WHOLE request. A single-thinking-block assistant
turn (``content == [thinking]``) makes the offending block ``content.0`` --
matching the exact ``messages.25.content.0.thinking`` seen in production.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from amplifier_core.message_models import ChatRequest, Message, ThinkingBlock

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import DummyResponse


def _make_provider(**config_overrides) -> AnthropicProvider:
    config = {"use_streaming": False, "enable_prompt_caching": True}
    config.update(config_overrides)
    return AnthropicProvider(api_key="test-key", config=config)


def _capture_params(provider: AnthropicProvider) -> dict:
    """Capture the params dict the provider would send to the Anthropic SDK,
    instead of making a real network call."""
    captured: dict = {}

    async def _fake_create(**params):
        captured.update(params)
        raw = MagicMock()
        raw.parse.return_value = DummyResponse()
        raw.headers = {}
        return raw

    provider.client.messages.with_raw_response.create = AsyncMock(side_effect=_fake_create)
    return captured


def _run(provider: AnthropicProvider, request: ChatRequest) -> dict:
    params = _capture_params(provider)
    asyncio.run(provider.complete(request))
    return params


def _thinking_blocks_with_cache_control(params: dict) -> list[tuple[int, int, str]]:
    """Every (message_index, block_index, type) that is a thinking or
    redacted_thinking block carrying cache_control -- the exact shape
    Anthropic rejects with a 400."""
    offenders: list[tuple[int, int, str]] = []
    for i, msg in enumerate(params.get("messages") or []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, block in enumerate(content):
            if not isinstance(block, dict) or "cache_control" not in block:
                continue
            if block.get("type") in ("thinking", "redacted_thinking"):
                offenders.append((i, j, block["type"]))
    return offenders


def _cache_control_count(params: dict) -> int:
    count = 0
    for msg in params.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    count += 1
    return count


def test_breakpoint_never_lands_on_a_thinking_block():
    """The stable boundary is an assistant turn whose only stored payload is a
    thinking block (content == [thinking]). Anthropic forbids cache_control on
    thinking blocks; the breakpoint must move to a cacheable block instead."""
    provider = _make_provider()

    messages = [
        Message(role="system", content="System prompt."),
        Message(role="user", content="first question"),
        Message(role="assistant", content="first answer"),
        Message(role="user", content="second question"),
        # Assistant turn carrying ONLY a thinking block -> content == [thinking].
        Message(role="assistant", content=[ThinkingBlock(thinking="pondering deeply")]),
        # Ephemeral regenerated-per-turn tail (supplies the metadata signal the
        # conversation-region caching path requires before it places anything).
        Message(
            role="user",
            content="<system-reminder>live</system-reminder>",
            metadata={"ephemeral": True},
        ),
    ]
    params = _run(provider, ChatRequest(messages=messages))

    offenders = _thinking_blocks_with_cache_control(params)
    assert not offenders, (
        "cache_control was stamped on a thinking block, which Anthropic rejects "
        "with a 400 (messages.N.content.M.thinking.cache_control: Extra inputs "
        f"are not permitted): {offenders}"
    )

    # The fix must relocate the breakpoint, not silently disable caching: at
    # least one legal (non-thinking) conversation breakpoint must remain.
    assert _cache_control_count(params) >= 1, (
        "expected the conversation-region breakpoint to move to a legal block, "
        "not to vanish"
    )


def test_breakpoint_never_lands_on_a_thinking_block_when_thinking_is_last():
    """A turn ending in thinking after some text (interleaved-thinking shape)
    is equally illegal to stamp -- content[-1] is still a thinking block."""
    provider = _make_provider()

    messages = [
        Message(role="system", content="System prompt."),
        Message(role="user", content="first question"),
        Message(role="assistant", content="first answer"),
        Message(role="user", content="second question"),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(thinking="reconsidering after the tool result"),
            ],
        ),
        Message(role="user", content="third question"),
        Message(role="assistant", content="third answer"),
        Message(
            role="user",
            content="<system-reminder>live</system-reminder>",
            metadata={"ephemeral": True},
        ),
    ]
    params = _run(provider, ChatRequest(messages=messages))

    offenders = _thinking_blocks_with_cache_control(params)
    assert not offenders, (
        "cache_control was stamped on a thinking block Anthropic rejects: "
        f"{offenders}"
    )
