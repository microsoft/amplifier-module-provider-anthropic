"""Tests for the prompt-cache breakpoint placement fix.

Background (see FINDINGS.md at the workspace root for the full writeup):
the provider used to stamp `cache_control` on `messages[-1]` unconditionally.
In a live session, `messages[-1]` is frequently an *ephemeral* hook-injected
`<system-reminder>` message (a live timestamp / git-status blob) that the
orchestrator appends fresh on every request and never stores in history. That
message can never be reproduced on the next call, so the cached prefix always
terminated in content that was written once and never re-read -- measured
impact: cache_read frozen at a single constant for the whole session in 22/31
(71%) sampled sessions, aggregate write:read ratio 5.96x.

These tests cover the three properties the fix must guarantee:

1. Never more than 4 cache breakpoints total (Anthropic hard limit).
2. A cache breakpoint is never placed on a message marked ephemeral via
   `Message.metadata["ephemeral"]`.
3. Breakpoint *placement* (which message gets stamped) is stable across two
   consecutive requests that differ ONLY in their ephemeral tail content --
   this is the regression test that would have caught the original bug,
   because the old code's placement moved (uselessly) every single call.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from amplifier_core.message_models import ChatRequest, Message, ToolCallBlock, ToolSpec

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import DummyResponse


def _make_provider(**config_overrides) -> AnthropicProvider:
    config = {"use_streaming": False, "enable_prompt_caching": True}
    config.update(config_overrides)
    return AnthropicProvider(api_key="test-key", config=config)


def _make_raw_mock(response: DummyResponse):
    raw = MagicMock()
    raw.parse.return_value = response
    raw.headers = {}
    return AsyncMock(return_value=raw)


def _capture_params(provider: AnthropicProvider) -> dict:
    """Wire up the provider's client so the request `params` dict sent to
    the Anthropic SDK is captured and returned, instead of making a real
    call."""
    captured: dict = {}

    async def _fake_create(**params):
        captured.update(params)
        raw = MagicMock()
        raw.parse.return_value = DummyResponse()
        raw.headers = {}
        return raw

    provider.client.messages.with_raw_response.create = AsyncMock(
        side_effect=_fake_create
    )
    return captured


def _count_cache_control_blocks(params: dict) -> int:
    """Count every cache_control marker across system, tools, and messages --
    exactly what Anthropic counts against the 4-breakpoint hard limit."""
    count = 0

    for block in params.get("system") or []:
        if isinstance(block, dict) and "cache_control" in block:
            count += 1

    for tool in params.get("tools") or []:
        if isinstance(tool, dict) and "cache_control" in tool:
            count += 1

    for msg in params.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    count += 1

    return count


def _long_tool_spec(name: str = "do_something") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="A test tool",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )


def _turn(user_text: str, assistant_text: str, ephemeral: bool = False) -> list[Message]:
    """Build one simple user/assistant turn (no tool calls)."""
    return [
        Message(role="user", content=user_text),
        Message(role="assistant", content=assistant_text),
    ]


def _ephemeral_tail(text: str) -> Message:
    """Build a message shaped exactly like the orchestrator's ephemeral
    hook-injection append (loop-streaming's `inject_context` handling):
    role from `context_injection_role` (default/observed: "user"), content
    is the injected text, and -- once the necessary upstream plumbing fix
    is applied -- `metadata={"ephemeral": True}`.
    """
    return Message(role="user", content=text, metadata={"ephemeral": True})


def _run(provider: AnthropicProvider, request: ChatRequest) -> dict:
    params = _capture_params(provider)
    asyncio.run(provider.complete(request))
    return params


# ---------------------------------------------------------------------------
# 1. Never exceed 4 breakpoints total
# ---------------------------------------------------------------------------


def test_never_exceeds_four_breakpoints_with_system_tools_and_conversation():
    provider = _make_provider()

    messages: list[Message] = [Message(role="system", content="You are a helpful assistant.")]
    for i in range(6):
        messages.extend(_turn(f"question {i}", f"answer {i}"))
    messages.append(_ephemeral_tail("<system-reminder>live status</system-reminder>"))

    request = ChatRequest(
        messages=messages,
        tools=[_long_tool_spec("tool_a"), _long_tool_spec("tool_b")],
    )

    params = _run(provider, request)

    assert _count_cache_control_blocks(params) <= 4


def test_never_exceeds_four_breakpoints_even_with_many_turns():
    """A long conversation must still cap at 4 -- the rolling scheme adds at
    most 2 conversation breakpoints, never one per turn."""
    provider = _make_provider()

    messages: list[Message] = [Message(role="system", content="System prompt.")]
    for i in range(40):
        messages.extend(_turn(f"question {i}", f"answer {i}"))
    messages.append(_ephemeral_tail("<system-reminder>live status</system-reminder>"))

    request = ChatRequest(messages=messages, tools=[_long_tool_spec()])
    params = _run(provider, request)

    assert _count_cache_control_blocks(params) <= 4


# ---------------------------------------------------------------------------
# 2. Never place a breakpoint on an ephemeral message
# ---------------------------------------------------------------------------


def test_breakpoint_never_lands_on_ephemeral_tail_message():
    provider = _make_provider()

    messages: list[Message] = [Message(role="system", content="System prompt.")]
    for i in range(3):
        messages.extend(_turn(f"question {i}", f"answer {i}"))
    ephemeral_text = "<system-reminder>10:30:31 live git status</system-reminder>"
    messages.append(_ephemeral_tail(ephemeral_text))

    request = ChatRequest(messages=messages)
    params = _run(provider, request)

    sent_messages = params["messages"]
    last_msg = sent_messages[-1]
    # Sanity: the ephemeral message really is last, and really did carry the
    # unstable content through to the wire format.
    assert ephemeral_text in str(last_msg.get("content"))

    content = last_msg.get("content")
    if isinstance(content, list):
        assert all("cache_control" not in block for block in content)
    else:
        # String content never carries cache_control at all in our wire format
        assert True


def test_all_messages_ephemeral_places_zero_conversation_breakpoints():
    """Degenerate case: every conversation message is ephemeral -- there is
    no stable content, so the provider must not guess; it must skip
    conversation-region caching (system/tools breakpoints are unaffected)."""
    provider = _make_provider()

    messages = [
        Message(role="system", content="System prompt."),
        Message(role="user", content="hi", metadata={"ephemeral": True}),
    ]
    request = ChatRequest(messages=messages)
    params = _run(provider, request)

    sent_messages = params["messages"]
    for msg in sent_messages:
        content = msg.get("content")
        if isinstance(content, list):
            assert all("cache_control" not in b for b in content)

    # System breakpoint should still be present (independent mechanism).
    system_blocks = params.get("system") or []
    assert any("cache_control" in b for b in system_blocks)


# ---------------------------------------------------------------------------
# 3. Placement is stable across two requests differing only in ephemeral tail
#    -- the actual regression test for the bug.
# ---------------------------------------------------------------------------


def test_breakpoint_placement_stable_across_changing_ephemeral_tail():
    """Two 'requests' with IDENTICAL stable history but DIFFERENT ephemeral
    tail content (simulating two consecutive turns where only the injected
    timestamp/git-status changed) must place their conversation-region
    breakpoint(s) on the exact same stable message content.

    This is the test that would have caught the original bug: the old
    `_apply_message_cache_control` stamped `messages[-1]` unconditionally, so
    changing the tail content changed WHERE (and onto what) the breakpoint
    landed on every single call -- guaranteeing a cache miss.
    """
    stable_messages: list[Message] = [Message(role="system", content="System prompt.")]
    for i in range(5):
        stable_messages.extend(_turn(f"question {i}", f"answer {i}"))

    def _params_for_tail(tail_text: str) -> dict:
        provider = _make_provider()
        messages = list(stable_messages) + [_ephemeral_tail(tail_text)]
        request = ChatRequest(messages=messages)
        return _run(provider, request)

    params_a = _params_for_tail("<system-reminder>10:30:31 clean tree</system-reminder>")
    params_b = _params_for_tail("<system-reminder>10:37:55 3 files changed</system-reminder>")

    def _cached_block_texts(params: dict) -> list[str]:
        texts = []
        for msg in params["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        texts.append(block.get("text", ""))
        return texts

    cached_a = _cached_block_texts(params_a)
    cached_b = _cached_block_texts(params_b)

    assert cached_a, "expected at least one conversation-region cache breakpoint"
    assert cached_a == cached_b, (
        "cache breakpoint landed on different stable content across two "
        "calls that only differed in ephemeral tail -- placement is not "
        "stable"
    )
    # And, of course, neither call ever cached the changing tail text itself.
    assert "clean tree" not in " ".join(cached_a)
    assert "3 files changed" not in " ".join(cached_b)


def test_rolling_secondary_breakpoint_matches_previous_primary():
    """As the conversation grows by one real user turn, the new secondary
    breakpoint should land on the same stable content as the previous call's
    primary breakpoint (Anthropic's documented rolling multi-turn pattern)."""
    provider_1 = _make_provider()
    turn_1 = _turn("question 0", "answer 0")
    messages_1 = (
        [Message(role="system", content="System prompt.")]
        + turn_1
        + [_ephemeral_tail("<system-reminder>t1</system-reminder>")]
    )
    params_1 = _run(provider_1, ChatRequest(messages=messages_1))

    provider_2 = _make_provider()
    turn_2 = _turn("question 1", "answer 1")
    messages_2 = (
        [Message(role="system", content="System prompt.")]
        + turn_1
        + turn_2
        + [_ephemeral_tail("<system-reminder>t2</system-reminder>")]
    )
    params_2 = _run(provider_2, ChatRequest(messages=messages_2))

    def _cached_texts(params: dict) -> set[str]:
        out = set()
        for msg in params["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        out.add(block.get("text", ""))
        return out

    primary_call_1 = _cached_texts(params_1)
    breakpoints_call_2 = _cached_texts(params_2)

    assert primary_call_1, "call 1 should have placed at least one breakpoint"
    assert primary_call_1 & breakpoints_call_2, (
        "call 2 should re-use at least one of call 1's cache breakpoints "
        "(the rolling secondary breakpoint) -- otherwise the cache can never "
        "be hit turn-over-turn"
    )


# ---------------------------------------------------------------------------
# No-signal case: metadata contract not wired up anywhere in this deployment
# ---------------------------------------------------------------------------


def test_no_metadata_anywhere_skips_conversation_caching_loudly(caplog):
    """If NO message in the request carries a metadata dict at all, the
    provider cannot trust 'not marked ephemeral' as a real signal (some
    orchestrator/context manager may simply not populate the field). It must
    not silently fall back to the old messages[-1] placement -- it must skip
    conversation-region caching and warn loudly."""
    provider = _make_provider()

    messages = [
        Message(role="system", content="System prompt."),
        Message(role="user", content="question"),
        Message(role="assistant", content="answer"),
        Message(role="user", content="<system-reminder>live</system-reminder>"),
    ]
    # Sanity: none of these carry metadata (simulating an orchestrator that
    # never populates Message.metadata at all).
    assert all(m.metadata is None for m in messages)

    request = ChatRequest(messages=messages)

    import logging

    with caplog.at_level(logging.WARNING, logger="amplifier_module_provider_anthropic"):
        params = _run(provider, request)

    for msg in params["messages"]:
        content = msg.get("content")
        if isinstance(content, list):
            assert all("cache_control" not in b for b in content)

    assert any(
        "cannot be distinguished from stable history" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Tool pairing safety
# ---------------------------------------------------------------------------


def test_breakpoint_never_splits_a_tool_use_tool_result_pair():
    """A breakpoint must never land on an assistant message's tool_use block
    when the very next message is that tool call's tool_result batch."""
    provider = _make_provider()

    messages = [
        Message(role="system", content="System prompt."),
        Message(role="user", content="do the thing"),
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_1", name="do_something", input={"value": "x"})],
        ),
        Message(role="tool", tool_call_id="call_1", content="tool output"),
        # No further stable content after the tool round -- the only
        # eligible position for "primary" is the assistant tool_use message
        # itself, which must be rejected in favor of walking back further
        # (or, if nothing earlier is safe, skipping the breakpoint).
        _ephemeral_tail("<system-reminder>live</system-reminder>"),
    ]
    request = ChatRequest(messages=messages)
    params = _run(provider, request)

    sent_messages = params["messages"]
    for i, msg in enumerate(sent_messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        has_cache = any(isinstance(b, dict) and "cache_control" in b for b in content)
        if not has_cache:
            continue
        # If this message has a cache breakpoint and contains a tool_use,
        # the *next* message must not be a pure tool_result batch.
        is_tool_use = any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in content
        )
        if is_tool_use and i + 1 < len(sent_messages):
            next_content = sent_messages[i + 1].get("content")
            if isinstance(next_content, list):
                next_is_tool_result = all(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in next_content
                )
                assert not next_is_tool_result


# ---------------------------------------------------------------------------
# enable_prompt_caching=False gate still respected
# ---------------------------------------------------------------------------


def test_prompt_caching_disabled_places_no_breakpoints_at_all():
    provider = _make_provider(enable_prompt_caching=False)

    messages = [Message(role="system", content="System prompt.")]
    messages.extend(_turn("question", "answer"))
    request = ChatRequest(messages=messages, tools=[_long_tool_spec()])

    params = _run(provider, request)

    assert _count_cache_control_blocks(params) == 0


# ---------------------------------------------------------------------------
# Extended (1h) TTL opt-in for the stable system/tools region
# ---------------------------------------------------------------------------


def test_cache_stable_region_ttl_1h_defaults_off():
    provider = _make_provider()
    assert provider.cache_stable_region_ttl_1h is False

    messages = [Message(role="system", content="System prompt.")]
    messages.extend(_turn("question", "answer"))
    request = ChatRequest(messages=messages, tools=[_long_tool_spec()])
    params = _run(provider, request)

    system_blocks = params.get("system") or []
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    tools = params.get("tools") or []
    tool_with_cache = next(t for t in tools if "cache_control" in t)
    assert tool_with_cache["cache_control"] == {"type": "ephemeral"}


def test_cache_stable_region_ttl_1h_opt_in_applies_to_system_and_tools_only():
    provider = _make_provider(cache_stable_region_ttl_1h=True)
    assert provider.cache_stable_region_ttl_1h is True
    assert "extended-cache-ttl-2025-04-11" in provider._beta_headers

    messages = [Message(role="system", content="System prompt.")]
    messages.extend(_turn("question", "answer"))
    messages.append(_ephemeral_tail("<system-reminder>live</system-reminder>"))
    request = ChatRequest(messages=messages, tools=[_long_tool_spec()])
    params = _run(provider, request)

    system_blocks = params.get("system") or []
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    tools = params.get("tools") or []
    tool_with_cache = next(t for t in tools if "cache_control" in t)
    assert tool_with_cache["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    # Conversation-region breakpoints stay on the default 5m TTL -- the
    # conversation changes almost every turn, unlike the system prompt/tools.
    for msg in params["messages"]:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    assert block["cache_control"] == {"type": "ephemeral"}
