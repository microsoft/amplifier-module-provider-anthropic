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

    async def _complete_and_close() -> None:
        # Close the provider's underlying async HTTP client before this
        # event loop (created fresh by asyncio.run for every call to _run)
        # is torn down. Every test in this module builds its own
        # AnthropicProvider via _make_provider() and never closes it
        # otherwise; the leaked client is later garbage-collected at an
        # unpredictable point -- often during an unrelated *later* test --
        # and its internal cleanup then fails with "Event loop is closed"
        # (that loop already closed when *this* asyncio.run() returned).
        # asyncio's default exception handler logs that failure at ERROR
        # level via the "asyncio" logger, which can land inside whichever
        # test's caplog window happens to be open at GC time. Closing here,
        # inside the same loop the client was created in, prevents this
        # module's own tests from contributing leaked clients to that
        # cross-test noise.
        await provider.complete(request)
        await provider.close()

    asyncio.run(_complete_and_close())
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
# Single-message request: no conversation region at all -- vacuous input
# ---------------------------------------------------------------------------
#
# Real-world source of this shape: one-shot utility provider calls that never
# go through the main conversation loop at all, e.g. hooks-session-naming's
# `_generate_name`, which sends a single user message with no history and no
# metadata on *every* turn of *every* session. Before this guard, that call
# hit the `has_ephemeral_signal` branch below and fired the "no message ...
# carries a metadata dict" warning twice per turn -- a false alarm for a
# request that was never a caching candidate in the first place, and alarm
# fatigue that would mask the real (multi-message, no metadata) case.


def test_single_message_no_metadata_returns_silently_with_no_warning(caplog):
    """A single-message request (no conversation region, nothing stable to
    cache) must return early and silently -- no warning of any kind, at any
    level -- rather than reaching the has_ephemeral_signal warning."""
    provider = _make_provider()

    messages = [Message(role="user", content="question")]
    request = ChatRequest(messages=messages)

    import logging

    with caplog.at_level(logging.DEBUG, logger="amplifier_module_provider_anthropic"):
        params = _run(provider, request)

    # No cache_control anywhere in the (single) sent message.
    for msg in params["messages"]:
        content = msg.get("content")
        if isinstance(content, list):
            assert all("cache_control" not in b for b in content)

    # Nothing at all logged by _apply_conversation_cache_control -- not a
    # warning, not even a debug-level message. A silent return, full stop.
    #
    # Scoped to this module's own logger. Root cause of a flaky failure here
    # (ubuntu/py3.11, observed in CI run 31654412681): a *different* test's
    # AnthropicProvider (not this one's) leaks its httpx AsyncClient because
    # nothing in this file closed it before this fix. When Python garbage
    # collects that leaked client at some later, unpredictable point, its
    # cleanup tries to run against the (now-closed) event loop that created
    # it and fails with `RuntimeError("Event loop is closed")`. asyncio's
    # default exception handler logs that failure at ERROR level via the
    # "asyncio" logger -- unrelated to this module and to the cache-control
    # code path this assertion exists to guard -- and it can land inside
    # whichever test's caplog window happens to be open when the GC event
    # fires, which is nondeterministic and version-sensitive (observed only
    # on py3.11, never on py3.12/macOS in the same CI matrix). `_run()` now
    # closes the client used by each test in this file, removing this file
    # as a source of that noise; this scope narrowing additionally protects
    # this specific assertion against equivalent noise from elsewhere in the
    # (600+ test) suite that this file does not control.
    own_records = [
        r for r in caplog.records if r.name == "amplifier_module_provider_anthropic"
    ]
    assert not any("Prompt caching" in r.message for r in own_records), (
        "single-message request must not log anything from the cache-control path"
    )
    assert not any(r.levelno >= logging.WARNING for r in own_records)


def test_multi_message_no_metadata_still_warns_loudly(caplog):
    """Regression guard: the new single-message guard must NOT swallow the
    real has_ephemeral_signal warning for a genuine multi-message request
    that carries no metadata at all. That warning is the one this method
    exists to surface and must keep firing exactly as before."""
    provider = _make_provider()

    messages = [
        Message(role="system", content="System prompt."),
        Message(role="user", content="question"),
        Message(role="assistant", content="answer"),
        Message(role="user", content="follow-up, still no metadata anywhere"),
    ]
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
    ), "multi-message request with no metadata must still warn loudly"


def test_multi_message_with_ephemeral_metadata_places_breakpoints_as_before(caplog):
    """Regression guard on placement: a genuine multi-message conversation
    with proper `metadata={"ephemeral": True}` tagging must still get its
    conversation-region cache breakpoint(s) placed exactly as before the
    single-message guard was added -- the guard must only affect the
    len(all_messages) < 2 vacuous case, nothing else."""
    provider = _make_provider()

    messages: list[Message] = [Message(role="system", content="System prompt.")]
    for i in range(3):
        messages.extend(_turn(f"question {i}", f"answer {i}"))
    messages.append(_ephemeral_tail("<system-reminder>live status</system-reminder>"))

    request = ChatRequest(messages=messages)

    import logging

    with caplog.at_level(logging.WARNING, logger="amplifier_module_provider_anthropic"):
        params = _run(provider, request)

    # No spurious warnings for this well-formed, multi-message, properly
    # tagged request. Scoped to this module's own logger for the same reason
    # as the single-message case above: a leaked AsyncClient finalized by the
    # GC logs an ERROR on the unrelated "asyncio" logger, and it can land in
    # whichever caplog window happens to be open when that fires.
    own_records = [
        r for r in caplog.records if r.name == "amplifier_module_provider_anthropic"
    ]
    assert not any(r.levelno >= logging.WARNING for r in own_records)

    # At least one conversation-region cache breakpoint was placed -- the
    # guard did not accidentally suppress placement for a real conversation.
    found_breakpoint = False
    for msg in params["messages"]:
        content = msg.get("content")
        if isinstance(content, list):
            if any(isinstance(b, dict) and "cache_control" in b for b in content):
                found_breakpoint = True
    assert found_breakpoint, "expected at least one conversation-region cache breakpoint"


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

# ---------------------------------------------------------------------------
# Empty text blocks must never carry a cache breakpoint
# ---------------------------------------------------------------------------


def _cache_controlled_empty_text_blocks(params: dict) -> list[tuple[int, dict]]:
    """Return every (message_index, block) that is an EMPTY text block carrying
    cache_control -- the exact shape Anthropic rejects with:

        messages.N.content.0.text: cache_control cannot be set for empty text blocks
    """
    offenders: list[tuple[int, dict]] = []
    for i, msg in enumerate(params.get("messages") or []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or "cache_control" not in block:
                continue
            if block.get("type") != "text":
                continue
            if not (block.get("text") or "").strip():
                offenders.append((i, block))
    return offenders


def test_breakpoint_never_lands_on_an_empty_text_block():
    """A content-less assistant turn must never receive a cache breakpoint.

    Reproduces a live 400 from the Anthropic API. The chain:

    1. A caller represents a turn that carries no text (for example an
       assistant turn whose only payload is tool calls) as empty string
       content. This is legitimate and reaches providers routinely.
    2. That empty STRING reaches this provider. ``_stamp_last_block`` converts
       string content into a block array and stamps ``cache_control`` on it in
       the same step -- synthesising ``{"type": "text", "text": ""}`` WITH a
       cache breakpoint on it.
    3. Anthropic rejects the request outright.

    ``_last_safe_breakpoint_index`` already refuses to split a tool_use /
    tool_result pair; it must likewise refuse a candidate whose stamped block
    would be empty.
    """
    provider = _make_provider()

    messages = [
        Message(role="system", content="System prompt."),
        Message(role="user", content="do the thing"),
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_1", name="do_something", input={"value": "x"})],
        ),
        Message(role="tool", tool_call_id="call_1", content="tool output"),
        # The content-less assistant turn: tool calls already replayed above, so
        # this turn carries no text and no tool_use of its own.
        Message(role="assistant", content=""),
        Message(role="user", content="summarise what you found"),
        _ephemeral_tail("<system-reminder>live</system-reminder>"),
    ]
    request = ChatRequest(messages=messages)
    params = _run(provider, request)

    offenders = _cache_controlled_empty_text_blocks(params)
    assert not offenders, (
        "cache_control was stamped on an empty text block, which Anthropic "
        f"rejects with a 400: {offenders}"
    )


def test_breakpoint_never_lands_on_a_whitespace_only_text_block():
    """Same invariant for whitespace-only content.

    Anthropic treats a whitespace-only text block the same as an empty one, so
    a fix that only checks ``text == ""`` would still 400 in production.
    """
    provider = _make_provider()

    messages = [
        Message(role="system", content="System prompt."),
        Message(role="user", content="do the thing"),
        Message(role="assistant", content="   \n  "),
        Message(role="user", content="summarise what you found"),
        _ephemeral_tail("<system-reminder>live</system-reminder>"),
    ]
    request = ChatRequest(messages=messages)
    params = _run(provider, request)

    offenders = _cache_controlled_empty_text_blocks(params)
    assert not offenders, (
        "cache_control was stamped on a whitespace-only text block: "
        f"{offenders}"
    )


# ---------------------------------------------------------------------------
# Agentic tool loop: one real user turn, then N tool rounds
# ---------------------------------------------------------------------------


def _tool_round(n: int) -> list[Message]:
    """One agentic tool round: an assistant tool_use plus its tool_result.

    This is the shape of *every* iteration of an agentic loop -- a sub-agent
    delegation, a /goal run, a recipe step, or simply the tool-heavy first
    turn of an ordinary session. Critically, it contains NO new real user
    message: the human spoke once, at the very start, and everything after
    that is the model talking to its own tools.
    """
    return [
        Message(
            role="assistant",
            content=[ToolCallBlock(id=f"call_{n}", name="read_file", input={"n": n})],
        ),
        Message(role="tool", tool_call_id=f"call_{n}", content=f"file contents {n}"),
    ]


def _agentic_conversation(rounds: int) -> list[Message]:
    """A conversation with exactly ONE real user turn followed by `rounds`
    tool rounds, terminated by the orchestrator's ephemeral tail."""
    messages: list[Message] = [
        Message(role="system", content="System prompt."),
        Message(role="user", content="do the task"),
    ]
    for n in range(rounds):
        messages.extend(_tool_round(n))
    messages.append(
        _ephemeral_tail(f"<system-reminder>tail {rounds}</system-reminder>")
    )
    return messages


def _cached_block_ids(params: dict) -> set[str]:
    """Stable identity of every cache_control-stamped block.

    Message *indices* are stable across consecutive agentic requests (the
    prefix never changes, it only grows), but identifying by content makes
    a failure message far easier to read.
    """
    out: set[str] = set()
    for msg in params.get("messages") or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or "cache_control" not in block:
                continue
            btype = block.get("type")
            if btype == "tool_use":
                out.add(f"tool_use:{block.get('id')}")
            elif btype == "tool_result":
                out.add(f"tool_result:{block.get('tool_use_id')}")
            else:
                out.add(f"text:{block.get('text', '')}")
    return out


def test_agentic_tool_loop_reuses_a_breakpoint_across_iterations():
    """An agentic tool loop must get a rolling cache hit, exactly like a
    multi-turn chat does.

    Regression test for the case where `_find_rolling_secondary_index`
    bailed out whenever the only real user turn was at index 0 -- which is
    *always* true in an agentic loop. That left a single advancing primary
    breakpoint which can never overlap itself, so every request rewrote the
    whole prefix at the 1.25x write premium and read back nothing, for the
    entire run.

    The suite already covered this property for multi-turn chat (see
    `test_rolling_secondary_breakpoint_matches_previous_primary`), but every
    fixture there supplies a fresh real user turn each round -- the shape
    that takes the working code path. This test pins the agentic shape.
    """
    params_a = _run(_make_provider(), ChatRequest(messages=_agentic_conversation(3)))
    params_b = _run(_make_provider(), ChatRequest(messages=_agentic_conversation(4)))

    cached_a = _cached_block_ids(params_a)
    cached_b = _cached_block_ids(params_b)

    assert cached_a, "iteration 3 should have placed at least one breakpoint"
    assert cached_b, "iteration 4 should have placed at least one breakpoint"
    assert cached_a & cached_b, (
        "an agentic tool loop placed NO overlapping cache breakpoint between "
        "consecutive iterations, so the cache can never be read -- every call "
        f"pays the write premium for nothing. iter3={sorted(cached_a)} "
        f"iter4={sorted(cached_b)}"
    )


def test_agentic_tool_loop_places_two_breakpoints_once_a_round_completes():
    """Once at least one full tool round sits behind the primary, both the
    primary and the rolling secondary should be placed.

    Guards the `primary_idx < 2` early-out: it must suppress the secondary
    only when there is genuinely no completed round behind the primary, not
    swallow the whole agentic case.
    """
    params = _run(_make_provider(), ChatRequest(messages=_agentic_conversation(3)))
    assert len(_cached_block_ids(params)) == 2, (
        "expected a primary and a rolling secondary breakpoint in an "
        f"established agentic loop, got {sorted(_cached_block_ids(params))}"
    )


def test_agentic_breakpoints_never_land_on_the_ephemeral_tail():
    """The agentic fallback must respect the same ephemeral exclusion as
    every other placement path -- otherwise it reintroduces the original
    bug it exists to fix."""
    params = _run(_make_provider(), ChatRequest(messages=_agentic_conversation(4)))
    cached = _cached_block_ids(params)
    assert not any("system-reminder" in c for c in cached), (
        f"a breakpoint landed on the regenerated-per-turn tail: {sorted(cached)}"
    )
