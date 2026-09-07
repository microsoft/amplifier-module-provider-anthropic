"""Tests for observation-based conversation-cache stability inference.

Background -- the measured bug this file exists to prevent regressing:

`_apply_conversation_cache_control` refuses to place a conversation-region
breakpoint unless SOME message in the request carries a `Message.metadata`
dict, because without it "not marked ephemeral" cannot be distinguished from
"this deployment never populates the field". That guard is correct in intent
and was a total dead end in practice: an orchestrator that never stamps
metadata got NO conversation caching at all, forever.

Measured on one coding-agent node visit (164 provider calls in 60 minutes,
session c2e6940c-4582-4569-9f51-d8d90ff44c48):

    cache_read_input_tokens == 10995 on every single one of the 164 calls
    (the system + tools blocks, and nothing else), while input_tokens
    climbed 24,334 -> 228,000. Totals: 1.79M cache_read against 24.7M
    uncached input == a 6.8% hit ratio. The provider's own
    "no message in this request carries a `metadata` dict" warning appears
    618 times in that run's engine-stdout.log.

The fix replaces the missing *declaration* with a *measurement*: fingerprint
each request's message array, and on the next request in the same
conversation take `len(previous) - longest_common_prefix` as the observed
unstable-suffix length.

The properties under test:

1. A second request in the same conversation places a conversation
   breakpoint even with NO metadata anywhere (the regression test).
2. That breakpoint lands on content the NEXT request still carries
   byte-identically -- the actual precondition for a cache hit.
3. The first request in a conversation still skips (no observation yet),
   so the existing loud warning is preserved rather than silenced.
4. A conversation whose leading message is regenerated per request places
   NOTHING, rather than burning a 1.25x cache write every turn.
5. Observation is a floor, never a ceiling: it can only make placement more
   conservative than metadata already asked for.
6. The 4-breakpoint Anthropic hard limit still holds.
7. A retry (byte-identical re-issue) is not mistaken for evidence of a
   stable tail.
8. The knob turns the whole inference off.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

from amplifier_core.message_models import ChatRequest, Message, ToolSpec
from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import DummyResponse


def _make_provider(**config_overrides) -> AnthropicProvider:
    config = {"use_streaming": False, "enable_prompt_caching": True}
    config.update(config_overrides)
    return AnthropicProvider(api_key="[REDACTED:SECRET]", config=config)


def _run_sequence(
    provider: AnthropicProvider, requests: list[ChatRequest]
) -> list[dict]:
    """Issue several requests through ONE provider instance, in one event
    loop, capturing the params dict sent to the SDK for each.

    A single provider across requests is the whole point: the fingerprint
    memory that makes the inference possible lives on the instance.
    """
    captured: list[dict] = []

    async def _fake_create(**params):
        captured.append(params)
        raw = MagicMock()
        raw.parse = AsyncMock(return_value=DummyResponse())
        raw.headers = {}
        return raw

    provider.client.messages.with_raw_response.create = AsyncMock(
        side_effect=_fake_create
    )

    async def _drive() -> None:
        try:
            for request in requests:
                await provider.complete(request)
        finally:
            await provider.close()

    asyncio.run(_drive())
    return captured


def _breakpoint_message_indices(params: dict) -> list[int]:
    """Indices of messages carrying a cache_control marker."""
    out = []
    for i, msg in enumerate(params.get("messages") or []):
        content = msg.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and "cache_control" in b for b in content
        ):
            out.append(i)
    return out


def _total_breakpoints(params: dict) -> int:
    """Every cache_control marker Anthropic counts against the hard limit."""
    count = 0
    for block in params.get("system") or []:
        if isinstance(block, dict) and "cache_control" in block:
            count += 1
    for tool in params.get("tools") or []:
        if isinstance(tool, dict) and "cache_control" in tool:
            count += 1
    count += sum(
        1
        for msg in params.get("messages") or []
        if isinstance(msg.get("content"), list)
        for b in msg["content"]
        if isinstance(b, dict) and "cache_control" in b
    )
    return count


def _strip_cache_control(obj):
    """Canonical form of a wire message, ignoring breakpoint markers.

    Two requests hit the same cache entry when their prefixes are identical
    *as content*; a cache_control marker is metadata about the boundary, not
    part of the cached bytes.
    """
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Fixtures shaped like the measured failing deployment: NO metadata anywhere,
# and a `<system-reminder>` block regenerated per request sitting immediately
# BEFORE the real user message (amplifier's pre_user reminder placement).
# ---------------------------------------------------------------------------


def _stable_history(turns: int) -> list[Message]:
    """Append-only conversation history -- what canonical storage holds."""
    messages: list[Message] = [Message(role="system", content="System prompt." * 40)]
    for i in range(turns):
        messages.append(Message(role="user", content=f"question {i} " * 20))
        messages.append(Message(role="assistant", content=f"answer {i} " * 20))
    return messages


def _request_with_pre_user_reminder(turns: int, clock: str) -> ChatRequest:
    """History + [regenerated reminder, new user message].

    NOTHING here carries `metadata`, exactly like the measured run.
    """
    messages = _stable_history(turns)
    messages.append(
        Message(
            role="user",
            content=f"<system-reminder>now={clock} git=clean</system-reminder>",
        )
    )
    messages.append(Message(role="user", content=f"next question at {clock}"))
    request = ChatRequest(messages=messages)
    assert all(m.metadata is None for m in messages), (
        "fixture must reproduce the measured deployment: no metadata anywhere"
    )
    return request


# ---------------------------------------------------------------------------
# 1 + 2. The regression test: caching engages, and lands somewhere reusable.
# ---------------------------------------------------------------------------


def test_second_request_places_breakpoint_without_any_metadata():
    """THE regression test for the 6.8% measurement.

    Two consecutive requests in one conversation, no metadata anywhere. The
    first has no prior observation and legitimately skips; the second has an
    observation and MUST place a conversation-region breakpoint. Before this
    fix, both placed zero -- for all 164 calls of the measured run.
    """
    provider = _make_provider()
    params = _run_sequence(
        provider,
        [
            _request_with_pre_user_reminder(3, "T1"),
            _request_with_pre_user_reminder(4, "T2"),
        ],
    )

    assert _breakpoint_message_indices(params[0]) == [], (
        "first request has no prior observation -- must still skip"
    )
    assert _breakpoint_message_indices(params[1]), (
        "second request has a measured observation and must place a "
        "conversation-region breakpoint"
    )


def test_breakpoint_lands_on_content_the_next_request_still_carries():
    """Placement must be *reusable*, not merely present.

    A breakpoint is only worth anything if the next request repeats the
    prefix up to it byte-for-byte. This asserts exactly that: take the
    breakpoint request 2 placed, and confirm request 3's messages reproduce
    every message up to and including that index, identically.

    This is the property the whole feature is for -- and the one a naive
    "just stamp the last message" implementation fails.
    """
    provider = _make_provider()
    params = _run_sequence(
        provider,
        [
            _request_with_pre_user_reminder(3, "T1"),
            _request_with_pre_user_reminder(4, "T2"),
            _request_with_pre_user_reminder(5, "T3"),
        ],
    )

    marks = _breakpoint_message_indices(params[1])
    assert marks, "request 2 must place at least one breakpoint"
    primary = max(marks)

    prefix_2 = [_strip_cache_control(m) for m in params[1]["messages"][: primary + 1]]
    prefix_3 = [_strip_cache_control(m) for m in params[2]["messages"][: primary + 1]]
    assert prefix_2 == prefix_3, (
        "the cached prefix must be reproduced byte-identically by the next "
        "request, otherwise the breakpoint is a write that is never read"
    )


def test_breakpoint_never_lands_on_the_regenerated_reminder():
    """The reminder block is regenerated every request. A breakpoint on or
    after it can never be re-read."""
    provider = _make_provider()
    params = _run_sequence(
        provider,
        [
            _request_with_pre_user_reminder(3, "T1"),
            _request_with_pre_user_reminder(4, "T2"),
        ],
    )

    messages = params[1]["messages"]
    reminder_idx = max(
        i
        for i, m in enumerate(messages)
        if "system-reminder" in str(m.get("content"))
    )
    for idx in _breakpoint_message_indices(params[1]):
        assert idx < reminder_idx, (
            f"breakpoint at {idx} lands at or after the regenerated reminder "
            f"at {reminder_idx}"
        )


def test_append_only_conversation_marks_the_true_tail():
    """When the orchestrator genuinely only appends (observed unstable
    suffix == 0), the breakpoint should reach the real tail -- the inference
    must not be needlessly conservative when the evidence says it can be."""
    provider = _make_provider()

    def _append_only(turns: int) -> ChatRequest:
        return ChatRequest(messages=_stable_history(turns))

    params = _run_sequence(provider, [_append_only(3), _append_only(4)])

    marks = _breakpoint_message_indices(params[1])
    assert marks, "append-only conversation must place a breakpoint"
    assert max(marks) == len(params[1]["messages"]) - 1, (
        "with a measured unstable suffix of 0 the primary breakpoint belongs "
        "on the true tail"
    )


# ---------------------------------------------------------------------------
# 3. First request still skips loudly -- the existing contract is preserved.
# ---------------------------------------------------------------------------


def test_first_request_still_warns_and_skips(caplog):
    """The metadata warning is not silenced -- it just stops repeating once
    an observation exists. On the FIRST request there is genuinely nothing to
    go on, so the pre-existing loud skip must still happen."""
    provider = _make_provider()
    with caplog.at_level(logging.WARNING, logger="amplifier_module_provider_anthropic"):
        params = _run_sequence(provider, [_request_with_pre_user_reminder(3, "T1")])

    assert _breakpoint_message_indices(params[0]) == []
    assert any(
        "cannot be distinguished from stable history" in r.message
        for r in caplog.records
    ), "first request must still warn loudly about the missing metadata contract"


# ---------------------------------------------------------------------------
# 4. The genuinely hopeless case is detected, not papered over.
# ---------------------------------------------------------------------------


def test_volatile_leading_message_places_nothing_and_says_why(caplog):
    """If the FIRST message is regenerated per request, Anthropic can never
    match a cached prefix (it matches from the start of the request). Placing
    a breakpoint anyway would bill a 1.25x cache write every single turn that
    is never read -- strictly worse than not caching. Skip, and name the
    cause."""
    provider = _make_provider()

    def _volatile_head(clock: str) -> ChatRequest:
        return ChatRequest(
            messages=[
                Message(role="system", content="System prompt." * 40),
                Message(role="developer", content=f"<context>session at {clock}"),
                Message(role="user", content="question " * 30),
                Message(role="assistant", content="answer " * 30),
            ]
        )

    with caplog.at_level(logging.WARNING, logger="amplifier_module_provider_anthropic"):
        params = _run_sequence(provider, [_volatile_head("T1"), _volatile_head("T2")])

    assert _breakpoint_message_indices(params[1]) == [], (
        "no breakpoint may be placed when no prefix can ever match"
    )
    assert any(
        "shares no leading message" in r.message for r in caplog.records
    ), "the hopeless case must name the likely cause, not fail silently"


# ---------------------------------------------------------------------------
# 5. Observation is a floor, never a ceiling.
# ---------------------------------------------------------------------------


def test_observation_never_loosens_a_metadata_declaration():
    """When metadata declares a LONGER unstable suffix than observation
    measured, the declaration wins. Observation may only ever exclude more
    of the tail, never less -- it is a safety floor, not a replacement."""
    provider = _make_provider()

    def _declared(turns: int, clock: str) -> ChatRequest:
        messages = _stable_history(turns)
        # Three trailing messages declared unstable, though only the last
        # actually changes between requests (observation would measure 1).
        for i in range(3):
            messages.append(
                Message(
                    role="user",
                    content=f"volatile {i} {clock}",
                    metadata={"ephemeral": True},
                )
            )
        return ChatRequest(messages=messages)

    params = _run_sequence(provider, [_declared(3, "T1"), _declared(4, "T2")])

    messages = params[1]["messages"]
    first_declared_idx = min(
        i for i, m in enumerate(messages) if "volatile 0" in str(m.get("content"))
    )
    for idx in _breakpoint_message_indices(params[1]):
        assert idx < first_declared_idx, (
            "a metadata-declared unstable message must never be stamped, even "
            "when observation measured a shorter unstable suffix"
        )


# ---------------------------------------------------------------------------
# 6. The Anthropic hard limit still holds with inference active.
# ---------------------------------------------------------------------------


def test_never_exceeds_four_breakpoints_with_inference_active():
    """A 5th cache_control is a 400, not a soft failure. System + tools +
    two conversation breakpoints is the ceiling."""
    provider = _make_provider()

    tools = [
        ToolSpec(
            name="do_something",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )
    ]

    def _with_tools(turns: int, clock: str) -> ChatRequest:
        base = _request_with_pre_user_reminder(turns, clock)
        return ChatRequest(messages=base.messages, tools=tools)

    params = _run_sequence(
        provider,
        [_with_tools(3, "T1"), _with_tools(4, "T2"), _with_tools(5, "T3")],
    )

    for i, p in enumerate(params):
        assert _total_breakpoints(p) <= 4, (
            f"request {i + 1} placed {_total_breakpoints(p)} breakpoints; "
            "Anthropic rejects more than 4"
        )


# ---------------------------------------------------------------------------
# 7. A retry is not evidence.
# ---------------------------------------------------------------------------


def test_identical_reissue_is_not_read_as_a_stable_tail():
    """A byte-identical re-issue (retry / fallback) has a longest common
    prefix equal to its whole length, which would naively measure an unstable
    suffix of 0 and stamp the volatile tail. It must instead reuse the last
    real observation."""
    provider = _make_provider()
    repeated = _request_with_pre_user_reminder(4, "T2")

    params = _run_sequence(
        provider,
        [
            _request_with_pre_user_reminder(3, "T1"),
            repeated,
            # Same request object -> byte-identical message array.
            ChatRequest(messages=list(repeated.messages)),
        ],
    )

    messages = params[2]["messages"]
    reminder_idx = max(
        i for i, m in enumerate(messages) if "system-reminder" in str(m.get("content"))
    )
    for idx in _breakpoint_message_indices(params[2]):
        assert idx < reminder_idx, (
            "an identical re-issue must not be mistaken for evidence that the "
            "regenerated tail is stable"
        )


# ---------------------------------------------------------------------------
# 8. The knob.
# ---------------------------------------------------------------------------


def test_knob_off_restores_strict_metadata_only_behavior(caplog):
    """`cache_infer_stability_from_history: false` reverts to the previous
    behavior exactly -- skip, and warn, on every request."""
    provider = _make_provider(cache_infer_stability_from_history=False)

    with caplog.at_level(logging.WARNING, logger="amplifier_module_provider_anthropic"):
        params = _run_sequence(
            provider,
            [
                _request_with_pre_user_reminder(3, "T1"),
                _request_with_pre_user_reminder(4, "T2"),
            ],
        )

    assert _breakpoint_message_indices(params[0]) == []
    assert _breakpoint_message_indices(params[1]) == []
    assert (
        sum(
            1
            for r in caplog.records
            if "cannot be distinguished from stable history" in r.message
        )
        >= 2
    ), "with inference disabled, every request must take the old skip path"


def test_prompt_caching_disabled_records_no_fingerprints():
    """`enable_prompt_caching: false` must not do fingerprint bookkeeping at
    all -- no work, no memory, for a feature that is off."""
    provider = _make_provider(enable_prompt_caching=False)
    _run_sequence(
        provider,
        [
            _request_with_pre_user_reminder(3, "T1"),
            _request_with_pre_user_reminder(4, "T2"),
        ],
    )
    assert provider._prefix_fingerprints == {}


# ---------------------------------------------------------------------------
# Unit-level properties of the measurement itself.
# ---------------------------------------------------------------------------


def test_fingerprint_ignores_cache_control_markers():
    """A message must hash identically whether or not this request happened
    to stamp it -- otherwise every comparison reports a spurious difference at
    exactly the boundary being reasoned about."""
    provider = _make_provider()
    plain = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    stamped = {
        "role": "user",
        "content": [
            {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
        ],
    }
    assert provider._message_fingerprint(plain) == provider._message_fingerprint(stamped)
    different = {"role": "user", "content": [{"type": "text", "text": "hello!"}]}
    assert provider._message_fingerprint(plain) != provider._message_fingerprint(
        different
    )


def test_observed_unstable_suffix_measures_the_dropped_tail():
    """Direct measurement check, independent of the request path."""
    provider = _make_provider()

    def _msg(text: str) -> dict:
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    first = [_msg("a"), _msg("b"), _msg("c"), _msg("R1"), _msg("u1")]
    assert provider._observed_unstable_suffix_length(first, None) == (None, "none")

    # R1 vanished; u1 shifted up; two new messages appended.
    second = [_msg("a"), _msg("b"), _msg("c"), _msg("u1"), _msg("A1"), _msg("R2")]
    observed, state = provider._observed_unstable_suffix_length(second, None)
    assert state == "ok"
    # lcp == 3 (a, b, c); previous had 5 -> 2 trailing entries did not survive.
    assert observed == 2


def test_fingerprint_memory_is_bounded():
    """A long-lived provider serving many conversations must not grow without
    limit."""
    from amplifier_module_provider_anthropic import _MAX_TRACKED_CONVERSATIONS

    provider = _make_provider()
    for i in range(_MAX_TRACKED_CONVERSATIONS * 3):
        provider._observed_unstable_suffix_length(
            [{"role": "user", "content": [{"type": "text", "text": f"convo {i}"}]}],
            None,
        )
    assert len(provider._prefix_fingerprints) <= _MAX_TRACKED_CONVERSATIONS
