"""Captured-client coverage for optional v1 instruction-layout lowering."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.message_models import ChatRequest, Message, ThinkingBlock, ToolCallBlock

from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import DummyResponse


def _provider(default_model: str = "claude-opus-4-8") -> AnthropicProvider:
    return AnthropicProvider(
        api_key="test-key",
        config={
            "default_model": default_model,
            "enable_prompt_caching": True,
            "max_retries": 0,
            "use_streaming": False,
        },
    )


def _capture(provider: AnthropicProvider, request: ChatRequest) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def create(**params: Any) -> MagicMock:
        captured.update(params)
        response = MagicMock()
        response.parse = AsyncMock(return_value=DummyResponse())
        response.headers = {}
        return response

    provider.client.messages.with_raw_response.create = AsyncMock(side_effect=create)
    asyncio.run(provider.complete(request))
    return captured


def _descriptor(
    *,
    key: str,
    placement: str,
    binding: str = "live",
    authority: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "version": 1,
        "source": "test-source",
        "key": key,
        "placement": placement,
        "binding": binding,
    }
    if authority is not None:
        result["authority"] = authority
    if binding == "fixed":
        target: dict[str, Any]
        if placement == "head":
            target = {"kind": "conversation_head", "session_id": "session"}
        elif placement == "before_human":
            target = {
                "input_id": "historical-input",
                "message_id": "historical-message",
                "origin": "human",
            }
        else:
            target = {"after_message_id": "historical-message"}
        result.update(
            entry_id=f"session:test-source:{key}",
            event_key=key,
            session_id="session",
            target=target,
            order=1,
            disposition="pending",
        )
    return result


def _instruction(
    content: str,
    *,
    key: str,
    placement: str,
    binding: str = "live",
    authority: str | None = "advisory",
) -> Message:
    return Message(
        role="system",
        content=content,
        metadata={
            "amplifier:instruction": _descriptor(
                key=key, placement=placement, binding=binding, authority=authority
            )
        },
    )


def _text_blocks(message: dict[str, Any]) -> list[str]:
    content = message["content"]
    if isinstance(content, str):
        return [content]
    return [
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def _cache_controlled_message_indices(params: dict[str, Any]) -> list[int]:
    return [
        index
        for index, message in enumerate(params["messages"])
        if isinstance(message.get("content"), list)
        and any(
            isinstance(block, dict) and "cache_control" in block
            for block in message["content"]
        )
    ]


def test_layout_version_is_advertised_only_for_supported_default_model() -> None:
    assert _provider("claude-opus-4-8").instruction_layout_version == 1
    assert _provider("claude-sonnet-4-7").instruction_layout_version is None
    assert _provider().instruction_layout_authority_v1 is True


def test_v1_rejects_an_unsupported_per_request_model() -> None:
    provider = _provider()
    request = ChatRequest(messages=[_instruction("head", key="head", placement="head")])

    with pytest.raises(ValueError, match="does not support"):
        asyncio.run(provider.complete(request, model="claude-sonnet-4-7"))


def test_v1_hoists_only_head_and_caches_stable_head_prefix() -> None:
    params = _capture(
        _provider(),
        ChatRequest(
            messages=[
                Message(role="system", content="legacy base"),
                _instruction(
                    "fixed head", key="fixed-head", placement="head", binding="fixed"
                ),
                _instruction("live head", key="live-head", placement="head"),
                _instruction(
                    "fixed before old input",
                    key="fixed-before-old",
                    placement="before_human",
                    binding="fixed",
                ),
                Message(role="user", content="older human"),
                Message(
                    role="assistant",
                    content=[ThinkingBlock(thinking="replay thinking", signature="sig")],
                ),
                _instruction("before newest human", key="current", placement="before_human"),
                Message(role="user", content="newest human"),
            ]
        ),
    )

    assert [block["text"] for block in params["system"]] == [
        "legacy base",
        "fixed head",
        "live head",
    ]
    assert "cache_control" not in params["system"][0]
    assert "cache_control" in params["system"][1]
    assert "cache_control" not in params["system"][2]
    assert sum("fixed head" in block["text"] for block in params["system"]) == 1

    # The positioned record is a declared carrier, merged with the next native
    # user message rather than becoming a fabricated human boundary.
    assert [message["role"] for message in params["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert params["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "replay thinking", "signature": "sig"}
    ]
    assert _text_blocks(params["messages"][0]) == [
        '[Amplifier system instruction {"source":"test-source","placement":"before_human","binding":"fixed"}]\n'
        "fixed before old input",
        "older human",
    ]
    assert _text_blocks(params["messages"][-1]) == [
        '[Amplifier system instruction {"source":"test-source","placement":"before_human","binding":"live"}]\n'
        "before newest human",
        "newest human",
    ]


@pytest.mark.parametrize("placement", ["before_human", "tail"])
def test_live_inline_instruction_carrier_excludes_its_native_user_message_from_cache(
    placement: str,
) -> None:
    params = _capture(
        _provider(),
        ChatRequest(
            messages=[
                Message(role="user", content="stable earlier human"),
                Message(role="assistant", content="stable earlier reply"),
                _instruction(
                    "live current instruction",
                    key=f"live-{placement}",
                    placement=placement,
                ),
                Message(role="user", content="current human"),
            ]
        ),
    )

    carrier_index = next(
        index
        for index, message in enumerate(params["messages"])
        if "live current instruction" in " ".join(_text_blocks(message))
    )
    cache_indices = _cache_controlled_message_indices(params)

    assert carrier_index == len(params["messages"]) - 1
    assert cache_indices == [0]
    assert all(index < carrier_index for index in cache_indices)


@pytest.mark.parametrize("placement", ["before_human", "tail"])
def test_delivered_fixed_inline_instruction_carrier_remains_cache_eligible(
    placement: str,
) -> None:
    instruction = _instruction(
        "fixed historical instruction",
        key=f"fixed-{placement}",
        placement=placement,
        binding="fixed",
    )
    instruction.metadata["amplifier:instruction"]["disposition"] = "delivered"
    params = _capture(
        _provider(),
        ChatRequest(
            messages=[
                Message(role="user", content="stable earlier human"),
                Message(role="assistant", content="stable earlier reply"),
                instruction,
                Message(role="user", content="current human"),
            ]
        ),
    )

    carrier_index = next(
        index
        for index, message in enumerate(params["messages"])
        if "fixed historical instruction" in " ".join(_text_blocks(message))
    )

    assert carrier_index == len(params["messages"]) - 1
    assert carrier_index in _cache_controlled_message_indices(params)


def test_v1_mixed_legacy_system_messages_hoist_without_mutating_canonical_input() -> None:
    request = ChatRequest(
        messages=[
            Message(role="system", content="legacy base system"),
            _instruction(
                "live positioned instruction",
                key="positioned",
                placement="before_human",
            ),
            Message(role="user", content="actual human work"),
            Message(role="system", content="legacy reminder"),
        ]
    )
    original = request.model_dump()

    params = _capture(_provider(), request)

    assert [block["text"] for block in params["system"]] == [
        "legacy base system",
        "legacy reminder",
    ]
    assert _text_blocks(params["messages"][-1]) == [
        '[Amplifier system instruction {"source":"test-source","placement":"before_human","binding":"live"}]\n'
        "live positioned instruction",
        "actual human work",
    ]
    assert _cache_controlled_message_indices(params) == []
    assert request.model_dump() == original


def test_v1_tail_after_parallel_tool_batch_preserves_grouping_and_order() -> None:
    params = _capture(
        _provider(),
        ChatRequest(
            messages=[
                Message(role="user", content="run both"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {"id": "one", "tool": "first", "arguments": {}},
                        {"id": "two", "tool": "second", "arguments": {}},
                    ],
                ),
                Message(role="tool", tool_call_id="one", content="result one"),
                Message(role="tool", tool_call_id="two", content="result two"),
                _instruction(
                    "batch feedback",
                    key="feedback",
                    placement="tail",
                    binding="fixed",
                ),
                Message(role="user", content="continue"),
            ]
        ),
    )

    assert [message["role"] for message in params["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    result_and_feedback = params["messages"][-1]["content"]
    assert [block["type"] for block in result_and_feedback] == [
        "tool_result",
        "tool_result",
        "text",
        "text",
    ]
    assert [block["tool_use_id"] for block in result_and_feedback[:2]] == ["one", "two"]
    assert result_and_feedback[2]["text"].endswith("batch feedback")
    assert result_and_feedback[3]["text"] == "continue"


@pytest.mark.parametrize(
    "message",
    [
        Message(
            role="system",
            content="must not disappear",
            metadata={"amplifier:instruction": {"version": 1}},
        ),
        Message(
            role="user",
            content="must not become an instruction",
            metadata={
                "amplifier:instruction": _descriptor(
                    key="wrong-role", placement="before_human"
                )
            },
        ),
    ],
)
def test_malformed_marked_metadata_fails_before_wire_conversion(message: Message) -> None:
    provider = _provider()
    original = message.model_dump()
    request = ChatRequest(messages=[message])

    with pytest.raises(ValueError, match="amplifier:instruction"):
        asyncio.run(provider.complete(request))

    assert request.messages[0].model_dump() == original


def test_malformed_marked_data_is_not_repaired_before_it_fails() -> None:
    malformed = Message(
        role="system",
        content="must not disappear",
        metadata={"amplifier:instruction": {"version": 1}},
    )
    unfinished_call = Message(
        role="assistant",
        content=[ToolCallBlock(id="missing", name="tool", input={})],
    )
    request = ChatRequest(messages=[malformed, unfinished_call])
    original = request.model_dump()

    with pytest.raises(ValueError, match="amplifier:instruction"):
        asyncio.run(_provider().complete(request))

    assert request.model_dump() == original


@pytest.mark.parametrize(
    "descriptor",
    [
        {
            **_descriptor(key="retired-head", placement="head", binding="fixed"),
            "disposition": "retired",
            "retire_reason": "source retired it",
        },
        {
            **_descriptor(key="pruned-tail", placement="tail", binding="fixed"),
            "disposition": "anchor_pruned",
        },
        {
            **_descriptor(key="deferred-tail", placement="tail", binding="fixed"),
            "deferred_origin": True,
        },
    ],
)
def test_unlowerable_fixed_records_fail_without_sdk_dispatch(
    descriptor: dict[str, Any],
) -> None:
    request = ChatRequest(
        messages=[
            Message(
                role="system",
                content="must not be resurrected",
                metadata={"amplifier:instruction": descriptor},
            )
        ]
    )
    original = request.model_dump()
    provider = _provider()
    create = AsyncMock()
    provider.client.messages.with_raw_response.create = create

    with pytest.raises(ValueError, match="fixed amplifier:instruction"):
        asyncio.run(provider.complete(request))

    assert create.await_count == 0
    assert request.model_dump() == original


def test_unmarked_system_messages_keep_legacy_hoisting_and_wire_shape() -> None:
    params = _capture(
        _provider("claude-sonnet-4-7"),
        ChatRequest(
            messages=[
                Message(role="system", content="legacy first"),
                Message(role="user", content="human"),
                Message(role="system", content="legacy inline"),
            ]
        ),
    )

    assert params["system"] == [
        {
            "type": "text",
            "text": "legacy first\n\nlegacy inline",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert params["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "human"}]}
    ]


def test_authoritative_tail_uses_consolidated_native_system_without_mutating_request() -> None:
    request = ChatRequest(
        messages=[
            Message(role="user", content="work"),
            _instruction(
                "historical authority default",
                key="tail-one",
                placement="tail",
                authority=None,
            ),
            _instruction(
                "explicit authority",
                key="tail-two",
                placement="tail",
                authority="authoritative",
            ),
            Message(role="assistant", content="previous reply"),
        ]
    )
    original = request.model_dump()

    params = _capture(_provider(), request)

    assert [message["role"] for message in params["messages"]] == [
        "user",
        "system",
        "assistant",
    ]
    assert _text_blocks(params["messages"][1]) == [
        "historical authority default\n\nexplicit authority"
    ]
    assert "system" not in params
    assert request.model_dump() == original


def test_authoritative_tail_after_completed_tool_batch_keeps_native_pairing() -> None:
    params = _capture(
        _provider(),
        ChatRequest(
            messages=[
                Message(role="user", content="run tool"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[{"id": "call", "tool": "tool", "arguments": {}}],
                ),
                Message(role="tool", tool_call_id="call", content="tool result"),
                _instruction(
                    "authoritative tool feedback",
                    key="tool-tail",
                    placement="tail",
                    authority="authoritative",
                ),
                Message(role="assistant", content="continued reply"),
            ]
        ),
    )

    assert [message["role"] for message in params["messages"]] == [
        "user",
        "assistant",
        "user",
        "system",
        "assistant",
    ]
    tool_results = [
        {key: block[key] for key in ("type", "tool_use_id", "content")}
        for block in params["messages"][2]["content"]
    ]
    assert tool_results == [
        {"type": "tool_result", "tool_use_id": "call", "content": "tool result"}
    ]
    assert _text_blocks(params["messages"][3]) == ["authoritative tool feedback"]


def test_authoritative_before_human_uses_global_system_and_warns_once_per_source(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = ChatRequest(
        messages=[
            Message(role="user", content="older input"),
            Message(role="assistant", content="older reply"),
            _instruction(
                "must not become user content",
                key="before-human",
                placement="before_human",
                authority="authoritative",
            ),
            _instruction(
                "same source only warns once",
                key="before-human-second",
                placement="before_human",
                authority="authoritative",
            ),
            Message(role="user", content="current input"),
        ]
    )
    original = request.model_dump()

    params = _capture(_provider(), request)

    assert [block["text"] for block in params["system"]] == [
        "must not become user content",
        "same source only warns once",
    ]
    assert all(
        "must not become user content" not in " ".join(_text_blocks(message))
        for message in params["messages"]
        if message["role"] == "user"
    )
    warnings = [
        record.message
        for record in caplog.records
        if "global system fallback" in record.message
    ]
    assert len(warnings) == 1
    assert "source=test-source" in warnings[0]
    assert "key=before-human" in warnings[0]
    assert "placement=before_human" in warnings[0]
    assert "cache" in warnings[0]
    assert request.model_dump() == original


def test_authoritative_tail_in_illegal_position_uses_global_system() -> None:
    params = _capture(
        _provider(),
        ChatRequest(
            messages=[
                Message(role="assistant", content="previous reply"),
                _instruction(
                    "tail after assistant is illegal",
                    key="illegal-tail",
                    placement="tail",
                    authority="authoritative",
                ),
                Message(role="user", content="next input"),
            ]
        ),
    )

    assert [block["text"] for block in params["system"]] == [
        "tail after assistant is illegal"
    ]
    assert all(message["role"] != "system" for message in params["messages"])
    assert all(
        "tail after assistant is illegal" not in " ".join(_text_blocks(message))
        for message in params["messages"]
        if message["role"] == "user"
    )


@pytest.mark.parametrize("authority", [True, "invalid"])
def test_authority_must_be_a_closed_string_field(authority: object) -> None:
    descriptor = _descriptor(key="bad-authority", placement="tail", authority=None)
    descriptor["authority"] = authority
    request = ChatRequest(
        messages=[
            Message(
                role="system",
                content="must fail before dispatch",
                metadata={"amplifier:instruction": descriptor},
            )
        ]
    )

    with pytest.raises(ValueError, match="authority"):
        asyncio.run(_provider().complete(request))


def test_authority_unknown_field_remains_rejected() -> None:
    descriptor = _descriptor(key="unknown-authority-field", placement="tail", authority=None)
    descriptor["unexpected"] = "value"
    request = ChatRequest(
        messages=[
            Message(
                role="system",
                content="must fail before dispatch",
                metadata={"amplifier:instruction": descriptor},
            )
        ]
    )

    with pytest.raises(ValueError, match="invalid fields"):
        asyncio.run(_provider().complete(request))


def test_v1_overload_fallback_skips_models_without_v1_support() -> None:
    provider = _provider()

    assert provider._fallback_target_for_model("claude-opus-4-8") == "claude-sonnet-5"
    assert (
        provider._fallback_target_for_request(
            "claude-opus-4-8", requires_instruction_layout=True
        )
        is None
    )


def test_v1_content_only_tool_call_serializes_as_native_tool_use() -> None:
    request = ChatRequest(
        messages=[
            _instruction("head", key="head", placement="head"),
            Message(role="user", content="run tool"),
            Message(
                role="assistant",
                content=[ToolCallBlock(id="content-call", name="content_tool", input={"x": 1})],
            ),
            Message(role="tool", tool_call_id="content-call", content="result"),
        ]
    )
    original = request.model_dump()

    params = _capture(_provider(), request)

    assert params["messages"][1]["content"] == [
        {
            "type": "tool_use",
            "id": "content-call",
            "name": "content_tool",
            "input": {"x": 1},
        }
    ]
    assert params["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "content-call", "content": "result"}
    ]
    assert request.model_dump() == original


def test_v1_top_level_name_alias_serializes_as_native_tool_use() -> None:
    request = ChatRequest(
        messages=[
            _instruction("head", key="head", placement="head"),
            Message(role="user", content="run tool"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "name-call", "name": "portable_tool", "arguments": {"x": 2}}
                ],
            ),
            Message(role="tool", tool_call_id="name-call", content="result"),
        ]
    )
    original = request.model_dump()

    params = _capture(_provider(), request)

    assert params["messages"][1]["content"] == [
        {
            "type": "tool_use",
            "id": "name-call",
            "name": "portable_tool",
            "input": {"x": 2},
        }
    ]
    assert request.model_dump() == original


def test_v1_equivalent_duplicate_tool_calls_emit_once() -> None:
    request = ChatRequest(
        messages=[
            _instruction("head", key="head", placement="head"),
            Message(role="user", content="run tool"),
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(id="duplicate-call", name="duplicate_tool", input={"x": 3})
                ],
                tool_calls=[
                    {
                        "id": "duplicate-call",
                        "tool": "duplicate_tool",
                        "arguments": {"x": 3},
                    }
                ],
            ),
            Message(role="tool", tool_call_id="duplicate-call", content="result"),
        ]
    )
    original = request.model_dump()

    params = _capture(_provider(), request)

    tool_uses = [
        block
        for block in params["messages"][1]["content"]
        if block["type"] == "tool_use"
    ]
    assert tool_uses == [
        {
            "type": "tool_use",
            "id": "duplicate-call",
            "name": "duplicate_tool",
            "input": {"x": 3},
        }
    ]
    assert request.model_dump() == original