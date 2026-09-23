"""Wire-level acceptance tests for Opus 5.5 preserved thinking."""

from __future__ import annotations

import asyncio

from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_anthropic import _preserved_thinking
from tests._wire import WireRecorder, make_wire_provider, opus55_message, sse_events


def _request(messages: list[Message], *, tools: bool = False) -> ChatRequest:
    specs = None
    if tools:
        specs = [
            ToolSpec(
                name="computer",
                parameters={},
                type="computer_20251124",
                display_width_px=1024,
                display_height_px=768,
            )
        ]
    return ChatRequest(messages=messages, tools=specs)


def test_exact_wire_replay_and_toolset_result_echo():
    first_content = [
        {"type": "thinking", "thinking": "", "signature": "sig-1"},
        {"type": "text", "text": "Checking."},
        {
            "type": "tool_use",
            "id": "tool-1",
            "name": "left_click",
            "input": {"coordinate": [1, 2]},
            "toolset_name": "computer",
        },
    ]
    recorder = WireRecorder(
        messages=[
            opus55_message(first_content, stop_reason="tool_use"),
            opus55_message([{"type": "text", "text": "Done."}]),
        ]
    )
    provider = make_wire_provider(recorder)
    first = asyncio.run(
        provider.complete(_request([Message(role="user", content="click")], tools=True))
    )
    assistant = Message(
        role="assistant",
        content=[block.model_dump() for block in first.content],
        tool_calls=first.tool_calls,
        metadata=first.metadata,
    )
    asyncio.run(
        provider.complete(
            _request(
                [
                    Message(role="user", content="click"),
                    assistant,
                    Message(role="tool", tool_call_id="tool-1", content="ok"),
                ],
                tools=True,
            )
        )
    )

    second = recorder.message_bodies()[1]
    replayed = next(m for m in second["messages"] if m["role"] == "assistant")
    assert replayed["content"] == first_content
    result = next(
        block
        for message in second["messages"]
        if message["role"] == "user" and isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    )
    assert result["toolset_name"] == "computer"
    assert (
        _preserved_thinking.BETA_HEADER_THINKING_BINDING
        in recorder.beta_headers(1)
    )


def test_prefix_binding_mismatch_retries_once_with_drop_block():
    recorder = WireRecorder(
        errors=[
            (
                400,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": (
                            "messages.1.content.0: Invalid `signature` in `thinking` "
                            "block. The block is bound to a different conversation. "
                            "Remove the block, or set "
                            "`thinking.block_binding.prefix_mismatch_behavior` "
                            'to "drop_block".'
                        ),
                    },
                },
            )
        ],
        messages=[opus55_message([{"type": "text", "text": "Recovered."}])],
    )
    provider = make_wire_provider(recorder)
    result = asyncio.run(
        provider.complete(_request([Message(role="user", content="hi")]))
    )
    bodies = recorder.message_bodies()
    assert len(bodies) == 2
    assert "block_binding" not in bodies[0]["thinking"]
    assert bodies[1]["thinking"]["block_binding"] == {
        "prefix_mismatch_behavior": "drop_block"
    }
    assert result.metadata["anthropic"]["thinking_binding"] == "drop_block"
    assert provider.coordinator is not None
    assert "provider:thinking_binding_retry" in provider.coordinator.hooks.emitted_names()


def test_streaming_updates_marks_progress_delta_and_preserves_signature():
    message = opus55_message(
        [{"type": "thinking", "thinking": "Checking.", "signature": "sig"}]
    )
    recorder = WireRecorder(messages=[sse_events(message)])
    provider = make_wire_provider(
        recorder, streaming=True, thinking_display="updates"
    )
    response = asyncio.run(
        provider.complete(_request([Message(role="user", content="hi")]))
    )
    assert provider.coordinator is not None
    deltas = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:stream_block_delta"
    ]
    assert deltas[0]["progress_update"] is True
    thinking = next(block for block in response.content if block.type == "thinking")
    assert thinking.signature == "sig"