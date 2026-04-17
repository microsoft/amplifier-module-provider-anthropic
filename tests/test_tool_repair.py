"""Tests for tool result repair and infinite loop prevention."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock


from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolCallBlock
from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import DummyResponse, FakeCoordinator


class MockStreamManager:
    """Mock for the streaming context manager returned by client.messages.stream().

    Separates the API message response (returned by get_final_message) from the
    HTTP response (accessed via .response for rate limit headers).
    """

    def __init__(self, api_response: DummyResponse):
        self._api_response = api_response
        # The real SDK exposes .response as the HTTP response (with headers).
        # Provide a minimal stub so header extraction doesn't crash.
        self.response = SimpleNamespace(headers={})

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def get_final_message(self):
        return self._api_response


def create_stream_mock(response: DummyResponse):
    """Create a mock for client.messages.stream that returns an async context manager."""
    return MagicMock(return_value=MockStreamManager(response))


def create_raw_response_mock(response: DummyResponse):
    """Create a mock for client.messages.with_raw_response.create (non-streaming path).

    The non-streaming path calls with_raw_response.create() which returns an
    object with .parse() → response and .headers → dict.
    """
    raw = MagicMock()
    raw.parse.return_value = response
    raw.headers = {}
    return AsyncMock(return_value=raw)


def test_tool_call_sequence_missing_tool_message_is_repaired():
    """Missing tool results should be repaired with synthetic results and emit event."""
    # use_streaming=False so we use with_raw_response.create (which we mock)
    provider = AnthropicProvider(api_key="test-key", config={"use_streaming": False})
    provider.client.messages.with_raw_response.create = create_raw_response_mock(
        DummyResponse()
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="do_something", input={"value": 1})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Should succeed (not raise validation error)
    provider.client.messages.with_raw_response.create.assert_awaited_once()

    # Should not emit validation error
    assert all(
        event_name != "provider:validation_error"
        for event_name, _ in fake_coordinator.hooks.events
    )

    # Should emit repair event
    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["provider"] == "anthropic"
    assert repair_events[0][1]["repair_count"] == 1
    assert repair_events[0][1]["repairs"][0]["tool_name"] == "do_something"


def test_repaired_tool_ids_are_not_detected_again():
    """Repaired tool IDs should be tracked and not trigger infinite detection loops.

    This test verifies the fix for the infinite loop bug where:
    1. Missing tool results are detected and synthetic results are injected
    2. Synthetic results are NOT persisted to message store
    3. On next iteration, same missing tool results are detected again
    4. This creates an infinite loop of detection -> injection -> detection

    The fix tracks repaired tool IDs to skip re-detection.
    """
    # use_streaming=False so we use with_raw_response.create (which we mock)
    provider = AnthropicProvider(api_key="test-key", config={"use_streaming": False})
    provider.client.messages.with_raw_response.create = create_raw_response_mock(
        DummyResponse()
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Create a request with missing tool result
    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_abc123", name="grep", input={"pattern": "test"})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    # First call - should detect and repair
    asyncio.run(provider.complete(request))

    # Verify repair happened
    assert "call_abc123" in provider._repaired_tool_ids  # pyright: ignore[reportAttributeAccessIssue]
    repair_events_1 = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_1) == 1

    # Clear events for second call
    fake_coordinator.hooks.events.clear()

    # Second call with SAME messages (simulating message store not persisting synthetic results)
    # This would previously cause infinite loop detection
    messages_2 = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_abc123", name="grep", input={"pattern": "test"})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request_2 = ChatRequest(messages=messages_2)

    asyncio.run(provider.complete(request_2))

    # Should NOT emit another repair event for the same tool ID
    repair_events_2 = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_2) == 0, "Should not re-detect already-repaired tool IDs"


def test_multiple_missing_tool_results_all_tracked():
    """Multiple missing tool results should all be tracked to prevent infinite loops."""
    # use_streaming=False so we use with_raw_response.create (which we mock)
    provider = AnthropicProvider(api_key="test-key", config={"use_streaming": False})
    provider.client.messages.with_raw_response.create = create_raw_response_mock(
        DummyResponse()
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Create request with 3 parallel tool calls, none with results
    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="grep", input={"pattern": "a"}),
                ToolCallBlock(id="call_2", name="grep", input={"pattern": "b"}),
                ToolCallBlock(id="call_3", name="grep", input={"pattern": "c"}),
            ],
        ),
        Message(role="user", content="No tool results"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # All 3 should be tracked
    assert provider._repaired_tool_ids == {"call_1", "call_2", "call_3"}  # pyright: ignore[reportAttributeAccessIssue]

    # Verify repair event has all 3
    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["repair_count"] == 3


# =============================================================================
# Streaming Mode Tests (default behavior)
# =============================================================================


def test_streaming_tool_call_sequence_missing_tool_message_is_repaired():
    """Missing tool results should be repaired with streaming API (default mode)."""
    # Default use_streaming=True, mock the streaming API
    provider = AnthropicProvider(api_key="test-key")
    provider.client.messages.stream = create_stream_mock(DummyResponse())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="do_something", input={"value": 1})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Should succeed (not raise validation error)
    provider.client.messages.stream.assert_called_once()

    # Should not emit validation error
    assert all(
        event_name != "provider:validation_error"
        for event_name, _ in fake_coordinator.hooks.events
    )

    # Should emit repair event
    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["provider"] == "anthropic"
    assert repair_events[0][1]["repair_count"] == 1
    assert repair_events[0][1]["repairs"][0]["tool_name"] == "do_something"


def test_streaming_repaired_tool_ids_are_not_detected_again():
    """Repaired tool IDs should be tracked with streaming API (default mode)."""
    # Default use_streaming=True
    provider = AnthropicProvider(api_key="test-key")
    provider.client.messages.stream = create_stream_mock(DummyResponse())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Create a request with missing tool result
    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(
                    id="call_stream_123", name="grep", input={"pattern": "test"}
                )
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    # First call - should detect and repair
    asyncio.run(provider.complete(request))

    # Verify repair happened
    assert "call_stream_123" in provider._repaired_tool_ids  # pyright: ignore[reportAttributeAccessIssue]
    repair_events_1 = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_1) == 1

    # Clear events for second call
    fake_coordinator.hooks.events.clear()

    # Second call with SAME messages (simulating message store not persisting synthetic results)
    messages_2 = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(
                    id="call_stream_123", name="grep", input={"pattern": "test"}
                )
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request_2 = ChatRequest(messages=messages_2)

    asyncio.run(provider.complete(request_2))

    # Should NOT emit another repair event for the same tool ID
    repair_events_2 = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_2) == 0, "Should not re-detect already-repaired tool IDs"


def test_streaming_multiple_missing_tool_results_all_tracked():
    """Multiple missing tool results should all be tracked with streaming API (default mode)."""
    # Default use_streaming=True
    provider = AnthropicProvider(api_key="test-key")
    provider.client.messages.stream = create_stream_mock(DummyResponse())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Create request with 3 parallel tool calls, none with results
    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="stream_1", name="grep", input={"pattern": "a"}),
                ToolCallBlock(id="stream_2", name="grep", input={"pattern": "b"}),
                ToolCallBlock(id="stream_3", name="grep", input={"pattern": "c"}),
            ],
        ),
        Message(role="user", content="No tool results"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # All 3 should be tracked
    assert provider._repaired_tool_ids == {"stream_1", "stream_2", "stream_3"}  # pyright: ignore[reportAttributeAccessIssue]

    # Verify repair event has all 3
    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["repair_count"] == 3


# =============================================================================
# Session Resume Tests — synthetic results must survive _convert_messages()
# =============================================================================


def test_convert_messages_content_blocks_contribute_to_valid_tool_ids():
    """_convert_messages() must honour tool IDs found in content blocks.

    Regression test for the session-resume failure:
      - Session is saved after an overloaded_error with an assistant message
        that has ToolCallBlock entries in its content array but NO following
        tool_result messages.
      - On resume, complete() injects synthetic tool results.
      - _convert_messages() receives model_dump() output of those messages.
      - Without the fix the orphan filter drops synthetic results whose IDs
        appear only in content blocks (not in tool_calls dict field), producing
        a 400 "tool_use ids were found without tool_result" from Anthropic.
      - With the fix, content blocks are also scanned so the results survive.
    """
    provider = AnthropicProvider(api_key="test-key")

    # Simulate model_dump() output for a resumed session where the assistant
    # message has content blocks but the tool_calls field is absent — the
    # specific edge-case that triggers the orphan-filter bug.
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "id": "toolu_resume_001",
                    "name": "read_file",
                    "input": {"path": "/foo.py"},
                },
                {
                    "type": "tool_call",
                    "id": "toolu_resume_002",
                    "name": "grep",
                    "input": {"pattern": "test"},
                },
            ],
            # Intentionally NO "tool_calls" field — simulates the format
            # inconsistency that caused the bug on session resume.
        },
        # Synthetic tool results injected by complete() during resume
        {
            "role": "tool",
            "tool_call_id": "toolu_resume_001",
            "content": "[error: provider overloaded — synthetic result]",
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_resume_002",
            "content": "[error: provider overloaded — synthetic result]",
        },
        {"role": "user", "content": "Please continue."},
    ]

    result = provider._convert_messages(messages)  # pyright: ignore[reportAttributeAccessIssue]

    # Expected: assistant block · batched tool-results user block · user block
    assert len(result) == 3, (
        f"Expected 3 Anthropic messages (assistant + tool-results + user), "
        f"got {len(result)}: {result}"
    )

    tool_results_msg = result[1]
    assert tool_results_msg["role"] == "user"
    assert isinstance(tool_results_msg["content"], list)
    assert len(tool_results_msg["content"]) == 2, (
        "Both synthetic tool results should have survived the orphan filter"
    )
    returned_ids = {blk["tool_use_id"] for blk in tool_results_msg["content"]}
    assert returned_ids == {"toulu_resume_001", "toulu_resume_002"} or returned_ids == {
        "toolu_resume_001",
        "toolu_resume_002",
    }


def test_convert_messages_tool_use_type_content_blocks_contribute_to_valid_ids():
    """_convert_messages() must also accept IDs in 'tool_use' typed content blocks.

    Complements the 'tool_call' variant above.  Anthropic's own wire format uses
    type='tool_use' in content blocks; either spelling must be accepted.
    """
    provider = AnthropicProvider(api_key="test-key")

    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_anthropic_fmt",
                    "name": "bash",
                    "input": {"cmd": "ls"},
                },
            ],
            # No tool_calls field — only content blocks carry the ID
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_anthropic_fmt",
            "content": "[error: provider overloaded — synthetic result]",
        },
        {"role": "user", "content": "What did you find?"},
    ]

    result = provider._convert_messages(messages)  # pyright: ignore[reportAttributeAccessIssue]

    assert len(result) == 3
    tool_results_msg = result[1]
    assert tool_results_msg["role"] == "user"
    assert len(tool_results_msg["content"]) == 1
    assert tool_results_msg["content"][0]["tool_use_id"] == "toolu_anthropic_fmt"


def test_resume_end_to_end_synthetic_results_reach_anthropic_api():
    """End-to-end: synthetic results injected during resume must reach the API.

    Simulates a full session-resume cycle:
      1. Session was saved after an overloaded_error mid-turn — the assistant
         message has three ToolCallBlock entries in content but NO tool_result
         messages follow it (mirrors the user's log:
         "[PROVIDER] Anthropic: Detected 3 missing tool result(s)").
      2. complete() is called for the resumed turn.
      3. complete() detects the missing results and injects 3 synthetic tool
         messages into request.messages.
      4. _convert_messages() must NOT drop those synthetics as "orphaned".
      5. The Anthropic API payload must contain a tool_result block for every
         tool_use block — otherwise Anthropic returns a 400.
    """
    import json

    provider = AnthropicProvider(api_key="test-key", config={"use_streaming": False})
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Capture the raw kwargs forwarded to the mocked API for inspection
    captured_kwargs: list[dict] = []

    async def capturing_create(**kwargs):  # type: ignore[return]
        captured_kwargs.append(kwargs)
        raw = MagicMock()
        raw.parse.return_value = DummyResponse()
        raw.headers = {}
        return raw

    provider.client.messages.with_raw_response.create = capturing_create  # type: ignore[method-assign]

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(
                    id="resume_call_1", name="read_file", input={"path": "/a.py"}
                ),
                ToolCallBlock(
                    id="resume_call_2", name="read_file", input={"path": "/b.py"}
                ),
                ToolCallBlock(
                    id="resume_call_3", name="grep", input={"pattern": "TODO"}
                ),
            ],
        ),
        Message(role="user", content="Please continue where you left off."),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Repair event must have been emitted
    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["repair_count"] == 3

    # API must have been called exactly once
    assert len(captured_kwargs) == 1, "API should be called exactly once after repair"

    api_messages = captured_kwargs[0]["messages"]

    # Gather every tool call id from assistant content blocks (the kernel uses
    # type="tool_call"; Anthropic wire format uses type="tool_use" — accept both)
    # and every tool_result id from the following user message.
    tool_call_ids: set[str] = set()
    tool_result_ids: set[str] = set()
    for msg in api_messages:
        if msg.get("role") == "assistant":
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") in ("tool_use", "tool_call"):
                    tool_call_ids.add(block["id"])
        if msg.get("role") == "user":
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_result_ids.add(block["tool_use_id"])

    assert tool_call_ids == {"resume_call_1", "resume_call_2", "resume_call_3"}, (
        f"Assistant payload should contain all 3 tool call blocks; got {tool_call_ids}\n"
        f"Full API payload:\n{json.dumps(api_messages, indent=2, default=str)}"
    )
    assert tool_result_ids == tool_call_ids, (
        f"Every tool call must have a matching tool_result — none must be dropped as orphaned.\n"
        f"  tool call ids:   {tool_call_ids}\n"
        f"  tool_result ids: {tool_result_ids}\n"
        f"  Full API payload:\n{json.dumps(api_messages, indent=2, default=str)}"
    )
