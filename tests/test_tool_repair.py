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


class DummyResponse:
    """Minimal response stub for provider tests."""

    def __init__(self, content=None):
        self.content = content or []
        self.usage = SimpleNamespace(input_tokens=0, output_tokens=0)
        self.stop_reason = "end_turn"
        self.model = "claude-sonnet-4-5-20250929"


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


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


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
