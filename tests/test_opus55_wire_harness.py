"""Smoke test for the real-SDK wire harness (tests/_wire.py)."""

from __future__ import annotations

import asyncio

from amplifier_core.message_models import ChatRequest, Message

from tests._wire import WireRecorder, make_wire_provider, opus55_message, sse_events


def test_harness_records_non_streaming_and_streaming():
    recorder = WireRecorder(
        messages=[
            opus55_message([{"type": "text", "text": "hi"}]),
            opus55_message([{"type": "text", "text": "hi"}]),
        ]
    )
    provider = make_wire_provider(recorder, streaming=False)
    request = ChatRequest(messages=[Message(role="user", content="hello")])
    response = asyncio.run(provider.complete(request))
    assert response.text == "hi"

    streaming_recorder = WireRecorder(
        messages=[sse_events(opus55_message([{"type": "text", "text": "hi"}]))]
    )
    streaming_provider = make_wire_provider(streaming_recorder, streaming=True)
    streamed_response = asyncio.run(streaming_provider.complete(request))
    assert streamed_response.text == "hi"

    bodies = recorder.message_bodies()
    assert len(bodies) == 1
    assert bodies[0]["model"] == "claude-opus-5-5"
    stream_bodies = streaming_recorder.message_bodies()
    assert len(stream_bodies) == 1
    assert stream_bodies[0]["model"] == "claude-opus-5-5"
