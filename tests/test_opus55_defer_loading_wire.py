"""Wire-level regression: defer_loading must survive the REAL Anthropic SDK
request path in the corrected per-member ``configs`` shape.

``tests/test_opus55_defer_loading.py`` already exercises
``_computer_toolset.translate_tools`` directly and drift-checks the result
against the installed SDK's typed ``ComputerToolsetConfigsParam`` /
``Computer<Member>ConfigParam`` annotations. That is a *unit*-level check of
the translation function's output shape; it never sends a request through
the real ``anthropic.AsyncAnthropic`` client, so it cannot catch a defect
that only appears once the SDK actually serializes the assembled request
(e.g. a client-side pydantic/TypedDict validation step silently dropping or
rejecting a key, or the provider's request-assembly path clobbering
``translate_tools``'s output before it reaches ``messages.create``).

These tests close that gap using the same real-SDK-over-``httpx2.MockTransport``
harness as ``tests/_wire.py`` (used elsewhere for preserved-thinking wire
assertions): the request body asserted on here is the actual JSON the real
SDK put on the wire for a genuine ``provider.complete()`` call, not a
hand-built stand-in and not a mocked ``messages.create`` call-arg capture.

Round 2 regression under test: ``defer_loading`` must appear inside each
enabled toolset member's own config entry (``configs[member]["defer_loading"]``),
never as ``configs["_defer_loading"]`` (the invalid Round-1 shape) and never
as a top-level ``tools[i]["defer_loading"]`` key.
"""

from __future__ import annotations

import asyncio

from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_anthropic import _computer_toolset
from tests._wire import WireRecorder, make_wire_provider, opus55_message


def _computer_tool_request(**tool_overrides) -> ChatRequest:
    tool = ToolSpec(
        name="computer",
        parameters={},
        type="computer_20251124",
        display_width_px=1024,
        display_height_px=768,
        **tool_overrides,
    )
    return ChatRequest(messages=[Message(role="user", content="hi")], tools=[tool])


def _send(request: ChatRequest) -> dict:
    """Round-trip ``request`` through the real SDK and return the single
    ``POST /v1/messages`` request body the mock transport actually received."""
    recorder = WireRecorder(messages=[opus55_message([{"type": "text", "text": "hi"}])])
    provider = make_wire_provider(recorder)
    asyncio.run(provider.complete(request))
    bodies = recorder.message_bodies()
    assert len(bodies) == 1
    return bodies[0]


class TestDeferLoadingWireLevel:
    def test_defer_loading_true_reaches_wire_in_per_member_shape(self):
        body = _send(_computer_tool_request(defer_loading=True))
        assert len(body["tools"]) == 1
        wire_tool = body["tools"][0]

        assert wire_tool["type"] == "computer_toolset_20260801"
        configs = wire_tool["configs"]

        # The invalid Round-1 shapes must never reach the actual wire body.
        assert "_defer_loading" not in configs
        assert "defer_loading" not in wire_tool

        # Zoom is disabled by default (legacy tool declared no zoom
        # capability) and carries no defer_loading -- only `enabled: False`.
        assert configs["zoom"] == {"enabled": False}

        expected_members = _computer_toolset.TOOLSET_MEMBERS - {"zoom"}
        assert set(configs.keys()) == _computer_toolset.TOOLSET_MEMBERS
        for member in expected_members:
            assert configs[member] == {"defer_loading": True}, member

    def test_defer_loading_with_zoom_enabled_reaches_wire_on_every_member(self):
        body = _send(_computer_tool_request(defer_loading=True, enable_zoom=True))
        configs = body["tools"][0]["configs"]

        assert "_defer_loading" not in configs
        assert set(configs.keys()) == _computer_toolset.TOOLSET_MEMBERS
        for member in _computer_toolset.TOOLSET_MEMBERS:
            assert configs[member] == {"defer_loading": True}, member

    def test_no_defer_loading_requested_wire_body_has_only_zoom_override(self):
        body = _send(_computer_tool_request())
        assert body["tools"] == [
            {
                "type": "computer_toolset_20260801",
                "configs": {"zoom": {"enabled": False}},
            }
        ]

    def test_wire_body_carries_defer_loading_alongside_rest_of_request_unaffected(self):
        """Sanity check that the fix is scoped: everything else the real SDK
        puts on the wire around the ``tools`` entry (model, messages,
        max_tokens) is unaffected by the per-member ``defer_loading``
        rewrite -- this is a real request body, not a hand-built stand-in,
        so a regression that corrupted request assembly more broadly (not
        just the ``configs`` shape) would also be caught here."""
        body = _send(_computer_tool_request(defer_loading=True))
        assert body["model"] == "claude-opus-5-5"
        assert body["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]
        configs = body["tools"][0]["configs"]
        assert configs["left_click"] == {"defer_loading": True}
