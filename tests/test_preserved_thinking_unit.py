"""Unit tests for the pure `_preserved_thinking` module (no provider, no I/O)."""

from __future__ import annotations

import asyncio

from amplifier_module_provider_anthropic import _preserved_thinking as pt
from tests._wire import WireRecorder, opus55_message


def _sdk_content(recorder_body: dict) -> list:
    """Parse a wire-shaped message body through the REAL SDK (over a mock
    transport, exactly as production traffic is parsed) so
    wire_content_snapshot is exercised against real SDK block objects,
    including unknown block types the SDK's response parser tolerates but
    its request-side discriminated union does not accept via direct
    model_validate()."""
    recorder = WireRecorder(messages=[recorder_body])

    async def _fetch():
        client = recorder.client()
        return await client.messages.create(
            model="claude-opus-5-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
        )

    message = asyncio.run(_fetch())
    return list(message.content)


class TestWireContentSnapshot:
    def test_matches_server_json_including_fallback_and_redacted(self):
        body = opus55_message(
            [
                {"type": "thinking", "thinking": "", "signature": "sig1"},
                {"type": "text", "text": "hi"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "left_click",
                    "input": {"coordinate": [1, 2]},
                    "toolset_name": "computer",
                },
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "fallback", "unknown_field": "x"},
            ]
        )
        blocks = _sdk_content(body)
        snapshot = pt.wire_content_snapshot(blocks)
        assert snapshot == body["content"]
        # No None-valued extras (e.g. caller: null) leaked in.
        for entry in snapshot:
            assert None not in entry.values()


class TestSnapshotMatches:
    def _message(self, tool_ids, text_blocks):
        content = [{"type": "text", "text": t} for t in text_blocks]
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [{"id": tid} for tid in tool_ids],
        }

    def test_matching_ids_and_text(self):
        snapshot = [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "a", "name": "f", "input": {}},
        ]
        message = self._message(["a"], ["hello"])
        assert pt.snapshot_matches(snapshot, message) is True

    def test_edited_text_mismatch(self):
        snapshot = [{"type": "text", "text": "hello"}]
        message = self._message([], ["different"])
        assert pt.snapshot_matches(snapshot, message) is False

    def test_different_tool_id_order_mismatch(self):
        snapshot = [
            {"type": "tool_use", "id": "a", "name": "f", "input": {}},
            {"type": "tool_use", "id": "b", "name": "g", "input": {}},
        ]
        message = self._message(["b", "a"], [])
        assert pt.snapshot_matches(snapshot, message) is False

    def test_empty_snapshot_never_matches(self):
        assert pt.snapshot_matches([], self._message([], [])) is False

    def test_plain_string_content_skips_text_check(self):
        snapshot = [{"type": "text", "text": "hello"}]
        message = {"role": "assistant", "content": "hello", "tool_calls": []}
        assert pt.snapshot_matches(snapshot, message) is True


class TestReplayFromContent:
    def test_ordering_and_shape(self):
        content = [
            {"type": "thinking", "thinking": "", "signature": "s1"},
            {"type": "text", "text": "hi"},
            {"type": "tool_call", "id": "t1", "name": "computer", "input": {"action": "left_click", "coordinate": [1, 2]}},
        ]
        out = pt.replay_from_content(content, computer_target_type="computer_toolset_20260801")
        assert out == [
            {"type": "thinking", "thinking": "", "signature": "s1"},
            {"type": "text", "text": "hi"},
            {"type": "tool_use", "id": "t1", "name": "left_click", "input": {"coordinate": [1, 2]}, "toolset_name": "computer"},
        ]

    def test_no_none_values_or_forbidden_keys(self):
        content = [{"type": "thinking", "thinking": "", "signature": None}]
        out = pt.replay_from_content(content)
        assert out == [{"type": "thinking", "thinking": ""}]
        assert "signature" not in out[0]
        assert "visibility" not in out[0]


class TestStripThinkingFromMetadata:
    def test_removes_thinking_and_redacted(self):
        metadata = {
            "anthropic": {
                "wire_content": [
                    {"type": "thinking", "thinking": "x"},
                    {"type": "redacted_thinking", "data": "y"},
                    {"type": "text", "text": "keep"},
                ]
            }
        }
        stripped = pt.strip_thinking_from_metadata(metadata)
        assert stripped["anthropic"]["wire_content"] == [{"type": "text", "text": "keep"}]
        # Original untouched.
        assert len(metadata["anthropic"]["wire_content"]) == 3

    def test_none_and_missing_namespace_passthrough(self):
        assert pt.strip_thinking_from_metadata(None) is None
        assert pt.strip_thinking_from_metadata({}) == {}


class TestIsPrefixBindingMismatch:
    def test_exact_pt_text(self):
        text = (
            "messages.1.content.0: Invalid `signature` in `thinking` block. "
            "The block is bound to a different conversation. Remove the "
            'block, or set `thinking.block_binding.prefix_mismatch_behavior` '
            'to "drop_block".'
        )
        assert pt.is_prefix_binding_mismatch(text) is True

    def test_tampered_signature_not_matched(self):
        assert pt.is_prefix_binding_mismatch("Invalid `signature` in `thinking` block.") is False

    def test_extra_inputs_not_matched(self):
        assert pt.is_prefix_binding_mismatch("block_binding: Extra inputs are not permitted") is False


class TestSummarizeTransformations:
    def test_empty_is_none(self):
        assert pt.summarize_transformations(None) is None
        assert pt.summarize_transformations([]) is None

    def test_counts_reasons_and_paths(self):
        items = [
            {"reason": "prefix_binding_mismatch", "path": "messages.1.content.0"},
            {"reason": "prefix_binding_mismatch", "path": "messages.2.content.0"},
            {"reason": "other"},
        ]
        summary = pt.summarize_transformations(items)
        assert summary["count"] == 3
        assert summary["reasons"] == {"prefix_binding_mismatch": 2, "other": 1}
        assert summary["paths"] == ["messages.1.content.0", "messages.2.content.0"]


class TestPlaceFloatingMessages:
    def _user(self, text="u"):
        return {"role": "user", "content": text}

    def _assistant(self, text="a"):
        return {"role": "assistant", "content": text}

    def _tool_result(self, tool_use_id="t1"):
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": "ok"}]}

    def _floating(self, kind, text="sys"):
        return {"role": "system" if kind == "system" else "user", "content": text, "_floating": kind}

    def test_system_after_user_at_end_stays(self):
        messages = [self._user("u1"), self._floating("system")]
        out = pt.place_floating_messages(messages)
        assert out == [
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "sys"},
        ]

    def test_system_after_assistant_moves_after_next_user(self):
        messages = [
            self._assistant("a1"),
            self._floating("system"),
            self._user("u2"),
        ]
        out = pt.place_floating_messages(messages)
        assert out == [
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "system", "content": "sys"},
        ]

    def test_system_between_tool_use_and_tool_result_moves_after_result(self):
        messages = [
            self._assistant("tool_use_turn"),
            self._floating("system"),
            self._tool_result(),
        ]
        out = pt.place_floating_messages(messages)
        assert out == [
            {"role": "assistant", "content": "tool_use_turn"},
            self._tool_result(),
            {"role": "system", "content": "sys"},
        ]

    def test_developer_before_system_at_same_flush(self):
        messages = [
            self._assistant("a1"),
            self._floating("developer", "dev"),
            self._floating("system", "sys"),
            self._user("u2"),
        ]
        out = pt.place_floating_messages(messages)
        assert out[-2:] == [
            {"role": "user", "content": "dev"},
            {"role": "system", "content": "sys"},
        ]

    def test_appending_messages_does_not_move_earlier_placement(self):
        base = [self._assistant("a1"), self._floating("system"), self._user("u2")]
        out1 = pt.place_floating_messages(base)
        extended = base + [self._assistant("a3")]
        out2 = pt.place_floating_messages(extended)
        assert out2[: len(out1)] == out1
