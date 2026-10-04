"""Transport-only adaptation of persisted unsigned Core reasoning; no network."""

import asyncio
import json
from copy import deepcopy

import httpx2
import pytest
from anthropic import AsyncAnthropic
from amplifier_core.message_models import ChatRequest, Message, TextBlock, ThinkingBlock

from amplifier_module_provider_anthropic import AnthropicProvider


MODEL = "claude-fable-5-1"


def provider():
    return AnthropicProvider(api_key="offline-test-placeholder", config={
        "default_model": MODEL, "enable_prompt_caching": False,
        "max_retries": 0, "use_streaming": False,
    })


@pytest.mark.parametrize("shape", ["structured", "legacy", "both"])
@pytest.mark.parametrize("signature", [None, "missing"])
def test_unsigned_thinking_is_wire_only_omission_with_tools_preserved(shape, signature):
    thinking = {"type": "thinking", "thinking": "internal-only", "content": ["opaque-openai-state", "rs_test"]}
    if signature is None:
        thinking["signature"] = None
    history = [{"role": "assistant", "content": "Visible answer", "metadata": {"openai:response_id": "resp_test"},
                "tool_calls": [{"id": "tool_1", "name": "lookup", "arguments": {}}]}]
    if shape != "legacy":
        history[0]["content"] = [thinking, {"type": "text", "text": "Visible answer"}]
    if shape != "structured":
        history[0]["thinking_block"] = thinking
    history += [{"role": "tool", "tool_call_id": "tool_1", "content": "result"}, {"role": "user", "content": "Next"}]
    before = deepcopy(history)
    wire = provider()._convert_messages(history)
    assert wire[0]["content"] == [{"type": "text", "text": "Visible answer"}, {"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {}}]
    assert wire[1]["content"] == [{"type": "tool_result", "tool_use_id": "tool_1", "content": "result"}]
    assert "internal-only" not in json.dumps(wire)
    assert "opaque-openai-state" not in json.dumps(wire)
    assert history == before


@pytest.mark.parametrize("shape", ["structured", "legacy"])
def test_unsigned_reasoning_only_turn_does_not_emit_empty_assistant(shape):
    thinking = {"type": "thinking", "thinking": "internal", "signature": None}
    row = {"role": "assistant", "content": [thinking]} if shape == "structured" else {"role": "assistant", "content": "", "thinking_block": thinking}
    before = deepcopy(row)
    assert provider()._convert_messages([row, {"role": "user", "content": "Continue"}]) == [{"role": "user", "content": "Continue"}]
    assert row == before


def test_signed_and_redacted_reasoning_stays_exact_and_in_order():
    blocks = [
        {"type": "thinking", "thinking": "unsigned", "signature": None},
        {"type": "thinking", "thinking": "signed", "signature": "opaque-signature"},
        {"type": "text", "text": "Visible"},
        {"type": "redacted_thinking", "data": "opaque-redacted-state"},
        {"type": "thinking", "thinking": "empty-string-contract", "signature": ""},
    ]
    original = deepcopy(blocks)
    assert provider()._convert_messages([{"role": "assistant", "content": blocks}])[0]["content"] == blocks[1:]
    assert blocks == original


def test_actual_core_persisted_openai_history_to_real_sdk_payload(tmp_path):
    """The live failure shape is rejected by the in-process API validator.

    The real SDK serializes the public complete() request. No account, network,
    fallback, or second completion is used to make an invalid request pass.
    """
    original = Message(role="assistant", content=[
        ThinkingBlock(thinking="private-reasoning", content=["encrypted-state", "rs_test"]),
        TextBlock(text="Baseline answer"),
    ], metadata={"openai:response_id": "resp_test"})
    path = tmp_path / "transcript.jsonl"
    path.write_text(original.model_dump_json() + "\n")
    before = path.read_bytes()
    history = Message.model_validate_json(before)
    request = ChatRequest(model=MODEL, reasoning_effort="xhigh", max_output_tokens=128,
                          messages=[Message(role="user", content="Baseline"), history,
                                    Message(role="user", content="Quoted baseline\n\nFollow up", metadata={"replyTo": {"messageId": "fixture"}})])
    before_request = request.model_dump()
    calls = []

    async def handler(http_request):
        body = json.loads(http_request.content)
        calls.append(body)
        for message in body.get("messages", []):
            if not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if block.get("type") == "thinking" and not isinstance(block.get("signature"), str):
                    return httpx2.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "thinking.signature.str: Input should be a valid string"}})
        return httpx2.Response(200, json={"id": "msg_test", "type": "message", "role": "assistant", "model": MODEL,
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1}})

    async def run():
        instance = provider()
        instance._runtime_model_info_cache[MODEL] = None
        instance._client = AsyncAnthropic(api_key="offline-test-placeholder", max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
        try:
            return await instance.complete(request)
        finally:
            await instance.close()

    response = asyncio.run(run())
    assert response.content[0].text == "ok"
    assert len(calls) == 1
    assert calls[0]["model"] == MODEL
    # Fable's existing always-on thinking policy omits this parameter.
    assert "thinking" not in calls[0]
    assert calls[0]["messages"][1]["content"] == [{"type": "text", "text": "Baseline answer"}]
    assert "private-reasoning" not in json.dumps(calls)
    assert "encrypted-state" not in json.dumps(calls)
    assert request.model_dump() == before_request
    assert path.read_bytes() == before
