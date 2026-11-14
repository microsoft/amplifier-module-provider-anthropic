"""Optional exact F1 candidate integration; no runtime Foundation dependency.

Run with an installed media-v1 Foundation candidate and Pillow. A skip is not
accepted-graph qualification. The native offline runner denies real transport.
"""

import base64
import io
import json

import httpx2
import pytest
from anthropic import AsyncAnthropic
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import ChatRequest, Message, TextBlock, ToolCall

media = pytest.importorskip("amplifier_foundation.session.media")
pil = pytest.importorskip("PIL.Image")

from amplifier_foundation.session.media_provider import MediaPreparedProvider
from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import FakeHooks


MODEL = "claude-sonnet-5"


class Coordinator:
    session_id = "media-candidate"

    def __init__(self):
        self.capabilities = {}
        self.hooks = FakeHooks()

    def get_capability(self, name):
        return self.capabilities.get(name)

    def register_capability(self, name, value):
        self.capabilities[name] = value


@pytest.mark.asyncio
@pytest.mark.parametrize("revoke_after_count", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_actual_store_wrapper_count_send_and_epoch_refusal(
    tmp_path, revoke_after_count, streaming,
):
    output = io.BytesIO()
    pil.new("RGB", (3, 2), "red").save(output, format="PNG")
    pixels = output.getvalue()
    encoded = base64.b64encode(pixels).decode()
    coordinator = Coordinator()
    store = media.SessionMediaStore(
        tmp_path / "media", coordinator.session_id,
        policy=media.MediaPolicy(1024 * 1024, 10000, 10),
        checkpoint_verifier=lambda receipt: receipt["refs"],
    )
    origin = {"kind": "tool", "session_id": coordinator.session_id, "id": "actual-call"}
    cap = media.bind_session_media(coordinator, store=store, enabled=True)
    ref = (await cap.ingest(
        pixels, source_format="image-bytes", origin=origin, policy=None,
    ))["value"]
    await cap.publish([ref], origin=origin, disposition="retain")
    store.acknowledge_checkpoint({"refs": [ref]})
    req = ChatRequest(messages=[
        Message(role="user", content="read"),
        Message(role="assistant", content="reading", tool_calls=[
            ToolCall(id="actual-call", name="read_file", arguments={}),
        ]),
        Message(
            role="tool", tool_call_id="actual-call",
            content=[TextBlock(text="before"), TextBlock(text=f"[Session image {ref['sha256']}]"),
                     TextBlock(text="after")],
            metadata={media.MEDIA_NAMESPACE: {"refs": [ref]}, "opaque": {"owner": "keep"}},
        ),
    ], max_output_tokens=32)
    canonical = req.model_dump_json()
    captured = []

    def handle(http_request):
        captured.append((http_request.url.path, json.loads(http_request.content)))
        if http_request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 20})
        response = {
            "id": "msg_candidate", "type": "message", "role": "assistant", "model": MODEL,
            "content": [{"type": "text", "text": "synthetic no-inference response"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 20, "output_tokens": 1},
        }
        if not streaming:
            return httpx2.Response(200, json=response)
        events = [
            {"type": "message_start", "message": {**response, "content": [],
                                                "stop_reason": None}},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "synthetic no-inference response"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn",
                                               "stop_sequence": None},
             "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"},
            content="".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            ).encode(),
        )

    provider = AnthropicProvider(api_key="offline", config={
        "default_model": MODEL, "use_streaming": streaming, "raw": True,
        "max_retries": 0, "enable_prompt_caching": False,
    })
    provider.coordinator = coordinator
    provider._client = AsyncAnthropic(
        api_key="offline", max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
    )
    wrapped = MediaPreparedProvider(
        provider, cap, provider_identity={
            "provider": "anthropic", "account": "offline", "generation": "candidate-1",
        }, model=MODEL, execution_scope={},
    )
    try:
        assert await wrapped.request_budget(req, context_estimate=100, request_options={})
        if revoke_after_count:
            store.epoch += 1
            with pytest.raises((InvalidRequestError, media.MediaRefusal)):
                await wrapped.complete(req)
            assert len(captured) == 1
        else:
            await wrapped.complete(req)
            assert len(captured) == 2
            count, sent = [body for _, body in captured]
            assert count["messages"] == sent["messages"]
            parts = sent["messages"][2]["content"][0]["content"]
            assert [part["type"] for part in parts] == ["text", "image", "text"]
            assert base64.b64decode(parts[1]["source"]["data"], validate=True) == pixels
        assert req.model_dump_json() == canonical
        assert encoded not in canonical
        assert encoded not in json.dumps(coordinator.hooks.events)
    finally:
        wrapped.close()
        await provider.close()
        store.close()