"""Offline public-entry regressions for ordinary Anthropic image transport."""

import base64
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx2
import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import (
    ChatRequest, ImageBlock, Message, TextBlock, ThinkingBlock,
    ToolCallBlock, ToolResultBlock,
)

from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._image_content import OMITTED, UNAVAILABLE
from tests._helpers import FakeCoordinator


PIXELS = base64.b64encode(b"private-image-sentinel").decode()
URL = "https://example.invalid/private-image.png?token=private-url-sentinel"
SOURCES = [
    {"type": "base64", "media_type": "image/png", "data": PIXELS},
    {"type": "url", "url": URL},
]
MODELS = ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5", "claude-fable-5-2"]


def provider(model=MODELS[0], stream=False, raw=False):
    p = AnthropicProvider(api_key="offline-test-key", config={
        "default_model": model, "use_streaming": stream, "raw": raw,
        "enable_prompt_caching": False, "max_retries": 0,
    })
    p.coordinator = FakeCoordinator()
    return p


def install_sdk(p, model, stream=False, error=None):
    response = anthropic.types.Message(
        id="msg_fixture", model=model, role="assistant",
        content=[{"type": "text", "text": "fixture answer"}],
        stop_reason="end_turn", type="message",
        usage={"input_tokens": 25, "output_tokens": 2},
    )
    raw = MagicMock()
    raw.parse = AsyncMock(return_value=response)
    raw.headers = {}
    create = AsyncMock(return_value=raw, side_effect=error)
    p.client.messages.with_raw_response.create = create
    p.client.messages.count_tokens = AsyncMock(
        return_value=SimpleNamespace(input_tokens=25)
    )
    if stream:
        class Stream:
            response = None

            def __aiter__(self):
                return self.events()

            async def events(self):
                if error:
                    raise error
                yield SimpleNamespace(type="message_start")

            async def get_final_message(self):
                return response

        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=Stream())
        manager.__aexit__ = AsyncMock(return_value=False)
        create = MagicMock(return_value=manager)
        p.client.messages.stream = create
    return create


def history(result):
    return [
        Message(role="user", content="inspect"),
        Message(role="assistant", content=[
            ThinkingBlock(thinking="preserved reasoning", signature="offline-signature"),
            ToolCallBlock(id="call_a", name="lookup", input={}),
            ToolCallBlock(id="call_b", name="lookup", input={}),
        ]),
        result,
        Message(role="tool", tool_call_id="call_b", name="lookup", content="second result"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("source", SOURCES)
async def test_public_user_image_count_send_and_history(model, stream, source):
    p = provider(model, stream)
    sdk = install_sdk(p, model, stream)
    req = ChatRequest(messages=[
        Message(role="user", content=[
            TextBlock(text="before"), ImageBlock(source=source), TextBlock(text="after"),
        ]),
        Message(role="assistant", content="earlier answer"),
        Message(role="user", content=[ImageBlock(source=source)]),
    ], max_output_tokens=200)
    before = req.model_dump()
    budget = await p.request_budget(req, context_estimate=100)
    await p.complete(req)
    counted = p.client.messages.count_tokens.call_args.kwargs
    sent = sdk.call_args.kwargs
    assert budget["measurement"]["input_tokens"] == 25
    assert counted["messages"] == sent["messages"]
    assert sent["messages"][0]["content"] == [
        {"type": "text", "text": "before"}, {"type": "image", "source": source},
        {"type": "text", "text": "after"},
    ]
    assert sent["messages"][-1]["content"] == [{"type": "image", "source": source}]
    assert req.model_dump() == before
    await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["role_tool", "user_result", "content_envelope", "output_envelope", "nested_result"])
@pytest.mark.parametrize("source", SOURCES)
async def test_ordinary_tool_images_public_entry_without_computer_adapter(shape, source):
    p = provider()
    sdk = install_sdk(p, MODELS[0])
    rich = [TextBlock(text="before"), ImageBlock(source=source), TextBlock(text="after")]
    output = rich
    if shape == "content_envelope":
        output = {"content": [b.model_dump() for b in rich]}
    elif shape == "output_envelope":
        output = {"success": False, "output": [b.model_dump() for b in rich]}
    elif shape == "nested_result":
        output = {"type": "tool_result", "tool_call_id": "call_a", "output": rich}
    if shape == "role_tool":
        result = Message(role="tool", name="lookup", tool_call_id="call_a", content=rich, is_error=True)
    else:
        result = Message(role="user", content=[
            ToolResultBlock(tool_call_id="call_a", output=output, is_error=True),
        ])
    # User-role results and following role-tool blocks are intentionally separate
    # canonical messages; wire batching preserves their order.
    req = ChatRequest(messages=history(result), max_output_tokens=200)
    before = req.model_dump()
    await p.request_budget(req, context_estimate=100)
    await p.complete(req)
    sent = sdk.call_args.kwargs["messages"]
    counted = p.client.messages.count_tokens.call_args.kwargs["messages"]
    assert counted == sent
    results = [b for m in sent if isinstance(m["content"], list)
               for b in m["content"] if b["type"] == "tool_result"]
    assert [b["tool_use_id"] for b in results] == ["call_a", "call_b"]
    assert results[0]["content"] == [
        {"type": "text", "text": "before"}, {"type": "image", "source": source},
        {"type": "text", "text": "after"},
    ]
    assert results[0]["is_error"] is True
    assert len([m for m in sent if isinstance(m["content"], list)
                and any(b["type"] == "tool_result" for b in m["content"])]) == 1
    assert sent[1]["content"][0]["signature"] == "offline-signature"
    assert req.model_dump() == before
    await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["assistant", "system", "developer", "user"])
async def test_unavailable_original_slot_does_not_poison_plain_followup(role):
    p = provider()
    sdk = install_sdk(p, MODELS[0])
    req = ChatRequest(messages=[
        Message(role=role, content=[TextBlock(text="before"), ImageBlock(
            source={"type": "file", "file_id": "unqualified-file"}
        ), TextBlock(text="after")]),
        Message(role="user", content="plain followup"),
    ], max_output_tokens=200)
    before = req.model_dump()
    await p.request_budget(req, context_estimate=100)
    await p.complete(req)
    sent = sdk.call_args.kwargs
    projected = {"messages": sent["messages"], "system": sent.get("system")}
    assert UNAVAILABLE in json.dumps(projected)
    assert "unqualified-file" not in json.dumps(projected)
    assert sent["messages"][-1]["content"] == "plain followup"
    assert req.model_dump() == before
    assert p.client.messages.count_tokens.call_args.kwargs["messages"] == sent["messages"]
    await p.close()


@pytest.mark.parametrize("source", [
    {"type": "base64", "media_type": "image/svg+xml", "data": PIXELS},
    {"type": "base64", "media_type": "image/png", "data": "bad***"},
    {"type": "base64", "media_type": "image/png", "data": ""},
    {"type": "url", "url": "file:///private/image.png"},
    {"type": "url", "url": "https://"},
    {"type": "url", "url": "https://bad host/image"},
    {"type": "other", "data": PIXELS},
])
def test_malformed_images_are_explicit_and_payload_free(source):
    p = provider()
    result = p._convert_messages([{"role": "user", "content": [{"type": "image", "source": source}]}])
    assert result == [{"role": "user", "content": [{"type": "text", "text": UNAVAILABLE}]}]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_first_raw_subscriber_omits_images_sdk_keeps_pixels(stream):
    p = provider(stream=stream, raw=True)
    sdk = install_sdk(p, MODELS[0], stream)
    observed = []

    async def first_subscriber(name, payload):
        observed.append(deepcopy(payload))
        serialized = json.dumps(payload)
        assert PIXELS not in serialized and URL not in serialized

    p.coordinator.hooks.emit = first_subscriber
    req = ChatRequest(messages=[Message(role="user", content=[
        ImageBlock(source=s) for s in SOURCES
    ])], max_output_tokens=200)
    await p.complete(req)
    assert OMITTED in json.dumps(observed)
    assert PIXELS in json.dumps(sdk.call_args.kwargs["messages"])
    assert URL in json.dumps(sdk.call_args.kwargs["messages"])
    await p.close()


@pytest.mark.asyncio
async def test_error_echo_body_and_log_omit_pixels_and_urls(caplog):
    p = provider(raw=True)
    error = anthropic.BadRequestError(
        f"echo {PIXELS} {URL}",
        response=httpx2.Response(400, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")),
        body={"error": {"message": f"echo {PIXELS} {URL}", "type": "invalid_request_error"}},
    )
    install_sdk(p, MODELS[0], error=error)
    req = ChatRequest(messages=[Message(role="user", content=[
        ImageBlock(source=s) for s in SOURCES
    ])], max_output_tokens=200)
    with pytest.raises(InvalidRequestError) as caught:
        await p.complete(req)
    observed = json.dumps(p.coordinator.hooks.events) + caplog.text + str(caught.value)
    assert PIXELS not in observed and URL not in observed
    assert "echo" in observed and OMITTED in observed
    await p.close()


def test_result_binding_and_user_order_fail_without_payloads():
    p = provider()
    for blocks in [
        [{"type": "tool_result", "tool_call_id": "missing", "output": []}],
        [{"type": "text", "text": "interposed"}, {"type": "tool_result", "tool_call_id": "call_a", "output": []}],
    ]:
        messages = [m.model_dump() for m in history(Message(role="user", content="placeholder"))[:2]]
        messages.append({"role": "user", "content": blocks})
        with pytest.raises(InvalidRequestError):
            p._convert_messages(messages)


@pytest.mark.parametrize("output", [
    {"type": "tool_result", "tool_call_id": "other", "output": []},
    {"content": [{"type": "tool_result", "tool_call_id": "other", "output": []}]},
    {"success": False, "output": {"type": "tool_result", "tool_call_id": "call_a", "is_error": False, "output": []}},
])
def test_nested_result_cannot_replace_id_or_error_status(output):
    with pytest.raises(InvalidRequestError):
        provider()._ordinary_tool_result({"tool_call_id": "call_a", "content": output})


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_real_sdk_http_wire_has_user_and_tool_images_in_count_and_create(model):
    calls = []
    response = {
        "id": "msg_wire_fixture", "type": "message", "role": "assistant",
        "model": model, "content": [{"type": "text", "text": "wire fixture"}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 30, "output_tokens": 2},
    }

    def transport(request):
        if request.method == "GET":
            return httpx2.Response(200, json={
                "id": model, "type": "model", "display_name": model,
                "created_at": "2026-11-01T00:00:00Z",
            })
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 30})
        assert request.url.path == "/v1/messages"
        return httpx2.Response(200, json=response)

    p = provider(model, raw=True)
    p._client = anthropic.AsyncAnthropic(
        api_key="offline-wire-placeholder", max_retries=0,
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(transport), trust_env=False
        ),
    )
    req = ChatRequest(messages=[
        Message(role="user", content=[ImageBlock(source=SOURCES[1])]),
        Message(role="assistant", content=[
            ToolCallBlock(id="wire_call", name="lookup", input={})
        ]),
        Message(role="user", content=[
            ToolResultBlock(tool_call_id="wire_call", output=[
                TextBlock(text="tool image"), ImageBlock(source=SOURCES[0]),
                ImageBlock(source=SOURCES[1]),
            ], is_error=False)
        ]),
    ], max_output_tokens=200)
    before = req.model_dump()
    assert await p.request_budget(req, context_estimate=100) is not None
    await p.complete(req)
    assert [path for path, _ in calls] == ["/v1/messages/count_tokens", "/v1/messages"]
    counted, sent = [body for _, body in calls]
    assert counted["messages"] == sent["messages"]
    assert sent["messages"][0]["content"][0]["source"] == SOURCES[1]
    result = sent["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "wire_call" and result["is_error"] is False
    assert result["content"][1]["source"] == SOURCES[0]
    assert result["content"][2]["source"] == SOURCES[1]
    assert PIXELS not in json.dumps(p.coordinator.hooks.events)
    assert URL not in json.dumps(p.coordinator.hooks.events)
    assert req.model_dump() == before
    await p.close()


@pytest.mark.parametrize("output", [
    ["alpha", {"count": 2}],
    {"content": ["alpha"], "count": 2},
    {"content": [{"type": "text", "text": "app data"}], "count": 2},
])
def test_non_image_application_json_is_not_mistaken_for_rich_content(output):
    result = provider()._ordinary_tool_result({"tool_call_id": "call_a", "output": output})
    assert json.loads(result["content"]) == output


@pytest.mark.parametrize("block", [
    {"tool_call_id": "call_a", "tool_use_id": "call_b", "output": []},
    {"tool_call_id": "call_a", "output": {"success": False, "is_error": False, "output": []}},
])
def test_conflicting_same_envelope_association_and_status_are_rejected(block):
    with pytest.raises(InvalidRequestError):
        provider()._ordinary_tool_result(block)


def test_cross_message_result_batch_cannot_follow_interposed_text():
    p = provider()
    messages = [m.model_dump() for m in history(Message(role="user", content="placeholder"))[:2]]
    messages.extend([
        {"role": "user", "content": [
            {"type": "tool_result", "tool_call_id": "call_a", "output": "first"},
            {"type": "text", "text": "interposed"},
        ]},
        {"role": "tool", "tool_call_id": "call_b", "content": "second"},
    ])
    with pytest.raises(InvalidRequestError):
        p._convert_messages(messages)


@pytest.mark.asyncio
async def test_bodyless_error_and_cause_omit_signed_image_url_before_credential_redaction(caplog):
    private_url = "https://user:password@example.invalid/private.png?token=private-query"
    p = provider()
    error = ValueError(f"image rejected {private_url}")
    error.__cause__ = RuntimeError(f"image source {private_url}")
    install_sdk(p, MODELS[0], error=error)
    req = ChatRequest(messages=[Message(role="user", content=[
        ImageBlock(source={"type": "url", "url": private_url})
    ])], max_output_tokens=200)
    with pytest.raises(Exception) as caught:
        await p.complete(req)
    observed = json.dumps(p.coordinator.hooks.events) + caplog.text + str(caught.value)
    assert "private.png" not in observed and "private-query" not in observed
    assert "password" not in observed and OMITTED in observed
    await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [0, False, "", None, []])
async def test_business_output_mapping_survives_public_entry_sdk(output):
    p = provider()
    sdk = install_sdk(p, MODELS[0])
    business = {
        "success": False, "output": output,
        "record_id": "r1", "error": "business-field",
    }
    req = ChatRequest(messages=history(Message(role="user", content=[
        ToolResultBlock(tool_call_id="call_a", output=business)
    ])), max_output_tokens=200)
    before = req.model_dump()
    await p.request_budget(req, context_estimate=100)
    await p.complete(req)
    result = sdk.call_args.kwargs["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "call_a"
    assert json.loads(result["content"]) == business
    assert "is_error" not in result  # Application data is not transport status.
    assert req.model_dump() == before
    assert p.client.messages.count_tokens.call_args.kwargs["messages"] == sdk.call_args.kwargs["messages"]
    await p.close()


@pytest.mark.asyncio
async def test_known_core_toolresult_transport_envelope_still_unwraps():
    from amplifier_core import ToolResult

    p = provider()
    sdk = install_sdk(p, MODELS[0])
    envelope = ToolResult(
        success=False, error={"message": "known execution error"},
        output=[{"type": "image", "source": SOURCES[0]}],
    ).model_dump()
    req = ChatRequest(messages=history(Message(role="user", content=[
        ToolResultBlock(tool_call_id="call_a", output=envelope)
    ])), max_output_tokens=200)
    await p.request_budget(req, context_estimate=100)
    await p.complete(req)
    result = sdk.call_args.kwargs["messages"][-1]["content"][0]
    assert result["is_error"] is True and result["tool_use_id"] == "call_a"
    assert result["content"] == [{"type": "image", "source": SOURCES[0]}]
    assert p.client.messages.count_tokens.call_args.kwargs["messages"] == sdk.call_args.kwargs["messages"]
    await p.close()