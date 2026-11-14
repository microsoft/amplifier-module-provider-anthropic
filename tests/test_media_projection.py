"""Offline media-v1 vendor boundary: real SDK serialization, never inference."""

import base64
import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from anthropic import AsyncAnthropic
from amplifier_core.message_models import (
    ChatRequest, ImageBlock, Message, TextBlock, ToolCall,
)
from amplifier_core.llm_errors import InvalidRequestError, ProviderUnavailableError

from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._media import (
    NOT_RETAINED, diagnostic_copy, safe_error_text,
)
from tests._helpers import FakeCoordinator


MODEL = "claude-sonnet-5"
PIXELS = (
    b"\x89PNG\r\n\x1a\nmedia-v1-original-encoded-fixture",
    b"\xff\xd8media-v1-second-original-encoded-fixture\xff\xd9",
)
ENCODED = [base64.b64encode(value).decode() for value in PIXELS]


def image(index=0):
    return ImageBlock(source={
        "type": "base64",
        "media_type": "image/png" if index == 0 else "image/jpeg",
        "data": ENCODED[index],
    })


def request():
    return ChatRequest(messages=[
        Message(role="user", content=[TextBlock(text="original user"), image(1)]),
        Message(
            role="assistant",
            content="Reading two images.",
            tool_calls=[
                ToolCall(id="call_a", name="read_file", arguments={"path": "first.png"}),
                ToolCall(id="call_b", name="read_file", arguments={"path": "second.jpg"}),
            ],
            metadata={"owner": {"opaque": "unchanged"}},
        ),
        Message(role="tool", tool_call_id="call_a",
                content=[TextBlock(text="before"), image(0), image(1),
                         TextBlock(text="after")]),
        Message(role="tool", tool_call_id="call_b", content="no image"),
        Message(role="user", content="Describe the original observations."),
    ], max_output_tokens=32)


def reply():
    return {
        "id": "msg_media",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "offline response, not image understanding"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 5},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [False, True])
async def test_sdk_count_send_original_user_and_tool_positions_and_privacy(raw):
    captured = []

    def handle(http_request):
        captured.append((http_request.url.path, json.loads(http_request.content)))
        if http_request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 20})
        return httpx2.Response(200, json=reply())

    provider = AnthropicProvider(api_key="offline", config={
        "default_model": MODEL, "use_streaming": False, "max_retries": 0,
        "raw": raw, "throttle_threshold": 0, "enable_prompt_caching": False,
    })
    coordinator = FakeCoordinator()
    provider.coordinator = coordinator
    provider._client = AsyncAnthropic(
        api_key="offline", max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
    )
    req = request()
    before = req.model_dump_json()
    try:
        assert await provider.request_budget(req, context_estimate=100, request_options={})
        await provider.complete(req)
    finally:
        await provider.close()
    assert req.model_dump_json() == before
    assert len(captured) == 2
    count, sent = [body for _, body in captured]
    assert count["messages"] == sent["messages"]
    assert count["model"] == sent["model"] == MODEL
    messages = sent["messages"]
    assert len(messages) == 4  # Tool results are batched, never image-relocated.
    assert messages[0]["content"][0]["text"] == "original user"
    results = messages[2]["content"]
    assert [part["tool_use_id"] for part in results] == ["call_a", "call_b"]
    parts = results[0]["content"]
    assert [part["type"] for part in parts] == ["text", "image", "image", "text"]
    assert parts[0]["text"] == "before"
    assert parts[-1]["text"] == "after"
    for part, original in zip(parts[1:3], PIXELS, strict=True):
        decoded = base64.b64decode(part["source"]["data"], validate=True)
        assert hashlib.sha256(decoded).digest() == hashlib.sha256(original).digest()
    diagnostic = json.dumps(coordinator.hooks.events)
    assert all(value not in diagnostic for value in ENCODED)
    assert (NOT_RETAINED in diagnostic) == raw


def test_diagnostic_copy_known_locations_only_preserves_opaque_owners():
    payload = {
        "messages": [{"role": "user", "content": [
            image().model_dump(),
            {"type": "tool_result", "tool_use_id": "a",
             "content": [image(1).model_dump()]},
            {"type": "thinking", "thinking": "opaque", "signature": "signature"},
            {"type": "tool_use", "input": {"source": {"data": "not an image"}}},
        ]}],
        "extra_body": {"messages": [{"role": "user", "content": [image().model_dump()]}]},
        "metadata": {"$blob_ref": "not SessionBlobRefV1"},
        "text": '{"type":"image","source":{"data":"literal human text"}}',
    }
    original = copy.deepcopy(payload)
    sanitized = diagnostic_copy(payload)
    assert payload == original
    assert sanitized["metadata"] == original["metadata"]
    assert sanitized["text"] == original["text"]
    assert sanitized["messages"][0]["content"][2:] == original["messages"][0]["content"][2:]
    assert all(value not in json.dumps(sanitized) for value in ENCODED)
    assert safe_error_text("echo: " + ENCODED[0], payload) == "echo: " + NOT_RETAINED


def test_orphan_warning_never_serializes_pixels(caplog):
    provider = AnthropicProvider(api_key="offline")
    provider._convert_messages([
        {"role": "tool", "tool_call_id": "orphan", "content": [image().model_dump()]},
    ])
    assert ENCODED[0] not in caplog.text
    assert "content_preview" not in caplog.text


class Witness:
    version = 1

    def __init__(self, req, options=None):
        self.request = req
        self.options = options or {}
        self.calls = []
        self.revoked = False
        self.snapshot = req.model_dump_json()

    def validate(self, value, *, purpose, request_options=None):
        self.calls.append(purpose)
        if (
            self.revoked or value is not self.request
            or request_options != self.options
            or value.model_dump_json() != self.snapshot
        ):
            raise ValueError("sensitive local cause " + ENCODED[0])
        return {"valid": True}


def guarded_provider(req, options=None, **config):
    provider = AnthropicProvider(api_key="offline", config={
        "default_model": MODEL, "use_streaming": False, "max_retries": 2,
        "retry_min_delay": 0, "retry_max_delay": 0,
        "throttle_threshold": 0, "enable_prompt_caching": False, **config,
    })
    coordinator = FakeCoordinator()
    witness = Witness(req, options)
    coordinator.get_capability = lambda name: witness
    provider.coordinator = coordinator
    return provider, witness


def test_support_facts_are_exact_not_name_prefix_guesses():
    provider = AnthropicProvider(api_key="offline")
    assert provider.get_media_request_support(model=MODEL)["positions"] == ["user", "tool"]
    for model in ("claude-sonnet-999", "not-claude-sonnet-5", "claude-opus-latest"):
        assert provider.get_media_request_support(model=model)["positions"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_revocation_after_count_refuses_before_first_transport(streaming):
    req = request()
    provider, witness = guarded_provider(req, use_streaming=streaming)
    provider._client = MagicMock()
    provider._client.messages.count_tokens = AsyncMock(
        return_value=SimpleNamespace(input_tokens=20)
    )
    assert await provider.request_budget(req, context_estimate=100)
    witness.revoked = True
    with pytest.raises(InvalidRequestError, match="request not sent") as raised:
        await provider.complete(req)
    assert not raised.value.retryable
    assert ENCODED[0] not in str(raised.value)
    assert witness.calls == ["count", "transport"]
    provider._client.messages.stream.assert_not_called()
    provider._client.messages.with_raw_response.create.assert_not_called()


@pytest.mark.asyncio
async def test_revocation_after_backoff_blocks_second_attempt(monkeypatch):
    req = request()
    provider, witness = guarded_provider(req)
    provider._client = MagicMock()
    provider._client.messages.with_raw_response.create = AsyncMock(
        side_effect=ProviderUnavailableError("offline 503", status_code=503, retryable=True)
    )

    async def retry(operation, config, on_retry=None):
        with pytest.raises(ProviderUnavailableError):
            await operation()
        witness.revoked = True  # stands for expiry/revocation while waiting
        return await operation()

    monkeypatch.setattr("amplifier_module_provider_anthropic.retry_with_backoff", retry)
    with pytest.raises(InvalidRequestError, match="request not sent"):
        await provider.complete(req)
    assert witness.calls == ["transport", "transport"]
    assert provider._client.messages.with_raw_response.create.await_count == 1


@pytest.mark.asyncio
async def test_count_refusal_is_not_count_unavailable():
    req = request()
    provider, witness = guarded_provider(req)
    provider._client = MagicMock()
    witness.revoked = True
    with pytest.raises(InvalidRequestError, match="request not sent"):
        await provider.request_budget(req, context_estimate=100)
    provider._client.messages.count_tokens.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_model_unavailable_before_count_and_same_send():
    captured = []

    def handle(http_request):
        captured.append(json.loads(http_request.content))
        return httpx2.Response(200, json=(
            {"input_tokens": 20} if http_request.url.path.endswith("/count_tokens") else reply()
        ))

    # A known window is not vision evidence for this unknown exact model.
    unknown = "custom-model"
    provider = AnthropicProvider(api_key="offline", config={
        "default_model": unknown, "use_streaming": False, "max_retries": 0,
        "enable_prompt_caching": False,
    })
    provider._client = AsyncAnthropic(
        api_key="offline", max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
    )
    req = request()
    try:
        assembly = provider._assemble_request_params(
            req, request_options={}, request_caps=provider._get_capabilities(unknown)
        )
        assert assembly is not None
        diagnostic = json.dumps(assembly.params)
        assert all(value not in diagnostic for value in ENCODED)
        assert diagnostic.count("no pixels supplied") == 3
        await provider.complete(req)
        assert captured[0]["messages"] == assembly.params["messages"]
    finally:
        await provider.close()


def test_document_content_images_omitted_not_pdf_or_opaque_strings():
    payload = {"messages": [{"role": "user", "content": [{
        "type": "document", "source": {
            "type": "content", "content": [image().model_dump()],
        },
    }]}]}
    assert ENCODED[0] not in json.dumps(diagnostic_copy(payload))
    assert safe_error_text(ENCODED[0], payload) == NOT_RETAINED


@pytest.mark.asyncio
async def test_wire_override_after_count_requires_recount():
    req = request()
    provider, witness = guarded_provider(req)
    provider._client = MagicMock()
    provider._client.messages.count_tokens = AsyncMock(
        return_value=SimpleNamespace(input_tokens=20)
    )
    assert await provider.request_budget(req, context_estimate=100)
    provider.extra_request_params["tool_choice"] = {"type": "none"}
    with pytest.raises(InvalidRequestError, match="changed after count"):
        await provider.complete(req)
    provider._client.messages.with_raw_response.create.assert_not_called()


@pytest.mark.asyncio
async def test_no_replay_after_first_sdk_stream_event():
    req = request()
    provider, witness = guarded_provider(req, use_streaming=True)
    provider._client = MagicMock()

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def __aiter__(self):
            async def events():
                yield SimpleNamespace(type="message_start")
                raise ProviderUnavailableError("offline interruption", retryable=True)
            return events()

    provider._client.messages.stream.return_value = Stream()
    with pytest.raises(ProviderUnavailableError) as raised:
        await provider.complete(req)
    assert not raised.value.retryable
    assert provider._client.messages.stream.call_count == 1
    assert witness.calls == ["transport"]


@pytest.mark.asyncio
async def test_ref_bearing_request_without_capability_refuses_before_count():
    req = request()
    req.messages[2].metadata = {"experimental.media.v1": {"refs": []}}
    provider = AnthropicProvider(api_key="offline", config={"default_model": MODEL})
    provider._client = MagicMock()
    with pytest.raises(InvalidRequestError, match="compatible private preparation"):
        await provider.request_budget(req, context_estimate=100)
    provider._client.messages.count_tokens.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [asyncio.TimeoutError(), OSError("response lost")])
async def test_unknown_transport_outcome_is_not_replayed(failure):
    req = request()
    provider, witness = guarded_provider(req)
    provider._client = MagicMock()
    provider._client.messages.with_raw_response.create = AsyncMock(side_effect=failure)
    from amplifier_core.llm_errors import LLMError
    with pytest.raises(LLMError) as raised:
        await provider.complete(req)
    assert not raised.value.retryable
    assert provider._client.messages.with_raw_response.create.await_count == 1
    assert witness.calls == ["transport"]


@pytest.mark.asyncio
async def test_prepared_initial_text_native_declaration_keeps_exact_identity():
    from tests.test_neutral_computer_use import neutral
    req = ChatRequest(messages=[Message(role="user", content="start")], tools=[neutral()])
    provider, witness = guarded_provider(req, default_model="claude-sonnet-5-5")
    provider._client = MagicMock()
    response = SimpleNamespace(
        model="claude-sonnet-5-5", content=[], stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    raw = MagicMock()
    raw.parse = AsyncMock(return_value=response)
    raw.headers = {}
    provider._client.messages.with_raw_response.create = AsyncMock(return_value=raw)
    await provider.complete(req)
    assert witness.calls == ["transport"]
    provider._client.messages.with_raw_response.create.assert_awaited_once()