"""Native computer boundaries through public Core and real SDK transports."""

from __future__ import annotations

import copy
import json
from typing import Any
from unittest.mock import AsyncMock

import httpx2 as httpx
import pytest
from anthropic import AsyncAnthropic
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import ChatRequest, Message, ToolCall, ToolSpec

from amplifier_module_provider_anthropic import AnthropicProvider

MODEL = "claude-sonnet-5-5"
GA = "computer_toolset_20260801"
LEGACY = "computer_20251124"


def function(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="ordinary function",
        parameters={"type": "object", "properties": {}},
    )


def native(name: str = "desktop", tool_type: str = LEGACY, **extra: Any) -> ToolSpec:
    tool = function(name)
    tool.type = tool_type
    tool.display_width_px = 1200
    tool.display_height_px = 800
    for key, value in extra.items():
        setattr(tool, key, value)
    return tool


def request(tools: list[ToolSpec], **extra: Any) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="Inspect the example application.")],
        tools=tools,
        max_output_tokens=128,
        **extra,
    )


def member(name: str = "left_click", input_data: Any = None, **extra: Any) -> dict:
    return {
        "type": "tool_use",
        "id": "call_native",
        "toolset_name": "computer",
        "name": name,
        "input": {} if input_data is None else input_data,
        **extra,
    }


def sdk_message(content: list[dict], model: str = MODEL) -> dict:
    return {
        "id": "msg_example",
        "type": "message",
        "role": "assistant",
        "model": model,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "content": content,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


def sse(content: list[dict], model: str = MODEL) -> bytes:
    initial = sdk_message([], model)
    initial["stop_reason"] = None
    initial["usage"]["output_tokens"] = 0
    events: list[tuple[str, dict]] = [("message_start", {"message": initial})]
    for index, block in enumerate(content):
        events += [
            (
                "content_block_start",
                {"index": index, "content_block": {**block, "input": {}}},
            ),
            (
                "content_block_delta",
                {
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"]),
                    },
                },
            ),
            ("content_block_stop", {"index": index}),
        ]
    events += [
        (
            "message_delta",
            {
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
        ),
        ("message_stop", {}),
    ]
    return "".join(
        f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n"
        for kind, payload in events
    ).encode()


class Wire:
    def __init__(
        self,
        content: list[dict] | None = None,
        model: str = MODEL,
        sdk_default_endpoint: bool = False,
        **config: Any,
    ):
        self.captured: list[tuple[str, dict, dict]] = []
        self.effects: list[Any] = []
        self.content = [member()] if content is None else content
        self.model = model
        self.provider = AnthropicProvider(
            api_key="offline-placeholder",
            config={
                "default_model": model,
                "enable_prompt_caching": False,
                "use_streaming": False,
                "max_retries": 0,
                **config,
            },
        )
        self.provider._runtime_model_info_cache[model] = None
        self.provider._client = AsyncAnthropic(
            api_key="offline-placeholder",
            max_retries=0,
            base_url=None
            if sdk_default_endpoint
            else config.get("base_url", "https://api.anthropic.com"),
            default_headers=self.provider._default_headers,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
        )
        self.hooks = AsyncMock()
        self.provider.coordinator = type("Coordinator", (), {"hooks": self.hooks})()

    async def handle(self, incoming: httpx.Request) -> httpx.Response:
        body = json.loads(incoming.content)
        self.captured.append((incoming.url.path, body, dict(incoming.headers)))
        if incoming.url.path.endswith("/count_tokens"):
            return httpx.Response(200, json={"input_tokens": 3})
        if body.get("stream"):
            return httpx.Response(
                200,
                content=sse(self.content, self.model),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=sdk_message(self.content, self.model))

    async def complete(self, req: ChatRequest, **options: Any):
        response = await self.provider.complete(req, model=self.model, **options)
        self.effects.extend(response.tool_calls or [])
        return response

    async def budget(self, req: ChatRequest):
        return await self.provider.request_budget(
            req,
            context_estimate=20,
            request_options={"model": self.model},
        )

    async def close(self):
        await self.provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "tool_type", [LEGACY, "computer_20250124", "computer_20241022", GA]
)
async def test_native_wire_count_and_dispatch(streaming: bool, tool_type: str):
    configs = {"screenshot": {"enabled": False}, "key": {"enabled": True}}
    declaration = native(
        tool_type=tool_type,
        configs=configs,
        enable_zoom=False,
        cache_control={"type": "ephemeral", "ttl": "1h"},
    )
    original = declaration.model_dump()
    wire = Wire([member(input_data={"coordinate": [31, 47]})], use_streaming=streaming)
    try:
        req = request([declaration])
        assert await wire.budget(req) is not None
        response = await wire.complete(req)
        assert response.tool_calls[0].name == "desktop"
        assert response.tool_calls[0].arguments == {
            "action": "left_click",
            "coordinate": [31, 47],
        }
        count, dispatch = [item[1] for item in wire.captured]
        expected = {
            "type": GA,
            "configs": {**configs, "zoom": {"enabled": False}},
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
        assert dispatch["tools"] == [expected]
        assert dispatch["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }
        for key in ("model", "tools", "messages", "tool_choice"):
            assert count[key] == dispatch[key]
        assert count["model"] == MODEL
        assert "max_tokens" not in count
        assert "computer-use-" not in wire.captured[0][2].get("anthropic-beta", "")
        assert "computer-use-" not in wire.captured[1][2].get("anthropic-beta", "")
        assert declaration.model_dump() == original
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model", ["claude-sonnet-5", "claude-sonnet-4-6", "claude-opus-5-5"]
)
async def test_existing_native_model_wire_controls(model: str):
    wire = Wire([member("screenshot")], model=model)
    try:
        response = await wire.complete(request([native()]))
        body, headers = wire.captured[0][1:]
        if model == "claude-opus-5-5":
            assert body["tools"][0]["type"] == GA
            assert response.tool_calls[0].name == "desktop"
        else:
            assert body["tools"][0]["type"] == LEGACY
            assert "computer-use-2025-11-24" in headers["anthropic-beta"]
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        {"model": "claude-sonnet-5"},
        {"tools": []},
        {"messages": [{"role": "user", "content": "replacement"}]},
        {"tool_choice": {"type": "any"}},
        {"extra_body": {"model": "claude-sonnet-5"}},
        {"extra_body": {"tools": []}},
        {"extra_body": {"messages": []}},
        {
            "extra_body": {
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": False}
            }
        },
    ],
)
async def test_sensitive_native_overrides_fail_before_callbacks_and_transport(
    override: dict,
):
    wire = Wire(extra_request_params=override)
    wire.provider._runtime_model_info_cache.clear()
    try:
        req = request([native()])
        with pytest.raises(InvalidRequestError, match="sensitive overrides"):
            await wire.budget(req)
        with pytest.raises(InvalidRequestError, match="sensitive overrides"):
            await wire.complete(req)
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_override_introducing_native_from_legacy_is_refused(nested: bool):
    payload = {"model": MODEL, "tools": [{"type": GA, "configs": {}}]}
    wire = Wire(
        model="claude-sonnet-5",
        extra_request_params={"extra_body": payload} if nested else payload,
    )
    try:
        with pytest.raises(InvalidRequestError, match="sensitive overrides"):
            await wire.complete(request([function("lookup")]))
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "https://gateway.example.test",
        "http://localhost:8000",
        "https://api.anthropic.com.example.test",
        "http://api.anthropic.com",
        "https://api.anthropic.com:8443",
        "https://api.anthropic.com/proxy",
    ],
)
async def test_native_custom_endpoint_is_refused(base_url: str):
    wire = Wire(base_url=base_url)
    try:
        with pytest.raises(InvalidRequestError, match="first-party"):
            await wire.budget(request([native()]))
        with pytest.raises(InvalidRequestError, match="first-party"):
            await wire.complete(request([native()]))
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-5-5-20261001",
        "claude-sonnet-5-5-latest",
        "claude-sonnet-5-5-",
        "claude-sonnet-5-5preview",
    ],
)
async def test_unverified_sonnet_suffix_is_refused(model: str):
    wire = Wire(model=model)
    try:
        with pytest.raises(InvalidRequestError, match="Unverified Sonnet"):
            await wire.complete(request([native()]))
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tools",
    [
        [native(), native("other")],
        [native(), function("desktop")],
        [native(tool_type="computer_20990101")],
        [native(configs={"unknown": {"enabled": True}})],
        [native(configs={"screenshot": {"enabled": "false"}})],
        [native(configs={"screenshot": {"surplus": True}})],
        [native(configs={"screenshot": {"defer_loading": True}})],
        [native(defer_loading=True)],
        [native(configs={"zoom": {"enabled": False}}, enable_zoom=True)],
        [native(enable_zoom="true")],
    ],
)
async def test_invalid_declarations_fail_before_callbacks_and_transport(tools: list):
    wire = Wire()
    try:
        with pytest.raises(InvalidRequestError):
            await wire.budget(request(tools))
        with pytest.raises(InvalidRequestError):
            await wire.complete(request(tools))
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {"extra_request_params": {"tool_choice": {"type": "tool", "name": "desktop"}}},
        {"extra_request_params": {"extra_body": {"fine_grained_tool_streaming": True}}},
        {"beta_headers": ["fine-grained-tool-streaming-2025-05-14"]},
        {
            "extra_request_params": {
                "extra_headers": {
                    "Anthropic-Beta": "fine-grained-tool-streaming-2025-05-14"
                }
            }
        },
    ],
)
async def test_forcing_and_fine_grained_native_streaming_fail_closed(config: dict):
    wire = Wire(**config)
    try:
        with pytest.raises(InvalidRequestError):
            await wire.budget(request([native()]))
        with pytest.raises(InvalidRequestError):
            await wire.complete(request([native()]))
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choice", ["required", {"type": "any"}, {"type": "tool", "name": "desktop"}]
)
async def test_public_forced_native_choices_fail_closed(choice: Any):
    wire = Wire()
    try:
        with pytest.raises(InvalidRequestError, match="auto/none"):
            await wire.complete(request([native()], tool_choice=choice))
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
async def test_noncomputer_overrides_and_custom_endpoint_remain_supported():
    ordinary = {
        "type": "tool_use",
        "id": "ordinary",
        "name": "computer",
        "input": {"target": "example"},
    }
    wire = Wire(
        [ordinary],
        base_url="https://gateway.example.test",
        extra_request_params={
            "model": "claude-sonnet-5",
            "tools": [{"name": "computer", "input_schema": {"type": "object"}}],
            "extra_body": {"messages": [{"role": "user", "content": "replacement"}]},
        },
    )
    try:
        response = await wire.complete(request([function("lookup")]))
        assert response.tool_calls[0].name == "computer"
        body = wire.captured[0][1]
        assert body["model"] == "claude-sonnet-5"
        assert body["messages"] == [{"role": "user", "content": "replacement"}]
        assert body["tools"][0]["name"] == "computer"
        assert "type" not in body["tools"][0]
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "content",
    [
        [member("screenshot", toolset_name=None)],
        [member("screenshot", toolset_name="other")],
        [member("screenshot")],
        [member("unknown")],
        [member(), member("right_click", id="call_second")],
        [member(input_data={"coordinate": [2, 3], "text": "ctrl"})],
        [member("key", {"text": "Tab", "repeat": 3})],
        [member(input_data={"coordinate": [True, 3]})],
        [member(input_data={"coordinate": [2, 3], "action": "left_click"})],
        [member(input_data={"x": 2, "y": 3})],
        [member("mouse_move", {"coordinate": None})],
        [member("zoom", {"region": None})],
        [member(id="")],
    ],
)
async def test_invalid_native_replies_never_return_dispatchable_calls(
    streaming: bool, content: list
):
    wire = Wire(content, use_streaming=streaming)
    try:
        with pytest.raises(InvalidRequestError):
            await wire.complete(
                request([native(configs={"screenshot": {"enabled": False}})])
            )
        assert len(wire.captured) == 1
        assert wire.effects == []
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_declared_ordinary_screenshot_collision_is_not_native(streaming: bool):
    wire = Wire([member("screenshot", toolset_name=None)], use_streaming=streaming)
    try:
        response = await wire.complete(
            request(
                [
                    native(configs={"screenshot": {"enabled": False}}),
                    function("screenshot"),
                ]
            )
        )
        assert response.tool_calls[0].name == "screenshot"
        assert response.tool_calls[0].arguments == {}
        assert (
            "_anthropic_computer_member_name" not in response.tool_calls[0].model_dump()
        )
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "member_name,input_data",
    [
        ("screenshot", {}),
        ("cursor_position", {}),
        ("left_click", {"coordinate": [2, 3]}),
        ("right_click", {}),
        ("middle_click", {}),
        ("double_click", {}),
        ("triple_click", {}),
        ("mouse_move", {"coordinate": [2, 3]}),
        ("left_mouse_down", {}),
        ("left_mouse_up", {}),
        ("left_click_drag", {"start_coordinate": [2, 3], "coordinate": [4, 5]}),
        (
            "scroll",
            {"coordinate": [2, 3], "scroll_direction": "down", "scroll_amount": 2},
        ),
        ("key", {"text": "Tab"}),
        ("hold_key", {"text": "Tab", "duration": 1}),
        ("type", {"text": "example"}),
        ("wait", {"duration": 1}),
        ("zoom", {"region": [2, 3, 4, 5]}),
    ],
)
async def test_basic_native_actions_preserve_inputs(member_name: str, input_data: dict):
    wire = Wire([member(member_name, input_data)])
    try:
        response = await wire.complete(request([native(enable_zoom=True)]))
        assert response.tool_calls[0].arguments == {"action": member_name, **input_data}
        assert response.content[0].input == {"action": member_name, **input_data}
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("representation", ["content", "tool_calls", "native_wire"])
@pytest.mark.parametrize(
    "result_content,is_error",
    [
        (
            [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AA==",
                    },
                }
            ],
            False,
        ),
        ("Example execution refused.", True),
    ],
)
async def test_public_serialization_and_second_turn_replay(
    representation: str, result_content: Any, is_error: bool
):
    wire = Wire([member("left_click", {"coordinate": [31, 47]})])
    try:
        req = request([native()])
        first = await wire.complete(req)
        persisted = json.loads(first.model_dump_json())
        if representation == "content":
            assistant = Message(role="assistant", content=persisted["content"])
        elif representation == "tool_calls":
            # Core's public ToolCall serializer must preserve both native tags.
            restored = ToolCall(**persisted["tool_calls"][0])
            assistant = Message(role="assistant", content="", tool_calls=[restored])
        else:
            assistant = Message(
                role="assistant",
                content="",
                tool_calls=[member("left_click", {"coordinate": [31, 47]})],
            )
        second = ChatRequest(
            messages=[
                *req.messages,
                Message(**json.loads(assistant.model_dump_json())),
                Message(
                    role="tool",
                    tool_call_id="call_native",
                    content=result_content,
                    is_error=is_error,
                ),
            ],
            tools=req.tools,
            max_output_tokens=128,
        )
        assert await wire.budget(second) is not None
        await wire.complete(second)
        count, dispatch = [item[1] for item in wire.captured[-2:]]
        assert count["messages"] == dispatch["messages"]
        action = dispatch["messages"][1]["content"][0]
        assert action == member("left_click", {"coordinate": [31, 47]})
        result = dispatch["messages"][2]["content"][0]
        assert result["toolset_name"] == "computer"
        assert result["tool_use_id"] == "call_native"
        assert result["is_error"] is is_error
        assert result["content"] == result_content
        assert (
            persisted["tool_calls"][0]["_anthropic_computer_member_name"]
            == "left_click"
        )
        assert persisted["content"][0]["_anthropic_computer_toolset_name"] == "computer"
    finally:
        await wire.close()


@pytest.mark.asyncio
async def test_history_provenance_alias_disagreement_fails_before_transport():
    wire = Wire()
    try:
        req = request([native()])
        first = await wire.complete(req)
        call = copy.deepcopy(first.content[0].model_dump())
        call["name"] = "other_alias"
        second = ChatRequest(
            messages=[
                *req.messages,
                Message(role="assistant", content=[call]),
                Message(role="tool", tool_call_id="call_native", content="done"),
            ],
            tools=req.tools,
        )
        with pytest.raises(InvalidRequestError, match="alias"):
            await wire.budget(second)
        with pytest.raises(InvalidRequestError, match="alias"):
            await wire.complete(second)
        assert len(wire.captured) == 1
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_native_tuple_override_is_refused(nested: bool):
    override = {"tools": ({"type": GA, "configs": {}},), "tool_choice": {"type": "any"}}
    wire = Wire(extra_request_params={"extra_body": override} if nested else override)
    try:
        req = request([function("lookup")])
        with pytest.raises(InvalidRequestError, match="sensitive overrides"):
            await wire.budget(req)
        with pytest.raises(InvalidRequestError, match="sensitive overrides"):
            await wire.complete(req)
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
async def test_sdk_default_fine_grained_beta_survives_empty_request_headers_and_is_refused():
    wire = Wire(
        beta_headers=["fine-grained-tool-streaming-2025-05-14"],
        extra_request_params={"extra_headers": {}},
    )
    try:
        with pytest.raises(InvalidRequestError, match="fine-grained"):
            await wire.budget(request([native()]))
        with pytest.raises(InvalidRequestError, match="fine-grained"):
            await wire.complete(request([native()]))
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
async def test_sdk_default_legacy_betas_are_suppressed_for_count_and_dispatch():
    wire = Wire(
        beta_headers=["computer-use-2025-11-24"],
        extra_request_params={"extra_headers": {}},
    )
    try:
        assert await wire.budget(request([native()])) is not None
        await wire.complete(request([native()]))
        assert len(wire.captured) == 2
        assert all(
            "computer-use-" not in headers.get("anthropic-beta", "")
            for _, _, headers in wire.captured
        )
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_type", ["function", "custom"])
async def test_explicit_function_discriminator_preserves_ordinary_collision(
    tool_type: str,
):
    ordinary = function("screenshot")
    ordinary.type = tool_type
    if tool_type == "custom":
        # SDK's public ToolParam uses type=custom for ordinary function tools.
        ordinary.input_schema = ordinary.parameters
    wire = Wire([member("screenshot", toolset_name=None)])
    try:
        response = await wire.complete(
            request(
                [
                    native(configs={"screenshot": {"enabled": False}}),
                    ordinary,
                ]
            )
        )
        assert response.tool_calls[0].name == "screenshot"
        assert response.tool_calls[0].arguments == {}
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrupt",
    [
        {"_anthropic_computer_toolset_name": "other"},
        {
            "_anthropic_computer_toolset_name": "computer",
            "_anthropic_computer_member_name": "right_click",
        },
        {"type": "text"},
    ],
)
async def test_raw_native_history_cannot_bypass_provenance(corrupt: dict):
    wire = Wire()
    try:
        req = request([native()])
        req.messages += [
            Message(role="assistant", content="", tool_calls=[{**member(), **corrupt}]),
            Message(role="tool", tool_call_id="call_native", content="done"),
        ]
        with pytest.raises(InvalidRequestError):
            await wire.budget(req)
        with pytest.raises(InvalidRequestError):
            await wire.complete(req)
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("representation", ["content", "tool_calls"])
async def test_native_history_cannot_silently_downgrade_without_declaration(
    representation: str,
):
    wire = Wire()
    try:
        first = await wire.complete(request([native()]))
        if representation == "content":
            assistant = Message(role="assistant", content=first.content)
        else:
            assistant = Message(
                role="assistant", content="", tool_calls=first.tool_calls
            )
        req = ChatRequest(
            messages=[
                Message(role="user", content="Continue the example."),
                Message(**json.loads(assistant.model_dump_json())),
                Message(role="tool", tool_call_id="call_native", content="done"),
            ],
            tools=[function("desktop")],
        )
        wire.hooks.reset_mock()
        with pytest.raises(InvalidRequestError, match="matching declaration"):
            await wire.budget(req)
        with pytest.raises(InvalidRequestError, match="matching declaration"):
            await wire.complete(req)
        assert len(wire.captured) == 1
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway", [None, "https://gateway.example.test"])
async def test_sdk_environment_endpoint_resolution_is_authoritative(
    monkeypatch, gateway: str | None
):
    if gateway is None:
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", gateway)
    wire = Wire(sdk_default_endpoint=True)
    try:
        req = request([native()])
        if gateway is None:
            assert await wire.budget(req) is not None
            response = await wire.complete(req)
            assert response.tool_calls[0].name == "desktop"
            assert len(wire.captured) == 2
        else:
            with pytest.raises(InvalidRequestError, match="first-party"):
                await wire.budget(req)
            with pytest.raises(InvalidRequestError, match="first-party"):
                await wire.complete(req)
            assert wire.captured == wire.effects == []
            assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
async def test_ordinary_tuple_override_is_unchanged():
    wire = Wire(
        [member("lookup", toolset_name=None)],
        extra_request_params={
            "tools": ({"name": "lookup", "input_schema": {"type": "object"}},)
        },
    )
    try:
        response = await wire.complete(request([function("lookup")]))
        assert response.tool_calls[0].name == "lookup"
        assert wire.captured[0][1]["tools"] == [
            {"name": "lookup", "input_schema": {"type": "object"}}
        ]
    finally:
        await wire.close()
