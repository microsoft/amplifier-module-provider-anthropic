"""Neutral v1 through real Core serialization and SDK transport-only fakes.

No executor, vendor account, paid call or platform qualification is implied.
"""

from __future__ import annotations

import asyncio
import copy
import json

import httpx2 as httpx
import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import ChatRequest, Message, ToolCall, ToolSpec

from amplifier_module_provider_anthropic._computer_toolset import (
    PROVENANCE_NEUTRAL,
    ComputerToolsetError,
    translate_tools,
)
from tests.test_sonnet55_computer_hotfix import (
    GA,
    Wire,
    function,
    member,
    native,
    request,
)

MODELS = ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"]
# Independent documented catalogue; no production enumeration imports.
ACTIONS = {
    "screenshot": {},
    "cursor_position": {},
    "left_click": {"coordinate": [31, 47]},
    "right_click": {},
    "middle_click": {},
    "double_click": {},
    "triple_click": {},
    "mouse_move": {"coordinate": [31, 47]},
    "left_mouse_down": {},
    "left_mouse_up": {},
    "left_click_drag": {"start_coordinate": [31, 47], "coordinate": [51, 67]},
    "scroll": {"scroll_direction": "down", "scroll_amount": 2, "coordinate": [31, 47]},
    "key": {"text": "ctrl+s", "repeat": 1},
    "hold_key": {"text": "Tab", "duration": 1},
    "type": {"text": "example"},
    "wait": {"duration": 1},
    "zoom": {"region": [31, 47, 51, 67]},
}
IMAGE = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
}


def neutral(alias="desktop", operations=None, **intent):
    return ToolSpec(
        name=alias,
        description="canonical screenshot-pixel operations",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "coordinate": {"type": "array", "items": {"type": "integer"}},
                "version": {"type": "integer", "const": 1},
                "actions": {"type": "array", "items": {"type": "object"}},
                "capture_after": {"type": "boolean"},
            },
        },
        type="function",
        computer_use={
            "version": 1,
            "operations": list(ACTIONS) if operations is None else operations,
            "image_space": {"width": 1280, "height": 720, "unit": "pixel"},
            "unavailable_operations": {},
            **intent,
        },
        # Internal diagnostics are not vendor parameters.
        diagnostics={"backend_observation": "test-only"},
    )


def batch(action, params):
    return {
        "version": 1,
        "actions": [{"action": action, **params}],
        "capture_after": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("action,params", ACTIONS.items())
async def test_real_sdk_closed_surface_count_dispatch_and_canonical_conversion(
    model, streaming, action, params
):
    tool = neutral(operations=[action])
    wire = Wire([member(action, params)], model=model, use_streaming=streaming)
    wire.provider._runtime_model_info_cache.clear()
    try:
        req = request([tool])
        original = req.model_dump_json()
        assert await wire.budget(req) is not None
        response = await wire.complete(req)
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert call.name == "desktop"
        assert call.arguments == batch(action, params)
        assert response.content[0].input == batch(action, params)
        assert call.model_dump()[PROVENANCE_NEUTRAL] == {
            "version": 1,
            "input": params,
            "image_space": [1280, 720],
        }
        count, dispatch = [entry[1] for entry in wire.captured]
        for key in ("model", "tools", "messages", "tool_choice"):
            assert count[key] == dispatch[key]
        assert count["model"] == model
        assert count["tools"] == [
            {
                "type": GA,
                "configs": {op: {"enabled": op == action} for op in sorted(ACTIONS)},
            }
        ]
        assert dispatch["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }
        assert "max_tokens" not in count
        for path, body, headers in wire.captured:
            assert path in {"/v1/messages", "/v1/messages/count_tokens"}
            assert "computer-use-" not in headers.get("anthropic-beta", "")
            assert "computer_use" not in json.dumps(body)
            assert "diagnostics" not in json.dumps(body)
        assert req.model_dump_json() == original
        if model == "claude-haiku-5-5":
            assert response.usage.cost_usd is None  # Pricing is separate, never zero.
    finally:
        await wire.close()


def loop_replay_request(req, first):
    """Persist the public response exactly as Loop does, with its lossy mirrors."""
    persisted = json.loads(first.model_dump_json())
    return ChatRequest(
        messages=[
            *req.messages,
            Message(
                role="assistant",
                content=persisted["content"],
                tool_calls=[
                    {
                        "id": call["id"],
                        "tool": call["name"],
                        "arguments": call["arguments"],
                    }
                    for call in persisted["tool_calls"]
                ],
            ),
            *[
                Message(role="tool", tool_call_id=call["id"], content="done")
                for call in persisted["tool_calls"]
            ],
        ],
        tools=req.tools,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("copies", ["original", "agreeing", "number-equivalent", "secondary-id"])
async def test_public_loop_mirror_binds_exact_canonical_call(model, streaming, copies):
    ordinary = {"type": "tool_use", "id": "call_lookup", "name": "lookup", "input": {}}
    original_member = member("left_click", {"coordinate": [1, 47]}, id="call_second")
    wire = Wire([ordinary, original_member], model=model, use_streaming=streaming)
    try:
        req = request([neutral(), function("lookup")])
        first = await wire.complete(req)
        second = loop_replay_request(req, first)
        mirror = second.messages[1].tool_calls[1]
        assert set(mirror) == {"id", "tool", "arguments"}
        if copies in {"agreeing", "number-equivalent"}:
            payload = batch("left_click", {"coordinate": [1, 47]})
            if copies == "number-equivalent":
                payload["version"] = 1.0
                payload["actions"][0]["coordinate"] = [1.0, 47.0]
            mirror.update(
                name="desktop", tool_call_id="call_second", call_id="call_second",
                input=copy.deepcopy(payload), arguments=copy.deepcopy(payload),
            )
        elif copies == "secondary-id":
            mirror["call_id"] = mirror.pop("id")
        before = second.model_dump_json()
        first_before = first.model_dump_json()
        assert await wire.budget(second) is not None
        await wire.complete(second)
        assert len(wire.captured) == 3
        count, dispatch = [entry[1] for entry in wire.captured[-2:]]
        assert count["messages"] == dispatch["messages"]
        for body in (count, dispatch):
            assert body["messages"][1]["content"] == [ordinary, original_member]
            results = body["messages"][2]["content"]
            assert [result["tool_use_id"] for result in results] == [
                "call_lookup", "call_second",
            ]
            assert "toolset_name" not in results[0]
            assert results[1]["toolset_name"] == "computer"
        assert second.model_dump_json() == before
        assert first.model_dump_json() == first_before
        assert set(mirror).isdisjoint({
            "_anthropic_computer_member_name",
            "_anthropic_computer_toolset_name",
            "_anthropic_computer_use_v1",
        })
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["computer", "screenshot", "workstation_2"])
async def test_actual_alias_is_not_a_vendor_member_or_hardcoded_dispatch(alias):
    wire = Wire([member("screenshot")])
    try:
        response = await wire.complete(request([neutral(alias)]))
        assert response.tool_calls[0].name == alias
        assert response.tool_calls[0].arguments == batch("screenshot", {})
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("name", ["computer", "screenshot"])
async def test_without_intent_collision_names_remain_ordinary(model, streaming, name):
    wire = Wire(
        [member(name, {"example": "ordinary"}, toolset_name=None)],
        model=model,
        use_streaming=streaming,
    )
    try:
        response = await wire.complete(request([function(name)]))
        assert response.tool_calls[0].name == name
        assert response.tool_calls[0].arguments == {"example": "ordinary"}
        assert PROVENANCE_NEUTRAL not in response.tool_calls[0].model_dump()
        assert wire.captured[0][1]["tools"] == [
            {
                "name": name,
                "description": "ordinary function",
                "input_schema": function(name).parameters,
            }
        ]
    finally:
        await wire.close()


INVALID_INTENTS = [
    None,
    [],
    {"version": True},
    {"version": 2},
    {"version": "1"},
    {"version": 1},
    {"version": 1, "operations": "screenshot"},
    {"version": 1, "operations": [True]},
    {"version": 1, "operations": ["path"]},
    {"version": 1, "operations": ["screenshot", "screenshot"]},
    {"version": 1, "operations": ["screenshot"], "surplus": True},
    {"version": 1, "operations": ["screenshot"], "native_required": "yes"},
    {"version": 1, "operations": ["screenshot"], "unavailable_operations": []},
    {
        "version": 1,
        "operations": ["screenshot"],
        "unavailable_operations": {"screenshot": "unavailable"},
    },
    {
        "version": 1,
        "operations": ["screenshot"],
        "unavailable_operations": {"path": "unavailable"},
    },
    {"version": 1, "operations": ["screenshot"], "image_space": {}},
    {
        "version": 1,
        "operations": ["screenshot"],
        "image_space": {"width": True, "height": 720, "unit": "pixel"},
    },
    {
        "version": 1,
        "operations": ["screenshot"],
        "image_space": {"width": 1280, "height": -1, "unit": "pixel"},
    },
    {
        "version": 1,
        "operations": ["screenshot"],
        "image_space": {"width": 1280, "height": 720, "unit": "point"},
    },
]


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", INVALID_INTENTS)
async def test_malformed_intent_is_not_silently_treated_as_ordinary(intent):
    tool = neutral()
    tool.computer_use = intent
    wire = Wire()
    wire.provider._runtime_model_info_cache.clear()
    try:
        req = request([tool])
        with pytest.raises(InvalidRequestError):
            await wire.budget(req)
        with pytest.raises(InvalidRequestError):
            await wire.complete(req)
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tools",
    [
        [neutral(), neutral("other")],
        [neutral(), function("desktop")],
        [neutral(), native("old")],
        [neutral(), function("computer")],
    ],
)
async def test_ambiguous_declarations_refuse_before_metadata_or_callbacks(tools):
    wire = Wire()
    wire.provider._runtime_model_info_cache.clear()
    try:
        with pytest.raises(InvalidRequestError):
            await wire.budget(request(tools))
        with pytest.raises(InvalidRequestError):
            await wire.complete(request(tools))
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


FALLBACK_CASES = [
    ("claude-sonnet-5", {}, {}),
    ("claude-haiku-4-5", {}, {}),
    ("claude-haiku-5-5-latest", {}, {}),
    ("claude-sonnet-5-5-20261001", {}, {}),
    ("claude-sonnet-5-5", {"base_url": "https://gateway.example.test"}, {}),
    ("claude-sonnet-5-5", {"base_url": "http://api.anthropic.com"}, {}),
    ("claude-sonnet-5-5", {"base_url": "https://api.anthropic.com/proxy"}, {}),
    ("claude-sonnet-5-5", {}, {"image_space": None}),
    (
        "claude-sonnet-5-5",
        {},
        {"image_space": {"width": 3000, "height": 720, "unit": "pixel"}},
    ),
    ("claude-sonnet-5-5", {}, {"operations": []}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model,config,intent", FALLBACK_CASES)
@pytest.mark.parametrize("required", [False, True])
async def test_supported_ordinary_fallback_or_precise_native_required_refusal(
    model, config, intent, required
):
    tool = neutral(native_required=required, **intent)
    wire = Wire(
        [member("desktop", {"action": "screenshot"}, toolset_name=None)],
        model=model,
        **config,
    )
    wire.provider._runtime_model_info_cache.clear()
    try:
        req = request([tool])
        original = req.model_dump_json()
        if required:
            with pytest.raises(InvalidRequestError, match="native-required"):
                await wire.budget(req)
            with pytest.raises(InvalidRequestError, match="native-required"):
                await wire.complete(req)
            assert wire.captured == wire.effects == []
            assert wire.hooks.emit.await_count == 0
        else:
            # Unknown model windows can make budget unavailable without a call.
            await wire.budget(req)
            response = await wire.complete(req)
            assert response.tool_calls[0].arguments == {"action": "screenshot"}
            assert PROVENANCE_NEUTRAL not in response.tool_calls[0].model_dump()
            for _, body, _ in wire.captured:
                assert body["tools"] == [
                    {
                        "name": "desktop",
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    }
                ]
                assert "computer_use" not in json.dumps(body)
                assert "diagnostics" not in json.dumps(body)
                assert not body.get("tool_choice")
        assert original == req.model_dump_json()
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "effective", ["claude-haiku-5-5", "claude-opus-5-5", "claude-sonnet-5"]
)
async def test_model_only_overrides_compile_the_final_effective_model(
    nested, effective
):
    override = {"model": effective}
    content = (
        member("desktop", {"action": "screenshot"}, toolset_name=None)
        if effective == "claude-sonnet-5"
        else member("screenshot")
    )
    wire = Wire(
        [content], extra_request_params={"extra_body": override} if nested else override
    )
    wire.provider._runtime_model_info_cache.clear()
    try:
        req = request([neutral()])
        assert await wire.budget(req) is not None
        response = await wire.complete(req)
        assert all(body["model"] == effective for _, body, _ in wire.captured)
        count, dispatch = [entry[1] for entry in wire.captured]
        assert count["tools"] == dispatch["tools"]
        assert response.tool_calls[0].name == "desktop"
        assert response.tool_calls[0].arguments == (
            {"action": "screenshot"}
            if effective == "claude-sonnet-5"
            else batch("screenshot", {})
        )
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "override",
    [
        {"tools": []},
        {"tools": ({"type": GA},)},
        {"messages": []},
        {"system": "replacement"},
        {"tool_choice": {"type": "auto"}},
    ],
)
async def test_sensitive_shadowing_cannot_change_the_local_binding(nested, override):
    wire = Wire(extra_request_params={"extra_body": override} if nested else override)
    try:
        with pytest.raises(InvalidRequestError, match="sensitive overrides"):
            await wire.budget(request([neutral()]))
        with pytest.raises(InvalidRequestError, match="sensitive overrides"):
            await wire.complete(request([neutral()]))
        assert wire.captured == wire.effects == []
        assert wire.hooks.emit.await_count == 0
    finally:
        await wire.close()


BAD_REPLIES = [
    [member("screenshot", toolset_name=None)],
    [member("desktop", {"action": "screenshot"}, toolset_name=None)],
    [member("screenshot", toolset_name="browser")],
    [member("unknown")],
    [member("zoom", {"region": [0, 0, 4, 5]})],
    [member("left_click"), member("key", {"text": "Tab"}, id="second")],
    [member("key", {"text": "Tab", "repeat": 2})],
    [member("key", {"text": "Tab", "repeat": None})],
    [member("key", {"text": ""})],
    [member("left_click", {"text": "ctrl"})],
    [member("left_click", {"text": ""})],
    [member("left_click", {"coordinate": [True, 5]})],
    [member("left_click", {"coordinate": [1280, 5]})],
    [member("left_click", {"path": [[0, 0], [1, 1]]})],
    [member("left_click", {"action": "left_click"})],
    [member("left_click", pending_safety_checks=[{"id": "check_1"}])],
    [member("left_click", safety_requirements={"approval": True})],
    [member("left_click", id="")],
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("content", BAD_REPLIES)
async def test_bad_native_reply_returns_no_dispatchable_call(model, streaming, content):
    wire = Wire(content, model=model, use_streaming=streaming)
    try:
        with pytest.raises(InvalidRequestError):
            await wire.complete(
                request([neutral(operations=["left_click", "key", "screenshot"])])
            )
        assert len(wire.captured) == 1
        assert wire.effects == []
        responses = [
            call.args[1]
            for call in wire.hooks.emit.await_args_list
            if call.args[0] == "llm:response"
        ]
        assert all(event["status"] == "error" for event in responses)
        assert all(not event.get("tool_calls") for event in responses)
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_declared_ordinary_member_collision_is_a_positive_control(streaming):
    wire = Wire([member("screenshot", toolset_name=None)], use_streaming=streaming)
    try:
        response = await wire.complete(
            request(
                [
                    neutral(operations=["left_click"]),
                    function("screenshot"),
                ]
            )
        )
        assert response.tool_calls[0].name == "screenshot"
        assert response.tool_calls[0].arguments == {}
        assert PROVENANCE_NEUTRAL not in response.tool_calls[0].model_dump()
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("representation", ["content", "tool_calls", "both"])
@pytest.mark.parametrize("structured_result", [False, True])
@pytest.mark.parametrize("is_error", [False, True])
async def test_second_turn_replays_original_vendor_call_and_preserves_result(
    model, streaming, representation, structured_result, is_error
):
    wire = Wire(
        [member("left_click", {"coordinate": [31, 47]})],
        model=model,
        use_streaming=streaming,
    )
    try:
        req = request([neutral()])
        first = await wire.complete(req)
        persisted = json.loads(first.model_dump_json())
        assistant = Message(
            role="assistant",
            content=persisted["content"] if representation != "tool_calls" else "",
            tool_calls=[ToolCall(**persisted["tool_calls"][0])]
            if representation != "content"
            else None,
        )
        result_content = [
            {"type": "text", "text": '{"__amplifier_computer_use__":{"completed":1}}'},
            copy.deepcopy(IMAGE),
        ]
        if structured_result:
            result = Message(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_native",
                        "output": result_content,
                        "is_error": is_error,
                    }
                ],
            )
        else:
            result = Message(
                role="tool",
                tool_call_id="call_native",
                content=result_content,
                is_error=is_error,
            )
        second = ChatRequest(
            messages=[
                *req.messages,
                Message(**json.loads(assistant.model_dump_json())),
                result,
            ],
            tools=req.tools,
            max_output_tokens=128,
        )
        before = second.model_dump_json()
        assert await wire.budget(second) is not None
        await wire.complete(second)
        count, dispatch = [entry[1] for entry in wire.captured[-2:]]
        assert count["messages"] == dispatch["messages"]
        assert dispatch["messages"][1]["content"] == [
            member("left_click", {"coordinate": [31, 47]})
        ]  # Canonical + compatibility mirrors never duplicate execution.
        assert dispatch["messages"][2]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "call_native",
                "toolset_name": "computer",
                "content": result_content,
                "is_error": is_error,
            }
        ]
        assert before == second.model_dump_json()
        assert len(wire.effects) == 2  # One newly returned call per turn, not history.
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["alias", "geometry", "canonical", "provenance", "removed", "downgraded"]
)
async def test_binding_or_provenance_changes_refuse_replay(change):
    wire = Wire([member("left_click", {"coordinate": [31, 47]})])
    try:
        req = request([neutral()])
        first = await wire.complete(req)
        block = first.content[0].model_dump()
        tools = copy.deepcopy(req.tools)
        if change == "alias":
            tools[0].name = "new_alias"
        elif change == "geometry":
            tools[0].computer_use["image_space"]["width"] = 1200
        elif change == "canonical":
            block["input"]["actions"][0]["coordinate"] = [5, 6]
        elif change == "provenance":
            block.pop(PROVENANCE_NEUTRAL)
        elif change == "removed":
            tools = [function("desktop")]
        else:
            tools[0].computer_use["image_space"] = None
        second = ChatRequest(
            messages=[
                *req.messages,
                Message(role="assistant", content=[block]),
                Message(role="tool", tool_call_id="call_native", content="done"),
            ],
            tools=tools,
        )
        with pytest.raises(InvalidRequestError):
            await wire.budget(second)
        with pytest.raises(InvalidRequestError):
            await wire.complete(second)
        assert len(wire.captured) == 1
    finally:
        await wire.close()


@pytest.mark.asyncio
async def test_concurrent_request_local_aliases_do_not_cross_bind():
    wire = Wire([member("screenshot")])
    try:
        responses = await asyncio.gather(
            *[
                wire.complete(request([neutral(alias)]))
                for alias in ["desktop_a", "desktop_b"]
            ]
        )
        assert [r.tool_calls[0].name for r in responses] == ["desktop_a", "desktop_b"]
        assert len(wire.captured) == 2
    finally:
        await wire.close()


@pytest.mark.asyncio
async def test_cancelled_stream_never_returns_executable_calls():
    wire = Wire(use_streaming=True)
    started = asyncio.Event()

    async def waiting_transport(incoming: httpx.Request):
        wire.captured.append((incoming.url.path, json.loads(incoming.content), {}))
        started.set()
        await asyncio.Event().wait()

    wire.provider._client._client._transport = httpx.MockTransport(waiting_transport)
    try:
        task = asyncio.create_task(wire.complete(request([neutral()])))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert wire.effects == []
        assert len(wire.captured) == 1
    finally:
        await wire.close()


def test_neutral_intent_cannot_mix_a_legacy_wire_discriminator():
    tool = neutral()
    tool.type = "computer_20251124"
    with pytest.raises(ComputerToolsetError):
        translate_tools([tool], model=MODELS[0], base_url=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_null_drag_start_never_becomes_executable(model, streaming):
    wire = Wire(
        [member("left_click_drag", {"coordinate": [51, 67], "start_coordinate": None})],
        model=model,
        use_streaming=streaming,
    )
    try:
        with pytest.raises(InvalidRequestError):
            await wire.complete(request([neutral()]))
        assert len(wire.captured) == 1
        assert wire.effects == []
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("defect", ["orphan", "future", "family"])
async def test_results_require_preceding_matching_binding(model, structured, defect):
    wire = Wire([member("screenshot")], model=model)
    try:
        req = request([neutral()])
        first = await wire.complete(req)
        assistant = Message(role="assistant", content=first.content)
        call_id = "unknown" if defect == "orphan" else "call_native"
        extra = {"toolset_name": "wrong_family"} if defect == "family" else {}
        result = (
            Message(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_call_id": call_id,
                        "output": [copy.deepcopy(IMAGE)],
                        **extra,
                    }
                ],
            )
            if structured
            else Message(
                role="tool",
                tool_call_id=call_id,
                content=[copy.deepcopy(IMAGE)],
                **extra,
            )
        )
        history = [result, assistant] if defect == "future" else [assistant, result]
        if defect == "future":
            # Do not let Haiku's independent final-assistant prefill guard
            # satisfy this test instead of the preceding-ID binding guard.
            history.append(Message(role="user", content="Continue"))
        second = ChatRequest(messages=[*req.messages, *history], tools=req.tools)
        before = second.model_dump_json()
        reason = "preceding call binding" if defect == "future" else None
        with pytest.raises(InvalidRequestError, match=reason):
            await wire.budget(second)
        with pytest.raises(InvalidRequestError, match=reason):
            await wire.complete(second)
        assert len(wire.captured) == 1
        assert len(wire.effects) == 1  # First reply only; no replay dispatch.
        assert second.model_dump_json() == before
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("structured", [False, True])
async def test_ordinary_result_cannot_claim_native_family(model, structured):
    ordinary = {"type": "tool_use", "id": "call_lookup", "name": "lookup", "input": {}}
    wire = Wire([ordinary], model=model)
    try:
        req = request([neutral(), function("lookup")])
        first = await wire.complete(req)
        assistant = Message(role="assistant", content=first.content)
        if structured:
            result = Message(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_lookup",
                        "output": "ordinary result",
                        "toolset_name": "computer",
                    }
                ],
            )
        else:
            result = Message(
                role="tool",
                tool_call_id="call_lookup",
                content="ordinary result",
                toolset_name="computer",
            )
        second = ChatRequest(
            messages=[*req.messages, assistant, result], tools=req.tools
        )
        before = second.model_dump_json()
        with pytest.raises(InvalidRequestError, match="Ordinary result cannot claim"):
            await wire.budget(second)
        with pytest.raises(InvalidRequestError, match="Ordinary result cannot claim"):
            await wire.complete(second)
        assert len(wire.captured) == 1
        assert second.model_dump_json() == before
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("structured", [False, True])
async def test_native_and_ordinary_results_coexist_without_rebinding(model, structured):
    ordinary = {"type": "tool_use", "id": "call_lookup", "name": "lookup", "input": {}}
    wire = Wire([member("screenshot"), ordinary], model=model)
    try:
        req = request([neutral(), function("lookup")])
        first = await wire.complete(req)
        assistant = Message(role="assistant", content=first.content)
        if structured:
            results = [
                Message(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_native",
                            "output": [copy.deepcopy(IMAGE)],
                        },
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_lookup",
                            "output": "ordinary result",
                        },
                    ],
                )
            ]
        else:
            results = [
                Message(
                    role="tool",
                    tool_call_id="call_native",
                    content=[copy.deepcopy(IMAGE)],
                ),
                Message(
                    role="tool", tool_call_id="call_lookup", content="ordinary result"
                ),
            ]
        second = ChatRequest(
            messages=[*req.messages, assistant, *results], tools=req.tools
        )
        before = second.model_dump_json()
        await wire.budget(second)
        await wire.complete(second)
        count, dispatch = [entry[1] for entry in wire.captured[-2:]]
        assert count["messages"] == dispatch["messages"]
        replay = dispatch["messages"][2]["content"]
        assert replay[0]["tool_use_id"] == "call_native"
        assert replay[0]["toolset_name"] == "computer"
        assert replay[1]["tool_use_id"] == "call_lookup"
        assert "toolset_name" not in replay[1]
        assert replay[1]["content"] == "ordinary result"
        assert second.model_dump_json() == before
    finally:
        await wire.close()


def replay_request(req, first, representation):
    """Use real Core mirrors, retaining every supplied compatibility extra."""
    persisted = json.loads(first.model_dump_json())
    assistant = Message(
        role="assistant",
        content=persisted["content"] if representation != "tool_calls" else "",
        tool_calls=persisted["tool_calls"] if representation != "content" else None,
    )
    return ChatRequest(
        messages=[
            *req.messages,
            assistant,
            Message(role="tool", tool_call_id="call_native", content="done"),
        ],
        tools=req.tools,
    )


def replay_copy_get(block, field, default=None):
    # Core retains legacy tool_calls dictionaries alongside structured models.
    # Edit the actual representation without coercing away null/malformed extras.
    if isinstance(block, dict):
        return block.get(field, default)
    return getattr(block, field, default)


def replay_copy_set(block, field, value):
    if isinstance(block, dict):
        block[field] = value
    elif field.startswith("_"):
        # Public extras are what Core serializes; setattr creates a shadow
        # attribute for underscore fields without replacing the persisted extra.
        block.model_extra[field] = value
    else:
        setattr(block, field, value)


def replay_copy_delete(block, field):
    if isinstance(block, dict):
        del block[field]
    else:
        delattr(block, field)


def change_replay_copy(block, path, value):
    # Assignment preserves malformed primary fields too; validating construction
    # would test Core's schema rather than the provider's public replay boundary.
    field, *nested = path
    if field == "payload":
        field = "input" if replay_copy_get(block, "type") == "tool_call" else "arguments"
    if not nested:
        replay_copy_set(block, field, copy.deepcopy(value))
        return
    target = replay_copy_get(block, field)
    for key in nested[:-1]:
        target = target[key]
    target[nested[-1]] = copy.deepcopy(value)


REPLAY_DISAGREEMENTS = [
    (("payload", "version"), True),
    (("payload", "actions", 0, "coordinate", 0), True),
    (("payload", "capture_after"), 1),
    (("payload", "version"), "1"),
    (("payload", "actions"), ({"action": "left_click", "coordinate": [1, 47]},)),
    (("payload", "actions", 0, "coordinate", 0), float("nan")),
    (("payload", "actions", 0, "coordinate", 0), float("inf")),
    ((PROVENANCE_NEUTRAL, "image_space", 0), True),
    *[
        ((field,), value)
        for field in ("name", "tool", "id", "tool_call_id", "call_id")
        for value in (None, [], ["malformed"], {"malformed": 1}, 1, "", "conflicting")
    ],
    *[
        ((field,), value)
        for field in ("input", "arguments")
        for value in (None, [], "not a mapping", batch("screenshot", {}))
    ],
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "storage", ["content", "tool_calls", "both-content", "both-tool_calls"]
)
@pytest.mark.parametrize("path,value", REPLAY_DISAGREEMENTS)
async def test_public_replay_rejects_every_supplied_disagreeing_copy(
    storage, path, value
):
    wire = Wire([member("left_click", {"coordinate": [1, 47]})])
    try:
        req = request([neutral()])
        first = await wire.complete(req)
        second = replay_request(req, first, "both" if storage.startswith("both") else storage)
        assistant = second.messages[1]
        block = (
            assistant.content[0]
            if storage in {"content", "both-content"}
            else assistant.tool_calls[0]
        )
        change_replay_copy(block, path, value)
        before = second.model_dump_json()
        first_before = first.model_dump_json()
        for operation in (wire.budget, wire.complete):
            with pytest.raises(InvalidRequestError) as error:
                await operation(second)
            assert error.value.retryable is False
            assert len(wire.captured) == 1  # No count/create/retry HTTP on bad history.
            assert len(wire.effects) == 1  # Only the genuine first reply.
            assert second.model_dump_json() == before
            assert first.model_dump_json() == first_before
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["content", "tool_calls", "both"])
@pytest.mark.parametrize("copies", ["original", "agreeing", "number-equivalent"])
async def test_public_replay_accepts_original_and_json_number_equivalent_copies(
    storage, copies
):
    wire = Wire([member("left_click", {"coordinate": [1, 47]})])
    try:
        req = request([neutral()])
        first = await wire.complete(req)
        second = replay_request(req, first, storage)
        assistant = second.messages[1]
        blocks = (
            (assistant.content if isinstance(assistant.content, list) else [])
            + (assistant.tool_calls or [])
        )
        for block in blocks:
            if copies == "original":
                continue
            canonical = batch("left_click", {"coordinate": [1, 47]})
            if copies == "number-equivalent":
                canonical["version"] = 1.0
                canonical["actions"][0]["coordinate"] = [1.0, 47.0]
            for key, value in (
                ("name", "desktop"), ("tool", "desktop"),
                ("id", "call_native"), ("tool_call_id", "call_native"),
                ("call_id", "call_native"), ("input", canonical),
                ("arguments", canonical),
            ):
                replay_copy_set(block, key, copy.deepcopy(value))
        before = second.model_dump_json()
        assert await wire.budget(second) is not None
        await wire.complete(second)
        assert len(wire.captured) == 3
        for _, body, _ in wire.captured[-2:]:
            assert body["messages"][1]["content"] == [
                member("left_click", {"coordinate": [1, 47]})
            ]
            assert body["messages"][2]["content"][0]["content"] == "done"
        assert second.model_dump_json() == before
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("difference", ["id", "vendor-input", "missing-provenance"])
async def test_public_replay_cross_mirror_consistency_is_not_a_warning(
    diagnostics, difference
):
    wire = Wire([member("left_click", {"coordinate": [1, 47]})], raw=diagnostics)
    try:
        req = request([neutral()])
        first = await wire.complete(req)
        second = replay_request(req, first, "both")
        mirror = second.messages[1].tool_calls[0]
        if difference == "id":
            replay_copy_set(mirror, "id", "different-call")
        elif difference == "vendor-input":
            # Internally agreeing but different from the canonical mirror.
            replay_copy_get(mirror, "arguments")["actions"][0]["coordinate"] = [5, 6]
            replay_copy_get(mirror, PROVENANCE_NEUTRAL)["input"]["coordinate"] = [5, 6]
        else:
            replay_copy_delete(mirror, PROVENANCE_NEUTRAL)
        before = second.model_dump_json()
        for operation in (wire.budget, wire.complete):
            with pytest.raises(InvalidRequestError) as error:
                await operation(second)
            assert error.value.retryable is False
            assert len(wire.captured) == 1
            assert len(wire.effects) == 1
            assert second.model_dump_json() == before
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("declaration", [function("desktop"), native()])
async def test_public_replay_unrelated_ordinary_and_legacy_precedence_is_unchanged(
    declaration
):
    wire = Wire()
    try:
        canonical = {
            "type": "tool_call", "id": "call_native", "name": "desktop",
            "input": {"action": "left_click", "coordinate": [1, 47]},
        }
        stale = ToolCall(
            id="call_native", name="desktop",
            arguments={"action": "screenshot"},
        )
        req = ChatRequest(
            messages=[
                Message(role="user", content="Continue"),
                Message(role="assistant", content=[canonical], tool_calls=[stale]),
                Message(role="tool", tool_call_id="call_native", content="done"),
            ],
            tools=[declaration],
        )
        before = req.model_dump_json()
        assert await wire.budget(req) is not None
        await wire.complete(req)
        for _, body, _ in wire.captured:
            call = body["messages"][1]["content"][0]
            expected = (
                {"coordinate": [1, 47]}
                if getattr(declaration, "type", None)
                else canonical["input"]
            )
            assert call["input"] == expected
        assert req.model_dump_json() == before
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["content", "tool_calls", "both"])
@pytest.mark.parametrize("agreeing", [False, True])
async def test_public_replay_checks_duplicates_before_discarding_them(storage, agreeing):
    wire = Wire([member("left_click", {"coordinate": [1, 47]})])
    try:
        req = request([neutral()])
        first = await wire.complete(req)
        second = replay_request(req, first, storage)
        assistant = second.messages[1]
        population = (
            assistant.content if storage != "tool_calls" else assistant.tool_calls
        )
        duplicate = copy.deepcopy(population[0])
        if not agreeing:
            change_replay_copy(
                duplicate, ("payload", "actions", 0, "coordinate"), [5, 6]
            )
            change_replay_copy(
                duplicate, (PROVENANCE_NEUTRAL, "input", "coordinate"), [5, 6]
            )
        population.append(duplicate)
        before = second.model_dump_json()
        for operation in (wire.budget, wire.complete):
            if agreeing:
                await operation(second)
                assert wire.captured[-1][1]["messages"][1]["content"] == [
                    member("left_click", {"coordinate": [1, 47]})
                ]
            else:
                with pytest.raises(InvalidRequestError) as error:
                    await operation(second)
                assert error.value.retryable is False
                assert len(wire.captured) == 1
            assert second.model_dump_json() == before
    finally:
        await wire.close()


LOOP_MIRROR_DISAGREEMENTS = [
    (path, value) for path, value in REPLAY_DISAGREEMENTS
    if path[0] != PROVENANCE_NEUTRAL
] + [
    ((field,), value)
    for field in (
        "_anthropic_computer_member_name",
        "_anthropic_computer_toolset_name",
        "_anthropic_computer_use_v1",
    )
    for value in (None, {})
] + [
    (("type",), value) for value in ("text", [], {}, True)
] + [
    (("toolset_name",), value) for value in ("wrong", "computer", [], {})
]


@pytest.mark.asyncio
@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("path,value", LOOP_MIRROR_DISAGREEMENTS)
async def test_public_loop_mirror_rejects_every_disagreeing_copy(
    diagnostics, path, value
):
    wire = Wire([member("left_click", {"coordinate": [1, 47]})], raw=diagnostics)
    try:
        req = request([neutral()])
        first = await wire.complete(req)
        second = loop_replay_request(req, first)
        change_replay_copy(second.messages[1].tool_calls[0], path, value)
        before = second.model_dump_json()
        first_before = first.model_dump_json()
        for operation in (wire.budget, wire.complete):
            with pytest.raises(InvalidRequestError) as error:
                await operation(second)
            assert error.value.retryable is False
            assert len(wire.captured) == len(wire.effects) == 1
            assert second.model_dump_json() == before
            assert first.model_dump_json() == first_before
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("primary", ["id", "tool_call_id", "call_id"])
@pytest.mark.parametrize("secondary", ["id", "tool_call_id", "call_id"])
@pytest.mark.parametrize("value", ["unknown", None, [], {}, True])
async def test_public_loop_mirror_checks_known_shadow_ids(primary, secondary, value):
    wire = Wire([member("left_click", {"coordinate": [1, 47]})])
    try:
        req = request([neutral()])
        first = await wire.complete(req)
        second = loop_replay_request(req, first)
        mirror = second.messages[1].tool_calls[0]
        mirror.pop("id")
        mirror[secondary] = "call_native"
        mirror[primary] = copy.deepcopy(value)
        before = second.model_dump_json()
        for operation in (wire.budget, wire.complete):
            with pytest.raises(InvalidRequestError):
                await operation(second)
            assert len(wire.captured) == len(wire.effects) == 1
            assert second.model_dump_json() == before
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect",
    [
        "no-content", "other-assistant", "other-id", "ordinary-id",
        "canonical-partial", "canonical-no-provenance", "canonical-null",
        "canonical-type", "canonical-unhashable-type", "canonical-toolset",
    ],
)
async def test_public_loop_mirror_requires_current_validated_native_canonical(defect):
    ordinary = {"type": "tool_use", "id": "call_lookup", "name": "lookup", "input": {}}
    wire = Wire([ordinary, member("left_click", {"coordinate": [1, 47]})])
    try:
        req = request([neutral(), function("lookup")])
        first = await wire.complete(req)
        second = loop_replay_request(req, first)
        assistant = second.messages[1]
        canonical = assistant.content[1]
        mirror = assistant.tool_calls[1]
        if defect == "no-content":
            assistant.content = ""
        elif defect == "other-assistant":
            assistant.tool_calls = None
            second.messages.insert(
                2, Message(role="assistant", content="", tool_calls=[mirror])
            )
        elif defect == "other-id":
            mirror["id"] = "different-call"
        elif defect == "ordinary-id":
            mirror["id"] = "call_lookup"
        elif defect == "canonical-partial":
            replay_copy_delete(canonical, PROVENANCE_NEUTRAL)
        elif defect == "canonical-no-provenance":
            for key in (
                "_anthropic_computer_member_name",
                "_anthropic_computer_toolset_name",
                "_anthropic_computer_use_v1",
            ):
                replay_copy_delete(canonical, key)
        elif defect == "canonical-null":
            replay_copy_set(canonical, PROVENANCE_NEUTRAL, None)
            serialized = json.loads(second.model_dump_json())["messages"][1]["content"][1]
            assert PROVENANCE_NEUTRAL in serialized
            assert serialized[PROVENANCE_NEUTRAL] is None
        elif defect == "canonical-type":
            replay_copy_set(canonical, "type", "text")
        elif defect == "canonical-unhashable-type":
            replay_copy_set(canonical, "type", [])
        else:
            replay_copy_set(canonical, "toolset_name", "wrong")
        before = second.model_dump_json()
        for operation in (wire.budget, wire.complete):
            with pytest.raises(InvalidRequestError):
                await operation(second)
            assert len(wire.captured) == 1
            assert len(wire.effects) == 2
            assert second.model_dump_json() == before
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("known_secondary", [False, True])
@pytest.mark.parametrize("value", [None, [], {}, "unknown"])
async def test_public_haiku_loop_mirror_scanner_defers_bad_ids(known_secondary, value):
    wire = Wire(
        [member("left_click", {"coordinate": [1, 47]})],
        model="claude-haiku-5-5",
    )
    try:
        req = request([neutral()])
        first = await wire.complete(req)
        second = loop_replay_request(req, first)
        mirror = second.messages[1].tool_calls[0]
        mirror["id"] = copy.deepcopy(value)
        if known_secondary:
            mirror["call_id"] = "call_native"
        before = second.model_dump_json()
        first_before = first.model_dump_json()
        for operation in (wire.budget, wire.complete):
            with pytest.raises(InvalidRequestError) as error:
                await operation(second)
            assert error.value.retryable is False
            assert len(wire.captured) == len(wire.effects) == 1
            assert second.model_dump_json() == before
            assert first.model_dump_json() == first_before
    finally:
        await wire.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["canonical", "compatibility"])
async def test_public_haiku_scanner_preserves_ordinary_missing_result_repair(storage):
    ordinary = {"type": "tool_use", "id": "call_lookup", "name": "lookup", "input": {}}
    wire = Wire([ordinary], model="claude-haiku-5-5")
    try:
        req = request([neutral(), function("lookup")])
        first = await wire.complete(req)
        second = loop_replay_request(req, first)
        second.messages.pop()  # The ordinary result is genuinely missing.
        second.messages.append(Message(role="user", content="Continue"))
        if storage == "compatibility":
            second.messages[1].content = ""
        before = second.model_dump_json()
        assert await wire.budget(second) is not None
        await wire.complete(second)
        assert len(wire.captured) == 3
        count, dispatch = [entry[1] for entry in wire.captured[-2:]]
        assert count["messages"] == dispatch["messages"]
        for body in (count, dispatch):
            assert body["messages"][1]["content"] == [ordinary]
            result = body["messages"][2]["content"][0]
            assert result["tool_use_id"] == "call_lookup"
            assert "[SYSTEM ERROR: Tool result missing" in result["content"]
            assert "toolset_name" not in result
        assert second.model_dump_json() == before
    finally:
        await wire.close()
