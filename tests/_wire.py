"""Real-SDK wire-level test harness for Opus 5.5 behavior.

Uses the REAL ``anthropic.AsyncAnthropic`` client over
``httpx2.MockTransport`` -- no network, but every request is serialized and
parsed by the actual SDK, so tests assert on what the SDK really puts on the
wire (and really hands back), not on a hand-built stand-in. This matters
specifically for preserved-thinking: ``block.model_dump(mode="json",
exclude_unset=True)`` behavior (see ``_preserved_thinking.wire_content_snapshot``)
can only be verified against the real SDK's parsed objects.

``import httpx2 as httpx`` follows the precedent in
``tests/test_cloudflare_real_sdk.py``: anthropic 1.x is built on ``httpx2``,
not ``httpx``, and does not depend on ``httpx`` at all.
"""

from __future__ import annotations

import json
from typing import Any, cast

import anthropic
import httpx2 as httpx
from amplifier_core import ModuleCoordinator

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import FakeCoordinator


class WireRecorder:
    """Real AsyncAnthropic over httpx2.MockTransport. Records every request."""

    def __init__(
        self,
        *,
        messages: list[dict[str, Any] | list[dict[str, Any]]] | None = None,
        model_info: dict[str, Any] | None = None,
        count_tokens: dict[str, Any] | None = None,
        errors: list[tuple[int, dict[str, Any]]] | None = None,
    ) -> None:
        # Each POST /v1/messages consumes one entry from `messages` (a dict
        # -> single JSON response; a list[dict] -> SSE stream of events).
        # Entries in `errors` are consumed FIRST, before `messages`, letting
        # a test script "error then success" (e.g. the drop_block recovery
        # retry) without juggling two queues.
        self._messages_queue: list[dict[str, Any] | list[dict[str, Any]]] = list(
            messages or []
        )
        self._errors_queue: list[tuple[int, dict[str, Any]]] = list(errors or [])
        self._model_info = model_info
        self._count_tokens = count_tokens or {"input_tokens": 10}
        self.requests: list[dict[str, Any]] = []

    async def _handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        try:
            body = json.loads(request.content) if request.content else None
        except (ValueError, UnicodeDecodeError):
            body = None
        self.requests.append(
            {
                "method": request.method,
                "path": path,
                "headers": dict(request.headers),
                "json": body,
            }
        )
        if path == "/v1/messages/count_tokens":
            return httpx.Response(200, json=self._count_tokens, request=request)
        if path.startswith("/v1/models/"):
            if self._model_info is None:
                return httpx.Response(
                    404,
                    json={
                        "type": "error",
                        "error": {"type": "not_found_error", "message": "not found"},
                    },
                    request=request,
                )
            return httpx.Response(200, json=self._model_info, request=request)
        if path == "/v1/messages":
            if self._errors_queue:
                status, error_body = self._errors_queue.pop(0)
                return httpx.Response(status, json=error_body, request=request)
            if not self._messages_queue:
                raise AssertionError("WireRecorder: no queued /v1/messages response left")
            entry = self._messages_queue.pop(0)
            if isinstance(entry, list):
                return _sse_response(entry, request)
            return httpx.Response(200, json=entry, request=request)
        raise AssertionError(f"WireRecorder: unexpected path {path!r}")

    def client(self) -> anthropic.AsyncAnthropic:
        transport = httpx.MockTransport(self._handler)
        return anthropic.AsyncAnthropic(
            api_key="x",
            http_client=httpx.AsyncClient(transport=transport),
            max_retries=0,
        )

    def message_bodies(self) -> list[dict[str, Any]]:
        """``json`` of every ``POST /v1/messages`` request, in order."""
        return [r["json"] for r in self.requests if r["path"] == "/v1/messages"]

    def beta_headers(self, i: int = -1) -> set[str]:
        """Split ``anthropic-beta`` header of the i-th ``/v1/messages`` POST."""
        posts = [r for r in self.requests if r["path"] == "/v1/messages"]
        header = posts[i]["headers"].get("anthropic-beta", "")
        return {h for h in header.split(",") if h}


def _sse_response(events: list[dict[str, Any]], request: httpx.Request) -> httpx.Response:
    lines = []
    for event in events:
        lines.append(f"event: {event['type']}\n")
        lines.append(f"data: {json.dumps(event)}\n\n")
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(lines).encode(),
        request=request,
    )


def make_wire_provider(
    recorder: WireRecorder,
    *,
    model: str = "claude-opus-5-5",
    streaming: bool = False,
    **config: Any,
) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="x",
        config={
            "default_model": model,
            "max_retries": 0,
            "use_streaming": streaming,
            **config,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    provider._client = recorder.client()
    return provider


def opus55_message(
    content: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
    stop_details: dict[str, Any] | None = None,
    input_transformations: list[dict[str, Any]] | None = None,
    model: str = "claude-opus-5-5",
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }
    if stop_details is not None:
        body["stop_details"] = stop_details
    if input_transformations is not None:
        body["input_transformations"] = input_transformations
    return body


def sse_events(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Minimal ordered SSE event sequence reproducing ``message`` (final
    accumulated message equals it once the SDK's stream helper folds the
    deltas back together)."""
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {**message, "content": [], "stop_reason": None},
        }
    ]
    for idx, block in enumerate(message["content"]):
        events.append(
            {"type": "content_block_start", "index": idx, "content_block": _empty_like(block)}
        )
        for key, delta_type, delta_field in (
            ("text", "text_delta", "text"),
            ("thinking", "thinking_delta", "thinking"),
        ):
            if key in block and block.get("type") in ("text", "thinking"):
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": delta_type, delta_field: block[key]},
                    }
                )
        if block.get("type") == "thinking" and block.get("signature"):
            events.append(
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "signature_delta", "signature": block["signature"]},
                }
            )
        if block.get("type") == "tool_use":
            events.append(
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block.get("input", {})),
                    },
                }
            )
        events.append({"type": "content_block_stop", "index": idx})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": message.get("stop_reason", "end_turn")},
            "usage": message.get("usage", {"output_tokens": 0}),
        }
    )
    events.append({"type": "message_stop"})
    return events


def _empty_like(block: dict[str, Any]) -> dict[str, Any]:
    block_type = block.get("type")
    if block_type == "text":
        return {"type": "text", "text": ""}
    if block_type == "thinking":
        return {"type": "thinking", "thinking": ""}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.get("id", ""),
            "name": block.get("name", ""),
            "input": {},
            **({"toolset_name": block["toolset_name"]} if "toolset_name" in block else {}),
        }
    return dict(block)
