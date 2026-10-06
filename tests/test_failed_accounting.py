"""Failed usage/cancellation at the real SDK loopback receiving boundary."""

import asyncio
import contextvars
import json
import time
from types import SimpleNamespace

import anthropic
import pytest
from amplifier_core import HookResult, ModuleCoordinator
from amplifier_module_provider_anthropic._failed_usage import FailedUsage
from amplifier_module_provider_anthropic._request_safety import RequestOutcomeUnknownError
from tests import test_transport_liveness as live


FULL = {
    "input_tokens": 125, "output_tokens": 3,
    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    "speed": "standard", "service_tier": "standard",
}


class SettlementWire(live.Wire):
    def __init__(self, streaming, usage, final, partial):
        super().__init__("settlement", streaming)
        self.usage = usage
        self.final = final
        self.partial = partial

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next((int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                           if line.lower().startswith(b"content-length:")), 0)
            body = json.loads(await reader.readexactly(length) or b"{}")
            path = headers.split(b" ")[1]
            if path.startswith(b"/v1/models"):
                await self.json_reply(writer, 200, {"data": [], "has_more": False})
                return
            assert path == b"/v1/messages"
            self.posts += 1
            self.accepted += 1
            self.entered.set()
            response = live.message()
            response.update(model=body["model"], usage=self.usage)
            if not self.streaming:
                await self.release.wait()
                # An actual parsed response with bad content is still not a
                # successful ChatResponse; its consumed usage can be known.
                response["content"] = None
                await self.json_reply(writer, 200, response)
                return
            response["stop_reason"] = None
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                         b"connection: close\r\n\r\n")
            writer.write(live.sse({"type": "message_start", "message": response}))
            if self.partial:
                writer.write(live.sse({
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }))
                writer.write(live.sse({
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": live.PRIVATE},
                }))
            if self.final:
                writer.write(live.sse({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": self.usage,
                }))
            await writer.drain()
            self.sent.set()
            await self.release.wait()
            # No message_stop. get_final_message's partial snapshot must never
            # be accepted as a result, even if final usage arrived.
            writer.transport.abort()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self.tasks.discard(task)


def fixture_wire(monkeypatch, usage, *, final=False, partial=True):
    monkeypatch.setattr(live, "Wire", lambda mode, streaming:
                        SettlementWire(streaming, usage, final, partial))


def responses(hooks):
    return [p for name, p in hooks.events if name == "llm:response"]


def assert_settlement(hooks, partial, status):
    starts = [p for name, p in hooks.events if name == "llm:stream_block_start"]
    deltas = [p for name, p in hooks.events if name == "llm:stream_block_delta"]
    aborts = [p for name, p in hooks.events if name == "llm:stream_aborted"]
    assert len(aborts) == int(partial)
    if partial:
        assert starts[0]["request_id"] == deltas[0]["request_id"] == aborts[0]["request_id"]
        names = hooks.emitted_names()
        assert names.index("llm:stream_block_start") < names.index("llm:stream_block_delta")
        assert names.index("llm:stream_block_delta") < names.index("llm:stream_aborted")
        assert names.index("llm:stream_aborted") < names.index("llm:response")
    terminal = responses(hooks)
    assert len(terminal) == 1 and terminal[0]["status"] == status
    sanitized = [p for name, p in hooks.events
                 if name in {"llm:response", "llm:progress", "llm:stream_aborted"}]
    assert live.PRIVATE not in json.dumps(sanitized)
    assert not any(name in {"provider:retry", "provider:fallback", "provider:tool_sequence_repaired"}
                   for name, _ in hooks.events)
    return terminal[0]["usage"]


@pytest.mark.asyncio
@pytest.mark.parametrize("usage,final,expected_cost", [
    ({"input_tokens": 125}, False, None),
    ({}, False, None),
    ({"input_tokens": 0, "output_tokens": 0}, False, None),
    (FULL, False, None),
    (FULL, True, "0.000420"),
    ({**FULL, "input_tokens": 0, "output_tokens": 0}, True, "0.00"),
    ({**FULL, "output_tokens": None}, True, None),
    ({**FULL, "cache_read_input_tokens": -1}, True, None),
    ({**FULL, "speed": "unknown-private-tier"}, True, None),
    ({k: v for k, v in FULL.items() if k != "service_tier"}, True, None),
    ({**FULL, "cache_creation_input_tokens": 10,
      "cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 2}}, True, None),
    ({**FULL, "cache_creation_input_tokens": 10,
      "cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 7}}, True, "0.00047325"),
])
async def test_failed_stream_raw_usage_never_replays(monkeypatch, usage, final, expected_cost):
    fixture_wire(monkeypatch, usage, final=final)
    async with live.wire_call(default_model="claude-sonnet-5") as (wire, provider, request, hooks):
        costs = []
        provider._add_cost = costs.append
        receipts = []
        task = asyncio.create_task(live.traced_complete(provider, request, hooks, receipts))
        try:
            await live.wait_receipt(receipts, lambda name, p: name == "llm:stream_block_delta")
            await asyncio.sleep(0.02)
            wire.release.set()
            with pytest.raises(RequestOutcomeUnknownError) as caught:
                await asyncio.wait_for(task, 2)
            live.assert_unknown(caught.value)
            measured = assert_settlement(hooks, True, "error")
            assert caught.value.usage == measured
            assert measured["input_tokens"] == (usage.get("input_tokens") if
                   type(usage.get("input_tokens")) is int and usage["input_tokens"] >= 0 else None)
            assert measured["output_tokens"] == (usage.get("output_tokens") if
                   type(usage.get("output_tokens")) is int and usage["output_tokens"] >= 0 else None)
            if expected_cost is None:
                assert measured["cost_usd"] is None and costs == []
            else:
                from decimal import Decimal
                assert Decimal(measured["cost_usd"]) == Decimal(expected_cost)
                assert costs == [Decimal(expected_cost)]
            assert wire.posts == wire.accepted == 1
            live.assert_logical_settlement(receipts, [1, 1])
            print("SETTLEMENT " + json.dumps({"status": "error", "posts": wire.posts, "usage": measured}))
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming,partial", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("cleanup_failure", [None, "raise", "cancel", "block"])
async def test_original_cancellation_and_bounded_cleanup(monkeypatch, streaming, partial, cleanup_failure):
    fixture_wire(monkeypatch, {"input_tokens": 125}, partial=partial)
    async with live.wire_call(streaming=streaming) as (wire, provider, request, hooks):
        costs = []
        provider._add_cost = costs.append
        receipts = []
        original = hooks.emit
        async def emit(name, data):
            await original(name, data)
            if (name == "llm:stream_aborted" or
                (name == "llm:response" and data["status"] == "cancelled")):
                if cleanup_failure == "raise":
                    raise RuntimeError(live.PRIVATE)
                if cleanup_failure == "cancel":
                    raise asyncio.CancelledError("wrong-cleanup-cancellation")
                if cleanup_failure == "block":
                    await asyncio.Event().wait()
        hooks.emit = emit
        task = asyncio.create_task(live.traced_complete(provider, request, hooks, receipts))
        try:
            if partial:
                await live.wait_receipt(receipts, lambda name, p: name == "llm:stream_block_delta")
            else:
                await asyncio.wait_for(wire.entered.wait(), 2)
                await asyncio.sleep(0.02)
            started = time.monotonic()
            task.cancel("original-local-stop")
            with pytest.raises(asyncio.CancelledError) as caught:
                await asyncio.wait_for(task, 1)
            assert caught.value.args == ("original-local-stop",)
            assert time.monotonic() - started < 0.5
            measured = assert_settlement(hooks, partial, "cancelled")
            assert caught.value.usage == measured
            assert measured["input_tokens"] == (125 if streaming else None)
            assert measured["output_tokens"] is None and measured["cost_usd"] is None
            assert costs == []
            assert wire.posts == wire.accepted == 1
            assert receipts[-1][1] == "logical:terminal"
            if partial:
                aborted = hooks.payload_for("llm:stream_aborted")
                assert aborted["error"]["type"] == "Cancelled"
            print("SETTLEMENT " + json.dumps({"status": "cancelled", "posts": wire.posts, "usage": measured}))
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_parsed_nonstream_failed_envelope_retains_complete_usage(monkeypatch):
    fixture_wire(monkeypatch, FULL)
    async with live.wire_call(streaming=False, default_model="claude-sonnet-5") as (wire, provider, request, hooks):
        costs = []
        provider._add_cost = costs.append
        wire.release.set()
        with pytest.raises(RequestOutcomeUnknownError) as caught:
            await provider.complete(request)
        measured = assert_settlement(hooks, False, "error")
        assert caught.value.usage == measured and measured["usage_complete"]
        assert str(costs[0]) == measured["cost_usd"] and len(costs) == 1
        assert wire.posts == wire.accepted == 1


@pytest.mark.asyncio
async def test_concurrent_same_provider_core_hook_origins(monkeypatch):
    fixture_wire(monkeypatch, {"input_tokens": 125})
    current_call = contextvars.ContextVar("settlement_call")
    coordinator = ModuleCoordinator()
    receipts = []
    async def record(name, data):
        receipts.append((current_call.get(), name, data))
        return HookResult()
    for name in ("llm:progress", "llm:stream_block_start", "llm:stream_block_delta",
                 "llm:stream_aborted", "llm:response"):
        coordinator.hooks.register(name, record)
    async with live.wire_call() as (wire, provider, request, _):
        provider.coordinator = coordinator
        async def complete(label):
            token = current_call.set(label)
            try:
                return await provider.complete(request.model_copy(deep=True))
            finally:
                current_call.reset(token)
        tasks = [asyncio.create_task(complete(label)) for label in ("root", "delegate")]
        try:
            async with asyncio.timeout(2):
                while not all(any(owner == label and name == "llm:stream_block_delta"
                                  for owner, name, _ in receipts) for label in ("root", "delegate")):
                    await asyncio.sleep(0.001)
            tasks[1].cancel("delegate-stop")
            with pytest.raises(asyncio.CancelledError):
                await tasks[1]
            wire.release.set()
            with pytest.raises(RequestOutcomeUnknownError):
                await tasks[0]
            assert wire.posts == wire.accepted == 2
            ids = []
            for label, status in (("root", "error"), ("delegate", "cancelled")):
                owned = SimpleNamespace(events=[(name, data) for owner, name, data in receipts if owner == label])
                aborts = [p for name, p in owned.events if name == "llm:stream_aborted"]
                terminal = [p for name, p in owned.events if name == "llm:response"]
                assert len(aborts) == len(terminal) == 1
                assert terminal[0]["status"] == status and terminal[0]["usage"]["input_tokens"] == 125
                ids.append(aborts[0]["request_id"])
                assert all(p["attempt"] == 1 for name, p in owned.events if name == "llm:progress")
            assert len(set(ids)) == 2 and current_call.get(None) is None
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("value", [True, -1, 2**63, "125", None, 1.5])
def test_invalid_counter_is_not_measured(value):
    usage = FailedUsage("claude-sonnet-5")
    usage.capture(anthropic.types.Usage.model_construct(**{**FULL, "input_tokens": value}), complete=True)
    measured, cost = usage.settlement()
    assert measured["input_tokens"] is None and cost is None


@pytest.mark.asyncio
async def test_partial_cancel_during_delta_hook_closes_display(monkeypatch):
    fixture_wire(monkeypatch, {"input_tokens": 125})
    async with live.wire_call() as (wire, provider, request, hooks):
        seen = asyncio.Event()
        original = hooks.emit
        async def emit(name, payload):
            await original(name, payload)
            if name == "llm:stream_block_delta":
                seen.set()
                await asyncio.Event().wait()
        hooks.emit = emit
        task = asyncio.create_task(provider.complete(request))
        try:
            await asyncio.wait_for(seen.wait(), 2)
            task.cancel("stop-in-delta")
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert caught.value.args == ("stop-in-delta",)
            assert_settlement(hooks, True, "cancelled")
            assert wire.posts == wire.accepted == 1
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_final_usage_cancel_retains_one_cost_callback(monkeypatch):
    fixture_wire(monkeypatch, FULL, final=True)
    async with live.wire_call(default_model="claude-sonnet-5") as (wire, provider, request, hooks):
        costs = []
        provider._add_cost = costs.append
        task = asyncio.create_task(provider.complete(request))
        try:
            await asyncio.wait_for(wire.sent.wait(), 2)
            await asyncio.sleep(0.02)
            task.cancel("stop-after-final-usage")
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            measured = assert_settlement(hooks, True, "cancelled")
            assert caught.value.usage == measured and len(costs) == 1
            assert str(costs[0]) == measured["cost_usd"]
            assert wire.posts == wire.accepted == 1
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_failure_without_hooks_still_attaches_measured_usage(monkeypatch):
    from amplifier_module_provider_anthropic._failed_usage import FailedUsage
    consumed = asyncio.Event()
    original_capture = FailedUsage.capture
    def capture(self, usage, **kwargs):
        original_capture(self, usage, **kwargs)
        if self.values.get("input_tokens") == 125:
            consumed.set()
    monkeypatch.setattr(FailedUsage, "capture", capture)
    fixture_wire(monkeypatch, {"input_tokens": 125}, partial=False)
    async with live.wire_call() as (wire, provider, request, hooks):
        provider.coordinator = None
        task = asyncio.create_task(provider.complete(request))
        try:
            await asyncio.wait_for(consumed.wait(), 2)
            wire.release.set()
            with pytest.raises(RequestOutcomeUnknownError) as caught:
                await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert caught.value.usage["input_tokens"] == 125
        assert caught.value.usage["output_tokens"] is None
        assert caught.value.usage["cost_usd"] is None
        assert hooks.events == [] and wire.posts == wire.accepted == 1