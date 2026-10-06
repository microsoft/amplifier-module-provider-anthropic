"""Three final seams through real SDK loopback and installed Core hooks."""
import asyncio
import json
import time
from decimal import Decimal

import anthropic
import pytest
from amplifier_core import HookResult, ModuleCoordinator
from amplifier_core.llm_errors import InvalidRequestError, LLMError
from amplifier_module_provider_anthropic import _terminal_delivery
from amplifier_module_provider_anthropic._request_safety import LocalRequestError, RequestOutcomeUnknownError
from tests import test_transport_liveness as live
from tests.test_failed_accounting import FULL


class FinalWire(live.Wire):
    def __init__(self, mode, streaming):
        super().__init__(mode, streaming)

    async def json_reply(self, writer, status, body, extra=b""):
        if isinstance(body, dict) and body.get("type") == "message":
            body["usage"] = dict(FULL)
            if self.mode == "bad_tool":
                # The SDK parses this vendor envelope/usage. Core's mapping
                # rejects the tool input rather than returning failed content.
                body["content"] = [{
                    "type": "tool_use", "id": "tool_fixture", "name": "fixture",
                    "input": live.PRIVATE,
                }]
        await super().json_reply(writer, status, body, extra)


def terminal(hooks):
    return [p for n, p in hooks.events if n == "llm:response" and p["status"] != "ok"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    "bad_tool", "notification", "callback", "bad_tool_callback",
    "callback_cancel", "bad_tool_callback_cancel",
])
async def test_cost_ownership_at_actual_callback(monkeypatch, failure):
    monkeypatch.setattr(live, "Wire", FinalWire)
    mode = "bad_tool" if failure.startswith("bad_tool") else failure
    async with live.wire_call(mode, streaming=False, default_model="claude-sonnet-5") as (wire, provider, request, hooks):
        costs = []
        def contribute(cost):
            costs.append(cost)
            if failure.endswith("_cancel"):
                raise asyncio.CancelledError("not-caller-stop")
            if failure in {"callback", "bad_tool_callback"}:
                raise RuntimeError(live.PRIVATE)
        provider._add_cost = contribute
        original = hooks.emit
        primary = RuntimeError(live.PRIVATE)
        async def emit(name, payload):
            if name == "llm:response" and payload["status"] == "ok" and failure == "notification":
                raise primary
            await original(name, payload)
        hooks.emit = emit
        with pytest.raises(LocalRequestError) as caught:
            await provider.complete(request)
        error = caught.value
        assert error.request_outcome == "received" and error.effects == "occurred"
        assert not error.retryable and live.PRIVATE not in str(error)
        assert wire.posts == wire.accepted == 1
        assert costs == [Decimal("0.000420")]
        receipt, = terminal(hooks)
        assert receipt["usage"] == error.usage
        assert receipt["request_outcome"] == "received"
        assert error.usage["cost_usd"] == "0.000420"
        assert error.usage["cost_callback_state"] == ("unknown" if "callback" in failure else "returned")
        if failure == "notification":
            assert error.__cause__ is primary
        if failure.startswith("bad_tool"):
            assert type(error.__cause__).__name__ == "ValidationError"
        assert live.PRIVATE not in json.dumps(receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["concurrency", "parameter", "missing_key", "invalid_url", "protocol"])
async def test_local_failure_zero_posts_no_phantom_usage(failure):
    async with live.wire_call("burst", streaming=False) as (wire, provider, request, hooks):
        original = hooks.emit
        async def emit(name, payload):
            if failure == "concurrency" and name == "provider:concurrency":
                raise RuntimeError(live.PRIVATE)
            await original(name, payload)
        hooks.emit = emit
        if failure == "parameter":
            provider.extra_request_params = {"max_tokens": object()}
        if failure == "missing_key":
            provider._client.api_key = None
            provider._client.auth_token = None
        if failure == "invalid_url":
            await provider.close()
            provider._base_url = "http://127.0.0.1:invalid"
        if failure == "protocol":
            provider._client.base_url = "ftp://127.0.0.1"
        with pytest.raises(LocalRequestError) as caught:
            await asyncio.wait_for(provider.complete(request), 2)
        error = caught.value
        assert error.request_outcome == "not_dispatched" and error.effects == "none"
        assert error.retryable is False and not hasattr(error, "usage")
        assert wire.posts == wire.accepted == 0
        assert not any(n == "provider:retry" for n, _ in hooks.events)
        assert all("usage" not in p for p in terminal(hooks))
        assert live.PRIVATE not in str(error)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["burst", "partial_reset"])
@pytest.mark.parametrize("failure", ["raise", "cancel", "block"])
@pytest.mark.parametrize("native_core", [False, True])
async def test_terminal_pending_activity_preserves_outcome(mode, failure, native_core):
    async with live.wire_call(mode) as (wire, provider, request, fake):
        receipts = []
        entered = asyncio.Event()
        finished = asyncio.Event()
        callback_release = asyncio.Event()
        consumed = asyncio.Event()
        activities = 0
        async def record(name, data):
            nonlocal activities
            receipts.append((name, data))
            if name == "llm:stream_block_delta":
                consumed.set()
            if name == "llm:progress" and data["observation"] == "response_activity":
                activities += 1
                if activities == 2:
                    entered.set()
                    try:
                        if failure == "raise":
                            raise RuntimeError(live.PRIVATE)
                        if failure == "cancel":
                            raise asyncio.CancelledError("not-a-caller-stop")
                        await callback_release.wait()
                    finally:
                        finished.set()
            return HookResult()
        if native_core:
            coordinator = ModuleCoordinator()
            for name in ("llm:progress", "llm:response", "llm:stream_aborted", "llm:stream_block_delta"):
                coordinator.hooks.register(name, record)
            provider.coordinator = coordinator
        else:
            original = fake.emit
            async def emit(name, data):
                await original(name, data)
                await record(name, data)
            fake.emit = emit
        task = asyncio.create_task(provider.complete(request))
        try:
            if mode == "partial_reset":
                await asyncio.wait_for(consumed.wait(), 2)
            wire.release.set()
            started = time.monotonic()
            if mode == "partial_reset":
                with pytest.raises(RequestOutcomeUnknownError) as caught:
                    await asyncio.wait_for(task, 2)
                live.assert_unknown(caught.value)
            else:
                assert (await asyncio.wait_for(task, 2)).finish_reason == "end_turn"
            assert time.monotonic() - started < 1
            assert entered.is_set() and activities == 2
            assert task.cancelling() == 0
            # These are this fixture's registered callbacks, not foreign tasks.
            # Native Core can complete callback cleanup after provider settlement.
            callback_release.set()
            await asyncio.wait_for(finished.wait(), 2)
            assert wire.posts == wire.accepted == 1
            assert not any(n == "provider:retry" for n, _ in receipts)
            assert live.PRIVATE not in json.dumps([d for n, d in receipts if n == "llm:progress"])
        finally:
            callback_release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_new_external_stop_has_priority_during_terminal_flush():
    async with live.wire_call("burst") as (wire, provider, request, hooks):
        seen = asyncio.Event()
        original = hooks.emit
        activities = 0
        async def emit(name, data):
            nonlocal activities
            await original(name, data)
            if name == "llm:progress" and data["observation"] == "response_activity":
                activities += 1
                if activities == 2:
                    seen.set()
                    await asyncio.Event().wait()
        hooks.emit = emit
        task = asyncio.create_task(provider.complete(request))
        try:
            await asyncio.wait_for(seen.wait(), 2)
            task.cancel("caller-stop-at-settlement")
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert caught.value.args == ("caller-stop-at-settlement",)
            assert task.cancelling() == 1 and wire.posts == 1
            assert caught.value.usage["input_tokens"] is not None
            assert caught.value.usage["output_tokens"] is not None
            assert caught.value.usage["cost_callback_state"] in {"returned", "not_invoked"}
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_second_external_stop_preserves_new_message_and_count():
    entered = asyncio.Event()
    task = None
    async def blocked():
        entered.set()
        await asyncio.Event().wait()
    async def owner():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await _terminal_delivery(blocked())
            raise
    task = asyncio.create_task(owner())
    await asyncio.sleep(0)
    task.cancel("first-stop")
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel("second-stop")
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.args == ("second-stop",)
    assert task.cancelling() == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("native_core", [False, True])
async def test_second_stop_at_real_cancel_cleanup(native_core):
    async with live.wire_call("partial_hold") as (wire, provider, request, fake):
        displayed = asyncio.Event()
        cleanup = asyncio.Event()
        finished = asyncio.Event()
        callback_release = asyncio.Event()
        async def record(name, data):
            if name == "llm:stream_block_delta":
                displayed.set()
            if name == "llm:stream_aborted":
                cleanup.set()
                try:
                    await callback_release.wait()
                finally:
                    finished.set()
            return HookResult()
        if native_core:
            coordinator = ModuleCoordinator()
            for name in ("llm:stream_block_delta", "llm:stream_aborted", "llm:response"):
                coordinator.hooks.register(name, record)
            provider.coordinator = coordinator
        else:
            original = fake.emit
            async def emit(name, data):
                await original(name, data)
                await record(name, data)
            fake.emit = emit
        task = asyncio.create_task(provider.complete(request))
        try:
            await asyncio.wait_for(displayed.wait(), 2)
            task.cancel("first-wire-stop")
            await asyncio.wait_for(cleanup.wait(), 2)
            task.cancel("second-wire-stop")
            with pytest.raises(asyncio.CancelledError) as caught:
                await asyncio.wait_for(task, 2)
            assert caught.value.args == ("second-wire-stop",)
            assert task.cancelling() == 2 and wire.posts == 1
            callback_release.set()
            await asyncio.wait_for(finished.wait(), 2)
        finally:
            callback_release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["tool_choice", "request_hook"])
async def test_predispatch_kernel_error_classification(failure):
    async with live.wire_call("burst", streaming=False) as (wire, provider, request, hooks):
        original = hooks.emit
        primary = LLMError(live.PRIVATE, retryable=True)
        async def emit(name, data):
            if failure == "request_hook" and name == "llm:request":
                raise primary
            await original(name, data)
        hooks.emit = emit
        if failure == "tool_choice":
            request.tool_choice = live.PRIVATE
        with pytest.raises(LLMError) as caught:
            await provider.complete(request)
        error = caught.value
        assert error.request_outcome == "not_dispatched" and error.effects == "none"
        assert not error.retryable and not hasattr(error, "usage")
        receipt, = terminal(hooks)
        assert "usage" not in receipt and receipt["request_outcome"] == "not_dispatched"
        assert live.PRIVATE not in str(error) and live.PRIVATE not in json.dumps(receipt)
        assert wire.posts == 0
        if failure == "tool_choice":
            assert isinstance(error, InvalidRequestError)
        else:
            assert isinstance(error, LocalRequestError) and error.__cause__ is primary


@pytest.mark.asyncio
@pytest.mark.parametrize("already_cancelled", [False, True])
async def test_new_stop_during_yielding_failing_child_cleanup(already_cancelled):
    started = asyncio.Event()
    cleanup = asyncio.Event()
    release = asyncio.Event()
    async def child():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup.set()
            await release.wait()
            raise RuntimeError(live.PRIVATE)
    async def owner():
        if already_cancelled:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await _terminal_delivery(child())
                raise
        else:
            await _terminal_delivery(child())
    task = asyncio.create_task(owner())
    try:
        await asyncio.sleep(0)
        if already_cancelled:
            task.cancel("old-stop")
        await asyncio.wait_for(started.wait(), 2)
        await asyncio.wait_for(cleanup.wait(), 2)
        task.cancel("new-stop-during-cleanup")
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.args == ("new-stop-during-cleanup",)
        assert task.cancelling() == (2 if already_cancelled else 1)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class FirstErrorWire(live.Wire):
    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            path = headers.split(b" ")[1]
            if path.startswith(b"/v1/models"):
                await self.json_reply(writer, 200, {"data": [], "has_more": False})
                return
            length = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                          if line.lower().startswith(b"content-length:"))
            await reader.readexactly(length)
            assert path == b"/v1/messages"
            self.posts += 1
            self.accepted += 1
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                         b"connection: close\r\n\r\n" + live.sse({
                             "type": "error", "error": {
                                 "type": "overloaded_error", "message": live.PRIVATE,
                             },
                         }))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            self.tasks.discard(task)


@pytest.mark.asyncio
async def test_first_sse_error_is_not_http_nonacceptance(monkeypatch):
    monkeypatch.setattr(live, "Wire", FirstErrorWire)
    async with live.wire_call() as (wire, provider, request, hooks):
        with pytest.raises(RequestOutcomeUnknownError) as caught:
            await provider.complete(request)
        live.assert_unknown(caught.value)
        assert caught.value.status_code == 200
        assert wire.posts == wire.accepted == 1
        assert not any(n == "provider:retry" for n, _ in hooks.events)
        assert caught.value.usage["input_tokens"] is None
        assert caught.value.usage["output_tokens"] is None
        assert caught.value.usage["cost_usd"] is None
        assert not any(p["observation"] == "response_activity" for p in live.progress(hooks))


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ["redirects", "retries"])
async def test_actual_embedder_sdk_replay_guards_remain_local(guard):
    async with live.wire_call("burst") as (wire, provider, request, hooks):
        if guard == "redirects":
            provider._client._client.follow_redirects = True
        else:
            provider._client.max_retries = 1
        with pytest.raises(InvalidRequestError) as caught:
            await provider.complete(request)
        error = caught.value
        assert error.request_outcome == "not_dispatched" and error.effects == "none"
        assert not error.retryable and not hasattr(error, "usage")
        assert wire.posts == 0 and live.progress(hooks) == []
        receipt, = terminal(hooks)
        assert "usage" not in receipt