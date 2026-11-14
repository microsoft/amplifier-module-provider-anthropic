"""Started streams must not disappear from session costs after failure/cancel.

Real SDK parsing over transport-only synthetic streams. No paid vendor calls.
"""

import asyncio
from decimal import Decimal
import json

import httpx2 as httpx
import pytest
from anthropic import AsyncAnthropic
from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import InvalidRequestError, LLMError
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_anthropic import mount

MODEL = "claude-sonnet-5-5"
KNOWN_COST = Decimal(".0003")  # 100 fresh input at $2/M + 10 output at $10/M.


def request():
    return ChatRequest(messages=[Message(role="user", content="hello")])


def sse(kind, **payload):
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n".encode()


class AttemptStream(httpx.AsyncByteStream):
    def __init__(self, scenario, ready):
        self.scenario = scenario
        self.ready = ready
        self.closed = False

    async def __aiter__(self):
        yield sse("message_start", message={
            "id": "msg_offline", "type": "message", "role": "assistant", "model": MODEL,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 0, "inference_geo": "global"},
        })
        if self.scenario == "fail_after_start":
            raise httpx.ReadError("synthetic started stream failure")
        yield sse("content_block_start", index=0, content_block={"type": "text", "text": ""})
        yield sse("content_block_delta", index=0, delta={"type": "text_delta", "text": "ok"})
        if self.scenario == "cancel":
            self.ready.set()
            await asyncio.Event().wait()
        if self.scenario == "fail_after_output":
            raise httpx.ReadError("synthetic output stream failure")
        yield sse("content_block_stop", index=0)
        yield sse("message_delta", delta={"stop_reason": "end_turn", "stop_sequence": None},
                  usage={"output_tokens": 10})
        yield sse("message_stop")

    async def aclose(self):
        self.closed = True


async def mounted(scenarios, *, max_retries=0):
    coordinator = ModuleCoordinator()
    cleanup = await mount(coordinator, {
        "api_key": "offline-placeholder", "default_model": MODEL,
        "use_streaming": True, "enable_prompt_caching": False,
        "max_retries": max_retries, "min_retry_delay": .001,
        "max_retry_delay": .001, "retry_jitter": False,
        "overload_fallback_enabled": False, "refusal_fallback_enabled": False,
        "max_concurrent_requests": 0,
    })
    provider = coordinator.get("providers", "anthropic")
    provider._runtime_model_info_cache[MODEL] = None
    seen = []
    costs = []
    ready = asyncio.Event()
    streams = []
    add_cost = provider._add_cost

    def record(cost):
        costs.append(cost)
        add_cost(cost)
    provider._add_cost = record

    def handler(req):
        seen.append(req.url.path)
        assert req.url.path == "/v1/messages"  # No discovery/count/other transport.
        scenario = scenarios[len(seen) - 1]
        if scenario == "rejected":
            return httpx.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error", "message": "synthetic request rejected"}})
        stream = AttemptStream(scenario, ready)
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    provider._client = AsyncAnthropic(api_key="offline-placeholder", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return coordinator, provider, cleanup, seen, costs, ready, streams


@pytest.mark.parametrize("prior_known", [False, True])
@pytest.mark.parametrize("scenario", ["fail_after_start", "fail_after_output"])
def test_interrupted_attempt_records_unknown_once(prior_known, scenario):
    async def run():
        scenarios = (["success"] if prior_known else []) + [scenario]
        c, p, cleanup, seen, costs, _, streams = await mounted(scenarios)
        try:
            assert await c.collect_contributions("session.cost") == []
            if prior_known:
                await p.complete(request())
                assert costs == [KNOWN_COST]
            with pytest.raises(LLMError) as error:
                await p.complete(request())
            assert isinstance(error.value.__cause__, httpx.ReadError)
            assert costs == ([KNOWN_COST] if prior_known else []) + [None]
            assert len(seen) == len(scenarios)
            assert all(s.closed for s in streams)
            assert await c.collect_contributions("session.cost") == [{"cost_usd": None}]
        finally:
            await cleanup()
    asyncio.run(run())


@pytest.mark.parametrize("prior_known", [False, True])
def test_cancelled_started_attempt_keeps_cancellation_and_unknown(prior_known):
    async def run():
        scenarios = (["success"] if prior_known else []) + ["cancel"]
        c, p, cleanup, seen, costs, ready, streams = await mounted(scenarios)
        try:
            if prior_known:
                await p.complete(request())
            task = asyncio.create_task(p.complete(request()))
            await asyncio.wait_for(ready.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert costs == ([KNOWN_COST] if prior_known else []) + [None]
            assert len(seen) == len(scenarios) and all(s.closed for s in streams)
            assert await c.collect_contributions("session.cost") == [{"cost_usd": None}]
        finally:
            await cleanup()
    asyncio.run(run())


def test_retry_success_does_not_hide_prior_started_attempt():
    async def run():
        c, p, cleanup, seen, costs, _, _ = await mounted(
            ["fail_after_output", "success"], max_retries=1)
        try:
            result = await p.complete(request())
            assert result.usage.cost_usd == KNOWN_COST
            assert costs == [None, KNOWN_COST]
            assert len(seen) == 2
            assert await c.collect_contributions("session.cost") == [{"cost_usd": None}]
        finally:
            await cleanup()
    asyncio.run(run())


def test_success_not_double_counted_even_when_later_hook_fails():
    async def run():
        c, p, cleanup, _, costs, _, _ = await mounted(["success"])
        emit = c.hooks.emit
        async def fail_after_conversion(event, data):
            if event == "llm:response":
                raise RuntimeError("synthetic post-accounting hook failure")
            return await emit(event, data)
        c.hooks.emit = fail_after_conversion
        try:
            with pytest.raises(RuntimeError, match="synthetic post-accounting"):
                await p.complete(request())
            assert costs == [KNOWN_COST]
            assert await c.collect_contributions("session.cost") == [{"cost_usd": str(KNOWN_COST)}]
        finally:
            await cleanup()
    asyncio.run(run())


def test_conversion_failure_after_stream_return_records_unknown(monkeypatch):
    async def run():
        c, p, cleanup, _, costs, _, _ = await mounted(["success"])
        def fail(*args, **kwargs):
            raise ValueError("synthetic conversion failure")
        monkeypatch.setattr(p, "_convert_to_chat_response", fail)
        try:
            with pytest.raises(ValueError, match="synthetic conversion"):
                await p.complete(request())
            assert costs == [None]
            assert await c.collect_contributions("session.cost") == [{"cost_usd": None}]
        finally:
            await cleanup()
    asyncio.run(run())


def test_validation_and_rejection_before_stream_start_do_not_poison_cost():
    async def run():
        c, p, cleanup, seen, costs, _, _ = await mounted(["rejected"])
        try:
            # Real Haiku prevalidation fails without a Models or Messages call.
            with pytest.raises(InvalidRequestError):
                await p.complete(request(), model="claude-haiku-5-5", thinking_type="enabled")
            assert seen == [] and costs == []
            assert await c.collect_contributions("session.cost") == []
            with pytest.raises(InvalidRequestError):
                await p.complete(request())
            assert len(seen) == 1 and costs == []
            assert await c.collect_contributions("session.cost") == []
        finally:
            await cleanup()
    asyncio.run(run())
