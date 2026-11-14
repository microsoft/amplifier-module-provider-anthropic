"""Provider pricing contracts: real Core models, no vendor calls."""

from decimal import Decimal
import asyncio
import json
from types import SimpleNamespace

import pytest
import httpx2 as httpx
from anthropic import AsyncAnthropic
from anthropic._models import construct_type
from anthropic.types import Message as SDKMessage
from amplifier_core import ModelInfo
from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._cost import (
    compute_cost, compute_cost_result, get_rates, pricing_metadata,
)

HAIKU = "claude-haiku-5-5"
FABLE = "claude-fable-5-2"


def provider():
    return AnthropicProvider(api_key="offline-placeholder", config={"enable_prompt_caching": False})


def response(model=HAIKU, **counts):
    usage = {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 0, **counts}
    return SimpleNamespace(model=model, usage=SimpleNamespace(**usage), content=[], stop_reason="end_turn")


@pytest.mark.parametrize("model", ["claude-sonnet-5", FABLE, "claude-opus-5-5"])
def test_fixed_model_info_uses_shared_rates(model):
    info = provider()._model_info(SimpleNamespace(id=model), model)
    assert isinstance(info, ModelInfo)
    rates = get_rates(model)
    if "pricing" in ModelInfo.model_fields:
        assert info.pricing.input_per_million == float(rates["input_per_m"])
        assert info.pricing.output_per_million == float(rates["output_per_m"])
        assert info.pricing.cache_read_per_million == float(rates["cache_read_per_m"])
        assert info.pricing.cache_write_per_million == float(rates["cache_write_per_m"])
    else:
        # Locked Core predates Pricing; extension still supplies every rate.
        assert "pricing" not in info.model_dump()
    data = json.loads(info.model_dump_json())
    assert data["pricing_details"] == pricing_metadata(model)
    assert data["pricing_details"]["kind"] == "fixed"
    assert "pricing_details" not in info.defaults
    assert "anthropic_pricing" not in info.defaults


def test_tiered_model_is_not_flattened_and_metadata_is_fresh():
    info = provider()._model_info(SimpleNamespace(id=HAIKU), HAIKU)
    assert getattr(info, "pricing", None) is None
    details = info.model_dump()["pricing_details"]
    assert details["version"] == 1 and details["basis"] == "standard_global"
    assert details["threshold_basis"] == "input_tokens+cache_read_input_tokens+cache_creation_input_tokens"
    assert details["tiers"][0]["up_to"] == 100_000
    assert details["tiers"][1]["above"] == 100_000
    assert details["tiers"][0]["rates"]["input"] == "0.10"
    details["tiers"][0]["rates"]["input"] = "999"
    assert pricing_metadata(HAIKU)["tiers"][0]["rates"]["input"] == "0.10"


@pytest.mark.parametrize("model", ["claude-mythos-5-2", FABLE + "-latest", HAIKU + "-20261007", "opaque-id"])
def test_no_guessed_prices(model):
    assert get_rates(model) is None
    assert pricing_metadata(model) is None
    info = provider()._model_info(SimpleNamespace(id=model), model)
    assert getattr(info, "pricing", None) is None
    assert getattr(info, "pricing_details", None) is None
    assert compute_cost(model, input_tokens=1) is None


@pytest.mark.parametrize("prompt", [99_999, 100_000, 100_001])
@pytest.mark.parametrize("category", ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"])
def test_threshold_counts_every_prompt_category_once(prompt, category):
    kwargs = {category: prompt, "output_tokens": 1000}
    if category == "cache_creation_input_tokens":
        kwargs.update(cache_creation_5m_input_tokens=prompt // 2,
                      cache_creation_1h_input_tokens=prompt - prompt // 2)
    result = compute_cost(HAIKU, **kwargs)
    factor = Decimal(5 if prompt > 100_000 else 1)
    unit = {"input_tokens": Decimal(".10"), "cache_read_input_tokens": Decimal(".01")}
    if category == "cache_creation_input_tokens":
        expected = Decimal(prompt // 2) * Decimal(".125") + Decimal(prompt - prompt // 2) * Decimal(".20")
    else:
        expected = Decimal(prompt) * unit[category]
    assert result == factor * (expected + Decimal("500")) / Decimal(1_000_000)


def test_cached_prompt_boundary_does_not_double_count_split_or_reprice_prior():
    low = compute_cost(HAIKU, input_tokens=10, cache_read_input_tokens=99_980,
                       cache_creation_input_tokens=10, cache_creation_5m_input_tokens=5,
                       cache_creation_1h_input_tokens=5, output_tokens=100)
    high = compute_cost(HAIKU, input_tokens=11, cache_read_input_tokens=99_980,
                        cache_creation_input_tokens=10, cache_creation_5m_input_tokens=5,
                        cache_creation_1h_input_tokens=5, output_tokens=100)
    assert low == Decimal(".001052425")
    assert high == Decimal(".005262625")
    assert compute_cost(HAIKU, input_tokens=10, cache_read_input_tokens=99_980,
                        cache_creation_input_tokens=10, cache_creation_5m_input_tokens=5,
                        cache_creation_1h_input_tokens=5, output_tokens=100) == low


@pytest.mark.parametrize("kwargs,reason", [
    ({"cache_creation_input_tokens": 10}, "cache_write_ttl_unavailable"),
    ({"cache_creation_input_tokens": 10, "cache_creation_5m_input_tokens": 10}, "cache_write_ttl_split_incomplete"),
    ({"cache_creation_input_tokens": 10, "cache_creation_5m_input_tokens": 0, "cache_creation_1h_input_tokens": 0}, "cache_write_ttl_split_mismatch"),
    ({"input_tokens": True}, "invalid_usage_counts"),
    ({"input_tokens": -1}, "invalid_usage_counts"),
])
def test_unresolved_haiku_counts_never_false_precision(kwargs, reason):
    result = compute_cost_result(HAIKU, **kwargs)
    assert result.cost_usd is None and result.unavailable_reason == reason


def test_authoritative_split_can_replace_absent_aggregate_and_zero_is_known():
    assert compute_cost(HAIKU) == Decimal(0)
    assert compute_cost(HAIKU, cache_creation_input_tokens=None,
                        cache_creation_5m_input_tokens=10, cache_creation_1h_input_tokens=0) == Decimal(".00000125")
    assert compute_cost(HAIKU, cache_creation_input_tokens=10, cache_write_ttl="1h") == Decimal(".000002")


def test_fable_confirmed_fast_and_us_stack_all_categories():
    kwargs = {"input_tokens": 1000, "output_tokens": 200,
              "cache_read_input_tokens": 3000, "cache_creation_input_tokens": 1000,
              "cache_creation_5m_input_tokens": 400, "cache_creation_1h_input_tokens": 600}
    base = compute_cost(FABLE, **kwargs)
    assert base == Decimal(".0302")
    assert compute_cost(FABLE, speed="fast", inference_geo="us", **kwargs) == base * Decimal("2.2")
    assert compute_cost(FABLE, speed="standard", requested_speed="fast", **kwargs) == base
    assert compute_cost(FABLE, requested_speed="fast", **kwargs) is None
    assert compute_cost(FABLE, requested_speed="fast", speed=[], **kwargs) is None
    assert compute_cost(FABLE, requested_speed={}, **kwargs) is None
    assert compute_cost(FABLE, inference_geo="unknown", **kwargs) is None
    # Pricing support does not enable a new request capability.
    assert not provider()._get_capabilities(FABLE).supports_speed


def test_runtime_final_wire_geography_response_speed_and_unknown_reason():
    p = provider()
    r = response(FABLE, input_tokens=1000, output_tokens=0, speed="fast")
    result = p._convert_to_chat_response(r, request_params={
        "inference_geo": "global", "extra_body": {"inference_geo": "us", "speed": "fast"}})
    assert result.usage.cost_usd == Decimal(".0176")
    r.usage.speed = None
    result = p._convert_to_chat_response(r, request_params={"extra_body": {"speed": "fast"}})
    assert result.usage.cost_usd is None
    assert result.metadata["anthropic_cost_unavailable"] == "response_speed_unavailable"


def test_runtime_aggregate_ttl_uses_request_not_config():
    p = provider()
    r = response(cache_creation_input_tokens=100, input_tokens=0, output_tokens=0)
    assert p._convert_to_chat_response(r, request_params={}).usage.cost_usd is None
    result = p._convert_to_chat_response(r, request_params={
        "system": [{"type": "text", "text": "prefix", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]})
    assert result.usage.cost_usd == Decimal(".00002")


@pytest.mark.parametrize("usage,reason", [
    ({"cache_creation_input_tokens": 10}, "cache_write_ttl_unavailable"),
    ({"cache_creation_input_tokens": 10, "cache_creation": {
        "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}},
     "cache_write_ttl_split_mismatch"),
    ({"cache_creation_input_tokens": 10, "cache_creation": {
        "ephemeral_5m_input_tokens": 10}}, "cache_write_ttl_split_incomplete"),
])
def test_real_sdk_absent_and_zero_cache_split_are_distinct(usage, reason):
    # SDK permissive response construction must not turn missing split fields
    # or an explicit all-zero split into a false exact cost.
    raw = {"id": "msg_offline", "type": "message", "role": "assistant",
           "model": HAIKU, "content": [], "stop_reason": "end_turn",
           "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0, **usage}}
    # The SDK transport constructs permissively (it does not model_validate
    # server JSON); use its own construction to exercise absent required keys.
    message = construct_type(value=raw, type_=SDKMessage)
    result = provider()._convert_to_chat_response(message, request_params={})
    assert result.usage.cost_usd is None
    assert result.metadata["anthropic_cost_unavailable"] == reason


@pytest.mark.parametrize("streaming", [False, True])
def test_real_sdk_transport_high_tier_cost_and_event_json(streaming):
    from amplifier_core.message_models import ChatRequest, Message
    from tests._helpers import FakeCoordinator

    body = {"id": "msg_offline", "type": "message", "role": "assistant",
            "model": HAIKU, "content": [], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 10, "cache_read_input_tokens": 100_000,
                      "cache_creation_input_tokens": 0}}
    initial = {**body, "stop_reason": None, "usage": {**body["usage"], "output_tokens": 0}}
    events = [
        {"type": "message_start", "message": initial},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 10}},
        {"type": "message_stop"},
    ]
    sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    async def run():
        p = AnthropicProvider(api_key="offline-placeholder", config={
            "default_model": HAIKU, "use_streaming": streaming, "max_retries": 0,
            "enable_prompt_caching": False}, coordinator=FakeCoordinator())
        p._client = AsyncAnthropic(api_key="offline-placeholder", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, **(
                    {"headers": {"content-type": "text/event-stream"}, "content": sse.encode()}
                    if streaming else {"json": body})))))
        try:
            result = await p.complete(ChatRequest(messages=[Message(role="user", content="hello")]))
            assert result.usage.cost_usd == Decimal(".0050255")
            event = p.coordinator.hooks.payload_for("llm:response")
            assert Decimal(json.loads(json.dumps(event))["usage"]["cost_usd"]) == Decimal(".0050255")
            assert "anthropic_cost_unavailable" not in (result.metadata or {})
        finally:
            await p.close()
    asyncio.run(run())
