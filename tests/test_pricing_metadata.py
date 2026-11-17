"""Provider pricing contracts: real Core models, no vendor calls."""

from decimal import Decimal
import asyncio
import gzip
import json
import zlib
from types import SimpleNamespace

import pytest
import httpx2 as httpx
from anthropic import AsyncAnthropic
from anthropic._models import construct_type
from anthropic.types import Message as SDKMessage
from amplifier_core import ModelInfo
from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._cost import (
    compute_cost, compute_cost_result, get_rates, pricing_metadata, resolve_inference_geo,
)

HAIKU = "claude-haiku-5-5"
FABLE = "claude-fable-5-2"
MYTHOS = "claude-mythos-5-2"


def provider():
    return AnthropicProvider(api_key="offline-placeholder", config={"enable_prompt_caching": False})


def response(model=HAIKU, **counts):
    usage = {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 0, **counts}
    return SimpleNamespace(model=model, usage=SimpleNamespace(**usage), content=[], stop_reason="end_turn")


@pytest.mark.parametrize("model", ["claude-sonnet-5", FABLE, MYTHOS, "claude-opus-5-5"])
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


@pytest.mark.parametrize("model", [MYTHOS + "-latest", MYTHOS + "-20261109",
    "claude-mythos-5-20", "claude-mythos-5", "claude-mythos-5-1",
    "claude-mythos-preview", FABLE + "-latest", HAIKU + "-20261007", "opaque-id"])
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
@pytest.mark.parametrize("model", [HAIKU, FABLE, MYTHOS])
def test_unresolved_counts_never_false_precision(model, kwargs, reason):
    result = compute_cost_result(model, strict=True, **kwargs)
    assert result.cost_usd is None and result.unavailable_reason == reason


@pytest.mark.parametrize("model", [HAIKU, FABLE, MYTHOS])
def test_true_zero_is_known(model):
    assert compute_cost(model, strict=True) == Decimal(0)


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
    result = p._convert_to_chat_response(r, request_params={
        "inference_geo": "global", "extra_body": {"speed": "fast"}})
    assert result.usage.cost_usd is None
    assert result.metadata["anthropic_cost_unavailable"] == "response_speed_unavailable"


def test_runtime_aggregate_ttl_uses_request_not_config():
    p = provider()
    r = response(cache_creation_input_tokens=100, input_tokens=0, output_tokens=0,
                 inference_geo="global")
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
@pytest.mark.parametrize("model", [HAIKU, MYTHOS])
def test_real_sdk_absent_and_zero_cache_split_are_distinct(model, usage, reason):
    # SDK permissive response construction must not turn missing split fields
    # or an explicit all-zero split into a false exact cost.
    raw = {"id": "msg_offline", "type": "message", "role": "assistant",
           "model": model, "content": [], "stop_reason": "end_turn",
           "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0, **usage}}
    # The SDK transport constructs permissively (it does not model_validate
    # server JSON); use its own construction to exercise absent required keys.
    message = construct_type(value=raw, type_=SDKMessage)
    result = provider()._convert_to_chat_response(message, request_params={})
    assert result.usage.cost_usd is None
    assert result.metadata["anthropic_cost_unavailable"] == reason


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("model", [HAIKU, MYTHOS])
@pytest.mark.parametrize("actual,wire,final_geo,expected", [
    ("us", "global", None, ".00552805"),
    ("global", "us", None, ".0050255"),
    (None, "us", None, ".00552805"),
    (None, "global", None, ".0050255"),
    (None, None, None, None),
    ("invalid", "global", None, None),
    ([], "us", None, None),
    ("global", "global", "us", ".00552805"),
])
def test_real_sdk_transport_high_tier_cost_and_event_json(
    model, streaming, actual, wire, final_geo, expected
):
    from amplifier_core.message_models import ChatRequest, Message
    from tests._helpers import FakeCoordinator

    if model == MYTHOS and expected is not None:
        expected = ".0224488" if Decimal(expected) == Decimal(".00552805") else ".020408"
    body = {"id": "msg_offline", "type": "message", "role": "assistant",
            "model": model, "content": [], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 10, "cache_read_input_tokens": 100_000,
                      "cache_creation_input_tokens": 0, "inference_geo": actual}}
    initial = {**body, "stop_reason": None, "usage": {**body["usage"], "output_tokens": 0}}
    delta_usage = {"output_tokens": 10}
    if final_geo is not None:
        delta_usage["inference_geo"] = final_geo
        # Nonstreaming final response directly contains the same final fact.
        body["usage"]["inference_geo"] = final_geo
    events = [
        {"type": "message_start", "message": initial},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": delta_usage},
        {"type": "message_stop"},
    ]
    sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)

    async def run():
        p = AnthropicProvider(api_key="offline-placeholder", config={
            "default_model": model, "use_streaming": streaming, "max_retries": 0,
            "enable_prompt_caching": False,
            "extra_request_params": {"inference_geo": wire} if wire is not None else {},
        }, coordinator=FakeCoordinator())
        p._client = AsyncAnthropic(api_key="offline-placeholder", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, **(
                    {"headers": {"content-type": "text/event-stream"}, "content": sse.encode()}
                    if streaming else {"json": body})))))
        try:
            result = await p.complete(ChatRequest(messages=[Message(role="user", content="hello")]))
            assert result.usage.cost_usd == (Decimal(expected) if expected is not None else None)
            event = p.coordinator.hooks.payload_for("llm:response")
            event_cost = json.loads(json.dumps(event))["usage"]["cost_usd"]
            if expected is None:
                assert event_cost is None
                assert result.metadata["anthropic_cost_unavailable"] == "inference_geo_unavailable"
                assert event["metadata"]["anthropic_cost_unavailable"] == "inference_geo_unavailable"
            else:
                assert Decimal(event_cost) == Decimal(expected)
                assert "anthropic_cost_unavailable" not in (result.metadata or {})
        finally:
            await p.close()
    asyncio.run(run())


@pytest.mark.parametrize("encoding", ["gzip", "deflate"])
@pytest.mark.parametrize("requested,observed,reason", [
    (FABLE, MYTHOS, "invalid_usage_counts"),
    (MYTHOS, MYTHOS + "-latest", "model_rates_unavailable"),
    (MYTHOS, FABLE, None),
])
def test_compressed_cache_guard_uses_exact_observed_model(encoding, requested, observed, reason):
    from amplifier_core.message_models import ChatRequest, Message

    body = {
        "id": "msg_offline", "type": "message", "role": "assistant",
        "model": observed, "content": [], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, "inference_geo": "global"},
    }

    async def run():
        p = AnthropicProvider(api_key="offline-placeholder", config={
            "default_model": requested, "use_streaming": True,
            "enable_prompt_caching": False, "max_retries": 0,
        })
        p._client = AsyncAnthropic(
            api_key="offline-placeholder", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: _cache_evidence_transport(
                    body, "delta_then_valid", {"cache_read_input_tokens": False}, encoding))),
        )
        try:
            result = await p.complete(ChatRequest(messages=[Message(role="user", content="hello")]))
            assert (result.metadata or {}).get("anthropic_cost_unavailable") == reason
            assert result.usage.cost_usd == (None if reason else Decimal("0"))
        finally:
            await p._client.close()

    asyncio.run(run())


@pytest.mark.parametrize("actual,requested,expected", [
    ("us", "global", "us"), ("global", "us", "global"),
    (None, "us", "us"), (None, "global", "global"),
    (None, None, None), ("", "us", None), ([], "global", None),
    ("future", "us", None), (None, {}, None),
])
@pytest.mark.parametrize("model", [FABLE, MYTHOS])
def test_geography_authority_and_absence(model, actual, requested, expected):
    assert resolve_inference_geo(model, response_geo=actual, request_geo=requested) == expected
    assert resolve_inference_geo("claude-sonnet-4-5-20250929") == "global"
    # Standalone helper keeps the documented legacy default, unlike runtime.
    assert compute_cost(FABLE, input_tokens=1000) == Decimal(".008")


def test_actual_geo_precedes_expert_wire_and_stacks_confirmed_fast():
    p = provider()
    r = response(FABLE, input_tokens=1000, output_tokens=0,
                 speed="fast", inference_geo="global")
    params = {"inference_geo": "global", "extra_body": {"inference_geo": "us", "speed": "fast"}}
    assert p._convert_to_chat_response(r, request_params=params).usage.cost_usd == Decimal(".016")
    r.usage.inference_geo = "us"
    assert p._convert_to_chat_response(r, request_params={}).usage.cost_usd == Decimal(".0176")
    del r.usage.inference_geo
    assert p._convert_to_chat_response(r, request_params=params).usage.cost_usd == Decimal(".0176")


@pytest.mark.parametrize("model", [FABLE, MYTHOS])
def test_missing_geo_keeps_session_total_incomplete_after_known_cost(model):
    from amplifier_module_provider_anthropic import mount
    from amplifier_core import ModuleCoordinator

    async def run():
        coordinator = ModuleCoordinator()
        cleanup = await mount(coordinator, {"api_key": "offline-placeholder"})
        p = coordinator.get("providers", "anthropic")
        try:
            # No transport: exact response conversion exercises the registered
            # contribution without reading any account/workspace configuration.
            known = response(model, inference_geo="global")
            assert p._convert_to_chat_response(known, request_params={}).usage.cost_usd is not None
            unknown = response(model)
            result = p._convert_to_chat_response(unknown, request_params={})
            assert result.usage.cost_usd is None
            assert result.metadata["anthropic_cost_unavailable"] == "inference_geo_unavailable"
            p._convert_to_chat_response(known, request_params={})
            assert await coordinator.collect_contributions("session.cost") == [{"cost_usd": None}]
        finally:
            await cleanup()
    asyncio.run(run())


@pytest.mark.parametrize("speed", [[], {}, 0, False, "", "future"])
@pytest.mark.parametrize("model", [FABLE, MYTHOS])
def test_converter_invalid_actual_speed_without_fast_request_keeps_unknown(model, speed):
    from amplifier_module_provider_anthropic import mount
    from amplifier_core import ModuleCoordinator

    async def run():
        coordinator = ModuleCoordinator()
        cleanup = await mount(coordinator, {"api_key": "offline-placeholder"})
        p = coordinator.get("providers", "anthropic")
        body = {
            "id": "msg_offline", "type": "message", "role": "assistant",
            "model": model, "content": [], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 0,
                      "inference_geo": "global", "speed": speed},
        }
        try:
            # Actual SDK construction retains malformed/unknown response
            # fields. No requested fast mode exists to trigger another guard.
            result = p._convert_to_chat_response(
                construct_type(value=body, type_=SDKMessage), request_params={})
            assert result.usage.cost_usd is None
            assert result.metadata["anthropic_cost_unavailable"] == "response_speed_unavailable"
            assert json.loads(result.model_dump_json())["usage"]["cost_usd"] is None
            # None is the documented optional absence and still allows normal
            # standard pricing; it must not erase the previous unknown call.
            body["usage"]["speed"] = None
            standard = p._convert_to_chat_response(
                construct_type(value=body, type_=SDKMessage), request_params={})
            assert standard.usage.cost_usd == Decimal(".008")
            assert "anthropic_cost_unavailable" not in (standard.metadata or {})
            assert await coordinator.collect_contributions("session.cost") == [{"cost_usd": None}]
        finally:
            await cleanup()
    asyncio.run(run())


def test_mythos_exact_rates_metadata_is_fresh_and_standard_only():
    rates = get_rates(MYTHOS)
    assert rates == {
        "input_per_m": Decimal("8"), "output_per_m": Decimal("40"),
        "cache_read_per_m": Decimal(".20"), "cache_write_per_m": Decimal("10"),
        "cache_write_1h_per_m": Decimal("16"),
    }
    details = pricing_metadata(MYTHOS)
    assert details["rates"] == {
        "input": "8.00", "output": "40.00", "cache_read": "0.20",
        "cache_write_5m": "10.00", "cache_write_1h": "16.00",
    }
    assert details["modifiers"] == {"us_inference": "1.1"}
    details["rates"]["input"] = "999"
    rates["input_per_m"] = Decimal("999")
    assert pricing_metadata(MYTHOS)["rates"]["input"] == "8.00"
    assert get_rates(MYTHOS)["input_per_m"] == Decimal("8")
    assert not provider()._get_capabilities(MYTHOS).supports_speed
    result = provider()._convert_to_chat_response(
        response(MYTHOS, speed="fast", inference_geo="global"), request_params={})
    assert result.usage.cost_usd is None
    assert result.metadata["anthropic_cost_unavailable"] == "fast_mode_rates_unavailable"


@pytest.mark.parametrize("requested,observed,expected", [
    (FABLE, MYTHOS, Decimal(".0012")),
    (MYTHOS, MYTHOS + "-latest", None),
])
def test_observed_model_controls_accounting_without_relabelling(requested, observed, expected):
    raw = response(observed, inference_geo="global")
    result = provider()._convert_to_chat_response(
        raw, request_params={"model": requested})
    assert result.usage.cost_usd == expected
    assert raw.model == observed
    if expected is None:
        assert result.metadata["anthropic_cost_unavailable"] == "model_rates_unavailable"


@pytest.mark.parametrize("counts,reason", [
    *[({"cache_read_input_tokens": value}, "invalid_usage_counts")
      for value in [False, "", [], True, -1, "1", 1.5]],
    *[({"cache_creation_input_tokens": value}, "invalid_usage_counts")
      for value in [False, True, "", [], -1, "10"]],
    *[({"cache_creation_input_tokens": 10, "cache_creation": split}, reason)
      for split, reason in [
          ("bad", "cache_write_ttl_split_incomplete"),
          ([], "cache_write_ttl_split_incomplete"),
          ({}, "cache_write_ttl_split_incomplete"),
          ({"ephemeral_5m_input_tokens": 10}, "cache_write_ttl_split_incomplete"),
          ({"ephemeral_5m_input_tokens": "bad", "ephemeral_1h_input_tokens": "bad"},
           "invalid_usage_counts"),
          *[({"ephemeral_5m_input_tokens": value, "ephemeral_1h_input_tokens": 0},
             "invalid_usage_counts") for value in [False, True, "", [], -1, "10"]],
          ({"ephemeral_5m_input_tokens": None, "ephemeral_1h_input_tokens": 0},
           "cache_write_ttl_split_incomplete"),
          ({"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": False},
           "invalid_usage_counts"),
          ({"ephemeral_5m_input_tokens": "10", "ephemeral_1h_input_tokens": False},
           "invalid_usage_counts"),
          ({"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": "10"},
           "invalid_usage_counts"),
      ]],
])
@pytest.mark.parametrize("wire_mode,encoding", [
    ("json", None),
    *[(mode, encoding) for mode in ("start", "delta", "delta_then_valid")
      for encoding in (None, "gzip", "deflate", "deflate-raw", "gzip, deflate")],
])
def test_mythos_raw_cache_evidence_transport_unknown_session(counts, reason, wire_mode, encoding):
    from amplifier_core import ModuleCoordinator
    from amplifier_core.message_models import ChatRequest, Message
    from amplifier_module_provider_anthropic import mount
    from tests._helpers import FakeCoordinator

    body = {
        "id": "msg_offline", "type": "message", "role": "assistant",
        "model": MYTHOS, "content": [], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0, "inference_geo": "global", **counts},
    }

    async def run():
        coordinator = ModuleCoordinator()
        cleanup = await mount(coordinator, {
            "api_key": "offline-placeholder", "default_model": MYTHOS,
            "use_streaming": wire_mode != "json", "enable_prompt_caching": False, "max_retries": 0,
            # Malformed split must NOT fall back to this unambiguous TTL.
            "extra_request_params": {"system": [{
                "type": "text", "text": "prefix",
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            }]},
        })
        p = coordinator.get("providers", "anthropic")
        p.coordinator = FakeCoordinator()
        hooks = p.coordinator.hooks
        p._client = AsyncAnthropic(
            api_key="offline-placeholder", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: _cache_evidence_transport(body, wire_mode, counts, encoding))),
        )
        try:
            known = response(MYTHOS, inference_geo="global")
            assert p._convert_to_chat_response(known, request_params={}).usage.cost_usd > 0
            result = await p.complete(ChatRequest(messages=[Message(role="user", content="hello")]))
            assert result.usage.cost_usd is None
            assert result.metadata["anthropic_cost_unavailable"] == reason
            assert json.loads(result.model_dump_json())["usage"]["cost_usd"] is None
            event = hooks.payload_for("llm:response")
            assert json.loads(json.dumps(event))["usage"]["cost_usd"] is None
            assert event["metadata"]["anthropic_cost_unavailable"] == reason
            p._convert_to_chat_response(known, request_params={})
            assert await coordinator.collect_contributions("session.cost") == [{"cost_usd": None}]
        finally:
            await cleanup()

    asyncio.run(run())


def _cache_evidence_transport(body, wire_mode, counts, encoding=None, delta_usage=None):
    if wire_mode == "json":
        return httpx.Response(200, json=body)
    start = json.loads(json.dumps(body))
    start["stop_reason"] = None
    if wire_mode != "start":
        start["usage"] = {
            "input_tokens": 0, "output_tokens": 0, "inference_geo": "global",
        }
    events = [
        {"type": "message_start", "message": start},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": delta_usage if delta_usage is not None else {
             "output_tokens": 0, **(counts if wire_mode != "start" else {})}},
    ]
    if wire_mode == "delta_then_valid":
        # Later valid/absent fields must not launder earlier malformed evidence.
        events.append({
            "type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0, "cache_creation": None},
        })
    events.append({"type": "message_stop"})
    wire = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode("utf-8")
    headers = {"content-type": "text/event-stream"}
    if encoding:
        headers["content-encoding"] = encoding.replace("deflate-raw", "deflate")
        for coding in encoding.split(", "):
            if coding == "gzip":
                wire = gzip.compress(wire, mtime=0)
            elif coding == "deflate":
                wire = zlib.compress(wire)
            elif coding == "deflate-raw":
                compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
                wire = compressor.compress(wire) + compressor.flush()
            else:
                raise AssertionError(f"Unsupported fixture encoding: {coding}")

    class FragmentedSSE(httpx.AsyncByteStream):
        async def __aiter__(self):
            for index in range(0, len(wire), 17):
                yield wire[index:index + 17]

    return httpx.Response(200, headers=headers, stream=FragmentedSSE())


@pytest.mark.parametrize("counts,expected", [
    ({}, "0"),
    ({"cache_read_input_tokens": None, "cache_creation": None}, "0"),
    ({"cache_creation_input_tokens": 10}, ".0001"),
    ({"cache_creation_input_tokens": 10, "cache_creation": {
        "ephemeral_5m_input_tokens": 4, "ephemeral_1h_input_tokens": 6}}, ".000136"),
    ({"cache_creation": {
        "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}}, "0"),
])
def test_mythos_valid_cache_evidence_remains_known(counts, expected):
    body = {
        "id": "msg_offline", "type": "message", "role": "assistant",
        "model": MYTHOS, "content": [], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0, "inference_geo": "global", **counts},
    }
    result = provider()._convert_to_chat_response(
        construct_type(value=body, type_=SDKMessage), request_params={
            "system": [{"cache_control": {"type": "ephemeral", "ttl": "5m"}}],
        })
    assert result.usage.cost_usd == Decimal(expected)
    assert "anthropic_cost_unavailable" not in (result.metadata or {})


@pytest.mark.parametrize("wire_mode,encoding", [
    ("json", None),
    *[(mode, encoding) for mode in ("start", "delta")
      for encoding in (None, "gzip", "deflate", "deflate-raw", "gzip, deflate")],
])
@pytest.mark.parametrize("requested,observed", [(MYTHOS, MYTHOS), (FABLE, MYTHOS)])
@pytest.mark.parametrize("counts,expected", [
    ({}, "0"),
    ({"cache_read_input_tokens": None, "cache_creation": None}, "0"),
    ({"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}, "0"),
    ({"cache_creation_input_tokens": 10}, ".0001"),
    ({"cache_creation_input_tokens": 10, "cache_creation": {
        "ephemeral_5m_input_tokens": 4, "ephemeral_1h_input_tokens": 6}}, ".000136"),
])
def test_mythos_valid_cache_transport_boundaries(wire_mode, encoding, requested, observed, counts, expected):
    from amplifier_core import ModuleCoordinator
    from amplifier_core.message_models import ChatRequest, Message
    from amplifier_module_provider_anthropic import mount
    from tests._helpers import FakeCoordinator

    body = {
        "id": "msg_offline", "type": "message", "role": "assistant",
        "model": observed, "content": [], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0, "inference_geo": "global", **counts},
    }

    async def run():
        coordinator = ModuleCoordinator()
        cleanup = await mount(coordinator, {
            "api_key": "offline-placeholder", "default_model": requested,
            "use_streaming": wire_mode != "json", "enable_prompt_caching": False, "max_retries": 0,
            "extra_request_params": {"system": [{
                "type": "text", "text": "prefix",
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            }]},
        })
        p = coordinator.get("providers", "anthropic")
        p.coordinator = FakeCoordinator()
        p._client = AsyncAnthropic(
            api_key="offline-placeholder", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: _cache_evidence_transport(body, wire_mode, counts, encoding))),
        )
        try:
            result = await p.complete(ChatRequest(messages=[Message(role="user", content="hello")]))
            assert result.usage.cost_usd == Decimal(expected)
            assert "anthropic_cost_unavailable" not in (result.metadata or {})
            event = p.coordinator.hooks.payload_for("llm:response")
            assert Decimal(json.loads(result.model_dump_json())["usage"]["cost_usd"]) == Decimal(expected)
            assert Decimal(event["usage"]["cost_usd"]) == Decimal(expected)
            contributions = await coordinator.collect_contributions("session.cost")
            assert len(contributions) == 1
            assert Decimal(contributions[0]["cost_usd"]) == Decimal(expected)
        finally:
            await cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("delta_counts", [
    {},
    {"cache_read_input_tokens": None},
    {"cache_creation_input_tokens": None},
    {"cache_creation": None},
    {"cache_read_input_tokens": None, "cache_creation_input_tokens": None, "cache_creation": None},
])
@pytest.mark.parametrize("start_counts,expected_write,expected_cost", [
    ({"cache_creation_input_tokens": 0}, None, ".0014"),
    ({"cache_creation_input_tokens": 10}, 10, ".0015"),
    ({"cache_creation_input_tokens": 10, "cache_creation": {
        "ephemeral_5m_input_tokens": 4, "ephemeral_1h_input_tokens": 6}}, 10, ".001536"),
])
def test_mythos_stream_optional_delta_preserves_measured_cache(
    delta_counts, start_counts, expected_write, expected_cost
):
    from amplifier_core import ModuleCoordinator
    from amplifier_core.message_models import ChatRequest, Message

    from amplifier_module_provider_anthropic import mount
    from tests._helpers import FakeCoordinator

    body = {
        "id": "msg_cache_preserved", "type": "message", "role": "assistant",
        "model": MYTHOS, "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 0, "inference_geo": "global",
                  "cache_read_input_tokens": 1000, **start_counts},
    }

    async def run():
        coordinator = ModuleCoordinator()
        cleanup = await mount(coordinator, {
            "api_key": "offline-placeholder", "default_model": MYTHOS,
            "use_streaming": True, "enable_prompt_caching": False, "max_retries": 0,
            "extra_request_params": {"system": [{
                "type": "text", "text": "prefix",
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            }]},
        })
        p = coordinator.get("providers", "anthropic")
        p.coordinator = FakeCoordinator()
        p._client = AsyncAnthropic(
            api_key="offline-placeholder", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: _cache_evidence_transport(
                    body, "start", {}, delta_usage={"output_tokens": 10, **delta_counts}))),
        )
        try:
            result = await p.complete(ChatRequest(messages=[Message(role="user", content="hello")]))
            event = p.coordinator.hooks.payload_for("llm:response")
            for usage in (json.loads(result.model_dump_json())["usage"], event["usage"]):
                assert usage["input_tokens"] == 1100
                assert usage["output_tokens"] == 10
                assert usage["cache_read_tokens"] == 1000
                assert usage.get("cache_write_tokens") == expected_write
                assert Decimal(usage["cost_usd"]) == Decimal(expected_cost)
            assert "anthropic_cost_unavailable" not in (result.metadata or {})
            contributions = await coordinator.collect_contributions("session.cost")
            assert len(contributions) == 1
            assert Decimal(contributions[0]["cost_usd"]) == Decimal(expected_cost)
        finally:
            await cleanup()

    asyncio.run(run())
