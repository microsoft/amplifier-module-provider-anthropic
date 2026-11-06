"""Canonical selectors and safety, with real SDK HTTP behavior and no network."""

import asyncio
from copy import deepcopy
from dataclasses import FrozenInstanceError
import inspect
import traceback

import httpx2
import pytest
from anthropic.resources.models import AsyncModels
from amplifier_core import ModelInfo
from amplifier_core.llm_errors import LLMError
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._model_listing import (
    DEFAULT_FILTERS, ModelFilters, model_identity, split_snapshot,
)
from tests.test_model_discovery import _page, wire  # noqa: F401


# Same common-filter contract as the reviewed OpenAI selector, with Anthropic
# compact-date coverage added. Keep fixtures explicit and provider-local.
@pytest.mark.parametrize("filters,expected", [
    ({}, ["Claude-Sonnet-5-5", "claude-sonnet-5-5-20261001", "older", "x/y"]),
    ({"include": []}, []),
    ({"include": ["CLAUDE-*"]}, ["Claude-Sonnet-5-5", "claude-sonnet-5-5-20261001"]),
    ({"include": ["*"], "exclude": ["CLAUDE-*", "x?y"]}, ["older"]),
    ({"include": ["older"], "exclude": ["OLDER"]}, []),
    ({"hide_dated": True}, ["Claude-Sonnet-5-5", "older", "x/y"]),
    ({"hide_dated": True, "exclude": ["Claude-Sonnet-5-5"]},
     ["claude-sonnet-5-5-20261001", "older", "x/y"]),
])
def test_common_filter_vectors(filters, expected):
    ids = ["Claude-Sonnet-5-5", "claude-sonnet-5-5-20261001", "older", "x/y"]
    assert ModelFilters.parse(filters).select(ids) == expected


@pytest.mark.parametrize("filters", [
    None, [], "*", {"bogus": True}, {"include": "*"}, {"exclude": [None]},
    {"include": [""]}, {"include": ["  "]}, {"hide_dated": 1},
    {"latest_per_family": "true"}, {"lifecycle": []}, {"lifecycle": "active"},
    {"lifecycle": ["ACTIVE"]}, {"lifecycle": ["active", "active"]},
    {"lifecycle": ["unknown"]},
])
def test_invalid_filters_fail_closed(filters):
    with pytest.raises((TypeError, ValueError)):
        ModelFilters.parse(filters)
    with pytest.raises((TypeError, ValueError)):
        AnthropicProvider(config={"model_filters": filters})


def test_filter_snapshot_and_fresh_metadata():
    values = {"include": ["claude-haiku-*"]}
    provider = AnthropicProvider(config={"model_filters": values, "filtered": True})
    values["include"].append("*")
    info = provider.get_model_list_info()
    assert set(info) == {
        "version", "source", "filters_keyword", "filters_schema",
        "default_filters", "unfiltered", "override",
    }
    assert info["source"] == "provider-api"
    assert info["override"] == "replace"
    assert info["filters_keyword"] == "filters"
    assert info["default_filters"]["include"] == ["claude-haiku-*"]
    before = deepcopy(info)
    info["default_filters"]["include"].append("*")
    info["filters_schema"]["properties"].clear()
    assert provider.get_model_list_info() == before
    assert provider._client is None
    with pytest.raises(FrozenInstanceError):
        provider._model_filters.hide_dated = True
    assert AnthropicProvider(config={"filtered": False})._model_filters == ModelFilters()
    assert AnthropicProvider(config={"filtered": True, "model_filters": {}})._model_filters == ModelFilters()


def test_numeric_latest_aliases_snapshots_and_distinct_newer_preview():
    ids = [
        "claude-opus-5-9", "claude-opus-5-10-20261101", "claude-opus-5-10",
        "claude-opus-6-preview-20261103", "claude-opus-6-preview",
        "claude-opus-5-10-preview", "claude-sonnet-5-5",
        "claude-haiku-5-5-2026-10-01", "claude-haiku-5-5-20261101",
        "claude-fable-5-1", "claude-unknown-99", "claude-mythos-7",
    ]
    assert DEFAULT_FILTERS.select(ids) == sorted([
        "claude-opus-5-10", "claude-opus-6-preview", "claude-sonnet-5-5",
        "claude-haiku-5-5-20261101", "claude-fable-5-1",
    ])
    assert ModelFilters().select(ids) == sorted(ids)
    assert DEFAULT_FILTERS.select(["claude-sonnet-6-preview"]) == ["claude-sonnet-6-preview"]


@pytest.mark.parametrize("ids,expected", [
    (["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-latest"],
     ["claude-3-5-sonnet-latest"]),
    (["claude-opus-6-20261101", "claude-opus-6"], ["claude-opus-6"]),
    (["claude-haiku-5-5-20260101", "claude-haiku-5-5-20261101"],
     ["claude-haiku-5-5-20261101"]),
    (["claude-fable-5-0", "claude-fable-5-1"], ["claude-fable-5-1"]),
    (["claude-opus-5-5-preview", "claude-opus-5-5"], ["claude-opus-5-5"]),
    (["claude-sonnet-5-5", "claude-sonnet-5-5-unknown"], ["claude-sonnet-5-5"]),
])
def test_latest_version_vectors(ids, expected):
    assert DEFAULT_FILTERS.select(ids) == expected


@pytest.mark.parametrize("model_id", [
    "claude-sonnet-5-5", "claude-sonnet-5-10", "claude-opus-6",
    "claude-sonnet-5-5-20260230", "claude-sonnet-5-5-2026-13-01",
])
def test_not_a_snapshot_date(model_id):
    assert split_snapshot(model_id) == (model_id, "")


@pytest.mark.asyncio
async def test_default_and_override_no_hidden_policy(wire):
    ids = ["claude-sonnet-5-5", "claude-sonnet-4-5-20250929", "private-route"]
    wire.responses.extend([_page(*ids) for _ in range(6)])
    provider = wire.provider()
    assert [m.id for m in await provider.list_models()] == ["claude-sonnet-5-5"]
    assert [m.id for m in await provider.list_models(filters=None)] == ["claude-sonnet-5-5"]
    assert {m.id for m in await provider.list_models(filters={})} == set(ids)
    assert [m.id for m in await provider.list_models(filters={"include": ["*4-5*"]})] == [ids[1]]
    assert await provider.list_models(filters={"include": []}) == []
    assert await provider.list_models(filters={"include": ["*"], "exclude": ["*"]}) == []
    assert all(r.url.params["limit"] == "1000" for r in wire.requests)
    assert wire.requests[0].url.params.get_list("lifecycle[]") == ["active"]
    assert wire.requests[2].url.params.get_list("lifecycle[]") == ["active", "deprecated", "retired"]
    assert not hasattr(provider, "discover_models")


@pytest.mark.asyncio
async def test_snapshot_before_await_and_independent_concurrent_filters(wire):
    provider = wire.provider()
    filters = {"include": ["claude-haiku-*"]}
    original = provider._fetch_models

    async def fetch(retry_config, *, lifecycle):
        filters["include"].append("*")
        await asyncio.sleep(0)
        return await original(retry_config, lifecycle=lifecycle)

    provider._fetch_models = fetch
    ids = ["claude-sonnet-5-5", "claude-haiku-5-5"]
    wire.responses.extend([_page(*ids), _page(*ids)])
    one, two = await asyncio.gather(
        provider.list_models(filters=filters),
        provider.list_models(filters={"include": ["claude-sonnet-*"]}),
    )
    assert [m.id for m in one] == ["claude-haiku-5-5"]
    assert [m.id for m in two] == ["claude-sonnet-5-5"]


@pytest.mark.asyncio
async def test_invalid_call_filters_before_network(wire):
    with pytest.raises(ValueError):
        await wire.provider().list_models(filters={"include": [None]})
    assert not wire.requests


@pytest.mark.asyncio
async def test_unknown_model_metadata_does_not_invent_sonnet(wire):
    page = _page("private-route", "other-route")
    page["data"][1].update(max_input_tokens=1234, max_tokens=321)
    wire.responses.append(page)
    models = {m.id: m for m in await wire.provider().list_models(filters={})}
    assert all(isinstance(m, ModelInfo) for m in models.values())
    assert models["private-route"].context_window == 0
    assert models["private-route"].max_output_tokens == 0
    assert models["private-route"].capabilities == []
    assert models["private-route"].defaults == {}
    assert models["other-route"].context_window == 1234
    assert models["other-route"].max_output_tokens == 321


def test_locked_sdk_surface_supports_native_lifecycle_and_limit():
    params = inspect.signature(AsyncModels.list).parameters
    assert {"limit", "after_id", "before_id"} <= params.keys()
    assert "lifecycle" in params and "status" not in params
    from anthropic.types.model_info import ModelInfo as SDKModelInfo
    assert "lifecycle" in SDKModelInfo.model_fields
    assert "status" not in SDKModelInfo.model_fields


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 529])
async def test_listing_failure_text_and_traceback_do_not_expose_body(wire, status):
    wire.responses.append(httpx2.Response(status, json={
        "type": "error", "error": {"type": "api_error", "message": "SECRET-CANARY"},
    }))
    with pytest.raises(LLMError) as caught:
        await wire.provider().list_models(filters={})
    assert "SECRET-CANARY" not in str(caught.value)
    assert "SECRET-CANARY" not in "".join(traceback.format_exception(caught.value))
    assert caught.value.status_code == status


@pytest.mark.asyncio
async def test_cancellation_propagates_without_partial_result_or_cache(wire):
    provider = wire.provider()
    wire.responses.append(_page("claude-opus-6", has_more=True))
    original = provider._list_models_page

    async def page(retry_config, *, after_id=None, lifecycle):
        if after_id:
            raise asyncio.CancelledError()
        return await original(retry_config, after_id=after_id, lifecycle=lifecycle)

    provider._list_models_page = page
    with pytest.raises(asyncio.CancelledError):
        await provider.list_models()
    assert provider._family_latest == {}
    assert len(wire.requests) == 1


@pytest.mark.asyncio
async def test_retry_override_is_honored_and_instance_not_mutated(wire):
    provider = wire.provider(max_retries=2)
    original = provider._retry_config
    wire.responses.append(httpx2.Response(500, json={"error": {"message": "offline"}}))
    with pytest.raises(LLMError):
        await provider.list_models(retry_config=provider._build_retry_config(0))
    assert len(wire.requests) == 1
    assert provider._retry_config is original


@pytest.mark.asyncio
async def test_hidden_explicit_inference_model_params_are_unchanged(wire):
    provider = wire.provider(default_model="claude-haiku-4-5")
    request = ChatRequest(
        messages=[Message(role="user", content="offline")],
    )
    def assemble():
        return provider._assemble_request_params(
            request, request_options={"model": "claude-haiku-4-5"},
            request_caps=provider._get_capabilities("claude-haiku-4-5"),
        ).params
    before = assemble()
    wire.responses.append(_page("claude-haiku-4-5", "claude-haiku-5-5"))
    assert [m.id for m in await provider.list_models()] == ["claude-haiku-5-5"]
    assert assemble() == before
    assert before["model"] == "claude-haiku-4-5"
    assert provider.default_model == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_active_default_native_query_and_unfiltered_lifecycle(wire):
    page = _page("claude-opus-5-5", "claude-opus-6", "claude-opus-7")
    page["data"][1]["lifecycle"] = "deprecated"
    page["data"][2]["lifecycle"] = "retired"
    # Deliberately ignore the server filter: the local boundary must still hold.
    wire.responses.extend([deepcopy(page) for _ in range(3)])
    provider = wire.provider()
    assert [m.id for m in await provider.list_models()] == ["claude-opus-5-5"]
    assert [m.id for m in await provider.list_models(filters={})] == [
        "claude-opus-5-5", "claude-opus-6", "claude-opus-7",
    ]
    assert [m.id for m in await provider.list_models(filters={"lifecycle": ["deprecated"]})] == [
        "claude-opus-6",
    ]
    assert provider._family_latest == {"opus": "claude-opus-5-5"}
    assert [r.url.params.get_list("lifecycle[]") for r in wire.requests] == [
        ["active"], ["active", "deprecated", "retired"], ["deprecated"],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, "ACTIVE", "unknown", 1, [], {}])
async def test_malformed_lifecycle_never_coerced_or_recommended(wire, value):
    page = _page("claude-sonnet-5-5")
    page["data"][0]["lifecycle"] = value
    wire.responses.append(page)
    with pytest.raises(LLMError, match="expected lifecycle"):
        await wire.provider().list_models()


@pytest.mark.asyncio
async def test_lifecycle_is_identical_on_every_native_page(wire):
    wire.responses.extend([
        _page("claude-opus-5", has_more=True), _page("claude-sonnet-5"),
    ])
    await wire.provider().list_models()
    assert [r.url.params.get_list("lifecycle[]") for r in wire.requests] == [
        ["active"], ["active"],
    ]
    assert [r.url.params["limit"] for r in wire.requests] == ["1000", "1000"]