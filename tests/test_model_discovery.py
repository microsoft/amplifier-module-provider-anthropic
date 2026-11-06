"""Strict discovery contracts through the real SDK and in-process HTTP only."""

from collections import deque
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
import pytest_asyncio
from anthropic import AsyncAnthropic
from amplifier_core.llm_errors import (
    AccessDeniedError,
    AuthenticationError,
    LLMError,
    NotFoundError,
    ProviderUnavailableError,
    RateLimitError,
)

import amplifier_module_provider_anthropic as module
from amplifier_module_provider_anthropic import AnthropicProvider


def _page(*ids, has_more=False, lifecycle="active"):
    return {
        "data": [
            {
                "id": model_id,
                "display_name": model_id,
                "type": "model",
                "created_at": "2026-01-01T00:00:00Z",
                "lifecycle": lifecycle,
            }
            for model_id in ids
        ],
        "has_more": has_more,
        "first_id": ids[0] if ids else None,
        "last_id": ids[-1] if ids else None,
    }


def _error(status):
    return httpx2.Response(
        status,
        json={"type": "error", "error": {"type": "api_error", "message": "test error"}},
        headers={"retry-after": "0"},
    )


def _ids(result):
    assert isinstance(result, list)
    return [model.id for model in result]


@pytest_asyncio.fixture
async def wire(monkeypatch):
    """Keep real SDK parsing, pagination, endpoint and authentication behavior."""
    requests = []
    responses = deque()
    clients = []
    constructor_args = []

    def handle(request):
        requests.append(request)
        # A successful inventory cannot hide inference or metadata-retrieve calls.
        assert request.method == "GET"
        assert request.url.path.endswith("/v1/models")
        response = responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, httpx2.Response) else httpx2.Response(
            200, json=response
        )

    def make_client(**kwargs):
        constructor_args.append(kwargs.copy())
        client = AsyncAnthropic(
            **kwargs,
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(handle), trust_env=False
            ),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(module, "AsyncAnthropic", make_client)
    # Tests use only explicit fake credentials and endpoints, never ambient auth.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-offline-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://unused.test")

    def provider(**config):
        return AnthropicProvider(
            api_key=config.pop("api_key", "offline-account-a"),
            config={
                "base_url": "https://inventory.test/account-a/",
                "max_retries": 0,
                "min_retry_delay": 0,
                "max_retry_delay": 0,
                "retry_jitter": False,
                **config,
            },
        )

    yield SimpleNamespace(
        requests=requests,
        responses=responses,
        constructor_args=constructor_args,
        provider=provider,
    )
    for client in clients:
        await client.close()


@pytest.mark.asyncio
async def test_success_preserves_exact_api_ids_without_family_filtering(wire):
    ids = [
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-5-5",
        "account-only-experimental-model",
    ]
    wire.responses.append(_page(*ids))
    provider = wire.provider(filtered=True, default_model="not-in-account")
    assert _ids(await provider.list_models(filters={})) == sorted(ids)
    assert len(wire.requests) == 1
    assert provider._family_latest == {"sonnet": "claude-sonnet-5-5"}
    assert provider.filtered is True


@pytest.mark.asyncio
async def test_empty_inventory_is_genuine_even_with_selection_and_cache(wire):
    wire.responses.append(_page())
    provider = wire.provider(
        default_model="claude-opus-5",
        fallback_models={"sonnet": "configured-not-discovered"},
    )
    provider._family_latest["sonnet"] = "cached-not-discovered"
    assert await provider.list_models(filters={}) == []
    assert len(wire.requests) == 1
    assert provider._family_latest == {"sonnet": "cached-not-discovered"}


@pytest.mark.asyncio
async def test_every_page_uses_the_same_configured_endpoint_and_account(wire):
    wire.responses.extend(
        [
            _page("claude-sonnet-4-5", has_more=True),
            _page("private-model:two", has_more=True),
            _page("private-model:three"),
        ]
    )
    provider = wire.provider(
        base_url="https://proxy.test/team-one/",
        api_key="offline-team-one-key",
    )
    assert _ids(await provider.list_models(filters={})) == [
        "claude-sonnet-4-5",
        "private-model:three",
        "private-model:two",
    ]
    assert len(wire.requests) == 3
    assert [r.url.params.get("after_id") for r in wire.requests] == [
        None,
        "claude-sonnet-4-5",
        "private-model:two",
    ]
    for request in wire.requests:
        assert request.url.host == "proxy.test"
        assert request.url.path == "/team-one/v1/models"
        assert request.headers["x-api-key"] == "offline-team-one-key"
        assert "authorization" not in request.headers
        assert not request.content
    assert wire.constructor_args[0]["base_url"] == "https://proxy.test/team-one/"
    assert wire.constructor_args[0]["max_retries"] == 0


@pytest.mark.asyncio
async def test_distinct_accounts_do_not_share_inventory(wire):
    wire.responses.extend([_page("account-a-only"), _page("account-b-only")])
    first = wire.provider(api_key="offline-a")
    second = wire.provider(
        api_key="offline-b", base_url="https://other.test/account-b/"
    )
    assert _ids(await first.list_models(filters={})) == ["account-a-only"]
    assert _ids(await second.list_models(filters={})) == ["account-b-only"]
    assert [r.headers["x-api-key"] for r in wire.requests] == ["offline-a", "offline-b"]
    assert [r.url.host for r in wire.requests] == ["inventory.test", "other.test"]


@pytest.mark.asyncio
async def test_selection_fields_do_not_change_inventory_or_request(wire):
    ids = ["claude-sonnet-4-5", "claude-sonnet-5-5", "unrecognized-family"]
    wire.responses.extend([_page(*ids), _page(*ids)])
    ordinary = wire.provider()
    selected = wire.provider(
        model="not-an-api-model",
        default_model="another-not-an-api-model",
        reasoning_effort="high",
        priority=1,
        filtered=True,
    )
    config_before = deepcopy(selected.config)
    assert _ids(await ordinary.list_models(filters={})) == ids
    assert _ids(await selected.list_models(filters={})) == ids
    assert wire.requests[0].url == wire.requests[1].url
    assert selected.config == config_before
    assert selected.default_model == "another-not-an-api-model"
    assert selected.priority == 1


@pytest.mark.asyncio
async def test_repeated_discovery_is_fresh_without_reading_or_warming_caches(wire):
    wire.responses.extend([_page("first-live"), _page("second-live"), _page()])
    provider = wire.provider()
    provider._family_latest = {"sonnet": "stale-cached-id"}
    provider._family_latest_attempted = True
    provider._runtime_model_info_cache = {"stale-cached-id": None}
    provider.complete = AsyncMock(side_effect=AssertionError("inference used"))
    assert _ids(await provider.list_models(filters={})) == ["first-live"]
    assert _ids(await provider.list_models(filters={})) == ["second-live"]
    assert _ids(await provider.list_models(filters={})) == []
    assert len(wire.requests) == 3
    assert provider._family_latest == {"sonnet": "stale-cached-id"}
    assert provider._family_latest_attempted is True
    assert provider._runtime_model_info_cache == {"stale-cached-id": None}
    provider.complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,error_type,retryable",
    [
        (400, LLMError, False),
        (401, AuthenticationError, False),
        (403, AccessDeniedError, False),
        (404, NotFoundError, False),
        (429, RateLimitError, True),
        (500, ProviderUnavailableError, True),
        (529, ProviderUnavailableError, True),
    ],
)
async def test_http_failures_raise_kernel_errors_not_empty_success(
    wire, status, error_type, retryable
):
    wire.responses.append(_error(status))
    with pytest.raises(error_type) as caught:
        await wire.provider().list_models(filters={})
    assert caught.value.provider == "anthropic"
    assert caught.value.status_code == status
    assert caught.value.retryable is retryable
    assert len(wire.requests) == 1


@pytest.mark.asyncio
async def test_missing_credentials_fail_before_transport(wire):
    provider = wire.provider(api_key=None)
    with pytest.raises(AuthenticationError):
        await provider.list_models(filters={})
    assert wire.requests == []
    assert wire.constructor_args == []


@pytest.mark.asyncio
async def test_auth_error_is_not_retried(wire):
    wire.responses.append(_error(401))
    with pytest.raises(AuthenticationError):
        await wire.provider(max_retries=2).list_models(filters={})
    assert len(wire.requests) == 1


@pytest.mark.asyncio
async def test_transient_failure_retries_then_returns_actual_success(wire):
    wire.responses.extend([_error(500), _page("actual-retry-result")])
    assert _ids(await wire.provider(max_retries=1).list_models(filters={})) == [
        "actual-retry-result"
    ]
    assert len(wire.requests) == 2


@pytest.mark.asyncio
async def test_exhausted_transient_failure_never_falls_back_to_previous_success(wire):
    wire.responses.extend([_page("previous-live"), _error(500), _error(500)])
    provider = wire.provider(max_retries=1)
    assert _ids(await provider.list_models(filters={})) == ["previous-live"]
    with pytest.raises(ProviderUnavailableError):
        await provider.list_models(filters={})
    assert len(wire.requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,error_type,retries",
    [(401, AuthenticationError, 2), (500, ProviderUnavailableError, 1)],
)
async def test_later_page_failure_cannot_return_partial_inventory(
    wire, status, error_type, retries
):
    wire.responses.append(_page("partial-not-success", has_more=True))
    wire.responses.extend([_error(status)] * (2 if status == 500 else 1))
    provider = wire.provider(max_retries=retries)
    with pytest.raises(error_type):
        await provider.list_models(filters={})
    assert [r.url.params.get("after_id") for r in wire.requests] == [
        None,
        *(["partial-not-success"] * (2 if status == 500 else 1)),
    ]
    assert provider._family_latest == {}


@pytest.mark.asyncio
async def test_later_page_retry_keeps_cursor_and_prior_page(wire):
    wire.responses.extend(
        [_page("first", has_more=True), _error(500), _page("second")]
    )
    assert _ids(await wire.provider(max_retries=1).list_models(filters={})) == [
        "first",
        "second",
    ]
    assert [r.url.params.get("after_id") for r in wire.requests] == [
        None,
        "first",
        "first",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        [],
        False,
        "not-an-object",
        {},
        {"data": [], "has_more": None},
        {"data": [], "has_more": "false"},
        {"data": None, "has_more": False},
        {"data": {}, "has_more": False},
        {"data": [{}], "has_more": False},
        {"data": [{"id": None}], "has_more": False},
        {"data": [{"id": 123}], "has_more": False},
        {"data": [{"id": ""}], "has_more": False},
        {"data": [{"id": "  "}], "has_more": False},
        {"data": [{"id": "valid", "display_name": None}], "has_more": False},
        {"data": [], "has_more": True, "last_id": "not-an-item"},
        {**_page("valid", has_more=True), "last_id": None},
        {**_page("valid"), "last_id": 123},
        {**_page("valid"), "first_id": False},
    ],
)
async def test_malformed_pages_are_errors_not_empty_or_partial_success(wire, body):
    wire.responses.append(body)
    with pytest.raises(LLMError):
        await wire.provider().list_models(filters={})
    assert len(wire.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("has_more", ["false", "true", "0", "1", 0, 1])
@pytest.mark.parametrize("later_page", [False, True])
@pytest.mark.parametrize("nonempty", [False, True])
async def test_raw_pagination_flags_cannot_be_coerced_into_success(
    wire, has_more, later_page, nonempty
):
    if later_page:
        wire.responses.append(_page("first-valid-page", has_more=True))
    body = _page("valid-id") if nonempty else _page()
    body["has_more"] = has_more
    wire.responses.append(body)
    provider = wire.provider(max_retries=2)
    with pytest.raises(LLMError, match="expected data and has_more") as caught:
        await provider.list_models(filters={})
    assert caught.value.retryable is False
    assert len(wire.requests) == (2 if later_page else 1)
    assert provider._family_latest == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["id", "last_id"])
@pytest.mark.parametrize("value", [0, 123, False])
@pytest.mark.parametrize("later_page", [False, True])
async def test_raw_ids_and_cursors_cannot_be_coerced(
    wire, field, value, later_page
):
    if later_page:
        wire.responses.append(_page("first-valid-page", has_more=True))
    body = _page("valid-id", has_more=True)
    if field == "id":
        body["data"][0]["id"] = value
    else:
        body["last_id"] = value
    wire.responses.append(body)
    with pytest.raises(LLMError) as caught:
        await wire.provider(max_retries=2).list_models(filters={})
    assert caught.value.retryable is False
    assert len(wire.requests) == (2 if later_page else 1)


@pytest.mark.asyncio
async def test_invalid_json_is_a_nonretryable_kernel_error(wire):
    wire.responses.append(
        httpx2.Response(
            200, content=b"{invalid", headers={"content-type": "application/json"}
        )
    )
    with pytest.raises(LLMError, match="expected JSON") as caught:
        await wire.provider(max_retries=2).list_models(filters={})
    assert caught.value.retryable is False
    assert len(wire.requests) == 1


@pytest.mark.asyncio
async def test_later_malformed_page_discards_partial_inventory(wire):
    wire.responses.extend([_page("first", has_more=True), {"has_more": False}])
    with pytest.raises(LLMError):
        await wire.provider().list_models(filters={})
    assert len(wire.requests) == 2


@pytest.mark.asyncio
async def test_repeated_pagination_cursor_fails_instead_of_looping(wire):
    wire.responses.extend(
        [_page("same", has_more=True), _page("same", has_more=True)]
    )
    with pytest.raises(LLMError, match="pagination"):
        await wire.provider().list_models(filters={})
    assert len(wire.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [httpx2.ConnectError, httpx2.ReadTimeout])
async def test_transport_failure_is_typed_and_never_becomes_empty(wire, failure):
    wire.responses.append(failure("offline transport failure"))
    with pytest.raises(LLMError) as caught:
        await wire.provider().list_models(filters={})
    assert caught.value.retryable is True
    assert len(wire.requests) == 1


@pytest.mark.asyncio
async def test_canonical_listing_filters_only_after_all_pages_and_warms_cache(wire):
    body = _page("claude-sonnet-4-5", "claude-sonnet-5-5", has_more=True)
    wire.responses.extend([deepcopy(body), _page("claude-sonnet-6"),
                           deepcopy(body), _page("claude-sonnet-6")])
    provider = wire.provider(filtered=True)
    recommended = await provider.list_models()
    assert _ids(recommended) == ["claude-sonnet-6"]
    assert len(wire.requests) == 2
    cache_before = provider._family_latest.copy()
    discovered = await provider.list_models(filters={})
    assert _ids(discovered) == [
        "claude-sonnet-4-5",
        "claude-sonnet-5-5",
        "claude-sonnet-6",
    ]
    assert discovered[2] == recommended[0]
    assert provider._family_latest == cache_before
    assert len(wire.requests) == 4


@pytest.mark.asyncio
async def test_discovery_preserves_bound_workspace_on_every_page(monkeypatch):
    requests = []

    def receive(request):
        requests.append(request)
        assert request.method == "GET"
        assert str(request.url).split("?")[0] == "https://workspace.invalid/v1/models"
        assert request.headers["x-api-key"] == "fixture-workspace-key"
        assert request.headers.get("anthropic-workspace-id") == "wrkspc_A"
        return httpx2.Response(
            200, json=_page("first", has_more=True) if len(requests) == 1 else _page("last")
        )

    async with AsyncAnthropic(
        api_key="fixture-workspace-key", auth_token="",
        base_url="https://workspace.invalid",
        default_headers={"anthropic-workspace-id": "wrkspc_A"},
        max_retries=0,
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(receive), trust_env=False
        ),
    ) as client:
        provider = AnthropicProvider(api_key="unused", config={"max_retries": 0})
        provider._client = client
        before = client.default_headers.copy()
        monkeypatch.setattr(
            module, "AsyncAnthropic",
            lambda **kwargs: pytest.fail("discovery reconstructed the bound client"),
        )
        result = await provider.list_models(filters={})
        assert _ids(result) == ["first", "last"]
        serialized = [item.model_dump(mode="json") for item in result]
        assert [item["id"] for item in json.loads(json.dumps(serialized))] == [
            "first", "last"
        ]
        assert provider.client is client
        assert client.default_headers == before
        assert client.max_retries == 0
        assert [r.url.params.get("after_id") for r in requests] == [None, "first"]


@pytest.mark.asyncio
@pytest.mark.parametrize("later_page", [False, True])
@pytest.mark.parametrize("cursor", ["a", "not-in-page"])
async def test_mismatched_last_id_fails_before_next_request(wire, later_page, cursor):
    if later_page:
        wire.responses.append(_page("previous", has_more=True))
    body = _page("a", "b", has_more=True)
    body["last_id"] = cursor
    wire.responses.extend([body, _page()])
    with pytest.raises(LLMError) as caught:
        await wire.provider().list_models(filters={})
    assert caught.value.provider == "anthropic"
    assert caught.value.retryable is False
    assert len(wire.requests) == (2 if later_page else 1)
    assert list(wire.responses) == [_page()]


@pytest.mark.asyncio
async def test_runtime_metadata_and_context_policy_are_preserved(wire):
    body = _page("claude-haiku-5-5")
    body["data"][0].update(max_input_tokens=1_000_000, max_tokens=128_000)
    wire.responses.extend([deepcopy(body), deepcopy(body)])
    capped = await wire.provider(enable_1m_context=False).list_models(filters={})
    expanded = await wire.provider(enable_1m_context=True).list_models(filters={})
    assert _ids(capped) == _ids(expanded) == ["claude-haiku-5-5"]
    assert capped[0].context_window == 200_000
    assert expanded[0].context_window == 1_000_000
    assert capped[0].max_output_tokens == 128_000