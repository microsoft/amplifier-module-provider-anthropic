"""Guards on the two SDK contracts the mocked suite cannot see.

Both of the breaks these tests exist to catch shipped to `main` while the full
suite stayed green, because the suite never touches the real SDK surface:

  1. `raw_response.parse()` awaitability. The non-streaming path awaits
     `parse()`. That is correct on anthropic 1.x, where
     `messages.with_raw_response` returns `AsyncAPIResponse`, and wrong on 0.x,
     where it returns `LegacyAPIResponse` and `parse()` is synchronous --
     awaiting it raises "object Message can't be used in 'await' expression"
     on every non-streaming call. The suite mocks `raw.parse` directly, so it
     encodes whichever contract the mock was written for and can never
     disagree with the installed SDK.

  2. Unknown keyword arguments. `AsyncMock` accepts any keyword, so a
     parameter the real SDK has removed (`temperature`, dropped from the typed
     Messages surface in 1.0.0) sails through every mocked assertion and fails
     only against the live API.

Neither test makes a network call.
"""

import asyncio
import inspect
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx2
from anthropic import AsyncAnthropic
from anthropic.resources.messages import AsyncMessages

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_anthropic import _TYPED_REQUEST_PARAMS, AnthropicProvider

from tests._helpers import DummyResponse, FakeCoordinator

_MESSAGE_BODY = {
    "id": "msg_contract",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _mock_client() -> AsyncAnthropic:
    """A real AsyncAnthropic wired to an in-process transport -- no network."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=_MESSAGE_BODY)

    return AsyncAnthropic(
        api_key="test-key",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


class TestRawResponseParseContract:
    """The non-streaming path awaits parse(). The installed SDK must agree."""

    def test_parse_is_awaitable_on_installed_sdk(self):
        async def _run():
            client = _mock_client()
            try:
                raw = await client.messages.with_raw_response.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=16,
                    messages=[{"role": "user", "content": "hi"}],
                )
                result = raw.parse()
                assert inspect.isawaitable(result), (
                    f"{type(raw).__name__}.parse() is synchronous on the installed "
                    "anthropic SDK, but the provider's non-streaming path awaits it. "
                    "Awaiting a Message raises "
                    "\"object Message can't be used in 'await' expression\" on every "
                    "non-streaming call."
                )
                message = await result
                assert message.id == "msg_contract"
            finally:
                await client.close()

        asyncio.run(_run())


class TestBuiltParamsBindToSdkSignature:
    """Every key the provider sends must be a parameter the SDK still accepts."""

    def test_params_bind_to_create_and_stream(self):
        provider = AnthropicProvider(
            api_key="test-key",
            config={
                "use_streaming": False,
                "max_retries": 0,
                # Sampling-capable, so `temperature` is actually populated --
                # Opus 4.7+ would skip the branch this test exists to guard.
                "default_model": "claude-sonnet-4-5-20250929",
                "temperature": 0.3,
            },
        )
        provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())

        raw = MagicMock()
        raw.parse = AsyncMock(return_value=DummyResponse())
        raw.headers = {}
        provider.client.messages.with_raw_response.create = AsyncMock(return_value=raw)

        asyncio.run(
            provider.complete(
                ChatRequest(messages=[Message(role="user", content="Hello")])
            )
        )

        _, params = provider.client.messages.with_raw_response.create.call_args
        assert "temperature" in (params.get("extra_body") or {}), (
            "temperature must travel in extra_body -- it is not a typed keyword "
            "argument on anthropic 1.x."
        )

        # The same params dict feeds both call sites, so it must bind to both.
        for method in (AsyncMessages.create, AsyncMessages.stream):
            bound: Any = inspect.signature(method)
            try:
                bound.bind(provider.client.messages, **params)
            except TypeError as exc:  # pragma: no cover - failure path
                raise AssertionError(
                    f"params built by the provider do not bind to "
                    f"AsyncMessages.{method.__name__}() on the installed anthropic "
                    f"SDK: {exc}"
                ) from exc


class TestTypedRequestParamsMatchSdkSignature:
    """T-D15: every name _merge_extra_request_params treats as "typed"
    must actually be a parameter the installed SDK's create() accepts --
    otherwise a config key routed onto the typed surface would raise
    "unexpected keyword argument" and trigger the retry-storm bug
    extra_request_params exists to avoid.
    """

    def test_typed_request_params_subset_of_create_signature(self):
        sdk_params = set(inspect.signature(AsyncMessages.create).parameters)
        # `messages.create()` is a keyword-only method on `self`; drop the
        # non-parameter names inspect always reports (args/kwargs catch-alls
        # are absent here, so this is just documentation of intent).
        missing = _TYPED_REQUEST_PARAMS - sdk_params
        assert not missing, (
            f"_TYPED_REQUEST_PARAMS names not present on the installed SDK's "
            f"AsyncMessages.create() signature: {missing}. Routing one of "
            f"these onto the typed surface would raise 'unexpected keyword "
            f"argument' and trigger the 5x retry-storm bug this allowlist "
            f"exists to prevent."
        )
