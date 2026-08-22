"""The configured timeout must reach the SDK -- at the client AND the request.

Two independent failures came from the same omission, and each needs its own
guard because each is fixed by a different half of the change.

  1. A configured `timeout` was never handed to `AsyncAnthropic`, so the SDK
     used its own default and long single-turn streams died at that default
     even when the operator had asked for more.

  2. Because the client timeout was unset, the SDK's non-streaming guard stayed
     armed:

         if not stream and not is_given(timeout) and
            self._client.timeout == DEFAULT_TIMEOUT:
             timeout = self._client._calculate_nonstreaming_timeout(...)

     which refuses any non-streaming call whose `max_tokens` exceeds 21,333 --
     and `self.max_tokens` defaults to the model's full output ceiling.

Setting the client timeout does NOT disarm (2) on its own: with the default
`timeout` of 600.0 the resulting Timeout is value-equal to the SDK's own
DEFAULT_TIMEOUT, so that comparison stays true. `is_given(timeout)` is checked
first, which is why the non-streaming call passes `timeout=` per request.

No test here makes a network call.
"""

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from anthropic import DEFAULT_TIMEOUT

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import DummyResponse, FakeCoordinator


def _make_provider(**config: Any) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": 0,
            "default_model": "claude-sonnet-4-5-20250929",
            **config,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


class TestClientTimeoutWiring:
    """The client must be built with the configured timeout, not the default."""

    def test_configured_timeout_reaches_the_client(self):
        provider = _make_provider(timeout=3000.0)
        assert provider.client.timeout.read == 3000.0
        assert provider.client.timeout != DEFAULT_TIMEOUT, (
            "A configured timeout that leaves the client at DEFAULT_TIMEOUT was "
            "never handed to the SDK."
        )

    def test_connect_timeout_is_not_stretched(self):
        """A bare float would apply the full timeout to connect as well.

        Passing `timeout=3000.0` directly would set connect=3000.0, so an
        unreachable endpoint would hang for 50 minutes instead of failing in
        5 seconds. The SDK uses connect=5.0 in its own timeout construction.
        """
        provider = _make_provider(timeout=3000.0)
        assert provider.client.timeout.connect == 5.0

    def test_default_timeout_is_ten_minutes(self):
        provider = _make_provider()
        assert provider.timeout == 600.0
        assert provider.client.timeout.read == 600.0


class TestNonStreamingRequestTimeout:
    """The non-streaming call must pass `timeout=` to disarm the SDK guard."""

    def _run_non_streaming(self, provider: AnthropicProvider) -> dict[str, Any]:
        raw = MagicMock()
        raw.parse = AsyncMock(return_value=DummyResponse())
        raw.headers = {}
        provider.client.messages.with_raw_response.create = AsyncMock(return_value=raw)

        asyncio.run(
            provider.complete(
                ChatRequest(messages=[Message(role="user", content="Hello")])
            )
        )

        _, kwargs = provider.client.messages.with_raw_response.create.call_args
        return kwargs

    def test_request_carries_an_explicit_timeout(self):
        provider = _make_provider(timeout=1234.0)
        kwargs = self._run_non_streaming(provider)
        assert kwargs.get("timeout") == 1234.0, (
            "Without an explicit per-request timeout the SDK estimates the "
            "request duration from max_tokens and refuses the call."
        )

    def test_default_config_still_carries_a_timeout(self):
        """The default is where the client-level timeout alone is not enough.

        `timeout=600.0` builds a Timeout value-equal to DEFAULT_TIMEOUT, so the
        guard's client-timeout comparison stays true and only the per-request
        timeout disarms it. This is the common configuration, so it is the case
        most worth pinning.
        """
        provider = _make_provider()
        assert provider.client.timeout == DEFAULT_TIMEOUT, (
            "Precondition for this test: the default config is value-equal to "
            "DEFAULT_TIMEOUT. If the SDK's default changes this assertion is "
            "the signal to re-check whether the per-request timeout is still "
            "load-bearing."
        )
        kwargs = self._run_non_streaming(provider)
        assert kwargs.get("timeout") == 600.0

    def test_large_max_tokens_is_not_refused(self):
        """The reported failure: default max_tokens is above the SDK's cutoff."""
        provider = _make_provider()
        kwargs = self._run_non_streaming(provider)
        assert kwargs["max_tokens"] > 21_333, (
            "Precondition: the default max_tokens must exceed the SDK's "
            "non-streaming cutoff, otherwise this test proves nothing."
        )
        assert kwargs.get("timeout") is not None
