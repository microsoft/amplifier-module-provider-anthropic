"""Retry behavior tests for list_models().

Verifies that list_models() uses the same shared retry_with_backoff()/
_retry_config machinery as complete(): transient failures (5xx) are
retried with backoff, non-retryable failures (401) raise immediately,
and persistent transient failures raise the translated kernel error
once retries are exhausted.

See test_retry.py for the equivalent tests on the complete() path --
this file mirrors that call shape for list_models().
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import AuthenticationError as KernelAuthenticationError
from amplifier_core.llm_errors import (
    ProviderUnavailableError as KernelProviderUnavailableError,
)

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import FakeCoordinator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(
    max_retries: int = 3, max_retry_delay: float = 60.0
) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": max_retries,
            "min_retry_delay": 0.01,  # Fast for tests
            "max_retry_delay": max_retry_delay,
            "retry_jitter": False,  # Deterministic for tests
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _fake_models_response(model_ids: list[str]) -> SimpleNamespace:
    """Create a fake Anthropic models.list() response."""
    data = [
        SimpleNamespace(id=mid, display_name=mid, created_at="2026-01-01")
        for mid in model_ids
    ]
    return SimpleNamespace(data=data)


def _make_sdk_server_error() -> anthropic.InternalServerError:
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.headers = {}
    return anthropic.InternalServerError(
        "server error", response=mock_response, body=None
    )


def _make_sdk_auth_error() -> anthropic.AuthenticationError:
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}
    return anthropic.AuthenticationError("bad key", response=mock_response, body=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_models_succeeds_first_try():
    """No transient failure: exactly one API call, result unchanged."""
    provider = _make_provider()
    response = _fake_models_response(["claude-sonnet-4-5-20250929"])
    provider.client.models.list = AsyncMock(return_value=response)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        models = asyncio.run(provider.list_models())

    assert provider.client.models.list.await_count == 1
    mock_sleep.assert_not_awaited()
    assert len(models) == 1
    assert models[0].id == "claude-sonnet-4-5-20250929"


def test_list_models_recovers_from_transient_500():
    """A single transient 500 is retried, then the call succeeds."""
    provider = _make_provider()
    response = _fake_models_response(["claude-sonnet-4-5-20250929"])
    provider.client.models.list = AsyncMock(
        side_effect=[_make_sdk_server_error(), response]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        models = asyncio.run(provider.list_models())

    assert provider.client.models.list.await_count == 2
    assert len(models) == 1
    assert models[0].id == "claude-sonnet-4-5-20250929"


def test_list_models_raises_after_retries_exhausted():
    """Persistent transient failure raises the kernel error after retries."""
    provider = _make_provider(max_retries=2)
    provider.client.models.list = AsyncMock(side_effect=_make_sdk_server_error())

    with (
        patch("asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(KernelProviderUnavailableError),
    ):
        asyncio.run(provider.list_models())

    # 1 initial + 2 retries = 3 total attempts
    assert provider.client.models.list.await_count == 3


def test_list_models_non_retryable_error_raised_immediately():
    """A non-retryable error (401) raises immediately without retrying."""
    provider = _make_provider()
    provider.client.models.list = AsyncMock(side_effect=_make_sdk_auth_error())

    with (
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        pytest.raises(KernelAuthenticationError),
    ):
        asyncio.run(provider.list_models())

    assert provider.client.models.list.await_count == 1
    mock_sleep.assert_not_awaited()
