"""Tests for Cloudflare 403 detection and retry behavior.

Cloudflare interposes HTML challenge pages (HTTP 403) in front of
api.anthropic.com when its bot-detection is triggered.  These look nothing
like real Anthropic API 403s (which return JSON bodies).  The provider must:

  1. Detect Cloudflare challenges via _is_cloudflare_challenge().
  2. Raise a *retryable* KernelProviderUnavailableError (not AccessDeniedError).
  3. Let retry_with_backoff handle the retry loop automatically.
  4. Still treat real API 403s (JSON body) as non-retryable AccessDeniedError.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import AccessDeniedError as KernelAccessDeniedError
from amplifier_core.llm_errors import (
    ProviderUnavailableError as KernelProviderUnavailableError,
)
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider
from anthropic import APIStatusError as AnthropicAPIStatusError

from tests._helpers import DummyResponse, FakeCoordinator, FakeHooks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLOUDFLARE_HTML = """<!DOCTYPE html>
<html><head><title>Just a moment...</title></head>
<body>
<div id="cf-browser-verification">
  Checking if the site connection is secure
</div>
</body></html>
"""


def _make_api_status_error(
    status_code: int,
    body: Any | None,
    content_type: str = "application/json",
    response_text: str = "",
) -> AnthropicAPIStatusError:
    """Build a fake AnthropicAPIStatusError with controllable fields."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-type": content_type}
    response.text = response_text
    # The SDK sets .body from parsed JSON; None when body isn't JSON-parseable
    error = AnthropicAPIStatusError.__new__(AnthropicAPIStatusError)
    error.status_code = status_code
    error.body = body
    error.response = response
    error.message = f"Error code: {status_code}"
    error.args = (error.message,)
    return error


class MockStreamManager:
    def __init__(self, api_response: DummyResponse):
        self._api_response = api_response
        self.response = SimpleNamespace(headers={})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get_final_message(self):
        return self._api_response



# ============================================================================
# Unit tests: _is_cloudflare_challenge()
# ============================================================================


class TestIsCloudflareChallenge:
    """Unit tests for the Cloudflare detection helper."""

    def test_cloudflare_html_403_detected(self):
        """HTML 403 with Cloudflare markers should be detected."""
        error = _make_api_status_error(
            status_code=403,
            body=None,  # SDK can't parse HTML as JSON
            content_type="text/html",
            response_text=CLOUDFLARE_HTML,
        )
        assert AnthropicProvider._is_cloudflare_challenge(error) is True

    def test_real_api_403_not_detected(self):
        """JSON 403 from the actual API should NOT be detected as Cloudflare."""
        error = _make_api_status_error(
            status_code=403,
            body={
                "type": "error",
                "error": {"type": "forbidden", "message": "Forbidden"},
            },
            content_type="application/json",
        )
        assert AnthropicProvider._is_cloudflare_challenge(error) is False

    def test_html_content_type_without_body(self):
        """text/html content-type with None body is sufficient signal."""
        error = _make_api_status_error(
            status_code=403,
            body=None,
            content_type="text/html; charset=utf-8",
            response_text="<html><body>Access Denied</body></html>",
        )
        assert AnthropicProvider._is_cloudflare_challenge(error) is True

    def test_cloudflare_markers_in_text(self):
        """Cloudflare markers in response text should trigger detection
        even without text/html content-type."""
        error = _make_api_status_error(
            status_code=403,
            body=None,
            content_type="",  # Missing content-type
            response_text="Just a moment... cloudflare verification",
        )
        assert AnthropicProvider._is_cloudflare_challenge(error) is True

    def test_no_response_object(self):
        """Error without a response object should not be detected."""
        error = AnthropicAPIStatusError.__new__(AnthropicAPIStatusError)
        error.status_code = 403
        error.body = None
        error.response = None
        error.message = "Error code: 403"
        error.args = (error.message,)
        assert AnthropicProvider._is_cloudflare_challenge(error) is False

    def test_json_body_takes_precedence(self):
        """Even if response has HTML content-type, a parsed JSON body means
        it's a real API error (SDK successfully parsed the body)."""
        error = _make_api_status_error(
            status_code=403,
            body={"type": "error"},  # SDK parsed this → real API error
            content_type="text/html",  # Unusual but body wins
        )
        assert AnthropicProvider._is_cloudflare_challenge(error) is False


# ============================================================================
# Integration tests: Cloudflare 403 raises retryable error
# ============================================================================


class TestCloudflare403Retry:
    """Verify that Cloudflare 403s are raised as retryable errors
    while real API 403s remain non-retryable."""

    def _make_provider(self) -> tuple[AnthropicProvider, FakeCoordinator]:
        provider = AnthropicProvider(
            api_key="test-key",
            config={"use_streaming": False, "max_retries": 0},
        )
        fake_coordinator = FakeCoordinator()
        provider.coordinator = cast(ModuleCoordinator, fake_coordinator)
        return provider, fake_coordinator

    def test_cloudflare_403_raises_retryable_provider_unavailable(self):
        """Cloudflare 403 should raise KernelProviderUnavailableError(retryable=True)."""
        provider, coordinator = self._make_provider()

        cf_error = _make_api_status_error(
            status_code=403,
            body=None,
            content_type="text/html",
            response_text=CLOUDFLARE_HTML,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=cf_error
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])

        with pytest.raises(KernelProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(request))

        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 403
        assert "Cloudflare" in str(exc_info.value)

    def test_real_api_403_raises_non_retryable_access_denied(self):
        """Real API 403 should raise KernelAccessDeniedError(retryable=False)."""
        provider, coordinator = self._make_provider()

        api_error = _make_api_status_error(
            status_code=403,
            body={
                "type": "error",
                "error": {"type": "forbidden", "message": "Forbidden"},
            },
            content_type="application/json",
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=api_error
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])

        with pytest.raises(KernelAccessDeniedError) as exc_info:
            asyncio.run(provider.complete(request))

        assert exc_info.value.retryable is False
        assert exc_info.value.status_code == 403

    def test_cloudflare_403_is_retried_then_succeeds(self):
        """Cloudflare 403 followed by success should work via retry loop."""
        provider = AnthropicProvider(
            api_key="test-key",
            config={"use_streaming": False, "max_retries": 2},
        )
        fake_coordinator = FakeCoordinator()
        provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

        cf_error = _make_api_status_error(
            status_code=403,
            body=None,
            content_type="text/html",
            response_text=CLOUDFLARE_HTML,
        )

        # First call: Cloudflare 403.  Second call: success.
        raw_ok = MagicMock()
        raw_ok.parse.return_value = DummyResponse()
        raw_ok.headers = {}

        call_count = 0

        async def flaky_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise cf_error
            return raw_ok

        provider.client.messages.with_raw_response.create = flaky_create  # type: ignore[method-assign]

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        response = asyncio.run(provider.complete(request))

        # Should succeed after retry
        assert response is not None
        assert call_count == 2

        # Should have emitted a retry event
        retry_events = [
            e for e in fake_coordinator.hooks.events if "retry" in e[0].lower()
        ]
        assert len(retry_events) >= 1

    def test_cloudflare_403_exhausts_retries(self):
        """Persistent Cloudflare 403 should exhaust retries and raise."""
        provider = AnthropicProvider(
            api_key="test-key",
            config={"use_streaming": False, "max_retries": 1, "min_retry_delay": 0.01},
        )
        fake_coordinator = FakeCoordinator()
        provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

        cf_error = _make_api_status_error(
            status_code=403,
            body=None,
            content_type="text/html",
            response_text=CLOUDFLARE_HTML,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=cf_error
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])

        with pytest.raises(KernelProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(request))

        assert exc_info.value.retryable is True
        assert "Cloudflare" in str(exc_info.value)
