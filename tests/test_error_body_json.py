"""Tests for json.dumps(e.body) in KernelLLMError messages.

Verifies that when Anthropic SDK errors have a .body attribute,
the kernel error message uses json.dumps(body) instead of str(e).
When body is None, str(e) is used as fallback.
"""

import asyncio
import json
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest

from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import (
    AccessDeniedError as KernelAccessDeniedError,
    AuthenticationError as KernelAuthenticationError,
    ContentFilterError as KernelContentFilterError,
    ContextLengthError as KernelContextLengthError,
    InvalidRequestError as KernelInvalidRequestError,
    LLMError as KernelLLMError,
    NotFoundError as KernelNotFoundError,
    ProviderUnavailableError as KernelProviderUnavailableError,
    RateLimitError as KernelRateLimitError,
)
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import FakeCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider() -> AnthropicProvider:
    """Create a provider with streaming disabled and max_retries=0 for isolation."""
    provider = AnthropicProvider(
        api_key="test-key",
        config={"use_streaming": False, "max_retries": 0},
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _make_anthropic_error_with_body(cls, message="error", status_code=400, body=None):
    """Construct an Anthropic SDK error with a body attribute."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    return cls(message, response=mock_response, body=body)


# ---------------------------------------------------------------------------
# Block 1: RateLimitError — uses json.dumps(body) when body present
# ---------------------------------------------------------------------------


class TestRateLimitErrorUsesBodyJson:
    def test_error_message_contains_json_body_when_body_present(self):
        provider = _make_provider()
        body = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "rate limited"},
        }
        sdk_error = _make_anthropic_error_with_body(
            anthropic.RateLimitError, "rate limited", status_code=429, body=body
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelRateLimitError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        # The error message should be the JSON dump of body, not str(e)
        assert json.dumps(body) == str(exc_info.value)

    def test_error_message_falls_back_to_str_when_body_none(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error_with_body(
            anthropic.RateLimitError, "rate limited", status_code=429, body=None
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelRateLimitError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        # Falls back to str(e) when body is None
        assert "rate limited" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Block 2: AuthenticationError — uses json.dumps(body) when body present
# ---------------------------------------------------------------------------


class TestAuthenticationErrorUsesBodyJson:
    def test_error_message_contains_json_body_when_body_present(self):
        provider = _make_provider()
        body = {
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid api key"},
        }
        sdk_error = _make_anthropic_error_with_body(
            anthropic.AuthenticationError, "invalid key", status_code=401, body=body
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelAuthenticationError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert json.dumps(body) == str(exc_info.value)

    def test_error_message_falls_back_to_str_when_body_none(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error_with_body(
            anthropic.AuthenticationError, "invalid key", status_code=401, body=None
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelAuthenticationError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        # Falls back to str(e) when body is None
        assert "invalid key" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Block 3: BadRequestError — uses json.dumps(body) but preserves str(e)
#           for keyword matching
# ---------------------------------------------------------------------------


class TestBadRequestErrorUsesBodyJson:
    def test_context_length_error_uses_json_body(self):
        """Even though keyword matching uses str(e), the raised error message should use body JSON."""
        provider = _make_provider()
        body = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long: context length exceeded",
            },
        }
        sdk_error = _make_anthropic_error_with_body(
            anthropic.BadRequestError,
            "prompt is too long: context length exceeded",
            status_code=400,
            body=body,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContextLengthError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert json.dumps(body) == str(exc_info.value)

    def test_content_filter_error_uses_json_body(self):
        provider = _make_provider()
        body = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "content blocked by safety filter",
            },
        }
        sdk_error = _make_anthropic_error_with_body(
            anthropic.BadRequestError,
            "content blocked by safety filter",
            status_code=400,
            body=body,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContentFilterError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert json.dumps(body) == str(exc_info.value)

    def test_invalid_request_error_uses_json_body(self):
        provider = _make_provider()
        body = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "invalid model name",
            },
        }
        sdk_error = _make_anthropic_error_with_body(
            anthropic.BadRequestError,
            "invalid model name",
            status_code=400,
            body=body,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelInvalidRequestError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert json.dumps(body) == str(exc_info.value)

    def test_context_length_error_falls_back_to_str_when_body_none(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error_with_body(
            anthropic.BadRequestError,
            "prompt is too long: context length exceeded",
            status_code=400,
            body=None,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContextLengthError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert "context length exceeded" in str(exc_info.value)

    def test_content_filter_error_falls_back_to_str_when_body_none(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error_with_body(
            anthropic.BadRequestError,
            "content blocked by safety filter",
            status_code=400,
            body=None,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContentFilterError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert "content blocked by safety filter" in str(exc_info.value)

    def test_invalid_request_error_falls_back_to_str_when_body_none(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error_with_body(
            anthropic.BadRequestError,
            "invalid model name",
            status_code=400,
            body=None,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelInvalidRequestError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert "invalid model name" in str(exc_info.value)

    def test_keyword_matching_still_works_with_body(self):
        """Keyword matching must still use str(e).lower(), not body JSON."""
        provider = _make_provider()
        # Body doesn't contain the keywords, but str(e) does
        body = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "tokens"},
        }
        sdk_error = _make_anthropic_error_with_body(
            anthropic.BadRequestError,
            "too many tokens in request",
            status_code=400,
            body=body,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        # Should still match "too many tokens" from str(e), not from body
        with pytest.raises(KernelContextLengthError):
            asyncio.run(provider.complete(_simple_request()))


# ---------------------------------------------------------------------------
# Block 4: APIStatusError — uses json.dumps(body) when body present
# ---------------------------------------------------------------------------


class TestAPIStatusErrorUsesBodyJson:
    def test_403_access_denied_uses_json_body(self):
        provider = _make_provider()
        body = {"type": "error", "error": {"type": "forbidden", "message": "forbidden"}}
        sdk_error = _make_anthropic_error_with_body(
            anthropic.APIStatusError, "forbidden", status_code=403, body=body
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelAccessDeniedError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert json.dumps(body) == str(exc_info.value)

    def test_404_not_found_uses_json_body(self):
        provider = _make_provider()
        body = {"type": "error", "error": {"type": "not_found", "message": "not found"}}
        sdk_error = _make_anthropic_error_with_body(
            anthropic.APIStatusError, "not found", status_code=404, body=body
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelNotFoundError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert json.dumps(body) == str(exc_info.value)

    def test_5xx_provider_unavailable_uses_json_body(self):
        provider = _make_provider()
        body = {
            "type": "error",
            "error": {"type": "api_error", "message": "internal server error"},
        }
        sdk_error = _make_anthropic_error_with_body(
            anthropic.InternalServerError,
            "internal server error",
            status_code=500,
            body=body,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert json.dumps(body) == str(exc_info.value)

    def test_403_access_denied_falls_back_to_str_when_body_none(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error_with_body(
            anthropic.APIStatusError, "forbidden", status_code=403, body=None
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelAccessDeniedError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert "forbidden" in str(exc_info.value)

    def test_404_not_found_falls_back_to_str_when_body_none(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error_with_body(
            anthropic.APIStatusError, "not found", status_code=404, body=None
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelNotFoundError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert "not found" in str(exc_info.value)

    def test_5xx_provider_unavailable_falls_back_to_str_when_body_none(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error_with_body(
            anthropic.InternalServerError,
            "internal server error",
            status_code=500,
            body=None,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert "internal server error" in str(exc_info.value)

    def test_other_status_falls_back_to_str_when_body_none(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error_with_body(
            anthropic.APIStatusError, "I'm a teapot", status_code=418, body=None
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert "I'm a teapot" in str(exc_info.value)

    def test_other_status_uses_json_body(self):
        provider = _make_provider()
        body = {"type": "error", "error": {"type": "teapot", "message": "I'm a teapot"}}
        sdk_error = _make_anthropic_error_with_body(
            anthropic.APIStatusError, "I'm a teapot", status_code=418, body=body
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert json.dumps(body) == str(exc_info.value)


# ---------------------------------------------------------------------------
# Block 7: Generic Exception catch-all — uses json.dumps(body) when body present
# ---------------------------------------------------------------------------


class TestGenericExceptionUsesBodyJson:
    def test_exception_with_body_uses_json(self):
        """Exceptions with a body attribute should use json.dumps(body)."""
        provider = _make_provider()
        body = {"type": "error", "error": {"message": "unexpected"}}
        original = Exception("something unexpected")
        original.body = body  # type: ignore[attr-defined]
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=original
        )

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert json.dumps(body) == str(exc_info.value)

    def test_exception_without_body_uses_str(self):
        """Exceptions without body should fall back to str(e)."""
        provider = _make_provider()
        original = RuntimeError("something unexpected")
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=original
        )

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert "something unexpected" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Unchanged blocks: verify they are NOT affected
# ---------------------------------------------------------------------------


class TestUnchangedBlocks:
    def test_timeout_error_message_unchanged(self):
        """asyncio.TimeoutError still uses hardcoded f-string, not body JSON."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        from amplifier_core.llm_errors import LLMTimeoutError as KernelLLMTimeoutError

        with pytest.raises(KernelLLMTimeoutError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert "timed out" in str(exc_info.value).lower()
