"""Tests for public generation error messages and private SDK bodies.

Known request errors retain their body/str behavior. Safe rate-limit refusals
use a fixed description; unknown outcomes never expose response or cause text.
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
    RateLimitError as KernelRateLimitError,
)
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._request_safety import (
    UNKNOWN_MESSAGE,
    RequestOutcomeUnknownError,
)

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
# Block 1: RateLimitError — structured refusal versus unknown outcome
# ---------------------------------------------------------------------------


class TestRateLimitErrorPublicMessage:
    def test_structured_refusal_uses_fixed_message(self):
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

        assert str(exc_info.value) == "Anthropic refused the request due to a rate limit."
        assert exc_info.value.retryable is True
        assert exc_info.value.__cause__ is sdk_error

    def test_missing_body_is_unknown_not_a_refusal(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error_with_body(
            anthropic.RateLimitError, "rate limited", status_code=429, body=None
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(RequestOutcomeUnknownError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert str(exc_info.value) == UNKNOWN_MESSAGE
        assert exc_info.value.retryable is False
        assert exc_info.value.request_outcome == "unknown"
        assert exc_info.value.effects == "may_have_occurred"
        assert exc_info.value.__cause__ is sdk_error
        assert provider.client.messages.with_raw_response.create.await_count == 1


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
                "message": "prompt is too long: 208310 tokens > 200000 maximum",
            },
        }
        sdk_error = _make_anthropic_error_with_body(
            anthropic.BadRequestError,
            "prompt is too long: 208310 tokens > 200000 maximum",
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
            "prompt is too long: 208310 tokens > 200000 maximum",
            status_code=400,
            body=None,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContextLengthError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert "prompt is too long" in str(exc_info.value)

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

    def test_5xx_unknown_outcome_keeps_body_private(self):
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

        with pytest.raises(RequestOutcomeUnknownError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert str(exc_info.value) == UNKNOWN_MESSAGE
        assert exc_info.value.retryable is False
        assert exc_info.value.request_outcome == "unknown"
        assert exc_info.value.effects == "may_have_occurred"
        assert exc_info.value.__cause__ is sdk_error
        assert provider.client.messages.with_raw_response.create.await_count == 1

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

    def test_5xx_unknown_outcome_keeps_sdk_message_private(self):
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

        with pytest.raises(RequestOutcomeUnknownError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert str(exc_info.value) == UNKNOWN_MESSAGE
        assert exc_info.value.retryable is False
        assert exc_info.value.__cause__ is sdk_error
        assert provider.client.messages.with_raw_response.create.await_count == 1

    def test_other_status_is_unknown_when_body_none(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error_with_body(
            anthropic.APIStatusError, "I'm a teapot", status_code=418, body=None
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelLLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert str(exc_info.value) == UNKNOWN_MESSAGE
        assert exc_info.value.request_outcome == "unknown"

    def test_other_status_does_not_publish_body(self):
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

        assert str(exc_info.value) == UNKNOWN_MESSAGE
        assert exc_info.value.request_outcome == "unknown"


# ---------------------------------------------------------------------------
# Generic exception catch-all — fixed public message and private cause
# ---------------------------------------------------------------------------


class TestGenericExceptionPrivacy:
    def test_exception_with_body_keeps_body_private(self):
        """An arbitrary body is not safe to include in the public message."""
        provider = _make_provider()
        body = {"type": "error", "error": {"message": "unexpected"}}
        original = Exception("something unexpected")
        original.body = body  # type: ignore[attr-defined]
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=original
        )

        with pytest.raises(RequestOutcomeUnknownError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert str(exc_info.value) == UNKNOWN_MESSAGE
        assert exc_info.value.retryable is False
        assert exc_info.value.request_outcome == "unknown"
        assert exc_info.value.effects == "may_have_occurred"
        assert exc_info.value.__cause__ is original
        assert provider.client.messages.with_raw_response.create.await_count == 1

    def test_exception_without_body_keeps_message_private(self):
        """Unknown-stage exception text stays in the private cause chain."""
        provider = _make_provider()
        original = RuntimeError("something unexpected")
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=original
        )

        with pytest.raises(RequestOutcomeUnknownError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert str(exc_info.value) == UNKNOWN_MESSAGE
        assert exc_info.value.retryable is False
        assert exc_info.value.__cause__ is original
        assert provider.client.messages.with_raw_response.create.await_count == 1


# ---------------------------------------------------------------------------
# Deadline errors — compatible timeout type, conservative outcome
# ---------------------------------------------------------------------------


class TestTimeoutPrivacy:
    def test_timeout_error_uses_fixed_unknown_message(self):
        """A deadline does not prove that the provider stopped generation."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        from amplifier_core.llm_errors import LLMTimeoutError as KernelLLMTimeoutError

        with pytest.raises(KernelLLMTimeoutError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert str(exc_info.value) == UNKNOWN_MESSAGE
        assert exc_info.value.retryable is False
        assert exc_info.value.request_outcome == "unknown"
        assert exc_info.value.effects == "may_have_occurred"
        assert provider.client.messages.with_raw_response.create.await_count == 1
