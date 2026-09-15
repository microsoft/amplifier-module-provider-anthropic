"""Tests for context-window overflow detection (BadRequestError -> ContextLengthError).

The message strings used in this file are Anthropic's ACTUAL production wording
for context-window overflow errors, verified against Anthropic's official docs
and independent verbatim production logs:

    "prompt is too long: 208310 tokens > 200000 maximum"
    "input length and `max_tokens` exceed context limit: 189127 + 16000 > 200000, ..."

Do NOT "simplify" these back into synthetic text like
"prompt is too long: context length exceeded" -- that string is not a message
Anthropic ever emits. A fabricated fixture that merely contains the substring
under test can pass while missing the real-world message shape entirely,
which is exactly the gap this test file exists to close.
"""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import (
    ContextLengthError as KernelContextLengthError,
)
from amplifier_core.llm_errors import (
    InvalidRequestError as KernelInvalidRequestError,
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


def _make_anthropic_error(cls, message="error", status_code=400):
    """Construct an Anthropic SDK error with the expected shape."""
    # Anthropic SDK errors take (message, response, body)
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    return cls(message, response=mock_response, body=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRealContextOverflowMessages:
    """Both real Anthropic overflow message shapes must classify as ContextLengthError."""

    def test_message_a_real_wording(self):
        """Message A: input alone exceeds the window (all models)."""
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "prompt is too long: 208310 tokens > 200000 maximum",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContextLengthError):
            asyncio.run(provider.complete(_simple_request()))

    def test_message_a_with_non_default_limit(self):
        """Regression guard: the maximum value is NOT hardcoded to 200000."""
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "prompt is too long: 103078 tokens > 102398 maximum",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContextLengthError):
            asyncio.run(provider.complete(_simple_request()))

    def test_message_b_real_wording_with_backticks(self):
        """Message B: input + max_tokens exceeds the window (legacy models).

        Backticks around `max_tokens` are literally present in the real API
        string.
        """
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "input length and `max_tokens` exceed context limit: "
            "189127 + 16000 > 200000, decrease input length or "
            "`max_tokens` and try again",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContextLengthError):
            asyncio.run(provider.complete(_simple_request()))

    def test_message_b_without_backticks(self):
        """Some gateways strip backticks when rewriting the message."""
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "input length and max_tokens exceed context limit: "
            "188240 + 21333 > 200000, decrease input length or "
            "max_tokens and try again",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContextLengthError):
            asyncio.run(provider.complete(_simple_request()))

    def test_legacy_marker_still_classifies(self):
        """Legacy phrasing ("maximum context length is ...") is retained
        deliberately -- other providers and rewriting gateways may still use it.
        """
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "This model's maximum context length is 200000 tokens",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContextLengthError):
            asyncio.run(provider.complete(_simple_request()))


class TestNegativeGuard:
    """Unrelated 400s must NOT be misclassified as context overflow."""

    def test_unrelated_bad_request_is_invalid_request_error(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "messages: at least one message is required",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelInvalidRequestError):
            asyncio.run(provider.complete(_simple_request()))


class TestContextLengthErrorAttributes:
    """The raised error must be non-retryable -- retrying a deterministic
    context overflow just burns attempts."""

    def test_context_length_error_is_not_retryable_and_status_400(self):
        provider = _make_provider()
        sdk_error = _make_anthropic_error(
            anthropic.BadRequestError,
            "prompt is too long: 208310 tokens > 200000 maximum",
            status_code=400,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )

        with pytest.raises(KernelContextLengthError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        e = exc_info.value
        assert e.status_code == 400
        assert e.retryable is False


class TestRecoverableInputOverflow:
    """Only strict structured input-only 400s authorize one provider retry."""

    def _structured_error(self, message: str, *, request_id: str | None = "req-1"):
        response = MagicMock()
        response.status_code = 400
        response.headers = {"request-id": request_id} if request_id else {}
        return anthropic.BadRequestError(
            message,
            response=response,
            body={"type": "invalid_request_error", "message": message},
        )

    def test_input_only_error_returns_one_bound_recovery_decision(self):
        provider = _make_provider()
        request = _simple_request()
        error = self._structured_error(
            "prompt is too long: 208310 tokens > 200000 maximum"
        )
        provider.client.messages.with_raw_response.create = AsyncMock(side_effect=error)

        with pytest.raises(KernelContextLengthError) as raised:
            asyncio.run(provider.complete(request))

        decision = provider.recover_context_overflow(
            request, raised.value, context_estimate=100_000
        )
        assert decision == {
            "estimated_input_tokens": 208310,
            "input_limit_tokens": 200000,
            "context_token_budget": 94043,
            "max_output_tokens": provider.max_tokens,
        }
        assert (
            provider.recover_context_overflow(
                request, raised.value, context_estimate=100_000
            )
            is None
        )
        assert provider.client.messages.with_raw_response.create.await_count == 1

    def test_recovery_rejects_changed_request_or_options(self):
        provider = _make_provider()
        request = _simple_request()
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=self._structured_error(
                "prompt is too long: 208310 tokens > 200000 maximum"
            )
        )

        with pytest.raises(KernelContextLengthError) as raised:
            asyncio.run(provider.complete(request))

        changed = ChatRequest(messages=[Message(role="user", content="Changed")])
        assert (
            provider.recover_context_overflow(
                changed, raised.value, context_estimate=100_000
            )
            is None
        )

        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=self._structured_error(
                "prompt is too long: 208310 tokens > 200000 maximum"
            )
        )
        with pytest.raises(KernelContextLengthError) as second:
            asyncio.run(provider.complete(request))
        assert (
            provider.recover_context_overflow(
                request,
                second.value,
                context_estimate=100_000,
                request_options={"stop_sequences": ["END"]},
            )
            is None
        )

    @pytest.mark.parametrize(
        "message,request_id",
        [
            (
                "input length and `max_tokens` exceed context limit: "
                "189127 + 16000 > 200000",
                "req-1",
            ),
            ("prompt is too long: 0 tokens > 200000 maximum", "req-1"),
            ("prompt is too long: 200000 tokens > 200000 maximum", "req-1"),
            ("prompt is too long: 208310 tokens > 200000 maximum", None),
        ],
    )
    def test_joint_malformed_or_unbound_errors_cannot_recover(
        self, message: str, request_id: str | None
    ):
        provider = _make_provider()
        request = _simple_request()
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=self._structured_error(message, request_id=request_id)
        )

        with pytest.raises(KernelContextLengthError) as raised:
            asyncio.run(provider.complete(request))

        assert (
            provider.recover_context_overflow(
                request, raised.value, context_estimate=100_000
            )
            is None
        )
