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
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolSpec,
)
from anthropic import AsyncAnthropic
from anthropic import _base_client as anthropic_base_client

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


def _rich_adaptive_request() -> ChatRequest:
    """Use the assembly features exercised by adaptive overflow recovery."""
    return ChatRequest(
        messages=[
            Message(role="system", content="System authority."),
            Message(role="developer", content="Developer context."),
            Message(role="user", content="Original question."),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="Internal reasoning.", signature="signed-history"),
                    TextBlock(text="Prior answer."),
                ],
            ),
            Message(role="user", content="Required reminder."),
        ],
        tools=[
            ToolSpec(
                name=f"lookup_{index}",
                description=f"Find value {index}.",
                parameters={"type": "object", "properties": {}},
            )
            for index in range(45)
        ],
        reasoning_effort="high",
        max_output_tokens=128_000,
    )


def _sdk_mock_client(
    body: dict[str, object], *, request_id: str | None = "req-sdk-outer"
) -> AsyncAnthropic:
    """Build a real SDK client over its own installed HTTP implementation."""
    sdk_httpx = getattr(anthropic_base_client, "httpx2", None)
    if sdk_httpx is None:
        sdk_httpx = anthropic_base_client.httpx

    def handler(request):
        return sdk_httpx.Response(
            400,
            headers={"request-id": request_id} if request_id else {},
            json=body,
            request=request,
        )

    return AsyncAnthropic(
        api_key="[REDACTED:SECRET]",
        max_retries=0,
        http_client=sdk_httpx.AsyncClient(transport=sdk_httpx.MockTransport(handler)),
    )


def _make_anthropic_error(cls, message="error", status_code=400):
    """Construct an Anthropic SDK error with the expected shape."""
    # Anthropic SDK errors take (message, response, body)
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    return cls(message, response=mock_response, body=None)


class _FirstEventThenOverflow:
    """SDK stream that proves even message_start makes recovery ineligible."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.response = MagicMock(headers={})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def __aiter__(self):
        async def events():
            yield type("RawMessageStartEvent", (), {})()
            raise self.error

        return events()


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

    def test_real_sdk_outer_error_envelope_recovers_bound_adaptive_request(self):
        """SDK MockTransport must preserve the actual outer error envelope."""
        provider = _make_provider()
        provider._enable_1m_context = True
        request = _rich_adaptive_request()
        body = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long: 1087533 tokens > 1000000 maximum",
            },
            "request_id": "req-sdk-outer",
        }
        client = _sdk_mock_client(body, request_id=None)
        provider._client = client

        try:
            with pytest.raises(KernelContextLengthError) as raised:
                asyncio.run(provider.complete(request))
        finally:
            asyncio.run(client.close())

        assert isinstance(raised.value.__cause__, anthropic.BadRequestError)
        assert raised.value.__cause__.body == body
        assert provider._prefix_fingerprints
        decision = provider.recover_context_overflow(
            request, raised.value, context_estimate=200_000
        )
        assert decision == {
            "estimated_input_tokens": 1_087_533,
            "input_limit_tokens": 1_000_000,
            "context_token_budget": 183_148,
            "max_output_tokens": 128_000,
        }

    def test_combined_error_returns_one_bound_recovery_decision(self):
        provider = _make_provider()
        request = _simple_request()
        error = self._structured_error(
            "input length and `max_tokens` exceed context limit: "
            "189127 + 16000 > 200000"
        )
        provider.client.messages.with_raw_response.create = AsyncMock(side_effect=error)

        with pytest.raises(KernelContextLengthError) as raised:
            asyncio.run(provider.complete(request))

        assert provider.recover_context_overflow(
            request, raised.value, context_estimate=100_000
        ) == {
            "estimated_input_tokens": 189127,
            "input_limit_tokens": 184000,
            "context_token_budget": 95122,
            "max_output_tokens": provider.max_tokens,
        }

    @pytest.mark.parametrize(
        "body",
        [
            {
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "prompt is too long: 208310 tokens > 200000 maximum",
                },
                "request_id": "req-wrong-type",
            },
            {
                "type": "error",
                "error": {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "prompt is too long: 208310 tokens > 200000 maximum",
                    },
                },
                "request_id": "req-nested",
            },
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        "input length and `max_tokens` exceed context limit: "
                        "189127 + 16000 > 205127"
                    ),
                },
                "request_id": "req-joint",
            },
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: 200000 tokens > 200000 maximum",
                },
            },
        ],
    )
    def test_real_sdk_outer_envelope_rejects_nonrecoverable_shapes(self, body):
        provider = _make_provider()
        request = _simple_request()
        client = _sdk_mock_client(body, request_id=None)
        provider._client = client

        try:
            with pytest.raises(KernelContextLengthError) as raised:
                asyncio.run(provider.complete(request))
        finally:
            asyncio.run(client.close())

        assert provider.recover_context_overflow(
            request, raised.value, context_estimate=100_000
        ) is None

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

        copied = request.model_copy(deep=True)
        assert (
            provider.recover_context_overflow(
                copied, raised.value, context_estimate=100_000
            )
            is None
        )

        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=self._structured_error(
                "prompt is too long: 208310 tokens > 200000 maximum"
            )
        )
        with pytest.raises(KernelContextLengthError) as changed:
            asyncio.run(provider.complete(request))
        request.messages[0].content = "Changed"
        assert (
            provider.recover_context_overflow(
                request, changed.value, context_estimate=100_000
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

    def test_recovery_rejects_another_provider_instance_without_consuming(self):
        provider = _make_provider()
        other_provider = _make_provider()
        request = _simple_request()
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=self._structured_error(
                "prompt is too long: 208310 tokens > 200000 maximum"
            )
        )

        with pytest.raises(KernelContextLengthError) as raised:
            asyncio.run(provider.complete(request))

        assert (
            other_provider.recover_context_overflow(
                request, raised.value, context_estimate=100_000
            )
            is None
        )
        assert (
            provider.recover_context_overflow(
                request, raised.value, context_estimate=100_000
            )
            is not None
        )

    @pytest.mark.parametrize(
        "message,request_id",
        [
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

    @pytest.mark.parametrize(
        "message",
        [
            "input length and `max_tokens` exceed context limit: 189127 + 16000",
            "input length and `max_tokens` exceed context limit: 0 + 16000 > 200000",
            "input length and `max_tokens` exceed context limit: 189127 + 0 > 200000",
            "input length and `max_tokens` exceed context limit: 100 + 200 > 100",
            "input length and `max_tokens` exceed context limit: 189127 + 16000 > 200000, decrease input length",
            "input length and max_tokens exceed context limit: 189127 + 16000 > 200000",
        ],
    )
    def test_malformed_combined_errors_cannot_recover(self, message: str):
        provider = _make_provider()
        request = _simple_request()
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=self._structured_error(message)
        )

        with pytest.raises(KernelContextLengthError) as raised:
            asyncio.run(provider.complete(request))

        assert (
            provider.recover_context_overflow(
                request, raised.value, context_estimate=100_000
            )
            is None
        )

    def test_first_sdk_message_start_makes_strict_input_overflow_unrecoverable(self):
        provider = _make_provider()
        provider.use_streaming = True
        request = _simple_request()
        error = self._structured_error(
            "prompt is too long: 208310 tokens > 200000 maximum"
        )
        provider.client.messages.stream = MagicMock(
            return_value=_FirstEventThenOverflow(error)
        )

        with pytest.raises(KernelContextLengthError) as raised:
            asyncio.run(provider.complete(request))

        assert (
            provider.recover_context_overflow(
                request, raised.value, context_estimate=100_000
            )
            is None
        )
