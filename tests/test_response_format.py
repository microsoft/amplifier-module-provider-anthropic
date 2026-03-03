"""Tests for response_format → output_config wiring in the Anthropic provider.

Verifies that ChatRequest.response_format is correctly translated to
Anthropic's output_config.format API parameter.

Cases:
- ResponseFormatJsonSchema → output_config with json_schema
- ResponseFormatJson       → output_config with json mode
- ResponseFormatText       → no output_config (default text)
- None                     → no output_config (baseline)
- response_format + thinking enabled → output_config skipped (mutually exclusive)
"""

import asyncio
from typing import Any, cast
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    ResponseFormatJson,
    ResponseFormatJsonSchema,
    ResponseFormatText,
)
from amplifier_module_provider_anthropic import AnthropicProvider


# ---------------------------------------------------------------------------
# Shared helpers (same pattern as test_reasoning_effort.py)
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


class DummyResponse:
    """Minimal Anthropic API response stub."""

    def __init__(self):
        self.content = [SimpleNamespace(type="text", text="ok")]
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.stop_reason = "end_turn"
        self.model = "claude-sonnet-4-5-20250929"


def _make_provider(
    default_model: str = "claude-sonnet-4-5-20250929",
) -> AnthropicProvider:
    # Patch RetryConfig to handle parameter naming differences across amplifier_core
    # versions (initial_delay vs min_delay). This is a local dev environment concern;
    # CI runs with the correct amplifier_core version.
    from unittest.mock import patch, MagicMock
    with patch("amplifier_module_provider_anthropic.RetryConfig") as mock_rc:
        mock_rc.return_value = MagicMock()
        provider = AnthropicProvider(
            api_key="test-key",
            config={
                "use_streaming": False,
                "max_retries": 0,
                "default_model": default_model,
            },
        )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _make_raw_mock() -> MagicMock:
    raw = MagicMock()
    raw.parse.return_value = DummyResponse()
    raw.headers = {}
    return raw


def _get_api_params(mock_create: AsyncMock) -> dict[str, Any]:
    """Extract the kwargs passed to the Anthropic API call."""
    assert mock_create.await_count == 1
    _, kwargs = mock_create.call_args
    return kwargs


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# ResponseFormatJsonSchema tests
# ---------------------------------------------------------------------------


class TestResponseFormatJsonSchema:
    """ResponseFormatJsonSchema sets output_config with json_schema."""

    SCHEMA = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }

    def test_json_schema_sets_output_config(self):
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="test")],
            response_format=ResponseFormatJsonSchema(json_schema=self.SCHEMA),
        )

        _run(provider._complete_chat_request(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)

        assert "output_config" in params, "output_config should be set for json_schema"
        assert params["output_config"]["format"]["type"] == "json_schema"
        assert params["output_config"]["format"]["schema"] == self.SCHEMA

    def test_json_schema_does_not_set_thinking(self):
        """Structured output should not accidentally enable thinking."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="test")],
            response_format=ResponseFormatJsonSchema(json_schema=self.SCHEMA),
        )

        _run(provider._complete_chat_request(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)

        assert "thinking" not in params


# ---------------------------------------------------------------------------
# ResponseFormatJson tests
# ---------------------------------------------------------------------------


class TestResponseFormatJson:
    """ResponseFormatJson sets output_config with json mode (schema-less)."""

    def test_json_mode_sets_output_config(self):
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="test")],
            response_format=ResponseFormatJson(),
        )

        _run(provider._complete_chat_request(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)

        assert "output_config" in params, "output_config should be set for json mode"
        assert params["output_config"]["format"]["type"] == "json"

    def test_json_mode_no_schema_key(self):
        """Schema-less json mode should not have a schema key."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="test")],
            response_format=ResponseFormatJson(),
        )

        _run(provider._complete_chat_request(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)

        assert "schema" not in params.get("output_config", {}).get("format", {})


# ---------------------------------------------------------------------------
# ResponseFormatText tests
# ---------------------------------------------------------------------------


class TestResponseFormatText:
    """ResponseFormatText produces no output_config (default text behaviour)."""

    def test_text_format_no_output_config(self):
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="test")],
            response_format=ResponseFormatText(),
        )

        _run(provider._complete_chat_request(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)

        assert "output_config" not in params, (
            "ResponseFormatText should not set output_config"
        )


# ---------------------------------------------------------------------------
# None (no response_format) — baseline
# ---------------------------------------------------------------------------


class TestNoResponseFormat:
    """When response_format is None, no output_config is set."""

    def test_none_response_format_no_output_config(self):
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="test")],
        )

        _run(provider._complete_chat_request(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)

        assert "output_config" not in params


# ---------------------------------------------------------------------------
# Mutual exclusion: response_format + thinking
# ---------------------------------------------------------------------------


class TestResponseFormatThinkingMutualExclusion:
    """output_config must be skipped when extended thinking is enabled."""

    SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}

    def test_thinking_takes_precedence_over_json_schema(self):
        """When thinking is requested, output_config is not set."""
        provider = _make_provider(default_model="claude-opus-4-5-20251101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="test")],
            response_format=ResponseFormatJsonSchema(json_schema=self.SCHEMA),
        )

        _run(
            provider._complete_chat_request(
                request, extended_thinking=True, thinking_budget_tokens=4096
            )
        )
        params = _get_api_params(provider.client.messages.with_raw_response.create)

        assert "thinking" in params, "Thinking should be enabled"
        assert "output_config" not in params, (
            "output_config must not be set when thinking is enabled"
        )

    def test_thinking_takes_precedence_over_json_mode(self):
        """Thinking wins over json mode too."""
        provider = _make_provider(default_model="claude-opus-4-5-20251101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="test")],
            response_format=ResponseFormatJson(),
        )

        _run(
            provider._complete_chat_request(
                request, extended_thinking=True, thinking_budget_tokens=4096
            )
        )
        params = _get_api_params(provider.client.messages.with_raw_response.create)

        assert "thinking" in params
        assert "output_config" not in params
