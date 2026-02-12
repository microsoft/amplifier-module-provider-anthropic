"""Tests for Phase 2 reasoning_effort support in thinking configuration.

Verifies:
- reasoning_effort="low"  → type="enabled", budget_tokens=4096
- reasoning_effort="medium" → type="adaptive" if supported, else "enabled" + default budget
- reasoning_effort="high"  → type="adaptive" if supported, else "enabled" + default budget
- reasoning_effort=None    → existing behavior unchanged
- kwargs["extended_thinking"]=True overrides reasoning_effort=None
- kwargs["extended_thinking"]=False overrides reasoning_effort="high" → no thinking
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock


from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider


# ---------------------------------------------------------------------------
# Helpers
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
    """Extract the kwargs passed to the API call."""
    assert mock_create.await_count == 1
    _, kwargs = mock_create.call_args
    return kwargs


# ---------------------------------------------------------------------------
# reasoning_effort mapping tests
# ---------------------------------------------------------------------------


class TestReasoningEffortLow:
    def test_low_enables_thinking_with_small_budget(self):
        """reasoning_effort='low' → type='enabled', budget_tokens=4096."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="low",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 4096


class TestReasoningEffortMedium:
    def test_medium_on_sonnet_uses_enabled_with_default_budget(self):
        """Sonnet doesn't support adaptive → type='enabled', default budget."""
        provider = _make_provider(default_model="claude-sonnet-4-5-20250929")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="medium",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params
        # Sonnet doesn't support adaptive, falls back to "enabled"
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 32000  # Sonnet default

    def test_medium_on_opus_uses_adaptive(self):
        """Opus 4.6+ supports adaptive → type='adaptive'."""
        provider = _make_provider(default_model="claude-opus-4-6-20250929")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="medium",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params
        assert params["thinking"]["type"] == "adaptive"


class TestReasoningEffortHigh:
    def test_high_on_sonnet_uses_enabled_with_default_budget(self):
        """Sonnet: high → type='enabled', default budget."""
        provider = _make_provider(default_model="claude-sonnet-4-5-20250929")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 32000

    def test_high_on_opus_uses_adaptive(self):
        """Opus 4.6+: high → type='adaptive'."""
        provider = _make_provider(default_model="claude-opus-4-6-20250929")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params
        assert params["thinking"]["type"] == "adaptive"


# ---------------------------------------------------------------------------
# Non-thinking models (Haiku) must silently skip thinking
# ---------------------------------------------------------------------------


class TestReasoningEffortOnNonThinkingModel:
    """Models that don't support thinking (e.g. Haiku) must never send the
    ``thinking`` parameter to the API, regardless of reasoning_effort value.

    Regression tests for: budget_tokens >= 1024 API error when Haiku receives
    thinking params with budget_tokens=0.
    """

    def test_haiku_low_reasoning_effort_no_thinking(self):
        """Haiku + reasoning_effort='low' → no thinking param sent."""
        provider = _make_provider(default_model="claude-haiku-4-5-20251001")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="low",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params

    def test_haiku_medium_reasoning_effort_no_thinking(self):
        """Haiku + reasoning_effort='medium' → no thinking param sent."""
        provider = _make_provider(default_model="claude-haiku-4-5-20251001")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="medium",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params

    def test_haiku_high_reasoning_effort_no_thinking(self):
        """Haiku + reasoning_effort='high' → no thinking param sent."""
        provider = _make_provider(default_model="claude-haiku-4-5-20251001")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params

    def test_haiku_explicit_extended_thinking_kwarg_no_thinking(self):
        """Haiku + kwargs extended_thinking=True → still no thinking param."""
        provider = _make_provider(default_model="claude-haiku-4-5-20251001")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request, extended_thinking=True))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params

    def test_haiku_temperature_not_forced_to_1(self):
        """When thinking is skipped for Haiku, temperature should NOT be forced to 1.0."""
        provider = _make_provider(default_model="claude-haiku-4-5-20251001")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
            temperature=0.5,
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params
        # Temperature should remain as requested, not forced to 1.0
        assert params.get("temperature") != 1.0


class TestReasoningEffortNone:
    def test_none_no_thinking(self):
        """reasoning_effort=None → no thinking (existing behavior)."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort=None,
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params


# ---------------------------------------------------------------------------
# Precedence / override tests
# ---------------------------------------------------------------------------


class TestKwargsOverrideReasoningEffort:
    def test_kwargs_extended_thinking_true_overrides_none(self):
        """kwargs['extended_thinking']=True enables thinking even with no reasoning_effort."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort=None,
        )
        asyncio.run(provider.complete(request, extended_thinking=True))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params

    def test_kwargs_extended_thinking_false_overrides_high(self):
        """kwargs['extended_thinking']=False disables thinking even with reasoning_effort='high'."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request, extended_thinking=False))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params

    def test_kwargs_thinking_budget_overrides_effort_budget(self):
        """kwargs['thinking_budget_tokens'] overrides the budget from reasoning_effort."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="low",  # Would set budget_tokens=4096
        )
        asyncio.run(provider.complete(request, thinking_budget_tokens=16000))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params
        assert params["thinking"]["budget_tokens"] == 16000


class TestTemperatureOverride:
    def test_thinking_forces_temperature_1(self):
        """When thinking is enabled (via reasoning_effort), temperature must be 1.0."""
        provider = _make_provider()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="low",
            temperature=0.5,  # Should be overridden to 1.0
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["temperature"] == 1.0
