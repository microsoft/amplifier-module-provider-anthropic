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
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import DummyResponse, FakeCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# Haiku 4.5 thinking support (version-gated)
# ---------------------------------------------------------------------------


class TestReasoningEffortOnHaiku45:
    """Haiku 4.5 supports extended thinking per Anthropic docs.

    These tests verify that reasoning_effort correctly enables thinking
    for Haiku 4.5, matching the behavior of Sonnet.
    """

    def test_haiku_45_low_reasoning_effort_enables_thinking(self):
        """Haiku 4.5 + reasoning_effort='low' → thinking enabled, budget=4096."""
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
        assert "thinking" in params
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 4096

    def test_haiku_45_medium_reasoning_effort_enables_thinking(self):
        """Haiku 4.5 + reasoning_effort='medium' → thinking enabled, default budget."""
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
        assert "thinking" in params
        # Haiku doesn't support adaptive, falls back to "enabled"
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 32000  # Haiku 4.5 default

    def test_haiku_45_high_reasoning_effort_enables_thinking(self):
        """Haiku 4.5 + reasoning_effort='high' → thinking enabled, default budget."""
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
        assert "thinking" in params
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 32000

    def test_haiku_45_explicit_extended_thinking_kwarg(self):
        """Haiku 4.5 + kwargs extended_thinking=True → thinking enabled."""
        provider = _make_provider(default_model="claude-haiku-4-5-20251001")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request, extended_thinking=True))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params

    def test_haiku_45_thinking_forces_temperature_1(self):
        """When thinking is enabled for Haiku 4.5, temperature must be 1.0."""
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
        assert "thinking" in params
        assert params["temperature"] == 1.0


# ---------------------------------------------------------------------------
# Non-thinking models (Haiku 3.5) must silently skip thinking
# ---------------------------------------------------------------------------


class TestReasoningEffortOnNonThinkingModel:
    """Models that don't support thinking (e.g. Haiku 3.5) must never send the
    ``thinking`` parameter to the API, regardless of reasoning_effort value.

    Regression tests for: budget_tokens >= 1024 API error when non-thinking
    models receive thinking params with budget_tokens=0.
    """

    def test_haiku_35_low_reasoning_effort_no_thinking(self):
        """Haiku 3.5 + reasoning_effort='low' → no thinking param sent."""
        provider = _make_provider(default_model="claude-haiku-3-5-20250929")
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

    def test_haiku_35_high_reasoning_effort_no_thinking(self):
        """Haiku 3.5 + reasoning_effort='high' → no thinking param sent."""
        provider = _make_provider(default_model="claude-haiku-3-5-20250929")
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

    def test_haiku_35_explicit_extended_thinking_kwarg_no_thinking(self):
        """Haiku 3.5 + kwargs extended_thinking=True → still no thinking param."""
        provider = _make_provider(default_model="claude-haiku-3-5-20250929")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request, extended_thinking=True))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params

    def test_haiku_35_temperature_not_forced_to_1(self):
        """When thinking is skipped for Haiku 3.5, temperature should NOT be forced to 1.0."""
        provider = _make_provider(default_model="claude-haiku-3-5-20250929")
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
# Config-level `effort` default (Phase 3)
# ---------------------------------------------------------------------------


def _make_provider_with_effort(
    effort: Any,
    default_model: str = "claude-sonnet-4-5-20250929",
) -> AnthropicProvider:
    """Provider whose config carries an `effort` default."""
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": 0,
            "default_model": default_model,
            "effort": effort,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


class TestConfigEffortDefault:
    """config['effort'] is the lowest-priority source for reasoning_effort.

    It follows the SAME coupling as request.reasoning_effort: a valid value
    enables extended thinking. request.reasoning_effort wins over it; an
    invalid config value is ignored (no thinking).
    """

    def test_config_effort_enables_thinking_when_no_request_effort(self):
        """config effort='high' + no request effort → thinking enabled."""
        provider = _make_provider_with_effort("high")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params

    def test_request_effort_wins_over_config_effort(self):
        """request.reasoning_effort='low' overrides config effort='high'."""
        provider = _make_provider_with_effort("high")
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
        # 'low' wins → enabled + 4096; would be adaptive/default if 'high' leaked
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 4096

    def test_invalid_config_effort_is_ignored(self):
        """An invalid config effort (e.g. 'ultra') is ignored → no thinking."""
        provider = _make_provider_with_effort("ultra")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params
        assert "output_config" not in params

    def test_config_effort_is_case_insensitive(self):
        """config effort='High' (mixed case / whitespace) is normalised."""
        provider = _make_provider_with_effort("  High ")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params

    def test_config_effort_max_on_opus_48_sets_output_config(self):
        """config effort='max' on Opus 4.8 → output_config.effort=max + adaptive."""
        provider = _make_provider_with_effort(
            "max", default_model="claude-opus-4-8-20260101"
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params.get("output_config", {}).get("effort") == "max"
        assert params["thinking"]["type"] == "adaptive"

    def test_config_effort_xhigh_unsupported_on_sonnet_omits_output_config(self):
        """config effort='xhigh' on a model that lacks output_config → omitted.

        Thinking still engages (the coupling), but output_config.effort is not
        sent because the model's capability matrix doesn't list it.
        """
        provider = _make_provider_with_effort(
            "xhigh", default_model="claude-sonnet-4-5-20250929"
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" not in params
        assert "thinking" in params


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


# ---------------------------------------------------------------------------
# Speed config plumbing — end-to-end request param and beta header
# ---------------------------------------------------------------------------


class TestSpeedConfigEndToEnd:
    def test_speed_fast_config_sends_speed_param_and_beta_header(self):
        """config speed='fast' + claude-opus-4-8 → params['speed']=='fast' and fast-mode beta header."""
        provider = AnthropicProvider(
            api_key="test-key",
            config={
                "use_streaming": False,
                "max_retries": 0,
                "default_model": "claude-opus-4-8",
                "speed": "fast",
            },
        )
        provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params.get("speed") == "fast"
        beta_header = params.get("extra_headers", {}).get("anthropic-beta", "")
        assert "fast-mode-2026-02-01" in beta_header
