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
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
import amplifier_module_provider_anthropic as anthropic_module
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

    def test_config_effort_max_on_sonnet_5_sets_output_config(self):
        """config effort='max' on Sonnet 5 → output_config.effort=max + adaptive.

        Regression guard: Sonnet 5 accepts the 'max' effort tier (confirmed
        2026-07-20). Previously 'max' was omitted from supported_efforts, so the
        provider dropped output_config.effort and logged a warning.
        """
        provider = _make_provider_with_effort("max", default_model="claude-sonnet-5")
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
# Canonical `reasoning_effort` config key (portable kernel key)
# ---------------------------------------------------------------------------


def _make_provider_with_config(
    config_overrides: dict[str, Any],
    default_model: str = "claude-sonnet-4-5-20250929",
) -> AnthropicProvider:
    """Provider with arbitrary effort-family config keys."""
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": 0,
            "default_model": default_model,
            **config_overrides,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


class TestCanonicalReasoningEffortConfigKey:
    """config['reasoning_effort'] is the canonical effort key (matches the
    kernel's portable request.reasoning_effort). config['effort'] remains a
    working alias; when both are set, 'reasoning_effort' wins.
    """

    def test_canonical_key_enables_thinking(self):
        """config reasoning_effort='high' + no request effort -> thinking enabled."""
        provider = _make_provider_with_config({"reasoning_effort": "high"})
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params

    def test_canonical_key_low_maps_to_small_budget(self):
        """config reasoning_effort='low' -> type='enabled', budget_tokens=4096."""
        provider = _make_provider_with_config({"reasoning_effort": "low"})
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 4096

    def test_canonical_key_wins_over_effort_alias(self):
        """Both set: reasoning_effort='low' beats effort='high' (budget=4096)."""
        provider = _make_provider_with_config(
            {"reasoning_effort": "low", "effort": "high"}
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        # 'low' wins -> enabled + 4096; would be default budget if 'high' leaked
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 4096

    def test_both_keys_set_warns_at_init(self, caplog):
        """Setting both keys logs a precedence warning once at provider init."""
        import logging

        with caplog.at_level(logging.WARNING):
            _make_provider_with_config({"reasoning_effort": "low", "effort": "high"})

        assert any(
            "reasoning_effort" in r.message and "effort" in r.message
            for r in caplog.records
        )

    def test_request_effort_wins_over_canonical_config_key(self):
        """request.reasoning_effort='low' overrides config reasoning_effort='high'."""
        provider = _make_provider_with_config({"reasoning_effort": "high"})
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="low",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 4096

    def test_invalid_canonical_key_is_ignored_with_warning(self, caplog):
        """Invalid reasoning_effort ('ultra') warns naming the key, no thinking."""
        import logging

        provider = _make_provider_with_config({"reasoning_effort": "ultra"})
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params
        assert any(
            "reasoning_effort" in r.message and "ultra" in r.message
            for r in caplog.records
        )

    def test_effort_alias_still_works_alone(self):
        """Legacy config effort='high' alone still enables thinking (unchanged)."""
        provider = _make_provider_with_config({"effort": "high"})
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params


class TestEffortConfigFieldIsCanonical:
    """get_info() must advertise the canonical key.

    ConfigField.id is written verbatim into settings.yaml by the app-CLI
    (provider install wizard and routing-matrix editor), so advertising the
    legacy "effort" alias would make the CLI generate the deprecated key.
    """

    def test_config_field_id_is_reasoning_effort(self):
        provider = _make_provider()
        field_ids = {f.id for f in provider.get_info().config_fields}

        assert "reasoning_effort" in field_ids
        assert "effort" not in field_ids

    def test_config_field_still_offers_all_effort_tiers(self):
        provider = _make_provider()
        field = next(
            f for f in provider.get_info().config_fields if f.id == "reasoning_effort"
        )

        assert field.choices == ["low", "medium", "high", "xhigh", "max"]


class TestNoSpuriousEffortWarnings:
    """A config without effort-family keys must emit no effort warnings."""

    def test_no_warning_when_no_effort_keys_set(self, caplog):
        """A config without effort-family keys emits no effort warnings."""
        import logging

        with caplog.at_level(logging.WARNING):
            _make_provider()

        assert not any(
            "not consumed" in r.message or "canonical" in r.message
            for r in caplog.records
        )


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


# ---------------------------------------------------------------------------
# Effort clamping (amplifier-support#289 follow-up)
#
# When a configured/requested effort exceeds a resolved model's ceiling, the
# provider must CLAMP to the highest tier the model supports instead of
# omitting output_config.effort entirely. Models that fully support the
# requested tier must see zero behavior change.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_effort_downgrade_state():
    """Each test starts with a clean one-time-notice cache."""
    anthropic_module._clear_effort_downgrade_notices()
    yield
    anthropic_module._clear_effort_downgrade_notices()


class TestEffortClampingPerModel:
    """Parametrized model-family x requested-effort clamp verification."""

    @pytest.mark.parametrize(
        ("default_model", "requested_effort", "expected_effort"),
        [
            # Sonnet 5+: ceiling is xhigh (no "max" tier) -> max clamps down.
            ("claude-sonnet-5", "max", "xhigh"),
            ("claude-sonnet-5", "xhigh", "xhigh"),  # exact, unchanged
            ("claude-sonnet-5", "high", "high"),  # exact, unchanged
            ("claude-sonnet-5", "low", "low"),  # exact, unchanged
            # Fable 5: supports every tier including "max" -> always passes
            # through untouched (zero behavior change).
            ("claude-fable-5", "max", "max"),
            ("claude-fable-5", "xhigh", "xhigh"),
            ("claude-fable-5", "high", "high"),
            # Opus 4.8+: supports every tier including "max" -> passthrough.
            ("claude-opus-4-8", "max", "max"),
            ("claude-opus-4-8", "xhigh", "xhigh"),
            # Opus 4.7: ceiling is xhigh (pre-4.8 "max" tier not yet added).
            ("claude-opus-4-7-20260416", "max", "xhigh"),
            ("claude-opus-4-7-20260416", "xhigh", "xhigh"),
        ],
    )
    def test_effort_clamps_to_expected_tier(
        self, default_model: str, requested_effort: str, expected_effort: str
    ):
        provider = _make_provider_with_effort(
            requested_effort, default_model=default_model
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params.get("output_config", {}).get("effort") == expected_effort

    def test_max_on_sonnet_5_never_omits_output_config(self):
        """Regression guard: max->xhigh clamp must still populate output_config,
        never fall back to the old omit-entirely behavior."""
        provider = _make_provider_with_effort("max", default_model="claude-sonnet-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" in params
        assert "effort" in params["output_config"]


class TestEffortDowngradeOneTimeLogging:
    """The downgrade notice is INFO-level and fires once per (model, effort)."""

    def test_downgrade_notice_is_info_not_warning(self, caplog):
        provider = _make_provider_with_effort("max", default_model="claude-sonnet-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(messages=[Message(role="user", content="Hello")])

        with caplog.at_level(logging.INFO):
            asyncio.run(provider.complete(request))

        info_notices = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "xhigh" in r.message
        ]
        warning_notices = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert info_notices, "expected an INFO downgrade notice"
        assert not warning_notices, "must not warn per-request on a known clamp"

    def test_second_identical_request_does_not_repeat_notice(self, caplog):
        """Two requests, same model + same effort -> exactly one notice."""
        provider = _make_provider_with_effort("max", default_model="claude-sonnet-5")

        def _notice_count() -> int:
            return sum(
                1
                for r in caplog.records
                if r.levelno == logging.INFO and "'max'" in r.message
            )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])

        with caplog.at_level(logging.INFO):
            provider.client.messages.with_raw_response.create = AsyncMock(
                return_value=_make_raw_mock()
            )
            asyncio.run(provider.complete(request))
            first_count = _notice_count()

            caplog.clear()
            provider.client.messages.with_raw_response.create = AsyncMock(
                return_value=_make_raw_mock()
            )
            asyncio.run(provider.complete(request))
            second_count = _notice_count()

        assert first_count == 1
        assert second_count == 0

    def test_different_requested_effort_gets_its_own_notice(self, caplog):
        """Same model, different requested effort -> separate notice fires."""
        request = ChatRequest(messages=[Message(role="user", content="Hello")])

        provider_max = _make_provider_with_effort(
            "max", default_model="claude-sonnet-5"
        )
        provider_max.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        with caplog.at_level(logging.INFO):
            asyncio.run(provider_max.complete(request))
        max_notices = [
            r for r in caplog.records if "'max'" in r.message and "xhigh" in r.message
        ]
        assert len(max_notices) == 1

        caplog.clear()

        # A different model+effort pair not previously seen must still notify.
        provider_opus47_max = _make_provider_with_effort(
            "max", default_model="claude-opus-4-7-20260416"
        )
        provider_opus47_max.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        with caplog.at_level(logging.INFO):
            asyncio.run(provider_opus47_max.complete(request))
        opus_notices = [
            r for r in caplog.records if "'max'" in r.message and "xhigh" in r.message
        ]
        assert len(opus_notices) == 1

    def test_different_model_gets_its_own_notice(self, caplog):
        """Same requested effort, different model -> separate notice fires."""
        request = ChatRequest(messages=[Message(role="user", content="Hello")])

        provider_sonnet = _make_provider_with_effort(
            "max", default_model="claude-sonnet-5"
        )
        provider_sonnet.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        with caplog.at_level(logging.INFO):
            asyncio.run(provider_sonnet.complete(request))
        sonnet_notices = [
            r
            for r in caplog.records
            if "claude-sonnet-5" in r.message and "supported range" in r.message
        ]
        assert len(sonnet_notices) == 1

        caplog.clear()

        provider_opus47 = _make_provider_with_effort(
            "max", default_model="claude-opus-4-7-20260416"
        )
        provider_opus47.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        with caplog.at_level(logging.INFO):
            asyncio.run(provider_opus47.complete(request))
        opus_notices = [
            r
            for r in caplog.records
            if "claude-opus-4-7-20260416" in r.message
            and "supported range" in r.message
        ]
        assert len(opus_notices) == 1


class TestClampEffortToSupportedHelper:
    """Direct unit tests for the _clamp_effort_to_supported helper.

    Covers the below-floor edge case (item 2 in the spec) which no current
    model capability table entry exercises end-to-end -- every real
    ``supported_efforts`` tuple today starts at "low", so we simulate a
    hypothetical model with a raised floor to prove the "up" branch works.
    """

    def test_exact_match_returns_exact(self):
        clamped, kind = anthropic_module._clamp_effort_to_supported(
            "high", ("low", "medium", "high", "xhigh")
        )
        assert (clamped, kind) == ("high", "exact")

    def test_above_ceiling_clamps_down(self):
        clamped, kind = anthropic_module._clamp_effort_to_supported(
            "max", ("low", "medium", "high", "xhigh")
        )
        assert (clamped, kind) == ("xhigh", "down")

    def test_below_floor_clamps_up(self):
        """Hypothetical model whose floor is 'high' -- 'low' clamps UP to 'high'."""
        clamped, kind = anthropic_module._clamp_effort_to_supported(
            "low", ("high", "xhigh", "max")
        )
        assert (clamped, kind) == ("high", "up")

    def test_below_floor_never_returns_none_when_supported_nonempty(self):
        """User intent is preserved -- never silently None when there's SOMETHING
        to clamp to, even in the below-floor case."""
        clamped, kind = anthropic_module._clamp_effort_to_supported(
            "medium", ("xhigh", "max")
        )
        assert clamped is not None
        assert kind == "up"
        assert clamped == "xhigh"

    def test_empty_supported_set_returns_none_unsupported(self):
        clamped, kind = anthropic_module._clamp_effort_to_supported("high", ())
        assert (clamped, kind) == (None, "unsupported")

    def test_unrecognized_effort_string_falls_back_to_highest_supported(self):
        clamped, kind = anthropic_module._clamp_effort_to_supported(
            "ultra", ("low", "medium", "high")
        )
        assert (clamped, kind) == ("high", "down")


class TestModelsFullySupportedEffortPassthrough:
    """Zero behavior change for models that fully support the requested tier.

    These mirror the existing test_config_effort_max_on_opus_48_sets_output_config
    test but add fable and confirm no downgrade notice fires when nothing was
    clamped.
    """

    @pytest.mark.parametrize(
        ("default_model", "effort"),
        [
            ("claude-opus-4-8", "max"),
            ("claude-fable-5", "max"),
            ("claude-sonnet-5", "xhigh"),
        ],
    )
    def test_no_downgrade_notice_when_fully_supported(
        self, default_model: str, effort: str, caplog
    ):
        provider = _make_provider_with_effort(effort, default_model=default_model)
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(messages=[Message(role="user", content="Hello")])

        with caplog.at_level(logging.INFO):
            asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"]["effort"] == effort
        assert not any(
            "supported range" in r.message or "exceeds" in r.message
            for r in caplog.records
        )
