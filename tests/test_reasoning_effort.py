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


# ---------------------------------------------------------------------------
# Effort clamping to the model's highest supported tier (amplifier-support#289)
#
# Root cause: config['effort'] / request.reasoning_effort is validated
# against the GLOBAL legal ladder (every value in EFFORT_ORDER), not against
# the ACTIVE model's capability tier. A value that passed global validation
# (e.g. "max") can still be unsupported by the model actually handling the
# request -- provider config is intentionally model-agnostic, so this
# mismatch is the NORMAL case in routing-matrix / mid-session model-switch
# deployments, not an edge case. Before this fix, the provider warned on
# EVERY request and omitted output_config.effort entirely, silently letting
# the API apply its own server-side default effort instead of the model's
# actual ceiling. The fix clamps to the highest supported tier <= what was
# requested (e.g. "max" -> "xhigh" on claude-sonnet-5) and only logs the
# downgrade once per (model, requested-effort) pair.
# ---------------------------------------------------------------------------


class TestEffortClampToSupportedTier:
    """output_config.effort clamps to the model's ceiling instead of being
    omitted when the requested/configured effort exceeds it."""

    def test_sonnet_5_max_clamps_to_xhigh(self):
        """Headline case from amplifier-support#289: effort='max' on
        claude-sonnet-5 clamps to 'xhigh' (sonnet-5's actual ceiling)
        instead of being omitted."""
        provider = _make_provider(default_model="claude-sonnet-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="max",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"]["effort"] == "xhigh"

    def test_fable_5_max_passes_through_unchanged(self):
        """Fable 5 declares 'max' in supported_efforts -- no clamping, the
        happy-path pass-through branch is untouched by this fix."""
        provider = _make_provider(default_model="claude-fable-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="max",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"]["effort"] == "max"

    def test_opus_48_max_passes_through_unchanged(self):
        """Opus 4.8+ declares 'max' in supported_efforts -- no clamping."""
        provider = _make_provider(default_model="claude-opus-4-8-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="max",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"]["effort"] == "max"

    def test_model_without_output_config_support_still_omits_key(self):
        """Models that don't support output_config AT ALL (e.g. sonnet-4-6)
        must never gain an output_config key -- clamping only kicks in when
        supports_output_config is already True for the model."""
        provider = _make_provider(default_model="claude-sonnet-4-6")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" not in params

    def test_unknown_effort_string_is_omitted_not_clamped(self, caplog):
        """A value that isn't on the EFFORT_ORDER ladder at all (typo/unknown)
        is NOT guessed at -- preserve the original warn-and-omit behavior
        rather than clamping to an arbitrary tier."""
        provider = _make_provider(default_model="claude-sonnet-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="ultra",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" not in params
        assert any("not supported by" in r.message for r in caplog.records)

    def test_kwargs_effort_precedence_is_also_clamped(self):
        """kwargs['effort'] overrides reasoning_effort for output_config
        purposes (see the precedence comment above the block) -- an
        unsupported kwargs value must be clamped too, not just values that
        arrive via reasoning_effort."""
        provider = _make_provider(default_model="claude-sonnet-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="low",  # would stay "low" if kwargs didn't win
        )
        asyncio.run(provider.complete(request, effort="max"))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"]["effort"] == "xhigh"


class TestEffortDowngradeLoggedOnce:
    """The clamp-downgrade notice logs once per (model, requested-effort)
    pair at INFO level, not as a per-request WARNING (the original bug
    report: an identical warning line on every single sonnet-5 request)."""

    def setup_method(self):
        """Clear the seen-pairs set before each test so prior tests/classes
        (e.g. TestEffortClampToSupportedTier, which also triggers clamping)
        can't leave state that makes the "logs on first use" assertion below
        flaky. Mirrors _clear_deprecated_model_warnings() in test_opus_47.py.
        """
        anthropic_module._clear_effort_downgrade_notices()

    def test_downgrade_logged_once_for_repeated_requests(self, caplog):
        """Two consecutive requests with the same (model, requested effort)
        log the downgrade notice exactly once."""
        provider = _make_provider(default_model="claude-sonnet-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="max",
        )

        with caplog.at_level(logging.INFO):
            asyncio.run(provider.complete(request))
        first_count = sum(1 for r in caplog.records if "clamp" in r.message.lower())
        assert first_count == 1

        caplog.clear()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        with caplog.at_level(logging.INFO):
            asyncio.run(provider.complete(request))
        second_count = sum(1 for r in caplog.records if "clamp" in r.message.lower())
        assert second_count == 0

    def test_downgrade_logged_again_for_a_different_pair(self, caplog):
        """A different (model, effort) pair logs its own downgrade notice
        even though a prior pair was already logged and suppressed."""
        provider = _make_provider(default_model="claude-sonnet-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request_max = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="max",
        )
        with caplog.at_level(logging.INFO):
            asyncio.run(provider.complete(request_max))
        caplog.clear()

        # Different model family (Opus 4.7 also lacks "max") -> a distinct
        # (model, effort) pair, so it must log again despite the sonnet-5
        # pair above already being marked as seen.
        provider2 = _make_provider(default_model="claude-opus-4-7-20260416")
        provider2.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request2 = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="max",
        )
        with caplog.at_level(logging.INFO):
            asyncio.run(provider2.complete(request2))
        second_count = sum(1 for r in caplog.records if "clamp" in r.message.lower())
        assert second_count == 1


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
