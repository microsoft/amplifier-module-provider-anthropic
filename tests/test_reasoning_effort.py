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
    raw.parse = AsyncMock(return_value=DummyResponse())
    raw.headers = {}
    return raw


def _get_api_params(mock_create: AsyncMock) -> dict[str, Any]:
    """Extract the effective wire params for the API call.

    `extra_body` entries are merged up to the top level. Params like
    `temperature` and `speed` are not typed keyword arguments on the SDK
    (`temperature` was removed in anthropic 1.0.0; `speed` was never typed),
    so the provider sends them via `extra_body`. Tests assert on what reaches
    the API, not on which transport carried it.
    """
    assert mock_create.await_count == 1
    _, kwargs = mock_create.call_args
    params = dict(kwargs)
    extra_body = params.pop("extra_body", None) or {}
    for key, value in extra_body.items():
        params.setdefault(key, value)
    return params


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
# Undifferentiated effort warning (xhigh/max degenerate to 'high' when the
# model has no output_config support) -- warn-and-continue, params unchanged.
# ---------------------------------------------------------------------------


class TestUndifferentiatedEffortWarning:
    """On models without output_config support, the effort ladder maps
    'high', 'xhigh', and 'max' to identical thinking params -- the only thing
    that differentiates them (output_config.effort) never gets set. This is a
    silent no-op for both the request path (request.reasoning_effort) and the
    config path (config['reasoning_effort']/['effort']). The fix is a single
    warn-and-continue check, placed after reasoning_effort is fully resolved
    from either source, so it covers both paths without changing any params.
    """

    def test_xhigh_produces_byte_identical_params_to_high_on_sonnet_4_5(self):
        """Pins the degeneracy: xhigh and high produce identical wire params
        on a model without output_config support."""
        provider_high = _make_provider(default_model="claude-sonnet-4-5-20250929")
        provider_high.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request_high = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider_high.complete(request_high))
        params_high = _get_api_params(
            provider_high.client.messages.with_raw_response.create
        )

        provider_xhigh = _make_provider(default_model="claude-sonnet-4-5-20250929")
        provider_xhigh.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request_xhigh = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="xhigh",
        )
        asyncio.run(provider_xhigh.complete(request_xhigh))
        params_xhigh = _get_api_params(
            provider_xhigh.client.messages.with_raw_response.create
        )

        assert params_high == params_xhigh
        assert "output_config" not in params_high
        assert "output_config" not in params_xhigh

    def test_xhigh_logs_warning_on_sonnet_4_5(self, caplog):
        """xhigh on a model without output_config support logs the new
        warn-and-continue message."""
        import logging

        provider = _make_provider(default_model="claude-sonnet-4-5-20250929")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="xhigh",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))

        assert any(
            "has no effect on" in r.message and "xhigh" in r.message
            for r in caplog.records
        )

    def test_max_logs_warning_on_sonnet_4_5(self, caplog):
        """max on a model without output_config support logs the new
        warn-and-continue message."""
        import logging

        provider = _make_provider(default_model="claude-sonnet-4-5-20250929")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="max",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))

        assert any(
            "has no effect on" in r.message and "max" in r.message
            for r in caplog.records
        )

    def test_high_does_not_log_warning_on_sonnet_4_5(self, caplog):
        """'high' is the actual ceiling on this model -- no surprise, no warning."""
        import logging

        provider = _make_provider(default_model="claude-sonnet-4-5-20250929")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))

        assert not any("has no effect on" in r.message for r in caplog.records)

    def test_xhigh_on_sonnet_5_does_not_warn_and_still_sets_output_config(
        self, caplog
    ):
        """Regression guard: on a model WITH output_config support, xhigh is
        NOT degenerate -- no new warning, and output_config.effort is still
        set (behavior unchanged)."""
        import logging

        provider = _make_provider(default_model="claude-sonnet-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="xhigh",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))

        assert not any("has no effect on" in r.message for r in caplog.records)
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"] == {"effort": "xhigh"}

    def test_opus_47_max_still_logs_original_gate_b_warning_only(self, caplog):
        """Regression guard: Opus 4.7 supports output_config, so the NEW
        warning must not fire. The ORIGINAL Gate B warning ('max' not in
        Opus 4.7's supported_efforts) must still fire, and output_config must
        still be omitted -- no double-warning."""
        import logging

        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="max",  # not in supported_efforts for 4.7
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" not in params

        # Original Gate B warning still fires unchanged
        assert any(
            "not supported by" in r.message and "omitting output_config" in r.message
            for r in caplog.records
        )
        # New warn-and-continue must NOT double-fire (4.7 supports output_config)
        assert not any("has no effect on" in r.message for r in caplog.records)

    def test_xhigh_with_extended_thinking_false_does_not_warn(self, caplog):
        """xhigh + kwargs extended_thinking=False on Sonnet 4.5 (no
        output_config support): thinking ends up disabled, so
        reasoning_effort is consumed by nothing at all -- not the effort
        ladder, not output_config. The warning's claim ('resolves
        identically to high') would be false here, so it must NOT fire."""
        import logging

        provider = _make_provider(default_model="claude-sonnet-4-5-20250929")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="xhigh",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request, extended_thinking=False))

        assert not any("has no effect on" in r.message for r in caplog.records)
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params
        assert "output_config" not in params

    def test_xhigh_on_haiku_35_does_not_warn(self, caplog):
        """xhigh on Haiku 3.5 (no thinking support at all, no output_config
        support): reasoning_effort is consumed by nothing -- the effort
        ladder never runs (thinking_enabled is forced False) and
        output_config is unavailable. The warning must NOT fire."""
        import logging

        provider = _make_provider(default_model="claude-haiku-3-5-20250929")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="xhigh",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))

        assert not any("has no effect on" in r.message for r in caplog.records)
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params

    def test_xhigh_with_thinking_active_still_warns_on_sonnet_4_5(self, caplog):
        """Regression guard: xhigh with thinking actually active (explicit
        extended_thinking=True) on Sonnet 4.5 still logs the warning -- proves
        the thinking_enabled guard didn't just delete the feature, only
        narrowed it to the case where the effort ladder actually runs."""
        import logging

        provider = _make_provider(default_model="claude-sonnet-4-5-20250929")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="xhigh",
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request, extended_thinking=True))

        assert any(
            "has no effect on" in r.message and "xhigh" in r.message
            for r in caplog.records
        )
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" in params
        assert "output_config" not in params


# ---------------------------------------------------------------------------
# kwargs["extended_thinking"]=False must fully suppress output_config.effort
# too, not just the `thinking` param block.
#
# Bug: an ambient/backfilled reasoning_effort (from provider config or
# request.reasoning_effort) was leaking into output_config.effort even when
# the caller explicitly opted out of thinking via extended_thinking=False.
# On models with supports_output_config, output_config.effort IS the
# thinking control surface, so this silently defeated the caller's explicit
# "no reasoning on this call" opt-out. An explicit kwargs["effort"] override
# still wins -- it's a deliberate per-call output_config-only request, not
# an ambient default.
# ---------------------------------------------------------------------------


class TestExtendedThinkingFalseSuppressesOutputConfig:
    """extended_thinking=False must omit output_config.effort too, unless
    the caller also passed an explicit kwargs["effort"] override."""

    def test_ambient_config_xhigh_with_extended_thinking_false_omits_output_config(
        self,
    ):
        """Bug case: ambient config reasoning_effort='xhigh' + kwargs
        extended_thinking=False on claude-sonnet-5 (supports_output_config)
        -> output_config is ABSENT, not backfilled with 'xhigh'."""
        provider = _make_provider_with_config(
            {"reasoning_effort": "xhigh"}, default_model="claude-sonnet-5"
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request, extended_thinking=False))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" not in params
        assert "thinking" not in params

    def test_request_reasoning_effort_xhigh_with_extended_thinking_false_omits_output_config(
        self,
    ):
        """Same bug via the request path: request.reasoning_effort='xhigh'
        + kwargs extended_thinking=False -> output_config ABSENT."""
        provider = _make_provider(default_model="claude-sonnet-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="xhigh",
        )
        asyncio.run(provider.complete(request, extended_thinking=False))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" not in params
        assert "thinking" not in params

    def test_explicit_effort_kwarg_still_wins_despite_thinking_opt_out(self):
        """An explicit kwargs['effort'] is a deliberate output_config-only
        override and must win even when extended_thinking=False."""
        provider = _make_provider_with_config(
            {"reasoning_effort": "xhigh"}, default_model="claude-sonnet-5"
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(
            provider.complete(request, extended_thinking=False, effort="high")
        )

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"] == {"effort": "high"}

    def test_no_extended_thinking_kwarg_unchanged_behavior(self):
        """Regression guard: without an explicit extended_thinking kwarg,
        ambient reasoning_effort='xhigh' still sets output_config.effort
        exactly as before this fix."""
        provider = _make_provider_with_config(
            {"reasoning_effort": "xhigh"}, default_model="claude-sonnet-5"
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"] == {"effort": "xhigh"}

    def test_extended_thinking_true_unchanged_behavior(self):
        """Regression guard: an explicit extended_thinking=True (opt-IN)
        does not suppress output_config.effort."""
        provider = _make_provider_with_config(
            {"reasoning_effort": "xhigh"}, default_model="claude-sonnet-5"
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request, extended_thinking=True))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"] == {"effort": "xhigh"}
