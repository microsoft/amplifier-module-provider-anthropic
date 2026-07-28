"""Tests for Claude Opus 4.7 support.

Phase 1: Validates capability detection, manual-thinking fallback,
and 1M beta header fix.
"""

import asyncio
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
import amplifier_module_provider_anthropic as anthropic_module
from amplifier_module_provider_anthropic import AnthropicProvider, _RuntimeModelInfo

from tests._helpers import DummyResponse, FakeCoordinator


# ---------------------------------------------------------------------------
# Helpers (same infrastructure as test_reasoning_effort.py)
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
    raw.parse.return_value = DummyResponse(model="claude-opus-4-7-20260416")
    raw.headers = {}
    return raw


def _get_api_params(mock_create: AsyncMock) -> dict[str, Any]:
    """Extract the kwargs passed to the API call."""
    assert mock_create.await_count == 1
    _, kwargs = mock_create.call_args
    return kwargs


# ---------------------------------------------------------------------------
# TestOpus47Capabilities — ModelCapabilities for Opus 4.7 models
# ---------------------------------------------------------------------------


class TestOpus47Capabilities:
    """ModelCapabilities for Opus 4.7 models."""

    def test_opus_47_supports_manual_thinking_false(self):
        """Opus 4.7 rejects type='enabled' — supports_manual_thinking must be False."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_manual_thinking is False

    def test_opus_47_supports_adaptive_thinking_true(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_adaptive_thinking is True

    def test_opus_47_max_output_tokens_128k(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.max_output_tokens == 128000

    def test_opus_47_supports_1m(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_1m is True

    def test_opus_47_default_thinking_budget_64k(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.default_thinking_budget == 64000

    def test_opus_47_family(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.family == "opus"

    def test_opus_47_supports_thinking(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_thinking is True

    def test_opus_46_still_supports_manual_thinking(self):
        """Opus 4.6 must retain manual thinking support (backward compat)."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.supports_manual_thinking is True

    def test_opus_45_still_supports_manual_thinking(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.supports_manual_thinking is True

    def test_opus_unknown_assumes_no_manual_thinking(self):
        """Unknown Opus → latest → no manual thinking."""
        caps = AnthropicProvider._get_capabilities("claude-opus-latest")
        assert caps.supports_manual_thinking is False

    def test_sonnet_unaffected(self):
        """Sonnet models retain manual thinking support."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-6-20260101")
        assert caps.supports_manual_thinking is True

    def test_haiku_unaffected(self):
        """Haiku models retain manual thinking support."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.supports_manual_thinking is True


# ---------------------------------------------------------------------------
# TestOpus47ThinkingFallback — thinking forced to adaptive on Opus 4.7
# ---------------------------------------------------------------------------


class TestOpus47ThinkingFallback:
    """Thinking config forced to adaptive on Opus 4.7."""

    def test_opus_47_low_effort_forces_adaptive(self):
        """reasoning_effort='low' on 4.7 → type='adaptive' (not 'enabled')."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="low",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in params["thinking"]

    def test_opus_47_medium_effort_uses_adaptive(self):
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="medium",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "adaptive"

    def test_opus_47_high_effort_uses_adaptive(self):
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "adaptive"

    def test_opus_47_config_thinking_type_enabled_forces_adaptive(self):
        """Even if config says thinking_type='enabled', 4.7 forces adaptive.

        COE FIX #1: Uses extended_thinking=True kwarg WITHOUT reasoning_effort.
        Do NOT use reasoning_effort='high' — it triggers the adaptive path via
        effort_thinking_type before the new elif branch is reached.
        """
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.config["thinking_type"] = "enabled"
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request, extended_thinking=True))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in params["thinking"]

    def test_opus_47_max_tokens_still_generous(self):
        """max_tokens ceiling calculation still works with forced adaptive."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["max_tokens"] >= 64000

    def test_opus_46_low_effort_still_uses_enabled(self):
        """Opus 4.6 + low → type='enabled', budget=4096 (backward compat)."""
        provider = _make_provider(default_model="claude-opus-4-6-20260101")
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

    def test_opus_47_extended_thinking_kwarg_forces_adaptive(self):
        """Old-style extended_thinking=True kwarg on 4.7 → adaptive (not enabled).

        COE FIX #3: Tests that the extended_thinking=True kwarg path (no reasoning_effort,
        default thinking_type='adaptive') still works correctly on Opus 4.7.
        """
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request, extended_thinking=True))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in params["thinking"]

    def test_apply_runtime_overrides_preserves_manual_thinking(self):
        """_apply_runtime_capability_overrides must not reset supports_manual_thinking to default.

        COE FIX #2: Tests the construction path in _apply_runtime_capability_overrides
        using a non-None _RuntimeModelInfo (all-None values trigger the ModelCapabilities
        construction path rather than the early-return path).
        """
        base_caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert base_caps.supports_manual_thinking is False
        # Use _RuntimeModelInfo() with all-None values to trigger the construction path
        # (not the early-return path that happens when runtime_info is None)
        runtime_info = _RuntimeModelInfo()
        overridden = AnthropicProvider._apply_runtime_capability_overrides(
            base_caps, runtime_info
        )
        assert overridden.supports_manual_thinking is False


# ---------------------------------------------------------------------------
# TestBetaHeader1MFix — 1M context beta header uses >= instead of ==
# ---------------------------------------------------------------------------


class TestBetaHeader1MFix:
    """1M context beta header uses >= instead of ==."""

    def _check(self, model_id: str) -> bool:
        provider = _make_provider(default_model=model_id)
        caps = AnthropicProvider._get_capabilities(model_id)
        return provider._should_add_context_1m_beta(model_id, caps)

    def test_opus_46_gets_1m_header(self):
        assert self._check("claude-opus-4-6-20260101") is True

    def test_opus_47_gets_1m_header(self):
        assert self._check("claude-opus-4-7-20260416") is True

    def test_opus_unknown_no_1m_header(self):
        # Unknown opus versions assume latest (4.8+) where 1M is GA — no header needed.
        assert self._check("claude-opus-latest") is False

    def test_opus_45_no_1m_header(self):
        assert self._check("claude-opus-4-5-20251101") is False

    def test_haiku_never_gets_1m_header(self):
        assert self._check("claude-haiku-4-5-20251001") is False

    def test_sonnet_46_gets_1m_header(self):
        assert self._check("claude-sonnet-4-6-20260101") is True

    def test_sonnet_45_gets_1m_header(self):
        assert self._check("claude-sonnet-4-5-20250929") is True

    def test_sonnet_unknown_gets_1m_header(self):
        assert self._check("claude-sonnet-latest") is True


# ---------------------------------------------------------------------------
# TestOpus47OutputConfig — output_config.effort on Opus 4.7
# ---------------------------------------------------------------------------


class TestOpus47OutputConfig:
    """output_config.effort on Opus 4.7."""

    def test_opus_47_high_effort_sends_output_config(self):
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"] == {"effort": "high"}

    def test_opus_47_xhigh_effort_sends_output_config(self):
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="xhigh",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"] == {"effort": "xhigh"}

    def test_opus_47_low_effort_sends_output_config(self):
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="low",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"] == {"effort": "low"}

    def test_opus_47_no_effort_no_output_config(self):
        """reasoning_effort=None → no output_config at all."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" not in params

    def test_opus_46_no_output_config(self):
        """Opus 4.6 doesn't support output_config — never sent."""
        provider = _make_provider(default_model="claude-opus-4-6-20260101")
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

    def test_opus_47_supported_efforts(self):
        """Opus 4.7 capabilities include xhigh."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert "xhigh" in caps.supported_efforts
        assert caps.supported_efforts == ("low", "medium", "high", "xhigh")

    def test_opus_46_no_xhigh(self):
        """Opus 4.6 capabilities don't include xhigh."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert "xhigh" not in caps.supported_efforts

    def test_opus_47_max_effort_clamps_to_xhigh(self):
        """effort='max' on Opus 4.7 is CLAMPED to 'xhigh' (its ceiling), not
        omitted (amplifier-support#289). 'max' is a recognized value on the
        EFFORT_ORDER ladder (Opus 4.8+ and Fable 5 support it) -- Opus 4.7
        just doesn't have that tier yet, so the highest tier <= 'max' that
        4.7 DOES support ("xhigh") is used instead of silently dropping
        output_config.effort and letting the API apply its own default.

        UPDATED CONTRACT (was test_opus_47_invalid_effort_omits_output_config):
        prior to the amplifier-support#289 clamp fix this asserted
        'output_config' not in params for the same input -- that omit-on-
        mismatch behavior was exactly the silent-downgrade bug being fixed.
        """
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="max",  # not in supported_efforts for 4.7 -> clamped
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"] == {"effort": "xhigh"}

    def test_opus_47_unknown_effort_string_omits_output_config(self):
        """A genuinely unrecognized effort value (not anywhere on the
        EFFORT_ORDER ladder) is still omitted with a warning -- clamping
        only applies to recognized values the ACTIVE model doesn't support,
        never to values the provider doesn't understand at all."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="ultra",  # not a real effort value at all
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" not in params


# ---------------------------------------------------------------------------
# TestOpus47ThinkingDisplay — thinking.display integration for Opus 4.7
# ---------------------------------------------------------------------------


class TestOpus47ThinkingDisplay:
    """thinking.display integration for Opus 4.7."""

    def test_opus_47_thinking_sends_display_summarized(self):
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["display"] == "summarized"

    def test_opus_47_display_config_override(self):
        """Config thinking_display='omitted' overrides default."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.config["thinking_display"] = "omitted"
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["display"] == "omitted"

    def test_opus_47_display_kwargs_override(self):
        """kwargs thinking_display overrides config."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.config["thinking_display"] = "omitted"
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request, thinking_display="summarized"))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["display"] == "summarized"

    def test_opus_46_no_display_field(self):
        """Opus 4.6 thinking dict has no display field."""
        provider = _make_provider(default_model="claude-opus-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "display" not in params["thinking"]

    def test_opus_47_thinking_display_required_flag(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.thinking_display_required is True

    def test_opus_46_thinking_display_required_false(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.thinking_display_required is False


# ---------------------------------------------------------------------------
# TestOpus47Temperature — temperature stripping for non-sampling models
# ---------------------------------------------------------------------------


class TestOpus47Temperature:
    """Temperature stripping for non-sampling models."""

    def test_opus_47_no_temperature_in_params(self):
        """Opus 4.7 requests should not include temperature."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "temperature" not in params

    def test_opus_47_explicit_temperature_ignored(self):
        """Even if user sets temperature, Opus 4.7 omits it."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            temperature=0.5,
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "temperature" not in params

    def test_opus_47_thinking_does_not_force_temperature_1(self):
        """With thinking on 4.7, temperature is omitted (not forced to 1.0)."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "temperature" not in params

    def test_opus_46_temperature_still_sent(self):
        """Opus 4.6 still includes temperature in params."""
        provider = _make_provider(default_model="claude-opus-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "temperature" in params

    def test_opus_47_supports_sampling_false(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_sampling is False

    def test_opus_46_supports_sampling_true(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.supports_sampling is True

    def test_sonnet_supports_sampling_true(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-6-20260101")
        assert caps.supports_sampling is True


# ---------------------------------------------------------------------------
# TestTemperatureZeroBug — temperature=0.0 must not be treated as falsy
# ---------------------------------------------------------------------------


class TestTemperatureZeroBug:
    """temperature=0.0 must be respected, not treated as falsy."""

    def test_temperature_zero_is_respected(self):
        """request.temperature=0.0 should send 0.0, not fall back to default 0.7."""
        provider = _make_provider(default_model="claude-sonnet-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            temperature=0.0,
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["temperature"] == 0.0

    def test_temperature_none_falls_back_to_default(self):
        """request.temperature=None should fall back to provider default (0.7)."""
        provider = _make_provider(default_model="claude-sonnet-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["temperature"] == 0.7

    def test_temperature_explicit_value_sent(self):
        """request.temperature=0.5 should send 0.5."""
        provider = _make_provider(default_model="claude-sonnet-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            temperature=0.5,
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["temperature"] == 0.5


# ---------------------------------------------------------------------------
# TestTokenizerBufferBump — default thinking buffer increased for Opus 4.7
# ---------------------------------------------------------------------------


class TestTokenizerBufferBump:
    """Default thinking_budget_buffer bumped from 4096 to 8192."""

    def test_default_buffer_is_8192(self):
        """Default buffer_tokens should be 8192 (not the old 4096)."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        # max_tokens = min(budget + buffer, ceiling) = min(64000 + 8192, 128000) = 72192
        # Old behavior: min(64000 + 4096, 128000) = 68096
        assert params["max_tokens"] >= 72192

    def test_config_buffer_override_still_works(self):
        """Config thinking_budget_buffer overrides the new default."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.config["thinking_budget_buffer"] = 16384
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        # min(64000 + 16384, 128000) = 80384
        assert params["max_tokens"] >= 80384


# ---------------------------------------------------------------------------
# TestDeprecationWarnings — warn once per process for deprecated models
# ---------------------------------------------------------------------------


class TestDeprecationWarnings:
    """Deprecation warnings for models approaching retirement."""

    def setup_method(self):
        """Clear warned set before each test."""
        anthropic_module._clear_deprecated_model_warnings()

    def test_deprecated_model_emits_warning(self, caplog):
        """Deprecated model emits a logger.warning on first use."""
        provider = _make_provider(default_model="claude-3-haiku-20240307")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))
        assert any("deprecated" in r.message.lower() for r in caplog.records)
        assert any("2026-04-19" in r.message for r in caplog.records)

    def test_warning_only_emitted_once(self, caplog):
        """Second call with same deprecated model does NOT warn again."""
        provider = _make_provider(default_model="claude-3-haiku-20240307")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))
        first_count = sum(
            1 for r in caplog.records if "deprecated" in r.message.lower()
        )

        caplog.clear()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))
        second_count = sum(
            1 for r in caplog.records if "deprecated" in r.message.lower()
        )
        assert first_count == 1
        assert second_count == 0

    def test_non_deprecated_model_no_warning(self, caplog):
        """Non-deprecated model emits no deprecation warning."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))
        assert not any("deprecated" in r.message.lower() for r in caplog.records)

    def test_deprecated_models_table_has_expected_entries(self):
        """Verify the deprecation table contains all known deprecated models."""
        deprecated = anthropic_module._DEPRECATED_MODELS
        assert "claude-3-haiku-20240307" in deprecated
        assert "claude-sonnet-4-20250514" in deprecated
        assert "claude-opus-4-20250514" in deprecated
        assert len(deprecated) == 3

    def test_clear_function_resets_warned_set(self):
        """_clear_deprecated_model_warnings() allows re-warning."""
        provider = _make_provider(default_model="claude-3-haiku-20240307")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request))

        anthropic_module._clear_deprecated_model_warnings()

        # After clearing, the next call should warn again (verified by checking
        # the set is empty — the warn-once test already covers the actual logging)
        assert len(anthropic_module._warned_deprecated_models) == 0


# ---------------------------------------------------------------------------
# TestTaskBudgets — task budget beta feature for Opus 4.7+
# ---------------------------------------------------------------------------


class TestTaskBudgets:
    """Task budget support (beta) for Opus 4.7+."""

    def test_opus_47_supports_task_budget(self):
        """Opus 4.7 capabilities include supports_task_budget=True."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_task_budget is True

    def test_opus_46_no_task_budget(self):
        """Opus 4.6 does not support task budgets."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.supports_task_budget is False

    def test_sonnet_no_task_budget(self):
        """Sonnet does not support task budgets."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-6-20260101")
        assert caps.supports_task_budget is False

    def test_task_budget_in_output_config(self):
        """task_budget_tokens kwarg adds task_budget to output_config."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request, task_budget_tokens=50000))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" in params
        assert params["output_config"]["task_budget"] == {
            "type": "tokens",
            "total": 50000,
        }

    def test_task_budget_min_20k_enforced(self):
        """Task budget below 20000 is clamped to 20000."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request, task_budget_tokens=5000))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"]["task_budget"]["total"] == 20000

    def test_task_budget_from_config(self):
        """Config-level task_budget_tokens is used when kwarg not provided."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.config["task_budget_tokens"] = 80000
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"]["task_budget"]["total"] == 80000

    def test_task_budget_adds_beta_header(self):
        """When task_budget is present, the beta header must be included."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request, task_budget_tokens=50000))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        beta_header = params.get("extra_headers", {}).get("anthropic-beta", "")
        assert "task-budgets-2026-03-13" in beta_header

    def test_no_task_budget_no_beta_header(self):
        """When task_budget is not set, the task-budgets beta header is absent."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        beta_header = params.get("extra_headers", {}).get("anthropic-beta", "")
        assert "task-budgets-2026-03-13" not in beta_header

    def test_task_budget_ignored_on_unsupported_model(self):
        """task_budget_tokens on Opus 4.6 (unsupported) is silently ignored."""
        provider = _make_provider(default_model="claude-opus-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request, task_budget_tokens=50000))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        # output_config should not exist for 4.6 at all
        assert "output_config" not in params

    def test_runtime_overrides_preserve_task_budget(self):
        """_apply_runtime_capability_overrides passes through supports_task_budget."""
        base_caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert base_caps.supports_task_budget is True
        runtime_info = _RuntimeModelInfo()
        overridden = AnthropicProvider._apply_runtime_capability_overrides(
            base_caps, runtime_info
        )
        assert overridden.supports_task_budget is True


# ---------------------------------------------------------------------------
# TestSonnet5Temperature — temperature stripping for claude-sonnet-5
# Regression guard for amplifier-support#299: sonnet-5 must NOT receive
# `temperature` (Anthropic rejects it: "temperature is deprecated for this
# model."). Mirrors TestOpus47Temperature; sonnet-4.x must be unaffected.
# ---------------------------------------------------------------------------


class TestSonnet5Temperature:
    """Temperature stripping for claude-sonnet-5 (amplifier-support#299)."""

    def test_sonnet_5_no_temperature_in_params(self):
        """No-thinking sonnet-5 request (the session-naming call) omits temperature."""
        provider = _make_provider(default_model="claude-sonnet-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(messages=[Message(role="user", content="Name this session")])
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "temperature" not in params

    def test_sonnet_5_explicit_temperature_ignored(self):
        """Even if the caller sets temperature, sonnet-5 omits it."""
        provider = _make_provider(default_model="claude-sonnet-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            temperature=0.7,
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "temperature" not in params

    def test_sonnet_5_thinking_does_not_force_temperature_1(self):
        """With thinking on sonnet-5, temperature is omitted (not forced to 1.0)."""
        provider = _make_provider(default_model="claude-sonnet-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "temperature" not in params

    def test_sonnet_46_temperature_still_sent(self):
        """Regression guard: sonnet-4.6 still includes temperature (Anthropic accepts it)."""
        provider = _make_provider(default_model="claude-sonnet-4-6")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(messages=[Message(role="user", content="Hello")])
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "temperature" in params

    def test_sonnet_5_supports_sampling_false(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-5")
        assert caps.supports_sampling is False

    def test_sonnet_46_supports_sampling_true(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-6")
        assert caps.supports_sampling is True
