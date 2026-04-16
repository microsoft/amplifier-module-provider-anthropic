"""Tests for Claude Opus 4.7 support.

Phase 1: Validates capability detection, manual-thinking fallback,
and 1M beta header fix.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider, _RuntimeModelInfo


# ---------------------------------------------------------------------------
# Helpers (same infrastructure as test_reasoning_effort.py)
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
        self.model = "claude-opus-4-7-20260416"


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

    def test_opus_unknown_gets_1m_header(self):
        assert self._check("claude-opus-latest") is True

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

    def test_opus_47_invalid_effort_omits_output_config(self):
        """Unknown effort level → output_config omitted (not a hard error)."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="max",  # not in supported_efforts for 4.7
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
