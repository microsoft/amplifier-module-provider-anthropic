"""Tests for Claude Fable 5.1 support.

Covers:
  (a) Capability detection: claude-fable-5-1 is detected as the fable family
  (b) Capability matrix: same as Fable 5 (128K output, 1M context, adaptive
      thinking always-on, all effort levels, no speed mode, no manual thinking)
  (c) Cost: input/output rates identical to Fable 5; cache_read is 75% cheaper
      ($0.25/MTok vs $1.00/MTok on Fable 5)
  (d) claude-fable-5-1 is NOT in _FAST_ELIGIBLE_MODELS
  (e) Fallback ladder: claude-fable-5-1 steps down to opus (same as Fable 5)
  (f) Version detection: (5, 1) parsed correctly from the model ID
"""

from decimal import Decimal
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._cost import (
    _FAST_ELIGIBLE_MODELS,
    compute_cost,
)
from tests._helpers import DummyResponse, FakeCoordinator


def _make_provider(default_model: str = "claude-fable-5-1") -> AnthropicProvider:
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


def _make_raw_mock():
    from unittest.mock import MagicMock

    raw = MagicMock()
    raw.parse = AsyncMock(return_value=DummyResponse())
    raw.headers = {}
    return raw


# ---------------------------------------------------------------------------
# (a) Family detection
# ---------------------------------------------------------------------------


class TestFable51FamilyDetection:
    """claude-fable-5-1 is detected as the fable family."""

    def test_family_detected_as_fable(self):
        assert AnthropicProvider._detect_family("claude-fable-5-1") == "fable"

    def test_version_parsed_correctly(self):
        """(5, 1) must be parsed from claude-fable-5-1."""
        assert AnthropicProvider._detect_version("claude-fable-5-1", "fable") == (5, 1)


# ---------------------------------------------------------------------------
# (b) Capability matrix
# ---------------------------------------------------------------------------


class TestFable51Capabilities:
    """Fable 5.1 inherits the fable family capability branch.

    All assertions here match the Fable 5 capability matrix documented in
    TestGetCapabilitiesFable5 (test_model_capabilities.py) -- Fable 5.1 is
    the same model surface with a different API identifier.
    """

    def test_family_tag(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.family == "fable"

    def test_max_output_128k(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.max_output_tokens == 128000

    def test_supports_1m_context(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.supports_1m is True

    def test_thinking_always_on(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.thinking_always_on is True

    def test_supports_adaptive_thinking(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.supports_adaptive_thinking is True

    def test_no_manual_thinking(self):
        """Manual thinking (budget_tokens) is not accepted on Fable 5.1."""
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.supports_manual_thinking is False

    def test_all_effort_levels(self):
        """Fable 5.1 supports all 5 effort levels including xhigh and max."""
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.supported_efforts == ("low", "medium", "high", "xhigh", "max")

    def test_no_speed_mode(self):
        """Speed mode is not supported on Fable 5.1."""
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.supports_speed is False

    def test_supports_inline_system(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.supports_inline_system is True

    def test_thinking_display_required(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.thinking_display_required is True

    def test_no_sampling(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.supports_sampling is False

    def test_supports_output_config(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.supports_output_config is True

    def test_supports_task_budget(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert caps.supports_task_budget is True

    def test_capability_tags(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
        assert "thinking" in caps.capability_tags
        assert "tools" in caps.capability_tags
        assert "streaming" in caps.capability_tags
        assert "vision" in caps.capability_tags

    def test_no_thinking_param_sent_with_reasoning_effort(self):
        """claude-fable-5-1 + reasoning_effort='high' must NOT send thinking param.

        Fable 5.1 has thinking always on -- the API controls it implicitly.
        Sending {type:disabled} (or any explicit thinking param) causes HTTP 400.
        """
        import asyncio

        provider = _make_provider(default_model="claude-fable-5-1")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))

        mock_create = provider.client.messages.with_raw_response.create
        assert mock_create.await_count == 1
        _, kwargs = mock_create.call_args
        params = dict(kwargs)
        extra_body = params.pop("extra_body", None) or {}
        for key, value in extra_body.items():
            params.setdefault(key, value)
        assert "thinking" not in params


# ---------------------------------------------------------------------------
# (c) Cost: cache_read is 75% cheaper than Fable 5
# ---------------------------------------------------------------------------


class TestFable51Cost:
    """Fable 5.1 pricing: same input/output as Fable 5, cache_read 75% cheaper."""

    def test_input_tokens_cost(self):
        """claude-fable-5-1: 1M input -> $10.00 (same as Fable 5)."""
        result = compute_cost("claude-fable-5-1", input_tokens=1_000_000)
        assert result == Decimal("10.00"), f"Expected Decimal('10.00'), got {result!r}"

    def test_output_tokens_cost(self):
        """claude-fable-5-1: 1M output -> $50.00 (same as Fable 5)."""
        result = compute_cost("claude-fable-5-1", output_tokens=1_000_000)
        assert result == Decimal("50.00"), f"Expected Decimal('50.00'), got {result!r}"

    def test_cache_read_cost_75pct_cheaper_than_fable5(self):
        """claude-fable-5-1: cache_read is $0.25/MTok (75% cheaper than Fable 5's $1.00/MTok).

        Source: https://www.anthropic.com/claude-fable-and-mythos-5-1
        "Cache reads now cost 75% less, or $0.25 per million tokens."
        """
        result = compute_cost("claude-fable-5-1", cache_read_input_tokens=1_000_000)
        assert result == Decimal("0.25"), f"Expected Decimal('0.25'), got {result!r}"

    def test_cache_read_is_75pct_cheaper_than_fable5(self):
        """Fable 5.1 cache_read is exactly 25% of Fable 5's cache_read rate."""
        fable5_read = compute_cost("claude-fable-5", cache_read_input_tokens=1_000_000)
        fable51_read = compute_cost(
            "claude-fable-5-1", cache_read_input_tokens=1_000_000
        )
        assert fable5_read is not None, "claude-fable-5 must be in _RATES"
        assert fable51_read is not None, "claude-fable-5-1 must be in _RATES"
        # 75% cheaper means 25% of the original price
        assert fable51_read == fable5_read * Decimal("0.25")

    def test_cache_write_5m_cost(self):
        """claude-fable-5-1: 1M cache write (5-min) -> $12.50 (same as Fable 5)."""
        result = compute_cost("claude-fable-5-1", cache_creation_input_tokens=1_000_000)
        assert result == Decimal("12.50"), f"Expected Decimal('12.50'), got {result!r}"

    def test_cache_write_1h_cost(self):
        """claude-fable-5-1: 1M cache write (1-hour) -> $20.00 (2x input rate)."""
        result = compute_cost(
            "claude-fable-5-1",
            cache_creation_1h_input_tokens=1_000_000,
            cache_creation_5m_input_tokens=0,
        )
        assert result == Decimal("20.00"), f"Expected Decimal('20.00'), got {result!r}"

    def test_unknown_model_returns_none(self):
        """Sanity check: a non-existent model still returns None."""
        result = compute_cost("claude-fable-5-1-does-not-exist", input_tokens=1_000)
        assert result is None


# ---------------------------------------------------------------------------
# (d) Not in fast-eligible models
# ---------------------------------------------------------------------------


def test_fable51_not_in_fast_eligible_models():
    """claude-fable-5-1 must NOT be in _FAST_ELIGIBLE_MODELS (no speed mode)."""
    assert "claude-fable-5-1" not in _FAST_ELIGIBLE_MODELS


# ---------------------------------------------------------------------------
# (e) Fallback ladder
# ---------------------------------------------------------------------------


class TestFable51FallbackLadder:
    """Fable 5.1 steps down to opus on the fallback ladder (same as Fable 5)."""

    def test_fable51_fallback_target_is_opus(self):
        """claude-fable-5-1 must fall back to the opus backstop (claude-opus-5)."""
        provider = _make_provider("claude-fable-5-1")
        target = provider._fallback_target_for_model("claude-fable-5-1")
        assert target == "claude-opus-5"

    def test_fable51_and_fable5_same_fallback_family(self):
        """Both Fable 5 and Fable 5.1 share the same fable->opus fallback family."""
        provider = _make_provider("claude-fable-5-1")
        target_51 = provider._fallback_target_for_model("claude-fable-5-1")
        target_5 = provider._fallback_target_for_model("claude-fable-5")
        assert target_51 == target_5
