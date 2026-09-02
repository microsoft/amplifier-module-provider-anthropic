"""Tests for Claude Fable 5.1 registration and capabilities.

Covers:
  (a) _detect_family returns 'fable' for claude-fable-5-1
  (b) _detect_version returns (5, 1) for claude-fable-5-1
  (c) _get_capabilities returns the correct capability matrix for Fable 5.1
  (d) Fable 5.1 capabilities are identical to Fable 5 (same context window,
      max output, thinking mode, effort tiers)
  (e) Fable 5.1 is in _RATES (cost lookup succeeds)
  (f) Fable 5.1 cache read rate is $0.25/MTok (75% reduction vs Fable 5's $1.00)
  (g) Fable 5.1 fallback target is claude-opus-5 (same as Fable 5)
  (h) _convert_to_chat_response stamps a non-None cost_usd for Fable 5.1

Source for facts (a)-(d): https://docs.anthropic.com/en/docs/about-claude/models/overview
                           (verified 2026-09-02)
Source for fact (f):       https://www.anthropic.com/pricing (verified 2026-09-02)
                           https://www.anthropic.com/claude-fable-and-mythos-5-1
"""

from decimal import Decimal
from typing import cast
from unittest.mock import MagicMock

from amplifier_core import ModuleCoordinator

from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._cost import _RATES
from tests._helpers import FakeCoordinator


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


# ---------------------------------------------------------------------------
# (a) Family detection
# ---------------------------------------------------------------------------
def test_fable51_family_detection():
    """claude-fable-5-1 must be classified as the 'fable' family."""
    assert AnthropicProvider._detect_family("claude-fable-5-1") == "fable"


# ---------------------------------------------------------------------------
# (b) Version detection
# ---------------------------------------------------------------------------
def test_fable51_version_detection():
    """claude-fable-5-1 must parse to version (5, 1)."""
    assert AnthropicProvider._detect_version("claude-fable-5-1", "fable") == (5, 1)


# ---------------------------------------------------------------------------
# (c) Capabilities matrix
# ---------------------------------------------------------------------------
def test_fable51_capabilities_correct():
    """_get_capabilities must return the expected matrix for claude-fable-5-1."""
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")

    assert caps.family == "fable"
    assert caps.max_output_tokens == 128000  # 128K tokens
    assert caps.base_context_window == 200000  # 1M enabled via supports_1m
    assert caps.supports_1m is True  # 1M context window
    assert caps.supports_thinking is True
    assert caps.supports_adaptive_thinking is True
    assert caps.supports_manual_thinking is False  # thinking is always-on
    assert caps.thinking_always_on is True
    assert caps.supports_output_config is True
    assert caps.supports_task_budget is True
    assert caps.supports_sampling is False
    assert caps.thinking_display_required is True
    assert "xhigh" in caps.supported_efforts
    assert "max" in caps.supported_efforts
    assert caps.supports_speed is False
    assert caps.supports_inline_system is True


# ---------------------------------------------------------------------------
# (d) Fable 5.1 capabilities are identical to Fable 5
# ---------------------------------------------------------------------------
def test_fable51_capabilities_match_fable5():
    """Fable 5.1 must have the same capability matrix as Fable 5.

    Whole-object equality rather than a field-by-field list: ModelCapabilities
    has 21 fields, and a hand-rolled comparison silently stops covering any
    field added after it was written.

    This guards a real regression. The sibling `mythos` branch in
    _get_capabilities IS version-gated (Mythos PREVIEW differs from Mythos 5 on
    effort tiers and cache minimum), so a future version gate on the `fable`
    branch that splits 5.1 off from 5 is exactly the kind of change this test
    exists to catch.
    """
    caps5 = AnthropicProvider._get_capabilities("claude-fable-5")
    caps51 = AnthropicProvider._get_capabilities("claude-fable-5-1")

    assert caps51 == caps5


# ---------------------------------------------------------------------------
# (e) Fable 5.1 is in _RATES
# ---------------------------------------------------------------------------
def test_fable51_in_rates():
    """claude-fable-5-1 must be registered in _RATES."""
    assert "claude-fable-5-1" in _RATES


# ---------------------------------------------------------------------------
# (f) Cache read rate is $0.25/MTok
# ---------------------------------------------------------------------------
def test_fable51_cache_read_rate():
    """Fable 5.1 cache_read_per_m must be $0.25 (75% reduction from Fable 5's $1.00)."""
    assert _RATES["claude-fable-5-1"]["cache_read_per_m"] == Decimal("0.25")
    # Verify the Fable 5 rate is still $1.00 (regression guard)
    assert _RATES["claude-fable-5"]["cache_read_per_m"] == Decimal("1.00")


# ---------------------------------------------------------------------------
# (g) Fallback target
# ---------------------------------------------------------------------------
def test_fable51_fallback_target_is_opus():
    """Fable 5.1 must fall back to claude-opus-5 (same as Fable 5).

    Asserts the CONCRETE target id, not just its family. _fallback_target_for_model
    branches only on _detect_family, so a family-level assertion here also holds
    for "claude-fable-banana" -- it cannot fail for a Fable-5.1-specific reason.
    Pinning the exact id at least catches a regression in the
    _STATIC_FALLBACK_MODELS backstop. Ladder mechanics themselves are covered by
    tests/test_fallback_ladder.py.
    """
    provider = _make_provider("claude-fable-5-1")
    target = provider._fallback_target_for_model("claude-fable-5-1")
    assert target == "claude-opus-5"


# ---------------------------------------------------------------------------
# (h) cost_usd is stamped by _convert_to_chat_response
# ---------------------------------------------------------------------------
def test_fable51_cost_usd_stamped():
    """_convert_to_chat_response must stamp a non-None cost_usd for Fable 5.1."""
    provider = _make_provider()
    response = MagicMock()
    response.model = "claude-fable-5-1"
    response.stop_reason = "end_turn"
    response.content = []
    response.usage.input_tokens = 1000
    response.usage.output_tokens = 200
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = 0
    del response.usage.speed

    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    assert result.usage.cost_usd is not None
    assert result.usage.cost_usd > Decimal(0)
