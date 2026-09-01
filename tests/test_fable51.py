"""Tests for Claude Fable 5.1 support.

Covers:
  (a) _RATES contains 'claude-fable-5-1' with correct pricing
  (b) Input tokens cost: 1M input -> $10.00 (same as Fable 5)
  (c) Output tokens cost: 1M output -> $50.00 (same as Fable 5)
  (d) Cache read cost: 1M cache read -> $0.25 (75% less than Fable 5's $1.00)
  (e) Cache write cost: 1M cache write (5-min) -> $12.50 (same as Fable 5)
  (f) Cache read rate is 75% less than Fable 5
  (g) Input/output rates identical to Fable 5
  (h) Not in _FAST_ELIGIBLE_MODELS (no speed mode)
  (i) _detect_family returns 'fable' for claude-fable-5-1
  (j) _detect_version returns (5, 1) for claude-fable-5-1
  (k) _get_capabilities returns correct capability matrix
  (l) Capabilities: family='fable', max_output_tokens=128000
  (m) Capabilities: supports_1m=True, thinking_always_on=True
  (n) Capabilities: supports_adaptive_thinking=True, supports_manual_thinking=False
  (o) Capabilities: all 5 effort levels (low/medium/high/xhigh/max)
  (p) Capabilities: supports_speed=False, supports_sampling=False
  (q) Capabilities: supports_task_budget=True, supports_output_config=True
  (r) list_models includes claude-fable-5-1 (family grouping)
  (s) 1h cache write billed at 2x input rate ($20.00/MTok)
"""

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator

from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._cost import (
    _FAST_ELIGIBLE_MODELS,
    _RATES,
    compute_cost,
)
from tests._helpers import FakeCoordinator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(filtered: bool = True) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": 0,
            "filtered": filtered,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _model(model_id: str, display_name: str, created_at: str) -> SimpleNamespace:
    """Minimal Anthropic Models API entry stub."""
    return SimpleNamespace(
        id=model_id,
        display_name=display_name,
        created_at=created_at,
    )


def _stub_models_list(
    provider: AnthropicProvider, models: list[SimpleNamespace]
) -> None:
    provider.client.models.list = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(data=models)
    )


# ---------------------------------------------------------------------------
# (a) _RATES contains 'claude-fable-5-1'
# ---------------------------------------------------------------------------
def test_fable51_in_rates():
    """claude-fable-5-1 must be registered in _RATES."""
    assert "claude-fable-5-1" in _RATES, "claude-fable-5-1 must be present in _RATES"


# ---------------------------------------------------------------------------
# (b) Input tokens cost: 1M input -> $10.00
# ---------------------------------------------------------------------------
def test_fable51_input_tokens_cost():
    """claude-fable-5-1: 1M input -> $10.00 (same as Fable 5)."""
    result = compute_cost("claude-fable-5-1", input_tokens=1_000_000)
    assert result == Decimal("10.00"), f"Expected Decimal('10.00'), got {result!r}"


# ---------------------------------------------------------------------------
# (c) Output tokens cost: 1M output -> $50.00
# ---------------------------------------------------------------------------
def test_fable51_output_tokens_cost():
    """claude-fable-5-1: 1M output -> $50.00 (same as Fable 5)."""
    result = compute_cost("claude-fable-5-1", output_tokens=1_000_000)
    assert result == Decimal("50.00"), f"Expected Decimal('50.00'), got {result!r}"


# ---------------------------------------------------------------------------
# (d) Cache read cost: 1M cache read -> $0.25 (75% less than Fable 5's $1.00)
# ---------------------------------------------------------------------------
def test_fable51_cache_read_cost():
    """claude-fable-5-1: 1M cache read -> $0.25 (reduced from Fable 5's $1.00)."""
    result = compute_cost("claude-fable-5-1", cache_read_input_tokens=1_000_000)
    assert result == Decimal("0.25"), f"Expected Decimal('0.25'), got {result!r}"


# ---------------------------------------------------------------------------
# (e) Cache write cost: 1M cache write (5-min) -> $12.50 (same as Fable 5)
# ---------------------------------------------------------------------------
def test_fable51_cache_write_cost():
    """claude-fable-5-1: 1M cache write (5-min) -> $12.50."""
    result = compute_cost("claude-fable-5-1", cache_creation_input_tokens=1_000_000)
    assert result == Decimal("12.50"), f"Expected Decimal('12.50'), got {result!r}"


# ---------------------------------------------------------------------------
# (f) Cache read rate is 75% less than Fable 5
# ---------------------------------------------------------------------------
def test_fable51_cache_read_75pct_cheaper_than_fable5():
    """Fable 5.1 cache reads must be 75% cheaper than Fable 5."""
    fable5_read = compute_cost("claude-fable-5", cache_read_input_tokens=1_000_000)
    fable51_read = compute_cost("claude-fable-5-1", cache_read_input_tokens=1_000_000)
    assert fable5_read is not None
    assert fable51_read is not None
    # $0.25 = 25% of $1.00 => 75% cheaper
    assert fable51_read == fable5_read * Decimal("0.25"), (
        f"Fable 5.1 cache read ({fable51_read}) should be 25% of Fable 5 ({fable5_read})"
    )


# ---------------------------------------------------------------------------
# (g) Input/output rates identical to Fable 5
# ---------------------------------------------------------------------------
def test_fable51_input_rate_identical_to_fable5():
    """Fable 5.1 input rate must equal Fable 5 input rate."""
    fable5_input = compute_cost("claude-fable-5", input_tokens=1_000_000)
    fable51_input = compute_cost("claude-fable-5-1", input_tokens=1_000_000)
    assert fable5_input is not None and fable51_input is not None
    assert fable51_input == fable5_input


def test_fable51_output_rate_identical_to_fable5():
    """Fable 5.1 output rate must equal Fable 5 output rate."""
    fable5_output = compute_cost("claude-fable-5", output_tokens=1_000_000)
    fable51_output = compute_cost("claude-fable-5-1", output_tokens=1_000_000)
    assert fable5_output is not None and fable51_output is not None
    assert fable51_output == fable5_output


# ---------------------------------------------------------------------------
# (h) Not in _FAST_ELIGIBLE_MODELS
# ---------------------------------------------------------------------------
def test_fable51_not_in_fast_eligible_models():
    """claude-fable-5-1 must NOT be in _FAST_ELIGIBLE_MODELS (no speed mode)."""
    assert "claude-fable-5-1" not in _FAST_ELIGIBLE_MODELS


# ---------------------------------------------------------------------------
# (i) _detect_family returns 'fable' for claude-fable-5-1
# ---------------------------------------------------------------------------
def test_fable51_family_detected():
    """_detect_family must return 'fable' for claude-fable-5-1."""
    family = AnthropicProvider._detect_family("claude-fable-5-1")
    assert family == "fable", f"Expected 'fable', got {family!r}"


# ---------------------------------------------------------------------------
# (j) _detect_version returns (5, 1) for claude-fable-5-1
# ---------------------------------------------------------------------------
def test_fable51_version_detected():
    """_detect_version must return (5, 1) for claude-fable-5-1."""
    version = AnthropicProvider._detect_version("claude-fable-5-1", "fable")
    assert version == (5, 1), f"Expected (5, 1), got {version!r}"


# ---------------------------------------------------------------------------
# (k) _get_capabilities returns correct capability matrix
# ---------------------------------------------------------------------------
def test_fable51_get_capabilities_does_not_raise():
    """_get_capabilities('claude-fable-5-1') must not raise."""
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
    assert caps is not None


# ---------------------------------------------------------------------------
# (l) Capabilities: family='fable', max_output_tokens=128000
# ---------------------------------------------------------------------------
def test_fable51_capabilities_family():
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
    assert caps.family == "fable"


def test_fable51_capabilities_max_output_128k():
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
    assert caps.max_output_tokens == 128000


# ---------------------------------------------------------------------------
# (m) Capabilities: supports_1m=True, thinking_always_on=True
# ---------------------------------------------------------------------------
def test_fable51_supports_1m():
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
    assert caps.supports_1m is True


def test_fable51_thinking_always_on():
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
    assert caps.thinking_always_on is True


# ---------------------------------------------------------------------------
# (n) Capabilities: supports_adaptive_thinking=True, supports_manual_thinking=False
# ---------------------------------------------------------------------------
def test_fable51_supports_adaptive_thinking():
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
    assert caps.supports_adaptive_thinking is True


def test_fable51_no_manual_thinking():
    """Manual thinking (budget_tokens) is not accepted on Fable 5.1."""
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
    assert caps.supports_manual_thinking is False


# ---------------------------------------------------------------------------
# (o) Capabilities: all 5 effort levels (low/medium/high/xhigh/max)
# ---------------------------------------------------------------------------
def test_fable51_all_effort_levels():
    """Fable 5.1 supports all 5 effort levels including xhigh and max."""
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
    assert set(caps.supported_efforts) == {"low", "medium", "high", "xhigh", "max"}


# ---------------------------------------------------------------------------
# (p) Capabilities: supports_speed=False, supports_sampling=False
# ---------------------------------------------------------------------------
def test_fable51_no_speed():
    """Speed mode is NOT supported on Fable 5.1."""
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
    assert caps.supports_speed is False


def test_fable51_no_sampling():
    """Sampling (temperature) is NOT supported on Fable 5.1."""
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
    assert caps.supports_sampling is False


# ---------------------------------------------------------------------------
# (q) Capabilities: supports_task_budget=True, supports_output_config=True
# ---------------------------------------------------------------------------
def test_fable51_supports_task_budget():
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
    assert caps.supports_task_budget is True


def test_fable51_supports_output_config():
    caps = AnthropicProvider._get_capabilities("claude-fable-5-1")
    assert caps.supports_output_config is True


# ---------------------------------------------------------------------------
# (r) list_models includes claude-fable-5-1 (family grouping)
# ---------------------------------------------------------------------------
def test_list_models_includes_fable51():
    """list_models must surface claude-fable-5-1 in the fable family."""
    provider = _make_provider(filtered=True)
    _stub_models_list(
        provider,
        [
            _model("claude-fable-5-1", "Claude Fable 5.1", "2026-09-01"),
            _model("claude-fable-5", "Claude Fable 5", "2026-01-01"),
            _model("claude-opus-5", "Claude Opus 5", "2026-07-01"),
            _model("claude-sonnet-5", "Claude Sonnet 5", "2026-06-30"),
            _model("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "2025-10-01"),
        ],
    )
    result = asyncio.run(provider.list_models())
    ids = {m.id for m in result}
    assert "claude-fable-5-1" in ids, (
        f"claude-fable-5-1 missing from list_models output: {ids}"
    )


def test_list_models_fable51_family_is_fable():
    """list_models must classify claude-fable-5-1 in the 'fable' family."""
    provider = _make_provider(filtered=True)
    _stub_models_list(
        provider,
        [_model("claude-fable-5-1", "Claude Fable 5.1", "2026-09-01")],
    )
    result = asyncio.run(provider.list_models())
    assert len(result) == 1
    family = AnthropicProvider._detect_family(result[0].id)
    assert family == "fable"


# ---------------------------------------------------------------------------
# (s) 1h cache write billed at 2x input rate ($20.00/MTok)
# ---------------------------------------------------------------------------
def test_fable51_1h_cache_write_at_2x_input_rate():
    """1h cache writes on Fable 5.1 must be billed at 2x input rate = $20.00/MTok."""
    result = compute_cost(
        "claude-fable-5-1",
        cache_creation_1h_input_tokens=1_000_000,
    )
    # 2x input rate = 2 * $10.00 = $20.00
    assert result == Decimal("20.00"), f"Expected Decimal('20.00'), got {result!r}"
