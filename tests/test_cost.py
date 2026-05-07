"""Tests for _cost.py: compute_cost() and _RATES.

Covers:
  (a) Known model: correct Decimal cost for input tokens
  (b) Output tokens cost
  (c) cache_read_input_tokens cost (10% of input rate)
  (d) cache_creation_input_tokens cost (125% of input rate)
  (e) All token types combined
  (f) Unknown model returns None
  (g) None != Decimal('0')
  (h) Result type is always Decimal, never float

Integration tests (i–k): _convert_to_chat_response stamps cost_usd on Usage
  (i)  Known model + tokens → cost_usd is Decimal > 0
  (j)  1M cache_creation_input_tokens → cost_usd == Decimal('3.75')
  (k)  Unknown model → cost_usd is None
"""

from decimal import Decimal
from typing import cast
from unittest.mock import MagicMock

from amplifier_core import ModuleCoordinator

from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._cost import compute_cost

from tests._helpers import FakeCoordinator


# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------


def _make_provider() -> AnthropicProvider:
    """Create a minimal AnthropicProvider for direct method testing."""
    provider = AnthropicProvider(
        api_key="test-key",
        config={"use_streaming": False, "max_retries": 0},
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _make_response(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> MagicMock:
    """Build a fake Anthropic API response for testing _convert_to_chat_response."""
    response = MagicMock()
    response.content = []
    response.model = model
    response.stop_reason = "end_turn"
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    response.usage.cache_read_input_tokens = cache_read_input_tokens
    response.usage.cache_creation_input_tokens = cache_creation_input_tokens
    return response


# ---------------------------------------------------------------------------
# (a) Known model: correct Decimal cost for 1M input tokens
# ---------------------------------------------------------------------------
def test_known_model_input_tokens_cost():
    """claude-sonnet-4-5-20250929: 1M input → $3.00"""
    result = compute_cost("claude-sonnet-4-5-20250929", input_tokens=1_000_000)
    assert result == Decimal("3.00"), f"Expected Decimal('3.00'), got {result!r}"


# ---------------------------------------------------------------------------
# (b) Output tokens cost: 1M output → $15.00
# ---------------------------------------------------------------------------
def test_known_model_output_tokens_cost():
    """claude-sonnet-4-5-20250929: 1M output → $15.00"""
    result = compute_cost("claude-sonnet-4-5-20250929", output_tokens=1_000_000)
    assert result == Decimal("15.00"), f"Expected Decimal('15.00'), got {result!r}"


# ---------------------------------------------------------------------------
# (c) cache_read_input_tokens: 1M → $0.30 (10% of $3.00)
# ---------------------------------------------------------------------------
def test_known_model_cache_read_tokens_cost():
    """claude-sonnet-4-5-20250929: 1M cache_read_input_tokens → $0.30"""
    result = compute_cost(
        "claude-sonnet-4-5-20250929", cache_read_input_tokens=1_000_000
    )
    assert result == Decimal("0.30"), f"Expected Decimal('0.30'), got {result!r}"


# ---------------------------------------------------------------------------
# (d) cache_creation_input_tokens: 1M → $3.75 (125% of $3.00)
# ---------------------------------------------------------------------------
def test_known_model_cache_write_tokens_cost():
    """claude-sonnet-4-5-20250929: 1M cache_creation_input_tokens → $3.75"""
    result = compute_cost(
        "claude-sonnet-4-5-20250929", cache_creation_input_tokens=1_000_000
    )
    assert result == Decimal("3.75"), f"Expected Decimal('3.75'), got {result!r}"


# ---------------------------------------------------------------------------
# (e) All token types combined: 10K input + 2K output + 5K cache_read + 3K cache_write
# ---------------------------------------------------------------------------
def test_combined_token_types():
    """Worked example: 10K+2K+5K+3K → $0.07275"""
    # 10_000 input   × $3.00/1M  = $0.03000
    # 2_000 output   × $15.00/1M = $0.03000
    # 5_000 cache_r  × $0.30/1M  = $0.00150
    # 3_000 cache_w  × $3.75/1M  = $0.01125
    # total                       = $0.07275
    result = compute_cost(
        "claude-sonnet-4-5-20250929",
        input_tokens=10_000,
        output_tokens=2_000,
        cache_read_input_tokens=5_000,
        cache_creation_input_tokens=3_000,
    )
    assert result == Decimal("0.07275"), f"Expected Decimal('0.07275'), got {result!r}"


# ---------------------------------------------------------------------------
# (f) Unknown model returns None
# ---------------------------------------------------------------------------
def test_unknown_model_returns_none():
    """An unrecognised model name must return None (not 0, not raise)."""
    result = compute_cost("claude-does-not-exist-9999", input_tokens=1_000_000)
    assert result is None, f"Expected None for unknown model, got {result!r}"


# ---------------------------------------------------------------------------
# (g) None != Decimal('0'): unknown is distinct from free
# ---------------------------------------------------------------------------
def test_unknown_distinct_from_zero():
    """None returned for unknown model must not equal Decimal('0')."""
    result = compute_cost("no-such-model", input_tokens=0)
    assert result is None
    assert result != Decimal("0")


# ---------------------------------------------------------------------------
# (h) Result type is always Decimal, never float
# ---------------------------------------------------------------------------
def test_result_type_is_decimal():
    """compute_cost must return a Decimal, not a float."""
    result = compute_cost("claude-sonnet-4-5-20250929", input_tokens=1_000)
    assert isinstance(result, Decimal), f"Expected Decimal, got {type(result)}"
    assert not isinstance(result, float), "Result must not be a float"


# ---------------------------------------------------------------------------
# (i) Integration: _convert_to_chat_response stamps cost_usd for known model
# ---------------------------------------------------------------------------
def test_convert_stamps_cost_on_usage():
    """Known model + tokens → result.usage.cost_usd is not None, Decimal, > 0."""
    provider = _make_provider()
    response = _make_response(
        model="claude-sonnet-4-5-20250929",
        input_tokens=1_000,
        output_tokens=500,
    )
    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    assert result.usage.cost_usd is not None, "cost_usd should be stamped for known model"
    assert isinstance(result.usage.cost_usd, Decimal), (
        f"cost_usd should be Decimal, got {type(result.usage.cost_usd)}"
    )
    assert result.usage.cost_usd > 0, f"cost_usd should be > 0, got {result.usage.cost_usd}"


# ---------------------------------------------------------------------------
# (j) Integration: _convert_to_chat_response includes cache write in cost
# ---------------------------------------------------------------------------
def test_convert_includes_cache_write_in_cost():
    """1M cache_creation_input_tokens on claude-sonnet-4-5-20250929 → cost_usd == Decimal('3.75')."""
    provider = _make_provider()
    response = _make_response(
        model="claude-sonnet-4-5-20250929",
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=1_000_000,
    )
    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    assert result.usage.cost_usd == Decimal("3.75"), (
        f"Expected Decimal('3.75') for 1M cache_creation_input_tokens, got {result.usage.cost_usd!r}"
    )


# ---------------------------------------------------------------------------
# (k) Integration: _convert_to_chat_response leaves cost_usd=None for unknown model
# ---------------------------------------------------------------------------
def test_convert_leaves_cost_none_for_unknown_model():
    """Unknown model → result.usage.cost_usd is None."""
    provider = _make_provider()
    response = _make_response(
        model="claude-unknown-model-9999",
        input_tokens=1_000,
        output_tokens=500,
    )
    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    assert result.usage.cost_usd is None, (
        f"cost_usd should be None for unknown model, got {result.usage.cost_usd!r}"
    )
