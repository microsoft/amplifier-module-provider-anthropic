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
"""

from decimal import Decimal


from amplifier_module_provider_anthropic._cost import compute_cost


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
