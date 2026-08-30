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

Cache-write TTL split tests (t–z): compute_cost() honors the per-TTL split
(usage.cache_creation.ephemeral_5m_input_tokens / .ephemeral_1h_input_tokens)
when present, billing 1h writes at 2x input rather than the 1.25x 5m rate.
  (t) Split present, all tokens 1h → billed at 2x input rate
  (u) Split present, all tokens 5m → identical to legacy aggregate behavior
  (v) Split present, mixed 5m + 1h → each portion billed at its own rate
  (w) Aggregate-only (no split kwargs) → unchanged legacy 1.25x behavior
  (x) Split kwargs explicitly 0/0 with no aggregate → zero cache-write cost,
      no crash
  (y) Split doesn't sum to aggregate → split wins, discrepancy logged at
      DEBUG (not WARNING)
  (z) Integration: _convert_to_chat_response wires usage.cache_creation into
      compute_cost() and stamps the corrected 1h cost on Usage.cost_usd
"""

import logging
from decimal import Decimal
from types import SimpleNamespace
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
    speed: str | None = None,
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
    response.usage.speed = speed
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
# Haiku 3.5 is keyed by its served model id
# ---------------------------------------------------------------------------
def test_haiku_35_real_model_id_is_priced():
    """The id the API actually returns must resolve to the Haiku 3.5 rates."""
    result = compute_cost("claude-3-5-haiku-20241022", input_tokens=1_000_000)
    assert result == Decimal("0.80")


def test_haiku_35_output_and_cache_rates():
    assert compute_cost(
        "claude-3-5-haiku-20241022", output_tokens=1_000_000
    ) == Decimal("4.00")
    assert compute_cost(
        "claude-3-5-haiku-20241022", cache_read_input_tokens=1_000_000
    ) == Decimal("0.08")
    assert compute_cost(
        "claude-3-5-haiku-20241022", cache_creation_input_tokens=1_000_000
    ) == Decimal("1.00")


def test_former_haiku_35_key_was_never_a_served_model_id():
    """No model was served as claude-haiku-3-5-20250929, so it must not price."""
    result = compute_cost("claude-haiku-3-5-20250929", input_tokens=1_000_000)
    assert result is None


# ---------------------------------------------------------------------------
# (g) None != Decimal('0'): unknown is distinct from free
# ---------------------------------------------------------------------------
def test_unknown_distinct_from_zero():
    """None returned for unknown model must not equal Decimal('0')."""
    result = compute_cost("no-such-model", input_tokens=0)
    assert result is None
    assert result != Decimal(0)


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
    assert result.usage.cost_usd is not None, (
        "cost_usd should be stamped for known model"
    )
    assert isinstance(result.usage.cost_usd, Decimal), (
        f"cost_usd should be Decimal, got {type(result.usage.cost_usd)}"
    )
    assert result.usage.cost_usd > 0, (
        f"cost_usd should be > 0, got {result.usage.cost_usd}"
    )


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


# ---------------------------------------------------------------------------
# (l) claude-opus-4-8 input cost: 1M input → $5.00
# ---------------------------------------------------------------------------
def test_opus_48_input_tokens_cost():
    """claude-opus-4-8: 1M input → $5.00"""
    result = compute_cost("claude-opus-4-8", input_tokens=1_000_000)
    assert result == Decimal("5.00"), f"Expected Decimal('5.00'), got {result!r}"


# ---------------------------------------------------------------------------
# (m) claude-opus-4-8 output cost: 1M output → $25.00
# ---------------------------------------------------------------------------
def test_opus_48_output_tokens_cost():
    """claude-opus-4-8: 1M output → $25.00"""
    result = compute_cost("claude-opus-4-8", output_tokens=1_000_000)
    assert result == Decimal("25.00"), f"Expected Decimal('25.00'), got {result!r}"


# ---------------------------------------------------------------------------
# (n) speed='fast' doubles cost for eligible model (claude-opus-4-8)
# ---------------------------------------------------------------------------
def test_fast_mode_doubles_cost_for_eligible_model():
    """speed='fast' on claude-opus-4-8: 1M input → $5.00 base, $10.00 fast"""
    base = compute_cost("claude-opus-4-8", input_tokens=1_000_000)
    assert base == Decimal("5.00"), f"Expected base Decimal('5.00'), got {base!r}"
    fast = compute_cost("claude-opus-4-8", input_tokens=1_000_000, speed="fast")
    assert fast == Decimal("10.00"), f"Expected fast Decimal('10.00'), got {fast!r}"


# ---------------------------------------------------------------------------
# (o) speed='fast' does NOT apply multiplier for ineligible model
# ---------------------------------------------------------------------------
def test_fast_mode_no_multiplier_for_ineligible_model():
    """speed='fast' on claude-sonnet-4-5-20250929: 1M input stays at $3.00"""
    base = compute_cost("claude-sonnet-4-5-20250929", input_tokens=1_000_000)
    assert base == Decimal("3.00"), f"Expected base Decimal('3.00'), got {base!r}"
    fast = compute_cost(
        "claude-sonnet-4-5-20250929", input_tokens=1_000_000, speed="fast"
    )
    assert fast == Decimal("3.00"), f"Expected fast Decimal('3.00'), got {fast!r}"


# ---------------------------------------------------------------------------
# (p) speed='standard' and speed=None both leave cost unchanged
# ---------------------------------------------------------------------------
def test_no_multiplier_when_speed_standard():
    """speed='standard' and speed=None both leave claude-opus-4-8 1M input at $5.00"""
    standard = compute_cost("claude-opus-4-8", input_tokens=1_000_000, speed="standard")
    assert standard == Decimal("5.00"), (
        f"Expected Decimal('5.00') for speed='standard', got {standard!r}"
    )
    no_speed = compute_cost("claude-opus-4-8", input_tokens=1_000_000, speed=None)
    assert no_speed == Decimal("5.00"), (
        f"Expected Decimal('5.00') for speed=None, got {no_speed!r}"
    )


# ---------------------------------------------------------------------------
# (q) Integration: _convert_to_chat_response applies fast multiplier from response speed
# ---------------------------------------------------------------------------
def test_convert_applies_fast_multiplier_from_response_speed():
    """response.usage.speed='fast' on claude-opus-4-8 with 1M input tokens → cost_usd == Decimal('10.00')."""
    provider = _make_provider()
    response = _make_response(
        model="claude-opus-4-8",
        input_tokens=1_000_000,
        output_tokens=0,
        speed="fast",
    )
    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    assert result.usage.cost_usd == Decimal("10.00"), (
        f"Expected Decimal('10.00') for speed='fast' on claude-opus-4-8, got {result.usage.cost_usd!r}"
    )


# ---------------------------------------------------------------------------
# (r) Integration: _convert_to_chat_response does NOT apply multiplier for speed='standard'
# ---------------------------------------------------------------------------
def test_convert_no_multiplier_for_standard_speed():
    """response.usage.speed='standard' on claude-opus-4-8 with 1M input tokens → cost_usd == Decimal('5.00')."""
    provider = _make_provider()
    response = _make_response(
        model="claude-opus-4-8",
        input_tokens=1_000_000,
        output_tokens=0,
        speed="standard",
    )
    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    assert result.usage.cost_usd == Decimal("5.00"), (
        f"Expected Decimal('5.00') for speed='standard' on claude-opus-4-8, got {result.usage.cost_usd!r}"
    )


# ---------------------------------------------------------------------------
# (s) Fable 5 pricing: 1M input -> $10.00, 1M output -> $50.00
# ---------------------------------------------------------------------------
def test_fable5_input_tokens_cost():
    """claude-fable-5: 1M input -> $10.00"""
    result = compute_cost("claude-fable-5", input_tokens=1_000_000)
    assert result == Decimal("10.00"), f"Expected Decimal('10.00'), got {result!r}"


def test_fable5_output_tokens_cost():
    """claude-fable-5: 1M output -> $50.00"""
    result = compute_cost("claude-fable-5", output_tokens=1_000_000)
    assert result == Decimal("50.00"), f"Expected Decimal('50.00'), got {result!r}"


def test_fable5_cache_read_cost():
    """claude-fable-5: 1M cache read -> $1.00"""
    result = compute_cost("claude-fable-5", cache_read_input_tokens=1_000_000)
    assert result == Decimal("1.00"), f"Expected Decimal('1.00'), got {result!r}"


def test_fable5_cache_write_cost():
    """claude-fable-5: 1M cache write (5-min) -> $12.50"""
    result = compute_cost("claude-fable-5", cache_creation_input_tokens=1_000_000)
    assert result == Decimal("12.50"), f"Expected Decimal('12.50'), got {result!r}"


def test_fable5_not_in_fast_eligible_models():
    """claude-fable-5 must NOT be in _FAST_ELIGIBLE_MODELS (no speed mode)."""
    from amplifier_module_provider_anthropic._cost import _FAST_ELIGIBLE_MODELS

    assert "claude-fable-5" not in _FAST_ELIGIBLE_MODELS


def test_fable5_exact_2x_opus48():
    """Every fable-5 rate is exactly 2x the corresponding opus-4-8 rate."""
    fable_input = compute_cost("claude-fable-5", input_tokens=1_000_000)
    opus_input = compute_cost("claude-opus-4-8", input_tokens=1_000_000)
    assert opus_input is not None, "claude-opus-4-8 must be in _RATES"
    assert fable_input is not None, "claude-fable-5 must be in _RATES"
    assert fable_input == opus_input * 2

    fable_output = compute_cost("claude-fable-5", output_tokens=1_000_000)
    opus_output = compute_cost("claude-opus-4-8", output_tokens=1_000_000)
    assert opus_output is not None, "claude-opus-4-8 must be in _RATES"
    assert fable_output is not None, "claude-fable-5 must be in _RATES"
    assert fable_output == opus_output * 2


# ---------------------------------------------------------------------------
# (r) Sonnet 5 pricing: standard rates $3 / $15 / $0.30 / $3.75 per MTok
#     (intro discount $2/$10 through 2026-08-31 is intentionally NOT encoded;
#     _RATES carries durable standard rates, matching the rest of the table.)
# ---------------------------------------------------------------------------
def test_sonnet_5_input_tokens_cost():
    """claude-sonnet-5: 1M input -> $3.00"""
    result = compute_cost("claude-sonnet-5", input_tokens=1_000_000)
    assert result == Decimal("3.00"), f"Expected Decimal('3.00'), got {result!r}"


def test_sonnet_5_output_tokens_cost():
    """claude-sonnet-5: 1M output -> $15.00"""
    result = compute_cost("claude-sonnet-5", output_tokens=1_000_000)
    assert result == Decimal("15.00"), f"Expected Decimal('15.00'), got {result!r}"


def test_sonnet_5_cache_read_cost():
    """claude-sonnet-5: 1M cache-read -> $0.30 (10% of input)."""
    result = compute_cost("claude-sonnet-5", cache_read_input_tokens=1_000_000)
    assert result == Decimal("0.30"), f"Expected Decimal('0.30'), got {result!r}"


def test_sonnet_5_cache_write_cost():
    """claude-sonnet-5: 1M cache-write -> $3.75 (125% of input)."""
    result = compute_cost("claude-sonnet-5", cache_creation_input_tokens=1_000_000)
    assert result == Decimal("3.75"), f"Expected Decimal('3.75'), got {result!r}"


# ---------------------------------------------------------------------------
# Claude Opus 5 pricing: same rates as Opus 4.8, and fast-mode eligible
# ---------------------------------------------------------------------------
def test_opus_5_standard_rate():
    cost = compute_cost(
        "claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert cost == Decimal("5.00") + Decimal("25.00")


def test_opus_5_fast_mode_multiplier():
    standard = compute_cost(
        "claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000
    )
    fast = compute_cost(
        "claude-opus-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        speed="fast",
    )
    assert fast == standard * 2


# ---------------------------------------------------------------------------
# Cache-write TTL split: bill 1h writes at 2x input, 5m writes at 1.25x input
# ---------------------------------------------------------------------------
#
# claude-sonnet-4-5-20250929 rates: input=$3.00/M, cache_write(5m)=$3.75/M
# (1.25x). The 1h rate is not a separate table entry -- it is always 2x the
# model's base input_per_m, per Anthropic's official pricing.


# (t) Split present, all tokens 1h -> billed at 2x input rate ($6.00/M)
def test_ttl_split_all_1h_billed_at_2x_input():
    """1M ephemeral_1h_input_tokens -> $6.00 (2x $3.00 input), not $3.75."""
    result = compute_cost(
        "claude-sonnet-4-5-20250929",
        cache_creation_1h_input_tokens=1_000_000,
        cache_creation_5m_input_tokens=0,
    )
    assert result == Decimal("6.00"), f"Expected Decimal('6.00'), got {result!r}"


# (u) Split present, all tokens 5m -> identical to legacy aggregate behavior
def test_ttl_split_all_5m_matches_legacy_behavior():
    """1M ephemeral_5m_input_tokens (split) == 1M cache_creation_input_tokens (legacy)."""
    split_result = compute_cost(
        "claude-sonnet-4-5-20250929",
        cache_creation_5m_input_tokens=1_000_000,
        cache_creation_1h_input_tokens=0,
    )
    legacy_result = compute_cost(
        "claude-sonnet-4-5-20250929", cache_creation_input_tokens=1_000_000
    )
    assert split_result == Decimal("3.75"), (
        f"Expected Decimal('3.75'), got {split_result!r}"
    )
    assert split_result == legacy_result


# (v) Split present, mixed 5m + 1h -> each portion billed at its own rate
def test_ttl_split_mixed_5m_and_1h():
    """400K 5m + 600K 1h -> (400K*3.75 + 600K*6.00) / 1M == $5.10."""
    result = compute_cost(
        "claude-sonnet-4-5-20250929",
        cache_creation_5m_input_tokens=400_000,
        cache_creation_1h_input_tokens=600_000,
        cache_creation_input_tokens=1_000_000,  # sums correctly, no discrepancy
    )
    assert result == Decimal("5.10"), f"Expected Decimal('5.10'), got {result!r}"


# (w) Aggregate-only (no split kwargs at all) -> unchanged legacy 1.25x behavior
def test_no_ttl_split_kwargs_preserves_legacy_behavior():
    """Omitting both split kwargs must behave exactly as before the fix."""
    result = compute_cost(
        "claude-sonnet-4-5-20250929", cache_creation_input_tokens=1_000_000
    )
    assert result == Decimal("3.75"), f"Expected Decimal('3.75'), got {result!r}"


# (x) Missing/None cache fields -> zero cost contribution, no crash
def test_ttl_split_none_fields_contribute_zero_cost_no_crash():
    """cache_creation_5m/1h=None (default) and cache_creation_input_tokens=0
    must contribute nothing to cost and must not raise."""
    result = compute_cost(
        "claude-sonnet-4-5-20250929",
        input_tokens=1_000,
        output_tokens=500,
        # cache_creation_input_tokens defaults to 0
        # cache_creation_5m_input_tokens / _1h_input_tokens default to None
    )
    baseline = compute_cost(
        "claude-sonnet-4-5-20250929", input_tokens=1_000, output_tokens=500
    )
    assert result == baseline
    assert result is not None and result > 0  # input/output cost still applies


def test_ttl_split_explicit_zero_zero_no_aggregate_contributes_zero():
    """Explicitly passing 0/0 for the split (split mode engaged, no tokens)
    must not crash and must contribute zero cache-write cost."""
    result = compute_cost(
        "claude-sonnet-4-5-20250929",
        input_tokens=1_000,
        cache_creation_5m_input_tokens=0,
        cache_creation_1h_input_tokens=0,
    )
    input_only = compute_cost("claude-sonnet-4-5-20250929", input_tokens=1_000)
    assert result == input_only


# (y) Split doesn't sum to aggregate -> split wins; discrepancy logged at DEBUG
def test_ttl_split_discrepancy_prefers_split_and_logs_debug_only(caplog):
    """Aggregate says 1M but split sums to 900K -> billed from the split
    (900K), and the mismatch is logged at DEBUG, never WARNING."""
    with caplog.at_level(
        logging.DEBUG, logger="amplifier_module_provider_anthropic._cost"
    ):
        result = compute_cost(
            "claude-sonnet-4-5-20250929",
            cache_creation_5m_input_tokens=400_000,
            cache_creation_1h_input_tokens=500_000,
            cache_creation_input_tokens=1_000_000,  # stale/mismatched aggregate
        )

    # Billed from the 900K split (400K*3.75 + 500K*6.00)/1M = $4.50, NOT the
    # legacy aggregate-at-5m-rate figure ($1M * 3.75/1M == $3.75).
    assert result == Decimal("4.50"), f"Expected Decimal('4.50'), got {result!r}"

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    warning_or_above = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("does not match aggregate" in r.message for r in debug_records), (
        "Expected a debug-level discrepancy note"
    )
    assert not warning_or_above, (
        f"Discrepancy must not be logged at WARNING or above, got: {warning_or_above}"
    )


# (z) Integration: _convert_to_chat_response wires usage.cache_creation split
#     into compute_cost() and stamps the corrected cost on Usage.cost_usd
def _make_response_with_ttl_split(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    ephemeral_5m_input_tokens: int | None = None,
    ephemeral_1h_input_tokens: int | None = None,
) -> MagicMock:
    """Fake Anthropic API response whose usage carries a real cache_creation
    TTL split object (SimpleNamespace, not an auto-attribute MagicMock)."""
    response = MagicMock()
    response.content = []
    response.model = model
    response.stop_reason = "end_turn"
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = cache_creation_input_tokens
    response.usage.speed = None
    if ephemeral_5m_input_tokens is None and ephemeral_1h_input_tokens is None:
        response.usage.cache_creation = None
    else:
        response.usage.cache_creation = SimpleNamespace(
            ephemeral_5m_input_tokens=ephemeral_5m_input_tokens or 0,
            ephemeral_1h_input_tokens=ephemeral_1h_input_tokens or 0,
        )
    return response


def test_convert_bills_1h_cache_writes_at_2x_via_usage_split():
    """End-to-end: response.usage.cache_creation.ephemeral_1h_input_tokens
    flows through _convert_to_chat_response into a corrected cost_usd."""
    provider = _make_provider()
    response = _make_response_with_ttl_split(
        model="claude-sonnet-4-5-20250929",
        cache_creation_input_tokens=1_000_000,
        ephemeral_5m_input_tokens=0,
        ephemeral_1h_input_tokens=1_000_000,
    )
    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    assert result.usage.cost_usd == Decimal("6.00"), (
        f"Expected Decimal('6.00') for 1M 1h cache write, got {result.usage.cost_usd!r}"
    )


def test_convert_without_ttl_split_object_preserves_legacy_cost():
    """When response.usage has no `cache_creation` object at all (older SDK
    shape), cost_usd must match the pre-fix aggregate-at-5m-rate behavior."""
    provider = _make_provider()
    response = _make_response_with_ttl_split(
        model="claude-sonnet-4-5-20250929",
        cache_creation_input_tokens=1_000_000,
        # ephemeral_5m_input_tokens / ephemeral_1h_input_tokens both None
        # -> response.usage.cache_creation is None
    )
    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    assert result.usage.cost_usd == Decimal("3.75"), (
        f"Expected Decimal('3.75') (legacy behavior), got {result.usage.cost_usd!r}"
    )
