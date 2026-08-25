"""Tests for config-level cost rate overrides (the ``rates`` config key).

Covers:
  (a) parse_rate_overrides: None / non-mapping input → empty dict
  (b) parse_rate_overrides: full row → Decimal values in _RATES row shape
  (c) parse_rate_overrides: missing cache fields → 10% / 125% derived defaults
  (d) parse_rate_overrides: float values parsed via Decimal(str(x)), no float artifacts
  (e) parse_rate_overrides: invalid entries skipped (missing input/output,
      negative, non-numeric, unknown field names)
  (f) compute_cost: exact override supplies rates for a model absent from _RATES
  (g) compute_cost: override takes precedence over _RATES
  (h) compute_cost: unknown model with no matching override still returns None
  (i) compute_cost: trailing-* glob matches; exact beats glob; longest glob wins
  (j) compute_cost: result is Decimal, never float
  (k) compute_cost: speed='fast' multiplier still applies to overridden rates
      for _FAST_ELIGIBLE_MODELS

Integration tests (l-n): provider config `rates` reaches _convert_to_chat_response
  (l) Override model → cost_usd stamped from override rates
  (m) Model not covered by overrides or _RATES → cost_usd is None
  (n) Override precedence over _RATES end-to-end
"""

from decimal import Decimal
from typing import cast
from unittest.mock import MagicMock

from amplifier_core import ModuleCoordinator

from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic._cost import (
    _RATES,
    compute_cost,
    parse_rate_overrides,
)

from tests._helpers import FakeCoordinator


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

# Fable 5 rates ($10 / $50 / $1.00 / $12.50) as they would appear in provider
# config.  claude-fable-5 is intentionally absent from _RATES.
_FABLE_RATES_CONFIG = {
    "claude-fable-5": {
        "input": 10.00,
        "output": 50.00,
        "cache_read": 1.00,
        "cache_write": 12.50,
    },
}


def _make_provider(rates=None) -> AnthropicProvider:
    """Create a minimal AnthropicProvider for direct method testing."""
    config = {"use_streaming": False, "max_retries": 0}
    if rates is not None:
        config["rates"] = rates
    provider = AnthropicProvider(api_key="test-key", config=config)
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
# (a) parse_rate_overrides: None / non-mapping → empty dict
# ---------------------------------------------------------------------------
def test_parse_none_returns_empty():
    """rates key absent from config → no overrides."""
    assert parse_rate_overrides(None) == {}


def test_parse_non_mapping_returns_empty():
    """Non-mapping rates value is ignored, not raised on."""
    assert parse_rate_overrides("claude-fable-5") == {}
    assert parse_rate_overrides([{"input": 1}]) == {}
    assert parse_rate_overrides(42) == {}


# ---------------------------------------------------------------------------
# (b) parse_rate_overrides: full row → Decimal values in _RATES row shape
# ---------------------------------------------------------------------------
def test_parse_full_row():
    """All four fields → _RATES-shaped row with Decimal values."""
    overrides = parse_rate_overrides(_FABLE_RATES_CONFIG)
    assert set(overrides) == {"claude-fable-5"}
    row = overrides["claude-fable-5"]
    assert row == {
        "input_per_m": Decimal("10.0"),
        "output_per_m": Decimal("50.0"),
        "cache_read_per_m": Decimal("1.0"),
        "cache_write_per_m": Decimal("12.5"),
    }
    for value in row.values():
        assert isinstance(value, Decimal), f"Expected Decimal, got {type(value)}"


# ---------------------------------------------------------------------------
# (c) parse_rate_overrides: missing cache fields → derived defaults
# ---------------------------------------------------------------------------
def test_parse_missing_cache_fields_derives_defaults():
    """cache_read defaults to 10% of input; cache_write to 125% of input."""
    overrides = parse_rate_overrides({"my-model": {"input": 10, "output": 50}})
    row = overrides["my-model"]
    assert row["cache_read_per_m"] == Decimal("1.00"), (
        f"Expected 10% of input, got {row['cache_read_per_m']!r}"
    )
    assert row["cache_write_per_m"] == Decimal("12.50"), (
        f"Expected 125% of input, got {row['cache_write_per_m']!r}"
    )


# ---------------------------------------------------------------------------
# (d) parse_rate_overrides: values parsed via Decimal(str(x))
# ---------------------------------------------------------------------------
def test_parse_float_values_no_float_artifacts():
    """0.1 must become Decimal('0.1'), not Decimal(0.1)'s binary expansion."""
    overrides = parse_rate_overrides(
        {"my-model": {"input": 0.1, "output": "0.3", "cache_read": 1, "cache_write": 2}}
    )
    row = overrides["my-model"]
    assert row["input_per_m"] == Decimal("0.1")
    assert row["output_per_m"] == Decimal("0.3")
    # Decimal(0.1) (raw float) would NOT equal Decimal('0.1').
    assert row["input_per_m"] != Decimal(0.1)


# ---------------------------------------------------------------------------
# (e) parse_rate_overrides: invalid entries skipped, valid ones kept
# ---------------------------------------------------------------------------
def test_parse_invalid_entries_skipped():
    """Bad rows are dropped with a warning; good rows survive."""
    overrides = parse_rate_overrides(
        {
            "missing-output": {"input": 1.0},
            "negative": {"input": -1.0, "output": 5.0},
            "non-numeric": {"input": "cheap", "output": 5.0},
            "unknown-field": {"input": 1.0, "output": 5.0, "input_per_m": 1.0},
            "not-a-mapping": 3.0,
            "good-model": {"input": 1.0, "output": 5.0},
        }
    )
    assert set(overrides) == {"good-model"}


# ---------------------------------------------------------------------------
# (f) compute_cost: exact override supplies rates for a model absent from _RATES
# ---------------------------------------------------------------------------
def test_override_hits_for_unlisted_model():
    """claude-fable-5 (absent from _RATES) costs $10/1M input via override."""
    assert "claude-fable-5" not in _RATES  # guard: absent from built-in table
    overrides = parse_rate_overrides(_FABLE_RATES_CONFIG)
    assert compute_cost("claude-fable-5", input_tokens=1_000_000) is None
    result = compute_cost(
        "claude-fable-5", input_tokens=1_000_000, rate_overrides=overrides
    )
    assert result == Decimal("10.00"), f"Expected Decimal('10.00'), got {result!r}"


def test_override_all_token_types_combined():
    """1M of each token type on claude-fable-5 → 10 + 50 + 1 + 12.50 = $73.50."""
    overrides = parse_rate_overrides(_FABLE_RATES_CONFIG)
    result = compute_cost(
        "claude-fable-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        rate_overrides=overrides,
    )
    assert result == Decimal("73.50"), f"Expected Decimal('73.50'), got {result!r}"


# ---------------------------------------------------------------------------
# (g) compute_cost: override takes precedence over _RATES
# ---------------------------------------------------------------------------
def test_override_precedence_over_rates_table():
    """Overriding claude-sonnet-4-5 replaces the built-in $3/1M input rate."""
    overrides = parse_rate_overrides(
        {"claude-sonnet-4-5": {"input": 1.00, "output": 2.00}}
    )
    result = compute_cost(
        "claude-sonnet-4-5", input_tokens=1_000_000, rate_overrides=overrides
    )
    assert result == Decimal("1.00"), f"Expected Decimal('1.00'), got {result!r}"
    # Without overrides the built-in table still applies.
    baseline = compute_cost("claude-sonnet-4-5", input_tokens=1_000_000)
    assert baseline == Decimal("3.00"), f"Expected Decimal('3.00'), got {baseline!r}"


# ---------------------------------------------------------------------------
# (h) compute_cost: unknown model with no matching override still returns None
# ---------------------------------------------------------------------------
def test_unknown_model_still_none_with_overrides_present():
    """Overrides for other models do not leak onto unrelated unknown models."""
    overrides = parse_rate_overrides(_FABLE_RATES_CONFIG)
    result = compute_cost(
        "claude-unknown-model-9999", input_tokens=1_000, rate_overrides=overrides
    )
    assert result is None, f"Expected None, got {result!r}"


# ---------------------------------------------------------------------------
# (i) compute_cost: trailing-* glob; exact beats glob; longest glob wins
# ---------------------------------------------------------------------------
def test_glob_pattern_matches_dated_model_id():
    """'claude-fable-*' covers dated ids like claude-fable-5-20260115."""
    overrides = parse_rate_overrides(
        {"claude-fable-*": {"input": 10.00, "output": 50.00}}
    )
    result = compute_cost(
        "claude-fable-5-20260115", input_tokens=1_000_000, rate_overrides=overrides
    )
    assert result == Decimal("10.00"), f"Expected Decimal('10.00'), got {result!r}"


def test_exact_override_beats_glob():
    """An exact model id entry wins over a matching glob entry."""
    overrides = parse_rate_overrides(
        {
            "claude-fable-*": {"input": 99.00, "output": 99.00},
            "claude-fable-5": {"input": 10.00, "output": 50.00},
        }
    )
    result = compute_cost(
        "claude-fable-5", input_tokens=1_000_000, rate_overrides=overrides
    )
    assert result == Decimal("10.00"), f"Expected Decimal('10.00'), got {result!r}"


def test_longest_glob_prefix_wins():
    """The most specific (longest-prefix) glob wins."""
    overrides = parse_rate_overrides(
        {
            "claude-*": {"input": 99.00, "output": 99.00},
            "claude-fable-*": {"input": 10.00, "output": 50.00},
        }
    )
    result = compute_cost(
        "claude-fable-5", input_tokens=1_000_000, rate_overrides=overrides
    )
    assert result == Decimal("10.00"), f"Expected Decimal('10.00'), got {result!r}"


def test_glob_does_not_shadow_rates_table_for_non_matches():
    """A glob that does not match leaves _RATES lookup intact."""
    overrides = parse_rate_overrides(
        {"claude-fable-*": {"input": 99.00, "output": 99.00}}
    )
    result = compute_cost(
        "claude-sonnet-4-5", input_tokens=1_000_000, rate_overrides=overrides
    )
    assert result == Decimal("3.00"), f"Expected Decimal('3.00'), got {result!r}"


# ---------------------------------------------------------------------------
# (j) compute_cost with overrides returns Decimal, never float
# ---------------------------------------------------------------------------
def test_override_result_is_decimal():
    """compute_cost with overrides must return a Decimal, not a float."""
    overrides = parse_rate_overrides(_FABLE_RATES_CONFIG)
    result = compute_cost(
        "claude-fable-5", input_tokens=1_000, rate_overrides=overrides
    )
    assert isinstance(result, Decimal), f"Expected Decimal, got {type(result)}"
    assert not isinstance(result, float), "Result must not be a float"
    assert result == Decimal("0.01"), f"Expected Decimal('0.01'), got {result!r}"


# ---------------------------------------------------------------------------
# (k) speed='fast' multiplier still applies to overridden rates
# ---------------------------------------------------------------------------
def test_fast_multiplier_applies_to_overridden_eligible_model():
    """claude-opus-4-8 is fast-eligible; overriding its rates keeps the 2x."""
    overrides = parse_rate_overrides(
        {"claude-opus-4-8": {"input": 6.00, "output": 30.00}}
    )
    fast = compute_cost(
        "claude-opus-4-8",
        input_tokens=1_000_000,
        speed="fast",
        rate_overrides=overrides,
    )
    assert fast == Decimal("12.00"), f"Expected Decimal('12.00'), got {fast!r}"


# ---------------------------------------------------------------------------
# (l) Integration: provider config rates → cost_usd stamped for override model
# ---------------------------------------------------------------------------
def test_convert_stamps_cost_from_config_rates():
    """Provider config rates for claude-fable-5 → cost_usd stamped on Usage."""
    provider = _make_provider(rates=_FABLE_RATES_CONFIG)
    response = _make_response(
        model="claude-fable-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    assert result.usage.cost_usd == Decimal("60.00"), (
        f"Expected Decimal('60.00'), got {result.usage.cost_usd!r}"
    )


# ---------------------------------------------------------------------------
# (m) Integration: model outside overrides and _RATES → cost_usd stays None
# ---------------------------------------------------------------------------
def test_convert_leaves_cost_none_for_model_not_in_overrides():
    """Overrides for one model leave other unknown models at cost_usd=None."""
    provider = _make_provider(rates=_FABLE_RATES_CONFIG)
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
# (n) Integration: override precedence over _RATES end-to-end
# ---------------------------------------------------------------------------
def test_convert_uses_override_over_rates_table():
    """Config rates for claude-sonnet-4-5-20250929 replace the built-in rate."""
    provider = _make_provider(
        rates={"claude-sonnet-4-5-20250929": {"input": 1.00, "output": 2.00}}
    )
    response = _make_response(
        model="claude-sonnet-4-5-20250929",
        input_tokens=1_000_000,
        output_tokens=0,
    )
    result = provider._convert_to_chat_response(response)
    assert result.usage is not None
    assert result.usage.cost_usd == Decimal("1.00"), (
        f"Expected Decimal('1.00'), got {result.usage.cost_usd!r}"
    )
