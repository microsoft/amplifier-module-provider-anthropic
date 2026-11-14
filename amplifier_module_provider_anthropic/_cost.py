"""Anthropic pricing rates and cost computation.

Current-model rates verified 2026-11-14:
https://platform.claude.com/docs/en/about-claude/pricing

Usage
-----
    from amplifier_module_provider_anthropic._cost import compute_cost
    from decimal import Decimal

    cost = compute_cost(
        "claude-sonnet-4-5-20250929",
        input_tokens=1_000,
        output_tokens=200,
    )
    # Returns Decimal or None if pricing is unavailable.

Haiku 5.5 uses the entire request's fresh + cache-read + cache-write input
to select its prompt tier. Above 100,000 tokens, every category (including
output) uses the higher rate. Earlier requests are never repriced.
None means unavailable, never zero. Rates describe standard first-party
token charges, not negotiated discounts or separately billed server tools.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_PER_M = Decimal("1_000_000")

# _RATES maps model-id → {
#   "input_per_m":      Decimal,   # fresh input tokens, per 1M
#   "output_per_m":     Decimal,   # output tokens, per 1M
#   "cache_read_per_m": Decimal,   # cache-read input tokens, per 1M
#   "cache_write_per_m":Decimal,   # cache-creation input tokens, per 1M
# }
#
# Rates are in USD.
# cache_read  ≈ 10 % of input_per_m (claude-fable-5-1: 2.5 % — Anthropic cut it 75 %)
# cache_write ≈ 125 % of input_per_m
_RATES: dict[str, dict[str, Decimal]] = {
    # ------------------------------------------------------------------
    # Claude Sonnet 4.5 family  ($3 / $15 / $0.30 / $3.75)
    # ------------------------------------------------------------------
    "claude-sonnet-4-5": {
        "input_per_m": Decimal("3.00"),
        "output_per_m": Decimal("15.00"),
        "cache_read_per_m": Decimal("0.30"),
        "cache_write_per_m": Decimal("3.75"),
    },
    "claude-sonnet-4-5-20250929": {
        "input_per_m": Decimal("3.00"),
        "output_per_m": Decimal("15.00"),
        "cache_read_per_m": Decimal("0.30"),
        "cache_write_per_m": Decimal("3.75"),
    },
    # claude-sonnet-4-6 is used as a fallback alias for Sonnet
    "claude-sonnet-4-6": {
        "input_per_m": Decimal("3.00"),
        "output_per_m": Decimal("15.00"),
        "cache_read_per_m": Decimal("0.30"),
        "cache_write_per_m": Decimal("3.75"),
    },
    # The introductory $2/$10 price is now permanent. The previously
    # scheduled September increase was cancelled (pricing page footnote 3).
    "claude-sonnet-5": {
        "input_per_m": Decimal("2.00"),
        "output_per_m": Decimal("10.00"),
        "cache_read_per_m": Decimal("0.20"),
        "cache_write_per_m": Decimal("2.50"),
    },
    # Claude Sonnet 5.5: cache reads reduced from $0.20 to $0.10/MTok.
    # First-party pricing verified 2026-10-07; only the documented fixed ID.
    "claude-sonnet-5-5": {
        "input_per_m": Decimal("2.00"),
        "output_per_m": Decimal("10.00"),
        "cache_read_per_m": Decimal("0.10"),
        "cache_write_per_m": Decimal("2.50"),
    },
    # ------------------------------------------------------------------
    # Claude Opus 4.5 / 4.6 / 4.7 family  ($5 / $25 / $0.50 / $6.25)
    # Source: anthropic.com/news/claude-opus-4-7 (verified 2026-05-07)
    # Anthropic lowered Opus pricing with the 4.5 launch (Nov 2025).
    # 4.6 and 4.7 kept the same rates: $5 input / $25 output.
    # cache_read = 10% of input ($0.50); cache_write = 125% of input ($6.25)
    # NOTE: the legacy claude-opus-4-20250514 row below retains $15/$75.
    # ------------------------------------------------------------------
    "claude-opus-4-5": {
        "input_per_m": Decimal("5.00"),
        "output_per_m": Decimal("25.00"),
        "cache_read_per_m": Decimal("0.50"),
        "cache_write_per_m": Decimal("6.25"),
    },
    "claude-opus-4-5-20251101": {
        "input_per_m": Decimal("5.00"),
        "output_per_m": Decimal("25.00"),
        "cache_read_per_m": Decimal("0.50"),
        "cache_write_per_m": Decimal("6.25"),
    },
    "claude-opus-4-6": {
        "input_per_m": Decimal("5.00"),
        "output_per_m": Decimal("25.00"),
        "cache_read_per_m": Decimal("0.50"),
        "cache_write_per_m": Decimal("6.25"),
    },
    "claude-opus-4-6-20260101": {
        "input_per_m": Decimal("5.00"),
        "output_per_m": Decimal("25.00"),
        "cache_read_per_m": Decimal("0.50"),
        "cache_write_per_m": Decimal("6.25"),
    },
    "claude-opus-4-7": {
        "input_per_m": Decimal("5.00"),
        "output_per_m": Decimal("25.00"),
        "cache_read_per_m": Decimal("0.50"),
        "cache_write_per_m": Decimal("6.25"),
    },
    "claude-opus-4-7-20260416": {
        "input_per_m": Decimal("5.00"),
        "output_per_m": Decimal("25.00"),
        "cache_read_per_m": Decimal("0.50"),
        "cache_write_per_m": Decimal("6.25"),
    },
    # ------------------------------------------------------------------
    # Claude Opus 4.8  ($5 / $25 / $0.50 / $6.25)
    # ------------------------------------------------------------------
    "claude-opus-4-8": {
        "input_per_m": Decimal("5.00"),
        "output_per_m": Decimal("25.00"),
        "cache_read_per_m": Decimal("0.50"),
        "cache_write_per_m": Decimal("6.25"),
    },
    # ------------------------------------------------------------------
    # Claude Opus 5  ($5 / $25 / $0.50 / $6.25 — same rates as Opus 4.8)
    # Source: docs.anthropic.com/en/docs/about-claude/pricing (verified 2026-07-24)
    # ------------------------------------------------------------------
    "claude-opus-5": {
        "input_per_m": Decimal("5.00"),
        "output_per_m": Decimal("25.00"),
        "cache_read_per_m": Decimal("0.50"),
        "cache_write_per_m": Decimal("6.25"),
    },
    # Claude Opus 5.5 ($4 / $20 / $0.20 / $5); a one-hour cache write is
    # derived by compute_cost() as 2x input ($8/MTok).
    "claude-opus-5-5": {
        "input_per_m": Decimal("4.00"),
        "output_per_m": Decimal("20.00"),
        "cache_read_per_m": Decimal("0.20"),
        "cache_write_per_m": Decimal("5.00"),
    },
    # ------------------------------------------------------------------
    # Claude Fable 5  ($10 / $50 / $1.00 / $12.50)
    # Exactly 2x Opus 4.8 on every rate -- true of Fable 5 ITSELF, not of
    # the fable family: Fable 5.1 below breaks it on cache reads.
    # A 1-hour cache write tier exists at $20.00/MTok (= 2x input_per_m, the
    # same relationship as every other model in this table). Anthropic's
    # usage object now reports a per-TTL split via `usage.cache_creation`
    # (`.ephemeral_5m_input_tokens` / `.ephemeral_1h_input_tokens`) — see the
    # TTL-aware billing in compute_cost() below. cache_write_per_m here is
    # the 5-minute rate ($12.50), used for the split's 5m portion and as the
    # legacy aggregate-only fallback rate.
    # ------------------------------------------------------------------
    "claude-fable-5": {
        "input_per_m": Decimal("10.00"),
        "output_per_m": Decimal("50.00"),
        "cache_read_per_m": Decimal("1.00"),
        "cache_write_per_m": Decimal("12.50"),
    },
    # ------------------------------------------------------------------
    # Claude Fable 5.1  ($10 / $50 / $0.25 / $12.50)
    # Same input/output/cache-write rates as Fable 5; cache reads reduced
    # 75% to $0.25/MTok (from $1.00/MTok on Fable 5).
    # Source: https://www.anthropic.com/pricing (verified 2026-09-02)
    #         https://www.anthropic.com/claude-fable-and-mythos-5-1
    # ------------------------------------------------------------------
    "claude-fable-5-1": {
        "input_per_m": Decimal("10.00"),
        "output_per_m": Decimal("50.00"),
        "cache_read_per_m": Decimal("0.25"),
        "cache_write_per_m": Decimal("12.50"),
    },
    # Fable 5.2 standard first-party USD/MTok, verified 2026-11-12:
    # https://platform.claude.com/docs/en/models/fable-5-2/overview
    # One-hour writes are derived below as 2x input ($16/MTok).
    "claude-fable-5-2": {
        "input_per_m": Decimal("8.00"),
        "output_per_m": Decimal("40.00"),
        "cache_read_per_m": Decimal("0.20"),
        "cache_write_per_m": Decimal("10.00"),
    },
    # ------------------------------------------------------------------
    # Claude Haiku 3.5  ($0.80 / $4.00 / $0.08 / $1.00)
    # ------------------------------------------------------------------
    "claude-haiku-3-5-20250929": {
        "input_per_m": Decimal("0.80"),
        "output_per_m": Decimal("4.00"),
        "cache_read_per_m": Decimal("0.08"),
        "cache_write_per_m": Decimal("1.00"),
    },
    # ------------------------------------------------------------------
    # Claude Haiku 4.5 family  ($1.00 / $5.00 / $0.10 / $1.25)
    # ------------------------------------------------------------------
    "claude-haiku-4-5": {
        "input_per_m": Decimal("1.00"),
        "output_per_m": Decimal("5.00"),
        "cache_read_per_m": Decimal("0.10"),
        "cache_write_per_m": Decimal("1.25"),
    },
    "claude-haiku-4-5-20251001": {
        "input_per_m": Decimal("1.00"),
        "output_per_m": Decimal("5.00"),
        "cache_read_per_m": Decimal("0.10"),
        "cache_write_per_m": Decimal("1.25"),
    },
    # ------------------------------------------------------------------
    # Deprecated models
    # ------------------------------------------------------------------
    "claude-3-haiku-20240307": {
        "input_per_m": Decimal("0.25"),
        "output_per_m": Decimal("1.25"),
        "cache_read_per_m": Decimal("0.025"),
        "cache_write_per_m": Decimal("0.3125"),
    },
    "claude-sonnet-4-20250514": {
        "input_per_m": Decimal("3.00"),
        "output_per_m": Decimal("15.00"),
        "cache_read_per_m": Decimal("0.30"),
        "cache_write_per_m": Decimal("3.75"),
    },
    "claude-opus-4-20250514": {
        "input_per_m": Decimal("15.00"),
        "output_per_m": Decimal("75.00"),
        "cache_read_per_m": Decimal("1.50"),
        "cache_write_per_m": Decimal("18.75"),
    },
}

# Models for which the 2x fast-mode multiplier applies when speed=='fast'.
# The 2x cost multiplier is applied ONLY when BOTH the response confirms
# speed=='fast' AND the model is listed here — this prevents a silent API
# fallback to standard speed (or misconfigured caller) from inflating
# tracked cost.
_FAST_ELIGIBLE_MODELS: set[str] = {
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-fable-5-2",
}

_HAIKU_MODEL = "claude-haiku-5-5"
_HAIKU_THRESHOLD = 100_000
_HAIKU_RATES = (
    {
        "input_per_m": Decimal(".10"),
        "output_per_m": Decimal(".50"),
        "cache_read_per_m": Decimal(".01"),
        "cache_write_per_m": Decimal(".125"),
    },
    {
        "input_per_m": Decimal(".50"),
        "output_per_m": Decimal("2.50"),
        "cache_read_per_m": Decimal(".05"),
        "cache_write_per_m": Decimal(".625"),
    },
)


def get_rates(model: str, *, prompt_tokens: int | None = None) -> dict[str, Decimal] | None:
    """Fresh Decimal rate mapping; exact IDs only, no guessed family aliases."""
    if model == _HAIKU_MODEL:
        if prompt_tokens is None:
            return None
        source = _HAIKU_RATES[int(prompt_tokens > _HAIKU_THRESHOLD)]
    else:
        source = _RATES.get(model)
    if source is None:
        return None
    return {**source, "cache_write_1h_per_m": source["input_per_m"] * 2}


def pricing_metadata(model: str) -> dict | None:
    """JSON-safe display contract in provider-local ModelInfo.pricing_details.

    These are base standard/global USD/MTok rates, not an actual request cost.
    Decimal strings preserve precision; flat Core Pricing is derived separately
    from get_rates only when the model is not tiered.
    """
    def display(rates: dict[str, Decimal]) -> dict[str, str]:
        return {
            "input": str(rates["input_per_m"]),
            "output": str(rates["output_per_m"]),
            "cache_read": str(rates["cache_read_per_m"]),
            "cache_write_5m": str(rates["cache_write_per_m"]),
            "cache_write_1h": str(rates["cache_write_1h_per_m"]),
        }

    rates = get_rates(model)
    if rates is None and model != _HAIKU_MODEL:
        return None
    result = {
        "version": 1,
        "kind": "tiered" if model == _HAIKU_MODEL else "fixed",
        "currency": "USD",
        "unit": "per_million_tokens",
        "basis": "standard_global",
        "modifiers": {},
    }
    if model in {
        "claude-sonnet-4-6", "claude-sonnet-5", "claude-sonnet-5-5",
        "claude-opus-4-6", "claude-opus-4-6-20260101", "claude-opus-4-7",
        "claude-opus-4-7-20260416", "claude-opus-4-8", "claude-opus-5",
        "claude-opus-5-5", "claude-fable-5", "claude-fable-5-1",
        "claude-fable-5-2", _HAIKU_MODEL,
    }:
        result["modifiers"]["us_inference"] = "1.1"
    if model in _FAST_ELIGIBLE_MODELS:
        result["modifiers"]["confirmed_fast"] = "2"
    if model == _HAIKU_MODEL:
        result.update({
            "threshold_tokens": _HAIKU_THRESHOLD,
            "threshold_basis": "input_tokens+cache_read_input_tokens+cache_creation_input_tokens",
            "tiers": [
                {"up_to": _HAIKU_THRESHOLD, "rates": display(get_rates(model, prompt_tokens=0))},
                {"above": _HAIKU_THRESHOLD, "rates": display(get_rates(model, prompt_tokens=100_001))},
            ],
        })
    else:
        result["rates"] = display(rates)
    return result


@dataclass(frozen=True)
class CostResult:
    cost_usd: Decimal | None
    unavailable_reason: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_cost_result(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int | None = 0,
    cache_creation_5m_input_tokens: int | None = None,
    cache_creation_1h_input_tokens: int | None = None,
    speed: str | None = None,
    inference_geo: str | None = "global",
    requested_speed: str | None = None,
    cache_write_ttl: str | None = None,
    strict: bool = False,
) -> CostResult:
    """Return a Decimal USD token cost or a machine-readable unknown reason.

    Parameters
    ----------
    model:
        Anthropic model identifier (e.g. ``"claude-sonnet-4-5-20250929"``).
    input_tokens:
        Fresh (non-cached) input tokens consumed.  This matches the
        ``input_tokens`` field returned by Anthropic's API, which already
        excludes cached tokens — no subtraction needed.
    output_tokens:
        Output tokens generated.
    cache_read_input_tokens:
        Tokens served from the prompt cache (cheaper than fresh input).
    cache_creation_input_tokens:
        Aggregate tokens written to the prompt cache, regardless of TTL
        (Anthropic's ``usage.cache_creation_input_tokens`` field). Used as
        the legacy fallback billing basis — billed at the 5-minute rate —
        when *neither* ``cache_creation_5m_input_tokens`` nor
        ``cache_creation_1h_input_tokens`` is supplied.
    cache_creation_5m_input_tokens:
        Tokens written to the prompt cache with a 5-minute TTL (Anthropic's
        ``usage.cache_creation.ephemeral_5m_input_tokens``), billed at 1.25x
        the base input rate. Pass ``None`` (the default) when the caller's
        usage object doesn't carry the per-TTL split; passing either split
        parameter (even as ``0``) switches billing to split mode for this
        call, so the discrepancy check below applies.
    cache_creation_1h_input_tokens:
        Tokens written to the prompt cache with a 1-hour TTL (Anthropic's
        ``usage.cache_creation.ephemeral_1h_input_tokens``), billed at 2x
        the base input rate (1.6x the 5m write rate). Undercounting this was the root cause of the
        cost-tracking bug this split-aware path fixes: before, all cache
        writes — including 1h ones — were billed at the 5-minute (1.25x)
        rate because the usage object only exposed the aggregate count.
    speed:
        When ``'fast'`` AND *model* is in :data:`_FAST_ELIGIBLE_MODELS` a 2x
        multiplier is applied; any other value leaves cost unchanged.

    Returns
    -------
    CostResult
        cost_usd is Decimal (including true zero), or None with a reason.
        Strict mode needs a complete consistent TTL split, or a known request
        TTL when nonzero cache writes are reported only as an aggregate.
    """
    # Public compute_cost retains legacy aggregate-at-5m behavior on fixed
    # models. Runtime strict mode refuses an unresolved TTL instead of silently
    # understating a one-hour write. Tiered Haiku always needs truthful usage.
    def unknown(reason: str) -> CostResult:
        return CostResult(None, reason)

    counts = (input_tokens, output_tokens, cache_read_input_tokens)
    optional_counts = (
        cache_creation_input_tokens, cache_creation_5m_input_tokens,
        cache_creation_1h_input_tokens,
    )
    if any(type(n) is not int or n < 0 for n in counts) or any(
        n is not None and (type(n) is not int or n < 0) for n in optional_counts
    ):
        return unknown("invalid_usage_counts")
    has_ttl_split = (
        cache_creation_5m_input_tokens is not None
        or cache_creation_1h_input_tokens is not None
    )
    split_total = (cache_creation_5m_input_tokens or 0) + (cache_creation_1h_input_tokens or 0)
    aggregate = cache_creation_input_tokens or 0
    if strict or model == _HAIKU_MODEL:
        if has_ttl_split and (
            cache_creation_5m_input_tokens is None or cache_creation_1h_input_tokens is None
        ):
            return unknown("cache_write_ttl_split_incomplete")
        if has_ttl_split and cache_creation_input_tokens is not None and aggregate != split_total:
            return unknown("cache_write_ttl_split_mismatch")
        if not has_ttl_split and aggregate and cache_write_ttl not in {"5m", "1h"}:
            return unknown("cache_write_ttl_unavailable")
    prompt_tokens = input_tokens + cache_read_input_tokens + (
        split_total if has_ttl_split else aggregate
    )
    rates = get_rates(model, prompt_tokens=prompt_tokens)
    if rates is None:
        return unknown("model_rates_unavailable")
    if not isinstance(inference_geo, str) or inference_geo not in {"global", "us"}:
        return unknown("inference_geo_unavailable")
    if requested_speed is not None and not isinstance(requested_speed, str):
        return unknown("request_speed_unavailable")
    if requested_speed == "fast" and speed not in {"fast", "standard"}:
        return unknown("response_speed_unavailable")
    if speed is not None and (not isinstance(speed, str) or speed not in {"fast", "standard"}):
        return unknown("response_speed_unavailable")
    if strict and speed == "fast" and model not in _FAST_ELIGIBLE_MODELS:
        return unknown("fast_mode_rates_unavailable")

    cost = (
        Decimal(input_tokens) * rates["input_per_m"] / _PER_M
        + Decimal(output_tokens) * rates["output_per_m"] / _PER_M
    )

    if cache_read_input_tokens > 0:
        cost += Decimal(cache_read_input_tokens) * rates["cache_read_per_m"] / _PER_M

    # Per-TTL split billing: only engaged when the caller actually supplies
    # part of the split (usage.cache_creation was present on the SDK usage
    # object). Passing neither param preserves the pre-split legacy path
    # below unchanged — a graceful fallback for older SDK response shapes
    # (or test doubles) that only carry the aggregate count.
    has_ttl_split = (
        cache_creation_5m_input_tokens is not None
        or cache_creation_1h_input_tokens is not None
    )

    if has_ttl_split:
        five_min_tokens = cache_creation_5m_input_tokens or 0
        one_hour_tokens = cache_creation_1h_input_tokens or 0

        # Sanity-check consistency: the split should sum to the aggregate.
        # If it doesn't, prefer the split (it is the more precise, billable
        # figure) and note the discrepancy at debug level only — this is
        # observability for an unexpected SDK response shape, not a user
        # actionable warning.
        split_total = five_min_tokens + one_hour_tokens
        if cache_creation_input_tokens and split_total != cache_creation_input_tokens:
            logger.debug(
                "cache_creation TTL split (5m=%s + 1h=%s = %s) does not match "
                "aggregate cache_creation_input_tokens=%s for model %s; "
                "billing from the split.",
                five_min_tokens,
                one_hour_tokens,
                split_total,
                cache_creation_input_tokens,
                model,
            )

        if five_min_tokens > 0:
            cost += Decimal(five_min_tokens) * rates["cache_write_per_m"] / _PER_M
        if one_hour_tokens > 0:
            # 1-hour cache writes cost 2x the base input rate (Anthropic
            # pricing), not the 1.25x 5-minute cache_write_per_m rate.
            cost += Decimal(one_hour_tokens) * (rates["input_per_m"] * 2) / _PER_M
    elif aggregate > 0:
        # Legacy path: no TTL split available, bill the full aggregate at
        # the 5-minute rate (unchanged historical behavior).
        cost += (
            Decimal(aggregate) * rates[
                "cache_write_1h_per_m" if cache_write_ttl == "1h" else "cache_write_per_m"
            ] / _PER_M
        )

    if speed == "fast" and model in _FAST_ELIGIBLE_MODELS:
        cost *= 2
    if inference_geo == "us":
        cost *= Decimal("1.1")
    return CostResult(cost)


def compute_cost(
    model: str, *, input_tokens: int = 0, output_tokens: int = 0,
    cache_read_input_tokens: int = 0, cache_creation_input_tokens: int | None = 0,
    cache_creation_5m_input_tokens: int | None = None,
    cache_creation_1h_input_tokens: int | None = None,
    speed: str | None = None, inference_geo: str | None = "global",
    requested_speed: str | None = None, cache_write_ttl: str | None = None,
    strict: bool = False,
) -> Decimal | None:
    """Backward-compatible Decimal-or-None API; see compute_cost_result."""
    return compute_cost_result(
        model, input_tokens=input_tokens, output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_creation_5m_input_tokens=cache_creation_5m_input_tokens,
        cache_creation_1h_input_tokens=cache_creation_1h_input_tokens,
        speed=speed, inference_geo=inference_geo, requested_speed=requested_speed,
        cache_write_ttl=cache_write_ttl, strict=strict,
    ).cost_usd
