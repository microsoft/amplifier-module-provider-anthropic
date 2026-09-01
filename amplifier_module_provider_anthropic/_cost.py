"""Anthropic pricing rates and cost computation.

Verification date: 2026-06-10
Source: https://www.anthropic.com/pricing

Usage
-----
    from amplifier_module_provider_anthropic._cost import compute_cost
    from decimal import Decimal

    cost = compute_cost(
        "claude-sonnet-4-5-20250929",
        input_tokens=1_000,
        output_tokens=200,
    )
    # Returns Decimal or None if the model is not recognised.
"""

from __future__ import annotations

import logging
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
# cache_read  ≈ 10 % of input_per_m
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
    # Claude Sonnet 5 (launched 2026-06-30; anthropic.com/news/claude-sonnet-5).
    # Standard rates $3 / $15 (same as the Sonnet 4.x tier). NOTE: an
    # introductory discount of $2 input / $10 output applies through
    # 2026-08-31 only; we deliberately encode the durable STANDARD rates here
    # (no time-windowed pricing logic anywhere in this table). The updated
    # Sonnet 5 tokenizer maps the same text to ~1.0-1.35x more tokens, which
    # raises effective per-request cost even at identical rates.
    "claude-sonnet-5": {
        "input_per_m": Decimal("3.00"),
        "output_per_m": Decimal("15.00"),
        "cache_read_per_m": Decimal("0.30"),
        "cache_write_per_m": Decimal("3.75"),
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
    # ------------------------------------------------------------------
    # Claude Fable 5  ($10 / $50 / $1.00 / $12.50)
    # Exactly 2x Opus 4.8 on every rate.
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
    # Same input and output rates as Fable 5, but cache reads are 75% cheaper
    # ($0.25/MTok vs $1.00/MTok on Fable 5). Cache writes (5-min) remain at
    # $12.50/MTok; 1-hour cache writes remain at 2x input ($20.00/MTok).
    # Source: https://www.anthropic.com/claude-fable-and-mythos-5-1
    #         (verified 2026-09-01: "Cache reads now cost 75% less, or $0.25
    #          per million tokens.")
    # ------------------------------------------------------------------
    "claude-fable-5-1": {
        "input_per_m": Decimal("10.00"),
        "output_per_m": Decimal("50.00"),
        "cache_read_per_m": Decimal("0.25"),
        "cache_write_per_m": Decimal("12.50"),
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
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_cost(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_creation_5m_input_tokens: int | None = None,
    cache_creation_1h_input_tokens: int | None = None,
    speed: str | None = None,
) -> Decimal | None:
    """Return the USD cost for an Anthropic API call as a :class:`~decimal.Decimal`.

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
        the base input rate (official Anthropic pricing — 1h writes cost
        double a 5m write). Undercounting this was the root cause of the
        cost-tracking bug this split-aware path fixes: before, all cache
        writes — including 1h ones — were billed at the 5-minute (1.25x)
        rate because the usage object only exposed the aggregate count.
    speed:
        When ``'fast'`` AND *model* is in :data:`_FAST_ELIGIBLE_MODELS` a 2x
        multiplier is applied; any other value leaves cost unchanged.

    Returns
    -------
    Decimal | None
        The computed cost in USD, or ``None`` if *model* is not recognised.
        ``None`` is semantically distinct from ``Decimal('0')`` (a free call).
    """
    rates = _RATES.get(model)
    if rates is None:
        return None

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
    elif cache_creation_input_tokens > 0:
        # Legacy path: no TTL split available, bill the full aggregate at
        # the 5-minute rate (unchanged historical behavior).
        cost += (
            Decimal(cache_creation_input_tokens) * rates["cache_write_per_m"] / _PER_M
        )

    if speed == "fast" and model in _FAST_ELIGIBLE_MODELS:
        cost *= 2

    return cost
