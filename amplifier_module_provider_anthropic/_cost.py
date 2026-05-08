"""Anthropic pricing rates and cost computation.

Verification date: 2026-05-06
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

from decimal import Decimal

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
        Tokens written to the prompt cache (slightly more expensive than
        fresh input).

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

    if cache_creation_input_tokens > 0:
        cost += (
            Decimal(cache_creation_input_tokens) * rates["cache_write_per_m"] / _PER_M
        )

    return cost
