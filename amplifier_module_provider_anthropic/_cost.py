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

Config-level rate overrides
---------------------------
Models absent from ``_RATES`` (private, brand-new, or fine-tuned models) would
otherwise always yield ``None``.  The provider's ``rates`` config key lets a
deployment supply rates for such models without a code change:

    overrides = parse_rate_overrides({
        "claude-fable-5": {"input": 10.00, "output": 50.00},
    })
    cost = compute_cost("claude-fable-5", input_tokens=1_000, rate_overrides=overrides)

Overrides take precedence over ``_RATES``.  See :func:`parse_rate_overrides`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

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
    # NOTE: A 1-hour cache write tier exists at $20.00/MTok but Anthropic's
    # usage object returns a single cache_creation_input_tokens count and
    # does not distinguish TTLs — track the 5-minute rate ($12.50) here.
    # ------------------------------------------------------------------
    "claude-fable-5": {
        "input_per_m": Decimal("10.00"),
        "output_per_m": Decimal("50.00"),
        "cache_read_per_m": Decimal("1.00"),
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


# Config field name → _RATES row key.  Config values are USD per 1M tokens,
# the same units as _RATES.
_OVERRIDE_FIELD_MAP: dict[str, str] = {
    "input": "input_per_m",
    "output": "output_per_m",
    "cache_read": "cache_read_per_m",
    "cache_write": "cache_write_per_m",
}

# Default cache rates as a fraction of the input rate.  Every row in _RATES
# follows these ratios (cache_read = 10% of input, cache_write = 125% of
# input), so derived defaults keep overrides consistent with the table.
_CACHE_READ_RATIO = Decimal("0.10")
_CACHE_WRITE_RATIO = Decimal("1.25")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_rate_overrides(raw: Any) -> dict[str, dict[str, Decimal]]:
    """Parse the provider's ``rates`` config value into ``_RATES``-shaped rows.

    Parameters
    ----------
    raw:
        The value of the ``rates`` config key: a mapping of model id (exact,
        or a trailing-``*`` glob such as ``"claude-fable-*"``) → mapping with
        keys ``input``, ``output`` (required) and ``cache_read``,
        ``cache_write`` (optional), all in USD per 1M tokens.  Values may be
        int, float, or string; each is parsed via ``Decimal(str(value))`` so
        no float arithmetic ever touches the cost path.

    Returns
    -------
    dict[str, dict[str, Decimal]]
        Mapping of model pattern → rate row with the same keys as ``_RATES``
        rows (``input_per_m``, ``output_per_m``, ``cache_read_per_m``,
        ``cache_write_per_m``).  Suitable for ``compute_cost``'s
        ``rate_overrides`` parameter.

    Notes
    -----
    * Missing ``cache_read`` defaults to 10% of ``input``; missing
      ``cache_write`` defaults to 125% of ``input`` — the ratios every
      ``_RATES`` row uses.
    * Invalid entries (missing ``input``/``output``, non-numeric or negative
      values, unknown field names) are skipped with a warning; they never
      raise, matching the provider's lenient config parsing.
    """
    overrides: dict[str, dict[str, Decimal]] = {}
    if raw is None:
        return overrides
    if not isinstance(raw, Mapping):
        logger.warning(
            "[PROVIDER] Invalid 'rates' config value %r (expected mapping); ignoring",
            raw,
        )
        return overrides

    for model_pattern, fields in raw.items():
        if not isinstance(model_pattern, str) or not model_pattern:
            logger.warning(
                "[PROVIDER] Invalid model pattern %r in 'rates' config; skipping",
                model_pattern,
            )
            continue
        if not isinstance(fields, Mapping):
            logger.warning(
                "[PROVIDER] Invalid rates entry for %r (expected mapping, got %r); skipping",
                model_pattern,
                fields,
            )
            continue

        unknown = set(fields) - set(_OVERRIDE_FIELD_MAP)
        if unknown:
            logger.warning(
                "[PROVIDER] Unknown field(s) %s in 'rates' entry for %r; skipping",
                sorted(unknown),
                model_pattern,
            )
            continue

        row: dict[str, Decimal] = {}
        valid = True
        for field, row_key in _OVERRIDE_FIELD_MAP.items():
            if field not in fields:
                continue
            value = fields[field]
            try:
                parsed = Decimal(str(value))
            except (InvalidOperation, ValueError, TypeError):
                parsed = None
            if parsed is None or not parsed.is_finite() or parsed < 0:
                logger.warning(
                    "[PROVIDER] Invalid %r value %r in 'rates' entry for %r; skipping entry",
                    field,
                    value,
                    model_pattern,
                )
                valid = False
                break
            row[row_key] = parsed
        if not valid:
            continue

        if "input_per_m" not in row or "output_per_m" not in row:
            logger.warning(
                "[PROVIDER] 'rates' entry for %r must define both 'input' and 'output'; skipping",
                model_pattern,
            )
            continue

        row.setdefault("cache_read_per_m", row["input_per_m"] * _CACHE_READ_RATIO)
        row.setdefault("cache_write_per_m", row["input_per_m"] * _CACHE_WRITE_RATIO)
        overrides[model_pattern] = row

    return overrides


def _resolve_rates(
    model: str,
    rate_overrides: Mapping[str, Mapping[str, Decimal]] | None,
) -> Mapping[str, Decimal] | None:
    """Resolve the rate row for *model*: overrides first, then ``_RATES``.

    Override precedence: exact model id, then trailing-``*`` glob patterns
    (longest matching prefix wins), then the built-in ``_RATES`` table.
    """
    if rate_overrides:
        exact = rate_overrides.get(model)
        if exact is not None:
            return exact
        best: Mapping[str, Decimal] | None = None
        best_len = -1
        for pattern, row in rate_overrides.items():
            if not pattern.endswith("*"):
                continue
            prefix = pattern[:-1]
            if model.startswith(prefix) and len(prefix) > best_len:
                best = row
                best_len = len(prefix)
        if best is not None:
            return best
    return _RATES.get(model)


def compute_cost(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    speed: str | None = None,
    rate_overrides: Mapping[str, Mapping[str, Decimal]] | None = None,
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
    speed:
        When ``'fast'`` AND *model* is in :data:`_FAST_ELIGIBLE_MODELS` a 2x
        multiplier is applied; any other value leaves cost unchanged.
    rate_overrides:
        Optional config-supplied rate rows keyed by exact model id or
        trailing-``*`` glob, as produced by :func:`parse_rate_overrides`.
        Overrides take precedence over ``_RATES``.

    Returns
    -------
    Decimal | None
        The computed cost in USD, or ``None`` if *model* is not recognised
        in either *rate_overrides* or ``_RATES``.  ``None`` is semantically
        distinct from ``Decimal('0')`` (a free call).
    """
    rates = _resolve_rates(model, rate_overrides)
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

    if speed == "fast" and model in _FAST_ELIGIBLE_MODELS:
        cost *= 2

    return cost
