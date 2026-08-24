"""Tests for _build_pricing(): wiring _RATES into ModelInfo.pricing.

list_models() surfaces pricing previously only used internally for cost
accounting (see _cost.py / compute_cost). _build_pricing() is the pure
function that does the _RATES -> Pricing translation; tested directly here
since no existing test mocks the async client.models.list() call.

TestListModelsPricingWiring (below) additionally mocks client.models.list()
to guard the pricing=_build_pricing(model_id) call site in list_models()
itself -- the unit tests above would all still pass even if that argument
were deleted from the ModelInfo(...) construction.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import Pricing
from amplifier_module_provider_anthropic import AnthropicProvider, _build_pricing
from amplifier_module_provider_anthropic._cost import _RATES


class TestBuildPricing:
    """_build_pricing() translates _RATES entries into Pricing objects."""

    def test_known_model_returns_populated_pricing(self):
        pricing = _build_pricing("claude-sonnet-4-5-20250929")

        assert pricing is not None
        assert isinstance(pricing, Pricing)
        assert pricing.input_per_million > 0
        assert pricing.output_per_million > 0
        assert pricing.currency == "USD"

    def test_pricing_matches_rates_table(self):
        rates = _RATES["claude-sonnet-4-5-20250929"]
        pricing = _build_pricing("claude-sonnet-4-5-20250929")

        assert pricing is not None
        assert pricing.input_per_million == float(rates["input_per_m"])
        assert pricing.output_per_million == float(rates["output_per_m"])
        assert pricing.cache_read_per_million == float(rates["cache_read_per_m"])
        assert pricing.cache_write_per_million == float(rates["cache_write_per_m"])

    def test_unknown_model_returns_none(self):
        assert _build_pricing("claude-mystery-9-9") is None

    def test_all_rate_table_entries_build_valid_pricing(self):
        """Every model in _RATES should produce a valid Pricing object."""
        for model_id in _RATES:
            pricing = _build_pricing(model_id)
            assert pricing is not None, f"Expected pricing for {model_id}"
            assert pricing.input_per_million > 0
            assert pricing.output_per_million > 0

    def test_dated_snapshot_of_bare_alias_resolves_via_find_rates(self):
        """claude-sonnet-4-6 is alias-only in _RATES; a fabricated dated
        snapshot of it must still resolve, via _find_rates() normalization.
        """
        rates = _RATES["claude-sonnet-4-6"]
        pricing = _build_pricing("claude-sonnet-4-6-20260201")

        assert pricing is not None
        assert pricing.input_per_million == float(rates["input_per_m"])

    def test_bare_alias_of_dated_only_entry_resolves_via_find_rates(self):
        """claude-haiku-3-5 has only a dated entry in _RATES; the bare alias
        must still resolve, via _find_rates() normalization.
        """
        rates = _RATES["claude-haiku-3-5-20250929"]
        pricing = _build_pricing("claude-haiku-3-5")

        assert pricing is not None
        assert pricing.input_per_million == float(rates["input_per_m"])


class _FakeApiModel:
    """Minimal stand-in for an Anthropic Models API list entry.

    Only carries the attributes list_models() actually reads (id,
    display_name, created_at); other lookups (e.g. capabilities metadata)
    resolve to None via getattr-with-default, matching how the real
    provider handles models the Models API doesn't annotate.
    """

    def __init__(self, model_id: str, display_name: str) -> None:
        self.id = model_id
        self.display_name = display_name
        self.created_at = "2026-01-01T00:00:00Z"


class TestListModelsPricingWiring:
    """Integration test: pricing=_build_pricing(model_id) wiring in list_models().

    Mocks client.models.list() so the assertion exercises the real
    list_models() code path (family grouping, filtering, ModelInfo
    construction) rather than calling _build_pricing() directly.
    """

    @pytest.mark.asyncio
    async def test_list_models_populates_pricing_from_rates(self):
        provider = AnthropicProvider(api_key="test-key")
        # _client is normally lazily created by the `client` property from a
        # real api_key; short-circuit it here with a MagicMock, matching the
        # pattern used in tests/test_close.py (SimpleNamespace fails the
        # AsyncAnthropic | None attribute type check under pyright).
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    # In _RATES -> pricing should be populated.
                    _FakeApiModel("claude-opus-4-8", "Claude Opus 4.8"),
                    # Not in _RATES (fabricated) -> pricing is None.
                    # Uses "sonnet" in the id so it lands in a different
                    # family bucket than claude-opus-4-8 above and isn't
                    # dropped by filtered=True (default) latest-only family
                    # filtering.
                    _FakeApiModel("claude-sonnet-9-9", "Claude Sonnet 9.9 (fake)"),
                ]
            )
        )
        provider._client = mock_client

        models = await provider.list_models()
        by_id = {m.id: m for m in models}

        assert "claude-opus-4-8" in by_id
        assert "claude-sonnet-9-9" in by_id

        rates = _RATES["claude-opus-4-8"]
        opus_pricing = by_id["claude-opus-4-8"].pricing
        assert opus_pricing is not None
        assert opus_pricing.input_per_million == float(rates["input_per_m"])
        assert opus_pricing.output_per_million == float(rates["output_per_m"])

        assert by_id["claude-sonnet-9-9"].pricing is None
