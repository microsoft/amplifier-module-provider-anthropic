"""Tests for pricing data, cost fields in ModelInfo, and vision capability.

Validates:
1. _ANTHROPIC_PRICING constant exists with correct values
2. list_models() populates cost_per_input_token, cost_per_output_token, metadata
3. All Claude models include 'vision' in capability_tags
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_provider_anthropic import AnthropicProvider


# --- 1. _ANTHROPIC_PRICING constant ---


class TestAnthropicPricingConstant:
    """Verify the module-level _ANTHROPIC_PRICING dict exists with correct values."""

    def test_pricing_constant_exists(self):
        from amplifier_module_provider_anthropic import _ANTHROPIC_PRICING

        assert isinstance(_ANTHROPIC_PRICING, dict)

    def test_opus_pricing(self):
        from amplifier_module_provider_anthropic import _ANTHROPIC_PRICING

        opus = _ANTHROPIC_PRICING["opus"]
        assert opus["input"] == 15.0e-6
        assert opus["output"] == 75.0e-6
        assert opus["tier"] == "high"

    def test_sonnet_pricing(self):
        from amplifier_module_provider_anthropic import _ANTHROPIC_PRICING

        sonnet = _ANTHROPIC_PRICING["sonnet"]
        assert sonnet["input"] == 3.0e-6
        assert sonnet["output"] == 15.0e-6
        assert sonnet["tier"] == "medium"

    def test_haiku_pricing(self):
        from amplifier_module_provider_anthropic import _ANTHROPIC_PRICING

        haiku = _ANTHROPIC_PRICING["haiku"]
        assert haiku["input"] == 0.80e-6
        assert haiku["output"] == 4.0e-6
        assert haiku["tier"] == "low"


# --- 2. list_models() cost fields ---


class TestListModelsCostFields:
    """Verify list_models() populates cost and metadata on ModelInfo."""

    @pytest.fixture
    def provider(self):
        """Create provider without API key (we mock the API call)."""
        return AnthropicProvider(api_key="test-key", config={"filtered": True})

    def _make_mock_model(self, model_id: str, display_name: str):
        m = MagicMock()
        m.id = model_id
        m.display_name = display_name
        m.created_at = "2025-01-01"
        return m

    @pytest.mark.asyncio
    async def test_sonnet_model_has_cost_fields(self, provider):
        mock_response = MagicMock()
        mock_response.data = [
            self._make_mock_model("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5"),
        ]

        with patch.object(provider, "_client", create=True) as mock_client:
            mock_client.models.list = AsyncMock(return_value=mock_response)
            # Override the client property
            provider._client = mock_client

            models = await provider.list_models()

        sonnet_models = [m for m in models if "sonnet" in m.id]
        assert len(sonnet_models) >= 1
        model = sonnet_models[0]
        assert model.cost_per_input_token == 3.0e-6
        assert model.cost_per_output_token == 15.0e-6
        assert model.metadata == {"cost_tier": "medium"}

    @pytest.mark.asyncio
    async def test_opus_model_has_cost_fields(self, provider):
        mock_response = MagicMock()
        mock_response.data = [
            self._make_mock_model("claude-opus-4-6-20260101", "Claude Opus 4.6"),
        ]

        with patch.object(provider, "_client", create=True) as mock_client:
            mock_client.models.list = AsyncMock(return_value=mock_response)
            provider._client = mock_client

            models = await provider.list_models()

        opus_models = [m for m in models if "opus" in m.id]
        assert len(opus_models) >= 1
        model = opus_models[0]
        assert model.cost_per_input_token == 15.0e-6
        assert model.cost_per_output_token == 75.0e-6
        assert model.metadata == {"cost_tier": "high"}

    @pytest.mark.asyncio
    async def test_haiku_model_has_cost_fields(self, provider):
        mock_response = MagicMock()
        mock_response.data = [
            self._make_mock_model("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
        ]

        with patch.object(provider, "_client", create=True) as mock_client:
            mock_client.models.list = AsyncMock(return_value=mock_response)
            provider._client = mock_client

            models = await provider.list_models()

        haiku_models = [m for m in models if "haiku" in m.id]
        assert len(haiku_models) >= 1
        model = haiku_models[0]
        assert model.cost_per_input_token == 0.80e-6
        assert model.cost_per_output_token == 4.0e-6
        assert model.metadata == {"cost_tier": "low"}


# --- 3. Vision capability in all models ---


class TestVisionCapability:
    """Verify 'vision' is in capability_tags for all Claude model families."""

    def test_opus_has_vision(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert "vision" in caps.capability_tags

    def test_opus_45_has_vision(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert "vision" in caps.capability_tags

    def test_sonnet_has_vision(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-5-20250929")
        assert "vision" in caps.capability_tags

    def test_haiku_35_has_vision(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert "vision" in caps.capability_tags

    def test_haiku_45_has_vision(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert "vision" in caps.capability_tags

    def test_unknown_model_falls_through_to_sonnet(self):
        """Unknown family falls through to sonnet (default) which has vision."""
        caps = AnthropicProvider._get_capabilities("claude-mystery-9-9")
        # _detect_family returns 'sonnet' for unknown models
        assert caps.family == "sonnet"
        assert "vision" in caps.capability_tags

    def test_vision_coexists_with_other_tags(self):
        """Vision tag should be added alongside existing tags, not replace them."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-5-20250929")
        assert "tools" in caps.capability_tags
        assert "thinking" in caps.capability_tags
        assert "streaming" in caps.capability_tags
        assert "json_mode" in caps.capability_tags
        assert "vision" in caps.capability_tags

    def test_haiku_fast_and_vision(self):
        """Haiku retains 'fast' tag alongside new 'vision' tag."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert "fast" in caps.capability_tags
        assert "vision" in caps.capability_tags
        assert "thinking" in caps.capability_tags
