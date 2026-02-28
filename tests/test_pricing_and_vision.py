"""Tests for vision capability on all Claude model families."""

from amplifier_module_provider_anthropic import AnthropicProvider


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
