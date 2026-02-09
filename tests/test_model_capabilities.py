"""Tests for model capability detection and version-gated token limits.

Validates that _get_capabilities returns correct max_output_tokens,
thinking budgets, and feature flags for each model family and version.
"""

from amplifier_module_provider_anthropic import AnthropicProvider


class TestDetectFamily:
    """Tests for _detect_family static method."""

    def test_opus_family(self):
        assert AnthropicProvider._detect_family("claude-opus-4-6-20260101") == "opus"

    def test_sonnet_family(self):
        assert (
            AnthropicProvider._detect_family("claude-sonnet-4-5-20250929") == "sonnet"
        )

    def test_haiku_family(self):
        assert AnthropicProvider._detect_family("claude-haiku-3-5-20250929") == "haiku"

    def test_unknown_defaults_to_sonnet(self):
        assert AnthropicProvider._detect_family("claude-mystery-9-9") == "sonnet"

    def test_bare_opus(self):
        assert AnthropicProvider._detect_family("claude-opus-4-6") == "opus"


class TestDetectVersion:
    """Tests for _detect_version static method."""

    def test_opus_46(self):
        assert AnthropicProvider._detect_version(
            "claude-opus-4-6-20260101", "opus"
        ) == (4, 6)

    def test_opus_45(self):
        assert AnthropicProvider._detect_version(
            "claude-opus-4-5-20251101", "opus"
        ) == (4, 5)

    def test_opus_bare_alias(self):
        # Bare alias without date — version not parseable
        assert AnthropicProvider._detect_version("claude-opus-4-6", "opus") == (4, 6)

    def test_unparseable_returns_zero(self):
        assert AnthropicProvider._detect_version("claude-opus-latest", "opus") == (0, 0)


class TestGetCapabilitiesOpus:
    """Tests for Opus model capabilities — the core of the issue #52 fix."""

    def test_opus_45_max_output_tokens(self):
        """Opus 4.5 must use 64000 max_output_tokens (API ceiling)."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.max_output_tokens == 64000

    def test_opus_46_max_output_tokens(self):
        """Opus 4.6+ gets 128000 max_output_tokens."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.max_output_tokens == 128000

    def test_opus_bare_alias_assumes_latest(self):
        """Bare alias 'claude-opus-4-6' should get 4.6+ capabilities."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6")
        assert caps.max_output_tokens == 128000
        assert caps.supports_1m is True
        assert caps.supports_adaptive_thinking is True

    def test_opus_unknown_version_assumes_latest(self):
        """Unknown version defaults to latest (128K) for forward compatibility."""
        caps = AnthropicProvider._get_capabilities("claude-opus-latest")
        assert caps.max_output_tokens == 128000

    def test_opus_45_thinking_budget(self):
        """Opus 4.5 gets reduced thinking budget to stay within 64K ceiling."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.default_thinking_budget == 32000

    def test_opus_46_thinking_budget(self):
        """Opus 4.6+ gets full 64K thinking budget."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.default_thinking_budget == 64000

    def test_opus_45_no_1m_no_adaptive(self):
        """Opus 4.5 does not support 1M context or adaptive thinking."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.supports_1m is False
        assert caps.supports_adaptive_thinking is False

    def test_opus_46_has_1m_and_adaptive(self):
        """Opus 4.6+ supports 1M context and adaptive thinking."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.supports_1m is True
        assert caps.supports_adaptive_thinking is True

    def test_all_opus_supports_thinking(self):
        """All Opus versions support extended thinking."""
        for model_id in ["claude-opus-4-5-20251101", "claude-opus-4-6-20260101"]:
            caps = AnthropicProvider._get_capabilities(model_id)
            assert caps.supports_thinking is True

    def test_opus_family_tag(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.family == "opus"

    def test_opus_thinking_budget_within_ceiling(self):
        """Thinking budget + reasonable buffer must not exceed max_output_tokens.

        This validates the secondary fix: with a 4096 buffer, the thinking
        budget must leave room within the model's output ceiling.
        """
        buffer = 4096
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.default_thinking_budget + buffer <= caps.max_output_tokens


class TestGetCapabilitiesSonnet:
    """Tests for Sonnet model capabilities (should be unaffected by fix)."""

    def test_sonnet_max_output_tokens_is_default(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-5-20250929")
        assert caps.max_output_tokens == 64000

    def test_sonnet_supports_thinking(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-5-20250929")
        assert caps.supports_thinking is True
        assert caps.supports_adaptive_thinking is False

    def test_sonnet_thinking_budget(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-5-20250929")
        assert caps.default_thinking_budget == 32000


class TestGetCapabilitiesHaiku:
    """Tests for Haiku model capabilities (should be unaffected by fix)."""

    def test_haiku_max_output_tokens_is_default(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert caps.max_output_tokens == 64000

    def test_haiku_no_thinking(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert caps.supports_thinking is False

    def test_haiku_family(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert caps.family == "haiku"
