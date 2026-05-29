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


class TestGetCapabilitiesOpus48:
    """Tests for Opus 4.8 capabilities — is_48_plus gate, speed/inline_system flags, max effort."""

    def test_opus_48_supports_speed(self):
        """Opus 4.8 accepts the speed parameter."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        assert caps.supports_speed is True

    def test_opus_48_supports_inline_system(self):
        """Opus 4.8 accepts role='system' in messages[]."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        assert caps.supports_inline_system is True

    def test_opus_48_has_max_effort(self):
        """Opus 4.8 has 'max' effort tier and the full effort tuple."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        assert "max" in caps.supported_efforts
        assert caps.supported_efforts == ("low", "medium", "high", "xhigh", "max")

    def test_opus_47_does_not_support_speed(self):
        """Opus 4.7 does NOT accept the speed parameter."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_speed is False
        assert caps.supports_inline_system is False

    def test_opus_47_no_max_effort(self):
        """Opus 4.7 does not have the 'max' effort tier."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert "max" not in caps.supported_efforts
        assert caps.supported_efforts == ("low", "medium", "high", "xhigh")

    def test_opus_unknown_version_assumes_48(self):
        """Unknown opus version (e.g. claude-opus-latest) assumes 4.8 for forward compatibility."""
        caps = AnthropicProvider._get_capabilities("claude-opus-latest")
        assert caps.supports_speed is True
        assert "max" in caps.supported_efforts


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
    """Tests for Haiku model capabilities — version-gated thinking support.

    Haiku 4.5+ supports extended thinking (per Anthropic docs).
    Haiku 3.5 does NOT support thinking.
    """

    # --- Haiku 3.5 (no thinking) ---

    def test_haiku_35_max_output_tokens_is_default(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert caps.max_output_tokens == 64000

    def test_haiku_35_no_thinking(self):
        """Haiku 3.5 does not support extended thinking."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert caps.supports_thinking is False
        assert caps.supports_adaptive_thinking is False
        assert caps.default_thinking_budget == 0

    def test_haiku_35_no_thinking_tag(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert "thinking" not in caps.capability_tags

    def test_haiku_35_family(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert caps.family == "haiku"

    # --- Haiku 4.5 (thinking supported) ---

    def test_haiku_45_supports_thinking(self):
        """Haiku 4.5 supports extended thinking per Anthropic docs."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.supports_thinking is True

    def test_haiku_45_no_adaptive_thinking(self):
        """Haiku 4.5 does NOT support adaptive thinking per Anthropic docs."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.supports_adaptive_thinking is False

    def test_haiku_45_thinking_budget(self):
        """Haiku 4.5 gets 32K default thinking budget."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.default_thinking_budget == 32000

    def test_haiku_45_has_thinking_tag(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert "thinking" in caps.capability_tags

    def test_haiku_45_has_fast_tag(self):
        """Haiku 4.5 retains the 'fast' tag."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert "fast" in caps.capability_tags

    def test_haiku_45_family(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.family == "haiku"

    def test_haiku_45_max_output_tokens_is_default(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.max_output_tokens == 64000

    # --- Unknown Haiku (defaults to latest = thinking enabled) ---

    def test_haiku_unknown_version_assumes_latest(self):
        """Unknown haiku version defaults to latest (thinking enabled)."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-latest")
        assert caps.supports_thinking is True
        assert caps.default_thinking_budget == 32000


class TestFastModeBetaHeader:
    """Tests for BETA_HEADER_FAST_MODE constant and fast_mode kwarg in _build_request_beta_headers."""

    def test_fast_mode_beta_header_constant(self):
        """BETA_HEADER_FAST_MODE must equal the expected beta header string."""
        from amplifier_module_provider_anthropic import BETA_HEADER_FAST_MODE

        assert BETA_HEADER_FAST_MODE == "fast-mode-2026-02-01"

    def test_beta_header_added_when_fast_mode(self):
        """fast_mode=True must include BETA_HEADER_FAST_MODE in returned headers."""
        from amplifier_module_provider_anthropic import BETA_HEADER_FAST_MODE

        provider = AnthropicProvider(api_key="test-key", config={"max_retries": 0})
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        headers = provider._build_request_beta_headers(
            model_id="claude-opus-4-8",
            request_caps=caps,
            tools_present=False,
            resolved_thinking_type=None,
            fast_mode=True,
        )
        assert BETA_HEADER_FAST_MODE in headers

    def test_beta_header_absent_when_not_fast_mode(self):
        """fast_mode=False must NOT include BETA_HEADER_FAST_MODE in returned headers."""
        from amplifier_module_provider_anthropic import BETA_HEADER_FAST_MODE

        provider = AnthropicProvider(api_key="test-key", config={"max_retries": 0})
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        headers = provider._build_request_beta_headers(
            model_id="claude-opus-4-8",
            request_caps=caps,
            tools_present=False,
            resolved_thinking_type=None,
            fast_mode=False,
        )
        assert BETA_HEADER_FAST_MODE not in headers


class TestContextBetaHeaderOpus48:
    """Opus 4.8+ should NOT get the 1M context beta header (1M is GA)."""

    def test_opus_48_no_1m_beta_header(self):
        from amplifier_module_provider_anthropic import BETA_HEADER_1M_CONTEXT

        provider = AnthropicProvider(api_key="test-key", config={"max_retries": 0})
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        headers = provider._build_request_beta_headers(
            model_id="claude-opus-4-8",
            request_caps=caps,
            tools_present=False,
            resolved_thinking_type=None,
        )
        assert BETA_HEADER_1M_CONTEXT not in headers

    def test_opus_47_still_gets_1m_beta_header(self):
        from amplifier_module_provider_anthropic import BETA_HEADER_1M_CONTEXT

        provider = AnthropicProvider(api_key="test-key", config={"max_retries": 0})
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        headers = provider._build_request_beta_headers(
            model_id="claude-opus-4-7-20260416",
            request_caps=caps,
            tools_present=False,
            resolved_thinking_type=None,
        )
        assert BETA_HEADER_1M_CONTEXT in headers

    def test_opus_unknown_version_no_1m_beta_header(self):
        """Unknown opus version assumes latest (4.8+), so no 1M header needed."""
        from amplifier_module_provider_anthropic import BETA_HEADER_1M_CONTEXT

        provider = AnthropicProvider(api_key="test-key", config={"max_retries": 0})
        caps = AnthropicProvider._get_capabilities("claude-opus-latest")
        headers = provider._build_request_beta_headers(
            model_id="claude-opus-latest",
            request_caps=caps,
            tools_present=False,
            resolved_thinking_type=None,
        )
        assert BETA_HEADER_1M_CONTEXT not in headers


class TestSpeedConfigPlumbing:
    """Tests for speed config key validation and beta header plumbing."""

    def test_supported_model_unsupported_speed_logs_and_omits(self):
        """Opus 4.7 does not support speed — provider omits the param and skips the beta header."""
        from amplifier_module_provider_anthropic import BETA_HEADER_FAST_MODE

        provider = AnthropicProvider(
            api_key="test-key", config={"max_retries": 0, "speed": "fast"}
        )
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_speed is False
        headers = provider._build_request_beta_headers(
            model_id="claude-opus-4-7-20260416",
            request_caps=caps,
            tools_present=False,
            resolved_thinking_type=None,
            fast_mode=False,
        )
        assert BETA_HEADER_FAST_MODE not in headers
