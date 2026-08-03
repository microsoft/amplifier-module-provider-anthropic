"""Tests for computer-use capability detection and model-aware wire-type selection.

Validates that `_get_capabilities` returns the correct `supports_native_computer_use`
flag and `computer_use_tool_type` for each model family/version, mirroring the
structure of `TestGetCapabilitiesOpus` / `TestGetCapabilitiesSonnet5` in
test_model_capabilities.py.

Every value asserted here is backed by a live probe against api.anthropic.com
(2026-08-03) — see the inline comments in `_get_capabilities` (opus/sonnet/haiku
branches) for the full evidence table. Each test below cites the specific live
result it encodes.
"""

from amplifier_module_provider_anthropic import (
    NATIVE_TOOL_BETA_HEADERS,
    AnthropicProvider,
    ModelCapabilities,
    _RuntimeModelInfo,
)


class TestModelCapabilitiesDefaults:
    """Bare dataclass defaults for the two new fields."""

    def test_default_supports_native_computer_use_false(self):
        caps = ModelCapabilities(family="test")
        assert caps.supports_native_computer_use is False

    def test_default_computer_use_tool_type_none(self):
        caps = ModelCapabilities(family="test")
        assert caps.computer_use_tool_type is None


class TestComputerUseOpus:
    """Opus family — live-probed 2026-08-03 against api.anthropic.com.

    Confirmed: claude-opus-4-1-20250805 + computer_20250124 -> 200;
    claude-opus-4-5-20251101 + computer_20250124 -> 200 (+ computer_20251124 -> 200,
    a dual-support transitional model — 20250124 is returned as the canonical
    answer since it also covers 4.1-4.4);
    claude-opus-4-6/4-7/4-8, claude-opus-5 + computer_20251124 -> 200
    (computer_20250124 -> 400 on all of these).
    """

    def test_opus_41_gets_older_type(self):
        """claude-opus-4-1-20250805: live 200 with computer_20250124."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-1-20250805")
        assert caps.computer_use_tool_type == "computer_20250124"
        assert caps.supports_native_computer_use is True

    def test_opus_45_dual_support_returns_older_type(self):
        """claude-opus-4-5-20251101: live 200 on BOTH computer_20250124 and
        computer_20251124. The older type is returned as canonical since it is
        the one confirmed across the whole 4.1-4.5 range."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.computer_use_tool_type == "computer_20250124"
        assert caps.supports_native_computer_use is True

    def test_opus_46_gets_newer_type(self):
        """claude-opus-4-6: live 200 with computer_20251124; 20250124 -> 400."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.computer_use_tool_type == "computer_20251124"

    def test_opus_47_gets_newer_type(self):
        """claude-opus-4-7: live 200 with computer_20251124; 20250124 -> 400."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.computer_use_tool_type == "computer_20251124"

    def test_opus_48_gets_newer_type(self):
        """claude-opus-4-8: live 200 with computer_20251124; 20250124 -> 400."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        assert caps.computer_use_tool_type == "computer_20251124"

    def test_opus_5_gets_newer_type(self):
        """claude-opus-5: live 200 with computer_20251124; 20250124/20241022 -> 400."""
        caps = AnthropicProvider._get_capabilities("claude-opus-5")
        assert caps.computer_use_tool_type == "computer_20251124"
        assert caps.supports_native_computer_use is True

    def test_opus_below_41_unsupported(self):
        """Below 4.1 is a NEGATIVE case: the only pre-4.1 opus model
        (claude-opus-4-20250514) returned HTTP 404 (retired) when probed live,
        so tool support could not be confirmed either way. Conservative default:
        unsupported."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-20250514")
        assert caps.computer_use_tool_type is None
        assert caps.supports_native_computer_use is False

    def test_opus_unknown_version_assumes_latest(self):
        """Unknown opus version assumes latest (4.8+ gate), same forward-compat
        convention as the rest of _get_capabilities."""
        caps = AnthropicProvider._get_capabilities("claude-opus-latest")
        assert caps.computer_use_tool_type == "computer_20251124"


class TestComputerUseSonnet:
    """Sonnet family — live-probed 2026-08-03.

    Confirmed: claude-sonnet-4-5-20250929 + computer_20250124 -> 200
    (computer_20251124/20241022 -> 400);
    claude-sonnet-4-6, claude-sonnet-5 + computer_20251124 -> 200
    (computer_20250124 -> 400 on both).
    """

    def test_sonnet_45_gets_older_type(self):
        """claude-sonnet-4-5-20250929: live 200 with computer_20250124 only."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-5-20250929")
        assert caps.computer_use_tool_type == "computer_20250124"
        assert caps.supports_native_computer_use is True

    def test_sonnet_46_gets_newer_type(self):
        """claude-sonnet-4-6: live 200 with computer_20251124; 20250124 -> 400."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-6")
        assert caps.computer_use_tool_type == "computer_20251124"

    def test_sonnet_5_gets_newer_type(self):
        """claude-sonnet-5: live 200 with computer_20251124; 20250124 -> 400."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-5")
        assert caps.computer_use_tool_type == "computer_20251124"
        assert caps.supports_native_computer_use is True

    def test_sonnet_below_45_unsupported(self):
        """NEGATIVE case: claude-sonnet-4-20250514 returned HTTP 404 (retired)
        when probed live — no sonnet model between 4.1 and 4.4 exists to test,
        so (unlike opus, confirmed live down to 4.1) the sonnet floor is set at
        the lowest version this evidence actually covers (4.5), not extrapolated
        down to match opus."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-20250514")
        assert caps.computer_use_tool_type is None
        assert caps.supports_native_computer_use is False

    def test_sonnet_unknown_version_assumes_latest(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-latest")
        assert caps.computer_use_tool_type == "computer_20251124"


class TestComputerUseHaiku:
    """Haiku family — live-probed 2026-08-03.

    Confirmed: claude-haiku-4-5-20251001 + computer_20250124 -> 200
    (computer_20251124/20241022 -> 400). No haiku model at 4.6+ exists live, so
    the opus/sonnet "newer generation at 4.6+" jump is NOT extrapolated for haiku.
    """

    def test_haiku_45_gets_older_type(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.computer_use_tool_type == "computer_20250124"
        assert caps.supports_native_computer_use is True

    def test_haiku_below_45_unsupported(self):
        """NEGATIVE case: claude-haiku-3-5-20250929 (pre-computer-use generation)
        returned HTTP 404 (retired) when probed live for a basic request — could
        not be confirmed either way; left at the conservative default."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert caps.computer_use_tool_type is None
        assert caps.supports_native_computer_use is False

    def test_haiku_unknown_version_assumes_latest_confirmed_type(self):
        """Unknown haiku version assumes latest -- but "latest confirmed" for
        haiku is still computer_20250124 (no live 4.6+ haiku exists to test
        computer_20251124), unlike opus/sonnet where "latest" is the newer type."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-latest")
        assert caps.computer_use_tool_type == "computer_20250124"


class TestComputerUseFable:
    """Fable family — NOT verified.

    claude-fable-5 rejected every request in this workspace with "organization
    or workspace must have data retention enabled" (HTTP 400, 2026-08-03) — the
    tool-support question itself could not be asked live. Left at the
    conservative dataclass default rather than assumed either way.
    """

    def test_fable_unsupported_by_default(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.computer_use_tool_type is None
        assert caps.supports_native_computer_use is False


class TestComputerUseUnknownFamily:
    def test_unrecognized_family_string_defaults_to_sonnet_and_assumes_latest(self):
        """_detect_family has no "unknown" outcome -- it defaults to "sonnet" for
        any unrecognized model id (see TestDetectFamily.test_unknown_defaults_to_sonnet
        in test_model_capabilities.py). An unrecognized model therefore takes the
        sonnet unknown-version path: version unparseable -> assume latest ->
        computer_20251124. This is the actual reachable behavior, not a fresh
        assumption -- it falls straight out of existing, already-tested
        family/version detection."""
        caps = AnthropicProvider._get_capabilities("claude-mystery-9-9")
        assert caps.family == "sonnet"
        assert caps.computer_use_tool_type == "computer_20251124"

    def test_unmatched_family_dataclass_fallback_is_unsupported(self):
        """The `ModelCapabilities(family=family)` fallback in _get_capabilities
        (for a family value that matches none of fable/opus/sonnet/haiku) is
        unreachable through the public model_id path today, since _detect_family
        always returns one of those four. Exercise the dataclass default
        directly -- unsupported, matching every other conservative default."""
        caps = ModelCapabilities(family="something-detect-family-would-never-return")
        assert caps.computer_use_tool_type is None
        assert caps.supports_native_computer_use is False


class TestComputerUseToolTypeConsistency:
    """The selected wire type must always be one this module can add a beta
    header for -- otherwise the capability answers a question the provider
    can't actually complete on the wire."""

    def test_every_asserted_tool_type_has_a_beta_header(self):
        models = [
            "claude-opus-4-1-20250805",
            "claude-opus-4-5-20251101",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-haiku-4-5-20251001",
        ]
        for model_id in models:
            caps = AnthropicProvider._get_capabilities(model_id)
            assert caps.computer_use_tool_type in NATIVE_TOOL_BETA_HEADERS, model_id


class TestComputerUseSurvivesRuntimeOverride:
    """Regression guard mirroring test_sonnet_5_caps_survive_runtime_override:
    a non-None _RuntimeModelInfo must not silently reset the computer-use
    fields to the dataclass default when ModelCapabilities is reconstructed."""

    def test_opus_5_computer_use_survives_override(self):
        base = AnthropicProvider._get_capabilities("claude-opus-5")
        overridden = AnthropicProvider._apply_runtime_capability_overrides(
            base, _RuntimeModelInfo()
        )
        assert overridden.supports_native_computer_use is True
        assert overridden.computer_use_tool_type == "computer_20251124"

    def test_sonnet_45_computer_use_survives_override(self):
        base = AnthropicProvider._get_capabilities("claude-sonnet-4-5-20250929")
        overridden = AnthropicProvider._apply_runtime_capability_overrides(
            base, _RuntimeModelInfo()
        )
        assert overridden.supports_native_computer_use is True
        assert overridden.computer_use_tool_type == "computer_20250124"

    def test_unsupported_model_stays_unsupported_after_override(self):
        base = AnthropicProvider._get_capabilities("claude-fable-5")
        overridden = AnthropicProvider._apply_runtime_capability_overrides(
            base, _RuntimeModelInfo()
        )
        assert overridden.supports_native_computer_use is False
        assert overridden.computer_use_tool_type is None
