"""Models API overlay narrows supports_manual_thinking / supported_efforts,
but never widens them (D10 / T7)."""

from __future__ import annotations

from amplifier_module_provider_anthropic import AnthropicProvider, _RuntimeModelInfo


def _extract(model_info: dict):
    return AnthropicProvider._extract_runtime_model_info(model_info)


class TestExtraction:
    def test_effort_supported_true_levels_collected(self):
        info = _extract(
            {
                "capabilities": {
                    "effort": {
                        "low": {"supported": True},
                        "medium": {"supported": True},
                        "high": {"supported": True},
                        "xhigh": {"supported": False},
                        "max": {"supported": False},
                    }
                }
            }
        )
        assert info.supported_efforts == ("low", "medium", "high")

    def test_no_effort_key_is_none(self):
        info = _extract({"capabilities": {}})
        assert info.supported_efforts is None

    def test_manual_thinking_enabled_supported_extracted(self):
        info = _extract(
            {"capabilities": {"thinking": {"types": {"enabled": {"supported": False}}}}}
        )
        assert info.supports_manual_thinking is False


class TestOverlayNarrowing:
    def test_runtime_narrows_supported_efforts(self):
        base = AnthropicProvider._get_capabilities("claude-opus-5-5")
        assert "xhigh" in base.supported_efforts
        runtime = _RuntimeModelInfo(supported_efforts=("low", "medium"))
        overridden = AnthropicProvider._apply_runtime_capability_overrides(
            base, runtime
        )
        assert overridden.supported_efforts == ("low", "medium")
        assert "xhigh" not in overridden.supported_efforts

    def test_runtime_true_never_widens_manual_thinking(self):
        """Anthropic's own sample Models API response reports
        enabled.supported=true for claude-opus-5, which contradicts this
        provider's live-verified static value -- so a runtime true must not
        flip a statically-False capability on."""
        base = AnthropicProvider._get_capabilities("claude-opus-5")
        assert base.supports_manual_thinking is False
        runtime = _RuntimeModelInfo(supports_manual_thinking=True)
        overridden = AnthropicProvider._apply_runtime_capability_overrides(
            base, runtime
        )
        assert overridden.supports_manual_thinking is False

    def test_runtime_false_narrows_manual_thinking_off(self):
        base = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert base.supports_manual_thinking is True
        runtime = _RuntimeModelInfo(supports_manual_thinking=False)
        overridden = AnthropicProvider._apply_runtime_capability_overrides(
            base, runtime
        )
        assert overridden.supports_manual_thinking is False

    def test_missing_capabilities_leaves_static_caps_unchanged(self):
        base = AnthropicProvider._get_capabilities("claude-opus-5-5")
        runtime = _RuntimeModelInfo()  # nothing extracted from the response
        overridden = AnthropicProvider._apply_runtime_capability_overrides(
            base, runtime
        )
        assert overridden.supported_efforts == base.supported_efforts
        assert overridden.supports_manual_thinking == base.supports_manual_thinking
