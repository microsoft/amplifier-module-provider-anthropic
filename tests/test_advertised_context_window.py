"""Tests for advertised context-window gating (_advertised_context_window).

A beta-gated 1M context window (Sonnet 4.6/4.7, Opus 4.6/4.7) must only be
advertised when the operator asserts entitlement via ``context_1m_entitled``;
otherwise the advertised window stays at the 200K base so downstream clients
don't skip compaction and overflow the real cap. GA 1M windows (Opus 4.8+)
are always advertised.
"""

from typing import cast

from amplifier_core import ModuleCoordinator

from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import FakeCoordinator

_BASE = 200000
_ONE_M = 1000000


def _make_provider(default_model: str, **config_overrides) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "default_model": default_model,
            **config_overrides,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _advertised(model_id: str, **config_overrides) -> int:
    provider = _make_provider(model_id, **config_overrides)
    caps = AnthropicProvider._get_capabilities(model_id)
    return provider._advertised_context_window(model_id, caps)


class TestBetaGatedNotEntitled:
    """Default (no entitlement): beta-gated 1M is NOT advertised."""

    def test_sonnet_46_advertises_base(self):
        assert _advertised("claude-sonnet-4-6") == _BASE

    def test_opus_47_advertises_base(self):
        assert _advertised("claude-opus-4-7-20260416") == _BASE


class TestBetaGatedEntitled:
    """With context_1m_entitled=True: beta-gated 1M IS advertised."""

    def test_sonnet_46_advertises_1m(self):
        assert _advertised("claude-sonnet-4-6", context_1m_entitled=True) == _ONE_M

    def test_opus_47_advertises_1m(self):
        assert (
            _advertised("claude-opus-4-7-20260416", context_1m_entitled=True) == _ONE_M
        )


class TestGaOneMillionAlwaysAdvertised:
    """GA 1M (Opus 4.8+) is advertised regardless of the entitlement flag."""

    def test_opus_48_advertises_1m_without_flag(self):
        assert _advertised("claude-opus-4-8") == _ONE_M

    def test_opus_48_advertises_1m_with_flag(self):
        assert _advertised("claude-opus-4-8", context_1m_entitled=True) == _ONE_M


class TestNoOneMillionSupport:
    """Models without 1M support always advertise the base window."""

    def test_haiku_advertises_base(self):
        assert _advertised("claude-haiku-4-5", context_1m_entitled=True) == _BASE

    def test_sonnet_45_advertises_base(self):
        # Sonnet 4.5 has supports_1m=False
        assert _advertised("claude-sonnet-4-5", context_1m_entitled=True) == _BASE


class TestEnable1mContextDisabled:
    """enable_1m_context=False forces the base window even when entitled."""

    def test_sonnet_46_disabled(self):
        assert (
            _advertised(
                "claude-sonnet-4-6",
                enable_1m_context=False,
                context_1m_entitled=True,
            )
            == _BASE
        )
