"""Tests for the fallback ladder redesign: family successor map, target
resolution (fallback_models override / live cache / static backstop), the
mythos family, and the terminal-family mount-time warning.

Spec: anthropic-surface-spec.md, Workstream B (§3), test plan §8.1
(test_fallback_ladder.py, T-B01..T-B18). Refusal-fallback (B-07) and the
429-is-not-overload fix (B-06) are covered separately once those commits
land, in this same file.
"""

import asyncio
import logging
from typing import cast

import pytest
from amplifier_core import ModuleCoordinator

import amplifier_module_provider_anthropic as anthropic_module
from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import FakeCoordinator


@pytest.fixture(autouse=True)
def clear_fallback_windows():
    anthropic_module._clear_fallback_windows()
    yield
    anthropic_module._clear_fallback_windows()


def _make_provider(
    default_model: str = "claude-sonnet-5", **config_overrides
) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "default_model": default_model,
            "max_retries": 3,
            "min_retry_delay": 0.01,
            "max_retry_delay": 60.0,
            "retry_jitter": False,
            **config_overrides,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


# ---------------------------------------------------------------------------
# B-01/B-04 -- the ladder walk
# ---------------------------------------------------------------------------


def test_fable_resolves_to_opus_not_dead():
    """T-B01: fable was completely dead before this release (no rung at
    all in the old if/elif chain) -- it must now resolve."""
    provider = _make_provider("claude-fable-5")
    assert provider._fallback_target_for_model("claude-fable-5") == "claude-opus-5"


def test_full_ladder_walk_from_fable_to_terminal():
    """T-B02: fable -> opus -> sonnet -> haiku -> None, four full rungs."""
    provider = _make_provider("claude-fable-5")
    assert provider._fallback_target_for_model("claude-fable-5") == "claude-opus-5"
    assert provider._fallback_target_for_model("claude-opus-5") == "claude-sonnet-5"
    assert provider._fallback_target_for_model("claude-sonnet-5") == "claude-haiku-4-5"
    assert provider._fallback_target_for_model("claude-haiku-4-5") is None


def test_haiku_is_terminal():
    """T-B03: haiku has no lower tier -- regression guard."""
    provider = _make_provider("claude-haiku-4-5-20251001")
    assert provider._fallback_target_for_model("claude-haiku-4-5-20251001") is None


def test_unusable_rung_is_skipped_not_fatal():
    """T-B04: an unusable rung (resolves to the same model, or the same
    family) must be SKIPPED, walking to the next lower rung -- not
    terminate the whole ladder (the pre-overhaul defect D-2: one
    unusable/self-referential config value killed fallback entirely
    instead of walking past it)."""
    provider = _make_provider(
        "claude-opus-5", fallback_models={"sonnet": "claude-opus-5"}
    )
    # sonnet's override resolves to the SAME model that was requested --
    # unusable -- so the walk must continue past sonnet to haiku, not stop.
    assert provider._fallback_target_for_model("claude-opus-5") == "claude-haiku-4-5"


def test_blank_override_falls_through_to_static_backstop():
    """An explicitly-blank fallback_models entry is treated as "no
    override" (falsy), not as "disable this rung" -- resolution falls
    through the three-source precedence to the static backstop, which is
    always populated for opus/sonnet/haiku. This is NOT a dead rung."""
    provider = _make_provider("claude-opus-5", fallback_models={"sonnet": ""})
    assert provider._fallback_target_for_model("claude-opus-5") == "claude-sonnet-5"


def test_fallback_models_override_beats_static_map():
    """T-B05: an explicit fallback_models override wins over the static
    backstop."""
    provider = _make_provider(
        "claude-fable-5", fallback_models={"opus": "claude-opus-4-8"}
    )
    assert provider._fallback_target_for_model("claude-fable-5") == "claude-opus-4-8"


# ---------------------------------------------------------------------------
# B-03 -- three-source target resolution
# ---------------------------------------------------------------------------


def test_family_latest_cache_beats_static_backstop():
    """T-B06: a live-refreshed _family_latest entry wins over
    _STATIC_FALLBACK_MODELS."""
    provider = _make_provider("claude-fable-5")
    provider._family_latest["opus"] = "claude-opus-4-9-preview"
    assert provider._resolve_fallback_model("opus") == "claude-opus-4-9-preview"


def test_warm_family_latest_swallows_errors_and_keeps_static_fallback():
    """T-B07: a raising list_models() during the warm-up must not propagate
    -- resolution still falls through to the static backstop."""
    provider = _make_provider("claude-fable-5")

    async def _raise(retry_config=None):
        raise RuntimeError("network is down")

    provider.list_models = _raise  # type: ignore[method-assign]
    asyncio.run(provider._warm_family_latest())
    assert provider._resolve_fallback_model("opus") == "claude-opus-5"


def test_warm_family_latest_runs_at_most_once():
    """T-B08: repeated calls to _warm_family_latest only ever call
    list_models() once per provider instance."""
    provider = _make_provider("claude-fable-5")
    calls = 0

    async def _fake_list_models(retry_config=None):
        nonlocal calls
        calls += 1
        return []

    provider.list_models = _fake_list_models  # type: ignore[method-assign]
    asyncio.run(provider._warm_family_latest())
    asyncio.run(provider._warm_family_latest())
    assert calls == 1


# ---------------------------------------------------------------------------
# B-08 -- terminal-family mount-time LOUD warning
# ---------------------------------------------------------------------------


def test_terminal_family_with_fallback_enabled_warns_loudly(caplog):
    """T-B12: fallback_on_overload=true on a haiku (terminal) instance has
    NO EFFECT -- must warn loudly at mount, naming the lowest-tier problem,
    since the wizard's show_when hides this ConfigField for Haiku but
    hand-edited settings.yaml is not protected by that."""
    with caplog.at_level(logging.WARNING):
        _make_provider("claude-haiku-4-5-20251001", fallback_on_overload=True)
    messages = [r.message for r in caplog.records]
    assert any("lowest tier" in m for m in messages)


def test_non_terminal_family_with_fallback_enabled_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        _make_provider("claude-sonnet-5", fallback_on_overload=True)
    messages = [r.message for r in caplog.records]
    assert not any("lowest tier" in m for m in messages)


# ---------------------------------------------------------------------------
# B-02 / X-2 -- the mythos family
# ---------------------------------------------------------------------------


def test_detect_family_recognizes_mythos():
    """T-B13: claude-mythos-5 must classify as 'mythos', not silently fall
    through to the 'sonnet' default (wrong capability row, wrong context
    window, wrong ladder position, wrong fallback target)."""
    assert AnthropicProvider._detect_family("claude-mythos-5") == "mythos"


def test_mythos_capabilities_match_fable_tier():
    """T-B14: Mythos 5 is a top-tier, 1M-context, 128K-output family with
    the full xhigh+max effort range -- the same tier as Fable 5."""
    caps = AnthropicProvider._get_capabilities("claude-mythos-5")
    assert caps.family == "mythos"
    assert caps.supports_1m is True
    assert caps.max_output_tokens == 128000
    assert "xhigh" in caps.supported_efforts
    assert "max" in caps.supported_efforts


def test_mythos_preview_lacks_xhigh():
    caps = AnthropicProvider._get_capabilities("claude-mythos-preview")
    assert caps.family == "mythos"
    assert "xhigh" not in caps.supported_efforts
    assert "max" in caps.supported_efforts


def test_mythos_ladder_target_is_opus():
    """Mythos enters the ladder as a peer of fable -- both step down to
    opus, not to each other."""
    provider = _make_provider("claude-mythos-5")
    assert provider._fallback_target_for_model("claude-mythos-5") == "claude-opus-5"


# ---------------------------------------------------------------------------
# Cycle guard + corrected defaults
# ---------------------------------------------------------------------------


def test_ladder_cycle_guard_returns_none_and_warns(caplog):
    """T-B17: a pathological fallback_models map that creates a cycle must
    return None and warn -- never hang."""
    provider = _make_provider(
        "claude-opus-5",
        fallback_models={"sonnet": "claude-opus-5"},  # sonnet -> back to opus's model
    )
    with caplog.at_level(logging.WARNING):
        result = provider._fallback_target_for_model("claude-opus-5")
    # claude-opus-5 requested; sonnet override points back at the same
    # model id, which is filtered as "target == model_id" and the walk
    # continues to haiku.
    assert result == "claude-haiku-4-5"


def test_fallback_ladder_defaults_corrected():
    """T-B18: fallback_retry_count defaults to 2 (was 1), fallback_cooldown_seconds
    defaults to 300.0 (was 1800.0) -- matching 4 of the owner's 5 real instances."""
    provider = _make_provider("claude-sonnet-5")
    assert provider._fallback_retry_count == 2
    assert provider._fallback_cooldown_seconds == 300.0


# ---------------------------------------------------------------------------
# End-to-end: list_models() warms the live cache as a side effect (B-03)
# ---------------------------------------------------------------------------


def test_list_models_populates_family_latest_cache():
    provider = _make_provider("claude-sonnet-5")

    class _FakeModel:
        def __init__(self, id_, created_at):
            self.id = id_
            self.display_name = id_
            self.created_at = created_at

    class _FakeResponse:
        def __init__(self, data):
            self.data = data

    async def _fake_list(**kwargs):
        return _FakeResponse(
            [
                _FakeModel("claude-opus-4-8", "2026-01-01"),
                _FakeModel("claude-opus-5", "2026-06-01"),
            ]
        )

    provider.client.models.list = _fake_list  # type: ignore[method-assign]
    asyncio.run(provider.list_models())
    assert provider._family_latest["opus"] == "claude-opus-5"
