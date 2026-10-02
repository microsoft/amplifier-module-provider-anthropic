"""Tests for list_models() family grouping.

Regression coverage for the model-discovery path used to build the
provider's model menu. list_models() must classify every family that
_detect_family() knows about — not a hardcoded subset — so that newer
families (e.g. fable) are surfaced instead of silently dropped.
"""

import asyncio
import pytest
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator

from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import FakeCoordinator


def _make_provider(filtered: bool = True) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": 0,
            "filtered": filtered,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _model(model_id: str, display_name: str, created_at: str) -> SimpleNamespace:
    """Minimal Anthropic Models API entry stub."""
    return SimpleNamespace(
        id=model_id,
        display_name=display_name,
        created_at=created_at,
    )


def _stub_models_list(
    provider: AnthropicProvider, models: list[SimpleNamespace]
) -> None:
    provider.client.models.list = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(data=models)
    )


# One representative model per family, including fable.
_API_MODELS = [
    _model("claude-fable-5", "Claude Fable 5", "2026-01-01"),
    _model("claude-opus-4-6-20260101", "Claude Opus 4.6", "2026-01-01"),
    _model("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", "2025-09-29"),
    _model("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "2025-10-01"),
]


def test_list_models_includes_fable_family():
    """fable must appear in the discovered model list (was silently dropped)."""
    provider = _make_provider(filtered=True)
    _stub_models_list(provider, _API_MODELS)

    result = asyncio.run(provider.list_models())
    ids = {m.id for m in result}

    assert "claude-fable-5" in ids


def test_filtered_menu_and_fallback_cache_use_numeric_newest():
    provider = _make_provider()
    _stub_models_list(provider, [
        _model("claude-sonnet-5-9", "Sonnet 5.9", ""),
        _model("claude-sonnet-5-10-20261002", "Snapshot 5.10", ""),
        _model("claude-sonnet-5-10", "Sonnet 5.10", ""),
    ])
    models = asyncio.run(provider.list_models())
    assert [model.id for model in models] == ["claude-sonnet-5-10"]
    assert provider._family_latest["sonnet"] == "claude-sonnet-5-10"


@pytest.mark.parametrize("filtered", [False, True])
def test_major_successor_wins_in_menu_and_cache_with_snapshots(filtered):
    provider = _make_provider(filtered=filtered)
    ids = ["claude-opus-5-10", "claude-opus-6-20261002", "claude-opus-6"]
    _stub_models_list(provider, [_model(model_id, model_id, "") for model_id in ids])
    models = asyncio.run(provider.list_models())
    assert provider._family_latest["opus"] == "claude-opus-6"
    if filtered:
        assert [model.id for model in models] == ["claude-opus-6"]
    else:
        assert {model.id for model in models} == set(ids)


def test_list_models_returns_all_known_families():
    """No family recognized by _detect_family() should be dropped."""
    provider = _make_provider(filtered=True)
    _stub_models_list(provider, _API_MODELS)

    result = asyncio.run(provider.list_models())
    families = {AnthropicProvider._detect_family(m.id) for m in result}

    assert {"fable", "opus", "sonnet", "haiku"} <= families
