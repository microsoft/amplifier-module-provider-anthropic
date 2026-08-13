"""Behavioral tests for anthropic provider.

Inherits authoritative tests from amplifier-core.
"""

import os

import pytest
from amplifier_core.validation.behavioral import ProviderBehaviorTests

# ProviderBehaviorTests' shared `provider_module` fixture (amplifier-core's
# pytest plugin) calls the module's real mount() with no config. mount()
# treats a missing API key as "not configured" and returns None by design
# (see amplifier_module_provider_anthropic/__init__.py's mount()), so the
# fixture setup itself fails ("No provider was mounted") before any test body
# runs -- surfacing as 5 fixture-setup ERRORs with no stated cause. One of
# the inherited tests (test_list_models_returns_list) also makes a REAL call
# to the Anthropic models API, so a fake/dummy key would trade one confusing
# failure (fixture error) for another (a live 401 from Anthropic) instead of
# fixing anything. On every CI runner today (none of them carry the
# ANTHROPIC_API_KEY secret), skip explicitly and say why.
requires_anthropic_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason=(
        "ANTHROPIC_API_KEY not set - mount() requires a real key to "
        "construct the provider, and list_models() calls the real "
        "Anthropic API"
    ),
)


@requires_anthropic_api_key
class TestAnthropicProviderBehavior(ProviderBehaviorTests):
    """Run standard provider behavioral tests for anthropic.

    All tests from ProviderBehaviorTests run automatically.
    Add module-specific tests below if needed.
    """
