"""Merge-gate acceptance test: the owner's 5 real provider-anthropic
settings.yaml instances (anthropic-surface-spec.md §7), mounted with
post-overhaul code, must produce ZERO false-positive warnings -- only the
EXPECTED targeted inert-key messages for the keys this release retires
(fallback_sonnet_model, fallback_haiku_model, and -- once the ladder
commit for refusal fallback lands -- refusal_fallback_model where set).

This is the acceptance bar for §7.1 of the spec ("No instance produces a
false-positive unknown-key warning") and is re-asserted, not merely
implied, because it is the release's actual merge gate for the migration
story.
"""

import logging

import pytest

from amplifier_module_provider_anthropic import AnthropicProvider
from tests.test_config_surface import OWNER_LIVE_CONFIGS

# For each owner instance: the set of config keys that are EXPECTED to
# produce a targeted inert-key warning once mounted with post-overhaul
# code, keyed by the substring that must appear in the warning message.
# Everything else in that instance's config must be silent.
EXPECTED_INERT_KEY_WARNINGS: dict[str, set[str]] = {
    "opus-4.8": {"fallback_sonnet_model", "fallback_haiku_model"},
    "opus": {"fallback_sonnet_model", "fallback_haiku_model"},
    "sonnet": {
        "fallback_haiku_model"
    },  # sonnet instance never set fallback_sonnet_model
    "haiku": set(),
    "fable": {"fallback_sonnet_model", "fallback_haiku_model"},
}


@pytest.mark.parametrize("name,config", list(OWNER_LIVE_CONFIGS.items()))
def test_owner_instance_produces_only_expected_targeted_warnings(
    name: str, config: dict, caplog
):
    with caplog.at_level(logging.WARNING):
        AnthropicProvider(api_key="test-key", config=config)

    # 1. Never a generic "Unknown config key" warning -- every key any of
    #    the 5 instances sets is a recognized key (consumed, inert, or a
    #    deprecated alias).
    unknown_key_warnings = [
        r.message for r in caplog.records if "Unknown config key" in r.message
    ]
    assert not unknown_key_warnings, (
        f"instance {name!r} produced unexpected unknown-key warning(s): "
        f"{unknown_key_warnings}"
    )

    # 2. Exactly the expected targeted inert-key warnings fire -- no more,
    #    no fewer.
    expected = EXPECTED_INERT_KEY_WARNINGS[name]
    inert_key_warnings = {
        key
        for key in (
            "fallback_sonnet_model",
            "fallback_haiku_model",
            "refusal_fallback_model",
        )
        if any(f"'{key}'" in r.message for r in caplog.records)
    }
    assert inert_key_warnings == expected, (
        f"instance {name!r}: expected targeted inert-key warnings for "
        f"{expected}, got {inert_key_warnings}"
    )
