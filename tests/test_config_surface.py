"""Tests for the config-surface overhaul: numeric coercion, unknown-key
sweep, targeted inert-key messages, and the ``extra_request_params``
escape hatch.

Companion to ``test_config_bool_parsing.py`` (boolean keys) -- this file
covers the numeric keys (D-05/D-06), the unknown-key sweep (D-01..D-04),
and the ``extra_request_params`` merge (D-07).

Spec: anthropic-surface-spec.md, Workstream D (\u00a75), test plan \u00a78.1
(test_config_surface.py, T-D01..T-D15).
"""

import logging

import pytest

from amplifier_module_provider_anthropic import _CONSUMED_CONFIG_KEYS
from amplifier_module_provider_anthropic import AnthropicProvider

# ---------------------------------------------------------------------------
# The owner's 5 live provider-anthropic settings.yaml instances, verbatim
# (anthropic-surface-spec.md §7). Shared by the unknown-key-sweep tests here
# and by the full migration acceptance test added once the fallback-ladder
# key removals land (see test_fallback_ladder.py /
# test_migration_owner_instances.py).
# ---------------------------------------------------------------------------
OWNER_LIVE_CONFIGS: dict[str, dict] = {
    "opus-4.8": {
        "default_model": "claude-opus-4-8",
        "fallback_on_overload": "true",
        "fallback_retry_count": "2",
        "fallback_cooldown_seconds": "300",
        "fallback_sonnet_model": "claude-sonnet-4-6",
        "fallback_haiku_model": "claude-haiku-4-5",
        "persist_fallback_state": "true",
        "enable_1m_context": "true",
        "reasoning_effort": "xhigh",
        "enable_prompt_caching": "true",
    },
    "opus": {
        "default_model": "claude-opus-5",
        "fallback_on_overload": "true",
        "fallback_retry_count": "2",
        "fallback_cooldown_seconds": "300",
        "fallback_sonnet_model": "claude-sonnet-4-6",
        "fallback_haiku_model": "claude-haiku-4-5",
        "persist_fallback_state": "true",
        "enable_1m_context": "true",
        "reasoning_effort": "xhigh",
        "enable_prompt_caching": "true",
    },
    "sonnet": {
        "default_model": "claude-sonnet-5",
        "fallback_on_overload": "true",
        "fallback_retry_count": "2",
        "fallback_cooldown_seconds": "300",
        "fallback_haiku_model": "claude-haiku-4-5",
        "persist_fallback_state": "true",
        "enable_1m_context": "true",
        "reasoning_effort": "high",
        "enable_prompt_caching": "true",
    },
    "haiku": {
        "default_model": "claude-haiku-4-5-20251001",
        "reasoning_effort": "high",
        "enable_prompt_caching": "true",
    },
    "fable": {
        "default_model": "claude-fable-5",
        "fallback_on_overload": "true",
        "fallback_retry_count": "3",
        "fallback_cooldown_seconds": "300",
        "fallback_sonnet_model": "claude-sonnet-4-6",
        "fallback_haiku_model": "claude-haiku-4-5",
        "persist_fallback_state": "false",
        "enable_1m_context": "true",
        "reasoning_effort": "xhigh",
        "enable_prompt_caching": "true",
    },
}

# ---------------------------------------------------------------------------
# D-05 / D-06 -- numeric coercion via _config_int / _config_float
# ---------------------------------------------------------------------------

# (config key, attribute name, python type, documented default)
# max_tokens' default depends on the resolved model's capability ceiling,
# so it is exercised separately below rather than folded into this table.
NUMERIC_KEYS = [
    ("priority", "priority", int, 100),
    ("timeout", "timeout", float, 600.0),
    ("overloaded_delay_multiplier", "_overloaded_delay_multiplier", float, 10.0),
    ("throttle_threshold", "_throttle_threshold", float, 0.02),
    ("throttle_delay", "_throttle_delay", float, 1.0),
    ("max_concurrent_requests", "_max_concurrent_requests", int, 5),
    ("temperature", "temperature", float, 0.7),
]


def _make_provider(config: dict) -> AnthropicProvider:
    return AnthropicProvider(api_key="test-key", config=config)


@pytest.mark.parametrize("key,attr,pytype,default", NUMERIC_KEYS)
def test_invalid_numeric_string_warns_and_defaults(
    key: str, attr: str, pytype: type, default, caplog
):
    """T-D08: a typo'd numeric string (e.g. 'lots', 'ten', '2%') must warn
    and use the documented default -- never raise ValueError at mount.

    Before this fix, 4 of these 7 keys (overloaded_delay_multiplier,
    throttle_threshold, throttle_delay, max_concurrent_requests) raised
    ValueError on construction, killing the whole provider instance.
    """
    with caplog.at_level(logging.WARNING):
        provider = _make_provider({key: "not-a-number"})
    value = getattr(provider, attr)
    assert value == default
    assert isinstance(value, pytype)
    assert any("invalid" in record.message.lower() for record in caplog.records), (
        f"expected a warning log for invalid {key!r}, got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.parametrize(
    "key,attr,pytype,str_value,expected",
    [
        ("priority", "priority", int, "1", 1),
        ("timeout", "timeout", float, "600", 600.0),
        (
            "overloaded_delay_multiplier",
            "_overloaded_delay_multiplier",
            float,
            "10",
            10.0,
        ),
        ("throttle_threshold", "_throttle_threshold", float, "0.02", 0.02),
        ("throttle_delay", "_throttle_delay", float, "1.0", 1.0),
        ("max_concurrent_requests", "_max_concurrent_requests", int, "5", 5),
        ("temperature", "temperature", float, "0.7", 0.7),
    ],
)
def test_numeric_string_coerces_to_real_type(
    key: str, attr: str, pytype: type, str_value: str, expected
):
    """T-D09: a settings.yaml numeric string ('8192', '1', '600', ...) must
    coerce to a real int/float, not silently stay a str all the way to the
    wire.
    """
    provider = _make_provider({key: str_value})
    value = getattr(provider, attr)
    assert value == expected
    assert isinstance(value, pytype)


def test_max_tokens_invalid_string_warns_and_uses_model_ceiling(caplog):
    """max_tokens is keyed to the resolved model's capability ceiling, not
    a fixed literal -- verified separately from the table above."""
    with caplog.at_level(logging.WARNING):
        provider = _make_provider({"max_tokens": "lots"})
    assert provider.max_tokens == provider._default_caps.max_output_tokens
    assert isinstance(provider.max_tokens, int)
    assert any("invalid" in r.message.lower() for r in caplog.records)


def test_max_tokens_numeric_string_coerces_to_int():
    provider = _make_provider({"max_tokens": "8192"})
    assert provider.max_tokens == 8192
    assert isinstance(provider.max_tokens, int)


def test_numeric_real_type_passthrough():
    """Real int/float config values (already-correct config) pass through
    unchanged -- the coercion helpers are a safety net, not a requirement."""
    provider = _make_provider(
        {
            "priority": 7,
            "timeout": 45.0,
            "temperature": 0.3,
        }
    )
    assert provider.priority == 7
    assert provider.timeout == 45.0
    assert provider.temperature == 0.3


# ---------------------------------------------------------------------------
# D-01..D-04 -- unknown-key sweep, targeted inert messages, did-you-mean
# ---------------------------------------------------------------------------

# The eight keys read only in the deferred request path (_build_params /
# _build_web_search_tool), never at construction. An allowlist derived only
# from __init__ would false-positive on every one of these.
_DEFERRED_KEYS = {
    "thinking_budget_tokens",
    "thinking_budget_buffer",
    "thinking_type",
    "thinking_display",
    "task_budget_tokens",
    "speed",
    "web_search_max_uses",
    "web_search_user_location",
}


def test_consumed_config_keys_contains_all_deferred_keys():
    """T-D01: _CONSUMED_CONFIG_KEYS must include every key read outside the
    constructor, or every extended-thinking / web-search user gets a false
    unknown-key warning."""
    missing = _DEFERRED_KEYS - _CONSUMED_CONFIG_KEYS
    assert not missing, f"deferred keys missing from _CONSUMED_CONFIG_KEYS: {missing}"


@pytest.mark.parametrize("name,config", list(OWNER_LIVE_CONFIGS.items()))
def test_no_false_positive_unknown_key_warning_on_owner_live_configs(
    name: str, config: dict, caplog
):
    """T-D02 (unknown-key-sweep half): every key set by the owner's 5 real
    settings.yaml instances (anthropic-surface-spec.md §7) is a recognized
    key -- mounting must never emit a generic 'Unknown config key' warning
    for any of them. (Targeted inert-key warnings, e.g. for the
    fallback_sonnet_model/fallback_haiku_model keys once they are retired by
    the fallback-ladder change, are a SEPARATE, expected signal -- see the
    full migration acceptance test added alongside that change.)
    """
    with caplog.at_level(logging.WARNING):
        _make_provider(config)
    unknown_key_warnings = [
        r.message for r in caplog.records if "Unknown config key" in r.message
    ]
    assert not unknown_key_warnings, (
        f"instance {name!r} produced unexpected unknown-key warning(s): "
        f"{unknown_key_warnings}"
    )


def test_unknown_key_warns_with_did_you_mean(caplog):
    """T-D03: a typo'd key gets a 'did you mean' suggestion drawn from the
    known-key set."""
    with caplog.at_level(logging.WARNING):
        _make_provider({"temperture": 1})  # typo of "temperature"
    matches = [r.message for r in caplog.records if "Unknown config key" in r.message]
    assert len(matches) == 1
    assert "did you mean" in matches[0].lower()
    assert "temperature" in matches[0]


def test_single_inert_key_emits_exactly_one_targeted_warning(caplog):
    """T-D04 (adapted): a single recognized-inert key ('debug', a ghost key
    documented in the README but never implemented) emits exactly one
    warning, and it is the TARGETED inert-key message, not the generic
    unknown-key sweep warning."""
    with caplog.at_level(logging.WARNING):
        _make_provider({"debug": True})
    debug_warnings = [r.message for r in caplog.records if "'debug'" in r.message]
    assert len(debug_warnings) == 1
    assert "not consumed" in debug_warnings[0]
    assert not any("Unknown config key" in r.message for r in caplog.records)


def test_debug_and_raw_debug_each_emit_a_targeted_warning(caplog):
    """T-D05: both README-ghost keys warn independently."""
    with caplog.at_level(logging.WARNING):
        _make_provider({"debug": True, "raw_debug": True})
    assert any("'debug'" in r.message for r in caplog.records)
    assert any("'raw_debug'" in r.message for r in caplog.records)


def test_effort_alias_alone_warns_deprecated_not_both_set(caplog):
    """T-D06 (part 1): 'effort' alone -> a deprecation warning, never the
    both-set warning."""
    with caplog.at_level(logging.WARNING):
        _make_provider({"effort": "high"})
    messages = [r.message for r in caplog.records if "effort" in r.message.lower()]
    assert any("DEPRECATED" in m for m in messages)
    assert not any("Both 'reasoning_effort'" in m for m in messages)


def test_effort_and_reasoning_effort_both_set_warns_both_set_not_deprecated(caplog):
    """T-D06 (part 2): both keys set -> the both-set warning, never the
    bare-deprecation warning (never both)."""
    with caplog.at_level(logging.WARNING):
        _make_provider({"effort": "high", "reasoning_effort": "low"})
    messages = [r.message for r in caplog.records if "effort" in r.message.lower()]
    both_set = [m for m in messages if "Both 'reasoning_effort'" in m]
    bare_deprecated = [m for m in messages if "DEPRECATED" in m and "Both" not in m]
    assert len(both_set) == 1
    assert not bare_deprecated


def test_extra_known_config_keys_subclass_extension_point(caplog):
    """T-D07: a subclass's EXTRA_KNOWN_CONFIG_KEYS suppresses the
    unknown-key warning for its own key."""

    class _SubclassProvider(AnthropicProvider):
        EXTRA_KNOWN_CONFIG_KEYS = frozenset({"my_custom_key"})

    with caplog.at_level(logging.WARNING):
        _SubclassProvider(api_key="test-key", config={"my_custom_key": "value"})
    assert not any("Unknown config key" in r.message for r in caplog.records)
