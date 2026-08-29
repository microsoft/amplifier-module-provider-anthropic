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

from amplifier_module_provider_anthropic import AnthropicProvider

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
