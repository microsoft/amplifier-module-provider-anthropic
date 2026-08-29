"""Tests for boolean config parsing on the keys that bypassed `_config_bool()`.

Follow-up to `microsoft/amplifier-module-provider-openai#74`. That PR fixed the
same anti-pattern in the openai provider and, as a read-only cross-check
(HEAD `9916a68`), found this module already has a correct helper --
`AnthropicProvider._config_bool()` (`__init__.py:577-584`) -- used by six
config keys, but FIVE boolean-ish keys were read straight off
`self.config.get(key, default)` with no coercion at all:

    raw, use_streaming, filtered, enable_prompt_caching, enable_web_search

The app-cli wizard writes boolean `ConfigField` answers as the STRING
``"true"``/``"false"`` (see the ``field_type="boolean"`` fields in
``get_info()``), and a plain ``self.config.get(key, default)`` with no
coercion returns that string unchanged. Every one of these keys is then used
in a truthiness context (``if self.filtered:``, ``if self.enable_web_search:``,
etc.), and any non-empty string -- including the literal string ``"false"``
-- is truthy in Python. A user answering "false" in the wizard therefore gets
the feature turned ON.

``enable_prompt_caching`` is the live-reachable instance: it IS exposed as a
``field_type="boolean"`` `ConfigField` with string default ``"true"``
(`__init__.py:975-982`), so a wizard-driven ``"false"`` answer silently
enables prompt caching -- and, post-#104, also feeds the
``cache_stable_region_ttl_1h`` beta-header gate
(``if self.enable_prompt_caching:`` at `__init__.py:813`), which then
misreads too.

Each test below is written so it FAILS on pre-fix `main` (string ``"false"``
resolves truthy) and PASSES once the key is routed through the existing
``_config_bool()`` helper -- verified via `git stash` (see the PR body for
the exact before/after run).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import DummyResponse

# ---------------------------------------------------------------------------
# Per-key parametrized coverage
# ---------------------------------------------------------------------------

# (config key, attribute name on the provider instance, default value)
AFFECTED_KEYS = [
    ("raw", "raw", False),
    ("use_streaming", "use_streaming", True),
    ("filtered", "filtered", True),
    ("enable_prompt_caching", "enable_prompt_caching", True),
    ("enable_web_search", "enable_web_search", False),
]


def _make_provider(config: dict) -> AnthropicProvider:
    return AnthropicProvider(api_key="test-key", config=config)


@pytest.mark.parametrize("key,attr,default", AFFECTED_KEYS)
def test_string_false_resolves_to_real_false(key: str, attr: str, default: bool):
    """The exact bug: string "false" must resolve to boolean False.

    Fails on pre-fix main because `bool("false")` / bare truthiness on the
    string `"false"` is `True`.
    """
    provider = _make_provider({key: "false"})
    assert getattr(provider, attr) is False, (
        f"{key}='false' (string) resolved truthy -- the wizard-writes-strings "
        "bug is present"
    )


@pytest.mark.parametrize("key,attr,default", AFFECTED_KEYS)
def test_string_true_resolves_to_real_true(key: str, attr: str, default: bool):
    """String "true" must resolve to boolean True (sanity: not just always-False)."""
    provider = _make_provider({key: "true"})
    assert getattr(provider, attr) is True


@pytest.mark.parametrize("key,attr,default", AFFECTED_KEYS)
def test_real_bool_passthrough(key: str, attr: str, default: bool):
    """Real booleans (already-correct config, e.g. from a Python caller or a
    YAML `true`/`false` literal parsed by PyYAML) must pass through unchanged."""
    provider_true = _make_provider({key: True})
    provider_false = _make_provider({key: False})
    assert getattr(provider_true, attr) is True
    assert getattr(provider_false, attr) is False


@pytest.mark.parametrize("key,attr,default", AFFECTED_KEYS)
def test_absent_key_uses_documented_default(key: str, attr: str, default: bool):
    """An absent key must fall back to the key's documented default."""
    provider = _make_provider({})
    assert getattr(provider, attr) is default


# ---------------------------------------------------------------------------
# Integration-flavored assertion for the live-reachable key:
# enable_prompt_caching="false" (string, exactly what the wizard writes) must
# result in NO cache_control blocks anywhere in a built request -- mirrors
# `test_prompt_caching_disabled_places_no_breakpoints_at_all` in
# test_prompt_cache_breakpoints.py, but drives the config through the
# wizard's actual string shape instead of a real Python bool.
# ---------------------------------------------------------------------------


def _count_cache_control_blocks(params: dict) -> int:
    count = 0
    for block in params.get("system") or []:
        if isinstance(block, dict) and "cache_control" in block:
            count += 1
    for tool in params.get("tools") or []:
        if isinstance(tool, dict) and "cache_control" in tool:
            count += 1
    for msg in params.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    count += 1
    return count


def _capture_params(provider: AnthropicProvider) -> dict:
    captured: dict = {}

    async def _fake_create(**params):
        captured.update(params)
        raw = MagicMock()
        raw.parse = AsyncMock(return_value=DummyResponse())
        raw.headers = {}
        return raw

    provider.client.messages.with_raw_response.create = AsyncMock(
        side_effect=_fake_create
    )
    return captured


def test_wizard_string_false_disables_prompt_caching_end_to_end():
    """The live-reachable regression: `enable_prompt_caching: "false"` (a
    string, exactly what the app-cli wizard writes for a boolean ConfigField
    answer) must produce a request with ZERO cache_control blocks -- not the
    inverted "caching stays on" behavior the pre-fix truthiness bug produces.
    """
    provider = AnthropicProvider(
        api_key="test-key",
        config={"use_streaming": False, "enable_prompt_caching": "false"},
    )
    params = _capture_params(provider)

    messages = [
        Message(role="system", content="System prompt."),
        Message(role="user", content="question"),
        Message(role="assistant", content="answer"),
    ]
    request = ChatRequest(messages=messages)

    async def _complete_and_close() -> None:
        await provider.complete(request)
        await provider.close()

    asyncio.run(_complete_and_close())

    assert provider.enable_prompt_caching is False, (
        "enable_prompt_caching='false' (string) did not resolve to False -- "
        "the wizard-writes-strings bug is present"
    )
    assert _count_cache_control_blocks(params) == 0, (
        "enable_prompt_caching='false' (string) still placed cache_control "
        f"blocks in the request: {params}"
    )
