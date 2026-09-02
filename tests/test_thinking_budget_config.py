"""Config `thinking_budget_tokens` must reach the wire — or say why it didn't.

BACKGROUND (the defect these tests pin)

Lane d0q measured the provider at HEAD 6abfcff with a mocked transport and
found `thinking_budget_tokens` SILENTLY INERT from provider config on
claude-haiku-4-5, where `thinking.budget_tokens` is the entire reasoning dial
(`effort` is absent from every haiku request). Four of five configurations
that set an explicit budget had it discarded with no warning; the fifth never
enabled thinking, so the budget was never read at all:

    config budget=8000  + request effort=high  -> wire 32000  (discarded)
    config budget=64000 + request effort=high  -> wire 32000  (discarded)
    config budget=8000  + request effort=low   -> wire 4096   (discarded)
    config budget=8000  + config  effort=low   -> wire 4096   (discarded)
    config budget=8000  + no effort at all     -> thinking ABSENT

Root cause: the budget chain resolved `kwargs > effort_budget > config >
default`, and the effort ladder always produced a non-None `effort_budget`
whenever ANY reasoning_effort was set — so config sat below a value that was
always present. The only budgets reachable from config were {4096, 32000}.

The defect is the SILENCE, not the precedence: a documented, allow-listed
config key was accepted without complaint and then thrown away — the same
defect class a discarded `effort` was given a loader guard for.

WHAT THESE TESTS ASSERT

  1. FAIL-BEFORE (TestConfigBudgetReachesTheWire): each of the five
     configurations above now either lands on the wire or emits a targeted
     warning naming the key, the value asked for, and the value sent.
  2. BYTE-IDENTITY (TestDefaultPathByteIdentity): with NO explicit budget
     anywhere, the constructed request body is byte-for-byte what it was
     before this change — the non-regression promise for the daily driver.

NO API SPEND: every assertion is on the constructed request body.
"""

import asyncio
import json
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import DummyResponse, FakeCoordinator

HAIKU = "claude-haiku-4-5-20251001"
HAIKU_35 = "claude-haiku-3-5-20250929"
SONNET = "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(default_model: str = HAIKU, **config: Any) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": 0,
            "default_model": default_model,
            **config,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_mock()
    )
    return provider


def _make_raw_mock() -> MagicMock:
    raw = MagicMock()
    raw.parse = AsyncMock(return_value=DummyResponse())
    raw.headers = {}
    return raw


def _get_api_params(provider: AnthropicProvider) -> dict[str, Any]:
    """The effective wire params, with `extra_body` merged up to top level."""
    mock_create = provider.client.messages.with_raw_response.create
    assert mock_create.await_count == 1
    _, kwargs = mock_create.call_args
    params = dict(kwargs)
    extra_body = params.pop("extra_body", None) or {}
    for key, value in extra_body.items():
        params.setdefault(key, value)
    return params


def _run(
    provider: AnthropicProvider,
    *,
    reasoning_effort: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=reasoning_effort,
    )
    asyncio.run(provider.complete(request, **kwargs))
    return _get_api_params(provider)


def _budget_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "thinking_budget_tokens" in r.getMessage()
    ]


# ---------------------------------------------------------------------------
# FAIL-BEFORE: the five silently-inert configurations d0q measured
# ---------------------------------------------------------------------------


class TestConfigBudgetReachesTheWire:
    """Each of these FAILS on HEAD 6abfcff (the value is silently discarded)."""

    @pytest.mark.parametrize(
        "effort_source,effort",
        [
            ("request", "high"),
            ("request", "low"),
            ("request", "medium"),
            ("request", "xhigh"),
            ("config", "low"),
            ("config", "high"),
        ],
    )
    def test_config_budget_beats_the_effort_ladder_on_haiku(
        self, effort_source: str, effort: str
    ):
        """config thinking_budget_tokens=8000 reaches thinking.budget_tokens
        for EVERY effort, from either the request or the config.

        Before: 'low' -> 4096, everything else -> 32000; the config value never
        appeared on the wire.
        """
        config: dict[str, Any] = {"thinking_budget_tokens": 8000}
        request_effort = None
        if effort_source == "config":
            config["reasoning_effort"] = effort
        else:
            request_effort = effort

        provider = _make_provider(HAIKU, **config)
        params = _run(provider, reasoning_effort=request_effort)

        assert params["thinking"] == {"type": "enabled", "budget_tokens": 8000}

    def test_kwargs_budget_still_outranks_config_budget(self):
        """Per-request kwargs remain the highest-precedence budget source."""
        provider = _make_provider(HAIKU, thinking_budget_tokens=8000)
        params = _run(provider, reasoning_effort="high", thinking_budget_tokens=16000)

        assert params["thinking"]["budget_tokens"] == 16000

    def test_config_budget_above_the_ceiling_is_clamped_and_warned(
        self, caplog: pytest.LogCaptureFixture
    ):
        """d0q's 64000 cell. Haiku's ceiling is 64000 and a tool-less request
        reserves one token for the response, so 64000 -> 63999. The caller is
        told what they asked for and what was sent, instead of guessing.
        """
        provider = _make_provider(HAIKU, thinking_budget_tokens=64000)
        with caplog.at_level(logging.WARNING):
            params = _run(provider, reasoning_effort="high")

        assert params["thinking"]["budget_tokens"] == 63999
        warnings = _budget_warnings(caplog)
        assert warnings, "clamping an explicit budget must not be silent"
        assert "64000" in warnings[0] and "63999" in warnings[0]

    def test_config_budget_with_no_effort_warns_instead_of_vanishing(
        self, caplog: pytest.LogCaptureFixture
    ):
        """The fifth configuration: thinking is never enabled, so the budget is
        never read. Before, this produced no thinking param and no warning."""
        provider = _make_provider(HAIKU, thinking_budget_tokens=8000)
        with caplog.at_level(logging.WARNING):
            params = _run(provider)

        assert "thinking" not in params
        warnings = _budget_warnings(caplog)
        assert warnings, "a budget that is never read must not be silent"
        assert "8000" in warnings[0]
        # The warning must name a remedy the caller can actually apply.
        assert "reasoning_effort" in warnings[0]
        assert "extended_thinking" in warnings[0]

    def test_config_extended_thinking_turns_thinking_on_without_an_effort(self):
        """The remedy the warning above names, and the reason it exists: a
        config-only caller can now enable thinking WITHOUT also choosing an
        effort, so their budget becomes readable at all."""
        provider = _make_provider(
            HAIKU, thinking_budget_tokens=8000, extended_thinking=True
        )
        params = _run(provider)

        assert params["thinking"] == {"type": "enabled", "budget_tokens": 8000}

    def test_config_extended_thinking_false_opts_out_of_config_effort(self):
        """The inverse: an explicit config opt-out beats the effort
        implication, mirroring kwargs['extended_thinking']=False."""
        provider = _make_provider(
            HAIKU, reasoning_effort="high", extended_thinking=False
        )
        params = _run(provider)

        assert "thinking" not in params

    def test_kwargs_extended_thinking_still_outranks_config(self):
        provider = _make_provider(HAIKU, extended_thinking=True)
        params = _run(provider, extended_thinking=False)

        assert "thinking" not in params

    def test_config_budget_on_a_model_that_cannot_think_warns(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Haiku 3.5 has supports_thinking=False. The budget can never land;
        say so rather than dropping it."""
        provider = _make_provider(HAIKU_35, thinking_budget_tokens=8000)
        with caplog.at_level(logging.WARNING):
            params = _run(provider, reasoning_effort="high")

        assert "thinking" not in params
        warnings = _budget_warnings(caplog)
        assert warnings
        assert "does not support extended thinking" in warnings[0]

    def test_config_budget_reaches_the_wire_on_sonnet_too(self):
        """Not haiku-specific: the same shadowing hit every model whose
        thinking_type resolves to 'enabled'."""
        provider = _make_provider(SONNET, thinking_budget_tokens=8000)
        params = _run(provider, reasoning_effort="high")

        assert params["thinking"]["budget_tokens"] == 8000

    def test_invalid_config_budget_warns_and_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ):
        """A typo must not raise ValueError out of int() on every request."""
        provider = _make_provider(HAIKU, thinking_budget_tokens="not-a-number")
        with caplog.at_level(logging.WARNING):
            params = _run(provider, reasoning_effort="high")

        assert params["thinking"]["budget_tokens"] == 32000  # model default
        assert _budget_warnings(caplog)

    def test_adaptive_model_warns_and_says_where_the_value_went(
        self, caplog: pytest.LogCaptureFixture
    ):
        """On an adaptive-thinking model the API forbids budget_tokens
        outright. The value only feeds max_tokens sizing, and the warning must
        report the max_tokens actually resolved -- not imply the budget set it
        (here the model ceiling already exceeds budget+buffer, so it did not).
        """
        provider = _make_provider("claude-sonnet-4-6", thinking_budget_tokens=8000)
        with caplog.at_level(logging.WARNING):
            params = _run(provider, reasoning_effort="high")

        assert params["thinking"] == {"type": "adaptive"}
        warnings = _budget_warnings(caplog)
        assert warnings
        assert "adaptive" in warnings[0]
        assert f"max_tokens={params['max_tokens']}" in warnings[0]
        assert "thinking_type" in warnings[0]

    def test_always_on_model_warns_without_claiming_the_value_was_used(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Fable 5.1 always thinks and rejects a thinking param entirely, so
        the budget is not sent AND does not size anything. The warning must not
        borrow the adaptive branch's 'used to size max_tokens' claim."""
        provider = _make_provider("claude-fable-5-1", thinking_budget_tokens=8000)
        with caplog.at_level(logging.WARNING):
            params = _run(provider, reasoning_effort="high")

        assert "thinking" not in params
        warnings = _budget_warnings(caplog)
        assert warnings
        assert "always thinks" in warnings[0]
        assert "max_tokens=" not in warnings[0]

    def test_zero_config_budget_is_not_silently_ignored(
        self, caplog: pytest.LogCaptureFixture
    ):
        """0 is below the API's 1024 minimum. Falling back is correct; doing it
        without saying so is the defect this PR closes."""
        provider = _make_provider(HAIKU, thinking_budget_tokens=0)
        with caplog.at_level(logging.WARNING):
            params = _run(provider, reasoning_effort="high")

        assert params["thinking"]["budget_tokens"] == 32000
        assert _budget_warnings(caplog)

    def test_string_config_budget_is_coerced(self):
        """settings.yaml round-trips numbers as strings often enough that the
        string form must work, like every other numeric key here."""
        provider = _make_provider(HAIKU, thinking_budget_tokens="8000")
        params = _run(provider, reasoning_effort="high")

        assert params["thinking"]["budget_tokens"] == 8000


# ---------------------------------------------------------------------------
# BYTE-IDENTITY: the default path must not move
# ---------------------------------------------------------------------------

# Captured from HEAD 6abfcff (pre-fix) via
# `scripts/thinking_budget_reachability.py`, with NO thinking_budget_tokens and
# NO extended_thinking set anywhere. These are the exact request bodies the
# daily driver sends today. If a value here changes, the non-regression promise
# is broken and this test is the thing that says so.
_DEFAULT_PATH_BASELINE: dict[str, dict[str, Any]] = {
    "haiku-effort-high": {
        "max_tokens": 64000,
        "model": "claude-haiku-4-5-20251001",
        "temperature": 1.0,
        "thinking": {"budget_tokens": 32000, "type": "enabled"},
        "timeout": 600.0,
    },
    "haiku-effort-low": {
        "max_tokens": 64000,
        "model": "claude-haiku-4-5-20251001",
        "temperature": 1.0,
        "thinking": {"budget_tokens": 4096, "type": "enabled"},
        "timeout": 600.0,
    },
    "haiku-effort-medium": {
        "max_tokens": 64000,
        "model": "claude-haiku-4-5-20251001",
        "temperature": 1.0,
        "thinking": {"budget_tokens": 32000, "type": "enabled"},
        "timeout": 600.0,
    },
    "haiku-effort-xhigh": {
        "max_tokens": 64000,
        "model": "claude-haiku-4-5-20251001",
        "temperature": 1.0,
        "thinking": {"budget_tokens": 32000, "type": "enabled"},
        "timeout": 600.0,
    },
    "haiku-no-effort": {
        "max_tokens": 64000,
        "model": "claude-haiku-4-5-20251001",
        "temperature": 0.7,
        "timeout": 600.0,
    },
    "sonnet-effort-high": {
        "max_tokens": 64000,
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 1.0,
        "thinking": {"budget_tokens": 32000, "type": "enabled"},
        "timeout": 600.0,
    },
    "sonnet-effort-low": {
        "max_tokens": 64000,
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 1.0,
        "thinking": {"budget_tokens": 4096, "type": "enabled"},
        "timeout": 600.0,
    },
    "sonnet-no-effort": {
        "max_tokens": 64000,
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 0.7,
        "timeout": 600.0,
    },
}

_DEFAULT_PATH_CELLS: dict[str, tuple[str, str | None]] = {
    "haiku-effort-high": (HAIKU, "high"),
    "haiku-effort-medium": (HAIKU, "medium"),
    "haiku-effort-xhigh": (HAIKU, "xhigh"),
    "haiku-effort-low": (HAIKU, "low"),
    "haiku-no-effort": (HAIKU, None),
    "sonnet-effort-high": (SONNET, "high"),
    "sonnet-effort-low": (SONNET, "low"),
    "sonnet-no-effort": (SONNET, None),
}


class TestDefaultPathByteIdentity:
    @pytest.mark.parametrize("cell", sorted(_DEFAULT_PATH_CELLS))
    def test_default_request_body_is_byte_identical_to_pre_fix(self, cell: str):
        """ANTHROPIC IS THE DAILY DRIVER. With no explicit thinking budget the
        constructed request body must be byte-for-byte what HEAD 6abfcff sent.
        """
        model, effort = _DEFAULT_PATH_CELLS[cell]
        provider = _make_provider(model)
        params = _run(provider, reasoning_effort=effort)
        params.pop("messages", None)

        expected = _DEFAULT_PATH_BASELINE[cell]
        assert json.dumps(params, sort_keys=True) == json.dumps(
            expected, sort_keys=True
        )

    @pytest.mark.parametrize("cell", sorted(_DEFAULT_PATH_CELLS))
    def test_default_path_emits_no_budget_warning(
        self, cell: str, caplog: pytest.LogCaptureFixture
    ):
        """The new guard must be invisible unless a budget was explicitly
        asked for — no new log noise on the default path."""
        model, effort = _DEFAULT_PATH_CELLS[cell]
        provider = _make_provider(model)
        with caplog.at_level(logging.WARNING):
            _run(provider, reasoning_effort=effort)

        assert not _budget_warnings(caplog)

    def test_config_effort_default_path_unchanged(self):
        """Config-supplied effort with no budget: still the model default."""
        provider = _make_provider(HAIKU, reasoning_effort="high")
        params = _run(provider)

        assert params["thinking"] == {"type": "enabled", "budget_tokens": 32000}
