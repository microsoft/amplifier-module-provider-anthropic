"""Tests for model capability detection and version-gated token limits.

Validates that _get_capabilities returns correct max_output_tokens,
thinking budgets, and feature flags for each model family and version.
"""

import asyncio
import dataclasses
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_anthropic import (
    AnthropicProvider,
    ModelCapabilities,
    _RuntimeModelInfo,
)

from tests._helpers import DummyResponse, FakeCoordinator


def _make_provider(default_model: str = "claude-fable-5") -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": 0,
            "default_model": default_model,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _make_raw_mock() -> MagicMock:
    raw = MagicMock()
    raw.parse = AsyncMock(return_value=DummyResponse())
    raw.headers = {}
    return raw


def _get_api_params(mock_create: AsyncMock) -> dict[str, Any]:
    """Extract the effective wire params for the API call.

    `extra_body` entries are merged up to the top level. Params like
    `temperature` and `speed` are not typed keyword arguments on the SDK
    (`temperature` was removed in anthropic 1.0.0; `speed` was never typed),
    so the provider sends them via `extra_body`. Tests assert on what reaches
    the API, not on which transport carried it.
    """
    assert mock_create.await_count == 1
    _, kwargs = mock_create.call_args
    params = dict(kwargs)
    extra_body = params.pop("extra_body", None) or {}
    for key, value in extra_body.items():
        params.setdefault(key, value)
    return params


class TestDetectFamily:
    """Tests for _detect_family static method."""

    def test_opus_family(self):
        assert AnthropicProvider._detect_family("claude-opus-4-6-20260101") == "opus"

    def test_sonnet_family(self):
        assert (
            AnthropicProvider._detect_family("claude-sonnet-4-5-20250929") == "sonnet"
        )

    def test_haiku_family(self):
        assert AnthropicProvider._detect_family("claude-haiku-3-5-20250929") == "haiku"

    def test_unknown_defaults_to_sonnet(self):
        assert AnthropicProvider._detect_family("claude-mystery-9-9") == "sonnet"

    def test_bare_opus(self):
        assert AnthropicProvider._detect_family("claude-opus-4-6") == "opus"


class TestDetectVersion:
    """Tests for _detect_version static method."""

    def test_opus_46(self):
        assert AnthropicProvider._detect_version(
            "claude-opus-4-6-20260101", "opus"
        ) == (4, 6)

    def test_opus_45(self):
        assert AnthropicProvider._detect_version(
            "claude-opus-4-5-20251101", "opus"
        ) == (4, 5)

    def test_opus_bare_alias(self):
        # Bare alias without date — version not parseable
        assert AnthropicProvider._detect_version("claude-opus-4-6", "opus") == (4, 6)

    def test_unparseable_returns_zero(self):
        assert AnthropicProvider._detect_version("claude-opus-latest", "opus") == (0, 0)

    # Claude 3-generation ids put the version before the family. The family word
    # therefore sits directly against the snapshot date, which the major-only
    # fallback used to read as the major version (major 20241022 and friends).

    def test_legacy_haiku_35(self):
        assert AnthropicProvider._detect_version(
            "claude-3-5-haiku-20241022", "haiku"
        ) == (3, 5)

    def test_legacy_sonnet_35(self):
        assert AnthropicProvider._detect_version(
            "claude-3-5-sonnet-20241022", "sonnet"
        ) == (3, 5)

    def test_legacy_sonnet_37(self):
        assert AnthropicProvider._detect_version(
            "claude-3-7-sonnet-20250219", "sonnet"
        ) == (3, 7)

    def test_legacy_haiku_3_major_only(self):
        assert AnthropicProvider._detect_version(
            "claude-3-haiku-20240307", "haiku"
        ) == (3, 0)

    def test_legacy_opus_3_major_only(self):
        assert AnthropicProvider._detect_version(
            "claude-3-opus-20240229", "opus"
        ) == (3, 0)

    # Family-first ids must keep parsing exactly as before.

    def test_modern_major_only_id_with_snapshot(self):
        assert AnthropicProvider._detect_version(
            "claude-opus-4-20250514", "opus"
        ) == (4, 0)

    def test_modern_major_minor_id_with_snapshot(self):
        assert AnthropicProvider._detect_version(
            "claude-sonnet-4-5-20250929", "sonnet"
        ) == (4, 5)

    def test_bare_family_major(self):
        assert AnthropicProvider._detect_version("claude-fable-5", "fable") == (5, 0)


class TestLegacyModelCapabilities:
    """Claude 3-generation ids must land on the conservative capability tier.

    Before the version parse was fixed these ids reported majors like 20241022,
    which cleared every ``>=`` gate and handed retired models the newest tier —
    1M context, adaptive thinking, ``speed``, native computer use — none of
    which those models accept.
    """

    def test_haiku_35_has_no_thinking_or_computer_use(self):
        caps = AnthropicProvider._get_capabilities("claude-3-5-haiku-20241022")
        assert caps.supports_thinking is False
        assert caps.supports_native_computer_use is False
        assert caps.default_thinking_budget == 0

    def test_haiku_3_has_no_thinking_or_computer_use(self):
        caps = AnthropicProvider._get_capabilities("claude-3-haiku-20240307")
        assert caps.supports_thinking is False
        assert caps.supports_native_computer_use is False

    def test_sonnet_37_is_not_treated_as_latest(self):
        caps = AnthropicProvider._get_capabilities("claude-3-7-sonnet-20250219")
        assert caps.supports_1m is False
        assert caps.supports_adaptive_thinking is False
        assert caps.max_output_tokens == 64000
        assert caps.supported_efforts == ("low", "medium", "high")

    def test_opus_3_is_not_treated_as_latest(self):
        caps = AnthropicProvider._get_capabilities("claude-3-opus-20240229")
        assert caps.supports_1m is False
        assert caps.supports_speed is False
        assert caps.supports_native_computer_use is False
        assert caps.max_output_tokens == 64000

    def test_current_models_keep_their_capabilities(self):
        opus = AnthropicProvider._get_capabilities("claude-opus-4-8")
        assert opus.supports_speed is True
        assert opus.supports_1m is True
        assert opus.max_output_tokens == 128000

        sonnet = AnthropicProvider._get_capabilities("claude-sonnet-4-5-20250929")
        assert sonnet.supports_1m is False
        assert sonnet.max_output_tokens == 64000
        assert sonnet.supported_efforts == ("low", "medium", "high")

        fable = AnthropicProvider._get_capabilities("claude-fable-5")
        assert fable.supports_1m is True
        assert fable.supports_adaptive_thinking is True


class TestGetCapabilitiesOpus:
    """Tests for Opus model capabilities — the core of the issue #52 fix."""

    def test_opus_45_max_output_tokens(self):
        """Opus 4.5 must use 64000 max_output_tokens (API ceiling)."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.max_output_tokens == 64000

    def test_opus_46_max_output_tokens(self):
        """Opus 4.6+ gets 128000 max_output_tokens."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.max_output_tokens == 128000

    def test_opus_bare_alias_assumes_latest(self):
        """Bare alias 'claude-opus-4-6' should get 4.6+ capabilities."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6")
        assert caps.max_output_tokens == 128000
        assert caps.supports_1m is True
        assert caps.supports_adaptive_thinking is True

    def test_opus_unknown_version_assumes_latest(self):
        """Unknown version defaults to latest (128K) for forward compatibility."""
        caps = AnthropicProvider._get_capabilities("claude-opus-latest")
        assert caps.max_output_tokens == 128000

    def test_opus_45_thinking_budget(self):
        """Opus 4.5 gets reduced thinking budget to stay within 64K ceiling."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.default_thinking_budget == 32000

    def test_opus_46_thinking_budget(self):
        """Opus 4.6+ gets full 64K thinking budget."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.default_thinking_budget == 64000

    def test_opus_45_no_1m_no_adaptive(self):
        """Opus 4.5 does not support 1M context or adaptive thinking."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.supports_1m is False
        assert caps.supports_adaptive_thinking is False

    def test_opus_46_has_1m_and_adaptive(self):
        """Opus 4.6+ supports 1M context and adaptive thinking."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.supports_1m is True
        assert caps.supports_adaptive_thinking is True

    def test_all_opus_supports_thinking(self):
        """All Opus versions support extended thinking."""
        for model_id in ["claude-opus-4-5-20251101", "claude-opus-4-6-20260101"]:
            caps = AnthropicProvider._get_capabilities(model_id)
            assert caps.supports_thinking is True

    def test_opus_family_tag(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.family == "opus"

    def test_opus_thinking_budget_within_ceiling(self):
        """Thinking budget + reasonable buffer must not exceed max_output_tokens.

        This validates the secondary fix: with a 4096 buffer, the thinking
        budget must leave room within the model's output ceiling.
        """
        buffer = 4096
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.default_thinking_budget + buffer <= caps.max_output_tokens


class TestGetCapabilitiesOpus48:
    """Tests for Opus 4.8 capabilities — is_48_plus gate, speed/inline_system flags, max effort."""

    def test_opus_48_supports_speed(self):
        """Opus 4.8 accepts the speed parameter."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        assert caps.supports_speed is True

    def test_opus_48_supports_inline_system(self):
        """Opus 4.8 accepts role='system' in messages[]."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        assert caps.supports_inline_system is True

    def test_opus_48_has_max_effort(self):
        """Opus 4.8 has 'max' effort tier and the full effort tuple."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        assert "max" in caps.supported_efforts
        assert caps.supported_efforts == ("low", "medium", "high", "xhigh", "max")

    def test_opus_47_does_not_support_speed(self):
        """Opus 4.7 does NOT accept the speed parameter."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_speed is False
        assert caps.supports_inline_system is False

    def test_opus_47_has_xhigh_and_max(self):
        """Opus 4.7 has both 'xhigh' and 'max' effort tiers (C-05 correction:
        the doc's compatibility table places the xhigh/max split at 4.6/4.7,
        not 4.7/4.8 -- 'max' \u2287 'xhigh', so 4.7+ gets both)."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert "xhigh" in caps.supported_efforts
        assert "max" in caps.supported_efforts
        assert caps.supported_efforts == ("low", "medium", "high", "xhigh", "max")

    def test_opus_unknown_version_assumes_48(self):
        """Unknown opus version (e.g. claude-opus-latest) assumes 4.8 for forward compatibility."""
        caps = AnthropicProvider._get_capabilities("claude-opus-latest")
        assert caps.supports_speed is True
        assert "max" in caps.supported_efforts


class TestGetCapabilitiesSonnet:
    """Tests for Sonnet model capabilities (should be unaffected by fix)."""

    def test_sonnet_max_output_tokens_is_default(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-5-20250929")
        assert caps.max_output_tokens == 64000

    def test_sonnet_46_plus_max_output_tokens_is_128k(self):
        """Sonnet 4.6+ has a 1M context window and is entitled to 128K output
        (platform.claude.com/en/docs/build-with-claude/context-windows,
        2026-08-29). Regression guard for the bug where the sonnet branch
        never set max_output_tokens and silently inherited the dataclass
        default of 64000, clamping claude-sonnet-5 to half its real ceiling.
        """
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-6")
        assert caps.max_output_tokens == 128000
        caps5 = AnthropicProvider._get_capabilities("claude-sonnet-5")
        assert caps5.max_output_tokens == 128000

    def test_sonnet_supports_thinking(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-5-20250929")
        assert caps.supports_thinking is True
        assert caps.supports_adaptive_thinking is False

    def test_sonnet_thinking_budget(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-5-20250929")
        assert caps.default_thinking_budget == 32000


class TestGetCapabilitiesHaiku:
    """Tests for Haiku model capabilities — version-gated thinking support.

    Haiku 4.5+ supports extended thinking (per Anthropic docs).
    Haiku 3.5 does NOT support thinking.
    """

    # --- Haiku 3.5 (no thinking) ---

    def test_haiku_35_max_output_tokens_is_default(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert caps.max_output_tokens == 64000

    def test_haiku_35_no_thinking(self):
        """Haiku 3.5 does not support extended thinking."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert caps.supports_thinking is False
        assert caps.supports_adaptive_thinking is False
        assert caps.default_thinking_budget == 0

    def test_haiku_35_no_thinking_tag(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert "thinking" not in caps.capability_tags

    def test_haiku_35_family(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-3-5-20250929")
        assert caps.family == "haiku"

    # --- Haiku 4.5 (thinking supported) ---

    def test_haiku_45_supports_thinking(self):
        """Haiku 4.5 supports extended thinking per Anthropic docs."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.supports_thinking is True

    def test_haiku_45_no_adaptive_thinking(self):
        """Haiku 4.5 does NOT support adaptive thinking per Anthropic docs."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.supports_adaptive_thinking is False

    def test_haiku_45_thinking_budget(self):
        """Haiku 4.5 gets 32K default thinking budget."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.default_thinking_budget == 32000

    def test_haiku_45_has_thinking_tag(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert "thinking" in caps.capability_tags

    def test_haiku_45_has_fast_tag(self):
        """Haiku 4.5 retains the 'fast' tag."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert "fast" in caps.capability_tags

    def test_haiku_45_family(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.family == "haiku"

    def test_haiku_45_max_output_tokens_is_default(self):
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.max_output_tokens == 64000

    # --- Unknown Haiku (defaults to latest = thinking enabled) ---

    def test_haiku_unknown_version_assumes_latest(self):
        """Unknown haiku version defaults to latest (thinking enabled)."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-latest")
        assert caps.supports_thinking is True
        assert caps.default_thinking_budget == 32000


class TestFastModeBetaHeader:
    """Tests for BETA_HEADER_FAST_MODE constant and fast_mode kwarg in _build_request_beta_headers."""

    def test_fast_mode_beta_header_constant(self):
        """BETA_HEADER_FAST_MODE must equal the expected beta header string."""
        from amplifier_module_provider_anthropic import BETA_HEADER_FAST_MODE

        assert BETA_HEADER_FAST_MODE == "fast-mode-2026-02-01"

    def test_beta_header_added_when_fast_mode(self):
        """fast_mode=True must include BETA_HEADER_FAST_MODE in returned headers."""
        from amplifier_module_provider_anthropic import BETA_HEADER_FAST_MODE

        provider = AnthropicProvider(api_key="test-key", config={"max_retries": 0})
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        headers = provider._build_request_beta_headers(
            request_caps=caps,
            tools_present=False,
            resolved_thinking_type=None,
            fast_mode=True,
        )
        assert BETA_HEADER_FAST_MODE in headers

    def test_beta_header_absent_when_not_fast_mode(self):
        """fast_mode=False must NOT include BETA_HEADER_FAST_MODE in returned headers."""
        from amplifier_module_provider_anthropic import BETA_HEADER_FAST_MODE

        provider = AnthropicProvider(api_key="test-key", config={"max_retries": 0})
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        headers = provider._build_request_beta_headers(
            request_caps=caps,
            tools_present=False,
            resolved_thinking_type=None,
            fast_mode=False,
        )
        assert BETA_HEADER_FAST_MODE not in headers


class TestContextBetaHeaderNeverSent:
    """1M context is GA/default/standard-priced on every model that has it
    (platform.claude.com/en/docs/build-with-claude/context-windows, verified
    2026-08-29) -- no beta header is EVER required or sent for it, on any
    model, regardless of version. _should_add_context_1m_beta and
    BETA_HEADER_1M_CONTEXT are removed entirely (C-01)."""

    def test_opus_48_no_1m_beta_header(self):
        provider = AnthropicProvider(api_key="test-key", config={"max_retries": 0})
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        headers = provider._build_request_beta_headers(
            request_caps=caps,
            tools_present=False,
            resolved_thinking_type=None,
        )
        assert "context-1m-2025-08-07" not in headers

    def test_opus_47_no_1m_beta_header(self):
        provider = AnthropicProvider(api_key="test-key", config={"max_retries": 0})
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        headers = provider._build_request_beta_headers(
            request_caps=caps,
            tools_present=False,
            resolved_thinking_type=None,
        )
        assert "context-1m-2025-08-07" not in headers

    def test_opus_unknown_version_no_1m_beta_header(self):
        provider = AnthropicProvider(api_key="test-key", config={"max_retries": 0})
        caps = AnthropicProvider._get_capabilities("claude-opus-latest")
        headers = provider._build_request_beta_headers(
            request_caps=caps,
            tools_present=False,
            resolved_thinking_type=None,
        )
        assert "context-1m-2025-08-07" not in headers


class TestSpeedConfigPlumbing:
    """Tests for speed config key validation and beta header plumbing."""

    def test_supported_model_unsupported_speed_logs_and_omits(self):
        """Opus 4.7 does not support speed — provider omits the param and skips the beta header."""
        from amplifier_module_provider_anthropic import BETA_HEADER_FAST_MODE

        provider = AnthropicProvider(
            api_key="test-key", config={"max_retries": 0, "speed": "fast"}
        )
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_speed is False
        headers = provider._build_request_beta_headers(
            request_caps=caps,
            tools_present=False,
            resolved_thinking_type=None,
            fast_mode=False,
        )
        assert BETA_HEADER_FAST_MODE not in headers


class TestThinkingAlwaysOn:
    """thinking_always_on: False by default; True for the fable family (Task 3)."""

    def test_thinking_always_on_default_false(self):
        """ModelCapabilities defaults thinking_always_on to False."""
        caps = ModelCapabilities(family="test")
        assert caps.thinking_always_on is False

    def test_opus_thinking_always_on_false(self):
        """Opus models do NOT have always-on thinking."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-8")
        assert caps.thinking_always_on is False

    def test_sonnet_thinking_always_on_false(self):
        """Sonnet models do NOT have always-on thinking."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-6")
        assert caps.thinking_always_on is False

    def test_opus_5_capabilities_match_opus_48_gate(self):
        """claude-opus-5 must match claude-opus-4-8's capability matrix on
        every gate keyed to is_48_plus (the existing numeric version-gate
        already handles Opus 5 correctly, since (5, 0) >= (4, 8) -- no code
        change was required for Opus 5 capability detection on those axes).

        min_cacheable_tokens is the one EXPECTED exception: it is gated on
        is_5_plus specifically (Opus 5 = 512, Opus 4.8 = 1024) per
        Anthropic's own non-monotonic per-model cache-minimum table
        (platform.claude.com/en/docs/build-with-claude/prompt-caching,
        verified 2026-08-29) -- so a whole-object equality is no longer
        the right check here.
        """
        caps_5 = AnthropicProvider._get_capabilities("claude-opus-5")
        caps_48 = AnthropicProvider._get_capabilities("claude-opus-4-8")

        assert caps_5.family == "opus"
        assert caps_5.max_output_tokens == 128000
        assert caps_5.supports_1m is True
        assert caps_5.supports_adaptive_thinking is True
        assert caps_5.supports_manual_thinking is False
        assert caps_5.supported_efforts == ("low", "medium", "high", "xhigh", "max")
        assert caps_5.supports_speed is True
        assert caps_5.supports_inline_system is True
        assert caps_5.min_cacheable_tokens == 512
        assert caps_48.min_cacheable_tokens == 1024
        assert dataclasses.replace(caps_5, min_cacheable_tokens=1024) == caps_48


class TestGetCapabilitiesFable5:
    """Fable 5 capability matrix."""

    def test_fable5_family_detected(self):
        """claude-fable-5 detects family='fable'."""
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.family == "fable"

    def test_fable5_thinking_always_on(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.thinking_always_on is True

    def test_fable5_supports_1m(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.supports_1m is True

    def test_fable5_max_output_128k(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.max_output_tokens == 128000

    def test_fable5_supports_adaptive_thinking(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.supports_adaptive_thinking is True

    def test_fable5_no_manual_thinking(self):
        """Manual thinking (budget_tokens) is not accepted on Fable 5."""
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.supports_manual_thinking is False

    def test_fable5_all_effort_levels(self):
        """Fable 5 supports all 5 effort levels including max."""
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.supported_efforts == ("low", "medium", "high", "xhigh", "max")

    def test_fable5_no_speed(self):
        """Speed mode is NOT supported on Fable 5 (spike confirmed)."""
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.supports_speed is False

    def test_fable5_inline_system(self):
        """Inline system messages are supported (spike confirmed schema exists)."""
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.supports_inline_system is True

    def test_fable5_thinking_display_required(self):
        """display defaults to 'omitted' on Fable 5 — same as Opus 4.8."""
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.thinking_display_required is True

    def test_fable5_no_sampling(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.supports_sampling is False

    def test_fable5_supports_output_config(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.supports_output_config is True

    def test_fable5_supports_task_budget(self):
        caps = AnthropicProvider._get_capabilities("claude-fable-5")
        assert caps.supports_task_budget is True

    def test_unknown_fable_version_assumes_latest(self):
        """Unknown version assumes latest (thinking_always_on=True)."""
        caps = AnthropicProvider._get_capabilities("claude-fable-latest")
        assert caps.thinking_always_on is True


class TestThinkingAlwaysOnRequestBehavior:
    """thinking_always_on=True: the provider never injects a thinking param."""

    def test_fable5_no_thinking_param_with_reasoning_effort(self):
        """claude-fable-5 + reasoning_effort='high' must NOT send thinking param.

        Fable 5 has thinking always on — the API controls it implicitly.
        Sending {type:disabled} (or any explicit thinking param) causes HTTP 400.
        """
        provider = _make_provider(default_model="claude-fable-5")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))

        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "thinking" not in params


class TestGetCapabilitiesSonnet5:
    """Sonnet 5 (Jun 2026): output_config effort API through xhigh/max, adaptive-only
    thinking (manual type='enabled' -> HTTP 400), displayed thinking, task budget.
    Modeled on the Opus 4.7+ surface + the Sonnet 5 launch; no fast mode."""

    def test_sonnet_5_supports_output_config(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-5")
        assert caps.supports_output_config is True

    def test_sonnet_5_efforts_through_max(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-5")
        assert caps.supported_efforts == ("low", "medium", "high", "xhigh", "max")
        assert "max" in caps.supported_efforts

    def test_sonnet_5_thinking_surface(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-5")
        assert caps.supports_adaptive_thinking is True
        assert caps.supports_manual_thinking is False
        assert caps.thinking_display_required is True
        assert caps.supports_task_budget is True

    def test_sonnet_5_no_speed_no_fast_mode(self):
        """Sonnet 5 must NOT advertise Opus-only fast mode."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-5")
        assert caps.supports_speed is False

    def test_sonnet_5_no_sampling(self):
        """Sonnet 5 must NOT support sampling — Anthropic rejects `temperature`
        ("deprecated for this model"). Regression guard for amplifier-support#299."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-5")
        assert caps.supports_sampling is False

    def test_sonnet_46_unchanged_by_sonnet5_gate(self):
        """Sonnet 4.6 does NOT get Sonnet 5's is_5_plus-gated features
        (xhigh, manual-thinking hard-gate) -- but DOES get 'max' and
        output_config (C-05/C-06 widening: doc states "Supported models:
        ... Sonnet 4.6 and 5"; "max" \\ "xhigh" is Sonnet 4.6's exact case,
        confirmed live 2026-08-29 T-C06-live)."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-6")
        assert caps.supported_efforts == ("low", "medium", "high", "max")
        assert "xhigh" not in caps.supported_efforts
        assert caps.supports_output_config is True
        assert caps.supports_manual_thinking is True
        assert caps.manual_thinking_deprecated is True

    def test_sonnet_unknown_version_assumes_5(self):
        """Forward-compat: a version-less sonnet id assumes latest (5+) so new
        aliases get the current surface. Mirrors test_opus_unknown_version_assumes_48."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-latest")
        assert caps.supports_output_config is True
        assert "xhigh" in caps.supported_efforts
        assert caps.supports_manual_thinking is False

    def test_sonnet_5_caps_survive_runtime_override(self):
        """The is_5_plus flags must not silently reset to dataclass defaults when
        _apply_runtime_capability_overrides reconstructs ModelCapabilities on a
        live request. A non-None _RuntimeModelInfo triggers the construction path
        (None would early-return base_caps and prove nothing)."""
        base = AnthropicProvider._get_capabilities("claude-sonnet-5")
        overridden = AnthropicProvider._apply_runtime_capability_overrides(
            base, _RuntimeModelInfo()
        )
        assert overridden.supports_output_config is True
        assert overridden.supported_efforts == ("low", "medium", "high", "xhigh", "max")
        assert overridden.supports_manual_thinking is False
        assert overridden.supports_task_budget is True
        assert overridden.thinking_display_required is True
