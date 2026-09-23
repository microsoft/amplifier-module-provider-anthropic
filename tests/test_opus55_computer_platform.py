"""Opus 5.5: platform-aware computer-use wire type resolution.

Anthropic's "What's new in Claude Opus 5.5" documents that the
computer-use wire type Opus 5.5 accepts depends on the serving platform:

* first-party Anthropic API and Google Cloud/Vertex AI -- toolset only
  (``computer_toolset_20260801``).
* Amazon Bedrock -- the legacy ``computer_20251124`` type still works.
* Microsoft Foundry and the separate "Claude Platform on AWS" offering --
  undocumented; treated as unsupported rather than guessed.
* any other/unknown ``base_url`` -- unsupported, fail safe.

An explicit ``computer_use_tool_type`` config override always wins over
platform inference, and this platform-awareness must not affect any model
below Opus 5.5.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import InvalidRequestError as KernelInvalidRequestError
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_anthropic import AnthropicProvider, _computer_toolset
from tests._helpers import DummyResponse, FakeCoordinator


def _make_provider(
    model: str = "claude-opus-5-5", *, base_url: str | None = None, **overrides: Any
) -> AnthropicProvider:
    config: dict[str, Any] = {
        "default_model": model,
        "max_retries": 0,
        "use_streaming": False,
        **overrides,
    }
    if base_url is not None:
        config["base_url"] = base_url
    provider = AnthropicProvider(api_key="x", config=config)
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _computer_tool_request() -> ChatRequest:
    tool = ToolSpec(
        name="computer",
        parameters={},
        type="computer_20251124",
        display_width_px=1024,
        display_height_px=768,
    )
    return ChatRequest(messages=[Message(role="user", content="hi")], tools=[tool])


def _stub_create(provider: AnthropicProvider) -> AsyncMock:
    raw_response = MagicMock()
    raw_response.parse = AsyncMock(
        return_value=DummyResponse(content=[], model=provider.default_model)
    )
    raw_response.headers = {}
    create = AsyncMock(return_value=raw_response)
    provider.client.messages.with_raw_response.create = create
    return create


def _run(provider: AnthropicProvider, request: ChatRequest):
    create = _stub_create(provider)
    response = asyncio.run(provider.complete(request))
    assert create.await_count == 1
    return create.call_args.kwargs, response


class TestClassifyPlatformPure:
    """The classifier itself is pure and side-effect-free."""

    @pytest.mark.parametrize(
        "base_url",
        [None, "", "https://api.anthropic.com", "https://api.anthropic.com/v1"],
    )
    def test_first_party(self, base_url):
        assert (
            _computer_toolset.classify_platform(base_url)
            == _computer_toolset.PLATFORM_FIRST_PARTY
        )

    def test_google_vertex(self):
        url = (
            "https://claude.googleapis.com/v1alpha/projects/p/locations/"
            "global/workspaces/w/invoke"
        )
        assert (
            _computer_toolset.classify_platform(url)
            == _computer_toolset.PLATFORM_GOOGLE_VERTEX
        )

    def test_aws_bedrock_runtime(self):
        assert (
            _computer_toolset.classify_platform(
                "https://bedrock-runtime.us-east-1.amazonaws.com"
            )
            == _computer_toolset.PLATFORM_AWS_BEDROCK
        )

    def test_aws_bedrock_mantle(self):
        assert (
            _computer_toolset.classify_platform(
                "https://bedrock-mantle.us-east-1.api.aws/anthropic"
            )
            == _computer_toolset.PLATFORM_AWS_BEDROCK
        )

    def test_microsoft_foundry(self):
        assert (
            _computer_toolset.classify_platform(
                "https://example-resource.services.ai.azure.com/anthropic/"
            )
            == _computer_toolset.PLATFORM_MICROSOFT_FOUNDRY
        )

    def test_other_aws_surface_is_claude_platform_on_aws(self):
        assert (
            _computer_toolset.classify_platform("https://something-else.amazonaws.com")
            == _computer_toolset.PLATFORM_AWS_CLAUDE_PLATFORM
        )

    def test_unknown_custom_host(self):
        assert (
            _computer_toolset.classify_platform("https://my-proxy.example.com")
            == _computer_toolset.PLATFORM_UNKNOWN
        )

    @pytest.mark.parametrize(
        "base_url",
        [
            "not-a-url",
            "   ",
            "://missing-scheme",
            "https://",
            "amazonaws.com",  # scheme-less: no parseable hostname
        ],
    )
    def test_malformed_non_empty_base_url_is_unknown_not_first_party(self, base_url):
        """A non-empty base_url that is malformed or has no parseable
        hostname must NEVER be treated as first-party -- only a genuinely
        unset/empty base_url gets that default."""
        assert _computer_toolset.classify_platform(base_url) == (
            _computer_toolset.PLATFORM_UNKNOWN
        )

    def test_unparseable_ipv6_url_is_unknown(self):
        """urlsplit itself raises ValueError on some malformed URLs (e.g. an
        unterminated IPv6 host literal) -- the classifier must catch that
        and fail safe to unknown, not propagate the exception or default to
        first-party."""
        assert (
            _computer_toolset.classify_platform("https://[invalid")
            == _computer_toolset.PLATFORM_UNKNOWN
        )

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://evilanthropic.com",
            "https://anthropic.com.evil.com",
            "https://notanthropic.com",
        ],
    )
    def test_anthropic_com_lookalike_is_not_first_party(self, base_url):
        """A substring match (``"anthropic.com" in host``) would wrongly
        classify these as first-party; exact domain/suffix matching must
        not."""
        assert (
            _computer_toolset.classify_platform(base_url)
            != _computer_toolset.PLATFORM_FIRST_PARTY
        )

    def test_googleapis_com_lookalike_is_not_google_vertex(self):
        assert (
            _computer_toolset.classify_platform("https://evilgoogleapis.com")
            != _computer_toolset.PLATFORM_GOOGLE_VERTEX
        )

    def test_azure_foundry_lookalike_is_not_microsoft_foundry(self):
        assert (
            _computer_toolset.classify_platform(
                "https://services.ai.azure.com.evil.com"
            )
            != _computer_toolset.PLATFORM_MICROSOFT_FOUNDRY
        )

    def test_bedrock_prefix_on_non_aws_host_is_not_bedrock(self):
        """The ``bedrock-runtime.``/``bedrock-mantle.`` prefix alone is not
        enough -- the host must also actually end in an AWS domain."""
        assert (
            _computer_toolset.classify_platform("https://bedrock-runtime.evil.com")
            != _computer_toolset.PLATFORM_AWS_BEDROCK
        )

    def test_bedrock_prefix_on_amazonaws_lookalike_is_not_bedrock(self):
        assert (
            _computer_toolset.classify_platform(
                "https://bedrock-runtime.us-east-1.evilamazonaws.com"
            )
            != _computer_toolset.PLATFORM_AWS_BEDROCK
        )

    def test_amazonaws_lookalike_is_not_claude_platform_on_aws(self):
        assert (
            _computer_toolset.classify_platform("https://evilamazonaws.com")
            != _computer_toolset.PLATFORM_AWS_CLAUDE_PLATFORM
        )

    def test_resolve_platform_computer_type_first_party(self):
        computer_type, label = _computer_toolset.resolve_platform_computer_type(None)
        assert computer_type == _computer_toolset.TOOLSET_TYPE
        assert label == _computer_toolset.PLATFORM_FIRST_PARTY

    def test_resolve_platform_computer_type_bedrock(self):
        computer_type, label = _computer_toolset.resolve_platform_computer_type(
            "https://bedrock-runtime.us-west-2.amazonaws.com"
        )
        assert computer_type == "computer_20251124"
        assert label == _computer_toolset.PLATFORM_AWS_BEDROCK

    def test_resolve_platform_computer_type_foundry_unsupported(self):
        computer_type, label = _computer_toolset.resolve_platform_computer_type(
            "https://r.services.ai.azure.com/anthropic/"
        )
        assert computer_type is None
        assert label == _computer_toolset.PLATFORM_MICROSOFT_FOUNDRY


class TestOpus55PlatformResolvedWireForm:
    """Exact wire form/header for each supported transport."""

    def test_default_first_party_sends_toolset_no_legacy_header(self):
        provider = _make_provider()
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"] == [
            {
                "type": "computer_toolset_20260801",
                "configs": {"zoom": {"enabled": False}},
            }
        ]
        beta = (params.get("extra_headers") or {}).get("anthropic-beta", "")
        assert "computer-use-2025-11-24" not in beta

    def test_google_vertex_base_url_sends_toolset(self):
        provider = _make_provider(
            base_url=(
                "https://claude.googleapis.com/v1alpha/projects/p/locations/"
                "global/workspaces/w/invoke"
            )
        )
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"][0]["type"] == "computer_toolset_20260801"

    def test_bedrock_base_url_sends_legacy_type_with_header(self):
        provider = _make_provider(
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com"
        )
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"][0]["type"] == "computer_20251124"
        beta = (params.get("extra_headers") or {}).get("anthropic-beta", "")
        assert "computer-use-2025-11-24" in beta

    def test_bedrock_mantle_base_url_sends_legacy_type_with_header(self):
        provider = _make_provider(
            base_url="https://bedrock-mantle.us-east-1.api.aws/anthropic"
        )
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"][0]["type"] == "computer_20251124"
        beta = (params.get("extra_headers") or {}).get("anthropic-beta", "")
        assert "computer-use-2025-11-24" in beta

    def test_anthropic_base_url_env_used_when_config_unset(self, monkeypatch):
        monkeypatch.setenv(
            "ANTHROPIC_BASE_URL", "https://bedrock-runtime.eu-west-1.amazonaws.com"
        )
        provider = _make_provider()  # no config base_url
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"][0]["type"] == "computer_20251124"


class TestOpus55PlatformUnsupportedFailsSafe:
    def test_microsoft_foundry_raises_before_dispatch(self):
        provider = _make_provider(
            base_url="https://example-resource.services.ai.azure.com/anthropic/"
        )
        create = _stub_create(provider)
        with pytest.raises(KernelInvalidRequestError) as excinfo:
            asyncio.run(provider.complete(_computer_tool_request()))
        assert create.await_count == 0
        assert "claude-opus-5-5" in str(excinfo.value)

    def test_claude_platform_on_aws_raises_before_dispatch(self):
        provider = _make_provider(base_url="https://something-else.amazonaws.com")
        create = _stub_create(provider)
        with pytest.raises(KernelInvalidRequestError):
            asyncio.run(provider.complete(_computer_tool_request()))
        assert create.await_count == 0

    def test_unknown_host_raises_before_dispatch(self):
        provider = _make_provider(base_url="https://my-proxy.example.com")
        create = _stub_create(provider)
        with pytest.raises(KernelInvalidRequestError):
            asyncio.run(provider.complete(_computer_tool_request()))
        assert create.await_count == 0

    def test_no_computer_tool_declared_unsupported_platform_does_not_raise(self):
        """The fail-fast guard only fires when a native computer tool is
        actually declared -- an ordinary request to an unsupported platform
        must not be affected."""
        provider = _make_provider(base_url="https://my-proxy.example.com")
        request = ChatRequest(messages=[Message(role="user", content="hi")])
        params, _ = _run(provider, request)
        assert "tools" not in params


class TestOpus55PlatformOverrideWins:
    def test_explicit_override_wins_over_unsupported_platform(self):
        provider = _make_provider(
            base_url="https://example-resource.services.ai.azure.com/anthropic/",
            computer_use_tool_type="computer_20251124",
        )
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"][0]["type"] == "computer_20251124"

    def test_explicit_override_wins_over_bedrock_default(self):
        provider = _make_provider(
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            computer_use_tool_type="computer_toolset_20260801",
        )
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"][0]["type"] == "computer_toolset_20260801"


def _toolset_tool_request() -> ChatRequest:
    """A caller declares the toolset shape directly (not the legacy
    computer_20251124 ToolSpec that _computer_tool_request builds)."""
    tool = ToolSpec(
        name="computer",
        parameters={},
        type="computer_toolset_20260801",
        configs={"zoom": {"enabled": False}},
    )
    return ChatRequest(messages=[Message(role="user", content="hi")], tools=[tool])


class TestOpus55CallerDeclaredToolsetCannotSlipToLegacyTarget:
    """A caller-declared computer_toolset_20260801 tool must never reach a
    resolved legacy computer_* target (Bedrock, or an explicit legacy
    override) unchanged: the toolset schema carries no
    display_width_px/display_height_px, so there is nothing lossless to
    downgrade to, and Bedrock would reject the resulting request anyway."""

    def test_toolset_shape_raises_on_bedrock_resolved_platform(self):
        provider = _make_provider(
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com"
        )
        create = _stub_create(provider)
        with pytest.raises(KernelInvalidRequestError) as excinfo:
            asyncio.run(provider.complete(_toolset_tool_request()))
        assert create.await_count == 0
        assert "computer_toolset_20260801" in str(excinfo.value)

    def test_toolset_shape_raises_on_explicit_legacy_override(self):
        """Even on the first-party platform (which would otherwise accept
        the toolset shape), an explicit computer_use_tool_type override to a
        legacy type must still reject a caller-declared toolset tool rather
        than silently forwarding it unchanged to a target that expects the
        legacy shape."""
        provider = _make_provider(computer_use_tool_type="computer_20251124")
        create = _stub_create(provider)
        with pytest.raises(KernelInvalidRequestError):
            asyncio.run(provider.complete(_toolset_tool_request()))
        assert create.await_count == 0

    def test_legacy_shape_still_translates_normally_on_bedrock(self):
        """Regression guard: the raise above must be scoped to a
        caller-declared TOOLSET_TYPE tool -- an ordinary legacy tool
        declaration on Bedrock keeps working exactly as before."""
        provider = _make_provider(
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com"
        )
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"][0]["type"] == "computer_20251124"


class TestOpus5UnaffectedByPlatform:
    """Do not change earlier model behavior: Opus 5 always used
    computer_20251124 regardless of base_url, and must continue to."""

    def test_opus_5_ignores_platform_and_keeps_legacy_type_on_bedrock_url(self):
        provider = _make_provider(
            model="claude-opus-5",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        )
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"][0]["type"] == "computer_20251124"

    def test_opus_5_ignores_platform_and_keeps_legacy_type_on_unknown_url(self):
        provider = _make_provider(
            model="claude-opus-5", base_url="https://my-proxy.example.com"
        )
        params, _ = _run(provider, _computer_tool_request())
        assert params["tools"][0]["type"] == "computer_20251124"
