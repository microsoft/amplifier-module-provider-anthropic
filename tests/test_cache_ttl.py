"""Unit tests for the `cache_ttl` config knob (extended 1h prompt caching).

Contract under test:

- Config key ``cache_ttl`` is OFF by default (absent, empty, or None):
  every cache_control block is the plain ``{"type": "ephemeral"}`` and the
  ``extended-cache-ttl-2025-04-11`` beta header is NOT sent.
- Only the exact value ``"1h"`` enables the knob:
  cache_control blocks become ``{"type": "ephemeral", "ttl": "1h"}`` and the
  beta header IS appended.
- Any other value (``"2h"``, ``"5m"``, garbage) behaves exactly as off.

Covered sites (all in AnthropicProvider):
- _apply_tool_cache_control (last tool definition)
- _format_system_with_cache (system content block; contract pointer named it
  ``_format_system_content`` but the real method is ``_format_system_with_cache``)
- _apply_message_cache_control (both list-content and str-content paths)
- _build_request_beta_headers (per-model beta header builder)

No network calls are made.
"""

from amplifier_core.message_models import Message

from amplifier_module_provider_anthropic import AnthropicProvider, ModelCapabilities

EXTENDED_TTL_BETA = "extended-cache-ttl-2025-04-11"
PLAIN = {"type": "ephemeral"}
ONE_HOUR = {"type": "ephemeral", "ttl": "1h"}


def _make_provider(config: dict | None = None) -> AnthropicProvider:
    """Construct a provider directly, following existing test conventions."""
    return AnthropicProvider(api_key="test-key", config=config or {})


def _beta_headers(provider: AnthropicProvider) -> list[str]:
    return provider._build_request_beta_headers(
        model_id="claude-sonnet-4-5-20250929",
        request_caps=ModelCapabilities(family="test"),
        tools_present=False,
        resolved_thinking_type=None,
    )


class TestToolCacheControl:
    """(a) tool cache_control gets ttl iff knob == '1h'."""

    def test_ttl_present_when_knob_on(self):
        provider = _make_provider({"cache_ttl": "1h"})
        tools = [{"name": "alpha"}, {"name": "beta"}]
        result = provider._apply_tool_cache_control(tools)
        assert result[-1]["cache_control"] == ONE_HOUR
        assert "cache_control" not in result[0]

    def test_plain_when_knob_absent(self):
        provider = _make_provider()
        result = provider._apply_tool_cache_control([{"name": "alpha"}])
        assert result[-1]["cache_control"] == PLAIN

    def test_plain_when_knob_bogus(self):
        provider = _make_provider({"cache_ttl": "2h"})
        result = provider._apply_tool_cache_control([{"name": "alpha"}])
        assert result[-1]["cache_control"] == PLAIN


class TestSystemBlockCacheControl:
    """(b) system block gets ttl iff knob == '1h'."""

    def test_ttl_present_when_knob_on(self):
        provider = _make_provider({"cache_ttl": "1h"})
        blocks = provider._format_system_with_cache(
            [Message(role="system", content="You are helpful.")]
        )
        assert blocks is not None
        assert blocks[0]["cache_control"] == ONE_HOUR

    def test_plain_when_knob_absent(self):
        provider = _make_provider()
        blocks = provider._format_system_with_cache(
            [Message(role="system", content="You are helpful.")]
        )
        assert blocks is not None
        assert blocks[0]["cache_control"] == PLAIN

    def test_plain_when_knob_bogus(self):
        provider = _make_provider({"cache_ttl": "5m"})
        blocks = provider._format_system_with_cache(
            [Message(role="system", content="You are helpful.")]
        )
        assert blocks is not None
        assert blocks[0]["cache_control"] == PLAIN


class TestMessageCacheControlListContent:
    """(c) message cache_control ttl iff on -- list-content path."""

    @staticmethod
    def _messages() -> list[dict]:
        return [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    def test_ttl_present_when_knob_on(self):
        provider = _make_provider({"cache_ttl": "1h"})
        result = provider._apply_message_cache_control(self._messages())
        assert result[-1]["content"][-1]["cache_control"] == ONE_HOUR

    def test_plain_when_knob_absent(self):
        provider = _make_provider()
        result = provider._apply_message_cache_control(self._messages())
        assert result[-1]["content"][-1]["cache_control"] == PLAIN

    def test_plain_when_knob_bogus(self):
        provider = _make_provider({"cache_ttl": "garbage"})
        result = provider._apply_message_cache_control(self._messages())
        assert result[-1]["content"][-1]["cache_control"] == PLAIN


class TestMessageCacheControlStrContent:
    """(c) message cache_control ttl iff on -- str-content path."""

    @staticmethod
    def _messages() -> list[dict]:
        return [{"role": "user", "content": "hello there"}]

    def test_ttl_present_when_knob_on(self):
        provider = _make_provider({"cache_ttl": "1h"})
        result = provider._apply_message_cache_control(self._messages())
        block = result[-1]["content"][-1]
        assert block["type"] == "text"
        assert block["text"] == "hello there"
        assert block["cache_control"] == ONE_HOUR

    def test_plain_when_knob_absent(self):
        provider = _make_provider()
        result = provider._apply_message_cache_control(self._messages())
        assert result[-1]["content"][-1]["cache_control"] == PLAIN

    def test_plain_when_knob_bogus(self):
        provider = _make_provider({"cache_ttl": "1H"})
        result = provider._apply_message_cache_control(self._messages())
        assert result[-1]["content"][-1]["cache_control"] == PLAIN


class TestBetaHeader:
    """(d) beta header 'extended-cache-ttl-2025-04-11' appended iff knob == '1h'."""

    def test_header_present_when_knob_on(self):
        provider = _make_provider({"cache_ttl": "1h"})
        assert EXTENDED_TTL_BETA in _beta_headers(provider)

    def test_header_absent_when_knob_absent(self):
        provider = _make_provider()
        assert EXTENDED_TTL_BETA not in _beta_headers(provider)

    def test_header_absent_when_knob_bogus(self):
        provider = _make_provider({"cache_ttl": "2h"})
        assert EXTENDED_TTL_BETA not in _beta_headers(provider)


class TestDefaultOffSemantics:
    """(e) default-off and (f) bogus/empty values normalize to off."""

    def test_absent_key_is_off(self):
        provider = _make_provider()
        assert provider._cache_ttl == ""

    def test_empty_string_is_off(self):
        provider = _make_provider({"cache_ttl": ""})
        assert provider._cache_ttl == ""

    def test_none_value_is_off(self):
        provider = _make_provider({"cache_ttl": None})
        assert provider._cache_ttl == ""

    def test_only_exact_1h_enables(self):
        assert _make_provider({"cache_ttl": "1h"})._cache_ttl == "1h"
        # Non-"1h" values are stored but never treated as enabled anywhere.
        bogus = _make_provider({"cache_ttl": "2h"})
        assert EXTENDED_TTL_BETA not in _beta_headers(bogus)
        tools = bogus._apply_tool_cache_control([{"name": "t"}])
        assert tools[-1]["cache_control"] == PLAIN
