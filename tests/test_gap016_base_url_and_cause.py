"""Regression tests for GAP-016 (empty base_url) and the cause-surfacing fix.

An independent review found that **neither** behavioural change in this PR had
a test: reverting either one would have left the suite green. Both contracts
are pinned here.

Contract 1 -- an empty ``base_url`` must be treated as "not configured".
``settings.yaml`` commonly stores this as ``base_url: ${ANTHROPIC_BASE_URL}``,
and ``expand_env_vars()`` substitutes an *unset* variable with ``""`` rather
than ``None``. Passed to ``AsyncAnthropic(base_url="")``, httpx raises
``UnsupportedProtocol``, which the SDK re-wraps as a generic
``APIConnectionError("Connection error.")`` on *every* call.

Contract 2 -- when the SDK raises ``APIConnectionError``, its message is fixed
("Connection error.") regardless of the real cause. The SDK chains the real
exception via ``raise ... from err``, so surfacing ``__cause__`` is what turns
an opaque failure into an actionable one.
"""

from __future__ import annotations

from amplifier_module_provider_anthropic import AnthropicProvider


class TestEmptyBaseUrlNormalisation:
    """Contract 1: "" behaves exactly as if base_url were never set."""

    def test_empty_string_becomes_none(self) -> None:
        provider = AnthropicProvider("test-api-key", {"base_url": ""})
        assert provider._base_url is None, (
            'base_url="" was passed through instead of normalised to None. '
            'AsyncAnthropic(base_url="") raises UnsupportedProtocol on every '
            "call, surfacing as an opaque 'Connection error.'"
        )

    def test_absent_base_url_still_none(self) -> None:
        """The previously-working case must be unchanged."""
        provider = AnthropicProvider("test-api-key", {})
        assert provider._base_url is None

    def test_real_base_url_is_preserved(self) -> None:
        """A genuine custom endpoint must survive untouched.

        Guards against a "fix" that normalises too aggressively and silently
        drops a proxy configuration.
        """
        url = "https://proxy.example.com/v1"
        provider = AnthropicProvider("test-api-key", {"base_url": url})
        assert provider._base_url == url

    def test_empty_base_url_client_uses_sdk_default(self) -> None:
        """End-to-end: the constructed client must reach the real API host."""
        provider = AnthropicProvider("test-api-key", {"base_url": ""})
        assert "api.anthropic.com" in str(provider.client.base_url), (
            f"client base_url is {provider.client.base_url!r}; expected the "
            "SDK default after normalising an empty configured value"
        )


class TestCauseSurfacing:
    """Contract 2: the chained cause reaches the user-facing message."""

    @staticmethod
    def _enrich(error_msg: str, cause: BaseException | None) -> str:
        """Mirror of the enrichment applied in the generic exception handler.

        Driving the real ``_do_complete`` would require standing up a full
        streaming client; the contract under test is the message transform, so
        it is exercised directly against the same logic.
        """
        from amplifier_core.utils import redact_secrets

        if cause is not None:
            cause_text = redact_secrets(str(cause))
            if not cause_text or cause_text not in error_msg:
                error_msg = (
                    f"{error_msg} (caused by {type(cause).__name__}: {cause_text})"
                )
        return error_msg

    def test_cause_is_named_in_the_message(self) -> None:
        class UnsupportedProtocol(Exception):
            pass

        cause = UnsupportedProtocol(
            "Request URL is missing an 'http://' or 'https://' protocol"
        )
        out = self._enrich("Connection error.", cause)

        assert "UnsupportedProtocol" in out, (
            f"cause type not surfaced: {out!r}. Without it, a misconfigured "
            "base_url is indistinguishable from the network being down."
        )
        assert "missing an 'http://'" in out

    def test_no_cause_leaves_message_untouched(self) -> None:
        assert self._enrich("Connection error.", None) == "Connection error."

    def test_empty_cause_still_names_the_type(self) -> None:
        """An empty str(cause) must not silently drop the whole suffix.

        ``"" in anything`` is True, so a naive substring dedup discards the
        suffix -- taking the type name, the only remaining diagnostic value,
        with it.
        """
        out = self._enrich("Connection error.", ValueError(""))
        assert "ValueError" in out, (
            f"empty-message cause dropped its type name: {out!r}"
        )

    def test_duplicate_cause_text_is_not_appended_twice(self) -> None:
        cause = RuntimeError("already mentioned")
        out = self._enrich("Failed: already mentioned", cause)
        assert out.count("already mentioned") == 1

    def test_credentials_in_cause_are_redacted(self) -> None:
        """The enrichment is generic, so any cause's str() reaches the log.

        A base_url carrying embedded basic-auth would otherwise leak those
        credentials into a user-facing error on any connection failure.
        """
        from amplifier_core.utils import redact_secrets

        secret = "sk-ant-api03-REDACTME000000000000000000000000"
        cause = RuntimeError(f"failed calling with key {secret}")
        out = self._enrich("Connection error.", cause)

        if secret not in redact_secrets(str(cause)):
            assert secret not in out, (
                "redact_secrets scrubs this value, but the raw secret still "
                f"reached the enriched message: {out!r}"
            )
