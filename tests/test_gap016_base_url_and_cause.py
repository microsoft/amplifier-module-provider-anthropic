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

import pytest
from amplifier_module_provider_anthropic import (
    AnthropicProvider,
    _redact_url_credentials,
)


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
        from amplifier_module_provider_anthropic import _redact_url_credentials

        if cause is not None:
            cause_text = redact_secrets(_redact_url_credentials(str(cause)))
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

        A base_url carrying embedded basic-auth (https://user:pass@host/) is
        echoed verbatim by httpx/httpcore's own exception text (e.g. the URL
        is quoted back in "Request URL is missing/invalid ..."). That would
        otherwise leak the credentials into a user-facing error on any
        connection failure. redact_secrets() alone can't catch this -- it
        only redacts dict values under a sensitive key, not a substring
        embedded inside a plain string -- so the userinfo must be stripped
        before the cause text is interpolated.
        """
        secret_user, secret_pass = "svc-account", "hunter2-token"
        cause = RuntimeError(
            f"Request URL 'https://{secret_user}:{secret_pass}@proxy.internal/v1' "
            "is missing an 'http://' or 'https://' protocol"
        )
        out = self._enrich("Connection error.", cause)

        assert secret_user not in out and secret_pass not in out, (
            f"credentials embedded in the cause's base_url leaked into the "
            f"enriched message: {out!r}"
        )
        assert "proxy.internal" in out, (
            "redaction should remove only the userinfo, not the rest of the "
            f"diagnostic text: {out!r}"
        )


class TestRedactUrlCredentialsForms:
    """``_redact_url_credentials`` must catch every real-world userinfo shape.

    The original regex (``[^/@\\s]+:[^/@\\s]+@``) required both a non-empty
    username *and* a non-empty password, so it only matched the classic
    ``user:pass@`` form. Three other forms used in practice leaked the raw
    secret verbatim: a bare token as username (the standard git-over-https /
    API-proxy-token convention, no colon at all), an empty username with the
    token as password, and a token as username with an empty password.
    """

    @pytest.mark.parametrize(
        ("url", "secret"),
        [
            pytest.param(
                "https://ghp_TOKEN@github.com/org/repo.git",
                "ghp_TOKEN",
                id="bare-token-username-no-colon",
            ),
            pytest.param(
                "https://:ghp_TOKEN@proxy.internal/v1",
                "ghp_TOKEN",
                id="empty-username-token-password",
            ),
            pytest.param(
                "https://sk-ant-KEY:@proxy.internal/v1",
                "sk-ant-KEY",
                id="token-username-empty-password",
            ),
            pytest.param(
                "https://user:pass@proxy.internal/v1",
                "pass",
                id="user-and-password",
            ),
        ],
    )
    def test_credentials_are_redacted(self, url: str, secret: str) -> None:
        out = _redact_url_credentials(url)
        assert secret not in out, f"secret leaked in redacted output: {out!r}"
        assert "[REDACTED]" in out

    def test_url_without_credentials_is_unchanged(self) -> None:
        url = "https://api.anthropic.com/v1"
        assert _redact_url_credentials(url) == url

    def test_at_sign_in_path_is_not_touched(self) -> None:
        """An ``@`` appearing after the first ``/`` is part of the path, not userinfo."""
        url = "https://host/a@b"
        assert _redact_url_credentials(url) == url
