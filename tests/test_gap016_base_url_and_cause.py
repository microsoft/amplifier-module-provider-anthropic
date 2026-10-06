"""Regression tests for GAP-016 (empty base_url) and private cause preservation.

An independent review found that **neither** behavioural change in this PR had
a test: reverting either one would have left the suite green. Both contracts
are pinned here.

Contract 1 -- an empty ``base_url`` must be treated as "not configured".
``settings.yaml`` commonly stores this as ``base_url: ${ANTHROPIC_BASE_URL}``,
and ``expand_env_vars()`` substitutes an *unset* variable with ``""`` rather
than ``None``. Passed to ``AsyncAnthropic(base_url="")``, httpx raises
``UnsupportedProtocol``, which the SDK re-wraps as a generic
``APIConnectionError("Connection error.")`` on *every* call.

Contract 2 -- unknown-stage connection failures retain the SDK's chained cause
for debugging, but never mirror its text into the fixed public error message
or automatically send a replacement generation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import (
    AnthropicProvider,
    _redact_url_credentials,
)
from amplifier_module_provider_anthropic._request_safety import (
    UNKNOWN_MESSAGE,
    RequestOutcomeUnknownError,
)
from tests._helpers import FakeCoordinator


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


class TestPrivateCausePreservation:
    """Contract 2: exercise actual provider translation, never a mirrored helper."""

    @staticmethod
    def _capture_error(
        error_msg: str, cause: BaseException | None
    ) -> RequestOutcomeUnknownError:
        coordinator = FakeCoordinator()
        provider = AnthropicProvider(
            "fixture",
            {
                "use_streaming": False,
                "max_retries": 3,
                "fallback_on_overload": True,
            },
            coordinator=coordinator,
        )
        sdk_error = anthropic.APIConnectionError(
            message=error_msg, request=MagicMock()
        )
        sdk_error.__cause__ = cause
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=sdk_error
        )
        request = ChatRequest(messages=[Message(role="user", content="hello")])

        with pytest.raises(RequestOutcomeUnknownError) as exc_info:
            asyncio.run(provider.complete(request))

        error = exc_info.value
        assert str(error) == UNKNOWN_MESSAGE
        assert error.retryable is False
        assert error.request_outcome == "unknown"
        assert error.effects == "may_have_occurred"
        assert error.__cause__ is sdk_error
        assert error.__cause__.__cause__ is cause
        assert provider.client.messages.with_raw_response.create.await_count == 1
        assert "provider:retry" not in coordinator.hooks.emitted_names()
        assert "provider:fallback_open" not in coordinator.hooks.emitted_names()
        return error

    def test_cause_is_private_not_named_in_the_message(self) -> None:
        class UnsupportedProtocol(Exception):
            pass

        cause = UnsupportedProtocol(
            "Request URL is missing an 'http://' or 'https://' protocol"
        )
        error = self._capture_error("Connection error.", cause)
        assert "UnsupportedProtocol" not in str(error)
        assert "missing an 'http://'" not in str(error)

    def test_no_cause_still_uses_fixed_message(self) -> None:
        self._capture_error("Connection error.", None)

    def test_empty_cause_is_preserved_privately(self) -> None:
        cause = ValueError("")
        error = self._capture_error("Connection error.", cause)
        assert "ValueError" not in str(error)

    def test_duplicate_cause_text_is_not_surfaced(self) -> None:
        cause = RuntimeError("already mentioned")
        error = self._capture_error("Failed: already mentioned", cause)
        assert "already mentioned" not in str(error)

    def test_credentials_and_endpoint_in_cause_stay_private(self) -> None:
        """The fixed public message must contain neither credentials nor URL."""
        secret_user, secret_pass = "svc-account", "hunter2-token"
        cause = RuntimeError(
            f"Request URL 'https://{secret_user}:{secret_pass}@proxy.internal/v1' "
            "is missing an 'http://' or 'https://' protocol"
        )
        out = str(self._capture_error("Connection error.", cause))

        assert secret_user not in out and secret_pass not in out, (
            f"credentials embedded in the cause's base_url leaked into the "
            f"public message: {out!r}"
        )
        assert "proxy.internal" not in out


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


class TestRedactUrlCredentialsNoScheme:
    """Credentials must be redacted even when there is no ``scheme://`` at all.

    This is the exact scenario GAP-016 exists to handle: httpx's own
    "Request URL ... is missing an 'http://' or 'https://' protocol" text
    echoes a malformed/missing-protocol ``base_url`` back verbatim, and a
    pattern anchored on a literal ``"://"`` never fires for it -- the one
    input this redaction exists to catch was the one input it didn't catch.
    """

    def test_no_scheme_user_pass_is_redacted(self) -> None:
        """The confirmed leak: user:pass@ with no scheme prefix at all."""
        secret_user, secret_pass = "svc-account", "hunter2-token"
        text = (
            f"Request URL '{secret_user}:{secret_pass}@proxy.internal/v1' "
            "is missing an 'http://' or 'https://' protocol"
        )
        out = _redact_url_credentials(text)

        assert secret_user not in out and secret_pass not in out, (
            f"credentials leaked with no scheme present: {out!r}"
        )
        assert "[REDACTED]" in out
        assert "proxy.internal" in out, (
            f"redaction should remove only the userinfo: {out!r}"
        )

    def test_no_scheme_empty_username_token_password_is_redacted(self) -> None:
        secret = "ghp_TOKEN"
        text = f"URL ':{secret}@proxy.internal/v1' is missing a protocol"
        out = _redact_url_credentials(text)
        assert secret not in out, f"secret leaked: {out!r}"
        assert "[REDACTED]" in out

    def test_no_scheme_token_username_empty_password_is_redacted(self) -> None:
        secret = "sk-ant-KEY"
        text = f"URL '{secret}:@proxy.internal/v1' is missing a protocol"
        out = _redact_url_credentials(text)
        assert secret not in out, f"secret leaked: {out!r}"
        assert "[REDACTED]" in out

    def test_bare_email_is_not_mangled(self) -> None:
        """A bare email's local part has no colon -- it must not be treated

        as leaked credentials. Over-redaction destroys diagnostic value for
        no security benefit: this pins the over-redaction risk instead of
        merely assuming it's handled.
        """
        text = "Contact admin at someone@example.com for help"
        assert _redact_url_credentials(text) == text

    def test_bare_token_no_scheme_is_left_alone_by_design(self) -> None:
        """A colon-less bare token with no scheme (``token@host``) is

        indistinguishable from an email's ``user@host`` shape without scheme
        context. This pins the deliberate design choice to leave it alone
        rather than risk destroying real diagnostic text on a guess -- the
        scheme-present form of the same case is still fully redacted
        (see ``bare-token-username-no-colon`` above).
        """
        text = "Request URL 'token@proxy.internal/v1' is missing a protocol"
        assert _redact_url_credentials(text) == text

    def test_scheme_present_case_is_unaffected(self) -> None:
        """The original, tested, scheme-anchored behaviour must be unchanged."""
        secret = "hunter2-token"
        text = f"https://user:{secret}@proxy.internal/v1"
        out = _redact_url_credentials(text)
        assert secret not in out

    def test_base64_padded_password_no_scheme_is_redacted(self) -> None:
        """Additional leak found during review: a base64-style secret with

        ``+``/``=`` padding characters (a common real-world token shape --
        e.g. a proxy password or bearer token) was left completely unredacted
        in the no-scheme form, because the first character class draft only
        allowed ``[A-Za-z0-9_.~%+-]`` and stopped matching at the unhandled
        ``=``, leaving no valid span reaching ``@`` at all -- not even a
        partial redaction. ``=`` was added to the no-scheme character classes
        to close this.
        """
        secret = "P4ss+word=="
        text = f"URL 'user:{secret}@proxy.internal/v1' missing protocol"
        out = _redact_url_credentials(text)
        assert secret not in out, f"secret leaked: {out!r}"
        assert "[REDACTED]" in out
