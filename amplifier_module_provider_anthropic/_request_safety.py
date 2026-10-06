"""Generation-only uncertainty policy. Read-only SDK operations keep their policy."""

from anthropic import APIConnectionError, APITimeoutError, _base_client
from amplifier_core.llm_errors import LLMError, LLMTimeoutError


UNKNOWN_MESSAGE = (
    "Provider wait ended without a confirmed result. The request may have been "
    "accepted; no automatic replacement request was sent."
)


class RequestOutcomeUnknownError(LLMError):
    """An admitted generation whose remote outcome cannot be established."""

    request_outcome = "unknown"
    effects = "may_have_occurred"

    def __init__(self, *, model: str, status_code: int | None = None):
        super().__init__(
            UNKNOWN_MESSAGE, provider="anthropic", model=model,
            status_code=status_code, retryable=False,
        )


class LocalRequestError(LLMError):
    """Known local failure, without exposing arbitrary exception text."""

    def __init__(self, *, model: str, received: bool = False):
        message = (
            "Provider received a result but local processing failed. "
            "No automatic replacement request was sent."
            if received else
            "Provider request failed locally before dispatch. No request was sent."
        )
        super().__init__(message, provider="anthropic", model=model, retryable=False)
        self.request_outcome = "received" if received else "not_dispatched"
        self.effects = "occurred" if received else "none"


def proved_local_url(error: Exception) -> bool:
    """Only immediate typed URL failures, not wrapper text or HTTP provenance."""
    transport = getattr(_base_client, "httpx2", None) or _base_client.httpx
    if type(error) is transport.InvalidURL:
        return True
    return (
        type(error) is APIConnectionError
        and type(error.__cause__) is transport.UnsupportedProtocol
    )


def unknown_timeout(model: str) -> LLMTimeoutError:
    error = LLMTimeoutError(
        UNKNOWN_MESSAGE, provider="anthropic", model=model, retryable=False,
    )
    error.request_outcome = "unknown"
    error.effects = "may_have_occurred"
    return error


def proved_pre_send(error: Exception) -> bool:
    """Accept only the SDK's immediate, typed transport connect/pool cause.

    SDK 1.x wraps the send boundary with `raise ... from err`. A generic wrapper,
    read/write timeout, protocol failure, or text mentioning connect is not proof.
    The SDK's transport import selects its installed httpx/httpx2 family.
    Public export classes can have __module__ rewritten to "anthropic".
    """
    if type(error) not in (APIConnectionError, APITimeoutError):
        return False
    transport = getattr(_base_client, "httpx2", None) or _base_client.httpx
    return type(error.__cause__) in (
        transport.ConnectError, transport.ConnectTimeout, transport.PoolTimeout,
    )


def proved_refusal(error: Exception, *, status: int, kind: str) -> bool:
    """Structured HTTP nonacceptance, never SSE error text or a bare status.

    Anthropic documents 429 rate_limit_error and 529 overloaded_error as request
    refusal responses. Stream errors can arrive after HTTP 200; they are unknown.
    """
    body = getattr(error, "body", None)
    if not isinstance(body, dict) or body.get("type") != "error":
        return False
    detail = body.get("error")
    return (
        getattr(error, "status_code", None) == status
        and isinstance(detail, dict)
        and detail.get("type") == kind
    )