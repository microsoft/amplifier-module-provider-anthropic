"""Provider-owned image wire/diagnostic boundaries.

This module knows Anthropic content locations, not stores, paths or ref authority.
It never parses strings or traverses arbitrary tool inputs/opaque reasoning.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import weakref
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
from typing import Any

from amplifier_core.llm_errors import InvalidRequestError
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
import httpx2
from pydantic import BaseModel


NAMESPACE = "experimental.media.v1"
# Explicit model identities from the vendor's vision supportedModels catalogue,
# retrieved 2026-11-13. No family/prefix inference for unknown future models.
# https://platform.claude.com/docs/en/build-with-claude/vision.md
VISION_MODELS = frozenset({
    "claude-fable-5-2", "claude-mythos-5-2", "claude-fable-5-1",
    "claude-mythos-5-1", "claude-fable-5", "claude-mythos-5",
    "claude-mythos-preview", "claude-opus-5-5", "claude-opus-5",
    "claude-sonnet-5-5", "claude-sonnet-5", "claude-haiku-5-5",
    "claude-haiku-4-5-20251001", "claude-opus-4-8", "claude-opus-4-7",
    "claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
})
_HAS_NATIVE_TRACING = "open_telemetry" in inspect.signature(AsyncAnthropic).parameters


NOT_RETAINED = (
    "[Image transport not retained in diagnostics; no retrievable pixel reference]"
)


def _refuse_connection() -> None:
    raise InvalidRequestError(
        "Prepared media SDK connection changed or is not qualified; request not sent. "
        "Prepare and fit a new authorized request.",
        provider="anthropic", retryable=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _method_key(value: Any) -> tuple[int, int]:
    return id(getattr(value, "__self__", None)), id(getattr(value, "__func__", value))


def _transport_state(transport: Any) -> tuple[Any, ...] | None:
    if transport is None:
        return None
    if type(transport) not in (httpx2.AsyncHTTPTransport, httpx2.MockTransport):
        _refuse_connection()
    execution = {
        name for base in type(transport).__mro__
        for name, value in vars(base).items() if callable(value)
    }
    if execution.intersection(vars(transport)):
        # A production/mounted transport wrapper runs AFTER HTTP client guards.
        # Refuse before it can observe the private request, not after sending.
        _refuse_connection()
    if type(transport) is httpx2.AsyncHTTPTransport:
        if getattr(getattr(transport, "_pool", None), "_retries", None) != 0:
            _refuse_connection()
    return (
        id(transport),
        _method_key(transport.handle_async_request),
        _method_key(transport.aclose),
    )


def _connection(provider: Any) -> tuple[Any, ...]:
    if provider._client is None and provider._api_key is None:
        return ("unconfigured",)
    client = provider.client
    if not isinstance(client, AsyncAnthropic):
        # Legacy unit/integration adapters do not establish real SDK qualification.
        return (id(client),)
    http = client._client
    if (
        http.is_closed
        or type(client.api_key) is not str or not client.api_key
        or getattr(client, "credentials", None) is not None
        or getattr(client, "_token_cache", None) is not None
        or getattr(client, "_custom_auth", None) is not None
        or getattr(client, "_middleware", ())
        or getattr(client, "_open_telemetry", {}).get("traces", {}).get(
            "content_mode", "metadata_only"
        ) != "metadata_only"
        or http.auth is not None
        or any(http.event_hooks.values())
        or client.base_url.username or client.base_url.password
        or logging.getLogger("anthropic._base_client").isEnabledFor(logging.DEBUG)
    ):
        # Without a post-refresh/post-redirect witness these paths cannot safely
        # bind selected account identity to the physical request.
        _refuse_connection()
    return (
        id(client), id(http), _transport_state(http._transport),
        tuple((id(pattern), _transport_state(transport))
              for pattern, transport in http._mounts.items()),
        _digest({
            "key": client.api_key, "token": client.auth_token,
            "url": str(client.base_url),
            "headers": client._custom_headers, "query": client._custom_query,
            "http_headers": dict(http.headers), "http_params": str(http.params),
            "cookies": list(http.cookies.items()),
            "redirects": http.follow_redirects,
            "mounts": [(str(k), id(v)) for k, v in http._mounts.items()],
            "tracing": getattr(client, "_open_telemetry", {}),
        }),
        *(_method_key(getattr(client, name)) for name in (
            "_prepare_options", "_prepare_request", "_build_request",
            "post", "request",
        )),
        *(_method_key(getattr(http, name)) for name in ("send", "_send_single_request")),
    )


class _ConnectionScope:
    def __init__(self, provider: Any, request: Any):
        self.provider, self.request = provider, request
        self.connection: tuple[Any, ...] | None = None
        self.transport_client: AsyncAnthropic | None = None
        self.purpose: str | None = None
        self.options: dict[str, Any] = {}
        self.expected_headers: dict[str, str] | None = None

    def check(self) -> None:
        actual = _connection(self.provider)
        if self.connection is not None and actual != self.connection:
            _refuse_connection()
        self.connection = actual


class _BoundTransport:
    """Last non-suspending handoff to the already accepted HTTP transport."""
    def __init__(self, scope: _ConnectionScope):
        self.scope = scope

    async def handle_async_request(self, request):
        self.scope.check()
        original = self.scope.provider.client
        if (
            request.url.scheme != original.base_url.scheme
            or request.url.host != original.base_url.host
            or request.url.port != original.base_url.port
            or request.url.username or request.url.password
        ):
            _refuse_connection()
        expected_headers = self.scope.expected_headers
        if expected_headers is None:
            _refuse_connection()
        for key in ("host", "x-api-key", "authorization", "anthropic-organization-id",
                    "anthropic-workspace-id"):
            if request.headers.get(key) != expected_headers.get(key):
                _refuse_connection()
        if self.scope.purpose is not None:
            validate_attempt(
                self.scope.provider.coordinator, self.scope.request,
                self.scope.purpose, self.scope.options,
            )
        transport = original._client._transport_for_url(request.url)
        return await transport.handle_async_request(request)

    async def aclose(self):
        # The original client owns the pool; closing this private facade must
        # never close a shared provider connection.
        pass


class _BoundAnthropic(AsyncAnthropic):
    _media_binding: _ConnectionScope

    async def _copy_and_prepare(self, options):
        # Includes the SDK's first-use platform-thread scheduling window.
        self._media_binding.check()
        result = await super()._copy_and_prepare(options)
        self._media_binding.check()
        return result

    async def _prepare_options(self, options):
        self._media_binding.check()
        result = await super()._prepare_options(options)
        self._media_binding.check()
        # Capture the actual SDK's authorized request-level header layering
        # after assembly, not just constructor defaults.
        expected = dict(self._client.headers)
        expected.update(self._build_headers(result))
        self._media_binding.expected_headers = {
            key.lower(): value for key, value in expected.items()
        }
        self._media_binding.expected_headers.setdefault(
            "host", self.base_url.netloc.decode("ascii")
        )
        return result

    async def _prepare_request(self, request):
        self._media_binding.check()
        await super()._prepare_request(request)
        self._media_binding.check()


def bound_transport_client(provider: Any) -> Any:
    scope = _CONNECTION_SCOPE.get()
    if scope is None or not isinstance(provider.client, AsyncAnthropic):
        return None
    scope.check()
    if scope.transport_client is None:
        original = provider.client
        # Independent immutable SDK auth/options and observer-free HTTP facade.
        # Keep real SDK count/create/raw/stream serialization; share ONLY the
        # selected bottom transport, behind the final private binding guard.
        client = _BoundAnthropic(
            api_key=original.api_key, auth_token=original.auth_token,
            base_url=original.base_url, default_headers=dict(original._custom_headers),
            default_query=dict(original._custom_query), max_retries=0,
            timeout=original.timeout,
            **({"open_telemetry": False} if _HAS_NATIVE_TRACING else {}),
            http_client=DefaultAsyncHttpxClient(
                transport=_BoundTransport(scope), trust_env=False, follow_redirects=False,
                headers=dict(original._client.headers),
                params=original._client.params, cookies=original._client.cookies,
            ),
        )
        # SDK constructors also read ANTHROPIC_CUSTOM_HEADERS even with
        # explicit credentials. Do not re-admit ambient routing after fit.
        client._custom_headers = dict(original._custom_headers)
        client._media_binding = scope
        scope.transport_client = client
    return scope.transport_client


_CONNECTION_SCOPE: ContextVar[_ConnectionScope | None] = ContextVar(
    "anthropic_media_connection", default=None
)


def bind_media_connection(operation):
    """Bind before the first await, including uncounted prepared requests.

    Retain only a weak original-request association and opaque connection digest
    between fit and send; never persist credentials or pixel transport values.
    The host remains authoritative for request/options/account/graph lifetime.
    """
    @wraps(operation)
    async def bound(provider, request, *args, **kwargs):
        if not (has_refs(request) or has_capability(provider.coordinator)):
            # A plain nested completion is not part of its caller's private
            # capability, including when it targets another provider/account.
            token = _CONNECTION_SCOPE.set(None)
            try:
                return await operation(provider, request, *args, **kwargs)
            finally:
                _CONNECTION_SCOPE.reset(token)
        scope = _ConnectionScope(provider, request)
        token = _CONNECTION_SCOPE.set(scope)
        try:
            scope.check()
            records = getattr(provider, "_media_connections", None)
            if records is None:
                records = provider._media_connections = {}
            previous = records.get(id(request))
            if previous is not None and previous[0]() is request and previous[1] != scope.connection:
                _refuse_connection()
            id_key = id(request)
            records[id_key] = (
                weakref.ref(request, lambda _ref: records.pop(id_key, None)),
                scope.connection,
            )
            result = await operation(provider, request, *args, **kwargs)
            # Count results are still pre-send authority. A completed transport
            # must not be relabeled "not sent" by a later response hook; the
            # original retained binding will refuse the next use instead.
            if operation.__name__ == "request_budget":
                scope.check()
            return result
        finally:
            try:
                if scope.transport_client is not None:
                    await scope.transport_client.close()
                    scope.transport_client = None
            finally:
                _CONNECTION_SCOPE.reset(token)
    return bound


def has_media(request: Any) -> bool:
    """Recognize only canonical media positions, never arbitrary JSON text."""
    for message in request.messages:
        metadata = getattr(message, "metadata", None)
        if isinstance(metadata, dict) and NAMESPACE in metadata:
            return True
        content = getattr(message, "content", None)
        if isinstance(content, list):
            for raw in content:
                block = raw.model_dump() if hasattr(raw, "model_dump") else raw
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        return True
                    if block.get("type") == "tool_result":
                        nested = block.get("output", block.get("content"))
                        if isinstance(nested, list) and any(
                            isinstance(part, dict) and part.get("type") == "image"
                            for part in nested
                        ):
                            return True
    return False


def has_refs(request: Any) -> bool:
    return any(
        isinstance(getattr(message, "metadata", None), dict)
        and NAMESPACE in message.metadata for message in request.messages
    )


def has_capability(coordinator: Any) -> bool:
    getter = getattr(coordinator, "get_capability", None)
    capability = getter(NAMESPACE) if callable(getter) else None
    return capability is not None and getattr(capability, "version", None) == 1


def tool_result_content(content: Any, *, allow_images: bool = True) -> Any:
    """Normalize Core text/image parts at the original tool-result position."""
    if not isinstance(content, list):
        return deepcopy(content)
    result = []
    for raw in content:
        block = raw.model_dump() if hasattr(raw, "model_dump") else raw
        if not isinstance(block, dict):
            raise InvalidRequestError(
                "Tool-result content requires structured text/image parts.",
                provider="anthropic",
                status_code=400,
            )
        if block.get("type") == "text":
            result.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            if not allow_images:
                result.append({
                    "type": "text",
                    "text": "[Image unavailable: selected model vision support is unknown; no pixels supplied]",
                })
                continue
            source = block.get("source")
            if not isinstance(source, dict) or source.get("type") != "base64":
                # Forward media-v1 is verified static PNG/JPEG, never a URL fetch.
                result.append({
                    "type": "text",
                    "text": "[Image unavailable: unsupported image source; no pixels supplied]",
                })
                continue
            if (
                source.get("media_type") not in {"image/png", "image/jpeg"}
                or not isinstance(source.get("data"), str)
                or not source["data"]
            ):
                raise InvalidRequestError(
                    "Tool-result image requires verified PNG/JPEG base64 transport.",
                    provider="anthropic",
                    status_code=400,
                )
            result.append({
                "type": "image",
                "source": {
                    key: source[key] for key in ("type", "media_type", "data")
                },
            })
        else:
            # Other provider-owned blocks retain their existing semantics.
            result.append(deepcopy(block))
    return result


def _diagnostic_content(content: Any) -> Any:
    if not isinstance(content, list):
        return deepcopy(content)
    result = []
    for block in content:
        if not isinstance(block, dict):
            result.append(deepcopy(block))
        elif block.get("type") == "image":
            # Omission is deliberately not a fake digest/ref that a reader can
            # mistake for retained pixels. Drop the WHOLE image transport copy.
            result.append({"type": "text", "text": NOT_RETAINED})
        elif block.get("type") == "tool_result":
            copied = deepcopy(block)
            for key in ("content", "output"):
                if key in copied:
                    copied[key] = _diagnostic_content(copied[key])
            result.append(copied)
        elif block.get("type") == "document":
            copied = deepcopy(block)
            source = copied.get("source")
            if isinstance(source, dict) and source.get("type") == "content":
                source["content"] = _diagnostic_content(source.get("content"))
            result.append(copied)
        else:
            result.append(deepcopy(block))
    return result


def diagnostic_copy(payload: Any) -> dict[str, Any]:
    """Omit known image wire fields BEFORE a Rust-backed hooks.emit call."""
    if isinstance(payload, BaseModel):
        # Pydantic warnings can include repr(raw image blocks), before copying.
        payload = payload.model_dump(warnings=False)
    elif callable(getattr(payload, "model_dump", None)):
        # Legacy test/host adapters have a zero-argument model_dump contract.
        payload = payload.model_dump()
    copied = deepcopy(payload)
    messages = copied.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                message["content"] = _diagnostic_content(message["content"])
    for key in ("content", "system"):
        if key in copied:
            copied[key] = _diagnostic_content(copied[key])
    extra = copied.get("extra_body")
    if isinstance(extra, dict):
        copied["extra_body"] = diagnostic_copy(extra)
    return copied


def safe_error_text(text: str, params: dict[str, Any]) -> str:
    """Remove exact known pixel transport values if an SDK echoes the input."""
    def scrub(content: Any) -> None:
        nonlocal text
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                source = block.get("source")
                if isinstance(source, dict):
                    for key in ("data", "url"):
                        value = source.get(key)
                        if isinstance(value, str) and value:
                            text = text.replace(value, NOT_RETAINED)
            elif block.get("type") == "tool_result":
                scrub(block.get("content"))
            elif block.get("type") == "document":
                source = block.get("source")
                if isinstance(source, dict) and source.get("type") == "content":
                    scrub(source.get("content"))

    for envelope in (params, params.get("extra_body")):
        if not isinstance(envelope, dict):
            continue
        for message in envelope.get("messages", []):
            if isinstance(message, dict):
                scrub(message.get("content"))
        scrub(envelope.get("system"))
    return text


def validate_attempt(coordinator: Any, request: Any, purpose: str, options: Any) -> None:
    """Call the host's private request witness; never turn refusal into retry.

    Legacy unwrapped requests retain legacy behavior without a media capability.
    With a real v1 capability the exact projected Core object is the private key.
    """
    scope = _CONNECTION_SCOPE.get()
    if scope is not None:
        if scope.request is not request:
            _refuse_connection()
        scope.check()
        scope.purpose, scope.options = purpose, dict(options)
    getter = getattr(coordinator, "get_capability", None)
    capability = getter(NAMESPACE) if callable(getter) else None
    if capability is None or getattr(capability, "version", None) != 1:
        if has_refs(request):
            raise InvalidRequestError(
                "Session image references require compatible private preparation; "
                "no pixels supplied and request not sent.",
                provider="anthropic", retryable=False,
            )
        return
    validator = getattr(capability, "validate", None)
    if not callable(validator):
        raise InvalidRequestError(
            "Media validation unavailable; request not sent.", provider="anthropic"
        )
    try:
        result = validator(request, purpose=purpose, request_options=dict(options))
    except Exception:
        # Policy exceptions must not enter provider retry/backoff/overflow or
        # escape with arbitrary byte-bearing exception text.
        raise InvalidRequestError(
            "Media projection expired, changed or is no longer authorized; "
            "request not sent. Prepare and fit a new authorized request.",
            provider="anthropic", retryable=False,
        ) from None
    if not isinstance(result, dict) or result.get("valid") is not True:
        raise InvalidRequestError(
            "Media validation refused; request not sent.", provider="anthropic",
            retryable=False,
        )
    if scope is not None:
        scope.check()