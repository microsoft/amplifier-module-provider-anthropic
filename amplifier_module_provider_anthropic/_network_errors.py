"""Distinguish a lost connection from the SDK's configuration-error wrapper."""

from contextlib import contextmanager
from contextvars import ContextVar
import ssl

import httpx2

try:
    import httpx  # Optional legacy transport accepted by the Anthropic SDK.
except ImportError:
    httpx = None

_response_state = ContextVar("anthropic_response_state", default=None)


async def _received_headers(response):
    state = _response_state.get()
    if state is not None:
        state["received"] = True


def watch_response(http_client):
    clients = (httpx2.AsyncClient,)
    if httpx is not None:
        clients += (httpx.AsyncClient,)
    if isinstance(http_client, clients):
        hooks = http_client.event_hooks.setdefault("response", [])
        hooks[:] = [_received_headers, *(h for h in hooks if h is not _received_headers)]
        state = _response_state.get()
        if state is not None:
            state["client"] = http_client


def observation_confirmed():
    state = _response_state.get()
    client = state.get("client") if state is not None else None
    module = httpx2 if isinstance(client, httpx2.AsyncClient) else httpx
    if module is None or not isinstance(client, module.AsyncClient):
        return False
    hooks = client.event_hooks.get("response", [])
    if not hooks or hooks[0] is not _received_headers:
        return False
    # Custom send pipelines/transports can consume a body before HTTPX hooks.
    for method in ("send", "_send_handling_auth", "_send_handling_redirects", "_send_single_request"):
        if getattr(getattr(client, method, None), "__func__", None) is not getattr(module.AsyncClient, method):
            return False
    transports = [client._transport, *client._mounts.values()]
    return all(t is None or (
        type(t) is module.AsyncHTTPTransport and not getattr(t._pool, "_http2", True)
    ) for t in transports)


def certificate_failure(error):
    seen = set()
    pending = [error]
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        if isinstance(error, ssl.SSLCertVerificationError):
            return True
        pending.extend((error.__cause__, error.__context__))
    return False


def can_promote(error):
    return observation_confirmed() and not response_received() and not certificate_failure(error)


@contextmanager
def observe_response():
    # A mutable cell propagates observations from asyncio.wait_for's task.
    token = _response_state.set({"received": False, "client": None})
    try:
        yield
    finally:
        _response_state.reset(token)


def response_received():
    state = _response_state.get()
    return state is not None and state["received"]


def is_connection_failure(error):
    connection_errors = (httpx2.NetworkError, httpx2.RemoteProtocolError)
    if httpx is not None:
        connection_errors += (httpx.NetworkError, httpx.RemoteProtocolError)
    # UnsupportedProtocol and InvalidURL deliberately do not qualify.
    return isinstance(error.__cause__, connection_errors)