"""Distinguish a lost connection from the SDK's configuration-error wrapper."""

from contextlib import contextmanager
from contextvars import ContextVar
import ssl
from weakref import WeakSet

import httpx2

try:
    import httpx  # Optional legacy transport accepted by the Anthropic SDK.
except ImportError:
    httpx = None

_response_state = ContextVar("anthropic_response_state", default=None)
_installed = WeakSet()
_CLIENT_TYPES = (httpx2.AsyncClient,)
if httpx is not None:
    _CLIENT_TYPES += (httpx.AsyncClient,)
try:
    from anthropic._base_client import AsyncHttpxClientWrapper, DefaultAsyncHttpxClient

    _CLIENT_TYPES += (AsyncHttpxClientWrapper, DefaultAsyncHttpxClient)
except (ImportError, AttributeError):
    pass


def _client_module(client):
    if type(client) not in _CLIENT_TYPES:
        return None
    return httpx2 if isinstance(client, httpx2.AsyncClient) else httpx


async def _received_headers(response):
    state = _response_state.get()
    if state is not None:
        state["received"] = True


def watch_response(http_client):
    try:
        if _client_module(http_client) is None:
            return
        hooks = http_client.event_hooks.setdefault("response", [])
        if http_client not in _installed:
            _installed.add(http_client)
            hooks.insert(0, _received_headers)
        state = _response_state.get()
        if state is not None:
            state["client"] = http_client
    except Exception:
        return


def observation_confirmed():
    try:
        state = _response_state.get()
        client = state.get("client") if state is not None else None
        module = _client_module(client)
        if module is None or client not in _installed:
            return False
        hooks = client.event_hooks.get("response", [])
        if not hooks or hooks[0] is not _received_headers:
            return False
        if client.auth is not None or client.event_hooks.get("request"):
            return False
        transports = [client._transport, *client._mounts.values()]
        return all(t is None or (
            type(t) is module.AsyncHTTPTransport and not t._pool._http2
        ) for t in transports)
    except Exception:
        return False


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