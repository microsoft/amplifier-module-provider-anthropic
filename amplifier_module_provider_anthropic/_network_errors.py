"""Distinguish a lost connection from the SDK's configuration-error wrapper."""

from contextlib import contextmanager
from contextvars import ContextVar

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
        if _received_headers not in hooks:
            hooks.append(_received_headers)


@contextmanager
def observe_response():
    # A mutable cell propagates observations from asyncio.wait_for's task.
    token = _response_state.set({"received": False})
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