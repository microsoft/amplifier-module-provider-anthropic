"""Positive transport-trace evidence for an unreturned model request."""
from contextlib import contextmanager
from contextvars import ContextVar
import ssl

import httpx2

try:
    import httpx  # Optional legacy transport accepted by the SDK.
except ImportError:
    httpx = None

_response_state = ContextVar("anthropic_transport_evidence", default=None)


@contextmanager
def observe_response():
    token = _response_state.set({"seen": False, "received": False, "unknown": False})
    try:
        yield
    finally:
        _response_state.reset(token)


async def trace_request(request):
    state = _response_state.get()
    try:
        previous = request.extensions.get("trace")
        if getattr(previous, "_anthropic_evidence", None) is state:
            return  # Redirect/reused request: never chain our observer twice.
        if hasattr(previous, "_anthropic_previous"):
            previous = previous._anthropic_previous

        async def trace(event, info):
            try:
                state["seen"] = True
                if event.endswith("receive_response_headers.complete") or "receive_response_body" in event:
                    state["received"] = True
                if event.startswith("http2."):
                    state["unknown"] = True  # No local HTTP/2 qualification.
            except Exception:
                if state is not None:
                    state["unknown"] = True
            if previous is not None:
                await previous(event, info)  # Caller exceptions retain their SDK path.

        trace._anthropic_evidence = state
        trace._anthropic_previous = previous
        request.extensions["trace"] = trace
    except Exception:
        if state is not None:
            state["unknown"] = True


def watch_response(http_client):
    try:
        hooks = http_client.event_hooks.setdefault("request", [])
        if trace_request not in hooks:
            hooks.append(trace_request)
    except Exception:
        pass  # No trace report means unknown, never a new provider error.


def can_promote(error):
    try:
        state = _response_state.get()
        return bool(state and state["seen"] and not state["received"]
                    and not state["unknown"] and not certificate_failure(error))
    except Exception:
        return False


def certificate_failure(error):
    seen, pending = set(), [error]
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        if isinstance(error, ssl.SSLCertVerificationError):
            return True
        pending.extend((error.__cause__, error.__context__))
    return False


def is_connection_failure(error):
    classes = (httpx2.NetworkError, httpx2.RemoteProtocolError)
    if httpx is not None:
        classes += (httpx.NetworkError, httpx.RemoteProtocolError)
    return isinstance(error.__cause__, classes)
