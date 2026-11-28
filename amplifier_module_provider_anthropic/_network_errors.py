"""Distinguish a lost connection from the SDK's configuration-error wrapper."""

import httpx2

try:
    import httpx  # Optional legacy transport accepted by the Anthropic SDK.
except ImportError:
    httpx = None


def is_connection_failure(error):
    connection_errors = (httpx2.NetworkError, httpx2.RemoteProtocolError)
    if httpx is not None:
        connection_errors += (httpx.NetworkError, httpx.RemoteProtocolError)
    # UnsupportedProtocol and InvalidURL deliberately do not qualify.
    return isinstance(error.__cause__, connection_errors)