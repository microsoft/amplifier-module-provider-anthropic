"""Command-line login for Anthropic Claude Pro/Max OAuth."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import os
import sys
import webbrowser

from .auth import (
    AnthropicAuthError,
    CALLBACK_HOST,
    CALLBACK_PATH,
    CALLBACK_PORT,
    REDIRECT_URI,
    authorization_url,
    default_auth_path,
    exchange_authorization_code,
    generate_pkce,
    parse_authorization_input,
    write_credentials,
)

_SUCCESS = """<!doctype html><title>Anthropic login complete</title>
<h1>Authentication complete</h1><p>You can close this window.</p>"""
_ERROR = """<!doctype html><title>Anthropic login failed</title>
<h1>Authentication failed</h1><p>{message}</p>"""


async def _read_callback(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    expected_state: str,
    result: asyncio.Future[tuple[str, str]],
) -> None:
    status = "400 Bad Request"
    body = _ERROR.format(message="Missing authorization response.")
    try:
        request_line = (await reader.readline()).decode(errors="replace").strip()
        parts = request_line.split(" ")
        if len(parts) >= 2:
            code, state = parse_authorization_input(f"http://localhost{parts[1]}")
            if not parts[1].startswith(CALLBACK_PATH):
                status = "404 Not Found"
                body = _ERROR.format(message="Callback route not found.")
            elif not code or not state:
                body = _ERROR.format(message="Missing code or state parameter.")
            elif state != expected_state:
                body = _ERROR.format(message="OAuth state mismatch.")
            else:
                status = "200 OK"
                body = _SUCCESS
                if not result.done():
                    result.set_result((code, state))
    except Exception as exc:
        body = _ERROR.format(message=str(exc))
    payload = body.encode()
    writer.write(
        f"HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
        + payload
    )
    with suppress(Exception):
        await writer.drain()
    writer.close()
    with suppress(Exception):
        await writer.wait_closed()


async def login() -> None:
    verifier, challenge = generate_pkce()
    url = authorization_url(verifier, challenge)
    loop = asyncio.get_running_loop()
    callback: asyncio.Future[tuple[str, str]] = loop.create_future()

    server: asyncio.Server | None = None
    try:
        server = await asyncio.start_server(
            lambda reader, writer: _read_callback(reader, writer, verifier, callback),
            CALLBACK_HOST,
            CALLBACK_PORT,
        )
    except OSError as exc:
        print(f"Could not start the local callback server: {exc}", file=sys.stderr)

    print("Open this URL to authenticate with Anthropic:\n")
    print(url)
    print(f"\nWaiting for the callback at {REDIRECT_URI}.")
    opened = webbrowser.open(url)
    if not opened:
        print("Could not open a browser automatically; open the URL above manually.")

    manual: asyncio.Future[str] = loop.create_future()
    manual_input = bytearray()
    stdin_fd: int | None = None
    stdin_was_blocking: bool | None = None

    def read_manual_input() -> None:
        # An asyncio fd callback must never call TextIO.readline(): terminals
        # can report a partial/spurious readiness event, after which readline
        # blocks the entire event loop even though the browser callback won.
        assert stdin_fd is not None
        try:
            chunk = os.read(stdin_fd, 4096)
        except BlockingIOError:
            return
        if chunk:
            manual_input.extend(chunk)
        if not chunk or b"\n" in manual_input or b"\r" in manual_input:
            line = bytes(manual_input).splitlines()[0] if manual_input else b""
            if not manual.done():
                manual.set_result(line.decode(errors="replace"))

    has_stdin_reader = False
    try:
        stdin_fd = sys.stdin.fileno()
        stdin_was_blocking = os.get_blocking(stdin_fd)
        os.set_blocking(stdin_fd, False)
        loop.add_reader(stdin_fd, read_manual_input)
        has_stdin_reader = True
        print(
            "No terminal input is needed when the browser is on this machine.\n"
            "If the browser is elsewhere, paste the final redirect URL or "
            "authorization code here, then press Enter:\n> ",
            end="",
            flush=True,
        )
    except (AttributeError, NotImplementedError, OSError):
        if stdin_fd is not None and stdin_was_blocking is not None:
            with suppress(OSError):
                os.set_blocking(stdin_fd, stdin_was_blocking)
        stdin_fd = None

    waiters = {callback, manual} if has_stdin_reader else {callback}
    try:
        try:
            async with asyncio.timeout(5 * 60):
                done, _ = await asyncio.wait(
                    waiters, return_when=asyncio.FIRST_COMPLETED
                )
        except TimeoutError as exc:
            raise AnthropicAuthError(
                "Timed out waiting for the OAuth callback"
            ) from exc
        if callback in done:
            code, state = callback.result()
            print("\nOAuth callback received; exchanging authorization code...")
        else:
            code, supplied_state = parse_authorization_input(manual.result())
            state = supplied_state or verifier
            if supplied_state and supplied_state != verifier:
                raise AnthropicAuthError("OAuth state mismatch")
            if not code:
                raise AnthropicAuthError("Missing authorization code")

        credentials = await asyncio.to_thread(
            exchange_authorization_code, code, state, verifier
        )
        path = default_auth_path()
        await asyncio.to_thread(write_credentials, path, credentials)
        print(f"Anthropic OAuth credentials saved to {path}")
    finally:
        if has_stdin_reader and stdin_fd is not None:
            loop.remove_reader(stdin_fd)
        if stdin_fd is not None and stdin_was_blocking is not None:
            with suppress(OSError):
                os.set_blocking(stdin_fd, stdin_was_blocking)
        if server:
            server.close()
            await server.wait_closed()
        for task in (manual, callback):
            if not task.done():
                task.cancel()


def main() -> None:
    try:
        asyncio.run(login())
    except (AnthropicAuthError, KeyboardInterrupt) as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
