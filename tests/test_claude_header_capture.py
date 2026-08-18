"""Optional local capture of the headers emitted by an installed ``claude -p``."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import shutil
import socketserver
import ssl
import subprocess
import threading

import pytest

from amplifier_anthropic_oauth.auth import OAUTH_BETAS, oauth_request_headers


class _CaptureProxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler, tls_context: ssl.SSLContext):
        super().__init__(server_address, handler)
        self.tls_context = tls_context
        self.captured: queue.Queue[dict[str, str]] = queue.Queue()
        self.events: queue.Queue[str] = queue.Queue()


class _ConnectHandler(socketserver.BaseRequestHandler):
    """Minimal CONNECT proxy which records only redacted request headers."""

    def handle(self) -> None:
        connection = self.request
        connection.settimeout(5)
        connect_request = self._read_headers(connection)
        first_line = connect_request.split(b"\r\n", 1)[0]
        self.server.events.put(first_line.decode(errors="replace"))
        if first_line != b"CONNECT api.anthropic.com:443 HTTP/1.1":
            connection.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return

        connection.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        try:
            self.server.events.put("tls-start")
            tls = self.server.tls_context.wrap_socket(connection, server_side=True)
            self.server.events.put("tls-ok")
            request_head = self._read_headers(tls)
            self.server.events.put(
                request_head.split(b"\r\n", 1)[0].decode(errors="replace")
            )
            lines = request_head.decode("iso-8859-1").split("\r\n")
            request_line = lines[0].split(" ")
            if (
                len(request_line) < 2
                or request_line[1].split("?", 1)[0] != "/v1/messages"
            ):
                self._respond(tls)
                return

            raw_headers: dict[str, str] = {}
            for line in lines[1:]:
                if ":" not in line:
                    continue
                name, value = line.split(":", 1)
                raw_headers[name.strip().lower()] = value.strip()

            authorization = raw_headers.get("authorization", "")
            # Never retain or write the credential. Only preserve its scheme.
            normalized = {
                "authorization": (
                    "Bearer <redacted>"
                    if authorization.startswith("Bearer ")
                    else "<missing>"
                ),
                "user-agent": raw_headers.get("user-agent", ""),
                "x-app": raw_headers.get("x-app", ""),
                "anthropic-beta": raw_headers.get("anthropic-beta", ""),
                "anthropic-dangerous-direct-browser-access": raw_headers.get(
                    "anthropic-dangerous-direct-browser-access", ""
                ),
            }
            self.server.captured.put(normalized)
            self._respond(tls)
        except (OSError, ssl.SSLError, TimeoutError) as exc:
            self.server.events.put(f"{type(exc).__name__}: {exc}")
            return

    @staticmethod
    def _read_headers(connection) -> bytes:
        data = bytearray()
        while b"\r\n\r\n" not in data and len(data) < 128 * 1024:
            chunk = connection.recv(4096)
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    @staticmethod
    def _respond(connection) -> None:
        body = json.dumps(
            {
                "type": "error",
                "error": {"type": "authentication_error", "message": "captured"},
            }
        ).encode()
        connection.sendall(
            b"HTTP/1.1 401 Unauthorized\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )


def _generate_ca_and_leaf(tmp_path: Path) -> tuple[Path, ssl.SSLContext]:
    openssl = shutil.which("openssl")
    if not openssl:
        pytest.skip("openssl is required for local Claude header capture")

    ca_key = tmp_path / "ca.key"
    ca_cert = tmp_path / "ca.pem"
    leaf_key = tmp_path / "leaf.key"
    leaf_csr = tmp_path / "leaf.csr"
    leaf_cert = tmp_path / "leaf.pem"
    extensions = tmp_path / "leaf.ext"
    extensions.write_text(
        "subjectAltName=DNS:api.anthropic.com\n"
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
    )

    commands = [
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(ca_key),
            "-out",
            str(ca_cert),
            "-days",
            "1",
            "-subj",
            "/CN=Amplifier Claude Header Capture CA",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
        ],
        [
            openssl,
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(leaf_key),
            "-out",
            str(leaf_csr),
            "-subj",
            "/CN=api.anthropic.com",
        ],
        [
            openssl,
            "x509",
            "-req",
            "-in",
            str(leaf_csr),
            "-CA",
            str(ca_cert),
            "-CAkey",
            str(ca_key),
            "-CAcreateserial",
            "-out",
            str(leaf_cert),
            "-days",
            "1",
            "-extfile",
            str(extensions),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True)

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(leaf_cert, leaf_key)
    context.set_alpn_protocols(["http/1.1"])
    return ca_cert, context


@pytest.mark.local_header_capture
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="local credential-bearing test")
def test_claude_p_stable_headers_match_provider(tmp_path):
    """Capture a real local Claude request and compare its stable OAuth headers."""
    claude = shutil.which("claude")
    if not claude:
        pytest.skip("Claude Code is not installed")
    if not (Path.home() / ".claude" / ".credentials.json").exists():
        pytest.skip("Claude Code credentials were not found")

    ca_cert, tls_context = _generate_ca_and_leaf(tmp_path)
    proxy = _CaptureProxy(("127.0.0.1", 0), _ConnectHandler, tls_context)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()

    proxy_url = f"http://127.0.0.1:{proxy.server_address[1]}"
    env = os.environ.copy()
    env.update(
        {
            "HTTPS_PROXY": proxy_url,
            "https_proxy": proxy_url,
            "NO_PROXY": "",
            "no_proxy": "",
            "NODE_EXTRA_CA_CERTS": str(ca_cert),
            "NODE_USE_SYSTEM_CA": "1",
        }
    )
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        env.pop(name, None)

    process = subprocess.Popen(
        [claude, "-p", "Reply exactly OK", "--no-session-persistence"],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        try:
            captured = proxy.captured.get(timeout=20)
        except queue.Empty:
            events = list(proxy.events.queue)
            process.poll()
            pytest.fail(
                "Claude did not send a capturable request through the local proxy; "
                f"returncode={process.returncode}, proxy_events={events}"
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=3)

    expected = oauth_request_headers()
    assert captured["authorization"] == "Bearer <redacted>"
    assert captured["user-agent"] == expected["User-Agent"]
    assert captured["x-app"] == expected["x-app"]
    assert captured["anthropic-dangerous-direct-browser-access"] == expected[
        "anthropic-dangerous-direct-browser-access"
    ]
    assert set(OAUTH_BETAS).issubset(
        set(captured["anthropic-beta"].split(","))
    )
