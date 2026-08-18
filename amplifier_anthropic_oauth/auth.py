"""Anthropic OAuth credential management for Claude Pro/Max accounts."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen


CLIENT_ID = base64.b64decode(
    "OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl"
).decode()
AUTHORIZE_URL = "https://claude.com/cai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 53692
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPES = (
    "org:create_api_key user:profile user:inference "
    "user:sessions:claude_code user:mcp_servers user:file_upload"
)
# Stable OAuth identity betas. Feature-specific betas (thinking, context,
# tools, caching) are selected by the upstream provider per request.
OAUTH_BETAS = (
    "claude-code-20250219",
    "oauth-2025-04-20",
)


def installed_claude_code_version() -> str:
    """Use the installed CLI version in the identity header when available."""
    configured = os.environ.get("AMPLIFIER_CLAUDE_CODE_VERSION")
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        version = result.stdout.strip().split(" ", 1)[0]
        if version and all(part.isdigit() for part in version.split(".")):
            return version
    except (OSError, subprocess.SubprocessError):
        pass
    # This is only an attribution header. Authentication does not depend on
    # the installed CLI, so retain a known-compatible fallback.
    return "2.1.75"


def oauth_request_headers() -> dict[str, str]:
    """Headers used by pi for Anthropic Claude Pro/Max OAuth requests."""
    return {
        "Accept": "application/json",
        "anthropic-dangerous-direct-browser-access": "true",
        "anthropic-beta": ",".join(OAUTH_BETAS),
        # Match the SDK's canonical key casing so this replaces, rather than
        # appends to, its default AsyncAnthropic/Python user-agent.
        "User-Agent": (
            f"claude-cli/{installed_claude_code_version()} (external, sdk-cli)"
        ),
        "x-app": "cli",
    }


@dataclass(frozen=True)
class AnthropicAuth:
    """Resolved request authentication."""

    token: str
    oauth: bool


class AnthropicAuthError(RuntimeError):
    """Raised when Anthropic credentials cannot be resolved or refreshed."""


def default_auth_path() -> Path:
    configured = os.environ.get("AMPLIFIER_ANTHROPIC_AUTH_FILE") or os.environ.get(
        "AMPLIFIER_CLAUDE_AUTH_FILE"
    )
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".amplifier" / "anthropic-auth.json"
    )


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def generate_pkce() -> tuple[str, str]:
    import hashlib

    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


def authorization_url(verifier: str, challenge: str) -> str:
    query = urlencode(
        {
            "code": "true",
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": verifier,
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


def parse_authorization_input(value: str) -> tuple[str | None, str | None]:
    value = value.strip()
    if not value:
        return None, None
    try:
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            query = parse_qs(parsed.query)
            return query.get("code", [None])[0], query.get("state", [None])[0]
    except ValueError:
        pass
    if "#" in value:
        code, state = value.split("#", 1)
        return code or None, state or None
    if "code=" in value:
        query = parse_qs(value)
        return query.get("code", [None])[0], query.get("state", [None])[0]
    return value, None


def _post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    identity_headers = oauth_request_headers()
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": identity_headers["User-Agent"],
            "anthropic-beta": "oauth-2025-04-20",
            "x-app": identity_headers["x-app"],
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS endpoint
            payload = response.read().decode()
    except HTTPError as exc:
        response_body = exc.read().decode(errors="replace")
        raise AnthropicAuthError(
            f"Anthropic OAuth request failed: HTTP {exc.code}: {response_body}"
        ) from exc
    except Exception as exc:
        raise AnthropicAuthError(f"Anthropic OAuth request failed: {exc}") from exc
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AnthropicAuthError("Anthropic OAuth returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise AnthropicAuthError("Anthropic OAuth returned an invalid response")
    return result


def _credentials_from_token_response(data: dict[str, Any]) -> dict[str, Any]:
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")
    if not isinstance(access, str) or not isinstance(refresh, str):
        raise AnthropicAuthError("Anthropic OAuth response did not contain tokens")
    if not isinstance(expires_in, (int, float)):
        raise AnthropicAuthError("Anthropic OAuth response did not contain an expiry")
    return {
        "type": "oauth",
        "access": access,
        "refresh": refresh,
        # Refresh five minutes early, matching pi's credential handling.
        "expires": int(time.time() * 1000 + expires_in * 1000 - 5 * 60 * 1000),
    }


def exchange_authorization_code(code: str, state: str, verifier: str) -> dict[str, Any]:
    return _credentials_from_token_response(
        _post_json(
            TOKEN_URL,
            {
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": REDIRECT_URI,
                "code_verifier": verifier,
            },
        )
    )


def refresh_oauth_credentials(credentials: dict[str, Any]) -> dict[str, Any]:
    refresh = credentials.get("refresh")
    if not isinstance(refresh, str) or not refresh:
        raise AnthropicAuthError("Stored Anthropic OAuth credentials have no refresh token")
    return _credentials_from_token_response(
        _post_json(
            TOKEN_URL,
            {
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh,
                "scope": SCOPES,
            },
        )
    )


def read_credentials(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise AnthropicAuthError(f"Could not read Anthropic credentials at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnthropicAuthError(f"Invalid Anthropic credentials at {path}")
    return value


def write_credentials(path: Path, credentials: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(credentials, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class AnthropicAuthManager:
    """Resolve API-key or OAuth auth and refresh OAuth tokens when needed."""

    def __init__(self, path: Path | None = None, api_key: str | None = None) -> None:
        self.path = path or default_auth_path()
        self.api_key = api_key
        self._lock = asyncio.Lock()

    async def get_auth(self) -> AnthropicAuth:
        oauth_token = os.environ.get("ANTHROPIC_OAUTH_TOKEN")
        if oauth_token:
            return AnthropicAuth(oauth_token, oauth=True)

        async with self._lock:
            credentials = await asyncio.to_thread(read_credentials, self.path)
            if credentials and credentials.get("type") == "oauth":
                expires = credentials.get("expires", 0)
                if not isinstance(expires, (int, float)) or expires <= time.time() * 1000:
                    credentials = await asyncio.to_thread(refresh_oauth_credentials, credentials)
                    await asyncio.to_thread(write_credentials, self.path, credentials)
                access = credentials.get("access")
                if isinstance(access, str) and access:
                    return AnthropicAuth(access, oauth=True)

            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if key:
                return AnthropicAuth(key, oauth=False)

        raise AnthropicAuthError(
            "No Anthropic credentials. Run `amplifier-anthropic-login`, set "
            "ANTHROPIC_OAUTH_TOKEN`, or set `ANTHROPIC_API_KEY`."
        )
