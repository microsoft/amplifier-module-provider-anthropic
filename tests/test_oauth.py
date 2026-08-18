"""Tests for Anthropic subscription OAuth and native tool transport."""

import asyncio
import json
from io import BytesIO
import os
import subprocess
import sys
from urllib.error import HTTPError

from amplifier_core.message_models import ToolSpec
from anthropic.types import Message as AnthropicMessage
import pytest

from amplifier_module_provider_anthropic import AnthropicProvider
import amplifier_anthropic_oauth.auth as auth_module
import amplifier_anthropic_oauth.login as login_module
from amplifier_anthropic_oauth.auth import (
    AnthropicAuth,
    AnthropicAuthError,
    AnthropicAuthManager,
    OAUTH_BETAS,
    oauth_request_headers,
    read_credentials,
    refresh_oauth_credentials,
    write_credentials,
)


def test_browser_callback_finishes_with_partial_terminal_input(tmp_path, monkeypatch):
    """A readable partial stdin value must not block callback completion."""
    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd, "r")
    os.write(write_fd, b"partial input without a newline")

    monkeypatch.setattr(login_module.sys, "stdin", stdin)
    monkeypatch.setattr(login_module.webbrowser, "open", lambda url: False)
    monkeypatch.setattr(login_module, "generate_pkce", lambda: ("verifier", "challenge"))
    monkeypatch.setattr(login_module, "authorization_url", lambda *_: "https://example.test")
    monkeypatch.setattr(
        login_module,
        "exchange_authorization_code",
        lambda code, state, verifier: {"access": code},
    )
    monkeypatch.setattr(login_module, "default_auth_path", lambda: tmp_path / "auth.json")
    saved = {}
    monkeypatch.setattr(
        login_module,
        "write_credentials",
        lambda path, credentials: saved.update(path=path, credentials=credentials),
    )

    original_start_server = asyncio.start_server
    servers = []

    async def start_test_server(callback, host, port):
        server = await original_start_server(callback, host, 0)
        servers.append(server)
        return server

    monkeypatch.setattr(login_module.asyncio, "start_server", start_test_server)

    async def run_login_and_callback():
        task = asyncio.create_task(login_module.login())
        while not servers:
            await asyncio.sleep(0)
        port = servers[0].sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(
            b"GET /callback?code=test-code&state=verifier HTTP/1.1\r\n"
            b"Host: localhost\r\n\r\n"
        )
        await writer.drain()
        await reader.read()
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(task, timeout=2)

    try:
        asyncio.run(run_login_and_callback())
        assert os.get_blocking(read_fd) is True
    finally:
        stdin.close()
        os.close(write_fd)

    assert saved == {
        "path": tmp_path / "auth.json",
        "credentials": {"access": "test-code"},
    }


def test_login_module_does_not_import_amplifier_core():
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'amplifier_core' or name.startswith('amplifier_core.'):
        raise AssertionError('standalone login imported amplifier_core')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import amplifier_anthropic_oauth.login
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_token_exchange_uses_claude_identity_headers(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("AMPLIFIER_CLAUDE_CODE_VERSION", "9.8.7")
    monkeypatch.setattr(auth_module, "urlopen", fake_urlopen)
    assert auth_module._post_json("https://example.test/token", {"code": "x"}) == {
        "ok": True
    }
    headers = dict(captured["request"].header_items())
    assert headers["User-agent"] == "claude-cli/9.8.7 (external, sdk-cli)"
    assert headers["Anthropic-beta"] == "oauth-2025-04-20"
    assert headers["X-app"] == "cli"


def test_refresh_uses_oauth_endpoint_and_scopes(monkeypatch):
    captured = {}

    def fake_post_json(url, body):
        captured["url"] = url
        captured["body"] = body
        return {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        }

    monkeypatch.setattr(auth_module, "_post_json", fake_post_json)
    refreshed = refresh_oauth_credentials({"refresh": "old-refresh"})
    assert captured["url"] == auth_module.TOKEN_URL
    assert captured["body"] == {
        "grant_type": "refresh_token",
        "client_id": auth_module.CLIENT_ID,
        "refresh_token": "old-refresh",
        "scope": auth_module.SCOPES,
    }
    assert refreshed["access"] == "new-access"
    assert refreshed["refresh"] == "new-refresh"


def test_token_exchange_error_includes_response_body(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            403,
            "Forbidden",
            {},
            BytesIO(b'{"error":"invalid_request"}'),
        )

    monkeypatch.setattr(auth_module, "urlopen", fake_urlopen)
    with pytest.raises(AnthropicAuthError, match="invalid_request"):
        auth_module._post_json("https://example.test/token", {"code": "x"})


def test_oauth_headers_have_claude_code_identity(monkeypatch):
    monkeypatch.setenv("AMPLIFIER_CLAUDE_CODE_VERSION", "9.8.7")
    headers = oauth_request_headers()
    assert headers["x-app"] == "cli"
    assert headers["User-Agent"] == "claude-cli/9.8.7 (external, sdk-cli)"
    assert set(headers["anthropic-beta"].split(",")) == set(OAUTH_BETAS)
    assert headers["anthropic-dangerous-direct-browser-access"] == "true"


def test_credentials_are_written_atomically_with_private_permissions(tmp_path):
    path = tmp_path / "auth.json"
    credentials = {
        "type": "oauth",
        "access": "sk-ant-oat-access",
        "refresh": "refresh",
        "expires": 9999999999999,
    }
    write_credentials(path, credentials)
    assert read_credentials(path) == credentials
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_auth_manager_prefers_oauth_over_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-key")
    path = tmp_path / "auth.json"
    write_credentials(
        path,
        {
            "type": "oauth",
            "access": "sk-ant-oat-access",
            "refresh": "refresh",
            "expires": 9999999999999,
        },
    )
    auth = asyncio.run(AnthropicAuthManager(path=path).get_auth())
    assert auth == AnthropicAuth("sk-ant-oat-access", oauth=True)


def test_api_key_config_is_optional_for_oauth_users():
    provider = AnthropicProvider(
        "sk-ant-oat-test",
        initial_auth=AnthropicAuth("sk-ant-oat-test", oauth=True),
    )
    api_key_field = next(
        field for field in provider.get_info().config_fields if field.id == "api_key"
    )
    assert api_key_field.required is False


@pytest.mark.parametrize("api_key", [None, ""])
def test_direct_provider_instance_discovers_stored_oauth(tmp_path, api_key):
    path = tmp_path / "auth.json"
    write_credentials(
        path,
        {
            "type": "oauth",
            "access": "sk-ant-oat-discovered",
            "refresh": "refresh",
            "expires": 9999999999999,
        },
    )
    provider = AnthropicProvider(api_key=api_key, config={"auth_file": str(path)})
    asyncio.run(provider._refresh_auth())
    assert provider._auth_state == AnthropicAuth(
        "sk-ant-oat-discovered", oauth=True
    )
    assert provider._default_headers["x-app"] == "cli"
    assert "oauth-2025-04-20" in provider._default_headers["anthropic-beta"]
    assert provider.client.api_key is None
    assert provider.client.auth_token == "sk-ant-oat-discovered"


def test_oauth_client_uses_bearer_auth_and_identity_headers():
    provider = AnthropicProvider(
        "sk-ant-oat-test",
        initial_auth=AnthropicAuth("sk-ant-oat-test", oauth=True),
    )
    client = provider.client
    assert client.api_key is None
    assert client.auth_token == "sk-ant-oat-test"
    assert client.default_headers["x-app"] == "cli"
    assert "oauth-2025-04-20" in client.default_headers["anthropic-beta"]


def test_native_tools_are_sent_structurally():
    provider = AnthropicProvider(
        "sk-ant-oat-test",
        initial_auth=AnthropicAuth("sk-ant-oat-test", oauth=True),
    )
    tools = provider._convert_tools_from_request(
        [
            ToolSpec(
                name="read",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ]
    )
    params = {
        "messages": [{"role": "user", "content": "Read the README"}],
        "tools": tools,
    }
    provider._apply_oauth_request_contract(params)

    response = AnthropicMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"path": "README.md"},
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    result = provider._convert_to_chat_response(response)

    assert params["tools"][0]["name"] == "Read"
    assert params["tools"][0]["input_schema"]["required"] == ["path"]
    assert params["messages"] == [
        {"role": "user", "content": "Read the README"}
    ]
    assert "<tools>" not in json.dumps(params["messages"])
    assert "[tool]:" not in json.dumps(params["messages"])
    assert params["system"][0]["text"].startswith("You are Claude Code")
    assert result.tool_calls[0].name == "read"
    assert result.tool_calls[0].arguments == {"path": "README.md"}
