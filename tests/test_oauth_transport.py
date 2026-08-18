"""Transport-level and opt-in live tests for Claude subscription OAuth."""

from __future__ import annotations

from amplifier_core.message_models import ChatRequest, Message, ToolSpec
from anthropic import AsyncAnthropic
import httpx
import pytest

from amplifier_anthropic_oauth.auth import (
    AnthropicAuth,
    OAUTH_BETAS,
    default_auth_path,
    oauth_request_headers,
)
from amplifier_module_provider_anthropic import AnthropicProvider


def _assert_oauth_headers(headers: httpx.Headers) -> None:
    expected = oauth_request_headers()
    assert headers["authorization"] == "Bearer sk-ant-oat-transport-test"
    assert "x-api-key" not in headers
    assert headers["User-Agent"] == expected["User-Agent"]
    assert headers["x-app"] == expected["x-app"]
    assert set(OAUTH_BETAS).issubset(set(headers["anthropic-beta"].split(",")))


@pytest.mark.asyncio
async def test_models_and_messages_emit_oauth_headers(monkeypatch):
    """Inspect the final HTTP requests emitted by the Anthropic SDK."""
    monkeypatch.setenv("AMPLIFIER_CLAUDE_CODE_VERSION", "9.8.7")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "type": "model",
                            "id": "claude-haiku-4-5",
                            "display_name": "Claude Haiku 4.5",
                            "created_at": "2025-10-01T00:00:00Z",
                        }
                    ],
                    "has_more": False,
                    "first_id": "claude-haiku-4-5",
                    "last_id": "claude-haiku-4-5",
                },
            )
        if path.startswith("/v1/models/"):
            return httpx.Response(
                200,
                json={
                    "type": "model",
                    "id": "claude-haiku-4-5",
                    "display_name": "Claude Haiku 4.5",
                    "created_at": "2025-10-01T00:00:00Z",
                },
            )
        if path == "/v1/messages":
            return httpx.Response(
                200,
                json={
                    "id": "msg_oauth_transport",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        return httpx.Response(404, json={"error": {"message": path}})

    auth = AnthropicAuth("sk-ant-oat-transport-test", oauth=True)
    provider = AnthropicProvider(
        auth.token,
        config={
            "default_model": "claude-haiku-4-5",
            "use_streaming": False,
            "max_retries": 0,
        },
        initial_auth=auth,
    )
    sdk_client = provider.client
    await sdk_client._client.aclose()
    sdk_client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    models = await provider.list_models()
    assert models
    response = await provider.complete(
        ChatRequest(
            messages=[Message(role="user", content="Reply with ok")],
            metadata={"stream": False},
            max_output_tokens=64,
        )
    )
    assert response.text == "ok"

    assert any(request.url.path == "/v1/models" for request in requests)
    assert any(request.url.path == "/v1/messages" for request in requests)
    for request in requests:
        _assert_oauth_headers(request.headers)


@pytest.mark.live_oauth
@pytest.mark.asyncio
async def test_live_oauth_models_and_native_tool_call():
    """Opt-in smoke test against Anthropic using the stored OAuth credential."""
    auth_file = default_auth_path()
    provider = AnthropicProvider(
        config={
            "auth_file": str(auth_file),
            "default_model": "claude-haiku-4-5",
            "use_streaming": False,
            "max_retries": 0,
            "max_tokens": 64,
        }
    )

    models = await provider.list_models()
    assert any(model.id.startswith("claude-haiku-4-5") for model in models)

    result = await provider.complete(
        ChatRequest(
            messages=[Message(role="user", content="Call oauth_probe with value ok")],
            tools=[
                ToolSpec(
                    name="oauth_probe",
                    description="OAuth integration probe; always call this tool",
                    parameters={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                )
            ],
            metadata={"stream": False},
            max_output_tokens=64,
        ),
        tool_choice={"type": "tool", "name": "oauth_probe"},
    )
    assert result.tool_calls
    assert result.tool_calls[0].name == "oauth_probe"
