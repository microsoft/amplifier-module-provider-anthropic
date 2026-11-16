"""Optional real CI observer integration, not an invented JSON subscriber."""
import asyncio
import json
from pathlib import Path

import anthropic
import httpx2
import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
import amplifier_module_provider_anthropic as provider_module


def test_validation_detail_reaches_actual_ci_logging_handler(tmp_path: Path, monkeypatch):
    ci = pytest.importorskip("amplifier_module_hook_context_intelligence")
    from amplifier_module_hook_context_intelligence.handlers.logging_handler import LoggingHandler
    monkeypatch.setenv("AMPLIFIER_CONTEXT_INTELLIGENCE_BASE_PATH", str(tmp_path))
    private = "private-thinking-prompt-body-header-credential-config-toolname"

    async def run():
        coordinator = ModuleCoordinator()
        coordinator.hooks.set_default_fields(session_id="count-validation-capture")
        coordinator.register_capability("session.working_dir", str(tmp_path))
        ci_cleanup = await ci.mount(coordinator, {"base_path": str(tmp_path),
            "project_slug": "synthetic-count", "destinations": {}, "exclude_events": []})
        provider_cleanup = None
        calls = []

        def transport(request):
            calls.append(request.url.path)
            assert calls[-1] == "/v1/messages/count_tokens"
            return httpx2.Response(400, headers={"request-id": "req_validation_capture", "x-private": private},
                json={"type": "error", "error": {"type": "invalid_request_error",
                      "message": "thinking.type: Field required", "private": private}})

        try:
            provider_cleanup = await provider_module.mount(coordinator, {
                "api_key": private, "default_model": "claude-fable-5-2", "max_retries": 0})
            await ci.on_session_ready(coordinator)
            state = coordinator.get_capability("context_intelligence._hook_state")
            handler = state["logging_handler"]
            assert isinstance(handler, LoggingHandler) and handler._dispatchers == []
            assert state["destinations"] == {}
            assert coordinator.hooks.list_handlers()["provider:token_count_unavailable"] == ["LoggingHandler"]
            provider = coordinator.get("providers", "anthropic")
            async with anthropic.AsyncAnthropic(api_key=private, max_retries=0,
                http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(transport), trust_env=False)) as sdk:
                provider._client = sdk
                request = ChatRequest(messages=[Message(role="user", content=private)])
                assert await provider.request_budget(request, context_estimate=200000) is None
            provider._client = None
            assert calls == ["/v1/messages/count_tokens"]
            resolver = coordinator.get_capability("context_intelligence.hook_config_resolver")
            return resolver.session_dir("count-validation-capture") / "events.jsonl"
        finally:
            if provider_cleanup:
                await provider_cleanup()
            await ci_cleanup()

    capture = asyncio.run(run())
    # Synthetic bounded test capture only, not a user session transcript.
    serialized = capture.read_text()
    assert len(serialized) < 16384 and private not in serialized
    events = [json.loads(line) for line in serialized.splitlines()]
    event = next(row for row in events if row["event"] == "provider:token_count_unavailable")
    data = event["data"]
    assert data["diagnostic_version"] == 1 and data["reason_code"] == "required_field"
    assert data["schema_path"] == "thinking.type" and data["detail_status"] == "recognized"
    assert data["request_id"] == "req_validation_capture"
    assert "Field required" not in serialized
