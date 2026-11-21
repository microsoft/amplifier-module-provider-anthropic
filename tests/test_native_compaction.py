"""Exercise qualification, SDK wire replay and billing without any live discovery."""

import json
import traceback

from anthropic import AsyncAnthropic
import httpx2 as httpx
import pytest

from amplifier_core import ChatRequest, Message
from amplifier_core.llm_errors import LLMError
from amplifier_core.message_models import ImageBlock
from amplifier_module_provider_anthropic import AnthropicProvider, _compaction

SONNET = "claude-sonnet-5-5"
FABLE = "claude-fable-5-2"
BLOCK = {"type": "compaction", "content": "The earlier decision was amber.", "signature": "test-signature"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("source", "target"), [(SONNET, FABLE), (FABLE, SONNET)])
async def test_signed_checkpoint_scope_first_block_and_beta_through_sdk(source, target):
    calls = []

    def serve(req):
        calls.append(req)
        payload = json.loads(req.content)
        assert _compaction.BETA in req.headers["anthropic-beta"].split(",")
        if req.url.path.endswith("/count_tokens"):
            assert payload["messages"][0]["content"] == [BLOCK]
            return httpx.Response(200, json={"input_tokens": 51})
        assert payload["model"] == source
        assert payload["compaction"] == {"type": "summarize"}
        return httpx.Response(200, json={
            "id": "msg_abc", "type": "message", "role": "assistant", "model": source,
            "stop_reason": "compaction", "stop_sequence": None, "content": [BLOCK],
            "usage": {"input_tokens": 0, "output_tokens": 0, "iterations": [
                {"type": "compaction", "input_tokens": 54, "output_tokens": 12,
                 "cache_creation_input_tokens": 4, "cache_read_input_tokens": 3}]},
        })

    raw = AnthropicProvider(api_key="test-key", config={"default_model": source, "enable_prompt_caching": False})
    raw._client = AsyncAnthropic(api_key="test-key", base_url="https://api.anthropic.com",
                                 http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)))
    other = AnthropicProvider(api_key="different-key", config={"default_model": target})
    try:
        assert raw.supports_native_compaction(model=source)
        assert not raw.supports_native_compaction(model="claude-haiku-5-5")
        assert calls == []
        history = [Message(role="user", content="Remember amber."), Message(role="assistant", content="Recorded.")]
        result = await raw.compact_context(ChatRequest(messages=history, model=source, max_output_tokens=512))
        assert result["usage"] == {"input_tokens": 57, "cache_write_tokens": 4, "cache_read_tokens": 3,
                                   "total_tokens": 69, "output_tokens": 12}
        assert result["input_tokens"] == 61
        assert raw.validate_compacted_context(result["message"], model=source)
        assert result["message"]["metadata"][_compaction.PORTABLE] is True
        assert result["message"]["content"] == BLOCK["content"]
        with pytest.raises(ValueError, match="differs"):
            raw.validate_compacted_context({**result["message"], "content": "changed"}, model=source)
        assert other.rebind_compacted_context(result["message"], source_model=source, model=target) is None
        assert raw.rebind_compacted_context(result["message"], source_model=source, model="other") is None
        rebound = raw.rebind_compacted_context(result["message"], source_model=source, model=target)
        assert rebound["metadata"][_compaction.KEY]["block"] == BLOCK
        assert "test-key" not in json.dumps(rebound)
        overridden = AnthropicProvider(api_key="test-key", config={"default_model": target})
        overridden._client = raw.client.with_options(default_headers={"X-API-Key": "overridden-key"})
        assert not overridden.supports_native_compaction(model=target)
        assert overridden.rebind_compacted_context(result["message"], source_model=source, model=target) is None
        assert len(calls) == 1
        request = ChatRequest(messages=[Message(role="developer", content="New instruction."),
                                        Message(**rebound), Message(role="user", content="What was the decision?")])
        budget = await raw.request_budget(request, context_estimate=50, request_options={"model": target},
                                          require_measurement=True)
        assert budget is not None
        params = raw._assemble_request_params(request, request_options={"model": target},
                                              request_caps=raw._get_capabilities(target)).params
        assert params["messages"][0] == {"role": "assistant", "content": [BLOCK]}
        assert "New instruction." in str(params["messages"][1])
        assert _compaction.BETA in params["extra_headers"]["anthropic-beta"].split(",")
        assert len(calls) == 2
    finally:
        await raw.close()
        await other.close()


@pytest.mark.parametrize("config", [{"base_url": "https://proxy.example/v1"},
                                    {"extra_request_params": {"extra_body": {"model": "other"}}}])
def test_unqualified_connection_and_override_fail_closed_locally(config):
    raw = AnthropicProvider(api_key="test-key", config=config)
    assert not raw.supports_native_compaction(model=SONNET)


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", [False, True])
async def test_failed_image_compaction_does_not_expose_sdk_request(echo):
    url = "https://example.invalid/private-compaction-image.png"
    requests = []

    def reject(request):
        requests.append(json.loads(request.content))
        detail = url.replace("/", r"\/") if echo else "invalid image"
        return httpx.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error", "message": detail,
        }})

    provider = AnthropicProvider(api_key="test-key", config={"default_model": SONNET})
    provider._client = AsyncAnthropic(api_key="test-key", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(reject)))
    try:
        image = ImageBlock(source={"type": "url", "url": url})
        request = ChatRequest(messages=[Message(role="user", content=[image])], max_output_tokens=512)
        with pytest.raises(LLMError) as caught:
            await provider.compact_context(request)
        assert requests[0]["messages"][0]["content"][0]["source"]["url"] == url
        error = caught.value
        assert error.status_code == 400
        assert error.__cause__ is None and error.__context__ is None
        assert "private-compaction-image" not in "".join(traceback.format_exception(error))
        assert not hasattr(error, "request") and not hasattr(error, "response")
    finally:
        await provider.close()
