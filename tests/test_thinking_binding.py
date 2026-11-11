"""Thinking bindings use one stateless assembly for budget and dispatch.

Offline tests assert envelope/preservation, not cryptographic validity. Live
qualification must use real signed blocks and inspect input_transformations.
"""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_anthropic import AnthropicProvider, _RuntimeModelInfo
from amplifier_module_provider_anthropic._thinking import (
    BINDING_BETA, NATIVE_PROVENANCE, configure_binding,
)
from tests._helpers import FakeCoordinator

MODEL = "claude-fable-5-2"
DROP = {"prefix_mismatch_behavior": "drop_block"}


def provider(**config):
    p = AnthropicProvider(api_key="test-key", config={
        "default_model": MODEL, "max_retries": 0, "use_streaming": False,
        "enable_prompt_caching": False, "reasoning_effort": "high", **config,
    }, coordinator=FakeCoordinator())
    # Fable5.2 has no guessed static budget window. A previously retrieved
    # runtime window is required before a pure budget probe can count.
    p._runtime_model_info_cache[MODEL] = _RuntimeModelInfo(max_input_tokens=200000)
    return p


def request(context="Context"):
    return ChatRequest(messages=[
        Message(role="user", content=context),
        Message(role="assistant", content=[
            {"type":"thinking", "thinking":"First", "signature":"sig1"},
            {"type":"text", "text":"Answer"},
            {"type":"redacted_thinking", "data":"opaque"},
            {"type":"thinking", "thinking":"Second", "signature":"sig2"},
            {"type":"tool_call", "id":"call1", "name":"lookup", "input":{}},
        ]),
        Message(role="tool", tool_call_id="call1", content="Result already executed"),
    ])


def assemble(p, r, **options):
    return p._assemble_request_params(r, request_options=options,
        request_caps=p._get_capabilities(options.get("model", p.default_model))).params


@pytest.mark.parametrize("model", [MODEL, "claude-fable-5-1", "claude-mythos-5-2",
    "claude-mythos-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"])
def test_documented_models_receive_native_selective_control(model):
    params = assemble(provider(default_model=model), request())
    assert params["thinking"]["block_binding"] == DROP
    assert BINDING_BETA in params["extra_headers"]["anthropic-beta"].split(",")


@pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-fable-5", "claude-fable-5-20", "claude-fable-6-1", "proxy-model"])
def test_unverified_models_are_not_opted_into_beta(model):
    params = {"model":model}
    configure_binding(params, base_url="https://api.anthropic.com")
    assert params == {"model":model}


@pytest.mark.parametrize("url", ["https://proxy.example", "http://api.anthropic.com",
    "https://api.anthropic.com/path", "https://api.anthropic.com:444",
    "https://user@api.anthropic.com", "https://api.anthropic.com?proxy=1"])
def test_custom_endpoints_untouched_unless_explicitly_opted_in(url):
    params = {"model":MODEL}
    configure_binding(params, base_url=url)
    assert params == {"model":MODEL}
    configure_binding(params, base_url=url, policy="drop_block")
    assert params["thinking"] == {"block_binding":DROP}


@pytest.mark.parametrize("context", ["Context", "Changed snapshot", "Compacted summary"])
def test_resume_edits_multiple_blocks_and_results_preserve_canonical_bytes(context):
    p = provider()
    r = request(context)
    before = deepcopy(r.model_dump())
    params = assemble(p, r)
    again = assemble(provider(), r)  # resumed with no cached binding state
    assert params == again
    assert r.model_dump() == before
    canonical = deepcopy(before["messages"][1]["content"])
    expected = [p._clean_content_block(block) for block in canonical]
    assert params["messages"][1]["content"] == expected
    assert params["messages"][2]["content"] == [{"type":"tool_result", "tool_use_id":"call1", "content":"Result already executed"}]


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"])
def test_model_switch_does_not_locally_strip_native_blocks(model):
    p = provider()
    params = assemble(p, request(), model=model)
    assert [b["type"] for b in params["messages"][1]["content"]] == [
        "thinking", "text", "redacted_thinking", "thinking", "tool_use"]


def test_native_refusal_fallback_preserves_compatible_thinking_and_tool_receipts():
    p = provider()
    p._refusal_fallback_target = lambda _: "claude-opus-5-5"
    fallback = p._convert_to_chat_response(native_response([]))
    p._complete_chat_request = AsyncMock(return_value=fallback)
    refusal = fallback.model_copy(update={"finish_reason":"refusal"})
    r = request()
    original = r.model_dump()
    assert asyncio.run(p._apply_refusal_fallback(refusal, r, MODEL)) is fallback
    retried = p._complete_chat_request.call_args.args[0]
    assert retried.model_dump() == original
    assert retried is not r
    assert r.model_dump() == original
    assert p._complete_chat_request.call_count == 1


@pytest.mark.parametrize("mode", ["disabled", "between_tools", "enabled"])
def test_native_unsupported_modes_are_not_silently_enabled(mode):
    params = {"model":"claude-sonnet-5-5", "thinking":{"type":mode}}
    before = deepcopy(params)
    configure_binding(params, base_url="https://api.anthropic.com")
    assert params == before


def test_missing_mode_on_sonnet_haiku_is_not_enabled():
    for model in ["claude-sonnet-5-5", "claude-haiku-5-5"]:
        params = {"model":model}
        configure_binding(params, base_url="https://api.anthropic.com")
        assert params == {"model":model}


def test_explicit_error_policy_and_beta_headers_win_without_mutating_config():
    config = {"thinking":{"type":"adaptive", "block_binding":{"prefix_mismatch_behavior":"error"}},
        "extra_headers":{"Anthropic-Beta":"another-beta", "x-custom":"value"}}
    before = deepcopy(config)
    params = assemble(provider(extra_request_params=config), request())
    assert params["thinking"]["block_binding"] == {"prefix_mismatch_behavior":"error"}
    assert params["extra_headers"] == {"anthropic-beta":f"another-beta,{BINDING_BETA}", "x-custom":"value"}
    assert config == before


def test_extra_body_effective_model_and_thinking_controls_are_shared_with_count():
    p = provider(extra_request_params={"extra_body":{
        "model":"claude-sonnet-5-5", "thinking":{"type":"adaptive", "block_binding":{"prefix_mismatch_behavior":"error"}}}})
    params = assemble(p, request())
    count = p._count_tokens_params(params)
    assert count["model"] == "claude-sonnet-5-5"
    assert count["thinking"] == params["extra_body"]["thinking"]
    assert count["thinking"]["block_binding"] == {"prefix_mismatch_behavior":"error"}


def test_extra_body_header_field_is_not_mistaken_for_sdk_transport_headers():
    params = {"model":MODEL, "extra_headers":{"anthropic-beta":BINDING_BETA},
        "extra_body":{"extra_headers":{"anthropic-beta":"not-a-transport-override"}}}
    assert AnthropicProvider._count_tokens_params(params)["extra_headers"] == {
        "anthropic-beta":BINDING_BETA
    }


def test_null_policy_leaves_vendor_default_and_request_override_wins():
    assert "block_binding" not in assemble(provider(thinking_prefix_mismatch_behavior=None), request()).get("thinking", {})
    p = provider(thinking_prefix_mismatch_behavior="error")
    assert assemble(p, request())["thinking"]["block_binding"] == {"prefix_mismatch_behavior":"error"}
    assert "block_binding" not in assemble(p, request(), thinking_prefix_mismatch_behavior=None).get("thinking", {})


@pytest.mark.parametrize("thinking", [None, {"type":"disabled"}, {"block_binding":None},
    {"block_binding":{}}, {"type":"adaptive", "block_binding":{"prefix_mismatch_behavior":"error"}}])
def test_explicit_native_controls_are_preserved_by_presence(thinking):
    params = {"model":MODEL, "thinking":deepcopy(thinking)}
    configure_binding(params, base_url="https://api.anthropic.com")
    assert params["thinking"] == thinking


@pytest.mark.parametrize("beta", ["", None, 17])
def test_removed_or_malformed_beta_never_leaves_half_injected_control(beta):
    params = {"model":MODEL, "extra_headers":{"Anthropic-Beta":beta}}
    before = deepcopy(params)
    configure_binding(params, base_url="https://api.anthropic.com")
    assert params == before


def test_sdk_resolved_endpoint_and_default_headers_are_authoritative(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example")
    p = provider()
    assert "block_binding" not in assemble(p, request()).get("thinking", {})
    params = {"model":MODEL}
    configure_binding(params, base_url="https://api.anthropic.com/",
        default_headers={"Anthropic-Beta":"from-host-client", "x-client":"yes"})
    assert params["extra_headers"]["anthropic-beta"] == f"from-host-client,{BINDING_BETA}"


def test_unknown_budget_model_override_is_unavailable_not_wrong_window():
    p = provider(extra_request_params={"extra_body":{"model":"proxy-unknown"}})
    p.client.messages.count_tokens = AsyncMock()
    assert asyncio.run(p.request_budget(request(), context_estimate=1000)) is None
    p.client.messages.count_tokens.assert_not_called()


@pytest.mark.parametrize("invalid", [False, {}, "strip_all"])
def test_invalid_policy_fails_before_network(invalid):
    with pytest.raises(Exception, match="thinking_prefix_mismatch_behavior"):
        assemble(provider(thinking_prefix_mismatch_behavior=invalid), request())


@pytest.mark.parametrize("origin", ["gemini", "github-copilot"])
def test_known_foreign_signatures_and_opaque_redaction_never_reach_native_wire(origin):
    p = provider()
    messages = [{"role":"assistant", "content":[
        {"type":"thinking", "thinking":"Readable summary", "signature":"foreign", "thinking_provenance":{"version":1,"provider":origin}},
        {"type":"redacted_thinking", "data":"private-opaque", "thinking_provenance":{"version":1,"provider":origin}},
        {"type":"text", "text":"Answer"}]}]
    before = deepcopy(messages)
    assert p._convert_messages(messages)[0]["content"] == [
        {"type":"text", "text":"Readable summary"}, {"type":"text", "text":"Answer"}]
    assert messages == before


def native_response(transformations=None):
    r = anthropic.types.Message(id="msg_synthetic", model=MODEL, role="assistant", type="message",
        content=[{"type":"thinking", "thinking":"Reason", "signature":"native"},
                 {"type":"redacted_thinking", "data":"opaque"}, {"type":"text", "text":"Done"}],
        stop_reason="end_turn", usage={"input_tokens":10,"output_tokens":10})
    if transformations is not None:
        r.input_transformations = transformations
    return r


def test_response_stamps_native_origin_and_retains_transformation_metadata():
    p = provider()
    report = [{"type":"thinking_dropped", "path":"messages.1.content.0", "reason":"prefix_binding_mismatch"}]
    response = p._convert_to_chat_response(native_response(report))
    assert response.metadata["anthropic_input_transformations"] == report
    for block in response.content[:2]:
        assert block.model_dump()["thinking_provenance"] == NATIVE_PROVENANCE
    # Provenance is internal metadata, never an unrecognized vendor field.
    wire = p._convert_messages([{"role":"assistant", "content":[b.model_dump() for b in response.content]}])
    assert "thinking_provenance" not in wire[0]["content"][0]


@pytest.mark.parametrize("report", [None, [], [{"type":"future", "reason":"future", "path":"messages.0.content.0"}]])
def test_report_absent_empty_and_unknown_values_stay_distinct(report):
    response = provider()._convert_to_chat_response(native_response(report))
    metadata = response.metadata or {}
    if report is None:
        assert "anthropic_input_transformations" not in metadata
    else:
        assert metadata["anthropic_input_transformations"] == report


@pytest.mark.parametrize("overload_fallback", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_signature_400_is_not_retried_stripped_or_misclassified_as_overflow(
    overload_fallback, streaming
):
    p = provider(max_retries=3, fallback_on_overload=overload_fallback,
        use_streaming=streaming, persist_fallback_state=False)
    response = MagicMock(status_code=400, headers={})
    error = anthropic.BadRequestError(
        "Invalid `signature` in `thinking` block. The block is bound to a different conversation.",
        response=response, body={"error":{"type":"invalid_request_error", "message":"Invalid signature"}})
    p.client.messages.with_raw_response.create = AsyncMock(side_effect=error)
    # Synchronous entry failure must also be recognized as one permanent SDK
    # generation attempt, not replayed by the overload fallback's outer loop.
    p.client.messages.stream = MagicMock(side_effect=error)
    r = request()
    before = r.model_dump()
    with pytest.raises(Exception) as caught:
        asyncio.run(p.complete(r))
    assert type(caught.value).__name__ == "InvalidRequestError"
    assert caught.value.retryable is False
    assert p.client.messages.with_raw_response.create.call_count == int(not streaming)
    assert p.client.messages.stream.call_count == int(streaming)
    assert r.model_dump() == before


@pytest.mark.parametrize("model", [MODEL, "claude-haiku-5-5", "claude-opus-5-5"])
def test_credential_free_budget_is_unavailable_without_client(model):
    p = AnthropicProvider(api_key=None, config={"default_model":model})
    p._runtime_model_info_cache[MODEL] = _RuntimeModelInfo(max_input_tokens=200000)
    assert asyncio.run(p.request_budget(request(), context_estimate=1000)) is None
    assert p._client is None


@pytest.mark.parametrize("streaming", [False, True])
def test_budget_and_real_send_share_controls_and_stream_reports(streaming):
    p = provider(use_streaming=streaming)
    r = request()
    before = deepcopy(r.model_dump())
    report = [{"type":"thinking_dropped", "path":"messages.1.content.0", "reason":"prefix_binding_mismatch"}]
    p.client.messages.count_tokens = AsyncMock(return_value=SimpleNamespace(input_tokens=20))
    raw = MagicMock(headers={})
    raw.parse = AsyncMock(return_value=native_response(report))
    p.client.messages.with_raw_response.create = AsyncMock(return_value=raw)
    captures = []
    class Stream:
        response = None
        def __aiter__(self):
            async def events():
                yield SimpleNamespace(type="message_start", message=SimpleNamespace(input_transformations=[]))
                yield SimpleNamespace(type="message_delta", input_transformations=report)
            return events()
        async def get_final_message(self):
            return native_response()  # stable SDK dropped beta final-delta field
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
    def stream(**params):
        captures.append(params)
        return Stream()
    p.client.messages.stream = stream
    asyncio.run(p.request_budget(r, context_estimate=1000))
    result = asyncio.run(p.complete(r))
    sent = captures[0] if streaming else p.client.messages.with_raw_response.create.call_args.kwargs
    counted = p.client.messages.count_tokens.call_args.kwargs
    assert counted["thinking"] == sent["thinking"]
    assert counted["messages"] == sent["messages"]
    assert counted["extra_headers"] == sent["extra_headers"]
    assert result.metadata["anthropic_input_transformations"] == report
    assert r.model_dump() == before
    assert len(captures) == int(streaming)
    assert p.client.messages.with_raw_response.create.call_count == int(not streaming)
