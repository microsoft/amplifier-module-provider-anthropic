"""Compaction transport, count and continuation without network or real history."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_core.llm_errors import ContextLengthError
from amplifier_core.message_models import ToolSpec

from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic.compaction import BETA, KEY, compacted_message

MODEL = "claude-sonnet-5"
BLOCK = {
    "type": "compaction",
    "content": "Keep the user's corrected requirement.",
    "signature": "signed-fixture",
}


def provider(**config):
    p = AnthropicProvider(
        api_key="test",
        config={
            "default_model": MODEL,
            "extended_thinking": False,
            "shared_rate_limit_state_path": "",
            **config,
        },
    )
    p._client = MagicMock(base_url="https://api.anthropic.com/")
    p._client.messages.count_tokens = AsyncMock(
        return_value=SimpleNamespace(input_tokens=500)
    )
    p._client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[copy.deepcopy(BLOCK)],
            stop_reason="compaction",
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "iterations": [
                    {"type": "compaction", "input_tokens": 500, "output_tokens": 40}
                ],
            },
        )
    )
    p._get_request_capabilities = AsyncMock(
        return_value=p._get_capabilities(p.default_model)
    )
    return p


def request(messages=None):
    return ChatRequest(
        messages=messages
        or [
            Message(role="system", content="Keep decisions."),
            Message(role="user", content="Before correction"),
            Message(role="assistant", content="Prior answer"),
            Message(role="user", content="Corrected requirement"),
        ],
        tools=[
            ToolSpec(
                name="lookup", description="Lookup only", parameters={"type": "object"}
            )
        ],
        max_output_tokens=2048,
    )


def assemble(p, req):
    return p._assemble_request_params(
        req, request_options={}, request_caps=p._get_capabilities(MODEL)
    ).params


@pytest.mark.asyncio
async def test_compact_count_continue_and_recompact_preserve_signed_block_and_history():
    p, req = provider(), request()
    original = req.model_dump()
    result = await p.compact_context(req)
    assert req.model_dump() == original
    assert result["usage"] == {"input_tokens": 500, "output_tokens": 40}
    sent = p.client.messages.create.call_args.kwargs
    assert sent["extra_body"]["compaction"] == {"type": "summarize"}
    assert BETA in sent["extra_headers"]["anthropic-beta"]
    assert sent["system"] and sent["tools"] and sent["max_tokens"] == 2048
    assert "compaction" not in p.client.messages.count_tokens.call_args.kwargs.get(
        "extra_body", {}
    )
    carrier = result["message"]
    # JSON persisted/reloaded Message follows the identical assembly path.
    tail = Message(role="user", content="Continue after compaction")
    continued = request(
        [req.messages[0], Message(**json.loads(json.dumps(carrier))), tail]
    )
    wire = assemble(p, continued)
    assert wire["messages"][0] == {"role": "assistant", "content": [BLOCK]}
    assert "Native compacted" not in str(wire)
    assert BETA in wire["extra_headers"]["anthropic-beta"]
    decision = await p.request_budget(continued, context_estimate=0)
    assert decision["measurement"]["input_tokens"] == 500
    assert (
        p.client.messages.count_tokens.call_args.kwargs["messages"] == wire["messages"]
    )
    await p.compact_context(continued)
    assert p.client.messages.create.call_args.kwargs["messages"][0]["content"] == [
        BLOCK
    ]
    assert carrier == result["message"]
    assert req.model_dump() == original


@pytest.mark.asyncio
async def test_sdk_model_excludes_synthetic_null_fields():
    # SDK model_construct preserves unknown beta properties without requiring a new SDK.
    from anthropic.types import TextBlock

    p = provider()
    p.client.messages.create.return_value.content = [TextBlock.model_construct(**BLOCK)]
    result = await p.compact_context(request())
    assert result["message"]["metadata"][KEY]["block"] == BLOCK


@pytest.mark.parametrize(
    "model,endpoint,expected",
    [
        (MODEL, "https://api.anthropic.com/", True),
        ("claude-opus-4-6", "https://api.anthropic.com/", True),
        ("claude-haiku-4-5", "https://api.anthropic.com/", False),
        ("claude-future-9", "https://api.anthropic.com/", False),
        (MODEL, "https://proxy.example/", False),
    ],
)
def test_capability_is_model_and_endpoint_scoped(model, endpoint, expected):
    p = provider(default_model=model)
    p._client.base_url = endpoint
    assert p.supports_native_compaction() is expected


@pytest.mark.parametrize(
    "bad", [None, {}, {**BLOCK, "signature": ""}, {**BLOCK, "type": "text"}]
)
@pytest.mark.asyncio
async def test_missing_or_malformed_summary_never_returns_checkpoint(bad):
    p, req = provider(), request()
    original = req.model_dump()
    p.client.messages.create.return_value.content = [] if bad is None else [bad]
    with pytest.raises(ValueError):
        await p.compact_context(req)
    assert req.model_dump() == original


@pytest.mark.asyncio
async def test_cancellation_keeps_originals():
    p, req = provider(), request()
    original = req.model_dump()
    p.client.messages.create.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await p.compact_context(req)
    assert req.model_dump() == original


@pytest.mark.asyncio
async def test_overflow_is_not_dispatched():
    p = provider()
    p.client.messages.count_tokens.return_value.input_tokens = 9_000_000
    with pytest.raises(ContextLengthError):
        await p.compact_context(request())
    p.client.messages.create.assert_not_called()


@pytest.mark.parametrize("case", ["mismatched-model", "duplicate", "misplaced", "null-envelope"])
def test_reject_invalid_checkpoint_before_transport(case):
    p = provider()
    carrier = Message(**compacted_message(MODEL, BLOCK))
    messages = [carrier, Message(role="user", content="new")]
    if case == "mismatched-model":
        carrier.metadata[KEY]["model"] = "claude-opus-5"
    elif case == "null-envelope":
        carrier.metadata[KEY] = None
    elif case == "duplicate":
        messages.insert(1, carrier)
    else:
        messages.reverse()
    with pytest.raises(ValueError):
        assemble(p, request(messages))
    p.client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_missing_tool_result_is_not_synthesized():
    from amplifier_core.message_models import ToolCallBlock

    p = provider()
    req = request(
        [
            Message(
                role="assistant",
                content=[ToolCallBlock(id="pending", name="lookup", input={})],
            )
        ]
    )
    with pytest.raises(ValueError, match="pending"):
        await p.compact_context(req)
    p.client.messages.count_tokens.assert_not_called()
    p.client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_model_override_is_used_for_compaction():
    p = provider()
    req = request().model_copy(update={"model": "claude-opus-5"})
    result = await p.compact_context(req)
    assert p.client.messages.create.call_args.kwargs["model"] == "claude-opus-5"
    assert result["message"]["metadata"][KEY]["model"] == "claude-opus-5"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key", ["messages", "system", "tools", "model", "compaction", "context_management"]
)
async def test_request_context_overrides_are_rejected_for_compact_and_continuation(key):
    p = provider(extra_request_params={key: "unrelated override"})
    with pytest.raises(ValueError, match="overrides"):
        await p.compact_context(request())
    with pytest.raises(ValueError, match="overrides"):
        assemble(
            p,
            request(
                [
                    Message(**compacted_message(MODEL, BLOCK)),
                    Message(role="user", content="Continue"),
                ]
            ),
        )
    p.client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_incomplete_compaction_does_not_install_partial_state():
    p = provider()
    p.client.messages.create.return_value.stop_reason = "max_tokens"
    with pytest.raises(ValueError, match="completed"):
        await p.compact_context(request())


@pytest.mark.asyncio
async def test_previously_repaired_but_missing_tool_result_still_blocks_compaction():
    from amplifier_core.message_models import ToolCallBlock

    p = provider()
    p._repaired_tool_ids.add("pending")
    req = request(
        [
            Message(
                role="assistant",
                content=[ToolCallBlock(id="pending", name="lookup", input={})],
            )
        ]
    )
    with pytest.raises(ValueError, match="pending"):
        await p.compact_context(req)
    p.client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_assistant_whitespace_is_normalized_only_on_wire():
    p = provider()
    req = request(
        [
            Message(role="user", content="Fixture"),
            Message(role="assistant", content="Completed evidence. \n"),
        ]
    )
    original = req.model_dump()
    await p.compact_context(req)
    created = p.client.messages.create.call_args.kwargs["messages"]
    counted = p.client.messages.count_tokens.call_args.kwargs["messages"]
    assert created == counted
    content = created[-1]["content"]
    assert (
        content if isinstance(content, str) else content[-1]["text"]
    ) == "Completed evidence."
    assert req.model_dump() == original


@pytest.mark.asyncio
async def test_older_context_manager_cannot_silently_omit_system_and_tools():
    p = provider()
    req = request().model_copy(
        update={"metadata": {"purpose": "context-compaction", "stream": False}}
    )
    with pytest.raises(ValueError, match="full continuation"):
        await p.compact_context(req)
    p.client.messages.count_tokens.assert_not_called()
    p.client.messages.create.assert_not_called()
