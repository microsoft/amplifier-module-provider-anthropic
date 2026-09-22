"""Anthropic on-demand compaction transport; callers own compaction policy.

No automatic triggers, summary prompts, history trimming, or retries live here.
The signed vendor block travels intact in a derived Message metadata envelope.
"""

import copy
import os
import re
from urllib.parse import urlparse

from amplifier_core.llm_errors import ContextLengthError

KEY = "anthropic:compaction"
BETA = "compact-2026-09-04"
# Documented on-demand support, not an inference from a model's context size.
MODELS = re.compile(
    r"^claude-(?:opus-(?:4-6|4-7|4-8|5)|sonnet-(?:4-6|5)|"
    r"fable-5(?:-1)?|mythos-(?:5(?:-1)?|preview))(?:-\d{8})?$"
)


def signed_block(message, model):
    metadata = message.get("metadata") or {}
    if KEY not in metadata:
        return None
    state = metadata[KEY]
    if (
        not isinstance(state, dict)
        or state.get("version") != 1
        or state.get("model") != model
    ):
        raise ValueError("Native Anthropic context has an incompatible model or format")
    block = state.get("block")
    if (
        not isinstance(block, dict)
        or block.get("type") != "compaction"
        or not isinstance(block.get("content"), str)
        or not block["content"]
        or not isinstance(block.get("signature"), str)
        or not block["signature"]
    ):
        raise ValueError("Native Anthropic context requires a complete signed block")
    return copy.deepcopy(block)


def compacted_message(model, block):
    message = {
        # The context fitter retains persisted user reminders. This carrier is
        # never sent as text; assembly restores the vendor's assistant block.
        "role": "user",
        "content": "Native compacted conversation (originals remain in the transcript).",
        "metadata": {
            "ephemeral": True,
            "persisted": True,
            "source": "context-managed",
            KEY: {"version": 1, "model": model, "block": copy.deepcopy(block)},
        },
    }
    signed_block(message, model)
    return message


def add_beta(params):
    headers = dict(params.get("extra_headers") or {})
    values = [
        value.strip()
        for value in headers.get("anthropic-beta", "").split(",")
        if value.strip()
    ]
    headers["anthropic-beta"] = ",".join(dict.fromkeys([*values, BETA]))
    params["extra_headers"] = headers


def extract_checkpoint(request, model):
    """Remove only the carrier; never convert its public label into model input."""
    block = None
    messages = []
    seen_conversation = False
    for message in request.messages:
        current = signed_block({"metadata": message.metadata}, model)
        if current is not None:
            if block is not None or seen_conversation:
                raise ValueError(
                    "Native Anthropic context must precede conversation history"
                )
            block = current
        else:
            messages.append(message)
            if message.role not in {"system", "developer"}:
                seen_conversation = True
    return request.model_copy(update={"messages": messages}), block


def compaction_usage(usage):
    """On-demand usage lives in iterations; the top-level counters are zero."""
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(exclude_none=True)
    if not isinstance(usage, dict):
        return {}
    rows = usage.get("iterations")
    if not isinstance(rows, list) or not rows:
        return {}  # Do not report a free compaction from top-level zeroes.
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for field, target in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("cache_read_input_tokens", "cache_read_tokens"),
            ("cache_creation_input_tokens", "cache_write_tokens"),
        ):
            value = row.get(field)
            if type(value) is int and value >= 0:
                result[target] = result.get(target, 0) + value
    # Core includes reads in input, but keeps cache writes separate.
    if "input_tokens" in result:
        result["input_tokens"] += result.get("cache_read_tokens", 0)
    return result


class NativeCompactionMixin:
    # Compaction preserves provider thinking state, which is only valid with
    # the actual system prompt and tool definitions used for continuation.
    native_compaction_requires_request_context = True

    def _validate_native_overrides(self):
        if {
            "messages",
            "system",
            "tools",
            "model",
            "compaction",
            "context_management",
        } & self.extra_request_params.keys():
            raise ValueError(
                "Native compaction cannot use overrides of its history, model or request context"
            )

    def _supports_compaction_model(self, model):
        endpoint = (
            getattr(self._client, "base_url", None)
            if self._client is not None
            else self._base_url
            or os.environ.get("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com"
        )
        parsed = urlparse(str(endpoint))
        return bool(
            MODELS.fullmatch(model)
            and parsed.scheme == "https"
            and parsed.hostname == "api.anthropic.com"
            and parsed.port in (None, 443)
            and parsed.path in {"", "/"}
        )

    def supports_native_compaction(self):
        return self._supports_compaction_model(self.default_model)

    def validate_compacted_context(self, message):
        return (
            self.supports_native_compaction()
            and signed_block(message, self.default_model) is not None
        )

    async def compact_context(self, request):
        metadata = request.metadata or {}
        if (
            metadata.get("purpose") == "context-compaction"
            and metadata.get("native_compaction_request_context") is not True
        ):
            # Older context-managed versions assemble a messages-only request.
            # Fail before transport so they safely use their semantic fallback.
            raise ValueError(
                "Native Anthropic compaction needs the full continuation request context"
            )
        model = request.model or self.default_model
        if not self._supports_compaction_model(model):
            raise NotImplementedError(
                "This Anthropic endpoint/model does not advertise on-demand compaction"
            )
        self._validate_native_overrides()
        pending = set()
        for message in request.messages:
            row = message.model_dump()
            for call in row.get("tool_calls") or []:
                pending.add(call.get("id"))
            for block in (
                row.get("content") if isinstance(row.get("content"), list) else []
            ):
                if block.get("type") in {"tool_call", "tool_use"}:
                    pending.add(block.get("id"))
                elif block.get("type") == "tool_result":
                    pending.discard(
                        block.get("tool_use_id") or block.get("tool_call_id")
                    )
            if row.get("role") == "tool":
                pending.discard(row.get("tool_call_id"))
        if pending:
            raise ValueError("Complete pending tool results before native compaction")
        caps = await self._get_request_capabilities(model)
        assembly = self._assemble_request_params(
            request, request_options={"model": model}, request_caps=caps
        )
        if assembly is None:
            raise ValueError("Could not assemble native compaction request")
        params = copy.deepcopy(assembly.params)
        # Prefix compaction often ends on an assistant turn. Anthropic rejects
        # trailing whitespace there even in summarization mode. Normalize only
        # the copied terminal text, never canonical history or signed blocks.
        if params["messages"] and params["messages"][-1]["role"] == "assistant":
            last = params["messages"][-1]
            if isinstance(last["content"], str):
                last["content"] = last["content"].rstrip()
            elif last["content"] and last["content"][-1].get("type") == "text":
                last["content"][-1]["text"] = last["content"][-1]["text"].rstrip()
        # These generation controls are rejected by the on-demand API. Keep
        # system, tools, thinking and the caller's output cap exactly as assembled.
        params.pop("stop_sequences", None)
        params.pop("tool_choice", None)
        params.pop("stream", None)
        extra = params.setdefault("extra_body", {})
        if "context_management" in extra or "context_management" in params:
            raise ValueError(
                "On-demand compaction cannot be combined with context_management"
            )
        for container in (params, extra):
            container.pop("stop_sequences", None)
            container.pop("tool_choice", None)
            if isinstance(container.get("output_config"), dict):
                container["output_config"].pop("format", None)
        add_beta(params)
        count = await self.client.messages.count_tokens(
            **self._count_tokens_params(params)
        )
        input_tokens = getattr(count, "input_tokens", None)
        if type(input_tokens) is not int or input_tokens < 0:
            raise ValueError("Native compaction input could not be measured")
        limit = self._budget_input_limit(model, caps)
        if limit is not None and input_tokens > limit:
            raise ContextLengthError(
                "Native compaction input exceeds the model input limit",
                provider=self.name,
            )
        extra["compaction"] = {"type": "summarize"}
        # extra_body supports the 1.0 SDK floor even before this beta is typed.
        response = await self.client.messages.create(**params)
        blocks = [
            block.model_dump(mode="json", exclude_unset=True, exclude_none=True)
            if hasattr(block, "model_dump")
            else copy.deepcopy(block)
            for block in response.content
        ]
        if response.stop_reason != "compaction" or len(blocks) != 1:
            raise ValueError(
                "Native compaction did not return a completed signed summary"
            )
        return {
            "kind": "native",
            "message": compacted_message(model, blocks[0]),
            "usage": compaction_usage(getattr(response, "usage", None)),
            "input_tokens": input_tokens,
        }
