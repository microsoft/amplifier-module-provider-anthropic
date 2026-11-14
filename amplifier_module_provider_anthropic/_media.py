"""Provider-owned image wire/diagnostic boundaries.

This module knows Anthropic content locations, not stores, paths or ref authority.
It never parses strings or traverses arbitrary tool inputs/opaque reasoning.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from amplifier_core.llm_errors import InvalidRequestError


NAMESPACE = "experimental.media.v1"
# Explicit model identities from the vendor's vision supportedModels catalogue,
# retrieved 2026-11-13. No family/prefix inference for unknown future models.
# https://platform.claude.com/docs/en/build-with-claude/vision.md
VISION_MODELS = frozenset({
    "claude-fable-5-2", "claude-mythos-5-2", "claude-fable-5-1",
    "claude-mythos-5-1", "claude-fable-5", "claude-mythos-5",
    "claude-mythos-preview", "claude-opus-5-5", "claude-opus-5",
    "claude-sonnet-5-5", "claude-sonnet-5", "claude-haiku-5-5",
    "claude-haiku-4-5-20251001", "claude-opus-4-8", "claude-opus-4-7",
    "claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
})


NOT_RETAINED = (
    "[Image transport not retained in diagnostics; no retrievable pixel reference]"
)


def has_media(request: Any) -> bool:
    """Recognize only canonical media positions, never arbitrary JSON text."""
    for message in request.messages:
        metadata = getattr(message, "metadata", None)
        if isinstance(metadata, dict) and NAMESPACE in metadata:
            return True
        content = getattr(message, "content", None)
        if isinstance(content, list):
            for raw in content:
                block = raw.model_dump() if hasattr(raw, "model_dump") else raw
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        return True
                    if block.get("type") == "tool_result":
                        nested = block.get("output", block.get("content"))
                        if isinstance(nested, list) and any(
                            isinstance(part, dict) and part.get("type") == "image"
                            for part in nested
                        ):
                            return True
    return False


def has_refs(request: Any) -> bool:
    return any(
        isinstance(getattr(message, "metadata", None), dict)
        and NAMESPACE in message.metadata for message in request.messages
    )


def has_capability(coordinator: Any) -> bool:
    getter = getattr(coordinator, "get_capability", None)
    capability = getter(NAMESPACE) if callable(getter) else None
    return capability is not None and getattr(capability, "version", None) == 1


def tool_result_content(content: Any, *, allow_images: bool = True) -> Any:
    """Normalize Core text/image parts at the original tool-result position."""
    if not isinstance(content, list):
        return deepcopy(content)
    result = []
    for raw in content:
        block = raw.model_dump() if hasattr(raw, "model_dump") else raw
        if not isinstance(block, dict):
            raise InvalidRequestError(
                "Tool-result content requires structured text/image parts.",
                provider="anthropic",
                status_code=400,
            )
        if block.get("type") == "text":
            result.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            if not allow_images:
                result.append({
                    "type": "text",
                    "text": "[Image unavailable: selected model vision support is unknown; no pixels supplied]",
                })
                continue
            source = block.get("source")
            if not isinstance(source, dict) or source.get("type") != "base64":
                # Forward media-v1 is verified static PNG/JPEG, never a URL fetch.
                result.append({
                    "type": "text",
                    "text": "[Image unavailable: unsupported image source; no pixels supplied]",
                })
                continue
            if (
                source.get("media_type") not in {"image/png", "image/jpeg"}
                or not isinstance(source.get("data"), str)
                or not source["data"]
            ):
                raise InvalidRequestError(
                    "Tool-result image requires verified PNG/JPEG base64 transport.",
                    provider="anthropic",
                    status_code=400,
                )
            result.append({
                "type": "image",
                "source": {
                    key: source[key] for key in ("type", "media_type", "data")
                },
            })
        else:
            # Other provider-owned blocks retain their existing semantics.
            result.append(deepcopy(block))
    return result


def _diagnostic_content(content: Any) -> Any:
    if not isinstance(content, list):
        return deepcopy(content)
    result = []
    for block in content:
        if not isinstance(block, dict):
            result.append(deepcopy(block))
        elif block.get("type") == "image":
            # Omission is deliberately not a fake digest/ref that a reader can
            # mistake for retained pixels. Drop the WHOLE image transport copy.
            result.append({"type": "text", "text": NOT_RETAINED})
        elif block.get("type") == "tool_result":
            copied = deepcopy(block)
            for key in ("content", "output"):
                if key in copied:
                    copied[key] = _diagnostic_content(copied[key])
            result.append(copied)
        elif block.get("type") == "document":
            copied = deepcopy(block)
            source = copied.get("source")
            if isinstance(source, dict) and source.get("type") == "content":
                source["content"] = _diagnostic_content(source.get("content"))
            result.append(copied)
        else:
            result.append(deepcopy(block))
    return result


def diagnostic_copy(payload: dict[str, Any]) -> dict[str, Any]:
    """Omit known image wire fields BEFORE a Rust-backed hooks.emit call."""
    copied = deepcopy(payload)
    messages = copied.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                message["content"] = _diagnostic_content(message["content"])
    for key in ("content", "system"):
        if key in copied:
            copied[key] = _diagnostic_content(copied[key])
    extra = copied.get("extra_body")
    if isinstance(extra, dict):
        copied["extra_body"] = diagnostic_copy(extra)
    return copied


def safe_error_text(text: str, params: dict[str, Any]) -> str:
    """Remove exact known pixel transport values if an SDK echoes the input."""
    def scrub(content: Any) -> None:
        nonlocal text
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                source = block.get("source")
                if isinstance(source, dict):
                    for key in ("data", "url"):
                        value = source.get(key)
                        if isinstance(value, str) and value:
                            text = text.replace(value, NOT_RETAINED)
            elif block.get("type") == "tool_result":
                scrub(block.get("content"))
            elif block.get("type") == "document":
                source = block.get("source")
                if isinstance(source, dict) and source.get("type") == "content":
                    scrub(source.get("content"))

    for envelope in (params, params.get("extra_body")):
        if not isinstance(envelope, dict):
            continue
        for message in envelope.get("messages", []):
            if isinstance(message, dict):
                scrub(message.get("content"))
        scrub(envelope.get("system"))
    return text


def validate_attempt(coordinator: Any, request: Any, purpose: str, options: Any) -> None:
    """Call the host's private request witness; never turn refusal into retry.

    Legacy unwrapped requests retain legacy behavior without a media capability.
    With a real v1 capability the exact projected Core object is the private key.
    """
    getter = getattr(coordinator, "get_capability", None)
    capability = getter(NAMESPACE) if callable(getter) else None
    if capability is None or getattr(capability, "version", None) != 1:
        if has_refs(request):
            raise InvalidRequestError(
                "Session image references require compatible private preparation; "
                "no pixels supplied and request not sent.",
                provider="anthropic", retryable=False,
            )
        return
    validator = getattr(capability, "validate", None)
    if not callable(validator):
        raise InvalidRequestError(
            "Media validation unavailable; request not sent.", provider="anthropic"
        )
    try:
        result = validator(request, purpose=purpose, request_options=dict(options))
    except Exception:
        # Policy exceptions must not enter provider retry/backoff/overflow or
        # escape with arbitrary byte-bearing exception text.
        raise InvalidRequestError(
            "Media projection expired, changed or is no longer authorized; "
            "request not sent. Prepare and fit a new authorized request.",
            provider="anthropic", retryable=False,
        ) from None
    if not isinstance(result, dict) or result.get("valid") is not True:
        raise InvalidRequestError(
            "Media validation refused; request not sent.", provider="anthropic",
            retryable=False,
        )