"""Small ordinary-message image projection and diagnostic-copy helpers.

No URL fetching, image decoding, model discovery, or saved-history rewriting.
"""

from __future__ import annotations

import base64
import binascii
import copy
import json
from typing import Any
from urllib.parse import urlsplit


UNAVAILABLE = (
    "[Image unavailable: this image source or message position is not supported "
    "by this adapter. Supply a base64 JPEG/PNG/GIF/WebP or HTTP(S) URL in a user "
    "message or a call-bound tool result; do not infer its contents.]"
)
OMITTED = "[Image payload omitted from diagnostics]"


def mapping(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    return dict(value) if isinstance(value, dict) else None


def unavailable() -> dict[str, str]:
    return {"type": "text", "text": UNAVAILABLE}


def image_block(block: dict[str, Any]) -> dict[str, Any]:
    """Project a documented native source, or mark its original slot unavailable."""
    source = block.get("source")
    if not isinstance(source, dict):
        return unavailable()
    kind = source.get("type")
    if kind == "base64":
        mime = source.get("media_type", "image/jpeg")
        data = source.get("data")
        if mime not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            return unavailable()
        if not isinstance(data, str) or not data:
            return unavailable()
        try:
            base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error):
            return unavailable()
        native = {"type": "base64", "media_type": mime, "data": data}
    elif kind == "url":
        url = source.get("url")
        if not isinstance(url, str) or not url or any(c.isspace() for c in url):
            return unavailable()
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                return unavailable()
        except ValueError:
            return unavailable()
        native = {"type": "url", "url": url}
    else:
        return unavailable()
    result = {"type": "image", "source": native}
    for key in ("cache_control", "transformations"):
        if key in block:
            result[key] = copy.deepcopy(block[key])
    return result


def text_position(content: Any) -> str:
    """Keep structured system/developer text and same-slot image unavailability."""
    if isinstance(content, str):
        return content
    parts = []
    for raw in content or []:
        block = mapping(raw)
        if block and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block and block.get("type") == "image":
            parts.append(UNAVAILABLE)
    return "\n".join(parts)


def tool_content(output: Any) -> Any:
    """Unwrap known ToolResult content envelopes, not arbitrary application data."""
    block = mapping(output)
    if block is not None:
        if block.get("type") == "image":
            return [image_block(block)]
        if block.get("type") == "text":
            return [{"type": "text", "text": block.get("text", "")}]
        if block.get("type") == "tool_result":
            return tool_content(block.get("output", block.get("content", "")))
        if content_envelope(block):
            return tool_content(block["content"])
        if output_envelope(block):
            return tool_content(block["output"])
        if image_values(block):
            return [unavailable()]
    if not isinstance(output, list):
        # Preserve the old scalar path; ordinary structured output must be text,
        # never a Python repr of image-bearing content.
        return output if isinstance(output, str) else json.dumps(output)
    if not rich_content(output):
        return [unavailable()] if image_values(output) else json.dumps(output)
    converted = []
    for raw in output:
        item = mapping(raw)
        if item and item.get("type") == "image":
            converted.append(image_block(item))
        elif item and item.get("type") == "text":
            converted.append({"type": "text", "text": item.get("text", "")})
        elif item and item.get("type") in ("document", "search_result"):
            converted.append(copy.deepcopy(item))
        else:
            # Unsupported rich output stays visible instead of becoming pixels
            # encoded as JSON text. Never move it to a fabricated user turn.
            converted.append(unavailable())
    return converted


def rich_content(value: Any) -> bool:
    return isinstance(value, list) and all(
        (item := mapping(raw)) is not None
        and item.get("type") in ("text", "image", "document", "search_result", "tool_result")
        for raw in value
    )


def content_envelope(value: dict[str, Any]) -> bool:
    return (
        set(value) <= {"content", "success", "error", "is_error", "metadata"}
        and rich_content(value.get("content"))
    )


def output_envelope(value: dict[str, Any]) -> bool:
    """Recognize the Core ToolResult fields, not application/business mappings."""
    return (
        set(value) <= {"output", "success", "error", "is_error"}
        and "output" in value
        and type(value.get("success")) is bool
        and (value.get("error") is None or isinstance(value["error"], dict))
    )


def image_values(value: Any) -> tuple[str, ...]:
    """Collect only explicit image-source values for diagnostic echo omission."""
    values = []
    if isinstance(value, dict):
        if value.get("type") == "image":
            source = value.get("source")
            if isinstance(source, dict):
                values.extend(
                    source[key] for key in ("data", "url")
                    if isinstance(source.get(key), str) and source[key]
                )
        for child in value.values():
            values.extend(image_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(image_values(child))
    return tuple(values)


def image_echoes(*payloads: Any) -> tuple[str, ...]:
    """Prepare literal and JSON-escaped image strings once for this request."""
    values = {value for payload in payloads for value in image_values(payload)}
    variants = values.copy()
    for value in values:
        if value.isascii() and value.isprintable() and '"' not in value and "\\" not in value:
            continue  # Base64 and ordinary ASCII URLs already have their JSON spelling.
        variants.add(json.dumps(value, ensure_ascii=False)[1:-1])
        variants.add(json.dumps(value)[1:-1])
    # JSON permits escaped slashes in URLs and base64 text.
    variants |= {value.replace("/", r"\/") for value in variants if "/" in value}
    return tuple(sorted(variants, key=len, reverse=True))


def diagnostic_copy(value: Any, images: tuple[str, ...] = ()) -> Any:
    """Omit image objects before any subscriber, including exact textual echoes.

    This operates only on an observation copy, never the SDK payload. Image-shaped
    objects inside arbitrary diagnostic metadata are omitted conservatively too.
    """
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {"type": "text", "text": OMITTED}
        return {key: diagnostic_copy(child, images) for key, child in value.items()}
    if isinstance(value, list):
        return [diagnostic_copy(child, images) for child in value]
    if isinstance(value, str):
        for image in images:
            value = value.replace(image, OMITTED)
        return value
    return copy.deepcopy(value)


def sanitize_image_error(error: Exception, images: tuple[str, ...]) -> None:
    """Detach SDK errors from image requests before public reporting or retry."""
    if not images:
        return
    error.args = tuple(diagnostic_copy(arg, images) for arg in error.args)
    for name in ("message", "body"):
        if hasattr(error, name):
            setattr(error, name, diagnostic_copy(getattr(error, name), images))
    # Even a clean HTTP error retains the outgoing media in its SDK request.
    error.__cause__ = error.__context__ = None
    error.__suppress_context__ = True
