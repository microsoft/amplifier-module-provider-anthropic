"""Closed scalar count-failure details, never arbitrary API text or payloads.

Only two ordinary JSON body envelopes and complete anchored validation messages
are recognized. Unknown vendor/proxy grammar stays explicitly unknown. No
recursive traversal, serialization, exception formatting or request access.
"""
from __future__ import annotations

import re
from typing import Any

_API_TYPES = frozenset({"invalid_request_error", "authentication_error", "permission_error",
    "not_found_error", "rate_limit_error", "api_error", "overloaded_error"})
_PATH = re.compile(
    r"(?:model|thinking(?:\.(?:type|budget_tokens|block_binding(?:\.prefix_mismatch_behavior)?))?"
    r"|output_config(?:\.(?:effort|format))?|tool_choice(?:\.type)?"
    r"|messages\.[0-9]{1,9}\.content(?:\.[0-9]{1,9}\.(?:type|thinking|signature|text|tool_use_id))?"
    r"|tools\.[0-9]{1,9}\.(?:name|input_schema))", re.ASCII)
_REASONS = {
    "Field required": "required_field",
    "Unable to extract tag using discriminator 'type'": "missing_discriminator",
    "Extra inputs are not permitted": "extra_field",
    "Input should be a valid string": "invalid_type",
    "Input should be a valid integer": "invalid_type",
    "Input should be a valid dictionary": "invalid_type",
    "Input should be a valid list": "invalid_type",
}


def validation_detail(body: Any) -> dict[str, Any]:
    """Return only constants or allowlisted structural paths with indices erased."""
    result: dict[str, Any] = {"diagnostic_version": 1, "api_error_type": "unknown",
        "reason_code": "unknown", "schema_path": "unknown", "detail_status": "malformed"}
    # Exact builtins only: don't call user/proxy mapping methods or stringify.
    if type(body) is not dict:
        return result
    inner = body.get("error", body)
    if type(inner) is not dict:
        return result
    api_type = inner.get("type")
    if type(api_type) is str and len(api_type) <= 64 and api_type in _API_TYPES:
        result["api_error_type"] = api_type
    message = inner.get("message")
    if type(message) is not str:
        return result
    if len(message) > 1024:
        result["detail_status"] = "oversize"
        return result
    result["detail_status"] = "unrecognized"
    # A full suffix match is intentional: messages that echo an offending value,
    # custom schema key, tool name, URL or other private text must not be parsed.
    path, separator, reason = message.partition(": ")
    if not separator or reason not in _REASONS or not _PATH.fullmatch(path):
        return result
    result.update(reason_code=_REASONS[reason], schema_path=re.sub(r"\.[0-9]+", ".[]", path),
                  detail_status="recognized")
    return result


def bounded_identifier(value: Any, maximum: int) -> str | None:
    """Existing public correlation fields stay bounded; no arbitrary coercion."""
    if type(value) is str and 0 < len(value) <= maximum and value.isascii() and re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        return value
    return None
