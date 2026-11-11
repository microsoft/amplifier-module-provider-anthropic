"""Stateless wire policy for Anthropic's server-validated thinking bindings.

The server owns prefix/model/account compatibility. Never guess it from the
current model, cache snapshots, or replay old context to make a signature valid.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import re
from typing import Any
from urllib.parse import urlparse

BINDING_BETA = "thinking-binding-controls-2026-08-01"
PROVENANCE_KEY = "thinking_provenance"
NATIVE_PROVENANCE = {"version": 1, "provider": "anthropic"}

# Documented preserved-thinking models, not a speculative future-version gate.
# https://platform.claude.com/docs/en/build-with-claude/preserved-thinking
_BINDING_MODELS = re.compile(
    r"claude-(?:(?:fable|mythos)-5-[12]|(?:opus|sonnet|haiku)-5-5)(?:-\d{8})?\Z"
)


def _first_party(base_url: str) -> bool:
    try:
        url = urlparse(base_url)
        return (
            url.scheme == "https"
            and url.hostname == "api.anthropic.com"
            and url.port in {None, 443}
            and url.path in {"", "/"}
            and not (url.username or url.password or url.query or url.fragment)
        )
    except (TypeError, ValueError):
        return False


def supports_binding(model: Any, base_url: str | Callable[[], str]) -> bool:
    """Only documented models on the SDK-resolved first-party endpoint."""
    return (
        isinstance(model, str) and bool(_BINDING_MODELS.fullmatch(model))
        and _first_party(base_url() if callable(base_url) else base_url)
    )


def configure_binding(
    params: dict[str, Any], *,
    base_url: str | Callable[[], str], policy: str | None = "auto",
    default_headers: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
) -> None:
    """Apply once to the final assembled envelope, shared by count and send.

    ``auto`` uses selective server dropping only on documented first-party
    models and accepted thinking modes. None leaves vendor defaults untouched.
    Explicit native block_binding always wins, including null/unknown values.
    Explicit drop_block/error opts a custom endpoint into the native contract.
    No request history, signatures, retries, or tool execution are retained.
    """
    if policy is not None and (
        not isinstance(policy, str) or policy not in {"auto", "drop_block", "error"}
    ):
        raise ValueError(
            "thinking_prefix_mismatch_behavior must be auto, drop_block, error, or null"
        )
    extra = params.get("extra_body")
    extra = extra if isinstance(extra, dict) else {}
    model = extra.get("model", params.get("model"))
    thinking = extra.get("thinking", params.get("thinking"))
    explicit = isinstance(thinking, dict) and "block_binding" in thinking
    if policy is None:
        return
    if policy == "auto" and not supports_binding(model, base_url):
        return
    if not explicit and thinking is None and (
        "thinking" in params or "thinking" in extra
    ):
        return  # Explicit null is an expert override, not an omitted default.
    # Some models reject block_binding with disabled/between_tools. Never
    # enable thinking or override a caller's mode just to use the beta.
    if not explicit and thinking is not None and (
        not isinstance(thinking, dict) or thinking.get("type") not in {None, "adaptive"}
    ):
        return
    if (
        not explicit
        and isinstance(model, str)
        and model.startswith(("claude-sonnet-5-5", "claude-haiku-5-5"))
        and (not isinstance(thinking, dict) or thinking.get("type") != "adaptive")
    ):
        return
    headers = dict(params.get("extra_headers") or {})
    beta_keys = [key for key in headers if key.lower() == "anthropic-beta"]
    defaults = default_headers() if callable(default_headers) else default_headers
    default_beta = next(
        (value for key, value in (defaults or {}).items()
         if key.lower() == "anthropic-beta"), ""
    )
    beta = headers[beta_keys[-1]] if beta_keys else default_beta
    if not isinstance(beta, str):
        return  # Leave malformed expert overrides untouched, atomically.
    # An explicit empty header suppresses SDK beta defaults as well. Do not
    # silently re-enable a control the caller removed.
    if beta_keys and not beta.strip():
        return
    if not explicit:
        thinking = deepcopy(thinking) if isinstance(thinking, dict) else {}
        thinking["block_binding"] = {
            "prefix_mismatch_behavior": "drop_block" if policy == "auto" else policy
        }
        if "thinking" in extra:
            params["extra_body"] = {**extra, "thinking": thinking}
        else:
            params["thinking"] = thinking
    values = [value.strip() for value in beta.split(",") if value.strip()]
    if BINDING_BETA not in values:
        values.append(BINDING_BETA)
    for key in beta_keys:
        del headers[key]
    headers["anthropic-beta"] = ",".join(dict.fromkeys(values))
    params["extra_headers"] = headers


def is_foreign(block: Mapping[str, Any]) -> bool:
    """Only a recognized explicit origin disqualifies legacy signed history."""
    origin = block.get(PROVENANCE_KEY)
    return (
        isinstance(origin, dict)
        and origin.get("version") == 1
        and isinstance(origin.get("provider"), str)
        and bool(origin["provider"])
        and origin["provider"] != "anthropic"
    )


def input_transformations(value: Any) -> list[dict[str, Any]] | None:
    """Preserve vendor reports without prompts/signatures or schema guessing.

    Unknown future types/reasons remain observable. None (absent) differs from
    [] (server explicitly reports no transformations). Ignore mock attributes.
    """
    if not isinstance(value, list):
        return None
    result = []
    for item in value:
        if hasattr(item, "model_dump"):
            item = item.model_dump(exclude_none=True)
        if isinstance(item, dict):
            result.append(deepcopy(item))
    return result
