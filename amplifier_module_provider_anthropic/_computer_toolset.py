"""Legacy computer-use tool <-> ``computer_toolset_20260801`` translation.

Anthropic's Opus 5.5 migration guidance ("Migrate from computer_20251124")
replaces the legacy versioned ``computer_*`` native tool type with a single
``computer_toolset_20260801`` declaration. The wire shapes differ in two
places this provider must translate so that an existing Amplifier tool that
dispatches on ``{"name": "computer", "arguments": {"action": <member>, ...}}``
(the shape every prior ``computer_*`` generation already produced) keeps
working unchanged on Opus 5.5+:

1. **Tool declaration.** A legacy ``{"type": "computer_20251124", "name":
   "computer", "display_width_px": ..., ...}`` entry becomes
   ``{"type": "computer_toolset_20260801", ...optional configs}``. The
   legacy type/name/dimension/enable_zoom/defer_loading keys are dropped
   (rejected as "Extra inputs are not permitted" on the toolset schema);
   ``cache_control`` is kept if present.
2. **Tool-use / tool-result shape.** Anthropic's toolset schema emits
   ``tool_use`` blocks per *member action* directly (``name`` is the member,
   e.g. ``"left_click"``, and ``toolset_name`` names the toolset,
   ``"computer"``) instead of the legacy single ``name: "computer"`` with
   ``input.action``. This module translates a wire toolset call back into
   the legacy ``name="computer", arguments={"action": <member>, **input}``
   shape (so the rest of the provider and every existing "computer" tool
   implementation need no change), and translates it back to the wire
   toolset shape when replaying a stored assistant turn to a toolset-target
   model.

This module also classifies a request's Anthropic-compatible ``base_url``
into a coarse platform and resolves the Opus 5.5+ computer-use wire type for
that platform (``resolve_platform_computer_type``). Earlier models are
unaffected -- they keep a single, version-gated wire type regardless of
platform.

This module has no side effects and imports nothing from ``__init__.py``.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from typing import Any

TOOLSET_TYPE = "computer_toolset_20260801"
TOOLSET_NAME = "computer"  # value of tool_use.toolset_name on the wire [CU]
LEGACY_COMPUTER_TYPES = frozenset(
    {"computer_20241022", "computer_20250124", "computer_20251124"}
)
# Keys on a legacy computer_* ToolSpec/native-tool dict that have no meaning
# on the toolset schema and are rejected ("Extra inputs are not permitted")
# if forwarded unchanged.
_LEGACY_ONLY_KEYS = (
    "type",
    "name",
    "display_width_px",
    "display_height_px",
    "display_number",
    "enable_zoom",
    "defer_loading",
)

# The complete, documented member set for `computer_toolset_20260801`'s
# `configs` field (anthropic-sdk-python `ComputerToolsetConfigsParam`,
# 2026-09-22: cursor_position, double_click, hold_key, key, left_click,
# left_click_drag, left_mouse_down, left_mouse_up, middle_click, mouse_move,
# right_click, screenshot, scroll, triple_click, type, wait, zoom). Every
# member's config accepts exactly two optional fields: `enabled` and
# `defer_loading`. Unknown keys (including the legacy tool's own
# `defer_loading`/`enable_zoom`/etc.) are rejected outright by the toolset
# schema, so a config we build must only ever use these member names and
# these two fields.
TOOLSET_MEMBERS = frozenset(
    {
        "cursor_position",
        "double_click",
        "hold_key",
        "key",
        "left_click",
        "left_click_drag",
        "left_mouse_down",
        "left_mouse_up",
        "middle_click",
        "mouse_move",
        "right_click",
        "screenshot",
        "scroll",
        "triple_click",
        "type",
        "wait",
        "zoom",
    }
)


class UnsupportedComputerToolsetDowngradeError(ValueError):
    """A caller-declared ``computer_toolset_20260801`` tool cannot be
    translated to the resolved legacy ``computer_*`` target.

    The toolset schema carries no ``display_width_px``/``display_height_px``
    (those are legacy-only fields the toolset schema rejects), so downgrading
    a toolset-shaped declaration to a legacy type would be lossy -- there is
    no display size to put on the wire. Raised by ``translate_tools``; the
    caller (``__init__.py``) wraps this as a local, actionable
    ``KernelInvalidRequestError`` before dispatch rather than sending a
    request Anthropic will reject anyway (a legacy target with no
    ``display_width_px``/``display_height_px`` at all).
    """


def translate_tools(
    tools: list[dict[str, Any]], target_type: str | None
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return ``(wire_tools, aliases)``.

    ``aliases`` maps the wire toolset name (``TOOLSET_NAME``) to the
    Amplifier-side tool name callers should see (the legacy entry's own
    ``name``, defaulting to ``"computer"``).

    When ``target_type`` is not ``TOOLSET_TYPE`` (a legacy type, or no
    computer-use tool is declared), tools are returned unchanged and
    ``aliases`` is empty -- existing behavior for every non-5.5 model.
    The one exception: a caller-declared ``TOOLSET_TYPE`` tool cannot be
    silently passed through to a legacy target (Bedrock, or an explicit
    legacy override) -- see ``UnsupportedComputerToolsetDowngradeError``.
    """
    if target_type != TOOLSET_TYPE:
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == TOOLSET_TYPE:
                raise UnsupportedComputerToolsetDowngradeError(
                    "a caller-declared computer_toolset_20260801 tool cannot "
                    "be translated to the resolved legacy computer-use tool "
                    f"type ({target_type!r}): the toolset schema carries no "
                    "display_width_px/display_height_px, so the downgrade "
                    "would be lossy and the legacy request would be "
                    "rejected outright. Declare a legacy computer_* tool "
                    "instead (with display dimensions), or target a "
                    "platform/computer_use_tool_type that accepts "
                    "computer_toolset_20260801."
                )
        return list(tools), {}

    wire_tools: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    for tool in tools:
        tool_type = tool.get("type")
        if tool_type in LEGACY_COMPUTER_TYPES or tool_type == TOOLSET_TYPE:
            aliases[TOOLSET_NAME] = tool.get("name") or TOOLSET_NAME
            if tool_type == TOOLSET_TYPE:
                # Already toolset-shaped (e.g. a caller-declared toolset
                # ToolSpec) -- pass through unchanged.
                wire_tools.append(dict(tool))
                continue
            translated: dict[str, Any] = {"type": TOOLSET_TYPE}
            zoom_enabled = bool(tool.get("enable_zoom", False))
            if not zoom_enabled:
                translated["configs"] = {"zoom": {"enabled": False}}
            if tool.get("defer_loading"):
                configs = translated.setdefault("configs", {})
                # Legacy `defer_loading` applied to the whole tool. The
                # toolset schema has no top-level/blanket equivalent (no
                # `_defer_loading` key exists in its schema -- every key
                # under `configs` must be one of TOOLSET_MEMBERS); it is
                # scoped per member instead, and "Must resolve to the same
                # value on every enabled member of the toolset" per
                # Anthropic's own field docs. Set it on every member's
                # config except a disabled zoom, which is withheld from the
                # served schema entirely and so carries no defer_loading.
                for member in TOOLSET_MEMBERS:
                    if member == "zoom" and not zoom_enabled:
                        continue
                    configs.setdefault(member, {})["defer_loading"] = True
            if "cache_control" in tool:
                translated["cache_control"] = tool["cache_control"]
            for key, value in tool.items():
                if key in _LEGACY_ONLY_KEYS or key == "cache_control":
                    continue
                translated.setdefault(key, value)
            wire_tools.append(translated)
        else:
            wire_tools.append(dict(tool))
    return wire_tools, aliases


def to_amplifier_call(
    name: str,
    arguments: dict[str, Any] | None,
    toolset_name: str | None,
    aliases: Mapping[str, str],
) -> tuple[str, dict[str, Any]]:
    """Translate one wire ``tool_use`` (already split into name/arguments/
    toolset_name) into the legacy ``(name, arguments)`` shape every existing
    "computer" tool implementation dispatches on.

    A wire call with no ``toolset_name`` (a plain function tool, or a legacy
    computer_* call where Anthropic already emits ``name="computer"``) is
    returned unchanged.
    """
    if toolset_name == TOOLSET_NAME:
        amplifier_name = aliases.get(TOOLSET_NAME, TOOLSET_NAME)
        merged = {"action": name, **(arguments or {})}
        return amplifier_name, merged
    return name, dict(arguments or {})


def to_wire_tool_call(
    name: str, arguments: dict[str, Any] | None, *, target_type: str | None
) -> tuple[str, dict[str, Any], str | None]:
    """Reverse of ``to_amplifier_call`` for replaying a stored assistant
    tool call back onto the wire.

    Returns ``(wire_name, wire_input, toolset_name)``. When ``target_type``
    is ``TOOLSET_TYPE`` and ``arguments`` contains ``"action"`` under the
    legacy ``name == TOOLSET_NAME`` convention, the member action becomes the
    wire ``name`` and ``toolset_name`` is set. Otherwise the call is
    returned unchanged with no ``toolset_name`` (legacy target, or a plain
    function tool).
    """
    arguments = arguments or {}
    if (
        target_type == TOOLSET_TYPE
        and name == TOOLSET_NAME
        and isinstance(arguments, dict)
        and "action" in arguments
    ):
        member = arguments["action"]
        wire_input = {k: v for k, v in arguments.items() if k != "action"}
        return member, wire_input, TOOLSET_NAME
    return name, dict(arguments), None


# --- Platform-aware computer-use wire type (Opus 5.5+) ---------------------
#
# Anthropic's "What's new in Claude Opus 5.5" documents that the toolset
# replacement applies "On the Claude API and Google Cloud"; Amazon Bedrock
# keeps accepting the legacy `computer_20251124` type. Computer use on
# Microsoft Foundry and on the separate "Claude Platform on AWS" offering is
# undocumented, so this provider treats both as unsupported rather than
# guessing a wire shape Anthropic has not published. An unrecognized/custom
# `base_url` (a proxy, a gateway, a typo) is unsupported for the same reason
# -- fail safe, not silently wrong.
PLATFORM_FIRST_PARTY = "first_party"  # api.anthropic.com, or no base_url configured
PLATFORM_GOOGLE_VERTEX = "google_vertex"  # Claude on Vertex AI / Google Cloud
PLATFORM_AWS_BEDROCK = (
    "aws_bedrock"  # Amazon Bedrock: bedrock-runtime.* or bedrock-mantle.*
)
PLATFORM_MICROSOFT_FOUNDRY = "microsoft_foundry"  # Microsoft Foundry (Azure AI)
PLATFORM_AWS_CLAUDE_PLATFORM = (
    "aws_claude_platform"  # "Claude Platform on AWS" (not Bedrock)
)
PLATFORM_UNKNOWN = "unknown"  # unrecognized/custom base_url

# Effective computer_use_tool_type for Opus 5.5+ on each platform. `None`
# means unsupported: no native computer-use tool type is safe to advertise.
_COMPUTER_TYPE_BY_PLATFORM: dict[str, str | None] = {
    PLATFORM_FIRST_PARTY: TOOLSET_TYPE,
    PLATFORM_GOOGLE_VERTEX: TOOLSET_TYPE,
    PLATFORM_AWS_BEDROCK: "computer_20251124",
    PLATFORM_MICROSOFT_FOUNDRY: None,
    PLATFORM_AWS_CLAUDE_PLATFORM: None,
    PLATFORM_UNKNOWN: None,
}


def _matches_domain(host: str, domain: str) -> bool:
    """True if *host* IS *domain*, or a subdomain of it (suffix-bounded on a
    dot, e.g. ``bedrock-runtime.us-east-1.amazonaws.com`` matches
    ``amazonaws.com`` but ``evilamazonaws.com`` and ``anthropic.com.evil.com``
    do not). A plain substring check (``domain in host``) is exactly the
    lookalike hole this guards against -- ``evilanthropic.com`` contains
    ``anthropic.com`` as a substring while being an entirely different,
    unrelated domain.
    """
    return host == domain or host.endswith("." + domain)


def classify_platform(base_url: str | None) -> str:
    """Classify an Anthropic-compatible ``base_url`` into a coarse platform.

    Pure and side-effect-free: this only inspects the hostname, matched by
    exact domain/suffix boundary (see ``_matches_domain``) rather than plain
    substring containment, so a lookalike host (``evilanthropic.com``,
    ``anthropic.com.evil.com``) is never mistaken for the real domain.
    Hostname patterns are taken from Anthropic's own SDK client integrations
    (``anthropic.lib.bedrock``, ``anthropic.lib.google_cloud``,
    ``anthropic.lib.foundry``):

    * unset/empty ``base_url`` -> first-party (nothing configured; the SDK's
      own default applies).
    * a non-empty ``base_url`` that is malformed or has no parseable
      hostname -> unknown/custom, NOT first-party -- an unparseable
      endpoint is never safely assumed to be Anthropic's own.
    * a host that IS ``anthropic.com`` or a subdomain of it -> first-party.
    * a host that IS ``googleapis.com`` or a subdomain of it -> Google
      Cloud / Vertex AI.
    * a host starting with ``bedrock-runtime.`` or ``bedrock-mantle.`` AND
      ending in a real AWS domain (``amazonaws.com`` or ``api.aws``) ->
      Amazon Bedrock (both the classic runtime API and the newer "mantle"
      surface continue to accept the legacy computer-use type). The prefix
      alone is not enough -- ``bedrock-runtime.evil.com`` is a lookalike,
      not Bedrock.
    * a host that IS ``services.ai.azure.com`` or a subdomain of it ->
      Microsoft Foundry.
    * any other host ending in ``amazonaws.com`` or ``api.aws`` -> a
      distinct AWS offering ("Claude Platform on AWS"), not Bedrock.
    * anything else -> unknown/custom.
    """
    if not base_url:
        return PLATFORM_FIRST_PARTY
    try:
        host = (urllib.parse.urlsplit(base_url).hostname or "").lower()
    except ValueError:
        host = ""
    if not host:
        # Non-empty base_url that is malformed or has no parseable
        # hostname -- fail safe as unknown, never assume first-party.
        return PLATFORM_UNKNOWN
    if _matches_domain(host, "anthropic.com"):
        return PLATFORM_FIRST_PARTY
    if _matches_domain(host, "googleapis.com"):
        return PLATFORM_GOOGLE_VERTEX
    if host.startswith(("bedrock-runtime.", "bedrock-mantle.")) and (
        _matches_domain(host, "amazonaws.com") or _matches_domain(host, "api.aws")
    ):
        return PLATFORM_AWS_BEDROCK
    if _matches_domain(host, "services.ai.azure.com"):
        return PLATFORM_MICROSOFT_FOUNDRY
    if _matches_domain(host, "amazonaws.com") or _matches_domain(host, "api.aws"):
        return PLATFORM_AWS_CLAUDE_PLATFORM
    return PLATFORM_UNKNOWN


def resolve_platform_computer_type(base_url: str | None) -> tuple[str | None, str]:
    """Return ``(effective_computer_use_tool_type, platform_label)`` for
    Opus 5.5+ given the request's ``base_url``.

    Only meaningful when ``ModelCapabilities.computer_use_platform_aware``
    is True -- every other model keeps its fixed, version-gated wire type
    regardless of platform (Anthropic does not document platform variance
    for those generations).
    """
    platform = classify_platform(base_url)
    return _COMPUTER_TYPE_BY_PLATFORM.get(platform), platform
