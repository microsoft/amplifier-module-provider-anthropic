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

This module has no side effects and imports nothing from ``__init__.py``.
"""

from __future__ import annotations

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
    """
    if target_type != TOOLSET_TYPE:
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
            if not tool.get("enable_zoom", False):
                translated["configs"] = {"zoom": {"enabled": False}}
            if tool.get("defer_loading"):
                configs = translated.setdefault("configs", {})
                # Legacy defer_loading applied to the whole tool; the
                # toolset schema scopes it per member. Apply it to every
                # config entry created above (zoom, if present) plus a
                # blanket entry so it is not silently dropped.
                configs["_defer_loading"] = True
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
