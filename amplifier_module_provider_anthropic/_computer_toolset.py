"""Narrow request-local adapter for Opus 5.5's native computer toolset.

The core ToolSpec surface represents computer execution as an ordinary function
tool.  Opus 5.5 instead expects one fixed native declaration and emits members
of that declaration.  Keep that dialect boundary here: no provider instance
state, no capability inference, and no broader endpoint classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse


COMPUTER_TOOLSET_TYPE = "computer_toolset_20260801"
COMPUTER_TOOLSET_NAME = "computer"
PROVENANCE_TOOLSET = "_anthropic_computer_toolset_name"
PROVENANCE_MEMBER = "_anthropic_computer_member_name"
_LEGACY_COMPUTER_TYPES = frozenset(
    {
        "computer_20241022",
        "computer_20250124",
        "computer_20251124",
    }
)
_SUPPORTED_COMPUTER_TYPES = _LEGACY_COMPUTER_TYPES | frozenset({COMPUTER_TOOLSET_TYPE})


class ComputerToolsetError(ValueError):
    """A local declaration/history error which must not reach the API."""


@dataclass(frozen=True)
class NativeComputerAdapter:
    """The single native declaration active for one assembled request."""

    alias: str
    toolset_name: str = COMPUTER_TOOLSET_NAME


def is_opus_55(model: str) -> bool:
    """Recognize only canonical hyphenated Opus 5.5 model IDs."""
    return bool(
        re.fullmatch(
            r"claude-opus-5-5(?:-\d{8})?",
            model.lower(),
        )
    )


def has_unverified_opus_55_suffix(model: str) -> bool:
    """Recognize malformed Opus 5.5 suffixes without accepting them as aliases."""
    return model.lower().startswith("claude-opus-5-5-") and not is_opus_55(model)


def is_first_party_base_url(base_url: str | None) -> bool:
    """True only for the SDK default or the exact public Anthropic hostname."""
    if base_url is None:
        return True
    try:
        return urlparse(base_url).hostname == "api.anthropic.com"
    except (TypeError, ValueError):
        return False


def _tool_mapping(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        return dict(tool)
    if hasattr(tool, "model_dump"):
        dumped = tool.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dict(dumped)
    result: dict[str, Any] = {}
    for name in (
        "type",
        "name",
        "parameters",
        "description",
        "configs",
        "cache_control",
        "allowed_callers",
        "display_width_px",
        "display_height_px",
        "display_number",
        "enable_zoom",
        "defer_loading",
    ):
        value = getattr(tool, name, None)
        if value is not None:
            result[name] = value
    return result


def _native_computer_type(tool: dict[str, Any]) -> bool:
    return tool.get("type") in _SUPPORTED_COMPUTER_TYPES


def _unknown_computer_type(tool: dict[str, Any]) -> bool:
    tool_type = tool.get("type")
    return isinstance(tool_type, str) and tool_type.startswith("computer_")


def translate_tools(
    tools: list[Any],
    *,
    model: str,
    base_url: str | None | Callable[[], str | None],
) -> tuple[list[Any], NativeComputerAdapter | None]:
    """Translate the one supported native computer declaration for Opus 5.5.

    Ordinary function tools and unrelated native tools are returned untouched.
    Non-first-party endpoints fail loudly instead of pretending they implement
    Anthropic's vendor-native toolset. A callable resolves the endpoint only
    after a native declaration is found.
    """
    if not is_opus_55(model):
        if has_unverified_opus_55_suffix(model) and any(
            _native_computer_type(_tool_mapping(tool))
            or _unknown_computer_type(_tool_mapping(tool))
            for tool in tools
        ):
            raise ComputerToolsetError(
                "Unrecognized Opus 5.5 model suffix; native computer declarations "
                "are supported only for claude-opus-5-5 or a dated 8-digit alias."
            )
        return list(tools), None

    converted: list[Any] = []
    adapter: NativeComputerAdapter | None = None
    for raw_tool in tools:
        tool = _tool_mapping(raw_tool)
        if _unknown_computer_type(tool) and not _native_computer_type(tool):
            raise ComputerToolsetError(
                f"Unsupported native computer declaration type {tool['type']!r} for "
                "Opus 5.5; supported types are computer_20241022, "
                "computer_20250124, computer_20251124, and "
                "computer_toolset_20260801."
            )
        if not _native_computer_type(tool):
            converted.append(raw_tool)
            continue
        if callable(base_url):
            base_url = base_url()
        if not is_first_party_base_url(base_url):
            raise ComputerToolsetError(
                "Native computer-toolset declarations require Anthropic's first-party "
                "API endpoint (the default or exact api.anthropic.com hostname); "
                "this configured endpoint is not supported for Opus 5.5 native tools."
            )
        if adapter is not None:
            raise ComputerToolsetError(
                "Only one native computer declaration is allowed per Opus 5.5 request."
            )
        alias = tool.get("name", COMPUTER_TOOLSET_NAME)
        if not isinstance(alias, str) or not alias:
            raise ComputerToolsetError(
                "A native computer declaration needs a non-empty ToolSpec name for dispatch."
            )
        if sum(1 for candidate in tools if _tool_mapping(candidate).get("name") == alias) > 1:
            raise ComputerToolsetError(
                f"Native computer dispatch alias {alias!r} is ambiguous in this request."
            )
        allowed = {
            "type",
            "name",
            "parameters",
            "description",
            "configs",
            "cache_control",
            "allowed_callers",
            "display_width_px",
            "display_height_px",
            "display_number",
            "enable_zoom",
            "defer_loading",
        }
        unknown = sorted(set(tool) - allowed)
        if unknown:
            raise ComputerToolsetError(
                "Native computer declaration has unrepresentable field(s): "
                + ", ".join(unknown)
            )
        if tool.get("defer_loading"):
            raise ComputerToolsetError(
                "Opus 5.5 computer_toolset_20260801 cannot represent legacy "
                "defer_loading safely; remove it rather than widening deferred members."
            )
        legacy_declaration = tool.get("type") != COMPUTER_TOOLSET_TYPE
        configs = tool.get("configs", {})
        if not isinstance(configs, dict):
            raise ComputerToolsetError("Native computer configs must be a mapping.")
        wire_configs = dict(configs)
        if "enable_zoom" in tool:
            if not isinstance(tool["enable_zoom"], bool):
                raise ComputerToolsetError(
                    "Legacy native computer enable_zoom must be a boolean."
                )
            zoom = wire_configs.get("zoom", {})
            if not isinstance(zoom, dict):
                raise ComputerToolsetError("Native computer configs.zoom must be a mapping.")
            wire_configs["zoom"] = {**zoom, "enabled": tool["enable_zoom"]}
        elif legacy_declaration and "zoom" not in wire_configs:
            wire_configs["zoom"] = {"enabled": False}
        wire: dict[str, Any] = {
            "type": COMPUTER_TOOLSET_TYPE,
            "configs": wire_configs,
        }
        for key in ("cache_control", "allowed_callers"):
            if key in tool:
                wire[key] = tool[key]
        converted.append(wire)
        adapter = NativeComputerAdapter(alias=alias)
    return converted, adapter


def native_wire_tool_use(
    block: dict[str, Any], adapter: NativeComputerAdapter | None
) -> dict[str, Any] | None:
    """Return an exact native wire block when persisted provenance authorizes it."""
    has_toolset_provenance = PROVENANCE_TOOLSET in block
    has_member_provenance = PROVENANCE_MEMBER in block
    if has_toolset_provenance or has_member_provenance:
        if not (has_toolset_provenance and has_member_provenance):
            raise ComputerToolsetError(
                "Native computer history has incomplete persisted provenance."
            )
        toolset_name = block[PROVENANCE_TOOLSET]
        action = block[PROVENANCE_MEMBER]
        if not isinstance(toolset_name, str) or not isinstance(action, str) or not action:
            raise ComputerToolsetError(
                "Native computer history has invalid persisted provenance."
            )
        if toolset_name != COMPUTER_TOOLSET_NAME:
            raise ComputerToolsetError(
                "Native computer history provenance does not match the native toolset."
            )
        if "type" in block and block["type"] not in {"tool_call", "tool_use"}:
            raise ComputerToolsetError(
                "Native computer history provenance belongs to an invalid block type."
            )
        source_input = block.get("input", block.get("arguments", {}))
        if not isinstance(source_input, dict):
            raise ComputerToolsetError("Native computer history has a non-mapping input.")
        if source_input.get("action") != action:
            raise ComputerToolsetError(
                "Native computer history action does not match persisted provenance."
            )
        if adapter is not None and toolset_name != adapter.toolset_name:
            raise ComputerToolsetError(
                "Native computer history provenance does not match the current toolset."
            )
        if adapter is not None:
            current_alias = block.get("name", block.get("tool"))
            if current_alias != adapter.alias:
                raise ComputerToolsetError(
                    "Native computer history alias does not match persisted provenance."
                )
        tagged = True
    else:
        tagged = False
    if adapter is None:
        return None
    # Legacy separate tool_calls have no content-block discriminator.
    # Only a matching native declaration below authorizes their translation.
    if not tagged and block.get("type") not in {None, "tool_call", "tool_use"}:
        return None
    legacy_matching_alias = (
        not has_toolset_provenance
        and block.get("name", block.get("tool")) == adapter.alias
    )
    if not (tagged or legacy_matching_alias):
        return None
    source_input = block.get("input", block.get("arguments", {}))
    if not isinstance(source_input, dict):
        raise ComputerToolsetError("Native computer history has a non-mapping input.")
    current_action = source_input.get("action")
    if tagged:
        action = block.get(PROVENANCE_MEMBER)
    else:
        action = current_action
    if not isinstance(action, str) or not action:
        raise ComputerToolsetError("Native computer history has no action member.")
    native_input = dict(source_input)
    if "action" in native_input:
        native_input.pop("action")
    return {
        "type": "tool_use",
        "id": block.get("id") or block.get("tool_call_id", ""),
        "toolset_name": adapter.toolset_name,
        "name": action,
        "input": native_input,
    }

