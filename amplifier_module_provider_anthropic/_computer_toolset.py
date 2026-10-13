"""Narrow request-local adapter for first-party native computer toolsets.

The core ToolSpec surface represents computer execution as an ordinary function
tool.  Opus and canonical Sonnet 5.5 expect one native declaration and emit members
of that declaration.  Keep that dialect boundary here: no provider instance
state, no capability inference, and no broader endpoint classifier.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse


COMPUTER_TOOLSET_TYPE = "computer_toolset_20260801"
COMPUTER_TOOLSET_NAME = "computer"
PROVENANCE_TOOLSET = "_anthropic_computer_toolset_name"
PROVENANCE_MEMBER = "_anthropic_computer_member_name"
PROVENANCE_NEUTRAL = "_anthropic_computer_use_v1"
_LEGACY_COMPUTER_TYPES = frozenset(
    {
        "computer_20241022",
        "computer_20250124",
        "computer_20251124",
    }
)
_SUPPORTED_COMPUTER_TYPES = _LEGACY_COMPUTER_TYPES | frozenset({COMPUTER_TOOLSET_TYPE})
_MEMBERS = frozenset(
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


class ComputerToolsetError(ValueError):
    """A local declaration/history error which must not reach the API."""


@dataclass(frozen=True)
class NativeComputerAdapter:
    """The single native declaration active for one assembled request."""

    alias: str
    toolset_name: str = COMPUTER_TOOLSET_NAME
    strict: bool = False
    enabled_members: frozenset[str] = _MEMBERS
    ordinary_names: frozenset[str] = frozenset()
    neutral: bool = False
    image_space: tuple[int, int] | None = None


def has_neutral_computer_intent(tools: list[Any]) -> bool:
    """Presence is the discriminator, including malformed intent."""
    return any("computer_use" in _tool_mapping(tool) for tool in tools)


def neutral_effective_model(
    tools: list[Any], model: str, overrides: dict[str, Any]
) -> str:
    """Resolve model-only overrides; refuse shadowing of local tool bindings."""
    extra = overrides.get("extra_body")
    envelopes = [overrides, extra] if isinstance(extra, dict) else [overrides]
    populations = [tools, *(item.get("tools", []) for item in envelopes)]
    neutral = any(
        has_neutral_computer_intent(list(population))
        for population in populations
        if isinstance(population, (list, tuple))
    )
    if not neutral:
        return model
    if extra is not None and not isinstance(extra, dict):
        raise ComputerToolsetError("Neutral computer extra_body must be a mapping.")
    if any(
        {"tools", "messages", "system", "tool_choice"}.intersection(item)
        for item in envelopes
    ):
        raise ComputerToolsetError(
            "Neutral computer sensitive overrides (tools/messages/system/tool_choice) "
            "are unsupported; use the public request fields."
        )
    effective = model
    for envelope in envelopes:
        effective = envelope.get("model", effective)
    if not isinstance(effective, str) or not effective:
        raise ComputerToolsetError(
            "Neutral computer effective model must be a nonempty string."
        )
    return effective


def _exact_first_party_endpoint(base_url: str | None) -> bool:
    try:
        endpoint = urlparse(base_url or "https://api.anthropic.com")
        return (
            endpoint.hostname == "api.anthropic.com"
            and endpoint.scheme == "https"
            and endpoint.port in {None, 443}
            and endpoint.path in {"", "/"}
            and not endpoint.username
            and not endpoint.password
            and not endpoint.query
            and not endpoint.fragment
        )
    except (TypeError, ValueError):
        return False


def _translate_neutral_tools(
    tools: list[Any], *, model: str, base_url: str | None | Callable[[], str | None]
) -> tuple[list[Any], NativeComputerAdapter | None]:
    """Compile v1 into a closed GA surface or a sanitized ordinary function.

    Verified protocol: first-party computer-use-tool and Haiku 5.5 migration
    docs, plus anthropic 1.x's ComputerToolset20260801Param. Legacy declarations
    remain on their separate shipped path.
    """
    neutral_tools = [tool for tool in tools if "computer_use" in _tool_mapping(tool)]
    if len(neutral_tools) != 1 or any(
        _unknown_computer_type(_tool_mapping(tool)) for tool in tools
    ):
        raise ComputerToolsetError("Ambiguous neutral/legacy computer declarations.")
    source = _tool_mapping(neutral_tools[0])
    intent = source["computer_use"]
    if source.get("type") not in {None, "function", "custom"}:
        raise ComputerToolsetError(
            "Neutral computer intent requires a function ToolSpec."
        )
    if (
        not isinstance(intent, dict)
        or type(intent.get("version")) is not int
        or intent["version"] != 1
    ):
        raise ComputerToolsetError("Neutral computer_use requires integer version=1.")
    if set(intent) - {
        "version",
        "operations",
        "image_space",
        "unavailable_operations",
        "native_required",
    }:
        raise ComputerToolsetError("Unknown neutral computer intent fields.")
    required = intent.get("native_required", False)
    if type(required) is not bool:
        raise ComputerToolsetError("Neutral native_required must be a boolean.")
    operations = intent.get("operations")
    if (
        not isinstance(operations, list)
        or any(not isinstance(op, str) or op not in _MEMBERS for op in operations)
        or len(set(operations)) != len(operations)
    ):
        raise ComputerToolsetError(
            "Neutral operations must be a closed, unique canonical list."
        )
    unavailable = intent.get("unavailable_operations", {})
    if (
        not isinstance(unavailable, dict)
        or any(
            op not in _MEMBERS or not isinstance(reason, str) or not reason
            for op, reason in unavailable.items()
        )
        or set(operations).intersection(unavailable)
    ):
        raise ComputerToolsetError(
            "Invalid or conflicting neutral unavailable_operations."
        )
    alias = source.get("name")
    if (
        not isinstance(alias, str)
        or not alias
        or sum(_tool_mapping(tool).get("name") == alias for tool in tools) != 1
    ):
        raise ComputerToolsetError(
            "Neutral computer dispatch alias is missing or ambiguous."
        )
    if not isinstance(source.get("parameters"), dict):
        raise ComputerToolsetError(
            "Neutral computer requires a canonical argument schema."
        )
    image = intent.get("image_space")
    dimensions = None
    if image is not None:
        if (
            not isinstance(image, dict)
            or set(image) != {"width", "height", "unit"}
            or image.get("unit") != "pixel"
            or any(
                type(image.get(axis)) is not int or image[axis] <= 0
                for axis in ("width", "height")
            )
        ):
            raise ComputerToolsetError(
                "Neutral image_space must contain positive integer pixel dimensions."
            )
        dimensions = (image["width"], image["height"])
    reason = None
    # Fixed verified model IDs only; no inferred aliases or capability discovery.
    if not (is_sonnet_55(model) or is_opus_55(model) or model == "claude-haiku-5-5"):
        reason = "unverified native model"
    elif not operations:
        reason = "no enabled native operations"
    elif dimensions is None:
        reason = "unknown screenshot pixel dimensions"
    elif (
        max(dimensions) > 2576
        or ((dimensions[0] + 27) // 28) * ((dimensions[1] + 27) // 28) > 4784
    ):
        reason = "screenshot exceeds native image limits"
    elif not _exact_first_party_endpoint(
        base_url() if callable(base_url) else base_url
    ):
        reason = "unsupported native endpoint"
    function = {
        "name": alias,
        "description": source.get("description"),
        "parameters": copy.deepcopy(source["parameters"]),
    }
    if reason:
        if required:
            raise ComputerToolsetError(
                f"Neutral native-required request refused: {reason}."
            )
        return [
            function if "computer_use" in _tool_mapping(tool) else tool
            for tool in tools
        ], None
    ordinary_names = frozenset(
        data["name"]
        for tool in tools
        if "computer_use" not in (data := _tool_mapping(tool))
        and data.get("type") in {None, "function", "custom"}
        and isinstance(data.get("name"), str)
    )
    if "computer" in ordinary_names:
        raise ComputerToolsetError(
            "Native computer toolset conflicts with an ordinary tool named computer."
        )
    wire = {
        "type": COMPUTER_TOOLSET_TYPE,
        "configs": {op: {"enabled": op in operations} for op in sorted(_MEMBERS)},
    }
    if "cache_control" in source:
        wire["cache_control"] = copy.deepcopy(source["cache_control"])
    return [
        wire if "computer_use" in _tool_mapping(tool) else tool for tool in tools
    ], NativeComputerAdapter(
        alias=alias,
        strict=True,
        enabled_members=frozenset(operations),
        ordinary_names=ordinary_names,
        neutral=True,
        image_space=dimensions,
    )


def canonical_native_batch(
    adapter: NativeComputerAdapter,
    member: str,
    input_data: dict[str, Any],
    *,
    check_enabled: bool = True,
) -> dict[str, Any]:
    """One vendor call ID, one canonical result; never merge distinct calls."""
    validate_member(adapter, member, input_data, check_enabled=check_enabled)
    return {
        "version": 1,
        "actions": [{"action": member, **copy.deepcopy(input_data)}],
        "capture_after": True,
    }


def is_sonnet_55(model: str) -> bool:
    """The dateless ID is the snapshot; no dated aliases are verified."""
    return model == "claude-sonnet-5-5"


def has_native_computer_history(messages: list[Any]) -> bool:
    """Only explicit family/provenance, never ownership inferred from names."""
    for message in messages:
        data = message.model_dump() if hasattr(message, "model_dump") else message
        if not isinstance(data, dict) or data.get("role") != "assistant":
            continue
        for population in (data.get("content"), data.get("tool_calls")):
            if not isinstance(population, list):
                continue
            for raw in population:
                block = _tool_mapping(raw)
                if (
                    PROVENANCE_TOOLSET in block
                    or PROVENANCE_MEMBER in block
                    or PROVENANCE_NEUTRAL in block
                    or block.get("toolset_name") == COMPUTER_TOOLSET_NAME
                ):
                    return True
    return False


def guard_sonnet_overrides(
    tools: list[Any],
    model: str,
    overrides: dict[str, Any],
    *,
    native_history: bool = False,
) -> None:
    """Fail before I/O when expert shadowing would invalidate native context."""
    extra = overrides.get("extra_body")
    envelopes = [overrides, extra] if isinstance(extra, dict) else [overrides]
    models = [model, *(item.get("model", model) for item in envelopes)]
    populations = [tools, *(item.get("tools", []) for item in envelopes)]
    sonnet_boundary = any(
        isinstance(candidate, str) and candidate.lower().startswith("claude-sonnet-5-5")
        for candidate in models
    )
    if sonnet_boundary and any(
        "tools" in item and not isinstance(item["tools"], (list, tuple))
        for item in envelopes
    ):
        raise ComputerToolsetError(
            "Sonnet 5.5 tools overrides must be list/tuple populations; uninspectable iterable overrides are unsupported."
        )
    native = native_history or any(
        _unknown_computer_type(_tool_mapping(tool))
        for population in populations
        if isinstance(population, (list, tuple))
        for tool in population
    )
    if not native or not sonnet_boundary:
        return
    sensitive = {"model", "tools", "messages", "system", "tool_choice"}
    if any(sensitive.intersection(item) for item in envelopes):
        raise ComputerToolsetError(
            "Sonnet 5.5 native computer sensitive overrides (model/tools/messages/"
            "system/tool_choice) are unsupported; use the public request fields."
        )
    if extra is not None and not isinstance(extra, dict):
        raise ComputerToolsetError("Native computer extra_body must be a mapping.")


def validate_member(
    adapter: NativeComputerAdapter,
    member: str,
    input_data: dict[str, Any],
    *,
    check_enabled: bool = True,
) -> None:
    """Refuse unrepresentable executor arguments, without dropping them."""
    if not adapter.strict:
        return
    if not isinstance(member, str) or member not in _MEMBERS:
        raise ComputerToolsetError("Unknown native computer action member.")
    if check_enabled and member not in adapter.enabled_members:
        raise ComputerToolsetError("Native computer action member is disabled.")
    if not isinstance(input_data, dict):
        raise ComputerToolsetError("Native computer action input must be a mapping.")
    if "action" in input_data:
        raise ComputerToolsetError(
            "Native computer action input contains reserved key 'action'."
        )
    click_members = {
        "left_click",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "mouse_move",
        "left_mouse_down",
        "left_mouse_up",
        "left_click_drag",
        "scroll",
    }
    allowed: set[str] = set()
    required: set[str] = set()
    if member in click_members:
        allowed = {"coordinate", "text"}
    if member in {"mouse_move", "left_click_drag"}:
        required.add("coordinate")
    if member == "left_click_drag":
        allowed.add("start_coordinate")
    if member == "scroll":
        allowed |= {"scroll_direction", "scroll_amount"}
        required |= {"scroll_direction", "scroll_amount"}
    if member in {"key", "type", "hold_key"}:
        allowed = {"text"}
        required = {"text"}
    if member == "key":
        allowed.add("repeat")
    if member in {"hold_key", "wait"}:
        allowed.add("duration")
        required.add("duration")
    if member == "zoom":
        allowed = required = {"region"}
    if set(input_data) - allowed or required - set(input_data):
        raise ComputerToolsetError("Unrepresentable native computer action fields.")
    if member in click_members and input_data.get("text"):
        raise ComputerToolsetError(
            "Native computer modifiers are not qualified by this executor boundary."
        )
    for field, size in (("coordinate", 2), ("start_coordinate", 2), ("region", 4)):
        value = input_data.get(field)
        if field in required and value is None:
            raise ComputerToolsetError(
                "Native computer required geometry cannot be null."
            )
        if value is not None and (
            not isinstance(value, list)
            or len(value) != size
            or any(type(axis) is not int or axis < 0 for axis in value)
        ):
            raise ComputerToolsetError(
                "Native computer coordinates must be nonnegative integer arrays."
            )
    if member in {"key", "type", "hold_key"} and (
        not isinstance(input_data.get("text"), str) or not input_data["text"]
    ):
        raise ComputerToolsetError("Native computer text must be a nonempty string.")
    if (
        "repeat" in input_data
        and input_data["repeat"] is not None
        and (type(input_data["repeat"]) is not int or input_data["repeat"] != 1)
    ):
        raise ComputerToolsetError(
            "Native computer key repeat is not qualified by this boundary."
        )
    for field in ("duration", "scroll_amount"):
        if field in input_data and (
            type(input_data[field]) is not int or input_data[field] <= 0
        ):
            raise ComputerToolsetError(
                "Native computer duration/scroll amount must be positive integers."
            )
    if member == "scroll" and (
        not isinstance(input_data["scroll_direction"], str)
        or input_data["scroll_direction"] not in {"up", "down", "left", "right"}
    ):
        raise ComputerToolsetError("Native computer scroll direction is invalid.")
    if adapter.neutral:
        if "text" in input_data and member in click_members:
            raise ComputerToolsetError("Neutral native modifiers are not supported.")
        if "repeat" in input_data and type(input_data["repeat"]) is not int:
            raise ComputerToolsetError("Neutral native repeat must be integer 1.")
        if member in {"left_mouse_down", "left_mouse_up"} and input_data:
            raise ComputerToolsetError(
                "Neutral native mouse button members take no arguments."
            )
        if member == "left_click_drag" and input_data.get("start_coordinate") is None:
            raise ComputerToolsetError(
                "Neutral native drag requires an explicit start_coordinate."
            )
        if member in {"wait", "hold_key"} and input_data["duration"] > 300:
            raise ComputerToolsetError("Neutral native duration exceeds 300 seconds.")
        width, height = adapter.image_space
        for field in ("coordinate", "start_coordinate"):
            point = input_data.get(field)
            if point is not None and (point[0] >= width or point[1] >= height):
                raise ComputerToolsetError(
                    "Neutral native coordinate is outside screenshot pixel space."
                )
        if member == "zoom":
            x0, y0, x1, y1 = input_data["region"]
            if not (x0 < x1 <= width and y0 < y1 <= height):
                raise ComputerToolsetError(
                    "Neutral native zoom region is outside screenshot pixel space."
                )


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
            # Presence is the discriminator; explicit null must reach validation.
            complete = tool.model_dump(exclude_none=False)
            if isinstance(complete, dict) and "computer_use" in complete:
                dumped["computer_use"] = complete["computer_use"]
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
        "computer_use",
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
    if has_neutral_computer_intent(tools):
        return _translate_neutral_tools(tools, model=model, base_url=base_url)
    sonnet55 = is_sonnet_55(model)
    if not is_opus_55(model) and not sonnet55:
        if model.lower().startswith("claude-sonnet-5-5") and any(
            _unknown_computer_type(_tool_mapping(tool)) for tool in tools
        ):
            raise ComputerToolsetError(
                "Unverified Sonnet 5.5 model suffix; only claude-sonnet-5-5 is supported."
            )
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
        if sonnet55:
            try:
                endpoint = urlparse(base_url or "https://api.anthropic.com")
                exact_endpoint = (
                    endpoint.scheme == "https"
                    and endpoint.port in {None, 443}
                    and endpoint.path in {"", "/"}
                    and not endpoint.username
                    and not endpoint.password
                    and not endpoint.query
                    and not endpoint.fragment
                )
            except ValueError:
                exact_endpoint = False
            if not exact_endpoint:
                raise ComputerToolsetError(
                    "Native computer requires the exact first-party HTTPS endpoint."
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
        if (
            sum(
                1
                for candidate in tools
                if _tool_mapping(candidate).get("name") == alias
            )
            > 1
        ):
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
        wire_configs = copy.deepcopy(configs)
        if "enable_zoom" in tool:
            if not isinstance(tool["enable_zoom"], bool):
                raise ComputerToolsetError(
                    "Legacy native computer enable_zoom must be a boolean."
                )
            zoom = wire_configs.get("zoom", {})
            if not isinstance(zoom, dict):
                raise ComputerToolsetError(
                    "Native computer configs.zoom must be a mapping."
                )
            if (
                sonnet55
                and "enabled" in zoom
                and zoom["enabled"] != tool["enable_zoom"]
            ):
                raise ComputerToolsetError(
                    "Conflicting native computer zoom configuration."
                )
            wire_configs["zoom"] = {**zoom, "enabled": tool["enable_zoom"]}
        elif legacy_declaration and "zoom" not in wire_configs:
            wire_configs["zoom"] = {"enabled": False}
        wire: dict[str, Any] = {
            "type": COMPUTER_TOOLSET_TYPE,
            "configs": wire_configs,
        }
        enabled_members = _MEMBERS
        if sonnet55:
            if "allowed_callers" in tool:
                raise ComputerToolsetError(
                    "Native computer allowed_callers is unrepresentable."
                )
            for member, config in wire_configs.items():
                if member not in _MEMBERS or not isinstance(config, dict):
                    raise ComputerToolsetError(
                        "Invalid native computer member configs."
                    )
                if set(config) - {"enabled", "defer_loading"} or any(
                    type(value) is not bool for value in config.values()
                ):
                    raise ComputerToolsetError(
                        "Invalid native computer member config fields."
                    )
                if config.get("defer_loading"):
                    raise ComputerToolsetError(
                        "Deferred native computer members are unsupported."
                    )
            enabled_members = frozenset(
                member
                for member in _MEMBERS
                if wire_configs.get(member, {}).get("enabled", True)
            )
        for key in ("cache_control", "allowed_callers"):
            if key in tool:
                wire[key] = tool[key]
        converted.append(wire)
        adapter = NativeComputerAdapter(
            alias=alias,
            strict=sonnet55,
            enabled_members=enabled_members,
            ordinary_names=frozenset(
                _tool_mapping(candidate).get("name")
                for candidate in tools
                if _tool_mapping(candidate).get("type") in {None, "function", "custom"}
                and isinstance(_tool_mapping(candidate).get("name"), str)
            ),
        )
    return converted, adapter


def _json_agrees(left: Any, right: Any) -> bool:
    """Compare JSON copies, not Python's bool-as-number/container equality."""
    if type(left) in {int, float} and type(right) in {int, float}:
        return (
            (type(left) is int or math.isfinite(left))
            and (type(right) is int or math.isfinite(right))
            and left == right
        )
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (
            all(type(key) is str for key in left)
            and all(type(key) is str for key in right)
            and left.keys() == right.keys()
            and all(_json_agrees(left[key], right[key]) for key in left)
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_agrees(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) in {str, bool, type(None)} and left == right


def _neutral_replay_id(
    block: dict[str, Any], name: str, input_data: dict[str, Any]
) -> str:
    # Every supplied copy is evidence, including explicit nulls. Never let
    # serializer precedence or Python's True == 1 hide a disagreement.
    ids = [block[key] for key in ("id", "tool_call_id", "call_id") if key in block]
    if not ids or type(ids[0]) is not str or not ids[0]:
        raise ComputerToolsetError("Neutral computer history requires a call ID.")
    for keys, value in (
        (("name", "tool"), name),
        (("input", "arguments"), input_data),
        (("id", "tool_call_id", "call_id"), ids[0]),
    ):
        copies = [block[key] for key in keys if key in block]
        if not copies or any(not _json_agrees(item, value) for item in copies):
            raise ComputerToolsetError(
                "Neutral computer history disagrees with its vendor provenance."
            )
    return ids[0]


def native_wire_tool_use(
    block: dict[str, Any], adapter: NativeComputerAdapter | None
) -> dict[str, Any] | None:
    """Return an exact native wire block when persisted provenance authorizes it."""
    if PROVENANCE_NEUTRAL in block:
        if adapter is None or not adapter.neutral:
            raise ComputerToolsetError(
                "Neutral computer history requires a matching v1 declaration."
            )
        provenance = block[PROVENANCE_NEUTRAL]
        member = block.get(PROVENANCE_MEMBER)
        if (
            not isinstance(provenance, dict)
            or set(provenance) != {"version", "input", "image_space"}
            or type(provenance.get("version")) is not int
            or provenance["version"] != 1
            or not _json_agrees(provenance["image_space"], list(adapter.image_space))
            or block.get(PROVENANCE_TOOLSET) != adapter.toolset_name
            or block.get("type") not in {None, "tool_call", "tool_use"}
            or block.get("toolset_name") is not None
        ):
            raise ComputerToolsetError(
                "Neutral computer history binding/provenance mismatch."
            )
        expected = canonical_native_batch(
            adapter, member, provenance["input"], check_enabled=False
        )
        call_id = _neutral_replay_id(block, adapter.alias, expected)
        return {
            "type": "tool_use",
            "id": call_id,
            "toolset_name": adapter.toolset_name,
            "name": member,
            "input": copy.deepcopy(provenance["input"]),
        }
    if (
        adapter is not None
        and adapter.neutral
        and (
            PROVENANCE_MEMBER in block
            or PROVENANCE_TOOLSET in block
            or any(block.get(key) == adapter.alias for key in ("name", "tool"))
            and block.get("type") in {None, "tool_call", "tool_use"}
            and block.get("toolset_name") is None
        )
    ):
        raise ComputerToolsetError(
            "Neutral computer history is missing v1 vendor provenance."
        )
    if adapter is not None and adapter.strict and block.get("toolset_name") is not None:
        if block.get("type") != "tool_use":
            raise ComputerToolsetError(
                "Native computer history has an invalid block type."
            )
        if PROVENANCE_TOOLSET in block or PROVENANCE_MEMBER in block:
            # Persisted dispatch tags belong to the local alias/action form,
            # never to the already-native wire representation.
            raise ComputerToolsetError(
                "Raw native computer history has conflicting persisted provenance."
            )
        if block["toolset_name"] != adapter.toolset_name:
            raise ComputerToolsetError("Native computer history family is undeclared.")
        validate_member(
            adapter, block.get("name"), block.get("input"), check_enabled=False
        )
        call_id = (
            _neutral_replay_id(block, block["name"], block["input"])
            if adapter.neutral
            else block.get("id", "")
        )
        return {
            "type": "tool_use",
            "id": call_id,
            "toolset_name": adapter.toolset_name,
            "name": block["name"],
            "input": copy.deepcopy(block["input"]),
        }
    has_toolset_provenance = PROVENANCE_TOOLSET in block
    has_member_provenance = PROVENANCE_MEMBER in block
    if has_toolset_provenance or has_member_provenance:
        if not (has_toolset_provenance and has_member_provenance):
            raise ComputerToolsetError(
                "Native computer history has incomplete persisted provenance."
            )
        toolset_name = block[PROVENANCE_TOOLSET]
        action = block[PROVENANCE_MEMBER]
        if (
            not isinstance(toolset_name, str)
            or not isinstance(action, str)
            or not action
        ):
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
            raise ComputerToolsetError(
                "Native computer history has a non-mapping input."
            )
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
    validate_member(adapter, action, native_input, check_enabled=False)
    return {
        "type": "tool_use",
        "id": block.get("id") or block.get("tool_call_id", ""),
        "toolset_name": adapter.toolset_name,
        "name": action,
        "input": native_input,
    }
