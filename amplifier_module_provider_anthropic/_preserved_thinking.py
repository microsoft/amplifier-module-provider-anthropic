"""Preserved-thinking support for Opus 5.5+ models.

Anthropic's Opus 5.5 migration guidance ("Preserve thinking across turns",
cited below as [PT]) documents that thinking blocks are bound to a fixed
conversation prefix: replaying a stored assistant turn on a later request
must reproduce the *exact* prior wire content, append-only, or the request
is rejected with HTTP 400 ("The block is bound to a different conversation").

This module is a pure, side-effect-free helper library for that contract:

- ``wire_content_snapshot`` captures the exact server JSON for an assistant
  turn's content array (including block types this provider does not
  otherwise understand, e.g. ``fallback``), for storage in
  ``ChatResponse.metadata`` / ``Message.metadata``.
- ``replay_from_snapshot`` / ``replay_from_content`` reproduce that content
  on a later request -- byte-exact when a valid snapshot is available, best
  -effort ordered reconstruction otherwise.
- ``place_floating_messages`` deterministically places a later
  ``role: "system"`` / ``"developer"`` message without disturbing the
  positions of anything already in the conversation (append-only).
- ``is_prefix_binding_mismatch`` / ``summarize_transformations`` support the
  automatic ``drop_block`` recovery path (see ``__init__.py``).

This module has no side effects and imports nothing from ``__init__.py``.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from . import _computer_toolset

# Namespace under ChatResponse.metadata / Message.metadata this provider uses
# for every Anthropic-specific extra (see also the `stop_details` writer in
# `_convert_to_chat_response`, which shares this same namespace).
METADATA_KEY = "anthropic"
# Exact server content array for an assistant turn, keyed under
# metadata["anthropic"]["wire_content"].
WIRE_CONTENT_KEY = "wire_content"
# Present ("drop_block") once this session has switched to
# thinking.block_binding.prefix_mismatch_behavior="drop_block" after a
# prefix-binding rejection -- so a NEW provider instance given this history
# (e.g. after a process restart) resumes with the same setting rather than
# hitting the same 400 again before recovering.
BINDING_KEY = "thinking_binding"

# [PT] "thinking-binding-controls-2026-08-01" is required to receive
# `input_transformations` on the response and to send `thinking.block_binding`
# on the request; sending `block_binding` without this header is itself
# rejected ("Extra inputs are not permitted").
BETA_HEADER_THINKING_BINDING = "thinking-binding-controls-2026-08-01"

# Exact substring from Anthropic's documented 400 body [PT]:
#   "The block is bound to a different conversation. Remove the block, or
#    set `thinking.block_binding.prefix_mismatch_behavior` to \"drop_block\"."
# A tampered-signature 400 does NOT contain this sentence and must not be
# retried the same way.
_PREFIX_MISMATCH_MARKER = "bound to a different conversation"

_DROPPED_THINKING_TYPES = ("thinking", "redacted_thinking")


def wire_content_snapshot(sdk_blocks: Iterable[Any]) -> list[dict[str, Any]]:
    """Return the exact server JSON for a response's content blocks.

    ``block.model_dump(mode="json", exclude_unset=True)`` is verified (see
    ``tests/test_opus55_preserved_thinking.py``) to equal the server's own
    JSON byte-for-byte, including unknown block types (e.g. ``fallback``),
    ``redacted_thinking``, empty ``thinking`` text, and ``toolset_name``.
    A plain ``model_dump()`` is NOT equivalent: it fills in fields the
    server never sent (e.g. ``caller: null`` on ``tool_use``, ``text: null``
    on an unknown block), and re-sending those extra keys counts as an edit
    to the previously-sent prefix.
    """
    snapshot: list[dict[str, Any]] = []
    for block in sdk_blocks:
        if hasattr(block, "model_dump"):
            snapshot.append(block.model_dump(mode="json", exclude_unset=True))
        elif isinstance(block, dict):
            snapshot.append(dict(block))
        # Anything else (e.g. a bare string) cannot occur in a real SDK
        # response's `content` and is silently skipped rather than crashing
        # a provider that must keep working on malformed test doubles.
    return snapshot


def tool_use_ids(blocks: Iterable[Mapping[str, Any]]) -> list[str]:
    """Public wrapper: ``tool_use`` block ids, in order, from a wire content
    list or a stored ``wire_content`` snapshot."""
    return _tool_use_ids(blocks)


def _tool_use_ids(blocks: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        block["id"]
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "tool_use" and block.get("id")
    ]


def _message_tool_call_ids(message: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "tool_call" and block.get("id"):
                ids.append(block["id"])
    if not ids:
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, Mapping):
                continue
            tc_id = tool_call.get("id") or tool_call.get("tool_call_id")
            if tc_id:
                ids.append(tc_id)
    return ids


def _message_text(message: Mapping[str, Any]) -> str | None:
    """Concatenated TextBlock text, or ``None`` when content is a plain str
    (skipped per the snapshot-validity contract -- a plain string here means
    the message went through a different code path than the one that wrote
    the snapshot, so text comparison is not meaningful)."""
    content = message.get("content")
    if not isinstance(content, list):
        return None
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, Mapping) and block.get("type") == "text"
    ]
    return "".join(parts)


def snapshot_matches(snapshot: list[dict[str, Any]], message: Mapping[str, Any]) -> bool:
    """True iff a stored ``wire_content`` snapshot still describes ``message``
    well enough to replay byte-exact.

    Guards against a context manager (compaction, screenshot pruning, an
    ``ephemeral_injection_mode: "tail"`` rewrite) having edited the message
    after it was stored: the tool_use ids in order must match the message's
    own tool_call ids in order, and -- when the message's content is a list
    -- the concatenated snapshot text must equal the concatenated message
    text.
    """
    if not snapshot:
        return False
    if _tool_use_ids(snapshot) != _message_tool_call_ids(message):
        return False
    message_text = _message_text(message)
    if message_text is None:
        return True
    snapshot_text = "".join(
        block.get("text", "")
        for block in snapshot
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    return snapshot_text == message_text


def replay_from_snapshot(snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep copy of the exact stored wire content. Nothing dropped or
    reordered -- this is the byte-exact replay path."""
    return copy.deepcopy(snapshot)


def replay_from_content(
    content: list[Any], *, computer_target_type: str | None = None
) -> list[dict[str, Any]]:
    """Ordered fallback reconstruction when no valid snapshot is available
    (e.g. history produced before this feature existed).

    Keeps every thinking block (including empty ``thinking`` text -- an
    empty progress-update block is still a real block that must not be
    silently dropped), preserves order, and never emits ``None`` values or
    the core-only ``visibility`` / ``progress_update`` / ``content`` keys
    the wire API does not accept.
    """
    out: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        block_type = block.get("type")
        if block_type == "thinking":
            entry: dict[str, Any] = {
                "type": "thinking",
                "thinking": block.get("thinking", ""),
            }
            signature = block.get("signature")
            if signature:
                entry["signature"] = signature
            out.append(entry)
        elif block_type == "redacted_thinking":
            out.append({"type": "redacted_thinking", "data": block.get("data", "")})
        elif block_type == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "tool_call":
            wire_name, wire_input, toolset_name = _computer_toolset.to_wire_tool_call(
                block.get("name", ""),
                block.get("input", {}),
                target_type=computer_target_type,
            )
            wire_block: dict[str, Any] = {
                "type": "tool_use",
                "id": block.get("id", ""),
                "name": wire_name,
                "input": wire_input,
            }
            if toolset_name:
                wire_block["toolset_name"] = toolset_name
            out.append(wire_block)
        # Every other core block type (image, tool_result, reasoning, ...)
        # is not a legal assistant-turn wire block on Anthropic and is
        # skipped rather than guessed at.
    return out


def strip_thinking_from_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a copy of ``metadata`` with thinking/redacted_thinking blocks
    removed from ``metadata["anthropic"]["wire_content"]``, if present.

    Used by the refusal-fallback strip path: stripping the *displayed*
    thinking blocks from a request but leaving the exact-replay snapshot
    intact would bring the stripped blocks right back on the next preserved
    -thinking turn.
    """
    if not metadata or METADATA_KEY not in metadata:
        return metadata
    anthropic_meta = metadata[METADATA_KEY]
    if not isinstance(anthropic_meta, Mapping):
        return metadata
    wire_content = anthropic_meta.get(WIRE_CONTENT_KEY)
    if not wire_content:
        return metadata
    filtered = [
        block
        for block in wire_content
        if not (isinstance(block, Mapping) and block.get("type") in _DROPPED_THINKING_TYPES)
    ]
    new_metadata = dict(metadata)
    new_anthropic_meta = dict(anthropic_meta)
    new_anthropic_meta[WIRE_CONTENT_KEY] = filtered
    new_metadata[METADATA_KEY] = new_anthropic_meta
    return new_metadata


_FLOATING_KEY = "_floating"


def place_floating_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically place messages tagged ``{"_floating": "system"|
    "developer"}`` (the tag is removed from the output).

    A floating entry is buffered and released (with the tag stripped)
    immediately after the next ``role: "user"`` message whose immediately
    -following non-floating anchor is ``"assistant"`` or does not exist
    (end of the conversation) -- i.e. exactly the position [PT] describes
    as safe: "leave it where the caller put it" when that is already a
    user-then-assistant boundary, and otherwise defer to the next such
    boundary. Buffered developer entries are released before system
    entries at the same release point. Never releases between an
    assistant tool_use message and its own tool_result message, because a
    tool_result batch is itself a ``role: "user"`` message and the release
    check runs only after a user message, never before or in place of one.

    Appending new messages to the end of an already-processed conversation
    never changes the placement of anything already resolved: this is a
    single forward pass with lookahead bounded by messages already present.
    """

    def next_anchor_role(start: int) -> str | None:
        j = start
        while j < len(messages) and messages[j].get(_FLOATING_KEY) in ("system", "developer"):
            j += 1
        if j >= len(messages):
            return None
        return messages[j].get("role")

    out: list[dict[str, Any]] = []
    pending: list[tuple[str, dict[str, Any]]] = []

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        for kind in ("developer", "system"):
            for entry_kind, entry in pending:
                if entry_kind == kind:
                    out.append(entry)
        pending = []

    for idx, message in enumerate(messages):
        tag = message.get(_FLOATING_KEY)
        if tag in ("system", "developer"):
            cleaned = {key: value for key, value in message.items() if key != _FLOATING_KEY}
            pending.append((tag, cleaned))
            continue
        out.append(message)
        if message.get("role") == "user":
            nxt = next_anchor_role(idx + 1)
            if nxt is None or nxt == "assistant":
                flush()
    flush()
    return out


def is_prefix_binding_mismatch(error_text: str) -> bool:
    """True iff ``error_text`` is Anthropic's documented prefix-binding 400
    [PT] ("...bound to a different conversation..."), as opposed to a
    tampered-signature 400 (no such sentence) or an unrelated 400 (e.g.
    "block_binding: Extra inputs are not permitted")."""
    return _PREFIX_MISMATCH_MARKER in error_text


def summarize_transformations(items: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """Summarize Anthropic's ``input_transformations`` list [PT: "Count the
    transformations you receive back and alert on them"], or ``None`` when
    empty/absent."""
    if not items:
        return None
    reasons: Counter[str] = Counter()
    paths: list[Any] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        reasons[str(item.get("reason", "unknown"))] += 1
        path = item.get("path")
        if path is not None:
            paths.append(path)
    return {"count": len(items), "reasons": dict(reasons), "paths": paths}
