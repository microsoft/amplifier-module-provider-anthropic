# Claude Opus 4.7 Support — Complete Implementation Specification

**Target:** `amplifier-module-provider-anthropic`
**Source file:** `amplifier_module_provider_anthropic/__init__.py` (3,205 lines)
**Approach:** Single-file module. All changes are surgical edits to existing structures.
**Date:** 2026-04-16

---

## Table of Contents

1. [Opus 4.7 Breaking Changes Summary](#1-opus-47-breaking-changes-summary)
2. [Phase Definitions](#2-phase-definitions)
3. [ModelCapabilities Changes](#3-modelcapabilities-changes)
4. [_get_capabilities() Changes](#4-_get_capabilities-changes)
5. [_apply_runtime_capability_overrides() Changes](#5-_apply_runtime_capability_overrides-changes)
6. [Request Building Changes (_complete_chat_request)](#6-request-building-changes-_complete_chat_request)
7. [Effort Mapping Redesign](#7-effort-mapping-redesign)
8. [thinking.display Integration](#8-thinkingdisplay-integration)
9. [Beta Header Fixes](#9-beta-header-fixes)
10. [Interleaved Thinking Header Changes](#10-interleaved-thinking-header-changes)
11. [SDK Compatibility](#11-sdk-compatibility)
12. [Task Budget Architecture](#12-task-budget-architecture)
13. [Test Strategy](#13-test-strategy)
14. [Implementation Order](#14-implementation-order)
15. [What NOT to Do](#15-what-not-to-do)
16. [Success Criteria](#16-success-criteria)

---

## 1. Opus 4.7 Breaking Changes Summary

| # | Change | Impact | Provider area |
|---|--------|--------|---------------|
| 1 | `thinking: {type: "enabled", budget_tokens: N}` returns 400 | Breaks all explicit-budget thinking | Thinking config (line 2015-2027) |
| 2 | `temperature`, `top_p`, `top_k` silently ignored | Misleading UX — users think they control sampling | Params dict (line 1899-1900) |
| 3 | `thinking.display` defaults to `"omitted"` | Thinking blocks come back empty — invisible chain-of-thought | Not implemented yet |
| 4 | Only `thinking: {type: "adaptive"}` works | Must force adaptive for ALL thinking requests | Thinking config (line 2015-2027) |
| 5 | New effort level `"xhigh"` | Between high and max, only on Opus 4.7 | Effort mapping (line 1972-1982) |
| 6 | `output_config` field GA | `{effort: str, format: {...}}` — new primary control | Not implemented yet |
| 7 | Task budgets (beta) | `output_config.task_budget = {type: "tokens", total: N}` | Not implemented yet |
| 8 | New tokenizer: 1.0-1.35x more tokens | Budget calculations may under-allocate | Budget math (line 1985-1997) |
| 9 | 1M context standard | Beta header harmless but no longer required | Beta headers (line 1019-1020) |
| 10 | Interleaved thinking native | No beta header needed with adaptive on 4.7 | Beta headers (line 1023-1040) |
| 11 | SDK 0.96.0 required | New types for output_config, adaptive thinking | pyproject.toml (line 12) |

---

## 2. Phase Definitions

### Phase 1 — P0: Don't Break Existing Users

A user who sets `default_model: claude-opus-4-7` gets a working provider, not 400 errors.

**Scope:**
- Add `supports_manual_thinking` field to `ModelCapabilities`
- Wire `is_47_plus` in `_get_capabilities()` for Opus family
- Gate `type="enabled"` off for Opus 4.7+ — force adaptive
- Fix 1M beta header from `==` to `>=`
- Bump SDK to `>=0.96.0`
- New test file `tests/test_opus_47.py`

**NOT in Phase 1:** output_config, thinking.display, temperature stripping, xhigh effort.

### Phase 2 — P1: Full Support

Opus 4.7 users get proper effort control, visible thinking, clean params.

**Scope:**
- Add `supports_output_config`, `supports_sampling`, `thinking_display_required`, `supported_efforts` fields
- Build `output_config` in request when model supports it
- Add `thinking.display = "summarized"` for 4.7+ models
- Strip `temperature` for non-sampling models
- Support `"xhigh"` effort level
- Move `request_caps` fetch earlier in `_complete_chat_request`

### Phase 3 — P2: Forward-Looking

**Scope:**
- Task budget support via `output_config.task_budget` (beta)
- Deprecation warnings for retiring models
- Tokenizer budget recalibration field
- (Future) Refactor reasoning_effort → output_config as primary surface

---

## 3. ModelCapabilities Changes

### Current (line 204-220):

```python
@dataclass(frozen=True)
class ModelCapabilities:
    """Per-model capability matrix — single source of truth.

    Every model-specific decision in the provider (context window size,
    thinking mode, output capacity, etc.) should be derived from this
    dataclass rather than scattered if/else checks.
    """

    family: str
    max_output_tokens: int = 64000
    base_context_window: int = 200000
    supports_1m: bool = False
    supports_thinking: bool = False
    supports_adaptive_thinking: bool = False
    default_thinking_budget: int = 0
    capability_tags: tuple[str, ...] = ("tools", "streaming", "json_mode")
```

### After Phase 1 (add 1 field):

```python
@dataclass(frozen=True)
class ModelCapabilities:
    """Per-model capability matrix — single source of truth.

    Every model-specific decision in the provider (context window size,
    thinking mode, output capacity, etc.) should be derived from this
    dataclass rather than scattered if/else checks.
    """

    family: str
    max_output_tokens: int = 64000
    base_context_window: int = 200000
    supports_1m: bool = False
    supports_thinking: bool = False
    supports_adaptive_thinking: bool = False
    supports_manual_thinking: bool = True      # P1: False on Opus 4.7+ (type="enabled" → 400)
    default_thinking_budget: int = 0
    capability_tags: tuple[str, ...] = ("tools", "streaming", "json_mode")
```

**Field:** `supports_manual_thinking: bool = True`
**Meaning:** When `False`, the model rejects `thinking: {type: "enabled", budget_tokens: N}` with HTTP 400. The provider must use `type: "adaptive"` exclusively.
**Default:** `True` — preserves backward compat for all existing models (Opus 4.5, 4.6, all Sonnet, all Haiku).
**Set to `False` for:** Opus 4.7+ and unknown Opus versions.

### After Phase 2 (add 4 more fields):

```python
@dataclass(frozen=True)
class ModelCapabilities:
    """Per-model capability matrix — single source of truth.

    Every model-specific decision in the provider (context window size,
    thinking mode, output capacity, etc.) should be derived from this
    dataclass rather than scattered if/else checks.
    """

    family: str
    max_output_tokens: int = 64000
    base_context_window: int = 200000
    supports_1m: bool = False
    supports_thinking: bool = False
    supports_adaptive_thinking: bool = False
    supports_manual_thinking: bool = True      # P1: False on Opus 4.7+
    supports_output_config: bool = False       # P2: output_config.effort GA
    supports_sampling: bool = True             # P2: False = temperature silently ignored
    thinking_display_required: bool = False    # P2: must send display:"summarized" to see thinking
    supported_efforts: tuple[str, ...] = ("low", "medium", "high")  # P2: valid effort levels
    default_thinking_budget: int = 0
    capability_tags: tuple[str, ...] = ("tools", "streaming", "json_mode")
```

**New Phase 2 fields:**

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `supports_output_config` | `bool` | `False` | Model supports `output_config` in the request body |
| `supports_sampling` | `bool` | `True` | `False` means temperature/top_p/top_k are silently ignored |
| `thinking_display_required` | `bool` | `False` | `True` means thinking.display defaults to "omitted" server-side |
| `supported_efforts` | `tuple[str, ...]` | `("low", "medium", "high")` | Valid values for output_config.effort |

### After Phase 3 (add 2 more fields):

```python
    supports_task_budget: bool = False         # P3: output_config.task_budget beta
    tokenizer_scale: float = 1.0              # P3: 4.7+ ≈ 1.15 (tokens are larger)
```

---

## 4. _get_capabilities() Changes

### Location: line 807-873

### Current Opus block (line 825-841):

```python
if family == "opus":
    is_46_plus = not version_known or (major, minor) >= (4, 6)
    return ModelCapabilities(
        family="opus",
        max_output_tokens=128000 if is_46_plus else 64000,
        supports_1m=is_46_plus,
        supports_thinking=True,
        supports_adaptive_thinking=is_46_plus,
        default_thinking_budget=64000 if is_46_plus else 32000,
        capability_tags=(
            "tools",
            "thinking",
            "streaming",
            "json_mode",
            "vision",
        ),
    )
```

### After Phase 1:

```python
if family == "opus":
    is_46_plus = not version_known or (major, minor) >= (4, 6)
    is_47_plus = not version_known or (major, minor) >= (4, 7)
    return ModelCapabilities(
        family="opus",
        max_output_tokens=128000 if is_46_plus else 64000,
        supports_1m=is_46_plus,
        supports_thinking=True,
        supports_adaptive_thinking=is_46_plus,
        supports_manual_thinking=not is_47_plus,
        default_thinking_budget=64000 if is_46_plus else 32000,
        capability_tags=(
            "tools",
            "thinking",
            "streaming",
            "json_mode",
            "vision",
        ),
    )
```

**Changes:** 2 lines added (`is_47_plus`, `supports_manual_thinking`).

**Forward-compat behavior:** Unknown Opus (`version_known=False`) → `is_47_plus=True` → `supports_manual_thinking=False`. Correct: unknown means latest, latest rejects manual thinking.

### After Phase 2 — full Opus block:

```python
if family == "opus":
    is_46_plus = not version_known or (major, minor) >= (4, 6)
    is_47_plus = not version_known or (major, minor) >= (4, 7)
    return ModelCapabilities(
        family="opus",
        max_output_tokens=128000 if is_46_plus else 64000,
        supports_1m=is_46_plus,
        supports_thinking=True,
        supports_adaptive_thinking=is_46_plus,
        supports_manual_thinking=not is_47_plus,
        supports_output_config=is_47_plus,
        supports_sampling=not is_47_plus,
        thinking_display_required=is_47_plus,
        supported_efforts=(
            ("low", "medium", "high", "xhigh") if is_47_plus
            else ("low", "medium", "high")
        ),
        default_thinking_budget=64000 if is_46_plus else 32000,
        capability_tags=(
            "tools",
            "thinking",
            "streaming",
            "json_mode",
            "vision",
        ),
    )
```

### Sonnet block — unchanged in Phase 1

No Sonnet model has the manual-thinking restriction, sampling restriction, or output_config.
All new fields use their defaults (`True` for manual_thinking, `True` for sampling, `False` for output_config, etc.).
The Sonnet block (line 843-859) stays exactly as-is through all three phases.

### Haiku block — unchanged in Phase 1

Same reasoning. The Haiku block (line 861-870) stays as-is.

### Unknown family block — unchanged

Line 872-873: `return ModelCapabilities(family=family)` — all defaults apply, which is correct.

### After Phase 3 — Opus block additions:

```python
        supports_task_budget=is_47_plus,
        tokenizer_scale=1.15 if is_47_plus else 1.0,
```

---

## 5. _apply_runtime_capability_overrides() Changes

### Location: line 923-969

This method overlays live Models API metadata onto static heuristics. It constructs a new `ModelCapabilities` but currently only passes through a subset of fields.

### Phase 1 change:

The `return ModelCapabilities(...)` call at line 960-969 must pass through the new field:

```python
return ModelCapabilities(
    family=base_caps.family,
    max_output_tokens=runtime_info.max_tokens or base_caps.max_output_tokens,
    base_context_window=base_context_window,
    supports_1m=supports_1m,
    supports_thinking=supports_thinking,
    supports_adaptive_thinking=supports_adaptive_thinking,
    supports_manual_thinking=base_caps.supports_manual_thinking,    # Pass through
    default_thinking_budget=default_thinking_budget,
    capability_tags=tuple(capability_tags),
)
```

### Phase 2 change:

Pass through all Phase 2 fields:

```python
return ModelCapabilities(
    family=base_caps.family,
    max_output_tokens=runtime_info.max_tokens or base_caps.max_output_tokens,
    base_context_window=base_context_window,
    supports_1m=supports_1m,
    supports_thinking=supports_thinking,
    supports_adaptive_thinking=supports_adaptive_thinking,
    supports_manual_thinking=base_caps.supports_manual_thinking,
    supports_output_config=base_caps.supports_output_config,
    supports_sampling=base_caps.supports_sampling,
    thinking_display_required=base_caps.thinking_display_required,
    supported_efforts=base_caps.supported_efforts,
    default_thinking_budget=default_thinking_budget,
    capability_tags=tuple(capability_tags),
)
```

**Why pass-through (not runtime-derived):** The Models API does not expose these
capabilities. They are version-derived heuristics. The runtime override layer only
adjusts fields that the Models API provides (max_tokens, thinking support, context window).

---

## 6. Request Building Changes (_complete_chat_request)

### Location: line 1821-2097 (the entire method)

### 6a. Phase 1 — Force adaptive for models without manual thinking

**Location: line 2015-2027**

**Current code:**

```python
            # Adaptive thinking: model controls its own budget.  The API schema
            # is a discriminated union — "adaptive" accepts NO extra fields
            # (budget_tokens is forbidden).  Fall back to "enabled" with an
            # explicit budget when the model doesn't support adaptive.
            if thinking_type == "adaptive" and request_caps.supports_adaptive_thinking:
                params["thinking"] = {"type": "adaptive"}
                resolved_thinking_type = "adaptive"
            else:
                # "enabled" mode (all thinking-capable models): explicit budget
                if thinking_type == "adaptive":
                    # Caller asked for adaptive but model doesn't support it
                    thinking_type = "enabled"
                resolved_thinking_type = thinking_type
                params["thinking"] = {
                    "type": thinking_type,
                    "budget_tokens": budget_tokens,
                }
```

**Replace with:**

```python
            # Adaptive thinking: model controls its own budget.  The API schema
            # is a discriminated union — "adaptive" accepts NO extra fields
            # (budget_tokens is forbidden).  Fall back to "enabled" with an
            # explicit budget when the model doesn't support adaptive.
            if thinking_type == "adaptive" and request_caps.supports_adaptive_thinking:
                params["thinking"] = {"type": "adaptive"}
                resolved_thinking_type = "adaptive"
            elif not request_caps.supports_manual_thinking:
                # Model rejects type="enabled" (e.g. Opus 4.7+) — force adaptive.
                # This is safe because models that don't support manual thinking
                # always support adaptive thinking.
                if thinking_type != "adaptive":
                    logger.info(
                        "[PROVIDER] Model %s does not support manual thinking "
                        "(type='enabled') — using adaptive instead of '%s'",
                        params["model"],
                        thinking_type,
                    )
                params["thinking"] = {"type": "adaptive"}
                resolved_thinking_type = "adaptive"
            else:
                # "enabled" mode (all thinking-capable models): explicit budget
                if thinking_type == "adaptive":
                    # Caller asked for adaptive but model doesn't support it
                    thinking_type = "enabled"
                resolved_thinking_type = thinking_type
                params["thinking"] = {
                    "type": thinking_type,
                    "budget_tokens": budget_tokens,
                }
```

**What changed:** One new `elif` branch (8 lines). Existing branches untouched.

**Invariant preserved:** `thinking_budget` (set at line 2002) and `budget_tokens` are
still computed normally. The `max_tokens` ceiling calculation at line 2037-2043 uses
`budget_tokens` which is available regardless of which branch was taken.

### 6b. Phase 1 — Temperature forcing adjustment

**Location: line 2029-2030**

**Current code:**

```python
            # CRITICAL: Anthropic requires temperature=1.0 when thinking is enabled
            params["temperature"] = 1.0
```

**Phase 1: No change.** This runs inside the `if thinking_enabled:` block. On 4.7, temperature
is silently ignored, so forcing 1.0 is harmless. Phase 2 removes it properly.

### 6c. Phase 2 — Move request_caps fetch BEFORE params dict

**Current flow (line 1893-1925):**

```python
        # Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "messages": all_messages,
            "max_tokens": request.max_output_tokens
            or kwargs.get("max_tokens", self.max_tokens),
            "temperature": request.temperature
            or kwargs.get("temperature", self.temperature),
        }
        # ... tools, web search ...
        request_caps = await self._get_request_capabilities(params["model"])  # line 1925
```

**Phase 2 flow:**

```python
        # Resolve model and capabilities BEFORE building params dict,
        # so per-model param gating (temperature, output_config) can apply.
        effective_model = kwargs.get("model", self.default_model)
        request_caps = await self._get_request_capabilities(effective_model)
        model_ceiling = request_caps.max_output_tokens

        # Prepare request parameters
        params: dict[str, Any] = {
            "model": effective_model,
            "messages": all_messages,
            "max_tokens": request.max_output_tokens
            or kwargs.get("max_tokens", self.max_tokens),
        }

        # Only include temperature for models that support sampling.
        # Opus 4.7+ silently ignores temperature — omitting it avoids user confusion
        # and keeps request payloads clean.
        if request_caps.supports_sampling:
            params["temperature"] = request.temperature or kwargs.get(
                "temperature", self.temperature
            )
        else:
            if request.temperature is not None or kwargs.get("temperature") is not None:
                logger.info(
                    "[PROVIDER] Model %s does not support sampling parameters"
                    " — ignoring temperature setting",
                    params["model"],
                )
```

**Then delete** the old `request_caps` and `model_ceiling` lines (1925-1926).

### 6d. Phase 2 — Temperature in thinking block

**Location: line 2029-2030**

**Current code:**

```python
            # CRITICAL: Anthropic requires temperature=1.0 when thinking is enabled
            params["temperature"] = 1.0
```

**Replace with:**

```python
            # Anthropic requires temperature=1.0 when thinking is enabled
            # on models that support sampling. Non-sampling models (4.7+)
            # ignore temperature entirely — don't inject it.
            if request_caps.supports_sampling:
                params["temperature"] = 1.0
```

### 6e. Phase 2 — output_config support

**Insert AFTER the thinking block (after line 2053), BEFORE stop_sequences (line 2064):**

```python
        # Build output_config for models that support it (Opus 4.7+).
        # output_config.effort is the primary control surface for thinking
        # intensity on these models, replacing the budget_tokens approach.
        if request_caps.supports_output_config and reasoning_effort is not None:
            effort = kwargs.get("effort", reasoning_effort)
            if effort in request_caps.supported_efforts:
                params["output_config"] = {"effort": effort}
                logger.info(
                    "[PROVIDER] output_config.effort=%s for %s",
                    effort,
                    params["model"],
                )
            else:
                logger.warning(
                    "[PROVIDER] Effort level '%s' not supported by %s "
                    "(supported: %s) — omitting output_config.effort",
                    effort,
                    params["model"],
                    request_caps.supported_efforts,
                )
```

**Also allow explicit effort via kwargs even without reasoning_effort:**

```python
        # Allow explicit effort override via kwargs (e.g. from orchestrator)
        if request_caps.supports_output_config and "effort" in kwargs:
            effort = kwargs["effort"]
            if effort in request_caps.supported_efforts:
                if "output_config" not in params:
                    params["output_config"] = {}
                params["output_config"]["effort"] = effort
```

---

## 7. Effort Mapping Redesign

### Current mapping (line 1972-1982):

```python
            effort_thinking_type: str | None = None
            effort_budget: int | None = None
            if reasoning_effort == "low":
                effort_thinking_type = "enabled"
                effort_budget = 4096
            elif reasoning_effort == "medium":
                effort_thinking_type = "adaptive"
                effort_budget = request_caps.default_thinking_budget
            elif reasoning_effort == "high":
                effort_thinking_type = "adaptive"
                effort_budget = request_caps.default_thinking_budget
```

### Phase 2 — add xhigh:

```python
            effort_thinking_type: str | None = None
            effort_budget: int | None = None
            if reasoning_effort == "low":
                effort_thinking_type = "enabled"
                effort_budget = 4096
            elif reasoning_effort == "medium":
                effort_thinking_type = "adaptive"
                effort_budget = request_caps.default_thinking_budget
            elif reasoning_effort == "high":
                effort_thinking_type = "adaptive"
                effort_budget = request_caps.default_thinking_budget
            elif reasoning_effort == "xhigh":
                effort_thinking_type = "adaptive"
                effort_budget = request_caps.default_thinking_budget
```

### Complete mapping table (all models × all efforts):

| `reasoning_effort` | Opus 4.5 (no adaptive) | Opus 4.6 (adaptive, manual) | Opus 4.7 (adaptive only) | Sonnet 4.6 (adaptive, manual) | Haiku 4.5 (manual only) |
|---|---|---|---|---|---|
| `"low"` | `enabled`, budget=4096 | `enabled`, budget=4096 | `adaptive` + `output_config.effort="low"` | `enabled`, budget=4096 | `enabled`, budget=4096 |
| `"medium"` | `enabled`, budget=32000 | `adaptive` | `adaptive` + `output_config.effort="medium"` | `adaptive` | `enabled`, budget=32000 |
| `"high"` | `enabled`, budget=32000 | `adaptive` | `adaptive` + `output_config.effort="high"` | `adaptive` | `enabled`, budget=32000 |
| `"xhigh"` | `enabled`, budget=32000 | `adaptive` | `adaptive` + `output_config.effort="xhigh"` | `adaptive` | `enabled`, budget=32000 |
| `None` | no thinking | no thinking | no thinking | no thinking | no thinking |

**Phase 1 known compromise:** `reasoning_effort="low"` on Opus 4.7 becomes plain `adaptive`
(losing the low-budget intent). Phase 2 restores granularity via `output_config.effort="low"`.
The logger.info from §6a makes this visible.

**xhigh on older models:** Falls through to adaptive (where supported) or enabled (where not).
Treated as max effort. No error — the effort_budget is model_default either way.

---

## 8. thinking.display Integration

### Background

Opus 4.7 defaults `thinking.display` to `"omitted"`, meaning thinking blocks in responses
have `content: ""` (empty string). Users see no chain-of-thought. The provider must send
`display: "summarized"` to restore visibility.

### Phase 2 implementation

**Insert inside the thinking config block, immediately after `params["thinking"]` is set
(after the adaptive/enabled/fallback branches, before the temperature line).**

The exact insertion point is AFTER line 2027 (after all three branches that set
`params["thinking"]`), BEFORE line 2029 (the temperature forcing):

```python
            # For models where thinking.display defaults to "omitted" (Opus 4.7+),
            # request "summarized" so thinking content is visible to users.
            # Users can override via config or kwargs to "omitted" if desired.
            if request_caps.thinking_display_required:
                display = kwargs.get(
                    "thinking_display",
                    self.config.get("thinking_display", "summarized"),
                )
                params["thinking"]["display"] = display
```

### Config surface

New config field `thinking_display` (string):
- Default: `"summarized"` (hardcoded in the code above — NOT from config, which defaults to None)
- Valid values: `"summarized"`, `"omitted"`
- Only used when `thinking_display_required=True` (Opus 4.7+)
- Ignored for older models (they don't support the field)

### kwargs surface

`kwargs["thinking_display"]` overrides config per-request. Useful for orchestrator
scenarios where some steps want visible thinking and others don't.

### Precedence

```
kwargs["thinking_display"] > config["thinking_display"] > "summarized"
```

### Effect on older models

None. `thinking_display_required` defaults to `False`. The code block is skipped entirely.
`params["thinking"]` remains exactly as before for Opus 4.6, Sonnet, Haiku.

---

## 9. Beta Header Fixes

### 9a. 1M context header — fix version check

**Location: line 1002-1021**

**Current code:**

```python
    def _should_add_context_1m_beta(
        self, model_id: str, request_caps: ModelCapabilities
    ) -> bool:
        """Return True when the effective model still needs the 1M beta header."""
        if not self._enable_1m_context:
            return False

        family = self._detect_family(model_id)
        if family == "haiku":
            return False

        major, minor = self._detect_version(model_id, family)
        version = (major, minor)

        # Anthropic's current context-window docs list 1M as a beta-header feature
        # for Opus 4.6 and Sonnet 4.x. Keep the header scoped to those families so
        # lower-tier fallbacks like Haiku don't inherit it.
        if family == "opus":
            return version == (4, 6)
        return family == "sonnet" and version in {(4, 0), (4, 5), (4, 6)}
```

**Replace with:**

```python
    def _should_add_context_1m_beta(
        self, model_id: str, request_caps: ModelCapabilities
    ) -> bool:
        """Return True when the effective model still needs the 1M beta header."""
        if not self._enable_1m_context:
            return False

        family = self._detect_family(model_id)
        if family == "haiku":
            return False

        major, minor = self._detect_version(model_id, family)
        version = (major, minor)

        # Send the 1M beta header for known 1M-capable versions and unknown
        # versions (forward-compat).  The header is harmless on models where
        # 1M context is already GA (e.g. Opus 4.7+), but omitting it breaks
        # models that still need it.
        if family == "opus":
            return version == (0, 0) or version >= (4, 6)
        return family == "sonnet" and (version == (0, 0) or version >= (4, 0))
```

**Changes:**
- Opus: `version == (4, 6)` → `version == (0, 0) or version >= (4, 6)`
- Sonnet: `version in {(4, 0), (4, 5), (4, 6)}` → `version == (0, 0) or version >= (4, 0)`

**Why `(0, 0)` explicit check:** `(0, 0) >= (4, 6)` is `False`, so unknown versions
would NOT get the header under a simple `>=` check. The explicit `(0, 0)` handles the
forward-compat case where we can't parse the version.

### 9b. Phase 3 — Task budget beta header

**New constant (after line 198):**

```python
BETA_HEADER_TASK_BUDGETS = "task-budgets-2026-03-13"
```

**Add to `_build_request_beta_headers` (line 1042-1060), after the interleaved check:**

```python
        # Task budget beta header — only when task_budget is present in output_config
        if "output_config" in params and "task_budget" in params.get("output_config", {}):
            headers.append(BETA_HEADER_TASK_BUDGETS)
```

**Problem:** `_build_request_beta_headers` doesn't currently receive `params`.
Its signature is:

```python
    def _build_request_beta_headers(
        self,
        *,
        model_id: str,
        request_caps: ModelCapabilities,
        tools_present: bool,
        resolved_thinking_type: str | None,
    ) -> list[str]:
```

**Phase 3 signature change:**

```python
    def _build_request_beta_headers(
        self,
        *,
        model_id: str,
        request_caps: ModelCapabilities,
        tools_present: bool,
        resolved_thinking_type: str | None,
        has_task_budget: bool = False,
    ) -> list[str]:
```

Then add at end before `return`:

```python
        if has_task_budget:
            headers.append(BETA_HEADER_TASK_BUDGETS)
```

And update the call site (line 2068-2073) to pass `has_task_budget`:

```python
        request_beta_headers = self._build_request_beta_headers(
            model_id=params["model"],
            request_caps=request_caps,
            tools_present=bool(params.get("tools")),
            resolved_thinking_type=resolved_thinking_type,
            has_task_budget="task_budget" in params.get("output_config", {}),
        )
```

---

## 10. Interleaved Thinking Header Changes

### Location: line 1023-1040

**Current `_should_add_interleaved_beta`:**

```python
    def _should_add_interleaved_beta(
        self,
        *,
        request_caps: ModelCapabilities,
        tools_present: bool,
        resolved_thinking_type: str | None,
    ) -> bool:
        """Return True when tool-use thinking should opt into interleaving beta."""
        if not tools_present or not request_caps.supports_thinking:
            return False
        if request_caps.family == "haiku":
            return False
        if (
            resolved_thinking_type == "adaptive"
            and request_caps.supports_adaptive_thinking
        ):
            return False
        return resolved_thinking_type is not None
```

**Analysis:** This is already correct for Opus 4.7. When `resolved_thinking_type == "adaptive"`
and `supports_adaptive_thinking == True` (which Opus 4.7 has), it returns `False` — no
interleaved-thinking beta header sent. Opus 4.7 does interleaving natively with adaptive.

**No changes needed** in any phase. The existing logic handles 4.7 correctly because
Phase 1 forces all 4.7 thinking to adaptive.

---

## 11. SDK Compatibility

### 11a. pyproject.toml change

**File:** `pyproject.toml`, line 11-13

**Current:**
```toml
dependencies = [
    "anthropic>=0.25.0",
]
```

**Change to:**
```toml
dependencies = [
    "anthropic>=0.96.0",
]
```

**Then run:** `cd amplifier-module-provider-anthropic && uv lock`

### 11b. Why 0.96.0

SDK 0.96.0 is the minimum version that includes:
- Opus 4.7 model ID recognition
- `output_config` type definitions
- `ThinkingConfigAdaptive` with `display` field support
- Updated `supported_models` lists

### 11c. OverloadedError import risk

**Location: line 55-57**

```python
from anthropic._exceptions import (
    OverloadedError as AnthropicOverloadedError,
)  # Not exported in public API as of SDK v0.75.0
```

This imports from `anthropic._exceptions` (private module). Between SDK 0.75.0 and 0.96.0,
this may have been promoted to `anthropic.OverloadedError` (public API).

**Action for implementer:** After bumping the SDK, check:

```python
# Try public import first
try:
    from anthropic import OverloadedError as AnthropicOverloadedError
except ImportError:
    from anthropic._exceptions import (
        OverloadedError as AnthropicOverloadedError,
    )
```

**Or** just test `from anthropic import OverloadedError` in a Python shell with 0.96.0 installed.
If it works, replace the private import with the public one and update the comment.

### 11d. Other SDK types used

All other imports (line 50-54) are stable public API:
- `anthropic.APIStatusError` — stable
- `anthropic.AsyncAnthropic` — stable
- `anthropic.AuthenticationError` — stable
- `anthropic.BadRequestError` — stable
- `anthropic.RateLimitError` — stable

### 11e. Response duck-typing

The provider accesses response attributes via duck-typing:
- `response.content` — list of content blocks
- `response.usage.input_tokens` / `.output_tokens`
- `response.stop_reason`
- `response.model`

These are stable across SDK versions. No changes expected.

---

## 12. Task Budget Architecture

### Phase 3 only.

### 12a. Config surface

New optional config field:

```yaml
providers:
  anthropic:
    task_budget_tokens: 50000  # Optional. Minimum: 20000.
```

### 12b. kwargs surface

```python
response = await provider.complete(request, task_budget_tokens=100000)
```

### 12c. Request building

**Insert after the output_config.effort block (Phase 2 §6e), still inside the
`if request_caps.supports_output_config:` guard:**

```python
        # Task budget (beta) — allocates a total token budget for multi-step tasks.
        # The model manages how it distributes tokens across thinking and output.
        if request_caps.supports_task_budget:
            task_budget = kwargs.get("task_budget_tokens") or self.config.get(
                "task_budget_tokens"
            )
            if task_budget is not None:
                task_budget = max(20000, int(task_budget))  # API minimum 20K
                if "output_config" not in params:
                    params["output_config"] = {}
                params["output_config"]["task_budget"] = {
                    "type": "tokens",
                    "total": task_budget,
                }
                logger.info(
                    "[PROVIDER] Task budget=%d tokens for %s",
                    task_budget,
                    params["model"],
                )
```

### 12d. Orchestrator connection

Task budgets are set per-request. The orchestrator layer passes `task_budget_tokens`
via kwargs when it detects multi-step agentic tasks. This is the caller's responsibility,
not the provider's. The provider just wires the value through.

### 12e. ModelCapabilities field

```python
supports_task_budget: bool = False  # Only Opus 4.7+
```

Set to `is_47_plus` in the Opus block of `_get_capabilities()`.

---

## 13. Test Strategy

### 13a. New file: `tests/test_opus_47.py`

Uses the same test infrastructure as `tests/test_reasoning_effort.py`:
- `_make_provider(default_model=...)` helper
- `_make_raw_mock()` for API response stubs
- `_get_api_params(mock_create)` to extract API call kwargs
- `FakeCoordinator` / `FakeHooks` for event emission

### 13b. Phase 1 test cases

```python
"""Tests for Claude Opus 4.7 support.

Phase 1: Validates capability detection, manual-thinking fallback,
and 1M beta header fix.
"""


class TestOpus47Capabilities:
    """ModelCapabilities for Opus 4.7 models."""

    def test_opus_47_supports_manual_thinking_false(self):
        """Opus 4.7 rejects type='enabled' — supports_manual_thinking must be False."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_manual_thinking is False

    def test_opus_47_supports_adaptive_thinking_true(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_adaptive_thinking is True

    def test_opus_47_max_output_tokens_128k(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.max_output_tokens == 128000

    def test_opus_47_supports_1m(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_1m is True

    def test_opus_47_default_thinking_budget_64k(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.default_thinking_budget == 64000

    def test_opus_47_family(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.family == "opus"

    def test_opus_47_supports_thinking(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_thinking is True

    def test_opus_46_still_supports_manual_thinking(self):
        """Opus 4.6 must retain manual thinking support (backward compat)."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.supports_manual_thinking is True

    def test_opus_45_still_supports_manual_thinking(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-5-20251101")
        assert caps.supports_manual_thinking is True

    def test_opus_unknown_assumes_no_manual_thinking(self):
        """Unknown Opus → latest → no manual thinking."""
        caps = AnthropicProvider._get_capabilities("claude-opus-latest")
        assert caps.supports_manual_thinking is False

    def test_sonnet_unaffected(self):
        """Sonnet models retain manual thinking support."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-6-20260101")
        assert caps.supports_manual_thinking is True

    def test_haiku_unaffected(self):
        """Haiku models retain manual thinking support."""
        caps = AnthropicProvider._get_capabilities("claude-haiku-4-5-20251001")
        assert caps.supports_manual_thinking is True


class TestOpus47ThinkingFallback:
    """Thinking config forced to adaptive on Opus 4.7."""

    def test_opus_47_low_effort_forces_adaptive(self):
        """reasoning_effort='low' on 4.7 → type='adaptive' (not 'enabled')."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="low",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in params["thinking"]

    def test_opus_47_medium_effort_uses_adaptive(self):
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="medium",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "adaptive"

    def test_opus_47_high_effort_uses_adaptive(self):
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "adaptive"

    def test_opus_47_config_thinking_type_enabled_forces_adaptive(self):
        """Even if config says thinking_type='enabled', 4.7 forces adaptive."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.config["thinking_type"] = "enabled"
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "adaptive"

    def test_opus_47_max_tokens_still_generous(self):
        """max_tokens ceiling calculation still works with forced adaptive."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["max_tokens"] >= 64000

    def test_opus_46_low_effort_still_uses_enabled(self):
        """Opus 4.6 + low → type='enabled', budget=4096 (backward compat)."""
        provider = _make_provider(default_model="claude-opus-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="low",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "enabled"
        assert params["thinking"]["budget_tokens"] == 4096


class TestBetaHeader1MFix:
    """1M context beta header uses >= instead of ==."""

    def _check(self, model_id: str) -> bool:
        provider = _make_provider(default_model=model_id)
        caps = AnthropicProvider._get_capabilities(model_id)
        return provider._should_add_context_1m_beta(model_id, caps)

    def test_opus_46_gets_1m_header(self):
        assert self._check("claude-opus-4-6-20260101") is True

    def test_opus_47_gets_1m_header(self):
        assert self._check("claude-opus-4-7-20260416") is True

    def test_opus_unknown_gets_1m_header(self):
        assert self._check("claude-opus-latest") is True

    def test_opus_45_no_1m_header(self):
        assert self._check("claude-opus-4-5-20251101") is False

    def test_haiku_never_gets_1m_header(self):
        assert self._check("claude-haiku-4-5-20251001") is False

    def test_sonnet_46_gets_1m_header(self):
        assert self._check("claude-sonnet-4-6-20260101") is True

    def test_sonnet_45_gets_1m_header(self):
        assert self._check("claude-sonnet-4-5-20250929") is True

    def test_sonnet_unknown_gets_1m_header(self):
        assert self._check("claude-sonnet-latest") is True
```

### 13c. Phase 2 test cases (append to same file)

```python
class TestOpus47OutputConfig:
    """output_config.effort on Opus 4.7."""

    def test_opus_47_high_effort_sends_output_config(self):
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"] == {"effort": "high"}

    def test_opus_47_xhigh_effort_sends_output_config(self):
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="xhigh",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"] == {"effort": "xhigh"}

    def test_opus_47_low_effort_sends_output_config(self):
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="low",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"] == {"effort": "low"}

    def test_opus_47_no_effort_no_output_config(self):
        """reasoning_effort=None → no output_config at all."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" not in params

    def test_opus_46_no_output_config(self):
        """Opus 4.6 doesn't support output_config — never sent."""
        provider = _make_provider(default_model="claude-opus-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" not in params

    def test_opus_47_supported_efforts(self):
        """Opus 4.7 capabilities include xhigh."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert "xhigh" in caps.supported_efforts
        assert caps.supported_efforts == ("low", "medium", "high", "xhigh")

    def test_opus_46_no_xhigh(self):
        """Opus 4.6 capabilities don't include xhigh."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert "xhigh" not in caps.supported_efforts


class TestOpus47ThinkingDisplay:
    """thinking.display integration for Opus 4.7."""

    def test_opus_47_thinking_sends_display_summarized(self):
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["display"] == "summarized"

    def test_opus_47_display_config_override(self):
        """Config thinking_display='omitted' overrides default."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.config["thinking_display"] = "omitted"
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["display"] == "omitted"

    def test_opus_47_display_kwargs_override(self):
        """kwargs thinking_display overrides config."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.config["thinking_display"] = "omitted"
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request, thinking_display="summarized"))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["display"] == "summarized"

    def test_opus_46_no_display_field(self):
        """Opus 4.6 thinking dict has no display field."""
        provider = _make_provider(default_model="claude-opus-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "display" not in params["thinking"]

    def test_opus_47_thinking_display_required_flag(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.thinking_display_required is True

    def test_opus_46_thinking_display_required_false(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.thinking_display_required is False


class TestOpus47Temperature:
    """Temperature stripping for non-sampling models."""

    def test_opus_47_no_temperature_in_params(self):
        """Opus 4.7 requests should not include temperature."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "temperature" not in params

    def test_opus_47_explicit_temperature_ignored(self):
        """Even if user sets temperature, Opus 4.7 omits it."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            temperature=0.5,
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "temperature" not in params

    def test_opus_47_thinking_does_not_force_temperature_1(self):
        """With thinking on 4.7, temperature is omitted (not forced to 1.0)."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "temperature" not in params

    def test_opus_46_temperature_still_sent(self):
        """Opus 4.6 still includes temperature in params."""
        provider = _make_provider(default_model="claude-opus-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "temperature" in params

    def test_opus_47_supports_sampling_false(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_sampling is False

    def test_opus_46_supports_sampling_true(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.supports_sampling is True

    def test_sonnet_supports_sampling_true(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-6-20260101")
        assert caps.supports_sampling is True
```

### 13d. Existing test files — impact assessment

| Test file | Phase 1 impact | Phase 2 impact |
|-----------|---------------|---------------|
| `test_model_capabilities.py` | **No changes.** All existing tests pass. `supports_manual_thinking` defaults to `True`, matching all tested models. | **No changes.** New fields all have backward-compat defaults. |
| `test_reasoning_effort.py` | **No changes.** Tests use Sonnet 4.5 and Opus 4.6 — both have `supports_manual_thinking=True`. | **No changes.** Tests don't check for absence of temperature or output_config. |
| `test_behavioral.py` | **No changes.** Uses default model (Sonnet 4.5). | **No changes.** |
| `test_retry.py` | **No changes.** | **No changes.** |
| `test_error_translation.py` | **No changes.** | **No changes.** |
| `test_fallback.py` | **No changes.** | **No changes.** |
| `test_cloudflare_retry.py` | **No changes.** | **No changes.** |
| `test_throttle.py` | **No changes.** | **No changes.** |
| All others | **No changes.** | **No changes.** |

**Zero regressions expected.** All new fields default to backward-compatible values.

---

## 14. Implementation Order

### Phase 1 implementation sequence:

```
Step 1: Add supports_manual_thinking to ModelCapabilities          (line 218)
Step 2: Wire is_47_plus + supports_manual_thinking in Opus block   (line 825-841)
Step 3: Pass through in _apply_runtime_capability_overrides        (line 960-969)
Step 4: Add elif branch in thinking resolution                     (line 2015-2027)
Step 5: Fix _should_add_context_1m_beta                            (line 1019-1021)
Step 6: Bump SDK in pyproject.toml                                 (line 12)
Step 7: Run uv lock
Step 8: Check/fix OverloadedError import                           (line 55-57)
Step 9: Create tests/test_opus_47.py with Phase 1 tests
Step 10: Run full test suite — zero failures expected
```

### Phase 2 implementation sequence:

```
Step 1: Add 4 new fields to ModelCapabilities                      (line 218)
Step 2: Wire all fields in Opus block of _get_capabilities         (line 825-841)
Step 3: Pass through all in _apply_runtime_capability_overrides    (line 960-969)
Step 4: Move request_caps fetch BEFORE params dict                 (line 1893-1925)
Step 5: Conditional temperature inclusion                          (line 1899)
Step 6: Add xhigh to effort mapping                                (line 1980-1982)
Step 7: Add thinking.display block                                 (after line 2027)
Step 8: Conditional temperature forcing in thinking block           (line 2029-2030)
Step 9: Add output_config block                                    (after line 2053)
Step 10: Append Phase 2 tests to tests/test_opus_47.py
Step 11: Run full test suite — zero failures expected
```

### Phase 3 implementation sequence:

```
Step 1: Add supports_task_budget, tokenizer_scale to ModelCapabilities
Step 2: Wire in _get_capabilities
Step 3: Add BETA_HEADER_TASK_BUDGETS constant
Step 4: Add task_budget block in request building
Step 5: Add has_task_budget param to _build_request_beta_headers
Step 6: Add task budget tests
Step 7: Add deprecation warning in _get_capabilities (optional)
```

---

## 15. What NOT to Do

1. **Don't split the file.** The single-file architecture is deliberate. Changes are
   ~5 new dataclass fields, ~30 lines in the request builder, ~5 lines in the beta
   header method.

2. **Don't add a model registry dict.** The `_get_capabilities` version-detection pattern
   is simple, forward-compatible, and the existing convention. Don't replace it with a
   `MODEL_REGISTRY = {"claude-opus-4-7": {...}}` dict.

3. **Don't abstract thinking config into a separate builder class.** The thinking logic
   is read-once, linear, and specific to this provider. Extracting it would create
   indirection without reuse benefit.

4. **Don't change the `complete()` public API.** All new features wire through existing
   surfaces: `reasoning_effort` on `ChatRequest`, `kwargs` for per-request overrides,
   `config` dict for session-level settings.

5. **Don't add `top_p` or `top_k` to the params dict.** They were never sent before.
   The `supports_sampling` field gates temperature only (which is already sent).

6. **Don't send `output_config` to older models.** The field didn't exist before 4.7.
   Always gate on `supports_output_config`.

7. **Don't remove the temperature=1.0 forcing for older models.** Anthropic still requires
   it when thinking is enabled on pre-4.7 models. Only skip it when `supports_sampling`
   is False.

---

## 16. Success Criteria

### Phase 1 Complete When:

- [ ] `AnthropicProvider._get_capabilities("claude-opus-4-7-20260416").supports_manual_thinking` is `False`
- [ ] `AnthropicProvider._get_capabilities("claude-opus-4-6-20260101").supports_manual_thinking` is `True`
- [ ] `AnthropicProvider._get_capabilities("claude-opus-latest").supports_manual_thinking` is `False`
- [ ] `reasoning_effort="low"` on Opus 4.7 → `params["thinking"]["type"] == "adaptive"`
- [ ] `reasoning_effort="low"` on Opus 4.7 → `"budget_tokens" not in params["thinking"]`
- [ ] `reasoning_effort="low"` on Opus 4.6 → `params["thinking"]["type"] == "enabled"` (unchanged)
- [ ] `_should_add_context_1m_beta("claude-opus-4-7-...", ...)` returns `True`
- [ ] `_should_add_context_1m_beta("claude-opus-latest", ...)` returns `True`
- [ ] SDK resolves to >=0.96.0 after `uv lock`
- [ ] All 205 lines of existing `test_model_capabilities.py` pass
- [ ] All 451 lines of existing `test_reasoning_effort.py` pass
- [ ] All other existing test files pass
- [ ] New `test_opus_47.py` Phase 1 tests pass

### Phase 2 Complete When:

- [ ] Opus 4.7 + `reasoning_effort="high"` → `params["output_config"] == {"effort": "high"}`
- [ ] Opus 4.7 + `reasoning_effort="xhigh"` → `params["output_config"] == {"effort": "xhigh"}`
- [ ] Opus 4.7 + no effort → `"output_config" not in params`
- [ ] Opus 4.6 + `reasoning_effort="high"` → `"output_config" not in params`
- [ ] Opus 4.7 + thinking → `params["thinking"]["display"] == "summarized"`
- [ ] Opus 4.7 + thinking + `config["thinking_display"] = "omitted"` → `display == "omitted"`
- [ ] Opus 4.6 + thinking → `"display" not in params["thinking"]`
- [ ] Opus 4.7 (no thinking) → `"temperature" not in params`
- [ ] Opus 4.7 + thinking → `"temperature" not in params`
- [ ] Opus 4.6 (no thinking) → `"temperature" in params`
- [ ] Opus 4.6 + thinking → `params["temperature"] == 1.0`
- [ ] All existing tests still pass (zero regressions)
- [ ] New `test_opus_47.py` Phase 2 tests pass

### Phase 3 Complete When:

- [ ] `task_budget_tokens=50000` in kwargs → `params["output_config"]["task_budget"]` present
- [ ] Task budget → `BETA_HEADER_TASK_BUDGETS` in request beta headers
- [ ] Task budget minimum enforced at 20000
- [ ] No task budget → no task budget beta header
- [ ] All existing tests still pass
