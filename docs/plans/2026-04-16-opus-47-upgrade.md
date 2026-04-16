# Opus 4.7 Provider Upgrade — Implementation Plan

> **Execution:** Use the subagent-driven-development workflow to implement this plan.

**Goal:** Upgrade `amplifier-module-provider-anthropic` to fully support Claude Opus 4.7, fixing P0 breakages (Phase 1) and adding full feature support (Phase 2).

**Architecture:** All changes are surgical edits to the existing single-file provider (`amplifier_module_provider_anthropic/__init__.py`). New capability fields on the `ModelCapabilities` dataclass gate all behavioral differences — no scattered model-name checks. A single new test file `tests/test_opus_47.py` covers both phases.

**Tech Stack:** Python 3.11+, Anthropic SDK >=0.96.0, pytest, uv

**Design Spec:** `OPUS_47_UPGRADE_SPEC.md` (in repo root — the authoritative source for all before/after code)

---

## Phase 1: P0 — Don't Break Existing Users

Phase 1 prevents HTTP 400 errors when someone sets `model: claude-opus-4-7`. Minimal blast radius — zero changes to existing model behavior.

---

### Task 1: Write Phase 1 Capability Tests

**Files:**
- Create: `tests/test_opus_47.py`

**Step 1: Create the test file with Phase 1 capability tests**

Create `tests/test_opus_47.py` with the following content. This tests that `_get_capabilities()` returns correct values for Opus 4.7 vs older models.

```python
"""Tests for Claude Opus 4.7 support.

Phase 1: Validates capability detection, manual-thinking fallback,
and 1M beta header fix.

Phase 2: Validates output_config, thinking.display, temperature
stripping, and xhigh effort.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_reasoning_effort.py)
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


class DummyResponse:
    """Minimal Anthropic API response stub."""

    def __init__(self):
        self.content = [SimpleNamespace(type="text", text="ok")]
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.stop_reason = "end_turn"
        self.model = "claude-opus-4-7-20260416"


def _make_provider(
    default_model: str = "claude-opus-4-7-20260416",
) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": 0,
            "default_model": default_model,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _make_raw_mock() -> MagicMock:
    raw = MagicMock()
    raw.parse.return_value = DummyResponse()
    raw.headers = {}
    return raw


def _get_api_params(mock_create: AsyncMock) -> dict[str, Any]:
    """Extract the kwargs passed to the API call."""
    assert mock_create.await_count == 1
    _, kwargs = mock_create.call_args
    return kwargs


# ---------------------------------------------------------------------------
# Phase 1: Capability detection
# ---------------------------------------------------------------------------


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
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestOpus47Capabilities -v 2>&1 | head -40
```

Expected: FAIL — `ModelCapabilities` has no `supports_manual_thinking` attribute.

**Step 3: Commit test file (tests-first checkpoint)**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && git add tests/test_opus_47.py && git commit -m "test: add Phase 1 Opus 4.7 capability tests (red)"
```

---

### Task 2: Add `supports_manual_thinking` to ModelCapabilities

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (line 218, between `supports_adaptive_thinking` and `default_thinking_budget`)

**Step 1: Add the field**

In `amplifier_module_provider_anthropic/__init__.py`, find the `ModelCapabilities` dataclass (line 204-220). Add the new field after `supports_adaptive_thinking` (line 218) and before `default_thinking_budget` (line 219):

```python
# FIND this line:
    supports_adaptive_thinking: bool = False
    default_thinking_budget: int = 0

# REPLACE with:
    supports_adaptive_thinking: bool = False
    supports_manual_thinking: bool = True      # P1: False on Opus 4.7+ (type="enabled" → 400)
    default_thinking_budget: int = 0
```

**Step 2: Run capability tests**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestOpus47Capabilities -v 2>&1 | head -40
```

Expected: Most tests PASS, but `test_opus_47_supports_manual_thinking_false` and `test_opus_unknown_assumes_no_manual_thinking` FAIL (the Opus block doesn't set the field yet — it defaults to `True`).

---

### Task 3: Wire `is_47_plus` in `_get_capabilities()` Opus Block

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (line 825-841)

**Step 1: Add `is_47_plus` and `supports_manual_thinking` to the Opus block**

Find the Opus block in `_get_capabilities()` (line 825-841). Replace it:

```python
# FIND (exact match, lines 825-841):
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

# REPLACE with:
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

**Step 2: Run capability tests**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestOpus47Capabilities -v
```

Expected: ALL PASS.

---

### Task 4: Pass Through `supports_manual_thinking` in `_apply_runtime_capability_overrides`

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (line 960-969)

This is the **COE-flagged critical fix**. Without this, the runtime override method reconstructs `ModelCapabilities` without the new field, silently resetting it to the default `True` — which breaks production for Opus 4.7 when the Models API is reachable.

**Step 1: Write the test for pass-through (COE issue #2)**

Append this test class to `tests/test_opus_47.py`, after `TestOpus47Capabilities`:

```python
class TestRuntimeCapabilityOverridePassthrough:
    """COE issue #2: _apply_runtime_capability_overrides must preserve
    supports_manual_thinking from base_caps, not reset to default True."""

    def test_supports_manual_thinking_preserved_through_override(self):
        """If base_caps has supports_manual_thinking=False, the runtime
        override must NOT reset it to True (the default)."""
        from amplifier_module_provider_anthropic import AnthropicProvider
        from amplifier_module_provider_anthropic import _RuntimeModelInfo

        base_caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert base_caps.supports_manual_thinking is False  # precondition

        # Simulate a runtime override with basic model info
        runtime_info = _RuntimeModelInfo(
            max_input_tokens=200000,
            max_tokens=128000,
            supports_thinking=True,
        )

        result = AnthropicProvider._apply_runtime_capability_overrides(
            base_caps, runtime_info
        )
        assert result.supports_manual_thinking is False, (
            "Runtime override must pass through supports_manual_thinking, "
            "not reset to default True"
        )
```

**Step 2: Run test to verify it fails**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestRuntimeCapabilityOverridePassthrough -v
```

Expected: FAIL — `result.supports_manual_thinking` is `True` (default) because the override method doesn't pass through the field.

**Step 3: Add pass-through in _apply_runtime_capability_overrides**

Find the `return ModelCapabilities(...)` call at line 960-969. Add the pass-through line:

```python
# FIND (exact match, lines 960-969):
        return ModelCapabilities(
            family=base_caps.family,
            max_output_tokens=runtime_info.max_tokens or base_caps.max_output_tokens,
            base_context_window=base_context_window,
            supports_1m=supports_1m,
            supports_thinking=supports_thinking,
            supports_adaptive_thinking=supports_adaptive_thinking,
            default_thinking_budget=default_thinking_budget,
            capability_tags=tuple(capability_tags),
        )

# REPLACE with:
        return ModelCapabilities(
            family=base_caps.family,
            max_output_tokens=runtime_info.max_tokens or base_caps.max_output_tokens,
            base_context_window=base_context_window,
            supports_1m=supports_1m,
            supports_thinking=supports_thinking,
            supports_adaptive_thinking=supports_adaptive_thinking,
            supports_manual_thinking=base_caps.supports_manual_thinking,
            default_thinking_budget=default_thinking_budget,
            capability_tags=tuple(capability_tags),
        )
```

**Step 4: Run test to verify it passes**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestRuntimeCapabilityOverridePassthrough -v
```

Expected: PASS.

---

### Task 5: Write Thinking Fallback Tests + COE-Fixed Test

**Files:**
- Modify: `tests/test_opus_47.py` (append)

**Step 1: Append thinking fallback tests**

Append these test classes to `tests/test_opus_47.py`. Note: the `test_opus_47_extended_thinking_kwarg_forces_adaptive` test is the **COE issue #3 fix**, and `test_opus_47_config_thinking_type_enabled_forces_adaptive` is rewritten per **COE issue #1** — using `extended_thinking=True` kwarg WITHOUT `reasoning_effort` to actually exercise the new elif branch.

```python
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

    def test_opus_47_extended_thinking_kwarg_forces_adaptive(self):
        """COE issue #3: extended_thinking=True kwarg (no reasoning_effort)
        on Opus 4.7 must produce adaptive, not enabled.
        This tests the old-style kwarg interface on a 4.7 model."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request, extended_thinking=True))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in params["thinking"]

    def test_opus_47_config_thinking_type_enabled_forces_adaptive(self):
        """COE issue #1 REWRITE: Even if config says thinking_type='enabled',
        4.7 forces adaptive. Uses extended_thinking=True kwarg WITHOUT
        reasoning_effort so the new elif branch is actually exercised.
        (The original spec test used reasoning_effort='high' which
        overrides config before the branch is reached — a false positive.)"""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.config["thinking_type"] = "enabled"
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        # No reasoning_effort — thinking_type comes from config ("enabled")
        asyncio.run(provider.complete(request, extended_thinking=True))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in params["thinking"]

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
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestOpus47ThinkingFallback -v 2>&1 | head -50
```

Expected: `test_opus_47_low_effort_forces_adaptive`, `test_opus_47_extended_thinking_kwarg_forces_adaptive`, and `test_opus_47_config_thinking_type_enabled_forces_adaptive` FAIL — the elif branch doesn't exist yet, so `type="enabled"` is set with `budget_tokens`.

---

### Task 6: Add `elif not supports_manual_thinking` Branch

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (line 2015-2027)

**Step 1: Add the new elif branch**

Find the thinking resolution block (line 2015-2027). Replace:

```python
# FIND (exact match, lines 2011-2027):
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

# REPLACE with:
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

**Step 2: Run thinking fallback tests**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestOpus47ThinkingFallback -v
```

Expected: ALL PASS.

---

### Task 7: Write and Fix 1M Beta Header Tests

**Files:**
- Modify: `tests/test_opus_47.py` (append)
- Modify: `amplifier_module_provider_anthropic/__init__.py` (line 1019-1021)

**Step 1: Append beta header tests to `tests/test_opus_47.py`**

```python
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

**Step 2: Run tests to verify failures**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestBetaHeader1MFix -v
```

Expected: `test_opus_47_gets_1m_header`, `test_opus_unknown_gets_1m_header`, and `test_sonnet_unknown_gets_1m_header` FAIL.

**Step 3: Fix `_should_add_context_1m_beta`**

Find the method at line 1002-1021. Replace the version checks:

```python
# FIND (exact match, lines 1016-1021):
        # Anthropic's current context-window docs list 1M as a beta-header feature
        # for Opus 4.6 and Sonnet 4.x. Keep the header scoped to those families so
        # lower-tier fallbacks like Haiku don't inherit it.
        if family == "opus":
            return version == (4, 6)
        return family == "sonnet" and version in {(4, 0), (4, 5), (4, 6)}

# REPLACE with:
        # Send the 1M beta header for known 1M-capable versions and unknown
        # versions (forward-compat).  The header is harmless on models where
        # 1M context is already GA (e.g. Opus 4.7+), but omitting it breaks
        # models that still need it.
        if family == "opus":
            return version == (0, 0) or version >= (4, 6)
        return family == "sonnet" and (version == (0, 0) or version >= (4, 0))
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestBetaHeader1MFix -v
```

Expected: ALL PASS.

---

### Task 8: Bump SDK and Verify Import

**Files:**
- Modify: `pyproject.toml` (line 12)
- Regenerate: `uv.lock`
- Possibly modify: `amplifier_module_provider_anthropic/__init__.py` (line 55-57)

**Step 1: Bump SDK version in pyproject.toml**

```python
# FIND (in pyproject.toml):
    "anthropic>=0.25.0",

# REPLACE with:
    "anthropic>=0.96.0",
```

**Step 2: Run `uv lock` to regenerate lockfile**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv lock
```

Expected: Resolves successfully with anthropic >= 0.96.0.

**Step 3: Install updated dependencies**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv sync
```

**Step 4: Check if OverloadedError has been promoted to public API**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -c "from anthropic import OverloadedError; print('PUBLIC')" 2>/dev/null || echo "STILL_PRIVATE"
```

**Step 5: If PUBLIC — update the import**

If step 4 printed `PUBLIC`, edit `amplifier_module_provider_anthropic/__init__.py` (line 55-57):

```python
# FIND:
from anthropic._exceptions import (
    OverloadedError as AnthropicOverloadedError,
)  # Not exported in public API as of SDK v0.75.0

# REPLACE with:
from anthropic import OverloadedError as AnthropicOverloadedError
```

If step 4 printed `STILL_PRIVATE`, leave the import unchanged.

**Step 6: Run full test suite to verify no SDK breakage**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: ALL tests pass (existing + new Phase 1 tests).

---

### Task 9: Phase 1 Full Verification and Commit

**Files:** None (verification only)

**Step 1: Run ALL tests**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/ -v 2>&1 | tail -40
```

Expected: ALL PASS, zero failures, zero errors.

**Step 2: Commit Phase 1**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && git add -A && git commit -m "feat: Phase 1 Opus 4.7 support — manual thinking fallback, 1M header fix, SDK bump

- Add supports_manual_thinking field to ModelCapabilities (default True)
- Wire is_47_plus in _get_capabilities() Opus block
- Pass through supports_manual_thinking in _apply_runtime_capability_overrides
- Add elif branch: force adaptive when model rejects type=enabled
- Fix _should_add_context_1m_beta from == to >= for forward-compat
- Bump SDK from >=0.25.0 to >=0.96.0
- New test file tests/test_opus_47.py with capability, fallback, and header tests
- Includes COE-mandated tests for runtime override pass-through and extended_thinking kwarg"
```

---

## Phase 2: P1 — Full Opus 4.7 Support

Phase 2 adds proper effort control (`output_config`), visible thinking (`display`), temperature stripping, and `xhigh` effort level.

---

### Task 10: Write Phase 2 Capability Tests

**Files:**
- Modify: `tests/test_opus_47.py` (append)

**Step 1: Append Phase 2 capability tests**

Append to `tests/test_opus_47.py`:

```python
# ---------------------------------------------------------------------------
# Phase 2: Output config, temperature, thinking display, xhigh effort
# ---------------------------------------------------------------------------


class TestOpus47Phase2Capabilities:
    """Phase 2 capability fields for Opus 4.7."""

    def test_opus_47_supports_output_config(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_output_config is True

    def test_opus_46_no_output_config(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.supports_output_config is False

    def test_opus_47_supports_sampling_false(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_sampling is False

    def test_opus_46_supports_sampling_true(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.supports_sampling is True

    def test_sonnet_supports_sampling_true(self):
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-6-20260101")
        assert caps.supports_sampling is True

    def test_opus_47_thinking_display_required(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.thinking_display_required is True

    def test_opus_46_thinking_display_not_required(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.thinking_display_required is False

    def test_opus_47_supported_efforts_includes_xhigh(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supported_efforts == ("low", "medium", "high", "xhigh")

    def test_opus_46_no_xhigh(self):
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert "xhigh" not in caps.supported_efforts
```

**Step 2: Run to verify failures**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/test_opus_47.py::TestOpus47Phase2Capabilities -v 2>&1 | head -30
```

Expected: FAIL — `supports_output_config`, `supports_sampling`, etc. don't exist yet.

---

### Task 11: Add Phase 2 Fields to ModelCapabilities + Wire in Opus Block

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (ModelCapabilities dataclass + Opus block + override method)

**Step 1: Add 4 new fields to ModelCapabilities**

Find `supports_manual_thinking` (the field we added in Task 2). Add the new fields after it:

```python
# FIND:
    supports_manual_thinking: bool = True      # P1: False on Opus 4.7+ (type="enabled" → 400)
    default_thinking_budget: int = 0

# REPLACE with:
    supports_manual_thinking: bool = True      # P1: False on Opus 4.7+ (type="enabled" → 400)
    supports_output_config: bool = False       # P2: output_config.effort GA
    supports_sampling: bool = True             # P2: False = temperature silently ignored
    thinking_display_required: bool = False    # P2: must send display:"summarized" to see thinking
    supported_efforts: tuple[str, ...] = ("low", "medium", "high")  # P2: valid effort levels
    default_thinking_budget: int = 0
```

**Step 2: Wire all fields in the Opus block of `_get_capabilities()`**

Replace the Opus block (which now has `is_47_plus` from Task 3):

```python
# FIND the current Opus block (from Task 3):
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

# REPLACE with:
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

**Step 3: Pass through all Phase 2 fields in `_apply_runtime_capability_overrides`**

Update the `return ModelCapabilities(...)` call (which now has `supports_manual_thinking` from Task 4):

```python
# FIND (current after Task 4):
        return ModelCapabilities(
            family=base_caps.family,
            max_output_tokens=runtime_info.max_tokens or base_caps.max_output_tokens,
            base_context_window=base_context_window,
            supports_1m=supports_1m,
            supports_thinking=supports_thinking,
            supports_adaptive_thinking=supports_adaptive_thinking,
            supports_manual_thinking=base_caps.supports_manual_thinking,
            default_thinking_budget=default_thinking_budget,
            capability_tags=tuple(capability_tags),
        )

# REPLACE with:
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

**Step 4: Run Phase 2 capability tests**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/test_opus_47.py::TestOpus47Phase2Capabilities -v
```

Expected: ALL PASS.

**Step 5: Run existing test suite to verify no regressions**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: ALL PASS.

---

### Task 12: Write Temperature + Output Config Tests

**Files:**
- Modify: `tests/test_opus_47.py` (append)

**Step 1: Append temperature and output_config tests**

Append to `tests/test_opus_47.py`:

```python
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
```

**Step 2: Run to verify failures**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/test_opus_47.py::TestOpus47Temperature tests/test_opus_47.py::TestOpus47OutputConfig -v 2>&1 | head -40
```

Expected: Temperature tests and output_config tests FAIL — temperature is always included and output_config doesn't exist in the request builder.

---

### Task 13: Move `request_caps` Fetch + Conditional Temperature

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (line 1893-1926)

This is the biggest single structural change. Move `request_caps` fetch before the params dict, gate temperature on `supports_sampling`.

**Step 1: Restructure the params block**

Find the block from line 1893 to line 1926. Replace:

```python
# FIND (exact match, lines 1893-1926):
        # Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "messages": all_messages,
            "max_tokens": request.max_output_tokens
            or kwargs.get("max_tokens", self.max_tokens),
            "temperature": request.temperature
            or kwargs.get("temperature", self.temperature),
        }

        if system_blocks:
            params["system"] = system_blocks

        # Add tools if provided
        if request.tools:
            tools = self._convert_tools_from_request(request.tools)
            params["tools"] = self._apply_tool_cache_control(tools)
            # Add tool_choice if specified
            if tool_choice := kwargs.get("tool_choice"):
                params["tool_choice"] = tool_choice

        # Add native web search tool if enabled (via config or kwargs)
        # This is a model-native tool that doesn't need function conversion
        web_search_enabled = kwargs.get("enable_web_search", self.enable_web_search)
        if web_search_enabled:
            web_search_tool = self._build_web_search_tool(kwargs)
            if "tools" not in params:
                params["tools"] = []
            # Add web search tool at the beginning (native tools typically come first)
            params["tools"].insert(0, web_search_tool)
            logger.info("[PROVIDER] Native web search tool enabled")

        request_caps = await self._get_request_capabilities(params["model"])
        model_ceiling = request_caps.max_output_tokens

# REPLACE with:
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

        if system_blocks:
            params["system"] = system_blocks

        # Add tools if provided
        if request.tools:
            tools = self._convert_tools_from_request(request.tools)
            params["tools"] = self._apply_tool_cache_control(tools)
            # Add tool_choice if specified
            if tool_choice := kwargs.get("tool_choice"):
                params["tool_choice"] = tool_choice

        # Add native web search tool if enabled (via config or kwargs)
        # This is a model-native tool that doesn't need function conversion
        web_search_enabled = kwargs.get("enable_web_search", self.enable_web_search)
        if web_search_enabled:
            web_search_tool = self._build_web_search_tool(kwargs)
            if "tools" not in params:
                params["tools"] = []
            # Add web search tool at the beginning (native tools typically come first)
            params["tools"].insert(0, web_search_tool)
            logger.info("[PROVIDER] Native web search tool enabled")
```

**Step 2: Conditional temperature forcing in thinking block**

Find the temperature forcing line inside the thinking block (was line 2029-2030, now shifted):

```python
# FIND:
            # CRITICAL: Anthropic requires temperature=1.0 when thinking is enabled
            params["temperature"] = 1.0

# REPLACE with:
            # Anthropic requires temperature=1.0 when thinking is enabled
            # on models that support sampling. Non-sampling models (4.7+)
            # ignore temperature entirely — don't inject it.
            if request_caps.supports_sampling:
                params["temperature"] = 1.0
```

**Step 3: Run temperature tests**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/test_opus_47.py::TestOpus47Temperature -v
```

Expected: ALL PASS.

**Step 4: Run existing tests to check for regressions**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/test_reasoning_effort.py -v 2>&1 | tail -20
```

Expected: ALL PASS (existing Sonnet/Opus 4.6 tests unaffected — they have `supports_sampling=True`).

---

### Task 14: Add `xhigh` Effort Mapping + `thinking.display` Block

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (effort mapping + thinking display)

**Step 1: Write thinking display tests**

Append to `tests/test_opus_47.py`:

```python
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
```

**Step 2: Run to verify failures**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/test_opus_47.py::TestOpus47ThinkingDisplay -v 2>&1 | head -20
```

Expected: FAIL — `display` is not in the thinking dict.

**Step 3: Add `xhigh` to effort mapping**

Find the effort mapping block (was line 1972-1982, now shifted due to Task 13 changes). Look for the `elif reasoning_effort == "high":` block and add `xhigh` after it:

```python
# FIND:
            elif reasoning_effort == "high":
                effort_thinking_type = "adaptive"
                effort_budget = request_caps.default_thinking_budget

# REPLACE with:
            elif reasoning_effort == "high":
                effort_thinking_type = "adaptive"
                effort_budget = request_caps.default_thinking_budget
            elif reasoning_effort == "xhigh":
                effort_thinking_type = "adaptive"
                effort_budget = request_caps.default_thinking_budget
```

**Step 4: Add `thinking.display` block**

Insert AFTER all three thinking branches (the `if/elif/else` that sets `params["thinking"]`) and BEFORE the temperature forcing line. The exact insertion point is right after the last branch that sets `params["thinking"]`:

```python
# INSERT AFTER the closing of the else block that sets params["thinking"]:
# (After the line: params["thinking"] = {"type": thinking_type, "budget_tokens": budget_tokens, })

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

**Step 5: Run thinking display tests**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/test_opus_47.py::TestOpus47ThinkingDisplay -v
```

Expected: ALL PASS.

---

### Task 15: Add `output_config` Block

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (after the thinking/max_tokens block, before stop_sequences)

**Step 1: Add output_config block**

Insert AFTER the `params["max_tokens"] = model_ceiling` clamping block (was around line 2055-2062) and BEFORE `# Add stop_sequences if specified`:

```python
# INSERT BEFORE "# Add stop_sequences if specified":

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

        # Allow explicit effort override via kwargs (e.g. from orchestrator)
        if request_caps.supports_output_config and "effort" in kwargs:
            effort = kwargs["effort"]
            if effort in request_caps.supported_efforts:
                if "output_config" not in params:
                    params["output_config"] = {}
                params["output_config"]["effort"] = effort
```

**Step 2: Run output_config tests**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/test_opus_47.py::TestOpus47OutputConfig -v
```

Expected: ALL PASS.

---

### Task 16: Phase 2 Full Verification and Commit

**Files:** None (verification only)

**Step 1: Run ALL tests (entire test suite)**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/ -v 2>&1 | tail -50
```

Expected: ALL PASS, zero failures, zero errors.

**Step 2: Run a specific count check**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/test_opus_47.py -v --tb=short 2>&1 | tail -60
```

Expected: All Phase 1 + Phase 2 tests pass. Expect roughly 50+ tests in the file.

**Step 3: Verify no regressions in existing test files**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && uv run python -m pytest tests/ --ignore=tests/test_opus_47.py -v --tb=short 2>&1 | tail -20
```

Expected: ALL existing tests pass unchanged.

**Step 4: Commit Phase 2**

```bash
cd /home/bkrabach/dev/opus-4-7-support/amplifier-module-provider-anthropic && git add -A && git commit -m "feat: Phase 2 Opus 4.7 full support — output_config, thinking.display, temperature strip, xhigh

- Add supports_output_config, supports_sampling, thinking_display_required,
  supported_efforts fields to ModelCapabilities
- Wire all Phase 2 fields in Opus block of _get_capabilities()
- Pass through all in _apply_runtime_capability_overrides
- Move request_caps fetch before params dict for per-model gating
- Conditional temperature: omit for non-sampling models (Opus 4.7+)
- Add xhigh effort level mapping
- Add thinking.display='summarized' for models that default to 'omitted'
- Conditional temperature forcing: skip for non-sampling models in thinking block
- Add output_config.effort block gated on supports_output_config
- Phase 2 tests appended to tests/test_opus_47.py"
```

---

## Post-Implementation: STOP Before Push

**DO NOT push to remote.** The user explicitly requires container-based smoke testing before push.

**Step 1: Summarize what was done**

Report:
- Number of lines changed in `__init__.py`
- Number of new tests in `test_opus_47.py`
- Full test count and pass rate
- The two commit SHAs

**Step 2: Wait for user**

The user will test in a container (amplifier-core smoke test pattern) before approving push.
