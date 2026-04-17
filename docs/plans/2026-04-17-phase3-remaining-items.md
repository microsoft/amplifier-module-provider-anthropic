# Phase 3: Task Budgets, Deprecation Warnings, Tokenizer Buffer, Temp Bug Fix

> **Execution:** Use the subagent-driven-development workflow to implement this plan.

**Goal:** Address all 5 remaining items from the Opus 4.7 design spec, COE adversarial review, and code quality review in a single PR.

**Architecture:** Five changes to `__init__.py` (comment cleanup, temperature falsy bug fix, buffer bump, deprecation warnings, task budgets beta) with corresponding test additions in `tests/test_opus_47.py`. Changes are ordered by dependency: cosmetic fixes first, then feature additions of increasing complexity.

**Tech Stack:** Python, pytest, `anthropic` SDK ≥0.96.0, `amplifier-core` (`ChatRequest`, `ConfigField`, etc.)

**Baseline:** 389 tests passing (49 Opus 4.7 + 340 existing). All existing tests must continue to pass after every task.

---

## File Reference

| File | Role |
|------|------|
| `amplifier_module_provider_anthropic/__init__.py` | Production code — all 5 changes here |
| `tests/test_opus_47.py` | Test additions — all new tests appended here |

## Test Infrastructure

Tests in `test_opus_47.py` use these helpers (already defined at top of file, lines 22–71):

```python
class FakeHooks:          # Mock event emitter
class FakeCoordinator:    # Wraps FakeHooks
class DummyResponse:      # Minimal API response stub (model="claude-opus-4-7-20260416")

def _make_provider(default_model="claude-sonnet-4-5-20250929") -> AnthropicProvider
def _make_raw_mock() -> MagicMock        # Returns raw response with DummyResponse inside
def _get_api_params(mock_create) -> dict  # Extracts kwargs from the mocked API call
```

All tests follow this pattern:
1. Create provider with `_make_provider(default_model="...")`
2. Mock `provider.client.messages.with_raw_response.create = AsyncMock(return_value=_make_raw_mock())`
3. Build `ChatRequest` with desired params
4. Call `asyncio.run(provider.complete(request, **kwargs))`
5. Extract params with `_get_api_params(...)` and assert

---

### Task 1: B3 — Comment Prefix Cleanup

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (lines 219–231)

No tests — pure cosmetic change with zero behavior impact.

**Step 1: Replace P1/P2 shorthand comments with descriptive text**

In `amplifier_module_provider_anthropic/__init__.py`, replace the `ModelCapabilities` field comments. Find this block (lines 219–231):

```python
    supports_manual_thinking: bool = (
        True  # P1: False on Opus 4.7+ (type="enabled" → 400)
    )
    supports_output_config: bool = False  # P2: output_config.effort GA
    supports_sampling: bool = True  # P2: False = temperature silently ignored
    thinking_display_required: bool = (
        False  # P2: must send display:"summarized" to see thinking
    )
    supported_efforts: tuple[str, ...] = (
        "low",
        "medium",
        "high",
    )  # P2: valid effort levels
```

Replace with:

```python
    supports_manual_thinking: bool = (
        True  # False on Opus 4.7+ (type="enabled" returns HTTP 400)
    )
    supports_output_config: bool = False  # True = model accepts output_config.effort
    supports_sampling: bool = True  # False = temperature silently ignored by model
    thinking_display_required: bool = (
        False  # True = must send thinking.display to see thinking content
    )
    supported_efforts: tuple[str, ...] = (
        "low",
        "medium",
        "high",
    )  # Valid effort levels for output_config and reasoning_effort
```

**Step 2: Run all tests to verify no regressions**

```bash
cd amplifier-module-provider-anthropic && python -m pytest tests/ -q
```

Expected: `389 passed`

**Step 3: Commit**

```bash
git add amplifier_module_provider_anthropic/__init__.py
git commit -m "chore: replace P1/P2 comment shorthand with descriptive text"
```

---

### Task 2: B4 — Fix temperature=0.0 Falsy Bug

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (line 1948)
- Modify: `tests/test_opus_47.py` (append new test class)

**Step 1: Write the failing test**

Append this test class to the **end** of `tests/test_opus_47.py`:

```python
# ---------------------------------------------------------------------------
# TestTemperatureZeroBug — temperature=0.0 must not be treated as falsy
# ---------------------------------------------------------------------------


class TestTemperatureZeroBug:
    """temperature=0.0 must be respected, not treated as falsy."""

    def test_temperature_zero_is_respected(self):
        """request.temperature=0.0 should send 0.0, not fall back to default 0.7."""
        provider = _make_provider(default_model="claude-sonnet-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            temperature=0.0,
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["temperature"] == 0.0

    def test_temperature_none_falls_back_to_default(self):
        """request.temperature=None should fall back to provider default (0.7)."""
        provider = _make_provider(default_model="claude-sonnet-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["temperature"] == 0.7

    def test_temperature_explicit_value_sent(self):
        """request.temperature=0.5 should send 0.5."""
        provider = _make_provider(default_model="claude-sonnet-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            temperature=0.5,
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["temperature"] == 0.5
```

**Step 2: Run the new tests to verify they fail**

```bash
cd amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestTemperatureZeroBug -v
```

Expected: `test_temperature_zero_is_respected` FAILS (temperature is 0.7 instead of 0.0). The other two may pass since they exercise the non-buggy path.

**Step 3: Fix the bug in `__init__.py`**

In `amplifier_module_provider_anthropic/__init__.py`, find this block (line 1948):

```python
            params["temperature"] = request.temperature or kwargs.get(
                "temperature", self.temperature
            )
```

Replace with:

```python
            params["temperature"] = (
                request.temperature
                if request.temperature is not None
                else kwargs.get("temperature", self.temperature)
            )
```

**Step 4: Run all tests to verify pass**

```bash
cd amplifier-module-provider-anthropic && python -m pytest tests/ -q
```

Expected: `392 passed` (389 + 3 new)

**Step 5: Commit**

```bash
git add amplifier_module_provider_anthropic/__init__.py tests/test_opus_47.py
git commit -m "fix: temperature=0.0 no longer treated as falsy"
```

---

### Task 3: A2 — Tokenizer Buffer Bump

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (lines 2054–2056)
- Modify: `tests/test_opus_47.py` (append new test class)

Per COE decision: NO new config field. Just bump the default buffer from 4096 to 8192 and add an explanatory comment.

**Step 1: Write the failing test**

Append this test class to the **end** of `tests/test_opus_47.py`:

```python
# ---------------------------------------------------------------------------
# TestTokenizerBufferBump — default thinking buffer increased for Opus 4.7
# ---------------------------------------------------------------------------


class TestTokenizerBufferBump:
    """Default thinking_budget_buffer bumped from 4096 to 8192."""

    def test_default_buffer_is_8192(self):
        """Default buffer_tokens should be 8192 (not the old 4096)."""
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
        # max_tokens = min(budget + buffer, ceiling) = min(64000 + 8192, 128000) = 72192
        # Old behavior: min(64000 + 4096, 128000) = 68096
        assert params["max_tokens"] >= 72192

    def test_config_buffer_override_still_works(self):
        """Config thinking_budget_buffer overrides the new default."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.config["thinking_budget_buffer"] = 16384
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        # min(64000 + 16384, 128000) = 80384
        assert params["max_tokens"] >= 80384
```

**Step 2: Run new tests to verify they fail**

```bash
cd amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestTokenizerBufferBump -v
```

Expected: `test_default_buffer_is_8192` FAILS (max_tokens is 68096, not ≥72192)

**Step 3: Bump the default buffer**

In `amplifier_module_provider_anthropic/__init__.py`, find this block (line 2054):

```python
            buffer_tokens = kwargs.get("thinking_budget_buffer") or self.config.get(
                "thinking_budget_buffer", 4096
            )
```

Replace with:

```python
            # Default buffer raised from 4096 → 8192 to accommodate Opus 4.7's
            # denser tokenizer (1.0–1.35× more tokens for equivalent text).
            buffer_tokens = kwargs.get("thinking_budget_buffer") or self.config.get(
                "thinking_budget_buffer", 8192
            )
```

**Step 4: Run all tests to verify pass**

```bash
cd amplifier-module-provider-anthropic && python -m pytest tests/ -q
```

Expected: `394 passed` (392 + 2 new)

**Step 5: Commit**

```bash
git add amplifier_module_provider_anthropic/__init__.py tests/test_opus_47.py
git commit -m "feat: bump default thinking buffer from 4096 to 8192 for denser tokenizers"
```

---

### Task 4: A3 — Deprecation Warnings

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (module-level + inside `_complete_chat_request`)
- Modify: `tests/test_opus_47.py` (append new test class)

**Step 1: Write the failing tests**

Append this test class to the **end** of `tests/test_opus_47.py`. First, add `logging` to the imports at the top of the file:

At line 7 (after `import asyncio`), the imports section already has what we need. Add `import logging` if not present. Current imports (lines 7–14):
```python
import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider, _RuntimeModelInfo
```

Change to:
```python
import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
import amplifier_module_provider_anthropic as anthropic_module
from amplifier_module_provider_anthropic import AnthropicProvider, _RuntimeModelInfo
```

Then append this test class to the **end** of `tests/test_opus_47.py`:

```python
# ---------------------------------------------------------------------------
# TestDeprecationWarnings — warn once per process for deprecated models
# ---------------------------------------------------------------------------


class TestDeprecationWarnings:
    """Deprecation warnings for models approaching retirement."""

    def setup_method(self):
        """Clear warned set before each test."""
        anthropic_module._clear_deprecated_model_warnings()

    def test_deprecated_model_emits_warning(self, caplog):
        """Deprecated model emits a logger.warning on first use."""
        provider = _make_provider(default_model="claude-3-haiku-20240307")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))
        assert any("deprecated" in r.message.lower() for r in caplog.records)
        assert any("2026-04-19" in r.message for r in caplog.records)

    def test_warning_only_emitted_once(self, caplog):
        """Second call with same deprecated model does NOT warn again."""
        provider = _make_provider(default_model="claude-3-haiku-20240307")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))
        first_count = sum(
            1 for r in caplog.records if "deprecated" in r.message.lower()
        )

        caplog.clear()
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))
        second_count = sum(
            1 for r in caplog.records if "deprecated" in r.message.lower()
        )
        assert first_count == 1
        assert second_count == 0

    def test_non_deprecated_model_no_warning(self, caplog):
        """Non-deprecated model emits no deprecation warning."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.complete(request))
        assert not any("deprecated" in r.message.lower() for r in caplog.records)

    def test_deprecated_models_table_has_expected_entries(self):
        """Verify the deprecation table contains all known deprecated models."""
        deprecated = anthropic_module._DEPRECATED_MODELS
        assert "claude-3-haiku-20240307" in deprecated
        assert "claude-sonnet-4-20250514" in deprecated
        assert "claude-opus-4-20250514" in deprecated
        assert len(deprecated) == 3

    def test_clear_function_resets_warned_set(self):
        """_clear_deprecated_model_warnings() allows re-warning."""
        provider = _make_provider(default_model="claude-3-haiku-20240307")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
        )
        asyncio.run(provider.complete(request))

        anthropic_module._clear_deprecated_model_warnings()

        # After clearing, the next call should warn again (verified by checking
        # the set is empty — the warn-once test already covers the actual logging)
        assert len(anthropic_module._warned_deprecated_models) == 0
```

**Step 2: Run new tests to verify they fail**

```bash
cd amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestDeprecationWarnings -v
```

Expected: FAIL — `_clear_deprecated_model_warnings`, `_DEPRECATED_MODELS`, `_warned_deprecated_models` don't exist yet.

**Step 3: Add deprecation infrastructure to `__init__.py`**

**3a. Add module-level constants and function.**

In `amplifier_module_provider_anthropic/__init__.py`, find this block (around line 201):

```python
FALLBACK_STATE_VERSION = 1
```

Immediately after that line (before the blank line and `@dataclass(frozen=True)`), insert:

```python

# ---------------------------------------------------------------------------
# Deprecated model retirement dates — warn once per process per model
# ---------------------------------------------------------------------------
_DEPRECATED_MODELS: dict[str, str] = {
    "claude-3-haiku-20240307": "2026-04-19",
    "claude-sonnet-4-20250514": "2026-06-15",
    "claude-opus-4-20250514": "2026-06-15",
}
_warned_deprecated_models: set[str] = set()


def _clear_deprecated_model_warnings() -> None:
    """Clear the warned-models set.

    Internal helper for tests. Follows the same pattern as _clear_fallback_windows().
    """
    _warned_deprecated_models.clear()
```

**3b. Add warning emission inside `_complete_chat_request()`.**

In `amplifier_module_provider_anthropic/__init__.py`, find this block (around line 1933):

```python
        request_caps = await self._get_request_capabilities(effective_model)
        model_ceiling = request_caps.max_output_tokens
```

Immediately after `model_ceiling = request_caps.max_output_tokens`, insert:

```python

        # Emit once-per-process deprecation warning for models nearing retirement
        if effective_model in _DEPRECATED_MODELS and effective_model not in _warned_deprecated_models:
            _warned_deprecated_models.add(effective_model)
            retire_date = _DEPRECATED_MODELS[effective_model]
            logger.warning(
                "[PROVIDER] Model %s is deprecated and will be retired on %s. "
                "Please migrate to a newer model.",
                effective_model,
                retire_date,
            )
```

**Step 4: Run all tests to verify pass**

```bash
cd amplifier-module-provider-anthropic && python -m pytest tests/ -q
```

Expected: `399 passed` (394 + 5 new)

**Step 5: Commit**

```bash
git add amplifier_module_provider_anthropic/__init__.py tests/test_opus_47.py
git commit -m "feat: emit once-per-process deprecation warning for retiring models"
```

---

### Task 5: A1 — Task Budgets (Beta)

**Files:**
- Modify: `amplifier_module_provider_anthropic/__init__.py` (multiple locations)
- Modify: `tests/test_opus_47.py` (append new test class)

This is the most complex task. Changes touch 6 locations in `__init__.py`:
1. Beta header constant (line ~201 area)
2. `ModelCapabilities` dataclass (line ~232 area)
3. `_get_capabilities()` — set `supports_task_budget=True` for Opus 4.7+ (line ~848)
4. `_apply_runtime_capability_overrides()` — pass through (line ~993)
5. `_build_request_beta_headers()` — add `has_task_budget` param (line ~1073)
6. `_complete_chat_request()` — wire task_budget into output_config (line ~2170 area)

**Step 1: Write the failing tests**

Append this test class to the **end** of `tests/test_opus_47.py`:

```python
# ---------------------------------------------------------------------------
# TestTaskBudgets — task budget beta feature for Opus 4.7+
# ---------------------------------------------------------------------------


class TestTaskBudgets:
    """Task budget support (beta) for Opus 4.7+."""

    def test_opus_47_supports_task_budget(self):
        """Opus 4.7 capabilities include supports_task_budget=True."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert caps.supports_task_budget is True

    def test_opus_46_no_task_budget(self):
        """Opus 4.6 does not support task budgets."""
        caps = AnthropicProvider._get_capabilities("claude-opus-4-6-20260101")
        assert caps.supports_task_budget is False

    def test_sonnet_no_task_budget(self):
        """Sonnet does not support task budgets."""
        caps = AnthropicProvider._get_capabilities("claude-sonnet-4-6-20260101")
        assert caps.supports_task_budget is False

    def test_task_budget_in_output_config(self):
        """task_budget_tokens kwarg adds task_budget to output_config."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request, task_budget_tokens=50000))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert "output_config" in params
        assert params["output_config"]["task_budget"] == {
            "type": "tokens",
            "total": 50000,
        }

    def test_task_budget_min_20k_enforced(self):
        """Task budget below 20000 is clamped to 20000."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request, task_budget_tokens=5000))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"]["task_budget"]["total"] == 20000

    def test_task_budget_from_config(self):
        """Config-level task_budget_tokens is used when kwarg not provided."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.config["task_budget_tokens"] = 80000
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        assert params["output_config"]["task_budget"]["total"] == 80000

    def test_task_budget_adds_beta_header(self):
        """When task_budget is present, the beta header must be included."""
        provider = _make_provider(default_model="claude-opus-4-7-20260416")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request, task_budget_tokens=50000))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        beta_header = params.get("extra_headers", {}).get("anthropic-beta", "")
        assert "task-budgets-2026-03-13" in beta_header

    def test_no_task_budget_no_beta_header(self):
        """When task_budget is not set, the task-budgets beta header is absent."""
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
        beta_header = params.get("extra_headers", {}).get("anthropic-beta", "")
        assert "task-budgets-2026-03-13" not in beta_header

    def test_task_budget_ignored_on_unsupported_model(self):
        """task_budget_tokens on Opus 4.6 (unsupported) is silently ignored."""
        provider = _make_provider(default_model="claude-opus-4-6-20260101")
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_mock()
        )
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort="high",
        )
        asyncio.run(provider.complete(request, task_budget_tokens=50000))
        params = _get_api_params(provider.client.messages.with_raw_response.create)
        # output_config should not exist for 4.6 at all
        assert "output_config" not in params

    def test_runtime_overrides_preserve_task_budget(self):
        """_apply_runtime_capability_overrides passes through supports_task_budget."""
        base_caps = AnthropicProvider._get_capabilities("claude-opus-4-7-20260416")
        assert base_caps.supports_task_budget is True
        runtime_info = _RuntimeModelInfo()
        overridden = AnthropicProvider._apply_runtime_capability_overrides(
            base_caps, runtime_info
        )
        assert overridden.supports_task_budget is True
```

**Step 2: Run new tests to verify they fail**

```bash
cd amplifier-module-provider-anthropic && python -m pytest tests/test_opus_47.py::TestTaskBudgets -v
```

Expected: FAIL — `supports_task_budget` attribute doesn't exist on `ModelCapabilities`.

**Step 3: Implement task budgets — 6 production code changes**

**3a. Add beta header constant.**

In `amplifier_module_provider_anthropic/__init__.py`, find (line ~198):

```python
BETA_HEADER_INTERLEAVED_THINKING = "interleaved-thinking-2025-05-14"
```

Add immediately after:

```python
BETA_HEADER_TASK_BUDGETS = "task-budgets-2026-03-13"
```

**3b. Add `supports_task_budget` to `ModelCapabilities`.**

In `amplifier_module_provider_anthropic/__init__.py`, find the `ModelCapabilities` dataclass. After the `supported_efforts` field and its comment (around line 231), find:

```python
    )  # Valid effort levels for output_config and reasoning_effort
    default_thinking_budget: int = 0
```

Insert between the comment closing and `default_thinking_budget`:

```python
    supports_task_budget: bool = False  # True = model accepts output_config.task_budget (beta)
```

So it becomes:

```python
    )  # Valid effort levels for output_config and reasoning_effort
    supports_task_budget: bool = False  # True = model accepts output_config.task_budget (beta)
    default_thinking_budget: int = 0
```

**3c. Set `supports_task_budget=True` for Opus 4.7+ in `_get_capabilities()`.**

In `amplifier_module_provider_anthropic/__init__.py`, find the Opus capabilities block in `_get_capabilities()` (around line 841). Find:

```python
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
```

Add `supports_task_budget=is_47_plus,` after `supports_output_config=is_47_plus,`. The block becomes:

```python
            return ModelCapabilities(
                family="opus",
                max_output_tokens=128000 if is_46_plus else 64000,
                supports_1m=is_46_plus,
                supports_thinking=True,
                supports_adaptive_thinking=is_46_plus,
                supports_manual_thinking=not is_47_plus,
                supports_output_config=is_47_plus,
                supports_task_budget=is_47_plus,
                supports_sampling=not is_47_plus,
                thinking_display_required=is_47_plus,
                supported_efforts=(
```

**3d. Pass through `supports_task_budget` in `_apply_runtime_capability_overrides()`.**

In `amplifier_module_provider_anthropic/__init__.py`, find the `return ModelCapabilities(...)` block in `_apply_runtime_capability_overrides()` (around line 985). Find:

```python
            supports_output_config=base_caps.supports_output_config,
            supports_sampling=base_caps.supports_sampling,
```

Add after `supports_output_config`:

```python
            supports_task_budget=base_caps.supports_task_budget,
```

So it becomes:

```python
            supports_output_config=base_caps.supports_output_config,
            supports_task_budget=base_caps.supports_task_budget,
            supports_sampling=base_caps.supports_sampling,
```

**3e. Add `has_task_budget` parameter to `_build_request_beta_headers()`.**

In `amplifier_module_provider_anthropic/__init__.py`, find (around line 1073):

```python
    def _build_request_beta_headers(
        self,
        *,
        model_id: str,
        request_caps: ModelCapabilities,
        tools_present: bool,
        resolved_thinking_type: str | None,
    ) -> list[str]:
        """Build the anthropic-beta header set for a specific effective model."""
        headers = list(self._beta_headers)
        if self._should_add_context_1m_beta(model_id, request_caps):
            headers.append(BETA_HEADER_1M_CONTEXT)
        if self._should_add_interleaved_beta(
            request_caps=request_caps,
            tools_present=tools_present,
            resolved_thinking_type=resolved_thinking_type,
        ):
            headers.append(BETA_HEADER_INTERLEAVED_THINKING)
        return self._dedupe_headers(headers)
```

Replace with:

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
        """Build the anthropic-beta header set for a specific effective model."""
        headers = list(self._beta_headers)
        if self._should_add_context_1m_beta(model_id, request_caps):
            headers.append(BETA_HEADER_1M_CONTEXT)
        if self._should_add_interleaved_beta(
            request_caps=request_caps,
            tools_present=tools_present,
            resolved_thinking_type=resolved_thinking_type,
        ):
            headers.append(BETA_HEADER_INTERLEAVED_THINKING)
        if has_task_budget:
            headers.append(BETA_HEADER_TASK_BUDGETS)
        return self._dedupe_headers(headers)
```

**3f. Wire task_budget into `_complete_chat_request()`.**

In `amplifier_module_provider_anthropic/__init__.py`, find the output_config block (around line 2147):

```python
        # Build output_config for models that support it (Opus 4.7+).
        # output_config.effort is the primary control surface for thinking
        # intensity on these models, replacing the budget_tokens approach.
        if request_caps.supports_output_config and reasoning_effort is not None:
```

We need to add task budget logic AFTER the output_config block and BEFORE the stop_sequences line. Find this exact block (the end of the output_config section through the start of stop_sequences):

```python
            else:
                logger.warning(
                    "[PROVIDER] Effort level '%s' not supported by %s "
                    "(supported: %s) — omitting output_config.effort",
                    effort,
                    params["model"],
                    request_caps.supported_efforts,
                )

        # Add stop_sequences if specified
```

Insert between the closing of the output_config `if` block and the `# Add stop_sequences` comment:

```python

        # Task budget (beta): output_config.task_budget for Opus 4.7+
        # COE CONSTRAINT: Use `is not None` (not `or`) to avoid falsy-zero bug.
        has_task_budget = False
        if request_caps.supports_task_budget:
            task_budget_tokens = kwargs.get("task_budget_tokens")
            if task_budget_tokens is None:
                task_budget_tokens = self.config.get("task_budget_tokens")
            if task_budget_tokens is not None:
                task_budget_tokens = max(20000, int(task_budget_tokens))
                if "output_config" not in params:
                    params["output_config"] = {}
                params["output_config"]["task_budget"] = {
                    "type": "tokens",
                    "total": task_budget_tokens,
                }
                has_task_budget = True
                logger.info(
                    "[PROVIDER] output_config.task_budget=%d for %s",
                    task_budget_tokens,
                    params["model"],
                )

```

Then update the `_build_request_beta_headers()` call site (around line 2175). Find:

```python
        request_beta_headers = self._build_request_beta_headers(
            model_id=params["model"],
            request_caps=request_caps,
            tools_present=bool(params.get("tools")),
            resolved_thinking_type=resolved_thinking_type,
        )
```

Replace with:

```python
        request_beta_headers = self._build_request_beta_headers(
            model_id=params["model"],
            request_caps=request_caps,
            tools_present=bool(params.get("tools")),
            resolved_thinking_type=resolved_thinking_type,
            has_task_budget=has_task_budget,
        )
```

**Step 4: Run all tests to verify pass**

```bash
cd amplifier-module-provider-anthropic && python -m pytest tests/ -q
```

Expected: `409 passed` (399 + 10 new)

**Step 5: Commit**

```bash
git add amplifier_module_provider_anthropic/__init__.py tests/test_opus_47.py
git commit -m "feat: add task budget beta support for Opus 4.7+"
```

---

## Final Verification

### Run the full test suite

```bash
cd amplifier-module-provider-anthropic && python -m pytest tests/ -v
```

Expected: **409 passed** (389 original + 20 new), 0 failed, 0 errors.

### New test breakdown

| Test Class | Count | Covers |
|------------|-------|--------|
| `TestTemperatureZeroBug` | 3 | B4: temperature=0.0 falsy fix |
| `TestTokenizerBufferBump` | 2 | A2: buffer 4096→8192 |
| `TestDeprecationWarnings` | 5 | A3: deprecation warnings |
| `TestTaskBudgets` | 10 | A1: task budgets beta |
| **Total new** | **20** | |

### Summary of production code changes

| Location | Change |
|----------|--------|
| Lines 219–231 | B3: Comment cleanup (cosmetic) |
| Line ~202 | A3: `_DEPRECATED_MODELS` dict + `_warned_deprecated_models` set + `_clear_deprecated_model_warnings()` |
| Line ~199 | A1: `BETA_HEADER_TASK_BUDGETS` constant |
| `ModelCapabilities` | A1: `supports_task_budget: bool = False` field |
| `_get_capabilities()` opus block | A1: `supports_task_budget=is_47_plus` |
| `_apply_runtime_capability_overrides()` | A1: Pass through `supports_task_budget` |
| `_build_request_beta_headers()` | A1: `has_task_budget` param + `BETA_HEADER_TASK_BUDGETS` |
| `_complete_chat_request()` line ~1935 | A3: Deprecation warning emission |
| `_complete_chat_request()` line ~1948 | B4: `is not None` fix for temperature |
| `_complete_chat_request()` line ~2055 | A2: Buffer default 4096→8192 |
| `_complete_chat_request()` line ~2170 | A1: Task budget wiring + output_config assembly |
| `_complete_chat_request()` line ~2175 | A1: Pass `has_task_budget` to beta header builder |

### Stop before push

Do NOT push. The user will review the changes and authorize the push.
