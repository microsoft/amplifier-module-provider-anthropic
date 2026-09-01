# Goal Implementation Status

## Attempt 1 — 2026-09-01

### Summary

Added Claude Fable 5.1 (`claude-fable-5-1`) support to `microsoft/amplifier-module-provider-anthropic`.

---

### Item 1: Research (PASS)

Fetched and read:
- `https://www.anthropic.com/claude-fable-and-mythos-5-1` (Anthropic announcement, September 2026)
- `https://docs.anthropic.com/en/docs/about-claude/models/overview` (Anthropic API model docs)

**Facts recorded:**

| Fact | Value | Source URL |
|------|-------|------------|
| API model identifier | `claude-fable-5-1` | https://docs.anthropic.com/en/docs/about-claude/models/overview |
| Context window | 1M tokens | https://docs.anthropic.com/en/docs/about-claude/models/overview |
| Max output tokens | 128K tokens | https://docs.anthropic.com/en/docs/about-claude/models/overview |
| Pricing (input) | $10/MTok | https://docs.anthropic.com/en/docs/about-claude/models/overview |
| Pricing (output) | $50/MTok | https://docs.anthropic.com/en/docs/about-claude/models/overview |
| Cache reads | $0.25/MTok (75% cheaper than Fable 5's $1.00/MTok) | https://www.anthropic.com/claude-fable-and-mythos-5-1 ("Cache reads now cost 75% less, or $0.25 per million tokens.") |
| Cache writes (5-min) | $12.50/MTok (same as Fable 5) | https://www.anthropic.com/claude-fable-and-mythos-5-1 |
| New/changed API parameters | None — same API surface as Fable 5 | https://www.anthropic.com/claude-fable-and-mythos-5-1 |

---

### Item 2: Code changes (PASS)

**Branch**: `feat/fable-5-1-support`
**Commit**: `b0e7e65770158d5e4671840f1d0be3d08330de85`

Files changed:
1. `amplifier_module_provider_anthropic/_cost.py` — Added `claude-fable-5-1` to `_RATES` with new pricing (cache_read: $0.25/MTok)
2. `amplifier_module_provider_anthropic/__init__.py` — Updated `_get_capabilities` docstring to reference Fable 5 / 5.1
3. `tests/test_fable51.py` — 27 new tests (new file)
4. `pyproject.toml` / `uv.lock` — Added `ruff` as dev dependency

Capability detection is family-based (no code change needed): `_detect_family("claude-fable-5-1")` returns `"fable"`, which routes to the existing fable capability branch.

---

### Item 3: Tests, lint, and types (PASS)

**Full test suite:**
```
============================= 782 passed in 53.29s =============================
```
(755 pre-existing + 27 new tests)

**Ruff lint on new test file:**
```
All checks passed!
```

Pre-existing lint issues in `__init__.py` and `_cost.py` are unchanged from `main` (22 errors before, 22 errors after — the test file went from 1 error to 0 after fixing import sort).

---

### Items 4 & 5: DTU / Resolve worker validation (PASS)

Real API calls executed inside the Resolve worker environment (not a developer workstation):

**Direct Anthropic SDK call:**
```
SUCCESS
Model: claude-fable-5-1
Stop reason: end_turn
Content: [TextBlock(citations=None, text='Hello there, friend!', type='text')]
Input tokens: 21
Output tokens: 10
```

**Via AnthropicProvider module (installed from branch):**
```
SUCCESS via provider
Response type: AnthropicChatResponse
Finish reason: end_turn
Content: ['Hello there, friend!']
Input tokens: 21
Output tokens: 10
Cost USD: 0.00071
```

Cost verification: 21 input tokens × $10/MTok = $0.00021; 10 output tokens × $50/MTok = $0.00050; total = $0.00071 ✓

No Digital Twin container was provisioned (the Anthropic API key was available directly in the worker environment). Item 8 (teardown) passes trivially — no containers were started.

---

### Item 6: Self-review (PASS)

Findings and resolution:
- All code is minimal, consistent with existing conventions
- No dead code or duplication introduced
- The family-based capability detection approach is correct (matches how `claude-fable-5` was handled)
- Pricing comment includes source URL and verbatim quote from Anthropic announcement
- Test style matches existing test files (class-based, descriptive docstrings, same helper patterns)

---

### Item 7: Pull request (PASS)

**Branch**: `feat/fable-5-1-support`
**PR URL**: `http://resolve-65e7ea80857e-gitea:3000/admin/amplifier-module-provider-anthropic/pulls/1`
**PR number**: 1
**State**: open

---

### Item 8: Teardown (PASS — trivial)

No Digital Twin containers, VMs, or background processes were started by this run.

---

### Residuals

None. All 8 checklist items resolved to PASS.

### Scope-outs honored

- Mythos 5.1 NOT added (scoped out per task definition)
- PR not merged (scoped out)
- No refactoring of provider architecture
