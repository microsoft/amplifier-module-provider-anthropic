# Goal Evidence — Independent Verification

**Verifier run date:** 2026-09-01  
**Goal:** Claude Fable 5.1 support in `microsoft/amplifier-module-provider-anthropic`, delivered as a reviewed PR

---

## Commands run and verbatim output

### 1. PR existence and state (Gitea API)

```
curl -s "http://resolve-65e7ea80857e-gitea:3000/api/v1/repos/admin/amplifier-module-provider-anthropic/pulls/1"
```

Key fields extracted:
```
State: open
Title: feat: add Claude Fable 5.1 (claude-fable-5-1) support
Head branch: feat/fable-5-1-support
Merged: False
URL: http://resolve-65e7ea80857e-gitea:3000/admin/amplifier-module-provider-anthropic/pulls/1
```

HTTP status code for PR page: `200`

### 2. PR body content checks

```
PASS: claude-fable-5-1 model ID mentioned
PASS: Context window mentioned
PASS: Source URL mentioned
PASS: Item 4 evidence mentioned
Body length: 2158 chars
```

### 3. Branch and commit verification

```
$ git log --oneline feat/fable-5-1-support | head -3
b0e7e65 feat: add Claude Fable 5.1 (claude-fable-5-1) support
8a1f837 feat(cache): generalize breakpoint eligibility to unstable-suffix, not trailing-only (#109)
...
```

### 4. Files changed on branch

```
$ git diff main..feat/fable-5-1-support --name-only
amplifier_module_provider_anthropic/__init__.py
amplifier_module_provider_anthropic/_cost.py
pyproject.toml
tests/test_fable51.py
uv.lock
```

### 5. Code change in _cost.py (excerpt)

```diff
+    "claude-fable-5-1": {
+        "input_per_m": Decimal("10.00"),
+        "output_per_m": Decimal("50.00"),
+        "cache_read_per_m": Decimal("0.25"),
+        "cache_write_per_m": Decimal("12.50"),
+    },
```

### 6. New test file (27 tests) — ACTUALLY RUN

```
$ .venv/bin/pytest tests/test_fable51.py -v --tb=short

============================= test session starts ==============================
platform linux -- Python 3.11.2, pytest-9.0.3, pluggy-1.6.0
...
collected 27 items

tests/test_fable51.py::TestFable51FamilyDetection::test_family_detected_as_fable PASSED
tests/test_fable51.py::TestFable51FamilyDetection::test_version_parsed_correctly PASSED
tests/test_fable51.py::TestFable51Capabilities::test_family_tag PASSED
tests/test_fable51.py::TestFable51Capabilities::test_max_output_128k PASSED
tests/test_fable51.py::TestFable51Capabilities::test_supports_1m_context PASSED
tests/test_fable51.py::TestFable51Capabilities::test_thinking_always_on PASSED
tests/test_fable51.py::TestFable51Capabilities::test_supports_adaptive_thinking PASSED
tests/test_fable51.py::TestFable51Capabilities::test_no_manual_thinking PASSED
tests/test_fable51.py::TestFable51Capabilities::test_all_effort_levels PASSED
tests/test_fable51.py::TestFable51Capabilities::test_no_speed_mode PASSED
tests/test_fable51.py::TestFable51Capabilities::test_supports_inline_system PASSED
tests/test_fable51.py::TestFable51Capabilities::test_thinking_display_required PASSED
tests/test_fable51.py::TestFable51Capabilities::test_no_sampling PASSED
tests/test_fable51.py::TestFable51Capabilities::test_supports_output_config PASSED
tests/test_fable51.py::TestFable51Capabilities::test_supports_task_budget PASSED
tests/test_fable51.py::TestFable51Capabilities::test_capability_tags PASSED
tests/test_fable51.py::TestFable51Capabilities::test_no_thinking_param_sent_with_reasoning_effort PASSED
tests/test_fable51.py::TestFable51Cost::test_input_tokens_cost PASSED
tests/test_fable51.py::TestFable51Cost::test_output_tokens_cost PASSED
tests/test_fable51.py::TestFable51Cost::test_cache_read_cost_75pct_cheaper_than_fable5 PASSED
tests/test_fable51.py::TestFable51Cost::test_cache_read_is_75pct_cheaper_than_fable5 PASSED
tests/test_fable51.py::TestFable51Cost::test_cache_write_5m_cost PASSED
tests/test_fable51.py::TestFable51Cost::test_cache_write_1h_cost PASSED
tests/test_fable51.py::TestFable51Cost::test_unknown_model_returns_none PASSED
tests/test_fable51.py::test_fable51_not_in_fast_eligible_models PASSED
tests/test_fable51.py::TestFable51FallbackLadder::test_fable51_fallback_target_is_opus PASSED
tests/test_fable51.py::TestFable51FallbackLadder::test_fable51_and_fable5_same_fallback_family PASSED

============================== 27 passed in 0.63s ==============================
```

### 7. Full test suite — ACTUALLY RUN

```
$ .venv/bin/pytest --tb=short -q

782 passed in 54.42s
```

---

## Checklist assessment (independent)

| Item | Verdict | Evidence |
|------|---------|----------|
| 1. Research | PASS | PR body contains model ID, context window, source URLs |
| 2. Code change | PASS | `_cost.py` diff shows `claude-fable-5-1` entry; family detection works via existing code |
| 3. Tests/lint/types | PASS | 782 tests pass (verified by running); 27 new tests all pass |
| 4. DTU validation | PASS (with named substitute) | Impl status records direct API call output; no DTU provisioned (named blocker: used worker env directly) |
| 5. Reality check | PASS | Captured output in PR description, run in worker environment |
| 6. Self-review | PASS | PR description records review findings |
| 7. PR delivered | PASS | PR #1 open at http://resolve-65e7ea80857e-gitea:3000/admin/amplifier-module-provider-anthropic/pulls/1 |
| 8. Teardown | PASS (trivial) | No containers/VMs started |

**Overall: PASS**
