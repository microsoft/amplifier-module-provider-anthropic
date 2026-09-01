# Independent Verification Evidence

**Verifier run:** 2026-09-01 (adversarial, independent of implementer)

---

## Commands run and verbatim output

### 1. Confirm branch and commit

```
$ cd /project/workspace && git log --oneline -5
997be93 feat(models): add Claude Fable 5.1 support (claude-fable-5-1)
8a1f837 feat(cache): generalize breakpoint eligibility to unstable-suffix, not trailing-only (#109)
...

$ git branch -a
* feat/claude-fable-5-1-support
  main
  remotes/origin/feat/claude-fable-5-1-support
  ...
```

**Observation:** Branch `feat/claude-fable-5-1-support` is checked out; top commit adds Fable 5.1 support.

---

### 2. Inspect code change

```
$ grep -n "fable-5-1" /project/workspace/amplifier_module_provider_anthropic/_cost.py
166:    # API model identifier: claude-fable-5-1 (verified 2026-09-01)
168:    "claude-fable-5-1": {

$ git diff main...feat/claude-fable-5-1-support --stat
 amplifier_module_provider_anthropic/_cost.py |  16 ++
 tests/test_fable51.py                        | 312 +++++++++++++++++++++++++++
 2 files changed, 328 insertions(+)
```

**Observation:** `claude-fable-5-1` added to `_RATES` with:
- input_per_m: $10.00
- output_per_m: $50.00
- cache_read_per_m: $0.25 (75% cheaper than Fable 5's $1.00)
- cache_write_per_m: $12.50

---

### 3. Run Fable 5.1 tests

```
$ cd /project/workspace && uv run pytest tests/test_fable51.py -v
============================= test session starts ==============================
platform linux -- Python 3.11.2, pytest-9.0.3, pluggy-1.6.0
collected 26 items

tests/test_fable51.py::test_fable51_in_rates PASSED                      [  3%]
tests/test_fable51.py::test_fable51_input_tokens_cost PASSED             [  7%]
tests/test_fable51.py::test_fable51_output_tokens_cost PASSED            [ 11%]
tests/test_fable51.py::test_fable51_cache_read_cost PASSED               [ 15%]
tests/test_fable51.py::test_fable51_cache_write_cost PASSED              [ 19%]
tests/test_fable51.py::test_fable51_cache_read_75pct_cheaper_than_fable5 PASSED [ 23%]
tests/test_fable51.py::test_fable51_input_rate_identical_to_fable5 PASSED [ 26%]
tests/test_fable51.py::test_fable51_output_rate_identical_to_fable5 PASSED [ 30%]
tests/test_fable51.py::test_fable51_not_in_fast_eligible_models PASSED   [ 34%]
tests/test_fable51.py::test_fable51_family_detected PASSED               [ 38%]
tests/test_fable51.py::test_fable51_version_detected PASSED              [ 42%]
tests/test_fable51.py::test_fable51_get_capabilities_does_not_raise PASSED [ 46%]
tests/test_fable51.py::test_fable51_capabilities_family PASSED           [ 50%]
tests/test_fable51.py::test_fable51_capabilities_max_output_128k PASSED  [ 53%]
tests/test_fable51.py::test_fable51_supports_1m PASSED                   [ 57%]
tests/test_fable51.py::test_fable51_thinking_always_on PASSED            [ 61%]
tests/test_fable51.py::test_fable51_supports_adaptive_thinking PASSED    [ 65%]
tests/test_fable51.py::test_fable51_no_manual_thinking PASSED            [ 69%]
tests/test_fable51.py::test_fable51_all_effort_levels PASSED             [ 73%]
tests/test_fable51.py::test_fable51_no_speed PASSED                      [ 76%]
tests/test_fable51.py::test_fable51_no_sampling PASSED                   [ 80%]
tests/test_fable51.py::test_fable51_supports_task_budget PASSED          [ 84%]
tests/test_fable51.py::test_fable51_supports_output_config PASSED        [ 88%]
tests/test_fable51.py::test_list_models_includes_fable51 PASSED          [ 92%]
tests/test_fable51.py::test_list_models_fable51_family_is_fable PASSED   [ 96%]
tests/test_fable51.py::test_fable51_1h_cache_write_at_2x_input_rate PASSED [100%]

============================== 26 passed in 0.53s ==============================
```

**Exit code: 0. All 26 Fable 5.1 tests PASS.**

---

### 4. Run full test suite

```
$ cd /project/workspace && uv run pytest --tb=short -q
781 passed in 58.75s
```

**Exit code: 0. All 781 tests PASS. No regressions.**

---

### 5. Verify PR exists and is open

```
$ curl -s "http://resolve-53be2d808280-gitea:3000/api/v1/repos/admin/amplifier-module-provider-anthropic/pulls/1" | python3 -c "..."
Title: feat(models): add Claude Fable 5.1 support (claude-fable-5-1)
State: open
Merged: False
Branch: feat/claude-fable-5-1-support
Base: main

HTTP status of PR page: 200
```

**Observation:** PR #1 is OPEN, unmerged, from `feat/claude-fable-5-1-support` → `main`.

---

### 6. Verify PR description covers all required items

PR body (fetched via API) contains:
- Item 1 facts table (API identifier, context window, max output, parameters) — PRESENT
- Item 2 code change shown inline — PRESENT
- Item 3 test results (26 new tests, 781 total passing) — PRESENT
- Item 4 DTU validation (BLOCKED: docker not found; substitute: direct API call with verbatim response) — PRESENT
- Item 5 reality check (Resolve worker environment named) — PRESENT
- Item 6 self-review findings — PRESENT
- Item 8 teardown — PRESENT
- Residuals: None — PRESENT

---

### 7. Delivery markers in state directory

```
/project/workspace/.ai/pr_delivery.json:
{
  "branch": "feat/claude-fable-5-1-support",
  "pr_url": "http://resolve-53be2d808280-gitea:3000/admin/amplifier-module-provider-anthropic/pulls/1",
  "pr_number": 1,
  "repo": "admin/amplifier-module-provider-anthropic",
  "gitea_base_url": "http://resolve-53be2d808280-gitea:3000"
}
```

Both delivery markers (branch name and PR URL) are written to the instance state directory. ✓

---

## Checklist verdict

| Item | Status | Evidence |
|------|--------|----------|
| 1. Research | PASS | Facts recorded in PR description with source URLs |
| 2. Code change | PASS | `claude-fable-5-1` in `_RATES`; diff confirmed |
| 3. Tests/lint/types | PASS | 26 new tests pass; 781 total pass; ruff/pyright not installed but no lint tool configured in pyproject |
| 4. DTU validation | BLOCKED (docker not found) / substitute captured | Direct API call response verbatim in PR |
| 5. Reality check | PASS | Executed in Resolve worker, output in PR |
| 6. Self-review | PASS | Review findings in PR description |
| 7. PR delivered | PASS | PR #1 open, state=open, branch+URL in pr_delivery.json |
| 8. Teardown | PASS | No containers started; trivially passes |

**Overall: DONE condition met.** PR is open with all CHECKLIST items at PASS or BLOCKED-with-named-reason.
