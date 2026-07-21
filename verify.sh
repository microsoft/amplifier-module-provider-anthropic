#!/usr/bin/env bash
# verify.sh — acceptance verification for the cache_ttl (1h) knob.
# See VERIFY.md for knob semantics and the manual live-probe step.
set -euo pipefail
cd "$(dirname "$0")"

SRC="amplifier_module_provider_anthropic/__init__.py"

echo "==> [1/3] cache_ttl unit tests"
uv run pytest tests/test_cache_ttl.py -q

echo "==> [2/3] full test suite"
uv run pytest tests/ -q

echo "==> [3/3] grep assertions on ${SRC}"
# The ttl-conditional expression guards every knob-controlled site:
# 4 cache_control sites + 1 beta-header guard = exactly 5 occurrences.
ttl_count=$(grep -c 'getattr(self, "_cache_ttl", "")' "$SRC" || true)
if [ "${ttl_count}" -ne 5 ]; then
    echo "FAIL: expected exactly 5 occurrences of the ttl-conditional expression, found ${ttl_count}" >&2
    exit 1
fi

beta_count=$(grep -c 'extended-cache-ttl-2025-04-11' "$SRC" || true)
if [ "${beta_count}" -ne 1 ]; then
    echo "FAIL: expected exactly 1 occurrence of extended-cache-ttl-2025-04-11, found ${beta_count}" >&2
    exit 1
fi

echo "OK: 5 ttl-conditional sites, 1 beta-header string"
echo "verify.sh: ALL CHECKS PASSED"
