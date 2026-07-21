# VERIFY — `cache_ttl` 1-hour prompt-cache knob

## Knob semantics

Config key: `cache_ttl` (string), read once at provider construction into
`AnthropicProvider._cache_ttl`.

| Config value            | Behavior |
|-------------------------|----------|
| absent / `""` / `None`  | **OFF (default).** All cache_control blocks are plain `{"type": "ephemeral"}`; no extra beta header. |
| `"1h"` (exact match)    | **ON.** All 4 cache_control sites emit `{"type": "ephemeral", "ttl": "1h"}` and the per-model beta header builder appends `extended-cache-ttl-2025-04-11`. |
| anything else (`"2h"`, `"5m"`, garbage, `"1H"`) | Behaves exactly as OFF. |

The knob is flag-deletable: with the key absent, request payloads are
byte-identical to pre-patch behavior.

Controlled sites in `amplifier_module_provider_anthropic/__init__.py`:

1. `_apply_tool_cache_control` — last tool definition
2. `_format_system_with_cache` — system content block
3. `_apply_message_cache_control` — list-content path
4. `_apply_message_cache_control` — str-content path
5. `_build_request_beta_headers` — beta-header guard (header appended iff `"1h"`)

## How to run verification

```bash
bash verify.sh
```

It performs, exiting nonzero on any failure:

1. `uv run pytest tests/test_cache_ttl.py -q` — knob contract unit tests
   (tool / system / message-list / message-str cache_control, beta header,
   default-off, bogus-value-off). No network calls.
2. `uv run pytest tests/ -q` — full suite (no regressions).
3. Grep assertions on `__init__.py`: exactly **5** occurrences of the
   ttl-conditional expression `getattr(self, "_cache_ttl", "")` and exactly
   **1** occurrence of `extended-cache-ttl-2025-04-11`.

## Note: pre-existing test fix (unrelated to the knob)

At the pinned base (`2e7232a`), three streaming tests in
`tests/test_tool_repair.py` failed with
`TypeError: 'async for' requires an object with __aiter__ method, got
MockStreamManager` — independently verified as pre-existing (they fail with
the knob patch fully reverted). Because acceptance requires the **full**
suite to exit 0, `MockStreamManager` was given a minimal async-iterator
protocol (`__aiter__`/`__anext__` yielding zero events), matching the real
SDK's `MessageStream` contract. This is a test-only fix; no production code
beyond the validated knob patch was touched.

## Live-probe acceptance (manual step)

The patch was E2-validated against the live Anthropic API. To re-validate
end-to-end:

1. Run a session with the provider configured with `cache_ttl: "1h"` and a
   prompt large enough to hit the cache-write minimum.
2. In `events.jsonl`, the **first** call's usage must report
   `cache_creation.ephemeral_1h_input_tokens > 0` (proving the 1h TTL was
   accepted, not the default 5m).
3. Wait **more than 5 minutes** (past the default TTL), re-run the same
   context, and confirm the follow-up call reports
   `cache_read_input_tokens > 0` — a cache hit only possible with the 1h TTL.
