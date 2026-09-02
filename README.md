# Amplifier Anthropic Provider Module

Claude model integration for Amplifier via Anthropic API.

## Prerequisites

- **Python 3.11+**
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package manager

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Purpose

Provides access to Anthropic's Claude models (Claude 4 series: Sonnet, Opus, Haiku) as an LLM provider for Amplifier.

## Contract

**Module Type:** Provider
**Mount Point:** `providers`
**Entry Point:** `amplifier_module_provider_anthropic:mount`

## Supported Models

- `claude-sonnet-5` - Claude Sonnet 5 (recommended, default)
- `claude-opus-5` - Claude Opus 5 (most capable)
- `claude-haiku-4-5` - Claude Haiku 4.5 (fastest, cheapest)

## Configuration

```toml
[[providers]]
module = "provider-anthropic"
name = "anthropic"
config = {
    default_model = "claude-sonnet-5",
    max_tokens = 8192,
    temperature = 1.0,   # Silently ignored by models without sampling support (sonnet-5, opus-4.7+)
    raw = false          # Include full request/response payloads in llm:* events
}
```

See the [Configuration Reference](#configuration-reference) table below for every surviving key.

### Reasoning Effort

The `reasoning_effort` config key (canonical — it matches the kernel's portable
`request.reasoning_effort` field) sets a session-level default reasoning effort
applied to **every** request, so you can opt into stronger reasoning once
instead of supplying it per-request. The legacy `effort` key remains a working
alias; when both are set, `reasoning_effort` wins (a warning is logged).

```yaml
providers:
  - module: provider-anthropic
    config:
      default_model: claude-opus-4-8
      reasoning_effort: xhigh  # legacy alias: effort
```

**This enables extended thinking.** In this provider, `effort` (like the kernel's
portable `request.reasoning_effort`) deliberately maps to Anthropic **extended
thinking** — setting it engages thinking and controls its depth, the same way
OpenAI's reasoning effort engages its reasoning. At the Anthropic API level
`effort` and `thinking` are independent primitives; coupling them is Amplifier's
"reason harder" product semantics. Because it turns thinking on for every call,
**leave `effort` blank unless you want stronger (and more expensive) reasoning by
default.**

**Accepted values:**

| Value | Meaning | Availability |
| --- | --- | --- |
| `low` | Minimal thinking, most token-efficient | All thinking-capable models |
| `medium` | Balanced | All thinking-capable models |
| `high` | Default intensity (same as omitting `effort`) | All thinking-capable models |
| `xhigh` | Extended capability for long-horizon agentic/coding work | Opus 4.7+ |
| `max` | Maximum capability, no token constraints | Opus 4.8+ |

**Precedence** (highest wins): per-call `effort` kwarg → `request.reasoning_effort`
(set by the orchestrator) → this `effort` config default. Note the per-call
`effort` kwarg is an `output_config.effort`-only override; the `reasoning_effort`
chain is what enables thinking.

**Notes**:
- Invalid values (e.g. `ultra`, `EXTRA HIGH`) are normalised (trimmed/lowercased)
  and, if still unrecognised, ignored with a warning — they never silently turn
  thinking on.
- `output_config.effort` is currently only emitted for models the capability
  matrix marks as supporting it (**Opus 4.7+** today). On other thinking-capable
  models the extended-thinking mapping still applies. Broadening this to
  Sonnet 4.6 and Opus 4.5/4.6 (which Anthropic also supports) is tracked as a
  follow-up.
- `xhigh`/`max` are omitted from `output_config.effort` on models whose capability
  matrix doesn't list them (a warning is logged), falling back to adaptive thinking.
- This key is exposed through `amplifier provider use` (shown for thinking-capable
  models), so it can be set interactively without hand-editing YAML.

### Thinking Budget

`thinking.budget_tokens` is the *entire* reasoning dial on models without
`output_config.effort` support -- Haiku 4.5 among them, where `effort` never
reaches the wire at all. Two config keys control it:

```yaml
providers:
  - module: provider-anthropic
    config:
      default_model: claude-haiku-4-5
      extended_thinking: true       # turn thinking on WITHOUT choosing an effort
      thinking_budget_tokens: 8000  # explicit budget, honored as written
```

**Precedence** (highest wins): per-call `thinking_budget_tokens` kwarg ->
`thinking_budget_tokens` config -> the budget implied by `reasoning_effort`
(`low` -> 4096, everything else -> the model default) -> the model default.

An **explicit** budget outranks the effort-implied one, because the effort ladder
is a derived default and a configured number is caller intent. Setting only
`reasoning_effort` is unchanged: you still get 4096 for `low` and the model
default otherwise.

**Turning thinking on:** `reasoning_effort` enables thinking as a side effect, but
it also selects a thinking depth and, on Opus 4.7+, sets `output_config.effort`.
`extended_thinking: true` enables thinking *and nothing else*, which is what you
want when the budget is the only dial you care about. `extended_thinking: false`
is an explicit opt-out that overrides a configured `reasoning_effort`; a per-call
`extended_thinking` kwarg overrides both.

**It is never silently discarded.** If an explicitly requested budget does not
reach `thinking.budget_tokens`, a warning names the key, the value you asked for,
the value actually sent, and why. That happens when:

| Situation | What is sent | Remedy |
| --- | --- | --- |
| Thinking is not enabled | no `thinking` block at all | set `reasoning_effort` or `extended_thinking: true` |
| Model has no thinking support (e.g. Haiku 3.5) | no `thinking` block at all | use a thinking-capable model |
| Resolved `thinking_type` is `adaptive` | `{"type": "adaptive"}` -- the API forbids `budget_tokens` here; your value sizes `max_tokens` instead | set `thinking_type: enabled` |
| Value is outside the model's limits | the clamped value (min 1024, max the model's output ceiling) | pick a value inside the range |

A non-integer value is ignored with a warning and the resolved default is used --
a typo does not raise out of every request.

### Debug / Raw Payload Capture

`debug` and `raw_debug` never existed as real keys -- this section documented
them in error for a long time. The real key is `raw` (boolean, default
`false`): when `true`, full request/response payloads are attached to the
`llm:request` / `llm:response` events for inspection.

### Retry and Error Handling

The provider disables SDK built-in retries (`max_retries=0`) and manages retries itself via `amplifier_core.utils.retry.retry_with_backoff()`. This gives the provider full control over backoff timing, retry-after header honoring, and per-error-class delay scaling.

#### Error Translation

SDK exceptions are translated to kernel errors before the retry loop sees them. All translations preserve the original exception as `__cause__` for debugging.

| SDK Exception | Condition | Kernel Error | Status | Retryable |
| --- | --- | --- | --- | --- |
| `RateLimitError` | 429 | `RateLimitError` | 429 | Yes |
| `OverloadedError` | 529 | `ProviderUnavailableError` | 529 | Yes (10× backoff) |
| `InternalServerError` | 5xx | `ProviderUnavailableError` | 5xx | Yes |
| `AuthenticationError` | 401 | `AuthenticationError` | 401 | No |
| `BadRequestError` | prompt/context window overflow (e.g. `prompt is too long: ... tokens > ... maximum`) | `ContextLengthError` | 400 | No |
| `BadRequestError` | safety / content filter / blocked | `ContentFilterError` | 400 | No |
| `BadRequestError` | other | `InvalidRequestError` | 400 | No |
| `APIStatusError` | 403 | `AccessDeniedError` | 403 | No |
| `APIStatusError` | 404 | `NotFoundError` | 404 | No |
| `APIStatusError` | other non-5xx | `LLMError` | — | No |
| `asyncio.TimeoutError` | — | `LLMTimeoutError` | — | Yes |
| Other | — | `LLMError` | — | Yes |

#### Backoff Formula

Each retry delay is computed as follows:

```
base_delay   = min_retry_delay × 2^(attempt - 1)
capped_delay = min(base_delay, max_retry_delay)
scaled_delay = capped_delay × delay_multiplier          # 1.0 for most errors, 10.0 for 529
final_delay  = max(scaled_delay, retry_after)            # server retry-after as floor
sleep        = final_delay ± (final_delay × jitter)      # randomised ± jitter fraction
```

**Example: 529 Overloaded (10× multiplier, defaults)**

| Attempt | base_delay | capped | ×10 | Sleep |
| --- | --- | --- | --- | --- |
| 1 | 1s | 1s | 10s | 10s |
| 2 | 2s | 2s | 20s | 20s |
| 3 | 4s | 4s | 40s | 40s |
| 4 | 8s | 8s | 80s | 80s |
| 5 | 16s | 16s | 160s | 160s |

Total wait ≈ 310s (~5 min) before the request is abandoned.

#### Retry Configuration

```yaml
providers:
  - module: provider-anthropic
    config:
      max_retries: 5
      min_retry_delay: 1.0
      max_retry_delay: 60.0
      retry_jitter: true
      overloaded_delay_multiplier: 10.0
```

| Key | Default | Description |
| --- | --- | --- |
| `max_retries` | `5` | Maximum retry attempts before giving up |
| `min_retry_delay` | `1.0` | Base delay in seconds for the first retry |
| `max_retry_delay` | `60.0` | Cap on the base delay (before multiplier) |
| `retry_jitter` | `true` | Randomise each retry delay to avoid thundering-herd retries. **Boolean.** A numeric value such as `0.2` is parsed as `false` and disables jitter -- use `true`/`false` |
| `overloaded_delay_multiplier` | `10.0` | Multiplier applied to delays for 529 Overloaded errors |

#### Events

A `provider:retry` event is emitted before each retry sleep with the following fields:

| Field | Description |
| --- | --- |
| `provider` | Provider name (`"anthropic"`) |
| `model` | Model being called |
| `attempt` | Current retry attempt number |
| `max_retries` | Configured maximum retries |
| `delay` | Computed sleep duration in seconds |
| `retry_after` | Server retry-after value (or `null`) |
| `error_type` | Kernel error class name |
| `error_message` | Error description |

## Prompt Cache TTL

By default, prompt-cache breakpoints use Anthropic's standard 5-minute TTL. The
`cache_stable_region_ttl_1h` config key opts into a 1-hour TTL for the two most
stable cache breakpoints only:

```yaml
providers:
  - module: provider-anthropic
    config:
      enable_prompt_caching: true          # required -- see below
      cache_stable_region_ttl_1h: true
```

**What it covers**: the **system prompt and tool-definition** breakpoints only.
It deliberately never applies to conversation-region breakpoints, which move
every turn on a rolling basis and would pay the higher write premium (below)
on content that's about to be superseded anyway. Extending the conversation
region to a longer TTL is a separate, currently-unimplemented idea tracked
upstream in [microsoft/amplifier#337](https://github.com/microsoft/amplifier/issues/337).
See the design comment above `self.cache_stable_region_ttl_1h` in
`amplifier_module_provider_anthropic/__init__.py` for the full rationale.

**The economics**: Anthropic bills 1-hour TTL cache *writes* at 2x the base
input-token rate, versus 1.25x for the default 5-minute TTL. That's a real
up-front cost -- but a longer TTL means the write survives more calls before
it needs to be repeated. Once a cache entry is reused across roughly two or
more reads inside the 1-hour window that a 5-minute TTL would have missed
(because the previous call was more than 5 minutes ago), the 1h TTL comes out
ahead. It helps most for:

- **Long-running sessions** with gaps between calls longer than 5 minutes
  (e.g. a human pausing between turns, a scheduled/cron-triggered agent).
- **Stable system prompts and tool sets** that don't change turn-to-turn --
  exactly the two regions this knob targets.

If your calls are consistently more frequent than every 5 minutes, the
default 5-minute TTL already keeps the cache warm at the cheaper 1.25x write
rate, and enabling this knob only adds cost.

**Requires prompt caching itself.** Because this knob only affects existing
cache breakpoints, it is a no-op when `enable_prompt_caching` is `false` (or
left at its `false` default) -- the provider will not append the
`extended-cache-ttl-2025-04-11` beta header in that case, and logs a one-line
notice explaining why. Enable both together.

**Left unset by default** (rather than defaulting to `false`): this makes
"no opinion, use the provider's own 5-minute default" a real, distinguishable
third state in the config wizard, separate from an explicit opt-out.

## Beta Headers

Anthropic provides experimental features through beta headers. Enable these features by adding the `beta_headers` configuration field.

### Configuration

**Single beta header:**
```yaml
providers:
  - module: provider-anthropic
    config:
      default_model: claude-sonnet-4-5
      beta_headers: "context-1m-2025-08-07"  # Enable 1M token context window
```

**Multiple beta headers:**
```yaml
providers:
  - module: provider-anthropic
    config:
      default_model: claude-sonnet-4-5
      beta_headers:
        - "context-1m-2025-08-07"
        - "future-feature-header"
```

### 1M Token Context Window

1M context is **generally available, on by default, and billed at standard
pricing** on every model that has it (Opus 5/4.8/4.7/4.6, Sonnet 5/4.6, Fable
5/5.1, Mythos 5/Preview). No beta header is required, and there is no
long-context price premium
([Anthropic: Context windows](https://platform.claude.com/en/docs/build-with-claude/context-windows),
verified 2026-08-29). Those models cap output at **128K tokens** per request
regardless of context size.

The `enable_1m_context` config key does **not** change what the API accepts.
It only sets the context window this provider *advertises* to Amplifier's
context manager (200K vs 1M), which determines how much conversation history
is kept per request -- a cost decision. It defaults to `false`.

### Notes

- Beta features are experimental and subject to change
- Check [Anthropic's documentation](https://docs.anthropic.com) for available beta headers
- Beta headers are optional - existing configurations work unchanged
- Invalid beta headers will cause API errors (fail fast)
- Beta header usage is logged at initialization for observability

## Configuration Reference

House-style key reference. ✅ = wizard-visible ConfigField, ⚙️ = settings-only (fully functional, not asked in the wizard).

| Key | Default | Surface | Description |
|---|---|---|---|
| `api_key` | *(env `ANTHROPIC_API_KEY`)* | ✅ | Anthropic API key |
| `base_url` | `https://api.anthropic.com` | ✅ | Custom endpoint |
| `default_model` | `claude-sonnet-5` | *(model picker)* | Model used when a request does not name one |
| `reasoning_effort` | *(unset)* | ✅ | `low`\|`medium`\|`high`\|`xhigh`\|`max`. Enables extended thinking. Legacy alias: `effort` (deprecated) |
| `extended_thinking` | *(unset)* | ⚙️ | Turn extended thinking on/off without choosing an effort. Overrides the `reasoning_effort` implication; a per-call kwarg overrides this |
| `thinking_budget_tokens` | *(model default)* | ⚙️ | Explicit `thinking.budget_tokens`. Outranks the effort-implied budget; warns if it can't reach the wire |
| `thinking_budget_buffer` | `8192` | ⚙️ | Headroom added to the budget when sizing `max_tokens` |
| `thinking_type` | `adaptive` | ⚙️ | `adaptive`\|`enabled`. `adaptive` lets the model manage its own budget (and forbids `budget_tokens`); falls back to `enabled` on models without adaptive support |
| `enable_1m_context` | `false` | ✅ | Advertise the 1M context window (more history kept = higher cost) |
| `cache_stable_region_ttl_1h` | *(unset)* | ✅ | 1h cache TTL for system prompt + tools. 2x write cost, fewer writes |
| `enable_prompt_caching` | `true` | ⚙️ | Place cache breakpoints |
| `max_tokens` | *(model ceiling)* | ⚙️ | Output token cap |
| `temperature` | `0.7` | ⚙️ | Ignored by non-sampling models (Sonnet 5, Opus 4.7+) |
| `timeout` | `600.0` | ⚙️ | API timeout, seconds |
| `priority` | `100` | ⚙️ | Provider selection priority (lower wins) |
| `raw` | `false` | ⚙️ | Include full request/response payloads in `llm:request`/`llm:response` events |
| `fallback_on_overload` | `false` | ⚙️ | Downgrade one ladder rung (fable/mythos → opus → sonnet → haiku) after persistent 529s |
| `fallback_retry_count` | `2` | ⚙️ | 529 retries before downgrading |
| `fallback_cooldown_seconds` | `300` | ⚙️ | How long a downgrade window stays open |
| `fallback_models` | *(unset)* | ⚙️ | Per-family target overrides, e.g. `{opus: claude-opus-4-8}` |
| `persist_fallback_state` | `false` | ⚙️ | Share downgrade windows across processes |
| `refusal_fallback_enabled` | `true` | ⚙️ | Retry once on `finish_reason="refusal"`, via the same ladder as overload |
| `extra_request_params` | *(unset)* | ⚙️ | Escape hatch for Messages API params this provider doesn't model; merged last, user-wins |

Removed keys (still recognized, warn with a migration message): `fallback_sonnet_model`,
`fallback_haiku_model` (use `fallback_models`), `refusal_fallback_model` (refusal fallback
now follows the same ladder as overload -- see Fallback section).

## Environment Variables

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

```python
# In amplifier configuration
[provider]
name = "anthropic"
default_model = "claude-sonnet-5"
```

## Features

- Streaming support
- Tool use (function calling)
- Vision capabilities (on supported models)
- Token counting and management
- **Message validation** before API calls (defense in depth)

## Graceful Error Recovery

The provider implements automatic repair for incomplete tool call sequences:

**The Problem**: If tool results are missing from conversation history (due to context compaction bugs, parsing errors, or state corruption), the Anthropic API rejects the entire request, breaking the user's session.

**The Solution**: The provider automatically detects and repairs missing tool_results by injecting synthetic results:

1. **Repair before validation** - Detects missing tool_results and injects synthetic ones
2. **Make failures visible** - Synthetic results contain `[SYSTEM ERROR: Tool result missing]` messages
3. **Maintain conversation validity** - API accepts repaired messages, session continues
4. **Enable recovery** - LLM acknowledges error and can ask user to retry
5. **Provide observability** - Emits `provider:tool_sequence_repaired` event with repair details
6. **Validate remaining** - After repair, strict validation catches any remaining inconsistencies

**Example**:
```python
# Anthropic format (after _convert_messages)
messages = [
    {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {...}}
        ]
    },
    # MISSING: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_123", ...}]}
    {"role": "user", "content": "Thanks"}
]

# Provider repairs by injecting synthetic result:
# Either appends to existing user message or inserts new one
{
    "role": "user",
    "content": [{
        "type": "tool_result",
        "tool_use_id": "toolu_123",
        "content": "[SYSTEM ERROR: Tool result missing]\n\nTool: get_weather\n..."
    }]
}
```

**Observability**: Repairs are logged as warnings and emit `provider:tool_sequence_repaired` events for monitoring.

**Philosophy**: This is **graceful degradation** following kernel philosophy - errors in other modules (context management) don't crash the provider or kill the user's session

## Dependencies

- `amplifier-core>=1.0.0`
- `anthropic>=1.0.0,<2.0.0`

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
