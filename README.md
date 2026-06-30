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

- `claude-sonnet-4-5` - Claude Sonnet 4.5 (recommended, default)
- `claude-opus-4-6` - Claude Opus 4.6 (most capable)
- `claude-haiku-4-5` - Claude Haiku 4.5 (fastest, cheapest)

## Configuration

```toml
[[providers]]
module = "provider-anthropic"
name = "anthropic"
config = {
    default_model = "claude-sonnet-4-5",
    max_tokens = 8192,
    temperature = 1.0,
    debug = false,      # Enable standard debug events
    raw_debug = false   # Enable ultra-verbose raw API I/O logging
}
```

### Reasoning Effort

The `effort` config key sets a session-level default reasoning effort applied to
**every** request, so you can opt into stronger reasoning once instead of
supplying it per-request.

```yaml
providers:
  - module: provider-anthropic
    config:
      default_model: claude-opus-4-8
      effort: xhigh
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

### Debug Configuration

**Standard Debug** (`debug: true`):
- Emits `llm:request:debug` and `llm:response:debug` events
- Contains request/response summaries with message counts, model info, usage stats
- Moderate log volume, suitable for development

**Raw Debug** (`debug: true, raw_debug: true`):
- Emits `llm:request:raw` and `llm:response:raw` events
- Contains complete, unmodified request params and response objects
- Extreme log volume, use only for deep provider integration debugging
- Captures the exact data sent to/from Anthropic API before any processing

**Example**:
```yaml
providers:
  - module: provider-anthropic
    config:
      debug: true      # Enable debug events
      raw_debug: true  # Enable raw API I/O capture
      default_model: claude-sonnet-4-5
```

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
| `BadRequestError` | context length / too many tokens | `ContextLengthError` | 400 | No |
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
      retry_jitter: 0.2
      overloaded_delay_multiplier: 10.0
```

| Key | Default | Description |
| --- | --- | --- |
| `max_retries` | `5` | Maximum retry attempts before giving up |
| `min_retry_delay` | `1.0` | Base delay in seconds for the first retry |
| `max_retry_delay` | `60.0` | Cap on the base delay (before multiplier) |
| `retry_jitter` | `0.2` | Jitter fraction (0.0–1.0). Also accepts `true` (→ 0.2) or `false` (→ 0.0) for backward compatibility |
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

Claude Sonnet 4.5 supports a 1M token context window when the `context-1m-2025-08-07` beta header is enabled:

```yaml
providers:
  - module: provider-anthropic
    config:
      default_model: claude-sonnet-4-5
      beta_headers: "context-1m-2025-08-07"
      max_tokens: 8192  # Output tokens remain separate from context window
```

With this configuration:
- **Context window**: Up to 1M tokens of input (messages, tools, system prompt)
- **Output tokens**: Controlled by `max_tokens` (separate from context window)
- **Use case**: Process large codebases, extensive documentation, or long conversation histories

### Notes

- Beta features are experimental and subject to change
- Check [Anthropic's documentation](https://docs.anthropic.com) for available beta headers
- Beta headers are optional - existing configurations work unchanged
- Invalid beta headers will cause API errors (fail fast)
- Beta header usage is logged at initialization for observability

## Environment Variables

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

```python
# In amplifier configuration
[provider]
name = "anthropic"
default_model = "claude-sonnet-4-5"
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
- `anthropic>=0.25.0`

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
