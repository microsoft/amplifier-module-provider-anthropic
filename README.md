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

### Rate Limit Configuration

The provider uses the Anthropic SDK's built-in retry mechanism for rate limit errors (429) and server errors (5xx). Configure retry behavior:

```yaml
providers:
  - module: provider-anthropic
    config:
      max_retries: 5  # Number of retry attempts (default: 2)
```

**Behavior:**
- SDK automatically retries 429 (rate limit) and 5xx errors with exponential backoff
- Default is 2 retries (SDK default)
- Set to `0` to disable retries
- When retries are exhausted, emits `anthropic:rate_limited` event with retry timing info

**Events emitted on rate limit:**
- `anthropic:rate_limited` - Detailed rate limit info including `retry_after_seconds`
- `llm:response` with `status: "rate_limited"` - Standard error event

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
