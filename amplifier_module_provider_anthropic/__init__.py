"""Anthropic provider module for Amplifier.

Integrates with Anthropic's Claude API for Claude models (Sonnet, Opus, Haiku).
Supports streaming, tool calling, extended thinking, and ChatRequest format.
"""

__all__ = ["mount", "AnthropicProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import logging
import os
import random
import time
from typing import Any

from amplifier_core import ConfigField
from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from amplifier_core import TextContent
from amplifier_core import ThinkingContent
from amplifier_core import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolCall
from anthropic import AsyncAnthropic
from anthropic import RateLimitError

logger = logging.getLogger(__name__)


class AnthropicChatResponse(ChatResponse):
    """ChatResponse with additional fields for streaming UI compatibility."""

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the Anthropic provider.

    Args:
        coordinator: Module coordinator
        config: Provider configuration including API key

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Get API key from config or environment
    api_key = config.get("api_key")
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        logger.warning("No API key found for Anthropic provider")
        return None

    provider = AnthropicProvider(api_key, config, coordinator)
    await coordinator.mount("providers", provider, name="anthropic")
    logger.info("Mounted AnthropicProvider")

    # Return cleanup function
    async def cleanup():
        if hasattr(provider.client, "close"):
            await provider.client.close()

    return cleanup


class AnthropicProvider:
    """Anthropic API integration.

    Provides Claude models with support for:
    - Text generation
    - Tool calling
    - Extended thinking
    - Streaming responses
    """

    name = "anthropic"
    api_label = "Anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ):
        """
        Initialize Anthropic provider.

        The SDK client is created lazily on first use, allowing get_info()
        to work without valid credentials.

        Args:
            api_key: Anthropic API key (can be None for get_info() calls)
            config: Additional configuration
            coordinator: Module coordinator for event emission
        """
        self._api_key = api_key
        self._client: AsyncAnthropic | None = None  # Lazy init
        self.config = config or {}
        self.coordinator = coordinator
        self.default_model = self.config.get("default_model", "claude-sonnet-4-5")
        self.max_tokens = self.config.get("max_tokens", 64000)
        self.temperature = self.config.get("temperature", 0.7)
        self.priority = self.config.get("priority", 100)  # Store priority for selection
        self.debug = self.config.get(
            "debug", False
        )  # Enable full request/response logging
        self.raw_debug = self.config.get(
            "raw_debug", False
        )  # Enable ultra-verbose raw API I/O logging
        self.debug_truncate_length = self.config.get(
            "debug_truncate_length", 180
        )  # Max string length in debug logs
        self.timeout = self.config.get(
            "timeout", 300.0
        )  # API timeout in seconds (default 5 minutes)

        # Rate limit retry configuration
        # We handle retries ourselves (SDK max_retries=0) to properly honor retry-after headers
        # and use longer backoffs that help with org-wide rate limit pressure
        self.max_retries = self.config.get(
            "max_retries", 5
        )  # Total retry attempts before failing
        self.retry_jitter = self.config.get(
            "retry_jitter", True
        )  # Add ±20% randomness to delays
        self.max_retry_delay = self.config.get(
            "max_retry_delay", 60.0
        )  # Cap individual wait at 60s
        self.min_retry_delay = self.config.get(
            "min_retry_delay", 1.0
        )  # Minimum delay if no retry-after header

        # Use streaming API by default to support large context windows (Anthropic requires streaming
        # for operations that may take > 10 minutes, e.g. with 300k+ token contexts)
        self.use_streaming = self.config.get("use_streaming", True)
        self.filtered = self.config.get(
            "filtered", True
        )  # Filter to curated model list by default
        self.enable_prompt_caching = self.config.get("enable_prompt_caching", True)

        # Get base_url from config for custom endpoints (proxies, local APIs, etc.)
        self._base_url = self.config.get("base_url")

        # Handle enable_1m_context from init wizard - translate to beta_headers
        # This bridges the config field (enable_1m_context boolean) to the actual
        # beta header that Anthropic API requires (context-1m-2025-08-07)
        enable_1m = self.config.get("enable_1m_context")
        if enable_1m and str(enable_1m).lower() in ("true", "1", "yes"):
            existing_beta = self.config.get("beta_headers", [])
            if isinstance(existing_beta, str):
                existing_beta = [existing_beta] if existing_beta else []
            if "context-1m-2025-08-07" not in existing_beta:
                existing_beta.append("context-1m-2025-08-07")
            self.config["beta_headers"] = existing_beta
            logger.info(
                "[PROVIDER] 1M context window enabled via enable_1m_context config"
            )

        # Beta headers support for enabling experimental features
        # Store as instance variable so we can merge with per-request headers later
        beta_headers_config = self.config.get("beta_headers")
        self._beta_headers: list[str] = []
        self._default_headers: dict[str, str] | None = None
        if beta_headers_config:
            # Normalize to list (supports string or list of strings)
            self._beta_headers = (
                [beta_headers_config]
                if isinstance(beta_headers_config, str)
                else list(beta_headers_config)
            )
            # Build anthropic-beta header value (comma-separated)
            beta_header_value = ",".join(self._beta_headers)
            self._default_headers = {"anthropic-beta": beta_header_value}
            logger.info(f"[PROVIDER] Beta headers enabled: {beta_header_value}")

        # Track tool call IDs that have been repaired with synthetic results.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations (since synthetic results
        # are injected into request.messages but not persisted to message store).
        self._repaired_tool_ids: set[str] = set()

    @property
    def client(self) -> AsyncAnthropic:
        """Lazily initialize the Anthropic client on first access."""
        if self._client is None:
            if self._api_key is None:
                raise ValueError("api_key must be provided for API calls")
            # Set SDK max_retries=0 - we handle retries ourselves to properly
            # honor retry-after headers with jitter and longer backoffs
            self._client = AsyncAnthropic(
                api_key=self._api_key,
                base_url=self._base_url,
                default_headers=self._default_headers,
                max_retries=0,
            )
        return self._client

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="anthropic",
            display_name="Anthropic",
            credential_env_vars=["ANTHROPIC_API_KEY"],
            capabilities=["streaming", "tools", "thinking", "batch"],
            defaults={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": 300.0,
                "context_window": 1000000
                if self.config.get("enable_1m_context")
                else 200000,
                "max_output_tokens": 64000,
            },
            config_fields=[
                ConfigField(
                    id="api_key",
                    display_name="API Key",
                    field_type="secret",
                    prompt="Enter your Anthropic API key",
                    env_var="ANTHROPIC_API_KEY",
                ),
                ConfigField(
                    id="base_url",
                    display_name="API Base URL",
                    field_type="text",
                    prompt="API base URL",
                    env_var="ANTHROPIC_BASE_URL",
                    required=False,
                    default="https://api.anthropic.com",
                ),
                ConfigField(
                    id="enable_1m_context",
                    display_name="1M Context Window",
                    field_type="boolean",
                    prompt="Enable 1M token context window? (Sonnet 4/4.5 only, sets beta header)",
                    required=False,
                    default="true",
                    requires_model=True,  # Shown after model selection
                    show_when={
                        "default_model": "contains:sonnet"
                    },  # Only show for Sonnet models
                ),
                ConfigField(
                    id="enable_prompt_caching",
                    display_name="Prompt Caching",
                    field_type="boolean",
                    prompt="Enable prompt caching? (Reduces cost by 90% on cached tokens)",
                    required=False,
                    default="true",
                ),
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """
        List available Claude models dynamically from Anthropic API.

        When filtered=True (default), returns only the latest version of each
        model family (opus, haiku, sonnet). When filtered=False, returns all
        available Claude models.

        Returns:
            List of ModelInfo for available Claude models.
        """
        response = await self.client.models.list()
        api_models = list(response.data)

        # Group models by family (opus, haiku, sonnet)
        families: dict[str, list[tuple[str, str, str]]] = {
            "opus": [],
            "haiku": [],
            "sonnet": [],
        }

        for model in api_models:
            model_id = model.id
            display_name = getattr(model, "display_name", model_id)

            # Determine family from model ID
            model_id_lower = model_id.lower()
            for family in families:
                if family in model_id_lower:
                    families[family].append(
                        (model_id, display_name, str(getattr(model, "created_at", "")))
                    )
                    break

        result: list[ModelInfo] = []

        for family, models in families.items():
            if not models:
                continue

            # Sort by model_id descending (IDs contain dates like claude-sonnet-4-5-20250929)
            models.sort(key=lambda x: x[0], reverse=True)

            # When filtered, only include the latest; otherwise include all
            models_to_include = [models[0]] if self.filtered else models

            for model_id, display_name, _ in models_to_include:
                # Determine capabilities based on family
                # Haiku is optimized for speed; Opus and Sonnet support extended thinking
                if family == "haiku":
                    capabilities = ["tools", "streaming", "json_mode", "fast"]
                else:
                    capabilities = ["tools", "thinking", "streaming", "json_mode"]

                result.append(
                    ModelInfo(
                        id=model_id,
                        display_name=display_name,
                        context_window=200000,  # All Claude models have 200K context
                        max_output_tokens=64000,  # All current Claude models support 64K output
                        capabilities=capabilities,
                        defaults={"temperature": 0.7, "max_tokens": 64000},
                    )
                )

        # Sort alphabetically by display name
        result.sort(key=lambda m: m.display_name.lower())

        return result

    def _truncate_values(self, obj: Any, max_length: int | None = None) -> Any:
        """Recursively truncate string values in nested structures.

        Preserves structure, only truncates leaf string values longer than max_length.
        Uses self.debug_truncate_length if max_length not specified.

        Args:
            obj: Any JSON-serializable structure (dict, list, primitives)
            max_length: Maximum string length (defaults to self.debug_truncate_length)

        Returns:
            Structure with truncated string values
        """
        if max_length is None:
            max_length = self.debug_truncate_length

        # Type guard: max_length is guaranteed to be int after this point
        assert max_length is not None, (
            "max_length should never be None after initialization"
        )

        if isinstance(obj, str):
            if len(obj) > max_length:
                return (
                    obj[:max_length] + f"... (truncated {len(obj) - max_length} chars)"
                )
            return obj
        if isinstance(obj, dict):
            return {k: self._truncate_values(v, max_length) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._truncate_values(item, max_length) for item in obj]
        return obj  # Numbers, booleans, None pass through unchanged

    def _find_missing_tool_results(
        self, messages: list[Message]
    ) -> list[tuple[int, str, str, dict]]:
        """Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing pairs WITH their
        source message index so they can be inserted in the correct position.

        Excludes tool call IDs that have already been repaired with synthetic
        results to prevent infinite detection loops.

        Returns:
            List of (msg_index, call_id, tool_name, tool_arguments) tuples for unpaired calls.
            msg_index is the index of the assistant message containing the tool_use block.
        """
        tool_calls = {}  # {call_id: (msg_index, name, args)}
        tool_results = set()  # {call_id}

        for idx, msg in enumerate(messages):
            # Check assistant messages for ToolCallBlock in content
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (idx, block.name, block.input)

            # Check tool messages for tool_call_id
            elif (
                msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id
            ):
                tool_results.add(msg.tool_call_id)

        # Exclude IDs that have already been repaired to prevent infinite loops
        return [
            (msg_idx, call_id, name, args)
            for call_id, (msg_idx, name, args) in tool_calls.items()
            if call_id not in tool_results and call_id not in self._repaired_tool_ids
        ]

    def _create_synthetic_result(self, call_id: str, tool_name: str) -> Message:
        """Create synthetic error result for missing tool response.

        This is a BACKUP for when tool results go missing AFTER execution.
        The orchestrator should handle tool execution errors at runtime,
        so this should only trigger on context/parsing bugs.
        """
        return Message(
            role="tool",
            content=(
                f"[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                f"This indicates the tool result was lost after execution.\n"
                f"Likely causes: context compaction bug, message parsing error, or state corruption.\n\n"
                f"The tool may have executed successfully, but the result was lost.\n"
                f"Please acknowledge this error and offer to retry the operation."
            ),
            tool_call_id=call_id,
            name=tool_name,
        )

    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """
        Generate completion from ChatRequest.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options (override request fields)

        Returns:
            ChatResponse with content blocks, tool calls, usage
        """
        # VALIDATE AND REPAIR: Check for missing tool results (backup safety net)
        missing = self._find_missing_tool_results(request.messages)

        if missing:
            logger.warning(
                f"[PROVIDER] Anthropic: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for _, call_id, _, _ in missing]}"
            )

            # Group missing results by source assistant message index
            # We need to insert synthetic results IMMEDIATELY after each assistant message
            # that contains tool_use blocks (not at the end of the list)
            from collections import defaultdict

            by_msg_idx: dict[int, list[tuple[str, str]]] = defaultdict(list)
            for msg_idx, call_id, tool_name, _ in missing:
                by_msg_idx[msg_idx].append((call_id, tool_name))

            # Insert synthetic results in reverse order of message index
            # (so earlier insertions don't shift later indices)
            for msg_idx in sorted(by_msg_idx.keys(), reverse=True):
                synthetics = []
                for call_id, tool_name in by_msg_idx[msg_idx]:
                    synthetics.append(self._create_synthetic_result(call_id, tool_name))
                    # Track this ID so we don't detect it as missing again in future iterations
                    self._repaired_tool_ids.add(call_id)

                # Insert all synthetic results immediately after the assistant message
                insert_pos = msg_idx + 1
                for i, synthetic in enumerate(synthetics):
                    request.messages.insert(insert_pos + i, synthetic)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for _, call_id, tool_name, _ in missing
                        ],
                    },
                )

        return await self._complete_chat_request(request, **kwargs)

    def _extract_rate_limit_headers(
        self, headers: dict[str, str] | Any
    ) -> dict[str, Any]:
        """Extract rate limit information from response headers.

        Anthropic returns rate limit headers on every response:
        - anthropic-ratelimit-requests-limit/remaining/reset
        - anthropic-ratelimit-tokens-limit/remaining/reset
        - retry-after (on 429 errors)

        Args:
            headers: Response headers (dict-like object)

        Returns:
            Dict with rate limit info, or empty dict if headers unavailable
        """
        if not headers:
            return {}

        # Helper to safely get header values
        def get_int(key: str) -> int | None:
            val = headers.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
            return None

        info: dict[str, Any] = {}

        # Request limits
        requests_remaining = get_int("anthropic-ratelimit-requests-remaining")
        requests_limit = get_int("anthropic-ratelimit-requests-limit")
        if requests_remaining is not None:
            info["requests_remaining"] = requests_remaining
        if requests_limit is not None:
            info["requests_limit"] = requests_limit

        # Token limits
        tokens_remaining = get_int("anthropic-ratelimit-tokens-remaining")
        tokens_limit = get_int("anthropic-ratelimit-tokens-limit")
        if tokens_remaining is not None:
            info["tokens_remaining"] = tokens_remaining
        if tokens_limit is not None:
            info["tokens_limit"] = tokens_limit

        # Retry-after (typically only on 429)
        if retry_after := headers.get("retry-after"):
            try:
                info["retry_after_seconds"] = float(retry_after)
            except (ValueError, TypeError):
                pass

        return info

    def _parse_rate_limit_info(self, error: RateLimitError) -> dict[str, Any]:
        """Extract rate limit details from RateLimitError.

        The SDK provides headers via error.response.headers when available.
        """
        info: dict[str, Any] = {
            "retry_after_seconds": None,
            "rate_limit_type": None,
        }

        # RateLimitError may have response with headers
        if hasattr(error, "response") and error.response:
            headers = getattr(error.response, "headers", {})

            # Parse retry-after (seconds as float)
            if retry_after := headers.get("retry-after"):
                try:
                    info["retry_after_seconds"] = float(retry_after)
                except (ValueError, TypeError):
                    pass

            # Determine limit type from remaining tokens
            tokens_remaining = headers.get("anthropic-ratelimit-tokens-remaining")
            requests_remaining = headers.get("anthropic-ratelimit-requests-remaining")

            if tokens_remaining == "0":
                info["rate_limit_type"] = "tokens"
            elif requests_remaining == "0":
                info["rate_limit_type"] = "requests"

        return info

    def _calculate_retry_delay(self, retry_after: float | None, attempt: int) -> float:
        """Calculate delay before next retry attempt.

        Uses retry-after header if available, otherwise exponential backoff.
        Applies jitter if enabled to spread load across time.

        Args:
            retry_after: Seconds from retry-after header (may be None)
            attempt: Current attempt number (1-based)

        Returns:
            Delay in seconds before next retry
        """
        if retry_after is not None and retry_after > 0:
            # Honor the retry-after header
            delay = retry_after
        else:
            # Exponential backoff: 1s, 2s, 4s, 8s, 16s, ...
            delay = self.min_retry_delay * (2 ** (attempt - 1))

        # Cap at max_retry_delay
        delay = min(delay, self.max_retry_delay)

        # Apply jitter (±20%) to spread load and avoid thundering herd
        if self.retry_jitter:
            jitter_factor = random.uniform(0.8, 1.2)
            delay = delay * jitter_factor

        return delay

    def _format_system_with_cache(
        self, system_msgs: list[Message]
    ) -> list[dict[str, Any]] | None:
        """Format system messages as content block array with cache_control.

        Anthropic requires system as array of content blocks for caching.
        Cache breakpoint goes on the LAST block.

        Returns:
            List of content blocks, or None if no system messages
        """
        if not system_msgs:
            return None

        # Combine into single text (preserves current behavior)
        combined = "\n\n".join(
            m.content if isinstance(m.content, str) else "" for m in system_msgs
        )

        if not combined:
            return None

        block: dict[str, Any] = {"type": "text", "text": combined}

        # Add cache_control if enabled
        if self.enable_prompt_caching:
            block["cache_control"] = {"type": "ephemeral"}

        return [block]

    async def _complete_chat_request(
        self, request: ChatRequest, **kwargs
    ) -> ChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        logger.debug(
            f"Received ChatRequest with {len(request.messages)} messages (debug={self.debug})"
        )

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [
            m for m in request.messages if m.role in ("user", "assistant", "tool")
        ]

        logger.debug(
            f"Separated: {len(system_msgs)} system, {len(developer_msgs)} developer, {len(conversation)} conversation"
        )

        # Format system messages as content block array (required for caching)
        system_blocks = self._format_system_with_cache(system_msgs)

        if system_blocks:
            logger.info(
                f"[PROVIDER] System message length: {len(system_blocks[0]['text'])} chars (caching={'cache_control' in system_blocks[0]})"
            )
        else:
            logger.info("[PROVIDER] No system messages")

        # Convert developer messages to XML-wrapped user messages (at top)
        context_user_msgs = []
        for i, dev_msg in enumerate(developer_msgs):
            content = dev_msg.content if isinstance(dev_msg.content, str) else ""
            content_preview = content[:100] + ("..." if len(content) > 100 else "")
            logger.info(
                f"[PROVIDER] Converting developer message {i + 1}/{len(developer_msgs)}: length={len(content)}"
            )
            logger.debug(f"[PROVIDER] Developer message preview: {content_preview}")
            wrapped = f"<context_file>\n{content}\n</context_file>"
            context_user_msgs.append({"role": "user", "content": wrapped})

        logger.info(
            f"[PROVIDER] Created {len(context_user_msgs)} XML-wrapped context messages"
        )

        # Convert conversation messages
        conversation_msgs = self._convert_messages(
            [m.model_dump() for m in conversation]
        )
        logger.info(
            f"[PROVIDER] Converted {len(conversation_msgs)} conversation messages"
        )

        # Combine: context THEN conversation
        all_messages = context_user_msgs + conversation_msgs
        # Apply cache control to last message for incremental context caching
        all_messages = self._apply_message_cache_control(all_messages)
        logger.info(f"[PROVIDER] Final message count for API: {len(all_messages)}")

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

        # Enable extended thinking if requested (equivalent to OpenAI's reasoning)
        thinking_enabled = bool(kwargs.get("extended_thinking"))
        thinking_budget = None
        interleaved_thinking_enabled = False
        if thinking_enabled:
            budget_tokens = (
                kwargs.get("thinking_budget_tokens")
                or self.config.get("thinking_budget_tokens")
                or 32000
            )
            buffer_tokens = kwargs.get("thinking_budget_buffer") or self.config.get(
                "thinking_budget_buffer", 4096
            )

            thinking_budget = budget_tokens
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }

            # CRITICAL: Anthropic requires temperature=1.0 when thinking is enabled
            params["temperature"] = 1.0

            # Ensure max_tokens accommodates thinking budget + response
            target_tokens = budget_tokens + buffer_tokens
            if params.get("max_tokens"):
                params["max_tokens"] = max(params["max_tokens"], target_tokens)
            else:
                params["max_tokens"] = target_tokens

            # Auto-enable interleaved thinking when extended thinking is enabled.
            # Interleaved thinking allows Claude 4 models to think between tool calls,
            # producing better reasoning on complex multi-step tasks.
            # Uses the beta header: interleaved-thinking-2025-05-14
            #
            # IMPORTANT: We must merge with the instance's configured beta headers
            # (e.g., context-1m-2025-08-07 for 1M context window). The extra_headers
            # in params will override the client's default_headers for the same key,
            # so we need to include ALL beta headers in the combined value.
            interleaved_thinking_enabled = True
            combined_beta_headers = list(
                self._beta_headers
            )  # Start with configured headers
            if "interleaved-thinking-2025-05-14" not in combined_beta_headers:
                combined_beta_headers.append("interleaved-thinking-2025-05-14")
            params["extra_headers"] = {
                "anthropic-beta": ",".join(combined_beta_headers)
            }

            logger.info(
                "[PROVIDER] Extended thinking enabled (budget=%s, buffer=%s, temperature=1.0, max_tokens=%s, interleaved=%s)",
                thinking_budget,
                buffer_tokens,
                params["max_tokens"],
                interleaved_thinking_enabled,
            )

        # Add stop_sequences if specified
        if stop_sequences := kwargs.get("stop_sequences"):
            params["stop_sequences"] = stop_sequences

        logger.info(
            f"[PROVIDER] Anthropic API call - model: {params['model']}, messages: {len(params['messages'])}, system: {bool(system_blocks)}, tools: {len(params.get('tools', []))}, thinking: {thinking_enabled}"
        )

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": "anthropic",
                    "model": params["model"],
                    "message_count": len(params["messages"]),
                    "has_system": bool(system_blocks),
                    "thinking_enabled": thinking_enabled,
                    "thinking_budget": thinking_budget,
                    "interleaved_thinking": interleaved_thinking_enabled,
                },
            )

            # DEBUG level: Full request payload with truncated values (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "provider": "anthropic",
                        "request": self._truncate_values(params),
                    },
                )

            # RAW level: Complete params dict as sent to Anthropic API (if debug AND raw_debug enabled)
            if self.debug and self.raw_debug:
                await self.coordinator.hooks.emit(
                    "llm:request:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": "anthropic",
                        "params": params,  # Complete untruncated params
                    },
                )

        start_time = time.time()

        # Call Anthropic API with retry loop for rate limits
        # We handle retries ourselves (SDK max_retries=0) to properly honor
        # retry-after headers with jitter and longer backoffs
        last_rate_limit_error: RateLimitError | None = None

        for attempt in range(
            1, self.max_retries + 2
        ):  # +2 because range is exclusive and attempt 1 is initial try
            try:
                # Use streaming API to support large context windows (Anthropic requires streaming
                # for operations that may take > 10 minutes)
                rate_limit_info: dict[str, Any] = {}
                if self.use_streaming:
                    async with asyncio.timeout(self.timeout):
                        async with self.client.messages.stream(**params) as stream:
                            response = await stream.get_final_message()
                            # Capture rate limit headers from stream response
                            if hasattr(stream, "response") and stream.response:
                                rate_limit_info = self._extract_rate_limit_headers(
                                    stream.response.headers
                                )
                else:
                    # Use with_raw_response to access headers
                    raw_response = await asyncio.wait_for(
                        self.client.messages.with_raw_response.create(**params),
                        timeout=self.timeout,
                    )
                    response = raw_response.parse()
                    rate_limit_info = self._extract_rate_limit_headers(
                        raw_response.headers
                    )

                # Success - break out of retry loop
                break

            except RateLimitError as e:
                last_rate_limit_error = e
                rate_info = self._parse_rate_limit_info(e)
                retry_after = rate_info["retry_after_seconds"]

                # Check if we have retries remaining
                if attempt <= self.max_retries:
                    delay = self._calculate_retry_delay(retry_after, attempt)

                    logger.info(
                        f"[PROVIDER] Rate limited (attempt {attempt}/{self.max_retries + 1}). "
                        f"Waiting {delay:.1f}s before retry..."
                    )

                    # Emit retry event for observability
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        await self.coordinator.hooks.emit(
                            "anthropic:rate_limit_retry",
                            {
                                "provider": "anthropic",
                                "model": params["model"],
                                "attempt": attempt,
                                "max_retries": self.max_retries,
                                "retry_after_header": retry_after,
                                "actual_delay": delay,
                                "rate_limit_type": rate_info["rate_limit_type"],
                            },
                        )

                    # Wait before retry
                    await asyncio.sleep(delay)
                    continue

                # No retries remaining - will be handled after loop
                break

        else:
            # This else belongs to the for loop - executes if loop completed without break
            # This shouldn't happen given our logic, but handle it gracefully
            pass

        # Check if we exited due to rate limit exhaustion
        if last_rate_limit_error is not None and attempt > self.max_retries:
            elapsed_ms = int((time.time() - start_time) * 1000)
            rate_info = self._parse_rate_limit_info(last_rate_limit_error)
            retry_after = rate_info["retry_after_seconds"]

            # Build clean, actionable error message
            error_msg = (
                f"Rate limited by Anthropic API after {self.max_retries} retries."
            )
            if retry_after:
                error_msg += f" (retry-after: {retry_after}s)"

            # Note: We don't log here - the error message will be displayed by the CLI
            # when it catches the exception. Logging would cause duplicate output.

            # Emit rate limit exhausted event for observability
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "anthropic:rate_limited",
                    {
                        "provider": "anthropic",
                        "model": params["model"],
                        "retry_after_seconds": retry_after,
                        "retries_attempted": self.max_retries,
                        "error_message": str(last_rate_limit_error),
                        "rate_limit_type": rate_info["rate_limit_type"],
                    },
                )

                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "anthropic",
                        "model": params["model"],
                        "status": "rate_limited",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                    },
                )

            # Raise with clean message (original exception as cause for debugging)
            raise RuntimeError(error_msg) from last_rate_limit_error

        # If we get here, request succeeded - continue with response handling
        try:
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from Anthropic API")
            logger.debug(f"[PROVIDER] Response type: {response.model}")

            # Log rate limit status if available
            if rate_limit_info:
                tokens_remaining = rate_limit_info.get("tokens_remaining")
                tokens_limit = rate_limit_info.get("tokens_limit")
                if tokens_remaining is not None and tokens_limit is not None:
                    pct_used = (
                        ((tokens_limit - tokens_remaining) / tokens_limit) * 100
                        if tokens_limit > 0
                        else 0
                    )
                    logger.debug(
                        f"[PROVIDER] Rate limit: {tokens_remaining:,}/{tokens_limit:,} tokens remaining ({pct_used:.1f}% used)"
                    )

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary with rate limit info
                response_event: dict[str, Any] = {
                    "provider": "anthropic",
                    "model": params["model"],
                    "usage": {
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                        **(
                            {"cache_read": response.usage.cache_read_input_tokens}
                            if hasattr(response.usage, "cache_read_input_tokens")
                            and response.usage.cache_read_input_tokens
                            else {}
                        ),
                        **(
                            {"cache_write": response.usage.cache_creation_input_tokens}
                            if hasattr(response.usage, "cache_creation_input_tokens")
                            and response.usage.cache_creation_input_tokens
                            else {}
                        ),
                    },
                    "status": "ok",
                    "duration_ms": elapsed_ms,
                }
                # Add rate limit info if available
                if rate_limit_info:
                    response_event["rate_limits"] = rate_limit_info

                await self.coordinator.hooks.emit("llm:response", response_event)

                # DEBUG level: Full response with truncated values (if debug enabled)
                if self.debug:
                    response_dict = response.model_dump()  # Pydantic model → dict
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "provider": "anthropic",
                            "response": self._truncate_values(response_dict),
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

                # RAW level: Complete response object from Anthropic API (if debug AND raw_debug enabled)
                if self.debug and self.raw_debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:raw",
                        {
                            "lvl": "DEBUG",
                            "provider": "anthropic",
                            "response": response.model_dump(),  # Complete untruncated response
                        },
                    )

            # Convert to ChatResponse
            return self._convert_to_chat_response(response)

        except TimeoutError:
            # Handle timeout specifically - TimeoutError has empty str() representation
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Request timed out after {self.timeout}s"
            logger.error(f"[PROVIDER] Anthropic API error: {error_msg}")

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "anthropic",
                        "model": params["model"],
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                    },
                )
            raise TimeoutError(error_msg) from None

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            # Ensure error message is never empty
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error(f"[PROVIDER] Anthropic API error: {error_msg}")

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "anthropic",
                        "model": params["model"],
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": error_msg,
                    },
                )
            # Re-raise with meaningful message if original was empty
            if not str(e):
                raise type(e)(error_msg) from e
            raise

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """
        Parse tool calls from ChatResponse.

        Filters out tool calls with empty/missing arguments to handle
        Anthropic API quirk where empty tool_use blocks are sometimes generated.

        Args:
            response: Typed chat response

        Returns:
            List of valid tool calls (with non-empty arguments)
        """
        if not response.tool_calls:
            return []

        # Filter out tool calls with empty arguments (Anthropic API quirk)
        # Claude sometimes generates tool_use blocks with empty input {}
        valid_calls = []
        for tc in response.tool_calls:
            # Skip tool calls with no arguments or empty dict
            if not tc.arguments:
                logger.debug(f"Filtering out tool '{tc.name}' with empty arguments")
                continue
            valid_calls.append(tc)

        if len(valid_calls) < len(response.tool_calls):
            logger.info(
                f"Filtered {len(response.tool_calls) - len(valid_calls)} tool calls with empty arguments"
            )

        return valid_calls

    def _clean_content_block(self, block: dict[str, Any]) -> dict[str, Any]:
        """Clean a content block for API by removing fields not accepted by Anthropic API.

        Anthropic API may include extra fields (like 'visibility') in responses,
        but does NOT accept these fields when blocks are sent as input in messages.

        Args:
            block: Raw content block dict (may include visibility, etc.)

        Returns:
            Cleaned content block dict with only API-accepted fields
        """
        block_type = block.get("type")

        if block_type == "text":
            return {"type": "text", "text": block.get("text", "")}
        if block_type == "thinking":
            cleaned = {"type": "thinking", "thinking": block.get("thinking", "")}
            if "signature" in block:
                cleaned["signature"] = block["signature"]
            return cleaned
        if block_type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            }
        if block_type == "tool_result":
            return {
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id", ""),
                "content": block.get("content", ""),
            }
        # Unknown block type - return as-is but remove visibility
        cleaned = dict(block)
        cleaned.pop("visibility", None)
        return cleaned

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Anthropic format.

        CRITICAL: Anthropic requires ALL tool_result blocks from one assistant's tool_use
        to be batched into a SINGLE user message with multiple tool_result blocks in the
        content array. We cannot send separate user messages for each tool result.

        This method batches consecutive tool messages into one user message.

        DEFENSIVE: Also validates that each tool_result has a corresponding tool_use
        in a preceding assistant message. Orphaned tool_results (from context compaction)
        are skipped to avoid API errors.
        """
        # First pass: collect all valid tool_use_ids from assistant messages
        valid_tool_use_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    tc_id = tc.get("id") or tc.get("tool_call_id")
                    if tc_id:
                        valid_tool_use_ids.add(tc_id)

        anthropic_messages = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content", "")

            # Skip system messages (handled separately)
            if role == "system":
                i += 1
                continue

            # Batch consecutive tool messages into ONE user message
            if role == "tool":
                # Collect all consecutive tool results, but only valid ones
                tool_results = []
                skipped_count = 0
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_msg = messages[i]
                    tool_use_id = tool_msg.get("tool_call_id")

                    # DEFENSIVE: Skip tool_results without valid tool_use_id
                    # This prevents API errors from orphaned tool_results after compaction
                    if not tool_use_id or tool_use_id not in valid_tool_use_ids:
                        logger.warning(
                            f"Skipping orphaned tool_result (no matching tool_use): "
                            f"tool_call_id={tool_use_id}, content_preview={str(tool_msg.get('content', ''))[:100]}"
                        )
                        skipped_count += 1
                        i += 1
                        continue

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_msg.get("content", ""),
                        }
                    )
                    i += 1

                # Only add user message if we have valid tool_results
                if tool_results:
                    anthropic_messages.append(
                        {
                            "role": "user",
                            "content": tool_results,  # Array of tool_result blocks
                        }
                    )
                elif skipped_count > 0:
                    logger.warning(
                        f"All {skipped_count} consecutive tool_results were orphaned and skipped"
                    )
                continue  # i already advanced in while loop
            if role == "assistant":
                # Assistant messages - check for tool calls or thinking blocks
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Assistant message with tool calls
                    content_blocks = []

                    # CRITICAL: Check for thinking block and add it FIRST
                    has_thinking = "thinking_block" in msg and msg["thinking_block"]
                    if has_thinking:
                        # Clean thinking block (remove visibility field not accepted by API)
                        cleaned_thinking = self._clean_content_block(
                            msg["thinking_block"]
                        )
                        content_blocks.append(cleaned_thinking)

                    # Add text content if present, BUT skip when we have thinking + tool_calls
                    # When all three are present (thinking + text + tool_use), the text was generated
                    # but not shown to user yet (tool calls execute first). Including it in history
                    # misleads the model into thinking it already communicated that info.
                    if content and not has_thinking:
                        if isinstance(content, list):
                            # Content is a list of blocks - extract text blocks only
                            for block in content:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    content_blocks.append(
                                        {"type": "text", "text": block.get("text", "")}
                                    )
                                elif (
                                    not isinstance(block, dict)
                                    and hasattr(block, "type")
                                    and block.type == "text"
                                ):
                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "text": getattr(block, "text", ""),
                                        }
                                    )
                        else:
                            # Content is a simple string
                            content_blocks.append({"type": "text", "text": content})

                    # Add tool_use blocks
                    for tc in msg["tool_calls"]:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc.get("tool", ""),
                                "input": tc.get("arguments", {}),
                            }
                        )

                    anthropic_messages.append(
                        {"role": "assistant", "content": content_blocks}
                    )
                elif "thinking_block" in msg and msg["thinking_block"]:
                    # Assistant message with thinking block
                    # Clean thinking block (remove visibility field not accepted by API)
                    cleaned_thinking = self._clean_content_block(msg["thinking_block"])
                    content_blocks = [cleaned_thinking]
                    if content:
                        if isinstance(content, list):
                            # Content is a list of blocks - extract text blocks only
                            for block in content:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    content_blocks.append(
                                        {"type": "text", "text": block.get("text", "")}
                                    )
                                elif (
                                    not isinstance(block, dict)
                                    and hasattr(block, "type")
                                    and block.type == "text"
                                ):
                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "text": getattr(block, "text", ""),
                                        }
                                    )
                        else:
                            # Content is a simple string
                            content_blocks.append({"type": "text", "text": content})
                    anthropic_messages.append(
                        {"role": "assistant", "content": content_blocks}
                    )
                else:
                    # Regular assistant message - may have structured content blocks
                    if isinstance(content, list):
                        # Content is a list of blocks - clean each block
                        cleaned_blocks = [
                            self._clean_content_block(block) for block in content
                        ]
                        anthropic_messages.append(
                            {"role": "assistant", "content": cleaned_blocks}
                        )
                    else:
                        # Content is a simple string
                        anthropic_messages.append(
                            {"role": "assistant", "content": content}
                        )
                i += 1
            elif role == "developer":
                # Developer messages -> XML-wrapped user messages (context files)
                wrapped = f"<context_file>\n{content}\n</context_file>"
                anthropic_messages.append({"role": "user", "content": wrapped})
                i += 1
            else:
                # User messages - handle structured content (text + images)
                if isinstance(content, list):
                    content_blocks = []
                    for block in content:
                        if isinstance(block, dict):
                            block_type = block.get("type")
                            if block_type == "text":
                                content_blocks.append(
                                    {"type": "text", "text": block.get("text", "")}
                                )
                            elif block_type == "image":
                                # Convert ImageBlock to Anthropic image format
                                source = block.get("source", {})
                                if source.get("type") == "base64":
                                    content_blocks.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": source.get(
                                                    "media_type", "image/jpeg"
                                                ),
                                                "data": source.get("data"),
                                            },
                                        }
                                    )
                                else:
                                    logger.warning(
                                        f"Unsupported image source type: {source.get('type')}"
                                    )

                    if content_blocks:
                        anthropic_messages.append(
                            {"role": "user", "content": content_blocks}
                        )
                else:
                    # Simple string content
                    anthropic_messages.append({"role": "user", "content": content})
                i += 1

        return anthropic_messages

    def _convert_tools_from_request(self, tools: list) -> list[dict[str, Any]]:
        """Convert ToolSpec objects from ChatRequest to Anthropic format.

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of Anthropic-formatted tool definitions
        """
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.parameters,
                }
            )
        return anthropic_tools

    def _apply_tool_cache_control(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add cache_control to the last tool definition.

        Per Anthropic spec: cache breakpoint on last tool creates
        checkpoint for entire tool list.

        Args:
            tools: List of Anthropic-formatted tool definitions

        Returns:
            Same list with cache_control on last tool (if caching enabled)
        """
        if not tools or not self.enable_prompt_caching:
            return tools

        # Add cache_control to last tool
        tools[-1]["cache_control"] = {"type": "ephemeral"}
        return tools

    def _apply_message_cache_control(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add cache_control to last content block of last message.

        Per Anthropic spec: this creates a checkpoint at the end of
        conversation history, caching the full context.

        Args:
            messages: Anthropic-formatted message list

        Returns:
            Same list with cache_control on last message's last block
        """
        if not messages or not self.enable_prompt_caching:
            return messages

        last_msg = messages[-1]
        content = last_msg.get("content")

        # Handle different content formats
        if isinstance(content, list) and content:
            # Array of content blocks - mark last block
            content[-1]["cache_control"] = {"type": "ephemeral"}
        elif isinstance(content, str):
            # String content - convert to block array with cache marker
            last_msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        return messages

    def _convert_to_chat_response(self, response: Any) -> ChatResponse:
        """Convert Anthropic response to ChatResponse format.

        Args:
            response: Anthropic API response

        Returns:
            AnthropicChatResponse with content blocks and streaming-compatible fields
        """
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import ThinkingBlock
        from amplifier_core.message_models import ToolCall
        from amplifier_core.message_models import ToolCallBlock
        from amplifier_core.message_models import Usage

        content_blocks = []
        tool_calls = []
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []
        text_accumulator: list[str] = []

        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextBlock(text=block.text))
                text_accumulator.append(block.text)
                event_blocks.append(TextContent(text=block.text))
            elif block.type == "thinking":
                content_blocks.append(
                    ThinkingBlock(
                        thinking=block.thinking,
                        signature=getattr(block, "signature", None),
                        visibility="internal",
                    )
                )
                event_blocks.append(ThinkingContent(text=block.thinking))
                # NOTE: Do NOT add thinking to text_accumulator - it's internal process, not response content
            elif block.type == "tool_use":
                content_blocks.append(
                    ToolCallBlock(id=block.id, name=block.name, input=block.input)
                )
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )
                event_blocks.append(
                    ToolCallContent(id=block.id, name=block.name, arguments=block.input)
                )

        # Build usage dict with cache metrics if available
        usage_kwargs: dict[str, Any] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        # Add cache metrics if available (Anthropic includes these when caching is active)
        if (
            hasattr(response.usage, "cache_creation_input_tokens")
            and response.usage.cache_creation_input_tokens
        ):
            usage_kwargs["cache_creation_input_tokens"] = (
                response.usage.cache_creation_input_tokens
            )
        if (
            hasattr(response.usage, "cache_read_input_tokens")
            and response.usage.cache_read_input_tokens
        ):
            usage_kwargs["cache_read_input_tokens"] = (
                response.usage.cache_read_input_tokens
            )

        usage = Usage(**usage_kwargs)

        combined_text = "\n\n".join(text_accumulator).strip()

        return AnthropicChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=response.stop_reason,
            content_blocks=event_blocks if event_blocks else None,
            text=combined_text or None,
        )
