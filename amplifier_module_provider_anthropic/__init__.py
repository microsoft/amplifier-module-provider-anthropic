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
import time
from typing import Any
from typing import Optional

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

    def __init__(
        self, api_key: str, config: dict[str, Any] | None = None, coordinator: ModuleCoordinator | None = None
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            config: Additional configuration
            coordinator: Module coordinator for event emission
        """
        self.config = config or {}
        self.coordinator = coordinator
        self.default_model = self.config.get("default_model", "claude-sonnet-4-5")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.7)
        self.priority = self.config.get("priority", 100)  # Store priority for selection
        self.debug = self.config.get("debug", False)  # Enable full request/response logging
        self.raw_debug = self.config.get("raw_debug", False)  # Enable ultra-verbose raw API I/O logging
        self.debug_truncate_length = self.config.get("debug_truncate_length", 180)  # Max string length in debug logs
        self.timeout = self.config.get("timeout", 300.0)  # API timeout in seconds (default 5 minutes)

        # Get base_url from config for custom endpoints (proxies, local APIs, etc.)
        base_url = self.config.get("base_url")

        # Beta headers support for enabling experimental features
        beta_headers_config = self.config.get("beta_headers")
        default_headers = None
        if beta_headers_config:
            # Normalize to list (supports string or list of strings)
            beta_headers_list = [beta_headers_config] if isinstance(beta_headers_config, str) else beta_headers_config
            # Build anthropic-beta header value (comma-separated)
            beta_header_value = ",".join(beta_headers_list)
            default_headers = {"anthropic-beta": beta_header_value}
            logger.info(f"[PROVIDER] Beta headers enabled: {beta_header_value}")

        # Initialize client with optional beta headers and base_url
        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url, default_headers=default_headers)

        # Track tool call IDs that have been repaired with synthetic results.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations (since synthetic results
        # are injected into request.messages but not persisted to message store).
        self._repaired_tool_ids: set[str] = set()

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="anthropic",
            display_name="Anthropic",
            credential_env_vars=["ANTHROPIC_API_KEY"],
            capabilities=["streaming", "tools", "vision", "thinking", "batch"],
            defaults={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": 300.0,
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
                    prompt="Enable 1M token context window? (sets beta header: context-1m-2025-08-07)",
                    required=False,
                    default="true",
                    requires_model=True,  # Shown after model selection
                    show_when={"default_model": "claude-sonnet-4-5-20250929"},
                ),
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """
        List available Claude models.

        Returns hardcoded list of current Claude models since Anthropic doesn't
        provide a model listing API.
        """
        return [
            ModelInfo(
                id="claude-sonnet-4-5-20250929",
                display_name="Claude Sonnet 4.5",
                context_window=200000,
                max_output_tokens=16000,
                capabilities=["tools", "vision", "thinking", "streaming", "json_mode"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="claude-haiku-4-5-20251001",
                display_name="Claude Haiku 4.5",
                context_window=200000,
                max_output_tokens=8192,
                capabilities=["tools", "vision", "streaming", "json_mode", "fast"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="claude-opus-4-5-20251101",
                display_name="Claude Opus 4.5",
                context_window=200000,
                max_output_tokens=32000,
                capabilities=["tools", "vision", "thinking", "streaming", "json_mode"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
            ModelInfo(
                id="claude-opus-4-1-20250805",
                display_name="Claude Opus 4.1",
                context_window=200000,
                max_output_tokens=32000,
                capabilities=["tools", "vision", "thinking", "streaming", "json_mode"],
                defaults={"temperature": 0.7, "max_tokens": 4096},
            ),
        ]

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
        assert max_length is not None, "max_length should never be None after initialization"

        if isinstance(obj, str):
            if len(obj) > max_length:
                return obj[:max_length] + f"... (truncated {len(obj) - max_length} chars)"
            return obj
        if isinstance(obj, dict):
            return {k: self._truncate_values(v, max_length) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._truncate_values(item, max_length) for item in obj]
        return obj  # Numbers, booleans, None pass through unchanged

    def _find_missing_tool_results(self, messages: list[Message]) -> list[tuple[str, str, dict]]:
        """Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing pairs.

        Returns:
            List of (call_id, tool_name, tool_arguments) tuples for unpaired calls
        """
        tool_calls = {}  # {call_id: (name, args)}
        tool_results = set()  # {call_id}

        for msg in messages:
            # Check assistant messages for ToolCallBlock in content
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (block.name, block.input)

            # Check tool messages for tool_call_id
            elif msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id:
                tool_results.add(msg.tool_call_id)

        return [
            (call_id, name, args)
            for call_id, (name, args) in tool_calls.items()
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
                f"Tool IDs: {[call_id for call_id, _, _ in missing]}"
            )

            # Inject synthetic results and track repaired IDs to prevent infinite loops
            for call_id, tool_name, _ in missing:
                synthetic = self._create_synthetic_result(call_id, tool_name)
                request.messages.append(synthetic)
                self._repaired_tool_ids.add(call_id)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name} for call_id, tool_name, _ in missing
                        ],
                    },
                )

        return await self._complete_chat_request(request, **kwargs)

    async def _complete_chat_request(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        logger.debug(f"Received ChatRequest with {len(request.messages)} messages (debug={self.debug})")

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [m for m in request.messages if m.role in ("user", "assistant", "tool")]

        logger.debug(
            f"Separated: {len(system_msgs)} system, {len(developer_msgs)} developer, {len(conversation)} conversation"
        )

        # Combine system messages
        system = (
            "\n\n".join(m.content if isinstance(m.content, str) else "" for m in system_msgs) if system_msgs else None
        )

        if system:
            logger.info(f"[PROVIDER] Combined system message length: {len(system)}")
        else:
            logger.info("[PROVIDER] No system messages")

        # Convert developer messages to XML-wrapped user messages (at top)
        context_user_msgs = []
        for i, dev_msg in enumerate(developer_msgs):
            content = dev_msg.content if isinstance(dev_msg.content, str) else ""
            content_preview = content[:100] + ("..." if len(content) > 100 else "")
            logger.info(f"[PROVIDER] Converting developer message {i + 1}/{len(developer_msgs)}: length={len(content)}")
            logger.debug(f"[PROVIDER] Developer message preview: {content_preview}")
            wrapped = f"<context_file>\n{content}\n</context_file>"
            context_user_msgs.append({"role": "user", "content": wrapped})

        logger.info(f"[PROVIDER] Created {len(context_user_msgs)} XML-wrapped context messages")

        # Convert conversation messages
        conversation_msgs = self._convert_messages([m.model_dump() for m in conversation])
        logger.info(f"[PROVIDER] Converted {len(conversation_msgs)} conversation messages")

        # Combine: context THEN conversation
        all_messages = context_user_msgs + conversation_msgs
        logger.info(f"[PROVIDER] Final message count for API: {len(all_messages)}")

        # Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "messages": all_messages,
            "max_tokens": request.max_output_tokens or kwargs.get("max_tokens", self.max_tokens),
            "temperature": request.temperature or kwargs.get("temperature", self.temperature),
        }

        if system:
            params["system"] = system

        # Add tools if provided
        if request.tools:
            params["tools"] = self._convert_tools_from_request(request.tools)
            # Add tool_choice if specified
            if tool_choice := kwargs.get("tool_choice"):
                params["tool_choice"] = tool_choice

        # Enable extended thinking if requested (equivalent to OpenAI's reasoning)
        thinking_enabled = bool(kwargs.get("extended_thinking"))
        thinking_budget = None
        if thinking_enabled:
            budget_tokens = kwargs.get("thinking_budget_tokens") or self.config.get("thinking_budget_tokens") or 10000
            buffer_tokens = kwargs.get("thinking_budget_buffer") or self.config.get("thinking_budget_buffer", 4096)

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

            logger.info(
                "[PROVIDER] Extended thinking enabled (budget=%s, buffer=%s, temperature=1.0, max_tokens=%s)",
                thinking_budget,
                buffer_tokens,
                params["max_tokens"],
            )

        # Add stop_sequences if specified
        if stop_sequences := kwargs.get("stop_sequences"):
            params["stop_sequences"] = stop_sequences

        logger.info(
            f"[PROVIDER] Anthropic API call - model: {params['model']}, messages: {len(params['messages'])}, system: {bool(system)}, tools: {len(params.get('tools', []))}, thinking: {thinking_enabled}"
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
                    "has_system": bool(system),
                    "thinking_enabled": thinking_enabled,
                    "thinking_budget": thinking_budget,
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

        # Call Anthropic API
        try:
            response = await asyncio.wait_for(self.client.messages.create(**params), timeout=self.timeout)
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from Anthropic API")
            logger.debug(f"[PROVIDER] Response type: {response.model}")

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "anthropic",
                        "model": params["model"],
                        "usage": {
                            "input": response.usage.input_tokens,
                            "output": response.usage.output_tokens,
                        },
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

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

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[PROVIDER] Anthropic API error: {e}")

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "anthropic",
                        "model": params["model"],
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                    },
                )
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
            logger.info(f"Filtered {len(response.tool_calls) - len(valid_calls)} tool calls with empty arguments")

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
        """
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
                # Collect all consecutive tool results
                tool_results = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_msg = messages[i]
                    tool_use_id = tool_msg.get("tool_call_id")
                    if not tool_use_id:
                        logger.warning(f"Tool result missing tool_call_id: {tool_msg}")
                        tool_use_id = "unknown"  # Fallback

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_msg.get("content", ""),
                        }
                    )
                    i += 1

                # Add ONE user message with ALL tool results
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": tool_results,  # Array of tool_result blocks
                    }
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
                        cleaned_thinking = self._clean_content_block(msg["thinking_block"])
                        content_blocks.append(cleaned_thinking)

                    # Add text content if present, BUT skip when we have thinking + tool_calls
                    # When all three are present (thinking + text + tool_use), the text was generated
                    # but not shown to user yet (tool calls execute first). Including it in history
                    # misleads the model into thinking it already communicated that info.
                    if content and not has_thinking:
                        if isinstance(content, list):
                            # Content is a list of blocks - extract text blocks only
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    content_blocks.append({"type": "text", "text": block.get("text", "")})
                                elif not isinstance(block, dict) and hasattr(block, "type") and block.type == "text":
                                    content_blocks.append({"type": "text", "text": getattr(block, "text", "")})
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

                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                elif "thinking_block" in msg and msg["thinking_block"]:
                    # Assistant message with thinking block
                    # Clean thinking block (remove visibility field not accepted by API)
                    cleaned_thinking = self._clean_content_block(msg["thinking_block"])
                    content_blocks = [cleaned_thinking]
                    if content:
                        if isinstance(content, list):
                            # Content is a list of blocks - extract text blocks only
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    content_blocks.append({"type": "text", "text": block.get("text", "")})
                                elif not isinstance(block, dict) and hasattr(block, "type") and block.type == "text":
                                    content_blocks.append({"type": "text", "text": getattr(block, "text", "")})
                        else:
                            # Content is a simple string
                            content_blocks.append({"type": "text", "text": content})
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    # Regular assistant message - may have structured content blocks
                    if isinstance(content, list):
                        # Content is a list of blocks - clean each block
                        cleaned_blocks = [self._clean_content_block(block) for block in content]
                        anthropic_messages.append({"role": "assistant", "content": cleaned_blocks})
                    else:
                        # Content is a simple string
                        anthropic_messages.append({"role": "assistant", "content": content})
                i += 1
            elif role == "developer":
                # Developer messages -> XML-wrapped user messages (context files)
                wrapped = f"<context_file>\n{content}\n</context_file>"
                anthropic_messages.append({"role": "user", "content": wrapped})
                i += 1
            else:
                # User messages
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
                content_blocks.append(ToolCallBlock(id=block.id, name=block.name, input=block.input))
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
                event_blocks.append(ToolCallContent(id=block.id, name=block.name, arguments=block.input))

        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        combined_text = "\n\n".join(text_accumulator).strip()

        return AnthropicChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=response.stop_reason,
            content_blocks=event_blocks if event_blocks else None,
            text=combined_text or None,
        )
