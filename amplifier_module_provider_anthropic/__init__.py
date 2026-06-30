"""Anthropic provider module for Amplifier.

Integrates with Anthropic's Claude API for Claude models (Sonnet, Opus, Haiku).
Supports streaming, tool calling, extended thinking, and ChatRequest format.
"""

__all__ = ["mount", "AnthropicProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import json
import logging
import os
import re
import time
import uuid
from decimal import Decimal
from threading import Lock
from typing import Any

from dataclasses import dataclass
from dataclasses import field

from amplifier_core import ConfigField
from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from amplifier_core import TextContent
from amplifier_core import ThinkingContent
from amplifier_core import ToolCallContent
from amplifier_core.events import PROVIDER_RETRY, PROVIDER_THROTTLE
from amplifier_core.llm_errors import AccessDeniedError as KernelAccessDeniedError
from amplifier_core.llm_errors import AuthenticationError as KernelAuthenticationError
from amplifier_core.llm_errors import ContentFilterError as KernelContentFilterError
from amplifier_core.llm_errors import ContextLengthError as KernelContextLengthError
from amplifier_core.llm_errors import InvalidRequestError as KernelInvalidRequestError
from amplifier_core.llm_errors import LLMError as KernelLLMError
from amplifier_core.llm_errors import LLMTimeoutError as KernelLLMTimeoutError
from amplifier_core.llm_errors import NotFoundError as KernelNotFoundError
from amplifier_core.llm_errors import (
    ProviderUnavailableError as KernelProviderUnavailableError,
)
from amplifier_core.llm_errors import RateLimitError as KernelRateLimitError
from amplifier_core.utils import redact_secrets
from amplifier_core.utils.retry import RetryConfig, retry_with_backoff
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolCall
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AsyncAnthropic
from anthropic import AuthenticationError as AnthropicAuthenticationError
from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic import RateLimitError as AnthropicRateLimitError
from anthropic._exceptions import (
    OverloadedError as AnthropicOverloadedError,
)  # Not exported in public API as of SDK v0.96.0 (private import still works)

from ._cost import compute_cost


@dataclass
class WebSearchContent:
    """Content block for web search results from native Anthropic web search."""

    type: str = "web_search"
    query: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, str]] = field(default_factory=list)


@dataclass
class _RateLimitState:
    """Tracks rate limit capacity from response headers for pre-emptive throttling.

    Internal to the provider — not exported, not in core.
    Updated after every successful API call. Resets when the provider is created.
    """

    # Requests dimension
    requests_remaining: int | None = None
    requests_limit: int | None = None
    requests_reset: str | None = None

    # Input tokens dimension
    input_tokens_remaining: int | None = None
    input_tokens_limit: int | None = None
    input_tokens_reset: str | None = None

    # Output tokens dimension
    output_tokens_remaining: int | None = None
    output_tokens_limit: int | None = None
    output_tokens_reset: str | None = None

    # Fast-mode token dimensions (present only when fast-mode is active)
    fast_input_tokens_remaining: int | None = None
    fast_input_tokens_limit: int | None = None
    fast_input_tokens_reset: str | None = None
    fast_output_tokens_remaining: int | None = None
    fast_output_tokens_limit: int | None = None
    fast_output_tokens_reset: str | None = None

    def update_from_headers(self, rate_limit_info: dict[str, Any] | None) -> None:
        """Update state from parsed rate limit headers dict."""
        if not rate_limit_info:
            return
        for attr in (
            "requests_remaining",
            "requests_limit",
            "requests_reset",
            "input_tokens_remaining",
            "input_tokens_limit",
            "input_tokens_reset",
            "output_tokens_remaining",
            "output_tokens_limit",
            "output_tokens_reset",
            "fast_input_tokens_remaining",
            "fast_input_tokens_limit",
            "fast_input_tokens_reset",
            "fast_output_tokens_remaining",
            "fast_output_tokens_limit",
            "fast_output_tokens_reset",
        ):
            val = rate_limit_info.get(attr)
            if val is not None:
                setattr(self, attr, val)

    def most_constrained_ratio(
        self,
    ) -> tuple[float, str, int | None, int | None, str | None]:
        """Find the dimension with the lowest remaining/limit ratio.

        Returns:
            Tuple of (ratio, dimension_name, remaining, limit, reset_timestamp).
            ratio is 1.0 if no data is available (meaning "no constraint known").
        """
        best_ratio = 1.0
        best_dimension = "unknown"
        best_remaining = None
        best_limit = None
        best_reset = None

        for dimension, remaining_attr, limit_attr, reset_attr in (
            ("requests", "requests_remaining", "requests_limit", "requests_reset"),
            (
                "input_tokens",
                "input_tokens_remaining",
                "input_tokens_limit",
                "input_tokens_reset",
            ),
            (
                "output_tokens",
                "output_tokens_remaining",
                "output_tokens_limit",
                "output_tokens_reset",
            ),
            (
                "fast_input_tokens",
                "fast_input_tokens_remaining",
                "fast_input_tokens_limit",
                "fast_input_tokens_reset",
            ),
            (
                "fast_output_tokens",
                "fast_output_tokens_remaining",
                "fast_output_tokens_limit",
                "fast_output_tokens_reset",
            ),
        ):
            remaining = getattr(self, remaining_attr)
            limit = getattr(self, limit_attr)
            if remaining is not None and limit is not None and limit > 0:
                ratio = remaining / limit
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_dimension = dimension
                    best_remaining = remaining
                    best_limit = limit
                    best_reset = getattr(self, reset_attr)

        return best_ratio, best_dimension, best_remaining, best_limit, best_reset


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process-wide concurrency gate
# ---------------------------------------------------------------------------
# Shared across ALL AnthropicProvider instances in this process (including
# parent + delegated child sessions). Prevents blast patterns that trigger
# Cloudflare bot detection when many sessions delegate simultaneously.
# Created lazily on the first API call; keyed by event loop so that tests
# using asyncio.run() get fresh semaphores rather than inheriting stale state.

_process_semaphore: asyncio.Semaphore | None = None
_process_semaphore_loop: Any = None  # asyncio.AbstractEventLoop
_process_semaphore_max: int = 0
_active_requests: int = 0  # currently holding semaphore (executing)
_waiting_requests: int = 0  # waiting to acquire semaphore


async def _get_process_semaphore(max_concurrent: int) -> asyncio.Semaphore | None:
    """Get or create the process-wide concurrency semaphore.

    Returns ``None`` when ``max_concurrent <= 0`` (semaphore disabled).
    Recreates the semaphore when called from a different event loop so that
    unit tests using ``asyncio.run()`` always get a fresh, valid semaphore.
    """
    global _process_semaphore, _process_semaphore_loop, _process_semaphore_max
    if max_concurrent <= 0:
        return None
    current_loop = asyncio.get_running_loop()
    if (
        _process_semaphore is None
        or _process_semaphore_loop is not current_loop
        or _process_semaphore_max != max_concurrent
    ):
        _process_semaphore = asyncio.Semaphore(max_concurrent)
        _process_semaphore_loop = current_loop
        _process_semaphore_max = max_concurrent
    return _process_semaphore


# Beta header constants — single source of truth for experimental feature headers
BETA_HEADER_1M_CONTEXT = "context-1m-2025-08-07"
BETA_HEADER_INTERLEAVED_THINKING = "interleaved-thinking-2025-05-14"
BETA_HEADER_TASK_BUDGETS = "task-budgets-2026-03-13"
BETA_HEADER_FAST_MODE = "fast-mode-2026-02-01"
PROVIDER_FALLBACK_OPEN = "provider:fallback_open"
PROVIDER_FALLBACK_ACTIVE = "provider:fallback_active"
FALLBACK_STATE_VERSION = 1

# ---------------------------------------------------------------------------
# Deprecated model retirement dates — warn once per process per model
# ---------------------------------------------------------------------------
_DEPRECATED_MODELS: dict[str, str] = {
    "claude-3-haiku-20240307": "2026-04-19",
    "claude-sonnet-4-20250514": "2026-06-15",
    "claude-opus-4-20250514": "2026-06-15",
}
_warned_deprecated_models: set[str] = set()


def _clear_deprecated_model_warnings() -> None:
    """Clear the warned-models set.

    Internal helper for tests. Follows the same pattern as _clear_fallback_windows().
    """
    _warned_deprecated_models.clear()


@dataclass(frozen=True)
class ModelCapabilities:
    """Per-model capability matrix — single source of truth.

    Every model-specific decision in the provider (context window size,
    thinking mode, output capacity, etc.) should be derived from this
    dataclass rather than scattered if/else checks.
    """

    family: str
    max_output_tokens: int = 64000
    base_context_window: int = 200000
    supports_1m: bool = False
    supports_thinking: bool = False
    supports_adaptive_thinking: bool = False
    supports_manual_thinking: bool = (
        True  # False on Opus 4.7+ (type="enabled" returns HTTP 400)
    )
    supports_output_config: bool = False  # True = model accepts output_config.effort
    supports_sampling: bool = True  # False = temperature silently ignored by model
    thinking_display_required: bool = (
        False  # True = must send thinking.display to see thinking content
    )
    supported_efforts: tuple[str, ...] = (
        "low",
        "medium",
        "high",
    )  # Valid effort levels for output_config and reasoning_effort
    supports_task_budget: bool = (
        False  # True = model accepts output_config.task_budget (beta)
    )
    default_thinking_budget: int = 0
    supports_speed: bool = False  # True = model accepts the speed parameter
    supports_inline_system: bool = (
        False  # True = model accepts role='system' in messages[]
    )
    capability_tags: tuple[str, ...] = ("tools", "streaming", "json_mode")


@dataclass(frozen=True)
class _RuntimeModelInfo:
    """Best-effort runtime model metadata from Anthropic's Models API."""

    max_input_tokens: int | None = None
    max_tokens: int | None = None
    supports_thinking: bool | None = None
    supports_adaptive_thinking: bool | None = None


@dataclass
class _FallbackWindow:
    """Temporary downgrade window for a model family."""

    requested_model: str
    fallback_model: str
    opened_at: float
    until: float
    opened_by_pid: int
    error_type: str
    error_message: str


_fallback_windows: dict[str, _FallbackWindow] = {}
_fallback_lock = Lock()


def _get_active_fallback_window(
    family: str, *, now: float | None = None
) -> _FallbackWindow | None:
    """Return the active fallback window for a family, if any."""
    current_time = time.time() if now is None else now
    with _fallback_lock:
        window = _fallback_windows.get(family)
        if window is None:
            return None
        if window.until <= current_time:
            _fallback_windows.pop(family, None)
            return None
        return window


def _set_fallback_window(family: str, window: _FallbackWindow) -> None:
    """Store a fallback window for a family."""
    with _fallback_lock:
        _fallback_windows[family] = window


def _clear_fallback_windows() -> None:
    """Clear all fallback windows.

    Internal helper for tests. The provider intentionally keeps fallback state
    process-wide so sibling sessions share the same temporary downgrade window.
    """
    with _fallback_lock:
        _fallback_windows.clear()


class AnthropicChatResponse(ChatResponse):
    """ChatResponse with additional fields for streaming UI compatibility."""

    content_blocks: (
        list[TextContent | ThinkingContent | ToolCallContent | WebSearchContent] | None
    ) = None
    text: str | None = None
    web_search_results: list[dict[str, Any]] | None = None


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

    _totals: dict = {"cost_usd": None, "has_data": False}

    def _add_cost(cost) -> None:
        if cost is not None:
            _totals["cost_usd"] = (_totals["cost_usd"] or Decimal("0")) + cost
            _totals["has_data"] = True

    # Get API key from config or environment
    api_key = config.get("api_key")
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        logger.warning("No API key found for Anthropic provider")
        return None

    provider = AnthropicProvider(api_key, config, coordinator, add_cost=_add_cost)
    await coordinator.mount("providers", provider, name="anthropic")
    coordinator.register_contributor(
        "session.cost",
        "provider-anthropic",
        lambda: (
            {
                "cost_usd": str(_totals["cost_usd"])
                if _totals["cost_usd"] is not None
                else None
            }
            if _totals["has_data"]
            else None
        ),
    )
    logger.info("Mounted AnthropicProvider")

    # Return cleanup function that delegates to provider.close().
    # close() handles lazy-client guard, asyncio.shield, and CancelledError.
    async def cleanup():
        await provider.close()

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

    @staticmethod
    def _config_bool(value: Any) -> bool:
        """Parse config booleans from YAML or CLI string values."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _config_int(value: Any, default: int) -> int:
        """Parse an int config value with a safe fallback."""
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(
                "[PROVIDER] Invalid integer config value %r; using default %s",
                value,
                default,
            )
            return default

    @staticmethod
    def _config_float(value: Any, default: float) -> float:
        """Parse a float config value with a safe fallback."""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning(
                "[PROVIDER] Invalid float config value %r; using default %s",
                value,
                default,
            )
            return default

    def __init__(
        self,
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
        add_cost=None,
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
        self._default_caps = self._get_capabilities(self.default_model)
        self.max_tokens = self.config.get(
            "max_tokens", self._default_caps.max_output_tokens
        )
        self.temperature = self.config.get("temperature", 0.7)
        self.priority = self.config.get("priority", 100)  # Store priority for selection
        self.raw = self.config.get("raw", False)  # Include raw payload in events
        self.timeout = self.config.get(
            "timeout", 600.0
        )  # API timeout in seconds (default 5 minutes)

        # Retry configuration — delegates to shared retry_with_backoff() from amplifier-core.
        # We handle retries ourselves (SDK max_retries=0) to properly honor retry-after headers
        # and use longer backoffs that help with org-wide rate limit pressure.
        self._retry_max_retries = self._config_int(self.config.get("max_retries", 5), 5)
        self._retry_min_delay = self._config_float(
            self.config.get("min_retry_delay", 1.0), 1.0
        )
        self._retry_max_delay = self._config_float(
            self.config.get("max_retry_delay", 60.0), 60.0
        )
        self._retry_jitter = self._config_bool(self.config.get("retry_jitter", True))
        self._retry_config = RetryConfig(
            max_retries=self._retry_max_retries,
            initial_delay=self._retry_min_delay,
            max_delay=self._retry_max_delay,
            jitter=self._retry_jitter,
        )
        self._overloaded_delay_multiplier = float(
            self.config.get("overloaded_delay_multiplier", 10.0)
        )

        # Temporary model downgrade on persistent overloads.
        # When enabled, a higher-tier family gets a short retry budget; if it still
        # overloads, a process-wide cooldown window routes subsequent requests to
        # the configured lower-tier model until the cooldown expires.
        self._fallback_on_overload = self._config_bool(
            self.config.get("fallback_on_overload", False)
        )
        self._fallback_retry_count = max(
            0, self._config_int(self.config.get("fallback_retry_count", 1), 1)
        )
        self._fallback_cooldown_seconds = max(
            0.0,
            self._config_float(
                self.config.get("fallback_cooldown_seconds", 1800.0), 1800.0
            ),
        )
        self._enable_1m_context = self._config_bool(
            self.config.get("enable_1m_context", True)
        )
        self._fallback_sonnet_model = str(
            self.config.get("fallback_sonnet_model", "claude-sonnet-4-6")
        )
        self._fallback_haiku_model = str(
            self.config.get("fallback_haiku_model", "claude-haiku-4-5")
        )
        self._persist_fallback_state = self._config_bool(
            self.config.get("persist_fallback_state", False)
        )

        # Pre-emptive throttle configuration
        # Threshold: fraction of remaining capacity below which we inject a delay.
        # Default 0.02 (2%) — only throttle when nearly exhausted, not at 10%.
        # Delay: fallback sleep when no reset timestamp is available.
        # Default 1.0s — just enough to ease pressure without punishing every request.
        self._throttle_threshold = float(self.config.get("throttle_threshold", 0.02))
        self._throttle_delay = float(self.config.get("throttle_delay", 1.0))
        self._rate_limit_state = _RateLimitState()

        # Process-wide concurrency gate.
        # Limits how many API calls this process has in-flight simultaneously,
        # shared across ALL provider instances (parent + delegated child sessions).
        # This prevents blast patterns (e.g. parallel: true recipes spawning 20+
        # concurrent calls) that trigger Cloudflare bot-detection on api.anthropic.com.
        # Set to 0 to disable the semaphore entirely.
        self._max_concurrent_requests = int(
            self.config.get("max_concurrent_requests", 5)
        )

        # Use streaming API by default to support large context windows (Anthropic requires streaming
        # for operations that may take > 10 minutes, e.g. with 300k+ token contexts)
        self.use_streaming = self.config.get("use_streaming", True)
        self.filtered = self.config.get(
            "filtered", True
        )  # Filter to curated model list by default
        self.enable_prompt_caching = self.config.get("enable_prompt_caching", True)
        self.enable_web_search = self.config.get(
            "enable_web_search", False
        )  # Enable native web search tool

        # Get base_url from config for custom endpoints (proxies, local APIs, etc.)
        self._base_url = self.config.get("base_url")

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

        # Shared rate-limit state file for cross-process awareness.
        # All Anthropic provider instances (across processes, Docker containers
        # sharing a filesystem, etc.) read this file before the per-emptive
        # throttle check and write to it after every successful API response.
        # This lets process B know that process A is almost out of tokens and
        # should back off — even though they each have independent _RateLimitState
        # instances.
        # Set to "" to disable cross-process sharing entirely.
        _default_shared_path = os.path.join(
            os.path.expanduser("~"), ".amplifier", "rate-limit-state.json"
        )
        self._shared_state_path: str = str(
            self.config.get("rate_limit_state_path", _default_shared_path)
        )
        self._last_shared_state_read: float = 0.0  # epoch time of last file read
        self._last_written_state: dict[
            str, Any
        ] = {}  # last written content (for change detection)

        # Optional persisted fallback-breaker state for cross-process overload
        # downgrade windows. Disabled by default so environments that should not
        # touch the filesystem stay process-local unless explicitly opted in.
        _default_fallback_state_path = os.path.join(
            os.path.expanduser("~"), ".amplifier", "anthropic-fallback-state.json"
        )
        configured_fallback_state_path = self.config.get(
            "fallback_state_path", _default_fallback_state_path
        )
        self._fallback_state_path: str = (
            str(configured_fallback_state_path)
            if self._persist_fallback_state
            and configured_fallback_state_path is not None
            else ""
        )
        self._last_fallback_state_read: float = 0.0
        self._runtime_model_info_cache: dict[str, _RuntimeModelInfo | None] = {}

        # Track tool call IDs that have been repaired with synthetic results.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations (since synthetic results
        # are injected into request.messages but not persisted to message store).
        self._repaired_tool_ids: set[str] = set()
        self._add_cost = add_cost or (lambda cost: None)

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
            capabilities=list(self._default_caps.capability_tags),
            defaults={
                "model": self.default_model,
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": 600.0,
                "context_window": 1000000
                if self._enable_1m_context and self._default_caps.supports_1m
                else self._default_caps.base_context_window,
                "max_output_tokens": self._default_caps.max_output_tokens,
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
                    prompt="Request 1M token context window when the selected model supports it",
                    required=False,
                    default="true",
                    requires_model=True,  # Shown after model selection
                    show_when={
                        "default_model": "not_contains:haiku"
                    },  # Hide for Haiku (doesn't support 1M)
                ),
                ConfigField(
                    id="enable_prompt_caching",
                    display_name="Prompt Caching",
                    field_type="boolean",
                    prompt="Enable prompt caching? (Reduces cost by 90% on cached tokens)",
                    required=False,
                    default="true",
                ),
                ConfigField(
                    id="effort",
                    display_name="Reasoning Effort",
                    field_type="choice",
                    choices=["low", "medium", "high", "xhigh", "max"],
                    prompt=(
                        "Default reasoning effort applied to every request. Like "
                        "request.reasoning_effort, this ENABLES extended thinking "
                        "and sets its depth, so it raises token cost on every call "
                        "\u2014 leave blank unless you want stronger reasoning by default. "
                        "low/medium/high work on all thinking-capable models; xhigh "
                        "requires Opus 4.7+; max requires Opus 4.8+/Sonnet 4.6. "
                        "On Opus 4.7+ it is also sent as output_config.effort. "
                        "Unsupported values for the selected model are ignored."
                    ),
                    required=False,
                    requires_model=True,  # Shown after model selection
                    # Gate on the EFFECT surface (extended thinking), not on
                    # output_config support: effort enables/sizes thinking on
                    # every thinking-capable model, so hide it only for models
                    # that don't support thinking at all (pre-4.5 Haiku).
                    show_when={"default_model": "not_contains:haiku-3"},
                ),
                ConfigField(
                    id="fallback_on_overload",
                    display_name="Temporary Overload Downgrade",
                    field_type="boolean",
                    prompt="Downgrade temporarily if a higher-tier Claude model stays overloaded?",
                    required=False,
                    default="false",
                    requires_model=True,
                    show_when={
                        "default_model": "not_contains:haiku"
                    },  # No lower Anthropic family exists below Haiku
                ),
                ConfigField(
                    id="fallback_retry_count",
                    display_name="Retries Before Downgrade",
                    field_type="text",
                    prompt="How many overload retries before downgrading?",
                    required=False,
                    default="1",
                    requires_model=True,
                    show_when={"fallback_on_overload": "true"},
                ),
                ConfigField(
                    id="fallback_cooldown_seconds",
                    display_name="Downgrade Cooldown (seconds)",
                    field_type="text",
                    prompt="How long should the downgrade stay active before retrying the higher model?",
                    required=False,
                    default="1800",
                    requires_model=True,
                    show_when={"fallback_on_overload": "true"},
                ),
                ConfigField(
                    id="persist_fallback_state",
                    display_name="Share Downgrade State",
                    field_type="boolean",
                    prompt="Persist temporary downgrade state across separate Amplifier processes?",
                    required=False,
                    default="false",
                    requires_model=True,
                    show_when={"fallback_on_overload": "true"},
                ),
                ConfigField(
                    id="fallback_sonnet_model",
                    display_name="Opus Fallback Model",
                    field_type="text",
                    prompt="Model to use when Opus is overloaded",
                    required=False,
                    default="claude-sonnet-4-6",
                    requires_model=True,
                    show_when={
                        "fallback_on_overload": "true",
                        "default_model": "contains:opus",
                    },
                ),
                ConfigField(
                    id="fallback_haiku_model",
                    display_name="Sonnet Fallback Model",
                    field_type="text",
                    prompt="Model to use when Sonnet is overloaded",
                    required=False,
                    default="claude-haiku-4-5",
                    requires_model=True,
                    show_when={
                        "fallback_on_overload": "true",
                        "default_model": "not_contains:haiku",
                    },
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
                raw_model = next(model for model in api_models if model.id == model_id)
                caps = self._apply_runtime_capability_overrides(
                    self._get_capabilities(model_id),
                    self._extract_runtime_model_info(raw_model),
                )

                has_1m = self._enable_1m_context and caps.supports_1m
                context_window = (
                    max(caps.base_context_window, 1000000)
                    if has_1m
                    else caps.base_context_window
                )

                result.append(
                    ModelInfo(
                        id=model_id,
                        display_name=display_name,
                        context_window=context_window,
                        max_output_tokens=caps.max_output_tokens,
                        capabilities=list(caps.capability_tags),
                        defaults={
                            "temperature": 0.7,
                            "max_tokens": caps.max_output_tokens,
                        },
                    )
                )

        # Sort alphabetically by display name
        result.sort(key=lambda m: m.display_name.lower())

        return result

    @staticmethod
    def _detect_family(model_id: str) -> str:
        """Detect the Claude model family from a model ID string."""
        model_lower = model_id.lower()
        for family in ("opus", "sonnet", "haiku"):
            if family in model_lower:
                return family
        return "sonnet"  # Default to sonnet for unknown models

    @staticmethod
    def _detect_version(model_id: str, family: str) -> tuple[int, int]:
        """Extract (major, minor) version from a model ID.

        Parses patterns like ``claude-opus-4-6``, ``claude-sonnet-4-5-20250929``.
        Returns ``(0, 0)`` when the version cannot be determined — callers
        should treat unknown versions conservatively.
        """
        model_lower = model_id.lower()

        # Prefer family-MAJOR-MINOR where MINOR is a short semantic version part,
        # not a snapshot suffix like 20250514.
        pattern = rf"{family}-(\d+)-(\d{{1,2}})(?:-|$)"
        match = re.search(pattern, model_lower)
        if match:
            return int(match.group(1)), int(match.group(2))

        # Fallback for ids like claude-sonnet-4-20250514 where only the major
        # semantic version is present before the snapshot date.
        major_only_pattern = rf"{family}-(\d+)(?:-|$)"
        match = re.search(major_only_pattern, model_lower)
        if match:
            return int(match.group(1)), 0
        return (0, 0)

    @classmethod
    def _get_capabilities(cls, model_id: str) -> ModelCapabilities:
        """Return the capability matrix for *model_id*.

        Version requirements
        --------------------
        * **Opus 4.6+** — 1M context, adaptive thinking, 128K output
        * **Sonnet 4.5+** — 1M context, extended thinking, 64K output
        * **Haiku 4.5+** — fast inference, extended thinking, no adaptive, no 1M

        When the version cannot be parsed from the model ID we assume the
        *latest* capabilities for that family so newly released models work
        out-of-the-box.
        """
        family = cls._detect_family(model_id)
        major, minor = cls._detect_version(model_id, family)
        version_known = (major, minor) != (0, 0)

        if family == "opus":
            is_46_plus = not version_known or (major, minor) >= (4, 6)
            is_47_plus = not version_known or (major, minor) >= (4, 7)
            is_48_plus = not version_known or (major, minor) >= (4, 8)
            return ModelCapabilities(
                family="opus",
                max_output_tokens=128000 if is_46_plus else 64000,
                supports_1m=is_46_plus,
                supports_thinking=True,
                supports_adaptive_thinking=is_46_plus,
                supports_manual_thinking=not is_47_plus,
                supports_output_config=is_47_plus,
                supports_task_budget=is_47_plus,
                supports_sampling=not is_47_plus,
                thinking_display_required=is_47_plus,
                supported_efforts=(
                    ("low", "medium", "high", "xhigh", "max")
                    if is_48_plus
                    else ("low", "medium", "high", "xhigh")
                    if is_47_plus
                    else ("low", "medium", "high")
                ),
                supports_speed=is_48_plus,
                supports_inline_system=is_48_plus,
                default_thinking_budget=64000 if is_46_plus else 32000,
                capability_tags=(
                    "tools",
                    "thinking",
                    "streaming",
                    "json_mode",
                    "vision",
                ),
            )

        if family == "sonnet":
            is_46_plus = not version_known or (major, minor) >= (4, 6)
            is_45_plus = is_46_plus or (major, minor) >= (4, 5)
            return ModelCapabilities(
                family="sonnet",
                supports_1m=is_46_plus,
                supports_thinking=True,
                supports_adaptive_thinking=is_46_plus,
                default_thinking_budget=32000,
                capability_tags=(
                    "tools",
                    "thinking",
                    "streaming",
                    "json_mode",
                    "vision",
                ),
            )

        if family == "haiku":
            is_45_plus = not version_known or (major, minor) >= (4, 5)
            return ModelCapabilities(
                family="haiku",
                supports_thinking=is_45_plus,
                supports_adaptive_thinking=False,
                default_thinking_budget=32000 if is_45_plus else 0,
                capability_tags=("tools", "streaming", "json_mode", "fast", "vision")
                + (("thinking",) if is_45_plus else ()),
            )

        # Unknown family — conservative defaults
        return ModelCapabilities(family=family)

    @staticmethod
    def _positive_int_or_none(value: Any) -> int | None:
        """Parse a positive integer from runtime metadata, treating 0 as unknown."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _resolve_model_info_value(model_info: Any, *path: str) -> Any:
        """Traverse dict/object model metadata without caring about concrete types."""
        current = model_info
        for key in path:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
        return current

    @classmethod
    def _capability_supported(cls, model_info: Any, *path: str) -> bool | None:
        """Return capability support from Anthropic Models API metadata."""
        value = cls._resolve_model_info_value(model_info, *path, "supported")
        if value is None:
            return None
        return bool(value)

    @classmethod
    def _extract_runtime_model_info(cls, model_info: Any) -> _RuntimeModelInfo:
        """Extract request-relevant metadata from a Models API response."""
        return _RuntimeModelInfo(
            max_input_tokens=cls._positive_int_or_none(
                cls._resolve_model_info_value(model_info, "max_input_tokens")
            ),
            max_tokens=cls._positive_int_or_none(
                cls._resolve_model_info_value(model_info, "max_tokens")
            ),
            supports_thinking=cls._capability_supported(
                model_info, "capabilities", "thinking"
            ),
            supports_adaptive_thinking=cls._capability_supported(
                model_info, "capabilities", "thinking", "types", "adaptive"
            ),
        )

    @classmethod
    def _apply_runtime_capability_overrides(
        cls,
        base_caps: ModelCapabilities,
        runtime_info: _RuntimeModelInfo | None,
    ) -> ModelCapabilities:
        """Overlay live Models API metadata onto the static family heuristics."""
        if runtime_info is None:
            return base_caps

        capability_tags = list(base_caps.capability_tags)
        supports_thinking = (
            runtime_info.supports_thinking
            if runtime_info.supports_thinking is not None
            else base_caps.supports_thinking
        )
        supports_adaptive_thinking = (
            runtime_info.supports_adaptive_thinking
            if runtime_info.supports_adaptive_thinking is not None
            else base_caps.supports_adaptive_thinking
        )

        if supports_thinking and "thinking" not in capability_tags:
            capability_tags.append("thinking")
        if not supports_thinking and "thinking" in capability_tags:
            capability_tags = [tag for tag in capability_tags if tag != "thinking"]

        base_context_window = (
            runtime_info.max_input_tokens or base_caps.base_context_window
        )
        supports_1m = (
            runtime_info.max_input_tokens >= 1_000_000
            if runtime_info.max_input_tokens is not None
            else base_caps.supports_1m
        )
        default_thinking_budget = base_caps.default_thinking_budget
        if supports_thinking and default_thinking_budget <= 0:
            default_thinking_budget = 32000

        return ModelCapabilities(
            family=base_caps.family,
            max_output_tokens=runtime_info.max_tokens or base_caps.max_output_tokens,
            base_context_window=base_context_window,
            supports_1m=supports_1m,
            supports_thinking=supports_thinking,
            supports_adaptive_thinking=supports_adaptive_thinking,
            supports_manual_thinking=base_caps.supports_manual_thinking,
            supports_output_config=base_caps.supports_output_config,
            supports_task_budget=base_caps.supports_task_budget,
            supports_sampling=base_caps.supports_sampling,
            thinking_display_required=base_caps.thinking_display_required,
            supported_efforts=base_caps.supported_efforts,
            supports_speed=base_caps.supports_speed,
            supports_inline_system=base_caps.supports_inline_system,
            default_thinking_budget=default_thinking_budget,
            capability_tags=tuple(capability_tags),
        )

    async def _get_runtime_model_info(self, model_id: str) -> _RuntimeModelInfo | None:
        """Retrieve and cache live model metadata from Anthropic's Models API."""
        if model_id in self._runtime_model_info_cache:
            return self._runtime_model_info_cache[model_id]

        try:
            model_info = await self.client.models.retrieve(model_id)
        except Exception:
            self._runtime_model_info_cache[model_id] = None
            return None

        runtime_info = self._extract_runtime_model_info(model_info)
        self._runtime_model_info_cache[model_id] = runtime_info
        return runtime_info

    async def _get_request_capabilities(self, model_id: str) -> ModelCapabilities:
        """Compute capabilities for an effective request model with live overrides."""
        base_caps = self._get_capabilities(model_id)
        runtime_info = await self._get_runtime_model_info(model_id)
        return self._apply_runtime_capability_overrides(base_caps, runtime_info)

    @staticmethod
    def _dedupe_headers(headers: list[str]) -> list[str]:
        """Preserve header order while dropping duplicates and blanks."""
        deduped: list[str] = []
        for header in headers:
            if not header or header in deduped:
                continue
            deduped.append(header)
        return deduped

    def _should_add_context_1m_beta(
        self, model_id: str, request_caps: ModelCapabilities
    ) -> bool:
        """Return True when the effective model still needs the 1M beta header."""
        if not self._enable_1m_context:
            return False

        family = self._detect_family(model_id)
        if family == "haiku":
            return False

        major, minor = self._detect_version(model_id, family)
        version = (major, minor)

        if family == "opus":
            # 1M context is GA for Opus 4.8+; beta header only needed for 4.6 and 4.7.
            # Unknown versions (0, 0) assume latest (4.8+), so no header.
            if version == (0, 0):
                return False
            return (4, 6) <= version < (4, 8)
        return family == "sonnet" and (version == (0, 0) or version >= (4, 0))

    def _should_add_interleaved_beta(
        self,
        *,
        request_caps: ModelCapabilities,
        tools_present: bool,
        resolved_thinking_type: str | None,
    ) -> bool:
        """Return True when tool-use thinking should opt into interleaving beta."""
        if not tools_present or not request_caps.supports_thinking:
            return False
        if request_caps.family == "haiku":
            return False
        if (
            resolved_thinking_type == "adaptive"
            and request_caps.supports_adaptive_thinking
        ):
            return False
        return resolved_thinking_type is not None

    def _build_request_beta_headers(
        self,
        *,
        model_id: str,
        request_caps: ModelCapabilities,
        tools_present: bool,
        resolved_thinking_type: str | None,
        has_task_budget: bool = False,
        fast_mode: bool = False,
    ) -> list[str]:
        """Build the anthropic-beta header set for a specific effective model."""
        headers = list(self._beta_headers)
        if self._should_add_context_1m_beta(model_id, request_caps):
            headers.append(BETA_HEADER_1M_CONTEXT)
        if self._should_add_interleaved_beta(
            request_caps=request_caps,
            tools_present=tools_present,
            resolved_thinking_type=resolved_thinking_type,
        ):
            headers.append(BETA_HEADER_INTERLEAVED_THINKING)
        if has_task_budget:
            headers.append(BETA_HEADER_TASK_BUDGETS)
        if fast_mode:
            headers.append(BETA_HEADER_FAST_MODE)
        return self._dedupe_headers(headers)

    @staticmethod
    def _is_cloudflare_challenge(error: AnthropicAPIStatusError) -> bool:
        """Detect Cloudflare bot-management challenge responses.

        Cloudflare interposes HTML challenge pages (HTTP 403) that look nothing
        like Anthropic API errors.  Signals:

        1. The SDK failed to parse the body as JSON (error.body is None).
        2. The Content-Type is text/html (not application/json).
        3. The raw response text contains Cloudflare markers.

        Any combination of (1 + 2) or (1 + 3) is sufficient.  If the SDK
        successfully parsed a JSON body, this is a real API error regardless
        of other signals.
        """
        # If the SDK parsed a JSON body, this is a real API error
        if getattr(error, "body", None) is not None:
            return False

        # Inspect the raw HTTP response for HTML / Cloudflare signals
        response = getattr(error, "response", None)
        if response is None:
            return False

        content_type = getattr(response, "headers", {}).get("content-type", "")
        if "text/html" in content_type:
            return True

        # Fallback: scan response text for Cloudflare markers
        text = getattr(response, "text", "") or ""
        cf_markers = (
            "Just a moment",
            "cf-browser-verification",
            "cloudflare",
            "Checking if the site connection is secure",
        )
        return any(marker in text for marker in cf_markers)

    def _build_retry_config(self, max_retries: int) -> RetryConfig:
        """Create a retry config that preserves current backoff settings."""
        return RetryConfig(
            max_retries=max_retries,
            initial_delay=self._retry_min_delay,
            max_delay=self._retry_max_delay,
            jitter=self._retry_jitter,
        )

    def _fallback_target_for_model(self, model_id: str) -> str | None:
        """Return the configured lower-tier fallback for a requested model."""
        family = self._detect_family(model_id)
        target: str | None
        if family == "opus":
            target = self._fallback_sonnet_model
        elif family == "sonnet":
            target = self._fallback_haiku_model
        else:
            return None

        if not target or target == model_id:
            return None

        target_family = self._detect_family(target)
        if target_family == family:
            logger.warning(
                "[PROVIDER] Ignoring invalid overload fallback %s -> %s (same family)",
                model_id,
                target,
            )
            return None

        return target

    @staticmethod
    def _is_overload_fallback_error(error: KernelLLMError) -> bool:
        """Return True when the error indicates model overload, not generic throttling."""
        status_code = getattr(error, "status_code", None)
        if isinstance(error, KernelProviderUnavailableError) and status_code == 529:
            return True
        if isinstance(error, KernelRateLimitError) and status_code == 429:
            msg = str(error).lower()
            return "overload" in msg or "overloaded" in msg
        return False

    def _resolve_effective_model(
        self, requested_model: str
    ) -> tuple[str, list[tuple[str, _FallbackWindow]]]:
        """Apply any active overload fallback windows to the requested model."""
        self._read_shared_fallback_state()
        effective_model = requested_model
        active_windows: list[tuple[str, _FallbackWindow]] = []
        seen_families: set[str] = set()

        while True:
            family = self._detect_family(effective_model)
            if family in seen_families:
                break
            seen_families.add(family)

            window = _get_active_fallback_window(family)
            if window is None or window.fallback_model == effective_model:
                break

            active_windows.append((family, window))
            effective_model = window.fallback_model

        return effective_model, active_windows

    async def _emit_provider_event(self, name: str, payload: dict[str, Any]) -> None:
        """Emit a provider event when hooks are available."""
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            await self.coordinator.hooks.emit(name, payload)

    async def _emit_active_fallback_window(
        self,
        requested_model: str,
        effective_model: str,
        active_windows: list[tuple[str, _FallbackWindow]],
    ) -> None:
        """Emit observability for an active temporary downgrade window."""
        if not active_windows:
            return

        now = time.time()
        payload = {
            "provider": "anthropic",
            "requested_model": requested_model,
            "effective_model": effective_model,
            "chain": [
                {
                    "family": family,
                    "fallback_model": window.fallback_model,
                    "until": window.until,
                    "remaining_seconds": max(0.0, window.until - now),
                }
                for family, window in active_windows
            ],
        }
        logger.warning(
            "[PROVIDER] Temporary downgrade active: %s -> %s",
            requested_model,
            effective_model,
        )
        await self._emit_provider_event(PROVIDER_FALLBACK_ACTIVE, payload)

    async def _open_fallback_window(
        self, attempted_model: str, error: KernelLLMError
    ) -> bool:
        """Open a temporary downgrade window for the attempted model family."""
        fallback_model = self._fallback_target_for_model(attempted_model)
        if not fallback_model:
            return False

        family = self._detect_family(attempted_model)
        now = time.time()
        until = now + self._fallback_cooldown_seconds
        window = _FallbackWindow(
            requested_model=attempted_model,
            fallback_model=fallback_model,
            opened_at=now,
            until=until,
            opened_by_pid=os.getpid(),
            error_type=type(error).__name__,
            error_message=str(error),
        )

        if self._fallback_cooldown_seconds > 0:
            _set_fallback_window(family, window)
            self._write_shared_fallback_state(family, window)

        logger.warning(
            "[PROVIDER] Opening temporary downgrade window for %s -> %s (cooldown %.0fs)",
            attempted_model,
            fallback_model,
            self._fallback_cooldown_seconds,
        )
        await self._emit_provider_event(
            PROVIDER_FALLBACK_OPEN,
            {
                "provider": "anthropic",
                "requested_model": attempted_model,
                "fallback_model": fallback_model,
                "family": family,
                "cooldown_seconds": self._fallback_cooldown_seconds,
                "until": until,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        return True

    @staticmethod
    def _fallback_window_to_dict(window: _FallbackWindow) -> dict[str, Any]:
        """Serialize a fallback window for JSON persistence."""
        return {
            "requested_model": window.requested_model,
            "fallback_model": window.fallback_model,
            "opened_at": window.opened_at,
            "until": window.until,
            "opened_by_pid": window.opened_by_pid,
            "error_type": window.error_type,
            "error_message": window.error_message,
        }

    @staticmethod
    def _fallback_window_from_dict(data: Any) -> _FallbackWindow | None:
        """Parse a persisted fallback window, ignoring malformed entries."""
        if not isinstance(data, dict):
            return None
        try:
            return _FallbackWindow(
                requested_model=str(data["requested_model"]),
                fallback_model=str(data["fallback_model"]),
                opened_at=float(data["opened_at"]),
                until=float(data["until"]),
                opened_by_pid=int(data.get("opened_by_pid", 0)),
                error_type=str(data.get("error_type", "")),
                error_message=str(data.get("error_message", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _load_shared_fallback_windows(
        self, *, now: float | None = None
    ) -> dict[str, _FallbackWindow]:
        """Load non-expired persisted fallback windows from disk."""
        if not self._fallback_state_path:
            return {}
        current_time = time.time() if now is None else now
        try:
            with open(self._fallback_state_path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}

            windows_data = data.get("windows", {})
            if not isinstance(windows_data, dict):
                return {}

            windows: dict[str, _FallbackWindow] = {}
            for family, raw_window in windows_data.items():
                if not isinstance(family, str):
                    continue
                window = self._fallback_window_from_dict(raw_window)
                if window is None or window.until <= current_time:
                    continue
                windows[family] = window
            return windows
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    def _write_shared_fallback_state(
        self, family: str, window: _FallbackWindow
    ) -> None:
        """Atomically persist fallback windows when cross-process sharing is enabled."""
        if not self._fallback_state_path:
            return
        try:
            windows = self._load_shared_fallback_windows(now=time.time())
            existing = windows.get(family)
            if existing is None or window.until > existing.until:
                windows[family] = window

            serialized_windows = {
                name: self._fallback_window_to_dict(active_window)
                for name, active_window in sorted(windows.items())
            }
            state: dict[str, Any] = {
                "version": FALLBACK_STATE_VERSION,
                "updated_at": time.time(),
                "updated_by_pid": os.getpid(),
                "windows": serialized_windows,
            }

            path = self._fallback_state_path
            tmp_path = path + ".tmp"
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            os.rename(tmp_path, path)
        except Exception:
            pass  # Never crash on I/O errors

    def _read_shared_fallback_state(self) -> None:
        """Merge persisted fallback windows into the process-local breaker state."""
        if not self._fallback_state_path:
            return
        now = time.time()
        if now - self._last_fallback_state_read < 1.0:
            return
        self._last_fallback_state_read = now

        windows = self._load_shared_fallback_windows(now=now)
        for family, window in windows.items():
            local_window = _get_active_fallback_window(family, now=now)
            if local_window is None or window.until > local_window.until:
                _set_fallback_window(family, window)

    def _write_shared_rate_limit_state(self, rate_limit_info: dict[str, Any]) -> None:
        """Atomically write rate-limit header data to the shared cross-process file.

        Uses write-to-tmp + os.rename() so concurrent readers never see a partial
        file.  Only writes if the rate-limit data actually changed (debounce by
        content equality) to avoid excessive I/O on every response.

        Wrapped entirely in try/except — file I/O failures must NEVER crash the
        provider.  The feature is completely silent when disabled (empty path).
        """
        if not self._shared_state_path:
            return
        try:
            _rate_fields = (
                "requests_remaining",
                "requests_limit",
                "requests_reset",
                "input_tokens_remaining",
                "input_tokens_limit",
                "input_tokens_reset",
                "output_tokens_remaining",
                "output_tokens_limit",
                "output_tokens_reset",
            )
            # Build the comparable payload (excludes volatile metadata)
            comparable: dict[str, Any] = {}
            for fname in _rate_fields:
                val = rate_limit_info.get(fname)
                if val is not None:
                    comparable[fname] = val

            # Skip write if nothing changed (debounce)
            if comparable == self._last_written_state:
                return

            state: dict[str, Any] = {
                "updated_at": time.time(),
                "updated_by_pid": os.getpid(),
                **comparable,
            }

            path = self._shared_state_path
            tmp_path = path + ".tmp"
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            os.rename(tmp_path, path)
            self._last_written_state = comparable
        except Exception:
            pass  # Never crash on I/O errors

    def _read_shared_rate_limit_state(self) -> None:
        """Read cross-process rate-limit state and merge it into local state.

        Only re-reads the file at most once per second (simple timestamp cache)
        to avoid hammering the filesystem on every throttle check.

        Merge strategy for *remaining* fields: take the LOWER value between local
        and shared state (conservative — don't assume capacity we can't confirm).
        For *limit* and *reset* fields: adopt the shared value only when local has
        no data yet.

        Ignores stale data (file older than 120 seconds) since stale rate-limit
        windows are meaningless.

        Wrapped entirely in try/except — file I/O failures must NEVER crash the
        provider.
        """
        if not self._shared_state_path:
            return
        now = time.time()
        if now - self._last_shared_state_read < 1.0:
            return  # Cache: don't re-read within 1 second
        self._last_shared_state_read = now
        try:
            with open(self._shared_state_path) as f:
                data: dict[str, Any] = json.load(f)

            updated_at = data.get("updated_at", 0)
            if now - updated_at > 120:
                return  # Stale — ignore

            # Merge remaining values: always take the lower of local vs shared
            _remaining_fields = (
                "requests_remaining",
                "input_tokens_remaining",
                "output_tokens_remaining",
            )
            # Limit / reset fields: adopt shared only when local is absent
            _limit_reset_fields = (
                "requests_limit",
                "requests_reset",
                "input_tokens_limit",
                "input_tokens_reset",
                "output_tokens_limit",
                "output_tokens_reset",
            )
            merged: dict[str, Any] = {}
            for fname in _remaining_fields:
                shared_val = data.get(fname)
                local_val = getattr(self._rate_limit_state, fname)
                if shared_val is not None and local_val is not None:
                    merged[fname] = min(int(shared_val), int(local_val))
                elif shared_val is not None:
                    merged[fname] = int(shared_val)
                # else: keep local value (don't override with absent shared data)

            for fname in _limit_reset_fields:
                shared_val = data.get(fname)
                local_val = getattr(self._rate_limit_state, fname)
                if shared_val is not None and local_val is None:
                    merged[fname] = shared_val

            if merged:
                self._rate_limit_state.update_from_headers(merged)

        except FileNotFoundError:
            pass  # Normal: file doesn't exist yet
        except Exception:
            pass  # Never crash on I/O errors

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

        if not self._fallback_on_overload:
            return await self._complete_chat_request(request, **kwargs)

        requested_model = str(kwargs.get("model", self.default_model))
        attempted_models: set[str] = set()
        full_retry_budget_used: set[str] = set()

        while True:
            effective_model, active_windows = self._resolve_effective_model(
                requested_model
            )

            # Guard against misconfigured fallback cycles.
            if (
                effective_model in attempted_models
                and effective_model not in full_retry_budget_used
            ):
                raise RuntimeError(
                    f"Overload fallback loop detected while resolving {requested_model}"
                )

            if active_windows:
                await self._emit_active_fallback_window(
                    requested_model, effective_model, active_windows
                )

            current_kwargs = dict(kwargs)
            current_kwargs["model"] = effective_model

            fallback_target = (
                self._fallback_target_for_model(effective_model)
                if self._fallback_on_overload
                else None
            )
            use_short_retry_budget = (
                fallback_target is not None
                and effective_model not in full_retry_budget_used
            )
            retry_config = (
                self._build_retry_config(self._fallback_retry_count)
                if use_short_retry_budget
                else self._retry_config
            )

            attempted_models.add(effective_model)

            try:
                return await self._complete_chat_request(
                    request,
                    retry_config=retry_config,
                    **current_kwargs,
                )
            except KernelLLMError as e:
                if use_short_retry_budget and not self._is_overload_fallback_error(e):
                    # Preserve the old retry behavior for non-overload failures:
                    # after the short downgrade budget is exhausted, retry the same
                    # model once more with the full configured retry policy.
                    full_retry_budget_used.add(effective_model)
                    attempted_models.discard(effective_model)
                    continue

                if (
                    not self._fallback_on_overload
                    or not self._is_overload_fallback_error(e)
                ):
                    raise

                if not await self._open_fallback_window(effective_model, e):
                    raise

    def _extract_rate_limit_headers(
        self, headers: dict[str, str] | Any
    ) -> dict[str, Any]:
        """Extract rate limit information from response headers.

        Anthropic returns rate limit headers on every response across
        multiple dimensions:
        - anthropic-ratelimit-requests-{limit,remaining,reset}
        - anthropic-ratelimit-tokens-{limit,remaining,reset}
        - anthropic-ratelimit-input-tokens-{limit,remaining,reset}
        - anthropic-ratelimit-output-tokens-{limit,remaining,reset}
        - retry-after (on 429 errors)

        Args:
            headers: Response headers (dict-like object)

        Returns:
            Dict with rate limit info, or empty dict if headers unavailable
        """
        if not headers:
            return {}

        # Helper to safely get integer header values
        def get_int(key: str) -> int | None:
            val = headers.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
            return None

        # Helper to safely get non-empty string header values (for reset timestamps)
        def get_str(key: str) -> str | None:
            val = headers.get(key)
            if val is not None and val != "":
                return str(val)
            return None

        info: dict[str, Any] = {}

        # Request limits
        requests_remaining = get_int("anthropic-ratelimit-requests-remaining")
        requests_limit = get_int("anthropic-ratelimit-requests-limit")
        requests_reset = get_str("anthropic-ratelimit-requests-reset")
        if requests_remaining is not None:
            info["requests_remaining"] = requests_remaining
        if requests_limit is not None:
            info["requests_limit"] = requests_limit
        if requests_reset is not None:
            info["requests_reset"] = requests_reset

        # Token limits (aggregate)
        tokens_remaining = get_int("anthropic-ratelimit-tokens-remaining")
        tokens_limit = get_int("anthropic-ratelimit-tokens-limit")
        tokens_reset = get_str("anthropic-ratelimit-tokens-reset")
        if tokens_remaining is not None:
            info["tokens_remaining"] = tokens_remaining
        if tokens_limit is not None:
            info["tokens_limit"] = tokens_limit
        if tokens_reset is not None:
            info["tokens_reset"] = tokens_reset

        # Input token limits (dimension-specific)
        input_tokens_remaining = get_int("anthropic-ratelimit-input-tokens-remaining")
        input_tokens_limit = get_int("anthropic-ratelimit-input-tokens-limit")
        input_tokens_reset = get_str("anthropic-ratelimit-input-tokens-reset")
        if input_tokens_remaining is not None:
            info["input_tokens_remaining"] = input_tokens_remaining
        if input_tokens_limit is not None:
            info["input_tokens_limit"] = input_tokens_limit
        if input_tokens_reset is not None:
            info["input_tokens_reset"] = input_tokens_reset

        # Output token limits (dimension-specific)
        output_tokens_remaining = get_int("anthropic-ratelimit-output-tokens-remaining")
        output_tokens_limit = get_int("anthropic-ratelimit-output-tokens-limit")
        output_tokens_reset = get_str("anthropic-ratelimit-output-tokens-reset")
        if output_tokens_remaining is not None:
            info["output_tokens_remaining"] = output_tokens_remaining
        if output_tokens_limit is not None:
            info["output_tokens_limit"] = output_tokens_limit
        if output_tokens_reset is not None:
            info["output_tokens_reset"] = output_tokens_reset

        # Fast-mode input token limits (present only when fast-mode is active)
        fast_input_tokens_remaining = get_int("anthropic-fast-input-tokens-remaining")
        fast_input_tokens_limit = get_int("anthropic-fast-input-tokens-limit")
        fast_input_tokens_reset = get_str("anthropic-fast-input-tokens-reset")
        if fast_input_tokens_remaining is not None:
            info["fast_input_tokens_remaining"] = fast_input_tokens_remaining
        if fast_input_tokens_limit is not None:
            info["fast_input_tokens_limit"] = fast_input_tokens_limit
        if fast_input_tokens_reset is not None:
            info["fast_input_tokens_reset"] = fast_input_tokens_reset

        # Fast-mode output token limits (present only when fast-mode is active)
        fast_output_tokens_remaining = get_int("anthropic-fast-output-tokens-remaining")
        fast_output_tokens_limit = get_int("anthropic-fast-output-tokens-limit")
        fast_output_tokens_reset = get_str("anthropic-fast-output-tokens-reset")
        if fast_output_tokens_remaining is not None:
            info["fast_output_tokens_remaining"] = fast_output_tokens_remaining
        if fast_output_tokens_limit is not None:
            info["fast_output_tokens_limit"] = fast_output_tokens_limit
        if fast_output_tokens_reset is not None:
            info["fast_output_tokens_reset"] = fast_output_tokens_reset

        # Retry-after (typically only on 429)
        if retry_after := headers.get("retry-after"):
            try:
                info["retry_after_seconds"] = float(retry_after)
            except (ValueError, TypeError):
                pass

        return info

    def _parse_rate_limit_info(self, error: AnthropicRateLimitError) -> dict[str, Any]:
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
        self,
        request: ChatRequest,
        retry_config: RetryConfig | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        active_retry_config = retry_config or self._retry_config

        logger.debug(
            f"Received ChatRequest with {len(request.messages)} messages (raw={self.raw})"
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

        # Resolve model and capabilities BEFORE building params dict,
        # so per-model param gating (temperature, output_config) can apply.
        effective_model = kwargs.get("model", self.default_model)
        request_caps = await self._get_request_capabilities(effective_model)
        model_ceiling = request_caps.max_output_tokens

        # Emit once-per-process deprecation warning for models nearing retirement
        if (
            effective_model in _DEPRECATED_MODELS
            and effective_model not in _warned_deprecated_models
        ):
            _warned_deprecated_models.add(effective_model)
            retire_date = _DEPRECATED_MODELS[effective_model]
            logger.warning(
                "[PROVIDER] Model %s is deprecated and will be retired on %s. "
                "Please migrate to a newer model.",
                effective_model,
                retire_date,
            )

        # Prepare request parameters
        params: dict[str, Any] = {
            "model": effective_model,
            "messages": all_messages,
            "max_tokens": request.max_output_tokens
            or kwargs.get("max_tokens", self.max_tokens),
        }

        # Only include temperature for models that support sampling.
        # Opus 4.7+ silently ignores temperature — omitting it avoids user confusion
        # and keeps request payloads clean.
        if request_caps.supports_sampling:
            params["temperature"] = (
                request.temperature
                if request.temperature is not None
                else kwargs.get("temperature", self.temperature)
            )
        else:
            if request.temperature is not None or kwargs.get("temperature") is not None:
                logger.info(
                    "[PROVIDER] Model %s does not support sampling parameters"
                    " — ignoring temperature setting",
                    params["model"],
                )

        if system_blocks:
            params["system"] = system_blocks

        # Add tools if provided
        if request.tools:
            tools = self._convert_tools_from_request(request.tools)
            params["tools"] = self._apply_tool_cache_control(tools)
            # Add tool_choice if specified
            if tool_choice := kwargs.get("tool_choice"):
                params["tool_choice"] = tool_choice

        # Add native web search tool if enabled (via config or kwargs)
        # This is a model-native tool that doesn't need function conversion
        web_search_enabled = kwargs.get("enable_web_search", self.enable_web_search)
        if web_search_enabled:
            web_search_tool = self._build_web_search_tool(kwargs)
            if "tools" not in params:
                params["tools"] = []
            # Add web search tool at the beginning (native tools typically come first)
            params["tools"].insert(0, web_search_tool)
            logger.info("[PROVIDER] Native web search tool enabled")
        resolved_thinking_type: str | None = None

        # Enable extended thinking if requested (equivalent to OpenAI's reasoning)
        #
        # Precedence chain (highest to lowest):
        #   1. kwargs["extended_thinking"]  — explicit per-request override
        #   2. request.reasoning_effort     — portable kernel interface (Phase 2)
        #   3. config defaults              — session-level settings
        #
        # kwargs["extended_thinking"]=False can disable thinking even when
        # reasoning_effort is set (explicit opt-out).
        thinking_enabled = bool(kwargs.get("extended_thinking"))

        # Phase 2: Check request.reasoning_effort when kwargs don't specify
        reasoning_effort = getattr(request, "reasoning_effort", None)
        # Phase 3: fall back to the provider's config-level `effort` default.
        # Lets users set effort once in their provider config (settings.yaml /
        # bundle `config:` block) instead of per-request or via kwargs.
        #
        # Two precedence chains are in play here and they are NOT the same:
        #   (1) reasoning_effort — drives extended thinking (on/off + depth) and,
        #       on Opus 4.7+, output_config.effort.  Precedence (highest wins):
        #           request.reasoning_effort > config["effort"]
        #   (2) kwargs["effort"] — an output_config.effort-ONLY override applied
        #       later (see the output_config block).  It does NOT feed this
        #       thinking path and does NOT enable thinking on its own.
        if reasoning_effort is None:
            config_effort = self.config.get("effort")
            if config_effort is not None:
                # Validate/normalise the config value so a typo (e.g. "ultra",
                # "High", "EXTRA HIGH") can't silently flip thinking on with a
                # value the ladder/output_config don't understand.
                normalized = str(config_effort).strip().lower()
                valid_efforts = ("low", "medium", "high", "xhigh", "max")
                if normalized in valid_efforts:
                    reasoning_effort = normalized
                else:
                    logger.warning(
                        "[PROVIDER] Ignoring invalid config 'effort'=%r "
                        "(valid values: %s)",
                        config_effort,
                        ", ".join(valid_efforts),
                    )
        if "extended_thinking" not in kwargs and reasoning_effort is not None:
            # reasoning_effort implies extended_thinking=True. This is a
            # deliberate Amplifier mapping (commit bc026a43): the portable
            # reasoning_effort hint enables Anthropic extended thinking, the
            # same way OpenAI's reasoning effort engages its reasoning. effort
            # and thinking are independent at the API level; coupling them is
            # Amplifier's "reason harder" product semantics.
            thinking_enabled = True

        thinking_budget = None
        interleaved_thinking_enabled = False
        if thinking_enabled:
            # Guard: skip thinking entirely for models that don't support it
            # (e.g. Haiku). Without this check we would send budget_tokens=0
            # which violates the API's >= 1024 minimum.
            if not request_caps.supports_thinking:
                logger.info(
                    "[PROVIDER] Model %s does not support extended thinking"
                    " — ignoring thinking request",
                    params["model"],
                )
                thinking_enabled = False

        if thinking_enabled:
            # Phase 2: reasoning_effort maps to thinking_type + budget_tokens.
            # This sits between kwargs (highest) and config (lowest) in precedence.
            #
            # | reasoning_effort | thinking_type | budget_tokens             |
            # |-----------------|---------------|---------------------------|
            # | "low"           | "enabled"     | 4096 (minimal thinking)   |
            # | "medium"        | "adaptive"*   | model default             |
            # | "high"          | "adaptive"*   | generous (model default)  |
            # | None            | (existing)    | (existing)                |
            # * falls back to "enabled" if model doesn't support adaptive
            # * On Opus 4.7+ "enabled" is intercepted → forced to "adaptive"
            #   (models without supports_manual_thinking reject type="enabled")

            # NOTE: effort_budget below is DEAD on adaptive-thinking models
            # (Opus 4.6+, Sonnet 4.6): those send thinking={"type":"adaptive"}
            # with no budget_tokens, so the budget is discarded and the real
            # intensity lever is output_config.effort. effort_budget only
            # matters on manual-thinking models (e.g. Haiku 4.5, Sonnet <4.6).
            effort_thinking_type: str | None = None
            effort_budget: int | None = None
            if reasoning_effort == "low":
                effort_thinking_type = "enabled"
                effort_budget = 4096
            elif reasoning_effort == "medium":
                effort_thinking_type = "adaptive"
                effort_budget = request_caps.default_thinking_budget
            elif reasoning_effort == "high":
                effort_thinking_type = "adaptive"
                effort_budget = request_caps.default_thinking_budget
            elif reasoning_effort == "xhigh":
                effort_thinking_type = "adaptive"
                effort_budget = request_caps.default_thinking_budget
            elif reasoning_effort == "max":
                # "max" (Opus 4.8+/Sonnet 4.6) uses adaptive thinking. This
                # branch only changes behaviour when a user set
                # config.thinking_type="enabled": it forces adaptive instead of
                # inheriting "enabled". (The resolved default is already
                # "adaptive", so without it "max" still resolves to adaptive.)
                # The real intensity for "max" is carried by output_config.effort.
                effort_thinking_type = "adaptive"
                effort_budget = request_caps.default_thinking_budget

            # Resolve budget: kwargs > reasoning_effort > config > model default
            budget_tokens = (
                kwargs.get("thinking_budget_tokens")
                or effort_budget
                or self.config.get("thinking_budget_tokens")
                or request_caps.default_thinking_budget
            )
            budget_tokens = max(1024, int(budget_tokens))
            max_budget_tokens = (
                model_ceiling if params.get("tools") else max(1024, model_ceiling - 1)
            )
            budget_tokens = min(budget_tokens, max_budget_tokens)
            # Default buffer raised from 4096 → 8192 to accommodate Opus 4.7's
            # denser tokenizer (1.0–1.35× more tokens for equivalent text).
            buffer_tokens = kwargs.get("thinking_budget_buffer") or self.config.get(
                "thinking_budget_buffer", 8192
            )

            thinking_budget = budget_tokens

            # Resolve thinking_type: kwargs > reasoning_effort > config > "adaptive"
            thinking_type = (
                kwargs.get("thinking_type")
                or effort_thinking_type
                or self.config.get("thinking_type", "adaptive")
            )

            # Adaptive thinking: model controls its own budget.  The API schema
            # is a discriminated union — "adaptive" accepts NO extra fields
            # (budget_tokens is forbidden).  Fall back to "enabled" with an
            # explicit budget when the model doesn't support adaptive.
            if thinking_type == "adaptive" and request_caps.supports_adaptive_thinking:
                params["thinking"] = {"type": "adaptive"}
                resolved_thinking_type = "adaptive"
            elif not request_caps.supports_manual_thinking:
                # Model rejects type="enabled" (e.g. Opus 4.7+) — force adaptive.
                # This is safe because models that don't support manual thinking
                # always support adaptive thinking.
                if thinking_type != "adaptive":
                    logger.info(
                        "[PROVIDER] Model %s does not support manual thinking "
                        "(type='enabled') — using adaptive instead of '%s'",
                        params["model"],
                        thinking_type,
                    )
                params["thinking"] = {"type": "adaptive"}
                resolved_thinking_type = "adaptive"
            else:
                # "enabled" mode (all thinking-capable models): explicit budget
                if thinking_type == "adaptive":
                    # Caller asked for adaptive but model doesn't support it
                    thinking_type = "enabled"
                resolved_thinking_type = thinking_type
                params["thinking"] = {
                    "type": thinking_type,
                    "budget_tokens": budget_tokens,
                }

            # For models where thinking.display defaults to "omitted" (Opus 4.7+),
            # request "summarized" so thinking content is visible to users.
            # Users can override via config or kwargs to "omitted" if desired.
            if request_caps.thinking_display_required:
                display = kwargs.get(
                    "thinking_display",
                    self.config.get("thinking_display", "summarized"),
                )
                params["thinking"]["display"] = display

            # Anthropic requires temperature=1.0 when thinking is enabled
            # on models that support sampling. Non-sampling models (4.7+)
            # ignore temperature entirely — don't inject it.
            if request_caps.supports_sampling:
                params["temperature"] = 1.0

            # Ensure max_tokens accommodates thinking budget + response.
            # For adaptive mode the model manages its own budget within
            # max_tokens, so we still need a generous ceiling.
            # Cap to the model's API-enforced output ceiling so we never
            # exceed what the backend allows (e.g. Opus 4.5 caps at 64K).
            target_tokens = min(budget_tokens + buffer_tokens, model_ceiling)
            if params.get("max_tokens"):
                params["max_tokens"] = min(
                    max(params["max_tokens"], target_tokens), model_ceiling
                )
            else:
                params["max_tokens"] = target_tokens

            interleaved_thinking_enabled = bool(params.get("tools"))

            logger.info(
                "[PROVIDER] Extended thinking enabled (budget=%s, buffer=%s, temperature=%s, max_tokens=%s, interleaved=%s)",
                thinking_budget,
                buffer_tokens,
                params.get("temperature", "n/a"),
                params["max_tokens"],
                interleaved_thinking_enabled,
            )

        if params.get("max_tokens") and params["max_tokens"] > model_ceiling:
            logger.info(
                "[PROVIDER] Clamping max_tokens from %s to %s for %s",
                params["max_tokens"],
                model_ceiling,
                params["model"],
            )
            params["max_tokens"] = model_ceiling

        # Build output_config for models that support it (Opus 4.7+).
        # output_config.effort is the primary control surface for thinking
        # intensity on these models, replacing the budget_tokens approach.
        if request_caps.supports_output_config and reasoning_effort is not None:
            # kwargs["effort"] allows overriding output_config.effort independently
            # of reasoning_effort (e.g. reasoning_effort="high" for thinking type,
            # but effort="xhigh" for output config intensity).
            effort = kwargs.get("effort", reasoning_effort)
            if effort in request_caps.supported_efforts:
                params["output_config"] = {"effort": effort}
                logger.info(
                    "[PROVIDER] output_config.effort=%s for %s",
                    effort,
                    params["model"],
                )
            else:
                logger.warning(
                    "[PROVIDER] Effort level '%s' not supported by %s "
                    "(supported: %s) — omitting output_config.effort",
                    effort,
                    params["model"],
                    request_caps.supported_efforts,
                )

        # Task budget (beta): output_config.task_budget for Opus 4.7+
        # COE CONSTRAINT: Use `is not None` (not `or`) to avoid falsy-zero bug.
        has_task_budget = False
        if request_caps.supports_task_budget:
            task_budget_tokens = kwargs.get("task_budget_tokens")
            if task_budget_tokens is None:
                task_budget_tokens = self.config.get("task_budget_tokens")
            if task_budget_tokens is not None:
                task_budget_tokens = max(20000, int(task_budget_tokens))
                if "output_config" not in params:
                    params["output_config"] = {}
                params["output_config"]["task_budget"] = {
                    "type": "tokens",
                    "total": task_budget_tokens,
                }
                has_task_budget = True
                logger.info(
                    "[PROVIDER] output_config.task_budget=%d for %s",
                    task_budget_tokens,
                    params["model"],
                )

        # Speed parameter (Opus 4.8+): inject into API params when model supports it.
        # Mirrors the supports_sampling pattern — if unsupported, log warning and omit.
        fast_mode_enabled = False
        speed = self.config.get("speed")
        if speed is not None:
            if request_caps.supports_speed:
                params["speed"] = speed
                fast_mode_enabled = speed == "fast"
                logger.info(
                    "[PROVIDER] speed=%s for %s",
                    speed,
                    params["model"],
                )
            else:
                logger.warning(
                    "[PROVIDER] Model %s does not support the speed parameter — omitting",
                    params["model"],
                )

        # Add stop_sequences if specified
        if stop_sequences := kwargs.get("stop_sequences"):
            params["stop_sequences"] = stop_sequences

        request_beta_headers = self._build_request_beta_headers(
            model_id=params["model"],
            request_caps=request_caps,
            tools_present=bool(params.get("tools")),
            resolved_thinking_type=resolved_thinking_type,
            has_task_budget=has_task_budget,
            fast_mode=fast_mode_enabled,
        )
        if request_beta_headers:
            extra_headers = dict(params.get("extra_headers", {}))
            extra_headers["anthropic-beta"] = ",".join(request_beta_headers)
            params["extra_headers"] = extra_headers

        logger.info(
            f"[PROVIDER] Anthropic API call - model: {params['model']}, messages: {len(params['messages'])}, system: {bool(system_blocks)}, tools: {len(params.get('tools', []))}, thinking: {thinking_enabled}"
        )

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            request_payload: dict[str, Any] = {
                "provider": "anthropic",
                "model": params["model"],
                "message_count": len(params["messages"]),
                "has_system": bool(system_blocks),
                "thinking_enabled": thinking_enabled,
                "thinking_budget": thinking_budget,
                "interleaved_thinking": interleaved_thinking_enabled,
            }
            if self.raw:
                request_payload["raw"] = redact_secrets(params)
            await self.coordinator.hooks.emit("llm:request", request_payload)

        start_time = time.time()

        # Call Anthropic API with shared retry_with_backoff from amplifier-core.
        # Error translation happens inside _do_complete() so that retry_with_backoff
        # sees LLMError (and checks retryable) rather than raw SDK exceptions.

        # Mutable container for rate_limit_info captured inside _do_complete
        captured_rate_limit_info: dict[str, Any] = {}

        async def _do_complete():
            """Single API call attempt with SDK → kernel error translation."""
            nonlocal captured_rate_limit_info
            try:
                # Use streaming API to support large context windows
                # (Anthropic requires streaming for operations > 10 min)
                rate_limit_info: dict[str, Any] = {}

                # Per-request non-streaming override via request.metadata:
                #   metadata={"stream": False}
                # Callers (e.g. session-naming background tasks) that must NOT
                # emit llm:stream_* events set this flag.  It overrides
                # self.use_streaming for this single call only — the shared
                # provider instance's default behavior is completely unchanged.
                _metadata = getattr(request, "metadata", None)
                _use_streaming = self.use_streaming
                if isinstance(_metadata, dict) and _metadata.get("stream") is False:
                    _use_streaming = False

                if _use_streaming:
                    # ----- Streaming path with per-block event emission --------
                    # We iterate the SDK's event stream rather than calling
                    # get_final_message() directly, so we can emit the full
                    # block lifecycle (start/delta/end) on the hook bus. The
                    # SDK still accumulates internally; get_final_message()
                    # after the loop returns the complete assembled Message.
                    #
                    # Events emitted on the hook bus, per content block
                    # (v3 — separate streaming-lifecycle channel):
                    #   llm:stream_block_start   when a new block begins (with
                    #                            block_type so the renderer knows
                    #                            to open a Live region or print a
                    #                            placeholder)
                    #   llm:stream_block_delta   for each text_delta AND thinking_delta
                    #                            fragment (block_type in payload
                    #                            distinguishes text vs thinking)
                    #   llm:stream_block_end     when the block streaming completes
                    #
                    # These events are on a SEPARATE channel from the atomic
                    # renderer's content_block:start/end events (synthesized by
                    # loop-streaming from the assembled response). The streaming
                    # overlay subscribes to llm:stream_* only. The atomic renderer
                    # subscribes to content_block:* only. No payload field-parity
                    # requirement between the two channels — eliminates the
                    # regression class that produced missing total_blocks/usage.
                    #
                    # If the stream aborts mid-flight (timeout / disconnect /
                    # mid-stream API error) and we already emitted at least one
                    # delta, we also emit llm:stream_aborted before re-raising
                    # so the renderer hook can close any open Live regions
                    # cleanly.
                    request_id = str(uuid.uuid4())
                    block_sequences: dict[int, int] = {}
                    block_types: dict[int, str] = {}
                    partial_emitted = False
                    hooks_available = self.coordinator and hasattr(
                        self.coordinator, "hooks"
                    )
                    try:
                        async with asyncio.timeout(self.timeout):
                            async with self.client.messages.stream(
                                **params
                            ) as stream:
                                async for event in stream:
                                    etype = type(event).__name__
                                    idx = getattr(event, "index", None)
                                    if etype == "RawContentBlockStartEvent":
                                        if idx is None:
                                            continue
                                        block = getattr(event, "content_block", None)
                                        btype = (
                                            getattr(block, "type", "text")
                                            if block is not None
                                            else "text"
                                        )
                                        block_types[idx] = btype
                                        if hooks_available:
                                            payload: dict[str, Any] = {
                                                "request_id": request_id,
                                                "block_index": idx,
                                                "block_type": btype,
                                            }
                                            # Tool-use blocks carry a name so the
                                            # streaming overlay's placeholder can
                                            # show "Building tool call: <name>..."
                                            if btype == "tool_use" and block is not None:
                                                name = getattr(block, "name", None)
                                                if name:
                                                    payload["name"] = name
                                            await self.coordinator.hooks.emit(
                                                "llm:stream_block_start",
                                                payload,
                                            )
                                    elif etype == "RawContentBlockDeltaEvent":
                                        delta = getattr(event, "delta", None)
                                        if delta is None or idx is None:
                                            continue
                                        seq = block_sequences.get(idx, 0)
                                        block_sequences[idx] = seq + 1
                                        dtype = getattr(delta, "type", "")
                                        if dtype == "text_delta":
                                            text = getattr(delta, "text", "") or ""
                                            if text and hooks_available:
                                                await self.coordinator.hooks.emit(
                                                    "llm:stream_block_delta",
                                                    {
                                                        "request_id": request_id,
                                                        "block_index": idx,
                                                        "block_type": block_types.get(
                                                            idx, "text"
                                                        ),
                                                        "sequence": seq,
                                                        "text": text,
                                                    },
                                                )
                                                partial_emitted = True
                                        elif dtype == "thinking_delta":
                                            text = (
                                                getattr(delta, "thinking", "") or ""
                                            )
                                            if text and hooks_available:
                                                await self.coordinator.hooks.emit(
                                                    "llm:stream_block_delta",
                                                    {
                                                        "request_id": request_id,
                                                        "block_index": idx,
                                                        "block_type": block_types.get(
                                                            idx, "thinking"
                                                        ),
                                                        "sequence": seq,
                                                        "text": text,
                                                    },
                                                )
                                                partial_emitted = True
                                        # signature_delta and any future delta
                                        # types are observed silently — the
                                        # SDK still accumulates them into the
                                        # final message.
                                    elif etype in (
                                        "ParsedContentBlockStopEvent",
                                        "RawContentBlockStopEvent",
                                    ):
                                        if idx is None:
                                            continue
                                        if hooks_available:
                                            btype_end = block_types.get(idx, "text")
                                            await self.coordinator.hooks.emit(
                                                "llm:stream_block_end",
                                                {
                                                    "request_id": request_id,
                                                    "block_index": idx,
                                                    "block_type": btype_end,
                                                },
                                            )
                                    # All other event types (RawMessageStart,
                                    # ParsedMessageStop, SignatureEvent, etc.)
                                    # flow through the SDK's internal
                                    # accumulator and are not surfaced.

                                # Stream drained. Final message is now ready.
                                response = await stream.get_final_message()

                                # Capture rate limit headers from stream response
                                if hasattr(stream, "response") and stream.response:
                                    rate_limit_info = self._extract_rate_limit_headers(
                                        stream.response.headers
                                    )
                    except Exception as e:
                        # Mid-stream failure. If we emitted any partial output,
                        # tell the renderer so it can close any open Live
                        # regions cleanly. Then re-raise so the outer except
                        # clauses below translate the SDK error to a kernel
                        # error type.
                        if partial_emitted and hooks_available:
                            await self.coordinator.hooks.emit(
                                "llm:stream_aborted",
                                {
                                    "request_id": request_id,
                                    "error": {
                                        "type": type(e).__name__,
                                        "msg": str(e),
                                    },
                                },
                            )
                        raise
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

                captured_rate_limit_info = rate_limit_info
                return response

            except AnthropicRateLimitError as e:
                rate_info = self._parse_rate_limit_info(e)
                retry_after = rate_info.get("retry_after_seconds")
                body = getattr(e, "body", None)
                msg = json.dumps(body) if body is not None else str(e)
                raise KernelRateLimitError(
                    msg,
                    provider="anthropic",
                    model=params["model"],
                    status_code=429,
                    retryable=True,
                    retry_after=retry_after,
                ) from e

            except AnthropicAuthenticationError as e:
                body = getattr(e, "body", None)
                msg = json.dumps(body) if body is not None else str(e)
                raise KernelAuthenticationError(
                    msg,
                    provider="anthropic",
                    model=params["model"],
                    status_code=getattr(e, "status_code", 401),
                ) from e

            except AnthropicBadRequestError as e:
                raw_msg = str(e).lower()
                body = getattr(e, "body", None)
                error_msg = json.dumps(body) if body is not None else str(e)
                if "context length" in raw_msg or "too many tokens" in raw_msg:
                    raise KernelContextLengthError(
                        error_msg,
                        provider="anthropic",
                        model=params["model"],
                        status_code=getattr(e, "status_code", 400),
                    ) from e
                elif (
                    "content filter" in raw_msg
                    or "safety" in raw_msg
                    or "blocked" in raw_msg
                ):
                    raise KernelContentFilterError(
                        error_msg,
                        provider="anthropic",
                        model=params["model"],
                        status_code=getattr(e, "status_code", 400),
                    ) from e
                else:
                    raise KernelInvalidRequestError(
                        error_msg,
                        provider="anthropic",
                        model=params["model"],
                        status_code=getattr(e, "status_code", 400),
                    ) from e

            except AnthropicOverloadedError as e:
                body = getattr(e, "body", None)
                error_msg = json.dumps(body) if body is not None else str(e)
                retry_after: float | None = None
                if hasattr(e, "response") and e.response:
                    raw = e.response.headers.get("retry-after")
                    if raw is not None:
                        try:
                            retry_after = float(raw)
                        except (ValueError, TypeError):
                            pass
                raise KernelProviderUnavailableError(
                    error_msg,
                    provider="anthropic",
                    model=params["model"],
                    status_code=529,
                    retryable=True,
                    retry_after=retry_after,
                    delay_multiplier=self._overloaded_delay_multiplier,
                ) from e

            except AnthropicAPIStatusError as e:
                status = getattr(e, "status_code", 500)
                body = getattr(e, "body", None)
                error_msg = json.dumps(body) if body is not None else str(e)
                if status == 403:
                    # Distinguish Cloudflare bot challenges (transient) from
                    # real API 403s (permanent).  Cloudflare returns HTML
                    # challenge pages that the SDK can't parse as JSON, so
                    # e.body is None and content-type is text/html.
                    if self._is_cloudflare_challenge(e):
                        logger.warning(
                            "[PROVIDER] Cloudflare challenge detected (HTTP 403 "
                            "with HTML body). Treating as transient — will retry."
                        )
                        if self.coordinator and hasattr(self.coordinator, "hooks"):
                            await self.coordinator.hooks.emit(
                                "provider:cloudflare_challenge",
                                {
                                    "provider": "anthropic",
                                    "model": params["model"],
                                    "active_requests": _active_requests,
                                    "waiting_requests": _waiting_requests,
                                    "max_concurrent": self._max_concurrent_requests,
                                    "process_id": os.getpid(),
                                    "timestamp": time.time(),
                                },
                            )
                        raise KernelProviderUnavailableError(
                            "Cloudflare bot challenge (transient 403 with HTML body). "
                            "This typically resolves on retry.",
                            provider="anthropic",
                            model=params["model"],
                            status_code=403,
                            retryable=True,
                        ) from e
                    raise KernelAccessDeniedError(
                        error_msg,
                        provider="anthropic",
                        model=params["model"],
                        status_code=403,
                    ) from e
                if status == 404:
                    raise KernelNotFoundError(
                        error_msg,
                        provider="anthropic",
                        model=params["model"],
                        status_code=404,
                    ) from e
                if status >= 500:
                    raise KernelProviderUnavailableError(
                        error_msg,
                        provider="anthropic",
                        model=params["model"],
                        status_code=status,
                        retryable=True,
                    ) from e
                raise KernelLLMError(
                    error_msg,
                    provider="anthropic",
                    model=params["model"],
                    status_code=status,
                    retryable=False,
                ) from e

            except asyncio.TimeoutError as e:
                raise KernelLLMTimeoutError(
                    f"Request timed out after {self.timeout}s",
                    provider="anthropic",
                    model=params["model"],
                    retryable=True,
                ) from e

            except KernelLLMError:
                raise  # Already translated, don't double-wrap

            except Exception as e:
                body = getattr(e, "body", None)
                error_msg = (
                    json.dumps(body)
                    if body is not None
                    else (str(e) or f"{type(e).__name__}: (no message)")
                )
                raise KernelLLMError(
                    error_msg,
                    provider="anthropic",
                    model=params["model"],
                    retryable=True,
                ) from e

        async def _on_retry(attempt: int, delay: float, error: KernelLLMError):
            """Callback invoked before each retry sleep."""
            error_type = type(error).__name__
            retry_after = getattr(error, "retry_after", None)

            # Always log retries at WARNING level — visible even without hooks
            logger.warning(
                "[PROVIDER] Retry %d/%d for %s: %s, sleeping %.1fs%s",
                attempt,
                active_retry_config.max_retries,
                error_type,
                str(error),
                delay,
                f" (server retry-after: {retry_after}s)" if retry_after else "",
            )

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    PROVIDER_RETRY,
                    {
                        "provider": "anthropic",
                        "model": params["model"],
                        "attempt": attempt,
                        "max_retries": active_retry_config.max_retries,
                        "delay": delay,
                        "retry_after": retry_after,
                        "error_type": error_type,
                        "error_message": str(error),
                    },
                )

        async def _do_complete_guarded():
            """Semaphore-gated wrapper around _do_complete with concurrency logging.

            Acquires the process-wide concurrency semaphore before each API call
            attempt so that at most ``max_concurrent_requests`` calls are in-flight
            simultaneously across all provider instances in this process.

            This is the function passed to retry_with_backoff so that:
            - the semaphore is *released* between retry attempts (during backoff sleep)
            - each fresh attempt must re-acquire before hitting the network
            """
            global _active_requests, _waiting_requests
            sem = await _get_process_semaphore(self._max_concurrent_requests)
            if sem is not None:
                _waiting_requests += 1
                async with sem:
                    _waiting_requests -= 1
                    _active_requests += 1
                    try:
                        if self.coordinator and hasattr(self.coordinator, "hooks"):
                            await self.coordinator.hooks.emit(
                                "provider:concurrency",
                                {
                                    "provider": "anthropic",
                                    "model": params["model"],
                                    "active_requests": _active_requests,
                                    "waiting_requests": _waiting_requests,
                                    "max_concurrent": self._max_concurrent_requests,
                                    "process_id": os.getpid(),
                                },
                            )
                        return await _do_complete()
                    finally:
                        _active_requests -= 1
            else:
                # Semaphore disabled (max_concurrent_requests=0) — still log
                _active_requests += 1
                try:
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        await self.coordinator.hooks.emit(
                            "provider:concurrency",
                            {
                                "provider": "anthropic",
                                "model": params["model"],
                                "active_requests": _active_requests,
                                "waiting_requests": _waiting_requests,
                                "max_concurrent": 0,
                                "process_id": os.getpid(),
                            },
                        )
                    return await _do_complete()
                finally:
                    _active_requests -= 1

        # Read shared rate-limit state from cross-process file before the
        # throttle check so we also account for capacity consumed by sibling
        # processes on the same API key (e.g. parallel sessions, Docker containers).
        self._read_shared_rate_limit_state()

        # Pre-emptive throttle check: if we're running low on any rate limit
        # dimension, inject a delay and warn the user before hitting a 429.
        if self._throttle_threshold > 0:
            ratio, dimension, remaining, limit, reset_ts = (
                self._rate_limit_state.most_constrained_ratio()
            )
            if ratio < self._throttle_threshold and remaining is not None:
                # Calculate delay: use reset timestamp if available, else fallback
                delay = self._throttle_delay
                if reset_ts:
                    try:
                        from datetime import datetime, timezone

                        reset_time = datetime.fromisoformat(
                            reset_ts.replace("Z", "+00:00")
                        )
                        seconds_until_reset = (
                            reset_time - datetime.now(timezone.utc)
                        ).total_seconds()
                        if seconds_until_reset > 0:
                            delay = min(seconds_until_reset, 60.0)  # Cap at 60s
                    except (ValueError, TypeError):
                        pass  # Fall back to default delay

                # Always log throttle at WARNING level — visible even without hooks
                logger.warning(
                    "[PROVIDER] Throttling: %s at %.1f%% remaining (%s/%s), sleeping %.1fs",
                    dimension,
                    ratio * 100,
                    remaining,
                    limit,
                    delay,
                )

                # Emit throttle event so CLI can warn the user
                if self.coordinator and hasattr(self.coordinator, "hooks"):
                    await self.coordinator.hooks.emit(
                        PROVIDER_THROTTLE,
                        {
                            "provider": "anthropic",
                            "model": params["model"],
                            "reason": f"{dimension}_low",
                            "dimension": dimension,
                            "remaining": remaining,
                            "limit": limit,
                            "ratio": ratio,
                            "reset_timestamp": reset_ts,
                            "delay": delay,
                        },
                    )

                await asyncio.sleep(delay)

        try:
            response = await retry_with_backoff(
                _do_complete_guarded,
                active_retry_config,
                on_retry=_on_retry,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from Anthropic API")
            logger.debug(f"[PROVIDER] Response type: {response.model}")

            # Log rate limit status if available
            rate_limit_info = captured_rate_limit_info
            # Update throttle state for next request's pre-emptive check
            self._rate_limit_state.update_from_headers(rate_limit_info)
            # Write shared state so sibling processes can see current capacity.
            if rate_limit_info:
                self._write_shared_rate_limit_state(rate_limit_info)
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

            # Build ChatResponse first
            chat_response = self._convert_to_chat_response(response)

            # Emit from canonical fields
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # Build usage dict per #69 schema — is-not-None guards for cache fields
                _event_usage: dict[str, Any] = {
                    "input_tokens": chat_response.usage.input_tokens,
                    "output_tokens": chat_response.usage.output_tokens,
                }
                if chat_response.usage.cache_read_tokens is not None:
                    _event_usage["cache_read_tokens"] = (
                        chat_response.usage.cache_read_tokens
                    )
                if chat_response.usage.cache_write_tokens is not None:
                    _event_usage["cache_write_tokens"] = (
                        chat_response.usage.cache_write_tokens
                    )
                _cost = chat_response.usage.cost_usd
                _event_usage["cost_usd"] = str(_cost) if _cost is not None else None
                response_event: dict[str, Any] = {
                    "provider": "anthropic",
                    "model": params["model"],
                    "duration_ms": elapsed_ms,
                    "status": "ok",
                    "usage": _event_usage,
                }
                # Add rate limit info if available
                if rate_limit_info:
                    response_event["rate_limits"] = rate_limit_info
                if self.raw:
                    response_event["raw"] = redact_secrets(response.model_dump())
                await self.coordinator.hooks.emit("llm:response", response_event)

            return chat_response  # Return the already-built response

        except KernelLLMError as e:
            # Phase 2: Kernel error types — emit llm:response error event, then propagate
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error("[PROVIDER] Anthropic API error: %s", error_msg)

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
            raise

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            # Ensure error message is never empty
            error_msg = str(e) or f"{type(e).__name__}: (no message)"
            logger.error(f"[PROVIDER] Anthropic response processing error: {error_msg}")

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
            # Skip tool calls with truly missing arguments (None).
            # Empty dict {} is valid -- many tools take no arguments.
            if tc.arguments is None:
                logger.debug(f"Filtering out tool '{tc.name}' with None arguments")
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
        if block_type == "web_search_tool_result":
            # Web search results are model-native and should be passed through
            # with minimal cleaning (just remove internal fields)
            cleaned: dict[str, Any] = {
                "type": "web_search_tool_result",
            }
            if "tool_use_id" in block:
                cleaned["tool_use_id"] = block["tool_use_id"]
            if "content" in block:
                cleaned["content"] = block["content"]
            return cleaned
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
            # ALSO scan content blocks for tool_use/tool_call entries.
            # On session resume, synthetic tool results are injected by complete() before
            # _convert_messages() runs. If the content blocks contain tool_use IDs that
            # don't appear in tool_calls (format mismatch), the defensive filter at line
            # ~1585 drops the synthetic results as "orphaned", causing a 400 from Anthropic.
            # Scanning content blocks here makes the valid-ID set robust to any such mismatch.
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_use" and block.get("id"):
                                valid_tool_use_ids.add(block["id"])
                            elif block.get("type") == "tool_call" and block.get("id"):
                                valid_tool_use_ids.add(block["id"])

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

        Handles both standard function tools (converted to Anthropic format) and
        model-native tools like web_search_20250305 (passed through unchanged).

        Model-native tools are identified by having a 'type' attribute that is NOT
        'function'. These tools use Anthropic's built-in capabilities and should
        NOT be converted to the standard function tool format.

        Args:
            tools: List of ToolSpec objects or native tool definitions

        Returns:
            List of Anthropic-formatted tool definitions
        """
        anthropic_tools = []
        for tool in tools:
            # Check if this is a model-native tool (has 'type' that's not 'function')
            # Native tools like web_search_20250305 are passed through unchanged
            tool_type = getattr(tool, "type", None)
            if tool_type and tool_type != "function":
                # Model-native tool - pass through as-is (converted to dict if needed)
                if hasattr(tool, "model_dump"):
                    anthropic_tools.append(tool.model_dump(exclude_none=True))
                elif isinstance(tool, dict):
                    anthropic_tools.append(tool)
                else:
                    # Fallback: build dict from known attributes
                    native_tool: dict[str, Any] = {"type": tool_type}
                    if hasattr(tool, "name") and tool.name:
                        native_tool["name"] = tool.name
                    # Add any additional config (e.g., max_uses for web search)
                    if hasattr(tool, "max_uses") and tool.max_uses is not None:
                        native_tool["max_uses"] = tool.max_uses
                    if (
                        hasattr(tool, "user_location")
                        and tool.user_location is not None
                    ):
                        native_tool["user_location"] = tool.user_location
                    anthropic_tools.append(native_tool)
                logger.debug(f"[PROVIDER] Added native tool: {tool_type}")
            else:
                # Standard function tool - convert to Anthropic format
                anthropic_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.parameters,
                    }
                )
        return anthropic_tools

    def _extract_web_search_citations(self, block: Any) -> list[dict[str, Any]]:
        """Extract citation information from a web search result block.

        Web search results contain citations with source information that can be
        displayed to users for transparency and attribution.

        Args:
            block: Web search tool result block from Anthropic response

        Returns:
            List of citation dicts with title, url, and optional snippet
        """
        citations = []

        # Web search results have a 'content' field with search results
        content = getattr(block, "content", None)
        if not content:
            return citations

        # Content may be a list of result items or a single object
        results = content if isinstance(content, list) else [content]

        for result in results:
            # Each result may have source information
            if hasattr(result, "type") and result.type == "web_search_result":
                citation: dict[str, Any] = {}

                # Extract URL (required)
                if hasattr(result, "url") and result.url:
                    citation["url"] = result.url
                elif hasattr(result, "source_url") and result.source_url:
                    citation["url"] = result.source_url

                # Extract title
                if hasattr(result, "title") and result.title:
                    citation["title"] = result.title

                # Extract snippet/description
                if hasattr(result, "snippet") and result.snippet:
                    citation["snippet"] = result.snippet
                elif hasattr(result, "description") and result.description:
                    citation["snippet"] = result.description
                elif hasattr(result, "encrypted_content") and result.encrypted_content:
                    # Some results use encrypted_content - just note it exists
                    citation["has_content"] = True

                # Only add if we have at least a URL
                if citation.get("url"):
                    citations.append(citation)

        return citations

    def _build_web_search_tool(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build the native web search tool definition.

        The web_search_20250305 tool is a model-native tool that enables Claude
        to search the web for current information. Unlike function tools, it uses
        Anthropic's built-in web search capability.

        Tool definition format:
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,  # optional, limits searches per request
                "user_location": {...}  # optional, for location-aware results
            }

        Args:
            kwargs: Request kwargs that may contain web search configuration

        Returns:
            Web search tool definition dict
        """
        tool: dict[str, Any] = {
            "type": "web_search_20250305",
            "name": "web_search",  # Anthropic requires this exact name
        }

        # Optional: max_uses limits number of searches per request
        max_uses = kwargs.get("web_search_max_uses") or self.config.get(
            "web_search_max_uses"
        )
        if max_uses is not None:
            tool["max_uses"] = max_uses

        # Optional: user_location for location-aware search results
        user_location = kwargs.get("web_search_user_location") or self.config.get(
            "web_search_user_location"
        )
        if user_location is not None:
            tool["user_location"] = user_location

        return tool

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
        web_search_results: list[dict[str, Any]] = []
        event_blocks: list[
            TextContent | ThinkingContent | ToolCallContent | WebSearchContent
        ] = []
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
            elif block.type == "web_search_tool_result":
                # Handle native web search results from Anthropic
                # Extract citations from search results for observability
                citations = self._extract_web_search_citations(block)
                web_search_results.append(
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": getattr(block, "tool_use_id", None),
                        "citations": citations,
                    }
                )
                # Add to event blocks for UI display
                event_blocks.append(
                    WebSearchContent(
                        query=getattr(block, "query", ""),
                        citations=citations,
                    )
                )
                logger.debug(
                    f"[PROVIDER] Web search returned {len(citations)} citations"
                )

        # Build usage with named kernel fields + provider-native extras for
        # backward compatibility.  reasoning_tokens is intentionally None:
        # Anthropic does not provide a separate reasoning token count (thinking
        # tokens are included in output_tokens).
        input_tokens = response.usage.input_tokens + (
            getattr(response.usage, "cache_read_input_tokens", None) or 0
        )
        output_tokens = response.usage.output_tokens

        cache_creation = (
            getattr(response.usage, "cache_creation_input_tokens", None) or None
        )
        cache_read = getattr(response.usage, "cache_read_input_tokens", None) or None

        usage_kwargs: dict[str, Any] = {
            # Required fields
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            # Named kernel fields (Phase 2)
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_creation,
        }

        # Keep provider-native extras for backward compat (extra="allow" on Usage)
        if cache_creation is not None:
            usage_kwargs["cache_creation_input_tokens"] = cache_creation
        if cache_read is not None:
            usage_kwargs["cache_read_input_tokens"] = cache_read

        usage = Usage(**usage_kwargs)

        cost = compute_cost(
            response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_input_tokens=getattr(
                response.usage, "cache_read_input_tokens", 0
            )
            or 0,
            cache_creation_input_tokens=getattr(
                response.usage, "cache_creation_input_tokens", 0
            )
            or 0,
            speed=getattr(response.usage, "speed", None),
        )
        usage = usage.model_copy(update={"cost_usd": cost})
        self._add_cost(cost)

        combined_text = "\n\n".join(text_accumulator).strip()

        return AnthropicChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=response.stop_reason,
            content_blocks=event_blocks if event_blocks else None,
            text=combined_text or None,
            web_search_results=web_search_results if web_search_results else None,
        )

    async def close(self) -> None:
        """Close the underlying Anthropic client to prevent resource leaks."""
        if self._client is not None:
            try:
                await asyncio.shield(self._client.close())
            except asyncio.CancelledError:
                pass
