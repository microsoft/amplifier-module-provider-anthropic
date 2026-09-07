"""Anthropic provider module for Amplifier.

Integrates with Anthropic's Claude API for Claude models (Sonnet, Opus, Haiku).
Supports streaming, tool calling, extended thinking, and ChatRequest format.
"""

__all__ = ["mount", "AnthropicProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import difflib
import hashlib
import json
import logging
import os
import re
import time
import uuid
from decimal import Decimal
from threading import Lock
from typing import Any
from typing import ClassVar

from collections import OrderedDict
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
from anthropic import Timeout as AnthropicTimeout
from anthropic._exceptions import (
    OverloadedError as AnthropicOverloadedError,
)  # Not exported in public API as of SDK v0.96.0 (private import still works)

from ._cost import compute_cost

# Params the Messages API still accepts on the wire but the SDK does not expose
# as typed keyword arguments.
#
#   temperature -- removed from the typed Messages surface in anthropic 1.0.0
#                  (0 occurrences anywhere in the 1.0.0 package). The API still
#                  honors it; only the SDK signature dropped it.
#   speed       -- never present in the typed surface on any 0.x or 1.x release,
#                  though the API accepts it when the fast-mode beta header is
#                  sent. Passing it as a keyword has always raised TypeError.
#
# Both must travel in extra_body. Sending them as keywords raises
# "got an unexpected keyword argument", which the retry loop then treats as a
# transient failure and retries five times before surfacing.
_WIRE_ONLY_PARAMS: tuple[str, ...] = ("temperature", "speed")


def _route_wire_only_params(params: dict[str, Any]) -> dict[str, Any]:
    """Relocate wire-only params from the typed surface into ``extra_body``.

    Mutates and returns ``params``. A key already present in ``extra_body``
    wins -- an explicit caller-supplied override is not clobbered by the
    value this module derived.
    """
    for key in _WIRE_ONLY_PARAMS:
        if key not in params:
            continue
        value = params.pop(key)
        extra_body = dict(params.get("extra_body") or {})
        extra_body.setdefault(key, value)
        params["extra_body"] = extra_body
    return params


# Messages API parameters the installed SDK exposes as TYPED keyword
# arguments. Anything NOT in this set must travel in extra_body: passing an
# unknown keyword raises "got an unexpected keyword argument", which
# _do_complete's catch-all translates into a RETRYABLE KernelLLMError -- so a
# permanent config typo would be retried max_retries times before surfacing.
# See the _WIRE_ONLY_PARAMS comment above for the incident this prevents.
#
# Verified live against the INSTALLED SDK (anthropic==1.0.0, this repo's
# floor pin) via inspect.signature(AsyncMessages.create) -- not assumed from
# API docs. `top_k`, `top_p`, and `betas` are NOT present on 1.0.0's typed
# surface (the same fate as `temperature`, already documented above: dropped
# from the typed Messages surface in the 1.0.0 major bump) -- listing them
# here would route a config value onto the typed surface and reproduce
# exactly the unexpected-keyword-argument retry bug this allowlist exists to
# prevent. Guarded by test_sdk_contract.py::TestTypedRequestParamsMatchSdkSignature,
# which fails loud on any future SDK drift instead of silently trusting this
# list.
_TYPED_REQUEST_PARAMS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "system",
        "tools",
        "tool_choice",
        "stop_sequences",
        "stream",
        "metadata",
        "thinking",
        "output_config",
        "service_tier",
        "cache_control",
        "container",
        "inference_geo",
        "user_profile_id",
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout",
    }
)


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
# Teardown hard bound (see AnthropicProvider.close)
# ---------------------------------------------------------------------------
# `httpx.AsyncClient.aclose()` -- what `AsyncAnthropic.close()` awaits
# (anthropic/_base_client.py) -- has NO deadline of its own. On a half-closed
# (CLOSE-WAIT) connection it can block indefinitely, and session cleanup runs
# BEFORE a CLI command returns its result, so an unbounded close turns a
# finished run into a silent hang (measured 28 minutes; recipes-8sr evidence).
# Overridable per instance via the `close_timeout` config key.
_DEFAULT_CLOSE_TIMEOUT: float = 5.0


def _retrieve_task_exception(task: "asyncio.Future[Any]") -> None:
    """Consume an abandoned close task's exception.

    Without this, a close task we stopped awaiting that later fails makes
    asyncio log "Task exception was never retrieved" at GC time -- noise
    that reads like a new defect. Cancelled tasks have nothing to retrieve.
    """
    if not task.cancelled():
        task.exception()


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
# Model-native (server-side) tool types that require an anthropic-beta header.
#
# A native tool is declared on the wire as {"type": "<tool_type>", ...} rather
# than as a function schema. Anthropic gates several of these behind a beta
# header, and omitting it makes the API reject the whole request. The tool type
# is the only thing that determines which header is needed, so the provider can
# derive it rather than requiring every caller to know the mapping and inject
# the header itself.
#
# Without this, a caller that wants a beta-gated native tool has to reach into
# the provider's private `_beta_headers` to add the header - a pattern that
# breaks whenever the attribute is renamed and fails silently when it does.
#
# Tool types absent from this map need no beta header (e.g. web_search_20250305,
# which is generally available) and are passed through unchanged.
NATIVE_TOOL_BETA_HEADERS: dict[str, str] = {
    "computer_20241022": "computer-use-2024-10-22",
    "computer_20250124": "computer-use-2025-01-24",
    "computer_20251124": "computer-use-2025-11-24",
}

BETA_HEADER_INTERLEAVED_THINKING = "interleaved-thinking-2025-05-14"
BETA_HEADER_TASK_BUDGETS = "task-budgets-2026-03-13"
BETA_HEADER_FAST_MODE = "fast-mode-2026-02-01"
PROVIDER_FALLBACK_OPEN = "provider:fallback_open"
PROVIDER_FALLBACK_ACTIVE = "provider:fallback_active"
FALLBACK_STATE_VERSION = 1

# Overload/refusal fallback ladder: which family a request steps DOWN to.
#
# Signed chain: fable -> opus -> sonnet -> haiku. `mythos` is a top-tier PEER
# of fable (Anthropic prices Claude Mythos 5 identically to Claude Fable 5 at
# $10/MTok base, 2026-08-29), so it enters the ladder at the same rung rather
# than between fable and opus -- this keeps the signed fable->opus edge exact.
#
# "haiku" is deliberately ABSENT: it is the terminal rung. A missing key means
# "no lower tier exists", which is a real answer, not an omission.
#
# This same ladder now drives BOTH overload fallback AND refusal fallback
# (owner-adjudicated: Anthropic's own guidance for a refusal is to retry
# with a less-restrictive model, and there is a product expectation that
# fallback never lands on a model MORE expensive than the one the user
# selected -- which the old hardcoded refusal-escalation target violated
# for sonnet/haiku users). See _refusal_fallback_target.
_FALLBACK_NEXT_FAMILY: dict[str, str] = {
    "fable": "opus",
    "mythos": "opus",
    "opus": "sonnet",
    "sonnet": "haiku",
}

# Backstop targets, used when neither an explicit per-family override nor a
# live list_models() refresh has supplied one. Values are the newest GA model
# in each family as of 2026-08-29. Going stale here degrades gracefully (the
# fallback still works, aimed at a slightly older model) -- it never fails.
# fable/mythos never appear as VALUES: nothing steps UP the ladder.
_STATIC_FALLBACK_MODELS: dict[str, str] = {
    "opus": "claude-opus-5",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
}

# ---------------------------------------------------------------------------
# Context-overflow detection markers
# ---------------------------------------------------------------------------
# Anthropic returns context-window overflow as HTTP 400
# invalid_request_error. There is no machine-readable code for it -- the
# error.type is the generic "invalid_request_error" shared with every other
# 400 -- so the message is the only discriminator. Two shapes exist:
#   "prompt is too long: 208310 tokens > 200000 maximum"
#   "input length and `max_tokens` exceed context limit: 189127 + 16000 > 200000, ..."
# The legacy markers are retained so gateways that rewrite the message
# (and other providers' phrasing) still classify correctly.
_CONTEXT_OVERFLOW_MESSAGE_MARKERS = (
    "prompt is too long",
    "exceed context limit",
    "context length",
    "too many tokens",
    "maximum context",
    "context window",
)


def _is_context_overflow(raw_msg: str) -> bool:
    """True when an Anthropic 400 denotes context-window overflow."""
    return any(m in raw_msg for m in _CONTEXT_OVERFLOW_MESSAGE_MARKERS)


# redact_secrets() (amplifier_core.utils) only redacts dict values keyed by a
# sensitive name -- it passes plain strings straight through unchanged. The
# httpx/httpcore exceptions surfaced via __cause__ are plain strings, and a
# base_url with embedded basic-auth (https://user:pass@proxy.internal/) shows
# up verbatim inside them (e.g. in the "Request URL is missing/invalid ..."
# text). Strip that userinfo before it reaches a log line or KernelLLMError.
#
# The scheme prefix ("://") is OPTIONAL, not required: GAP-016's whole reason
# for existing is a malformed/missing-protocol base_url, and that is exactly
# the case where httpx's own "Request URL ... is missing an 'http://' or
# 'https://' protocol" text echoes the configured value back with NO "://" in
# front of it. A pattern anchored on "://" never fires for the one input this
# redaction exists to catch.
#
# Making "://" optional everywhere would also swallow an ordinary bare email
# address (e.g. "contact admin@example.com") appearing in unrelated
# diagnostic text -- that's over-redaction, destroying diagnostic value for
# no security benefit. So the no-scheme branch additionally REQUIRES a colon
# in the userinfo (user:pass@, :pass@, user:@). An email's local part never
# contains a colon, so this cleanly separates "credentials" from "email"
# without needing scheme context. A bare token with no colon AND no scheme
# (e.g. "token@host") is genuinely ambiguous -- it looks exactly like an
# email's user@host shape -- and is deliberately left unredacted rather than
# risk destroying real diagnostic text on a guess; the scheme-present form of
# the same bare-token case (https://token@host) is still fully covered below.
#
# One pattern (not two) handles both shapes via Python's conditional-group
# syntax `(?(scheme)yes|no)`, keeping the "when does this fire" logic in a
# single place instead of two regexes a future edit could let drift apart.
# The replacement is a function (not a fixed string) because the correct
# output differs by branch: re-emit "://" only if it was actually present in
# the input, so a no-scheme value doesn't gain a fake "://" it never had.
_URL_CREDENTIALS_RE = re.compile(
    r"(?P<scheme>://)?"
    r"(?(scheme)"
    r"[^/@\s:]*(?::[^/@\s]*)?"  # scheme present: user, user:pass, :pass, or user: all count
    r"|"
    r"[A-Za-z0-9_.~%+=-]*:[A-Za-z0-9_.~%+=-]*"  # no scheme: colon in userinfo required
    r")"
    r"@"
)


def _redact_url_credentials_match(match: re.Match[str]) -> str:
    scheme = match.group("scheme") or ""
    return f"{scheme}[REDACTED]@"


def _redact_url_credentials(text: str) -> str:
    """Strip embedded basic-auth credentials (user:pass@) from any URL in text.

    Catches both `scheme://user:pass@host` and the scheme-less
    `user:pass@host` form (see the regex comment above for why the latter is
    the case that actually matters for GAP-016).
    """
    return _URL_CREDENTIALS_RE.sub(_redact_url_credentials_match, text)


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
    manual_thinking_deprecated: bool = (
        False  # True = type="enabled" still works (HTTP 200) but is deprecated --
        # verified live 2026-08-29 (T-C08-live) on Opus 4.6. Only meaningful
        # when supports_manual_thinking is True (the two never both apply on
        # the same model: 4.7+/5+ already hard-gate to adaptive).
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
    thinking_always_on: bool = (
        False  # True = thinking is always active; NEVER send thinking:{type:disabled}
    )
    supports_native_computer_use: bool = False  # True = model accepts a "computer_*" native tool type (see computer_use_tool_type)
    computer_use_tool_type: str | None = (
        None  # Which "computer_*" wire type this model accepts -- Anthropic's versioned
        # computer-use tool has three incompatible generations (computer_20241022,
        # computer_20250124, computer_20251124) and a model paired with the wrong one
        # is rejected outright (HTTP 400), unlike OpenAI's single bare "computer" type.
        # None means this model does not support the tool at all.
    )
    capability_tags: tuple[str, ...] = ("tools", "streaming", "json_mode")
    min_cacheable_tokens: int = (
        1024  # Below this, the API silently skips caching (no error) --
        # platform.claude.com/en/docs/build-with-claude/prompt-caching,
        # verified 2026-08-29. Per-family values set explicitly below;
        # this is only the dataclass fallback.
    )


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
    # close() handles the lazy-client guard, the shield, CancelledError, and
    # the hard `close_timeout` bound -- this cleanup runs inside the finally
    # that precedes a CLI command's return, so it must never block forever.
    async def cleanup():
        await provider.close()

    return cleanup


# ---------------------------------------------------------------------------
# Config-key allowlist -- unknown-key sweep with did-you-mean (D-01..D-04)
# ---------------------------------------------------------------------------
# Every config key this module actually reads -- audited against every
# `self.config.get(...)` call site, in BOTH the constructor AND the deferred
# request path. The eight *_build_params / _build_web_search_tool keys below
# are the reason this list cannot be derived from __init__ alone: they are
# never touched at construction, and an allowlist built only from the
# constructor would false-positive on every extended-thinking user.
#
# NOTE: fallback_sonnet_model/fallback_haiku_model/refusal_fallback_model are
# still consumed at this point in the release sequence -- they are moved to
# _INERT_CONFIG_KEY_MESSAGES once the fallback-ladder commits that replace
# them land (fallback_models supersedes the first two; the ladder itself
# supersedes the third).
_CONSUMED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        # --- constructor / mount ---
        "api_key",
        "base_url",
        "default_model",
        "max_tokens",
        "temperature",
        "priority",
        "raw",
        "timeout",
        "close_timeout",
        "reasoning_effort",
        "max_retries",
        "min_retry_delay",
        "max_retry_delay",
        "retry_jitter",
        "overloaded_delay_multiplier",
        "fallback_on_overload",
        "fallback_retry_count",
        "fallback_cooldown_seconds",
        "persist_fallback_state",
        "fallback_state_path",
        "fallback_models",
        "refusal_fallback_enabled",
        "throttle_threshold",
        "throttle_delay",
        "rate_limit_state_path",
        "max_concurrent_requests",
        "use_streaming",
        "filtered",
        "enable_1m_context",
        "enable_prompt_caching",
        "cache_stable_region_ttl_1h",
        "cache_infer_stability_from_history",
        "enable_web_search",
        "beta_headers",
        "extra_request_params",
        # --- deferred: read in _build_params (:3056-3230 pre-overhaul) ---
        "extended_thinking",
        "thinking_budget_tokens",
        "thinking_budget_buffer",
        "thinking_type",
        "thinking_display",
        "task_budget_tokens",
        "speed",
        # --- deferred: read in _build_web_search_tool ---
        "web_search_max_uses",
        "web_search_user_location",
    }
)

# Infrastructure keys the app/kernel places in (or alongside) a provider's
# config block that this module does not itself read: metadata fields a
# settings-shaped provider entry carries. `api_key` and `default_model` are
# already in _CONSUMED_CONFIG_KEYS (this module reads both).
_INFRASTRUCTURE_CONFIG_KEYS: frozenset[str] = frozenset({"id", "module", "source"})

# How many distinct conversations one provider instance tracks request
# fingerprints for (see `cache_infer_stability_from_history`). A provider
# instance is shared across a session's sub-agents and utility calls, so this
# is an LRU, not a per-session slot. 32 covers deep delegation trees with room
# to spare; each entry is a list of 32-char digests, so the whole map is
# kilobytes even at the limit.
_MAX_TRACKED_CONVERSATIONS = 32

# Keys that are recognized but currently do nothing. Each gets its own
# targeted warning naming what to do instead, and is therefore *excluded*
# from the generic unknown-key sweep so no key ever produces two warnings.
#
# `debug` and `raw_debug` are ghosts: grep for '"debug"|raw_debug' across
# this module returns zero matches. They exist only in README.md's
# now-deleted "Debug Configuration" section, which documented two options
# that were never implemented. Anyone who followed the README has them set
# and gets nothing -- they must warn, not vanish silently.
_INERT_CONFIG_KEY_MESSAGES: dict[str, str] = {
    "debug": (
        "not consumed by provider-anthropic and has no effect. It was never "
        "implemented; the README documented it in error. For request/response "
        "payload capture use `raw: true`."
    ),
    "raw_debug": (
        "not consumed by provider-anthropic and has no effect. See `debug`; "
        "use `raw: true` for full request/response payloads in llm:request / "
        "llm:response events."
    ),
    "fallback_sonnet_model": (
        "removed -- fallback targets are now resolved from the model ladder "
        "(fable/mythos -> opus -> sonnet -> haiku). To pin a specific target, "
        "use `fallback_models: {sonnet: <model-id>}`."
    ),
    "fallback_haiku_model": (
        "removed -- see fallback_sonnet_model. Use "
        "`fallback_models: {haiku: <model-id>}` to pin a target."
    ),
    "refusal_fallback_model": (
        "removed -- refusals now follow the same never-more-expensive "
        "downgrade ladder as overload (fable/mythos -> opus -> sonnet -> "
        "haiku), resolved via `fallback_models` the same way overload "
        "fallback is. `refusal_fallback_enabled` still gates whether a "
        "refusal is retried at all."
    ),
}
_RECOGNIZED_INERT_CONFIG_KEYS: frozenset[str] = frozenset(_INERT_CONFIG_KEY_MESSAGES)

# Deprecated aliases: still fully functional, but the canonical key should be
# used. Must stay "known" or they would draw a SECOND, generic unknown-key
# warning on top of the targeted deprecation warning each already gets.
_DEPRECATED_ALIAS_CONFIG_KEYS: dict[str, str] = {
    "effort": "reasoning_effort",
}

_KNOWN_CONFIG_KEYS: frozenset[str] = (
    _CONSUMED_CONFIG_KEYS
    | _RECOGNIZED_INERT_CONFIG_KEYS
    | frozenset(_DEPRECATED_ALIAS_CONFIG_KEYS)
    | _INFRASTRUCTURE_CONFIG_KEYS
)


def _warn_unknown_config_keys(
    config: dict[str, Any], extra_known: frozenset[str] = frozenset()
) -> None:
    """Warn on any config key that is neither consumed, recognized-inert,
    a deprecated alias, nor an infrastructure key -- with a did-you-mean
    suggestion drawn from the full known-key set.

    `extra_known` lets a subclass (via `EXTRA_KNOWN_CONFIG_KEYS`) extend the
    allowlist without needing to fork this function.
    """
    known = _KNOWN_CONFIG_KEYS | extra_known
    for key in config:
        if key in known:
            continue
        suggestion = difflib.get_close_matches(key, known, n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        logger.warning(
            "[PROVIDER] Unknown config key '%s' for provider-anthropic.%s",
            key,
            hint,
        )


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

    # Zero-cost extension point for subclasses that add their own config
    # keys: provider-openai learned in practice that it needed one after
    # the fact (its own unknown-key sweep shipped without it first). An
    # empty frozenset here costs nothing and avoids that retrofit.
    EXTRA_KNOWN_CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset()

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
        self.default_model = self.config.get("default_model", "claude-sonnet-5")
        self._default_caps = self._get_capabilities(self.default_model)

        # Effort-family config keys. Canonical key: "reasoning_effort" (matches
        # the kernel's portable request.reasoning_effort). Legacy alias:
        # "effort". Both are consumed in _build_params; when both are set the
        # canonical key wins. "effort" is a DEPRECATED alias (D-03): still
        # fully functional, but now warns even when used alone -- it
        # previously stayed silent unless BOTH keys were set, which meant an
        # "effort"-only config never learned it should migrate.
        if "effort" in self.config:
            if self.config.get("reasoning_effort") is not None:
                logger.warning(
                    "[PROVIDER] Both 'reasoning_effort' and 'effort' are set in "
                    "config; 'reasoning_effort' (canonical) wins and 'effort'=%r "
                    "is ignored. Remove 'effort'.",
                    self.config.get("effort"),
                )
            else:
                logger.warning(
                    "[PROVIDER] Config key 'effort' is a DEPRECATED alias for "
                    "'reasoning_effort' (the canonical key, matching the "
                    "kernel's portable request.reasoning_effort). It still "
                    "works; rename it to 'reasoning_effort'.",
                )

        # Targeted messages for recognized-but-inert keys (D-02), then the
        # generic unknown-key sweep with did-you-mean (D-04). Runs after all
        # config is read but before any other work, so it covers every key
        # this constructor is ever going to look at.
        for _inert_key, _inert_msg in _INERT_CONFIG_KEY_MESSAGES.items():
            if _inert_key in self.config:
                logger.warning(
                    "[PROVIDER] Config key '%s' is %s", _inert_key, _inert_msg
                )
        _warn_unknown_config_keys(self.config, self.EXTRA_KNOWN_CONFIG_KEYS)
        # Numeric config keys are coerced via the warn-and-default helpers
        # below (_config_int / _config_float), not raw self.config.get().
        # Before this, a settings.yaml string value like max_tokens: '8192'
        # stayed a str all the way to the wire, and a typo'd value (e.g.
        # overloaded_delay_multiplier: 'ten') raised ValueError at mount --
        # killing the whole provider instance instead of warning and using a
        # safe default. Every numeric key gets the same shape: fail soft,
        # log loudly, never crash mount on a config typo.
        self.max_tokens = self._config_int(
            self.config.get("max_tokens"), self._default_caps.max_output_tokens
        )
        self.temperature = self._config_float(self.config.get("temperature"), 0.7)
        self.priority = self._config_int(self.config.get("priority"), 100)
        self.raw = self._config_bool(
            self.config.get("raw", False)
        )  # Include raw payload in events
        self.timeout = self._config_float(
            self.config.get("timeout"), 600.0
        )  # API timeout in seconds (default 10 minutes)
        # Hard bound on teardown's httpx aclose() -- see close(). This is NOT
        # `timeout` above: that one bounds an API request, this one bounds
        # closing the connection pool at session cleanup, where a half-closed
        # (CLOSE-WAIT) connection can otherwise block forever.
        self._close_timeout = self._config_float(
            self.config.get("close_timeout"), _DEFAULT_CLOSE_TIMEOUT
        )

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
        self._overloaded_delay_multiplier = self._config_float(
            self.config.get("overloaded_delay_multiplier"), 10.0
        )

        # Temporary model downgrade on persistent overloads.
        # When enabled, a higher-tier family gets a short retry budget; if it still
        # overloads, a process-wide cooldown window routes subsequent requests to
        # the configured lower-tier model until the cooldown expires.
        self._fallback_on_overload = self._config_bool(
            self.config.get("fallback_on_overload", False)
        )
        # Defaults corrected to the values actually used in practice: 4 of 5
        # real deployed instances set retry_count='2' (one sets '3') and
        # cooldown_seconds='300' explicitly. A 30-minute cooldown after a
        # transient capacity blip stranded a whole session on a downgraded
        # model long after real capacity had returned.
        self._fallback_retry_count = max(
            0, self._config_int(self.config.get("fallback_retry_count", 2), 2)
        )
        self._fallback_cooldown_seconds = max(
            0.0,
            self._config_float(
                self.config.get("fallback_cooldown_seconds", 300.0), 300.0
            ),
        )
        # 1M context is GA, DEFAULT, and billed at STANDARD PRICING on every
        # model that has it (Opus 5/4.8/4.7/4.6, Sonnet 5/4.6, Fable 5/5.1,
        # Mythos 5/Preview) -- verified against
        # platform.claude.com/en/docs/build-with-claude/context-windows on
        # 2026-08-29: "For every model with a 1M-token context window, 1M is
        # the default: you don't need a beta header, and long-context
        # requests are billed at standard pricing."
        #
        # There is NO long-context premium tier. Do not add one to _cost.py.
        # The context-1m-2025-08-07 beta header is gone from Anthropic's
        # docs and has been removed from this provider (C-01); sending an
        # unrecognised beta header is a hard 400.
        #
        # This flag's ONLY remaining effect is the ADVERTISED context window
        # handed to the context manager (get_info().defaults["context_window"]
        # and ModelInfo.context_window in list_models()) -- i.e. how much
        # conversation history the caller keeps per request, which is a COST
        # decision, not a capability one. Default flipped True -> False: an
        # honest default now that the flag no longer changes what the API
        # will accept, only how much history is kept (and therefore billed
        # for) per request.
        self._enable_1m_context = self._config_bool(
            self.config.get("enable_1m_context", False)
        )

        # Fallback target resolution: fallback_models (settings-only,
        # per-family override) supersedes the two deleted scalar keys
        # (fallback_sonnet_model / fallback_haiku_model). See
        # _resolve_fallback_model for the full three-source precedence and
        # _INERT_CONFIG_KEY_MESSAGES for the migration message the two
        # retired keys now produce.
        _raw_fallback_models = self.config.get("fallback_models")
        self._fallback_models: dict[str, str] = {}
        if _raw_fallback_models:
            if isinstance(_raw_fallback_models, dict):
                for _fam, _target in _raw_fallback_models.items():
                    if _fam not in _FALLBACK_NEXT_FAMILY and _fam != "haiku":
                        _suggestion = difflib.get_close_matches(
                            str(_fam),
                            (*_FALLBACK_NEXT_FAMILY.values(), *_FALLBACK_NEXT_FAMILY),
                            n=1,
                        )
                        _hint = (
                            f" Did you mean '{_suggestion[0]}'?" if _suggestion else ""
                        )
                        logger.warning(
                            "[PROVIDER] Unknown family '%s' in config "
                            "'fallback_models' -- ignoring.%s",
                            _fam,
                            _hint,
                        )
                        continue
                    if _target:
                        self._fallback_models[str(_fam)] = str(_target)
            else:
                raise ValueError(
                    f"Invalid config 'fallback_models'={_raw_fallback_models!r} for "
                    f"provider-anthropic: must be a mapping of family name "
                    f"(opus/sonnet/haiku) to model id "
                    f"(got {type(_raw_fallback_models).__name__}). Fix the "
                    f"provider config (settings.yaml / bundle config block)."
                )
        # Live-refreshed "newest model per family" cache -- see
        # _resolve_fallback_model (source 2) and _warm_family_latest.
        self._family_latest: dict[str, str] = {}
        self._family_latest_attempted = False

        self._persist_fallback_state = self._config_bool(
            self.config.get("persist_fallback_state", False)
        )

        # Mount-time LOUD warning: fallback_on_overload has NO EFFECT on an
        # instance whose default_model is already on the ladder's terminal
        # rung (haiku) -- there is no lower tier to downgrade to. The wizard
        # already hides this ConfigField for Haiku (show_when), so this
        # specifically catches hand-edited settings.yaml.
        if self._fallback_on_overload:
            _fam = self._detect_family(self.default_model)
            if _fam not in _FALLBACK_NEXT_FAMILY:
                logger.warning(
                    "[PROVIDER] fallback_on_overload is enabled for "
                    "default_model=%r (family %r), but %r is the lowest tier "
                    "on the Anthropic fallback ladder -- there is no "
                    "lower-tier model to downgrade to, so this setting has "
                    "NO EFFECT. Remove it, or point this instance at a "
                    "higher-tier model (fable/mythos/opus/sonnet).",
                    self.default_model,
                    _fam,
                    _fam,
                )

        # Refusal fallback: when a model returns finish_reason="refusal", retry
        # once against the SAME fallback ladder overload fallback uses.
        # Orthogonal to the overload-fallback machinery above -- this is a
        # single-shot retry triggered by response content, not by request
        # errors. `refusal_fallback_model` (a hardcoded escalation target,
        # always Opus) is RETIRED (owner-adjudicated): refusals now follow
        # the same never-more-expensive downgrade ladder as overload,
        # resolved via the same three-source precedence
        # (fallback_models override -> live list_models cache -> static
        # backstop). See _refusal_fallback_target.
        self._refusal_fallback_enabled = self._config_bool(
            self.config.get("refusal_fallback_enabled", True)
        )

        # Pre-emptive throttle configuration
        # Threshold: fraction of remaining capacity below which we inject a delay.
        # Default 0.02 (2%) — only throttle when nearly exhausted, not at 10%.
        # Delay: fallback sleep when no reset timestamp is available.
        # Default 1.0s — just enough to ease pressure without punishing every request.
        self._throttle_threshold = self._config_float(
            self.config.get("throttle_threshold"), 0.02
        )
        self._throttle_delay = self._config_float(
            self.config.get("throttle_delay"), 1.0
        )
        self._rate_limit_state = _RateLimitState()

        # Process-wide concurrency gate.
        # Limits how many API calls this process has in-flight simultaneously,
        # shared across ALL provider instances (parent + delegated child sessions).
        # This prevents blast patterns (e.g. parallel: true recipes spawning 20+
        # concurrent calls) that trigger Cloudflare bot-detection on api.anthropic.com.
        # Set to 0 to disable the semaphore entirely.
        self._max_concurrent_requests = self._config_int(
            self.config.get("max_concurrent_requests"), 5
        )

        # extra_request_params: the documented escape hatch for Messages API
        # parameters this provider does not model. Merged into `params` LAST
        # (see _merge_extra_request_params / _build_params), so it overrides
        # every value the provider computed -- deliberately. Never a
        # ConfigField; settings-only.
        _extra = self.config.get("extra_request_params") or {}
        if not isinstance(_extra, dict):
            raise ValueError(  # noqa: TRY004 -- mount-time config-validation contract
                f"Invalid config 'extra_request_params'={_extra!r} for "
                f"provider-anthropic: must be a mapping of Messages API "
                f"parameter names to values (got {type(_extra).__name__}). "
                f"Fix the provider config (settings.yaml / bundle config block)."
            )
        self.extra_request_params: dict[str, Any] = dict(_extra)
        self._extra_params_warned_keys: set[str] = set()

        # Use streaming API by default to support large context windows (Anthropic requires streaming
        # for operations that may take > 10 minutes, e.g. with 300k+ token contexts)
        self.use_streaming = self._config_bool(self.config.get("use_streaming", True))
        self.filtered = self._config_bool(
            self.config.get("filtered", True)
        )  # Filter to curated model list by default
        self.enable_prompt_caching = self._config_bool(
            self.config.get("enable_prompt_caching", True)
        )
        self.enable_web_search = self._config_bool(
            self.config.get("enable_web_search", False)
        )  # Enable native web search tool

        # Extended (1-hour) TTL for the stable system/tools cache breakpoints.
        #
        # Default OFF. The system prompt + tool definitions are the most stable,
        # least-often-changing part of the request, so a 1h TTL is *structurally*
        # justified for them (unlike the conversation region, which changes most
        # turns). But it is not free: Anthropic bills 1h-TTL cache *writes* at a
        # higher (2x vs 1.25x) multiplier than the 5m default. Rather than
        # silently changing billing behavior, this stays opt-in until a
        # deployment has measured that its system prompt is stable long enough
        # (and reused often enough) for the longer TTL to pay for itself.
        #
        # C-10: NO beta header is required or sent for this. Anthropic's
        # prompt-caching docs (platform.claude.com/en/docs/build-with-claude/
        # prompt-caching, verified 2026-08-29) document the mechanism as the
        # `ttl` field alone: "To use the extended cache, include ttl in the
        # cache_control definition". The `extended-cache-ttl-2025-04-11` beta
        # header does not appear anywhere on that page, and was confirmed live
        # (2026-08-29) to be unnecessary: a request with `ttl: "1h"` and no
        # beta header produces `cache_creation.ephemeral_1h_input_tokens > 0`
        # exactly as expected. Sending it is harmless (also confirmed live --
        # no 400), but there is no reason to keep code that adds a header the
        # current docs don't mention.
        self.cache_stable_region_ttl_1h = self._config_bool(
            self.config.get("cache_stable_region_ttl_1h", False)
        )

        # Infer conversation stability by OBSERVING consecutive requests, when
        # the `Message.metadata` ephemeral contract is not populated.
        #
        # Default ON. Without it, a deployment whose orchestrator never stamps
        # `metadata={"ephemeral": True}` gets NO conversation-region breakpoint
        # at all -- `_apply_conversation_cache_control`'s `has_ephemeral_signal`
        # guard skips the whole region rather than risk a breakpoint on
        # unstable content. Measured cost of that dead end on a real
        # coding-agent node visit (164 provider calls, one session): cache_read
        # frozen at a single constant (10,995 = system + tools) for every call
        # while uncached input climbed 24K -> 228K; 1.79M cache_read against
        # 24.7M uncached input = a 6.8% hit ratio, ~26.5M input tokens billed.
        #
        # The inference replaces a *declaration* the orchestrator may never
        # make with a *measurement* this provider can always take: fingerprint
        # each request's message array, and on the next request in the same
        # conversation compare against it. `len(previous) - longest_common_
        # prefix` is the observed unstable-suffix length -- exactly the
        # quantity `_unstable_suffix_length` tries to derive from metadata,
        # but evidenced instead of asserted. It is used as a floor (max) with
        # the metadata value, so it can only ever make placement MORE
        # conservative, never less.
        #
        # Set False to revert to strict metadata-only behavior.
        self.cache_infer_stability_from_history = self._config_bool(
            self.config.get("cache_infer_stability_from_history", True)
        )

        # conversation key -> (message fingerprints, last observed unstable
        # suffix length). Bounded LRU: a long-lived provider instance may
        # serve many interleaved conversations, and this must never grow
        # without limit. Values are short hex digests, not message content.
        self._prefix_fingerprints: OrderedDict[
            str, tuple[list[str], int | None]
        ] = OrderedDict()

        # Get base_url from config for custom endpoints (proxies, local APIs, etc.)
        #
        # GAP-016: settings.yaml commonly stores this as an env-var template
        # (e.g. ``base_url: ${ANTHROPIC_BASE_URL}``). amplifier-app-cli's
        # expand_env_vars() substitutes an *unset* referenced variable with
        # "" (empty string), not None -- by design, so that provider
        # instances the user isn't actively using this session don't crash
        # config loading just because one of their optional env vars isn't
        # set. But "" is never a valid base_url: passed straight to
        # AsyncAnthropic(base_url=""), httpx/httpcore raises
        # `UnsupportedProtocol: Request URL is missing an 'http://' or
        # 'https://' protocol`, which the SDK re-wraps as a generic
        # `APIConnectionError("Connection error.")` -- indistinguishable
        # from a real network failure and hitting every call this client
        # makes (list_models() preflight *and* the primary completion).
        # An empty string is never a meaningful custom endpoint, so treat it
        # the same as "not configured" and fall back to the SDK's real
        # default (https://api.anthropic.com), exactly as if base_url had
        # never been set at all.
        raw_base_url = self.config.get("base_url")
        self._base_url = raw_base_url if raw_base_url else None

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

        if self.cache_stable_region_ttl_1h and not self.enable_prompt_caching:
            # The knob only affects cache breakpoints, which are never
            # placed at all when prompt caching is off. Log once so a user
            # who set this expecting an effect isn't left guessing why
            # nothing changed.
            logger.info(
                "[PROVIDER] cache_stable_region_ttl_1h is set but "
                "enable_prompt_caching is False -- the 1h cache TTL "
                "knob has no effect without prompt caching enabled."
            )

        if self._beta_headers:
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
            #
            # `timeout` must be passed. Two things break when it is omitted:
            #
            #   1. A configured `timeout` is silently ignored -- the SDK falls
            #      back to its own default and long single-turn streams die at
            #      that default even when the operator asked for more.
            #
            #   2. Without it, `self._client.timeout == DEFAULT_TIMEOUT` stays
            #      true, which arms a client-side guard in the SDK's Messages
            #      resource: for a NON-streaming call it estimates the request
            #      duration from `max_tokens` alone and raises
            #      "Streaming is required for operations that may take longer
            #      than 10 minutes" before issuing any HTTP request. The
            #      estimate is `3600 * max_tokens / 128_000 > 600`, i.e. any
            #      `max_tokens` above 21,333 -- and `self.max_tokens` defaults
            #      to the model's full output ceiling (64k-128k), so every
            #      non-streaming call is refused unless the caller also lowered
            #      `max_tokens`. Passing an explicit timeout skips that guess
            #      entirely; the guard exists to estimate a bound we already
            #      know and already enforce ourselves via `asyncio.wait_for` /
            #      `asyncio.timeout` on both completion paths.
            #
            # Pass a Timeout rather than a bare float: a bare float applies to
            # every phase, stretching connect from the SDK's 5s to the full
            # request timeout. `connect=5.0` mirrors what the SDK itself builds
            # in `_calculate_nonstreaming_timeout`.
            #
            # Imported from `anthropic`, not from the underlying HTTP package.
            # The SDK re-exports its own timeout type, so this survives another
            # transport swap like the 1.0 move from httpx to httpx2 -- a bare
            # `import httpx` is precisely what broke on that upgrade.
            self._client = AsyncAnthropic(
                api_key=self._api_key,
                base_url=self._base_url,
                default_headers=self._default_headers,
                max_retries=0,
                timeout=AnthropicTimeout(self.timeout, connect=5.0),
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
                    prompt="Anthropic API key",
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
                    prompt=(
                        "Use the full 1M-token context window "
                        "(more context kept per request = higher cost)"
                    ),
                    required=False,
                    # Default flipped true -> false alongside the constructor
                    # default (both must change together or the wizard and
                    # the constructor disagree) -- see the enable_1m_context
                    # comment in __init__ for the full C-01/C-02 rationale.
                    default="false",
                    requires_model=True,  # Shown after model selection
                    show_when={
                        "default_model": "not_contains:haiku"
                    },  # Hide for Haiku (doesn't support 1M)
                ),
                ConfigField(
                    # Canonical key -- matches the kernel's portable
                    # request.reasoning_effort and the key used by the shipped
                    # routing matrices. ConfigField.id is written verbatim into
                    # settings.yaml, so this must be the canonical name, never
                    # the legacy "effort" alias.
                    id="reasoning_effort",
                    display_name="Reasoning Effort",
                    field_type="choice",
                    choices=["low", "medium", "high", "xhigh", "max"],
                    prompt="Reasoning effort -- higher is smarter, slower, costlier",
                    required=False,
                    requires_model=True,  # Shown after model selection
                    # Gate on the EFFECT surface (extended thinking), not on
                    # output_config support: effort enables/sizes thinking on
                    # every thinking-capable model, so hide it only for models
                    # that don't support thinking at all (pre-4.5 Haiku).
                    show_when={"default_model": "not_contains:haiku-3"},
                ),
                ConfigField(
                    id="cache_stable_region_ttl_1h",
                    display_name="1-Hour Cache TTL (Stable Regions)",
                    field_type="boolean",
                    prompt="1-hour cache TTL -- 2x write cost, fewer writes",
                    required=False,
                    # No declared default -- an unset field is a real,
                    # visible third state ("use provider default") distinct
                    # from an explicit False, not just "off by default"
                    # spelled a different way. The app-cli wizard renders a
                    # None-default boolean as "(leave unset -- use provider
                    # default)" and omits the key entirely when left blank;
                    # the constructor already treats an absent key as False
                    # via `self.config.get("cache_stable_region_ttl_1h",
                    # False)` above, so behavior is unchanged either way.
                    default=None,
                    requires_model=False,
                    # show_when REMOVED (A-04): enable_prompt_caching is no
                    # longer a ConfigField (demoted to settings-only below),
                    # so the condition could never be satisfied in the
                    # wizard. The inert combination (this on, prompt caching
                    # off) is already handled loudly by the constructor
                    # guard, which logs and continues.
                ),
            ],
        )

    async def list_models(
        self, retry_config: RetryConfig | None = None
    ) -> list[ModelInfo]:
        """
        List available Claude models dynamically from Anthropic API.

        When filtered=True (default), returns only the latest version of each
        model family (e.g. fable, opus, sonnet, haiku). When filtered=False,
        returns all available Claude models.

        The query is retried with the same shared retry_with_backoff()/
        _retry_config machinery used by complete() on transient failures
        (5xx, connection errors, timeouts, rate limits). Raises the
        translated kernel error once retries are exhausted, or immediately
        for non-retryable errors (401/403/404) -- no fallback; caller
        handles empty lists.

        `retry_config` lets a caller (namely `_warm_family_latest`) use a
        cheaper retry policy than the instance default -- passed explicitly
        rather than by mutating `self._retry_config`, which would not be
        concurrency-safe under the process-wide semaphore.

        Returns:
            List of ModelInfo for available Claude models.
        """
        active_retry_config = retry_config or self._retry_config

        async def _do_list_models():
            """Single API call attempt with SDK -> kernel error translation.

            Mirrors the error-translation branches used by _do_complete()
            (rate limit, authentication, status errors 403/404/5xx, and a
            catch-all for connection/timeout errors) so list_models() shares
            the same retry policy as complete().
            """
            try:
                return await self.client.models.list()
            except AnthropicRateLimitError as e:
                rate_info = self._parse_rate_limit_info(e)
                retry_after = rate_info.get("retry_after_seconds")
                body = getattr(e, "body", None)
                msg = json.dumps(body) if body is not None else str(e)
                raise KernelRateLimitError(
                    msg,
                    provider="anthropic",
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
                    status_code=getattr(e, "status_code", 401),
                ) from e
            except AnthropicAPIStatusError as e:
                # Also catches AnthropicOverloadedError (529), a subclass of
                # AnthropicAPIStatusError -- it falls into the status >= 500
                # branch below and is retried like any other 5xx.
                status = getattr(e, "status_code", 500)
                body = getattr(e, "body", None)
                error_msg = json.dumps(body) if body is not None else str(e)
                if status == 403:
                    # Distinguish Cloudflare bot challenges (transient) from
                    # real API 403s (permanent), same detection _do_complete uses.
                    if self._is_cloudflare_challenge(e):
                        logger.warning(
                            "[PROVIDER] Cloudflare challenge detected (HTTP 403 "
                            "with HTML body) on list_models(). Treating as "
                            "transient -- will retry."
                        )
                        raise KernelProviderUnavailableError(
                            "Cloudflare bot challenge (transient 403 with HTML "
                            "body). This typically resolves on retry.",
                            provider="anthropic",
                            status_code=403,
                            retryable=True,
                        ) from e
                    raise KernelAccessDeniedError(
                        error_msg,
                        provider="anthropic",
                        status_code=403,
                    ) from e
                if status == 404:
                    raise KernelNotFoundError(
                        error_msg,
                        provider="anthropic",
                        status_code=404,
                    ) from e
                if status >= 500:
                    raise KernelProviderUnavailableError(
                        error_msg,
                        provider="anthropic",
                        status_code=status,
                        retryable=True,
                    ) from e
                raise KernelLLMError(
                    error_msg,
                    provider="anthropic",
                    status_code=status,
                    retryable=False,
                ) from e
            except KernelLLMError:
                raise  # Already translated, don't double-wrap
            except Exception as e:
                # Connection errors, timeouts, and anything unforeseen land
                # here -- same catch-all _do_complete() uses, treated as
                # transient and retryable.
                body = getattr(e, "body", None)
                error_msg = (
                    json.dumps(body)
                    if body is not None
                    else (str(e) or f"{type(e).__name__}: (no message)")
                )
                raise KernelLLMError(
                    error_msg,
                    provider="anthropic",
                    retryable=True,
                ) from e

        async def _on_retry(attempt: int, delay: float, error: KernelLLMError):
            """Callback invoked before each retry sleep."""
            error_type = type(error).__name__
            logger.warning(
                "[PROVIDER] Retry %d/%d for list_models(): %s, sleeping %.1fs",
                attempt,
                active_retry_config.max_retries,
                error_type,
                delay,
            )
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    PROVIDER_RETRY,
                    {
                        "provider": "anthropic",
                        "attempt": attempt,
                        "max_retries": active_retry_config.max_retries,
                        "delay": delay,
                        "error_type": error_type,
                        "error_message": str(error),
                    },
                )

        response = await retry_with_backoff(
            _do_list_models,
            self._retry_config,
            on_retry=_on_retry,
        )
        api_models = list(response.data)

        # Group models by family using _detect_family() as the single source of
        # truth for family classification. A previous version hardcoded a
        # separate {opus, haiku, sonnet} map here, which silently dropped any
        # family not in that list (e.g. fable) until it was updated by hand.
        families: dict[str, list[tuple[str, str, str]]] = {}
        for model in api_models:
            model_id = model.id
            display_name = getattr(model, "display_name", model_id)
            family = self._detect_family(model_id)
            families.setdefault(family, []).append(
                (model_id, display_name, str(getattr(model, "created_at", "")))
            )

        result: list[ModelInfo] = []

        for family, models in families.items():
            if not models:
                continue

            # Sort by model_id descending (IDs contain dates like claude-sonnet-4-5-20250929)
            models.sort(key=lambda x: x[0], reverse=True)

            # Free side-channel population of the fallback ladder's live
            # "newest model per family" cache (B-03/_resolve_fallback_model
            # source 2). Correct regardless of self.filtered -- models[0] is
            # the latest either way; only models_to_include below branches
            # on filtered.
            self._family_latest[family] = models[0][0]

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
        for family in ("fable", "mythos", "opus", "sonnet", "haiku"):
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
        * **Fable 5 / Fable 5.1** — always-on adaptive thinking, 128K output, no manual thinking
        * **Opus 4.6+** (incl. Opus 5 — confirmed via numeric version-gate, verified
          2026-07-24) — 1M context, adaptive thinking, 128K output
        * **Sonnet 4.5+** — 1M context, extended thinking, 64K output
        * **Haiku 4.5+** — fast inference, extended thinking, no adaptive, no 1M

        Computer-use wire type (``computer_use_tool_type``) — live-probed against
        api.anthropic.com 2026-08-03, see per-family comments below for the full
        evidence table:

        * **Opus 4.1-4.5** / **Sonnet/Haiku 4.5** — ``computer_20250124``
        * **Opus/Sonnet 4.6+** (incl. Opus/Sonnet 5) — ``computer_20251124``
        * Everything else (below the verified floor, or an unreachable/unverified
          model such as Fable) — unsupported (``None``)

        These two generations are NOT interchangeable: pairing a model with the
        wrong one returns HTTP 400, not a graceful fallback.

        When the version cannot be parsed from the model ID we assume the
        *latest* capabilities for that family so newly released models work
        out-of-the-box.
        """
        family = cls._detect_family(model_id)
        major, minor = cls._detect_version(model_id, family)
        version_known = (major, minor) != (0, 0)

        if family == "fable":
            return ModelCapabilities(
                family=family,
                max_output_tokens=128000,
                supports_1m=True,
                supports_thinking=True,
                supports_adaptive_thinking=True,
                supports_manual_thinking=False,
                thinking_always_on=True,
                supports_output_config=True,
                supports_task_budget=True,
                supports_sampling=False,
                thinking_display_required=True,
                supported_efforts=("low", "medium", "high", "xhigh", "max"),
                supports_speed=False,
                supports_inline_system=True,
                default_thinking_budget=0,  # not used — adaptive only
                # NOT verified: claude-fable-5 rejects every request in this
                # workspace ("organization or workspace must have data retention
                # enabled", HTTP 400, 2026-08-03) — the tool-support question
                # itself could not be asked live. Left at the conservative
                # dataclass default (False/None) rather than assumed.
                min_cacheable_tokens=512,
                capability_tags=(
                    "tools",
                    "thinking",
                    "streaming",
                    "json_mode",
                    "vision",
                ),
            )

        if family == "mythos":
            # Cloned from the fable branch above -- Mythos 5 is Anthropic's
            # other top-tier, always-on-adaptive-thinking family, priced
            # identically to Fable 5 ($10/MTok base, 2026-08-29). Mythos
            # PREVIEW differs from Mythos 5: no `xhigh` effort tier, and a
            # 2048-token (not 512) prompt-cache minimum -- version-gated on
            # (major, minor) exactly as the opus branch does; an unparseable
            # version is treated as Mythos 5 (the newer, more-capable member),
            # matching this provider's existing "unknown version assumes
            # latest" convention.
            major, minor = cls._detect_version(model_id, family)
            is_preview = (major, minor) == (0, 0) and "preview" in model_id.lower()
            return ModelCapabilities(
                family=family,
                max_output_tokens=128000,
                supports_1m=True,
                supports_thinking=True,
                supports_adaptive_thinking=True,
                supports_manual_thinking=False,
                thinking_always_on=True,
                supports_output_config=True,
                supports_task_budget=True,
                supports_sampling=False,
                thinking_display_required=True,
                supported_efforts=(
                    ("low", "medium", "high", "max")
                    if is_preview
                    else ("low", "medium", "high", "xhigh", "max")
                ),
                supports_speed=False,
                supports_inline_system=True,
                default_thinking_budget=0,  # not used — adaptive only
                min_cacheable_tokens=2048 if is_preview else 512,
                capability_tags=(
                    "tools",
                    "thinking",
                    "streaming",
                    "json_mode",
                    "vision",
                ),
            )

        if family == "opus":
            is_45_plus = not version_known or (major, minor) >= (4, 5)
            is_46_plus = not version_known or (major, minor) >= (4, 6)
            is_47_plus = not version_known or (major, minor) >= (4, 7)
            is_48_plus = not version_known or (major, minor) >= (4, 8)
            is_5_plus = not version_known or (major, minor) >= (5, 0)
            # Computer-use wire type, live-probed against api.anthropic.com
            # 2026-08-03 (bare {"type": ..., "name": "computer", "display_width_px":
            # 1024, "display_height_px": 768} declarations, matching anthropic-beta
            # header per NATIVE_TOOL_BETA_HEADERS):
            #   claude-opus-4-1-20250805 + computer_20250124 -> 200
            #   claude-opus-4-1-20250805 + computer_20241022/computer_20251124 -> 400
            #   claude-opus-4-5-20251101 + computer_20250124 -> 200 (also 20251124 -> 200;
            #     4.5 shipped the same day as the 20251124 generation and accepts both —
            #     20250124 is returned as the canonical answer since it is the one that
            #     works across the whole 4.1-4.5 range, not just this one model)
            #   claude-opus-4-6/4-7/4-8, claude-opus-5 + computer_20251124 -> 200
            #     (computer_20250124 -> 400 on all of these — the newer generation
            #     supersedes rather than extends the older one)
            # Below 4.1 is unverified: the only pre-4.1 opus model (claude-opus-4-20250514)
            # is retired (HTTP 404) in this workspace, so it could not be probed either way.
            if is_46_plus:
                computer_use_tool_type = "computer_20251124"
            elif version_known and (major, minor) >= (4, 1):
                computer_use_tool_type = "computer_20250124"
            else:
                computer_use_tool_type = None
            return ModelCapabilities(
                family="opus",
                max_output_tokens=128000 if is_46_plus else 64000,
                supports_1m=is_46_plus,
                supports_thinking=True,
                supports_adaptive_thinking=is_46_plus,
                supports_manual_thinking=not is_47_plus,
                # Deprecated (still HTTP 200, verified live 2026-08-29,
                # T-C08-live) on the one version where supports_manual_thinking
                # is still True but the API considers "enabled" legacy: 4.6.
                # 4.7+ already hard-gates to adaptive, so the two flags never
                # both apply on the same model.
                manual_thinking_deprecated=is_46_plus and not is_47_plus,
                # Widened from is_47_plus to is_45_plus (X-1/Q-2): Anthropic's
                # docs state the compatibility set directly --
                # "Supported models: ... Opus 4.5, 4.6, 4.7, 4.8, and 5 ..."
                # (platform.claude.com/en/docs/build-with-claude/effort,
                # verified 2026-08-29). Confirmed live (T-C06-live, merge
                # gate): output_config.effort="max" -> HTTP 200 on
                # claude-opus-4-6.
                supports_output_config=is_45_plus,
                supports_task_budget=is_47_plus,
                supports_sampling=not is_47_plus,
                thinking_display_required=is_47_plus,
                # xhigh: Fable 5/Mythos 5/Opus 5/4.8/4.7/Sonnet 5 (doc:
                # "Available on Claude Fable 5, Claude Mythos 5, Claude Opus
                # 5, Claude Opus 4.8, Claude Opus 4.7, and Claude Sonnet 5").
                # max: the above PLUS Opus 4.6 and Sonnet 4.6 (doc's own
                # explanation: "xhigh is a newer level; some models that
                # support max don't support xhigh"). Both verified against
                # platform.claude.com/en/docs/build-with-claude/effort,
                # 2026-08-29.
                supported_efforts=(
                    ("low", "medium", "high", "xhigh", "max")
                    if is_47_plus
                    else ("low", "medium", "high", "max")
                    if is_46_plus
                    else ("low", "medium", "high")
                ),
                supports_speed=is_48_plus,
                supports_inline_system=is_48_plus,
                default_thinking_budget=64000 if is_46_plus else 32000,
                supports_native_computer_use=computer_use_tool_type is not None,
                computer_use_tool_type=computer_use_tool_type,
                # Non-monotonic by design -- 4.6->4096, 4.7->2048, 4.8->1024,
                # 5->512 -- verified against
                # platform.claude.com/en/docs/build-with-claude/prompt-caching,
                # 2026-08-29. Written as an explicit descending version
                # chain, not a >= threshold, so a future "simplification"
                # doesn't flatten a real non-monotonic API constraint.
                min_cacheable_tokens=(
                    512
                    if is_5_plus
                    else 1024
                    if is_48_plus
                    else 2048
                    if is_47_plus
                    else 4096
                ),
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
            # TODO(verify-live): haiku has no 4.6+ branch. opus and sonnet both flip to
            # computer_20251124 at 4.6 - if that threshold is an org-wide API rollout
            # rather than per-family, the next haiku release gets the WRONG (older)
            # type here. It fails loud (the two generations mutually reject), but this
            # is the one spot that ASSERTS rather than returning None like every other
            # unverified case. Re-probe when a haiku 4.6+ ships.
            is_45_plus = is_46_plus or (major, minor) >= (4, 5)
            # Sonnet 5 (Jun 2026) gains the output_config effort API through the
            # "xhigh"/"max" tiers and the same thinking surface as Opus 4.7+:
            # adaptive thinking only (manual type="enabled" returns HTTP 400),
            # thinking block displayed by default, and task-budget support.
            # Verified live against claude-sonnet-5 (2026-07-01):
            # output_config.effort=xhigh -> 200; thinking.type=enabled -> 400.
            # Sonnet 5 also accepts the "max" effort tier (confirmed 2026-07-20);
            # it has no Opus-only fast mode.
            is_5_plus = not version_known or (major, minor) >= (5, 0)
            # Computer-use wire type, live-probed 2026-08-03 (same method as opus,
            # above):
            #   claude-sonnet-4-5-20250929 + computer_20250124 -> 200
            #   claude-sonnet-4-5-20250929 + computer_20241022/computer_20251124 -> 400
            #   claude-sonnet-4-6, claude-sonnet-5 + computer_20251124 -> 200
            #     (computer_20250124 -> 400 on both)
            # Below 4.5 is unverified for sonnet specifically: claude-sonnet-4-20250514
            # is retired (HTTP 404) in this workspace and no live sonnet model between
            # 4.1 and 4.4 exists to probe. Unlike opus (confirmed live down to 4.1), the
            # sonnet floor is set at the lowest version this evidence actually covers
            # (4.5) rather than extrapolated down to match opus's threshold.
            if is_46_plus:
                computer_use_tool_type = "computer_20251124"
            elif is_45_plus:
                computer_use_tool_type = "computer_20250124"
            else:
                computer_use_tool_type = None
            return ModelCapabilities(
                family="sonnet",
                # 1M-context models are entitled to 128K output (verified
                # platform.claude.com/en/docs/build-with-claude/context-windows,
                # 2026-08-29: "A single request to any model with a 1M-token
                # context window can generate up to 128k output tokens").
                # Sonnet 4.6+ has 1M context (supports_1m below) but was never
                # given the matching output ceiling and fell back to the
                # dataclass default of 64000 -- clamping claude-sonnet-5 to
                # half its real output capacity.
                max_output_tokens=128000 if is_46_plus else 64000,
                supports_1m=is_46_plus,
                supports_thinking=True,
                supports_adaptive_thinking=is_46_plus,
                supports_manual_thinking=not is_5_plus,
                # Deprecated (still HTTP 200) but not yet hard-gated on 4.6 --
                # mirrors the opus branch's manual_thinking_deprecated. 5+
                # already hard-gates to adaptive via supports_manual_thinking.
                manual_thinking_deprecated=is_46_plus and not is_5_plus,
                # Widened from is_5_plus to is_46_plus (X-1/Q-2): doc states
                # "Supported models: ... Sonnet 4.6 and 5"
                # (platform.claude.com/en/docs/build-with-claude/effort,
                # verified 2026-08-29). Confirmed live (T-C06-live, merge
                # gate): output_config.effort="max" -> HTTP 200 on
                # claude-sonnet-4-6.
                supports_output_config=is_46_plus,
                supports_task_budget=is_5_plus,
                thinking_display_required=is_5_plus,
                # Sonnet 5 rejects `temperature` ("deprecated for this model");
                # omit it, matching the Opus 4.7+ pattern. amplifier-support#299.
                supports_sampling=not is_5_plus,
                # xhigh: Sonnet 5 only (doc: "Available on ... Claude Sonnet
                # 5"). max: Sonnet 5 AND Sonnet 4.6 (doc: "some models that
                # support max don't support xhigh" -- Sonnet 4.6 is exactly
                # that case). Both verified against
                # platform.claude.com/en/docs/build-with-claude/effort,
                # 2026-08-29.
                supported_efforts=(
                    ("low", "medium", "high", "xhigh", "max")
                    if is_5_plus
                    else ("low", "medium", "high", "max")
                    if is_46_plus
                    else ("low", "medium", "high")
                ),
                default_thinking_budget=32000,
                supports_native_computer_use=computer_use_tool_type is not None,
                computer_use_tool_type=computer_use_tool_type,
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
            # Computer-use wire type, live-probed 2026-08-03 (same method as opus,
            # above): claude-haiku-4-5-20251001 + computer_20250124 -> 200;
            # computer_20241022/computer_20251124 -> 400. No haiku model at 4.6+
            # exists live to test whether haiku ever adopts computer_20251124 the
            # way opus/sonnet do at that threshold, so that jump is NOT assumed
            # here — is_45_plus (which also covers "unknown version") maps to the
            # one wire type actually confirmed for this family. Below 4.5
            # (e.g. the retired claude-haiku-3-5) is unverified — HTTP 404 in
            # this workspace — and left unsupported.
            computer_use_tool_type = "computer_20250124" if is_45_plus else None
            return ModelCapabilities(
                family="haiku",
                supports_thinking=is_45_plus,
                supports_adaptive_thinking=False,
                default_thinking_budget=32000 if is_45_plus else 0,
                supports_native_computer_use=computer_use_tool_type is not None,
                computer_use_tool_type=computer_use_tool_type,
                min_cacheable_tokens=4096,
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
            thinking_always_on=base_caps.thinking_always_on,
            default_thinking_budget=default_thinking_budget,
            # Not derived from the Models API — Anthropic's model metadata carries
            # no computer-use signal, so the family/version-gated static value is
            # always the source of truth here. Must still be forwarded explicitly
            # or a live-request override silently resets it to the dataclass
            # default (False/None), the same class of bug the sonnet-5 override
            # regression guard (test_sonnet_5_caps_survive_runtime_override) exists
            # to catch.
            supports_native_computer_use=base_caps.supports_native_computer_use,
            computer_use_tool_type=base_caps.computer_use_tool_type,
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

    def _derive_native_tool_betas(
        self, tools: list[dict[str, Any]] | None
    ) -> list[str]:
        """Return the beta headers required by native tool types in `tools`.

        A model-native tool declares itself on the wire as
        ``{"type": "<tool_type>", ...}``. Several are gated behind an
        anthropic-beta header, and omitting it makes the API reject the entire
        request. The tool type fully determines the header, so deriving it here
        means a caller can declare a native tool without also knowing which
        header it needs.

        Unknown types are ignored rather than guessed at: a type absent from
        `NATIVE_TOOL_BETA_HEADERS` either needs no header or is one this
        provider version has not seen, and inventing a header string would turn
        a working request into a rejected one.
        """
        if not tools:
            return []
        derived: list[str] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            beta = NATIVE_TOOL_BETA_HEADERS.get(tool.get("type") or "")
            if beta and beta not in derived:
                derived.append(beta)
        return derived

    def _build_request_beta_headers(
        self,
        *,
        request_caps: ModelCapabilities,
        tools_present: bool,
        resolved_thinking_type: str | None,
        has_task_budget: bool = False,
        fast_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Build the anthropic-beta header set for a specific effective model.

        `tools` is the wire-format tool list. When supplied, any beta headers
        required by model-native tool types present in it are derived here (see
        `NATIVE_TOOL_BETA_HEADERS`), so a caller declaring a beta-gated native
        tool does not also have to know - and inject - the matching header.

        No 1M-context beta header is ever added here: 1M context is GA,
        default, and standard-priced on every model that has it -- see the
        `enable_1m_context` comment in __init__ for the verified citation.
        """
        headers = list(self._beta_headers)
        headers.extend(self._derive_native_tool_betas(tools))
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

        1. The body did not parse as a JSON object/array. (When the SDK
           cannot parse the body as JSON it stores the RAW TEXT in
           ``error.body`` -- a str, NOT None; a parsed error is a dict/list.)
        2. The Content-Type is text/html (not application/json).
        3. The raw response text contains Cloudflare markers.

        Any combination of (1 + 2) or (1 + 3) is sufficient.  If the SDK
        successfully parsed a JSON body, this is a real API error regardless
        of other signals.
        """
        # Only a PARSED JSON body (dict/list) means a genuine, structured
        # API error. When the SDK cannot parse the body as JSON it stores the
        # RAW TEXT in ``error.body`` -- a str, NOT None -- so a "body is not
        # None" guard bails on exactly the HTML challenge pages this exists to
        # catch. Fall through for a str (or absent) body; bail only on parsed
        # JSON.
        body = getattr(error, "body", None)
        if isinstance(body, (dict, list)):
            return False

        # Inspect the raw HTTP response for HTML / Cloudflare signals
        response = getattr(error, "response", None)
        if response is None:
            return False

        # Case-fold before matching: neither the header value nor the page
        # text has a guaranteed casing. Cloudflare's own block pages render
        # "Cloudflare" capitalised in the footer while the marker below is
        # lowercase, so a case-sensitive scan misses real challenge pages --
        # and a missed challenge is a permanent AccessDeniedError raised for
        # a condition that would have cleared on retry.
        content_type = getattr(response, "headers", {}).get("content-type", "").lower()
        if "text/html" in content_type:
            return True

        # Fallback: scan response text for Cloudflare markers
        text = (getattr(response, "text", "") or "").lower()
        cf_markers = (
            "just a moment",
            "cf-browser-verification",
            "cloudflare",
            "checking if the site connection is secure",
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

    def _resolve_fallback_model(self, family: str) -> str | None:
        """Concrete model id for a target FAMILY. Pure/synchronous by design.

        Precedence:
          1. `fallback_models` config override  -- explicit operator intent
          2. live latest-per-family cache       -- refreshed from list_models()
          3. _STATIC_FALLBACK_MODELS backstop   -- always present, never stale-fails

        This must NOT do network I/O: it is called from the 529 error path
        (_open_fallback_window) and from the hot loop in complete(). A
        list_models() call there would add latency and its own 529 risk at
        exactly the moment the API is already saturated.
        """
        override = self._fallback_models.get(family)
        if override:
            return override
        live = self._family_latest.get(family)
        if live:
            return live
        return _STATIC_FALLBACK_MODELS.get(family)

    async def _warm_family_latest(self) -> None:
        """One best-effort list_models() to learn the newest model per family.

        Runs on the HEALTHY path (before the first completion), never on the
        error path -- so an overload event never pays for it. Single attempt
        (max_retries=0) under a short timeout: this is an optimisation over
        _STATIC_FALLBACK_MODELS, and a failure must cost nothing more than the
        timeout. Attempted at most once per provider instance.
        """
        if self._family_latest_attempted:
            return
        self._family_latest_attempted = True
        try:
            async with asyncio.timeout(10.0):
                await self.list_models(retry_config=self._build_retry_config(0))
        except Exception as e:  # noqa: BLE001 -- optimisation only
            logger.debug(
                "[PROVIDER] Could not refresh fallback model list (%s: %s); "
                "using static fallback targets.",
                type(e).__name__,
                e,
            )

    def _fallback_target_for_model(self, model_id: str) -> str | None:
        """Return the next lower-tier model on the fallback ladder, or None.

        Walks _FALLBACK_NEXT_FAMILY downward from the model's own family,
        SKIPPING any rung that resolves to nothing / to the model itself /
        to the same family, rather than terminating there (the pre-overhaul
        behaviour, which turned one blank config value into a dead ladder --
        this is exactly what silently killed fallback for the `fable`
        family, which had no rung at all in the old if/elif).

        Returns None only when the ladder is genuinely exhausted -- i.e. the
        model is already on the terminal rung (haiku), or every rung below it
        resolved to something unusable.
        """
        source_family = self._detect_family(model_id)
        family = source_family
        seen: set[str] = {family}

        while (family := _FALLBACK_NEXT_FAMILY.get(family)) is not None:
            if family in seen:  # cycle guard against a pathological override map
                logger.warning(
                    "[PROVIDER] Fallback ladder cycle at family %r while "
                    "resolving %s -- stopping.",
                    family,
                    model_id,
                )
                return None
            seen.add(family)

            target = self._resolve_fallback_model(family)
            if not target or target == model_id:
                continue  # rung unusable -- try the one below it
            if self._detect_family(target) == source_family:
                logger.warning(
                    "[PROVIDER] Ignoring invalid overload fallback %s -> %s (same family)",
                    model_id,
                    target,
                )
                continue  # was: return None
            return target

        return None

    def _refusal_fallback_target(self, current_model: str) -> str | None:
        """Resolve the model to retry against after a refusal.

        Owner-adjudicated: uses the SAME ladder as overload fallback --
        refusal fallback is not a separate escalation path (the old
        default hardcoded every family to Opus, which on inspection was
        already silently dead for both `opus` instances: opus ->
        claude-opus-4-8 -> same family -> None). Refusals now downgrade one
        rung exactly like overload, via the same three-source target
        resolution (fallback_models override -> live list_models cache ->
        static backstop, skipping unusable rungs). Rationale: Anthropic's
        own guidance for a refusal is to retry with a less-restrictive
        model (i.e. downward, not up to the most expensive tier), and
        fallback should never land on a model MORE expensive than the one
        the user selected -- which the old hardcoded Opus-escalation
        target violated for sonnet/haiku users. `refusal_fallback_model`
        (the old explicit-override escape hatch) is RETIRED; see
        _INERT_CONFIG_KEY_MESSAGES.

        haiku is the ladder's terminal rung, so a haiku refusal has no
        fallback target -- the refusal surfaces normally, exactly as
        before for that instance.
        """
        if not self._refusal_fallback_enabled:
            return None
        return self._fallback_target_for_model(current_model)

    @staticmethod
    def _strip_thinking_blocks(request: ChatRequest) -> ChatRequest:
        """Return a deep copy of request with thinking/redacted_thinking blocks
        removed from assistant messages.

        The refusal-fallback model may not support (or may reject) thinking
        blocks generated by the model that just refused, so they are stripped
        before retrying. The original request is left untouched.
        """
        stripped = request.model_copy(deep=True)
        for message in stripped.messages:
            if message.role != "assistant" or not isinstance(message.content, list):
                continue
            message.content = [
                block
                for block in message.content
                if getattr(block, "type", None) not in ("thinking", "redacted_thinking")
            ]
        return stripped

    @staticmethod
    def _is_overload_fallback_error(error: KernelLLMError) -> bool:
        """True only for genuine CAPACITY errors, which a downgrade can relieve.

        529 (overloaded_error) is capacity pressure "across all users"
        (platform.claude.com/docs/en/api/errors, verified 2026-08-29) -- a
        lower-tier model is a different capacity pool, so the ladder helps.

        429 (rate_limit_error) is NOT capacity. Per the same page it means
        the ORGANIZATION hit a rate limit, a usage-tier spend cap, or a
        workspace spend limit -- and "in rare cases, if your organization
        has a sharp increase in usage, you might see 429 errors because of
        acceleration limits ... ramp up your traffic gradually". Every one
        of those is per-account: the lower-tier model draws on the SAME
        org quota, so downgrading cannot help and merely hides the real
        fix (back off / ramp gradually / raise the quota). The previous
        substring test for "overload" in a 429 body is removed for that
        reason -- a 429 now always falls through to the full retry budget
        on the SAME model (exponential backoff honoring retry-after),
        which is the documented correct response.

        Spend-cap caveat, worth noting here: a tier spend-cap 429 has no
        retry-after header and keeps failing until access resumes.
        Retrying that is futile but harmless and bounded by max_retries;
        detecting it specifically is out of scope.
        """
        status_code = getattr(error, "status_code", None)
        return isinstance(error, KernelProviderUnavailableError) and status_code == 529

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
            # os.replace(), NOT os.rename(): on POSIX rename() atomically
            # replaces an existing target, but on Windows it raises
            # FileExistsError when the target exists. Both call sites sit
            # inside `except Exception: pass`, so on Windows every write
            # after the first one failed SILENTLY and this file froze at
            # its initial contents for the life of the process.
            # os.replace() has rename()'s atomic-replace semantics on both.
            os.replace(tmp_path, path)
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

        Uses write-to-tmp + os.replace() so concurrent readers never see a partial
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
            # os.replace(), NOT os.rename(): on POSIX rename() atomically
            # replaces an existing target, but on Windows it raises
            # FileExistsError when the target exists. Both call sites sit
            # inside `except Exception: pass`, so on Windows every write
            # after the first one failed SILENTLY and this file froze at
            # its initial contents for the life of the process.
            # os.replace() has rename()'s atomic-replace semantics on both.
            os.replace(tmp_path, path)
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

    async def _apply_refusal_fallback(
        self,
        response: ChatResponse,
        request: ChatRequest,
        effective_model: str,
        **complete_kwargs,
    ) -> ChatResponse:
        """If response is a refusal, retry once against the refusal-fallback model.

        Single-shot: whatever the fallback model returns (refusal or not) is
        returned as-is -- no recursive retry. Thinking/redacted_thinking blocks
        are stripped from the request before the fallback call, since they were
        produced by the model that just refused and may not be valid input to
        a different model.
        """
        if getattr(response, "finish_reason", None) != "refusal":
            return response

        fallback_model = self._refusal_fallback_target(effective_model)
        if fallback_model is None:
            return response

        logger.warning(
            "[PROVIDER] %s refused; retrying once with fallback model %s",
            effective_model,
            fallback_model,
        )

        fallback_kwargs = dict(complete_kwargs)
        fallback_kwargs["model"] = fallback_model
        fallback_request = self._strip_thinking_blocks(request)
        return await self._complete_chat_request(fallback_request, **fallback_kwargs)

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
            response = await self._complete_chat_request(request, **kwargs)
            return await self._apply_refusal_fallback(
                response,
                request,
                str(kwargs.get("model", self.default_model)),
                **kwargs,
            )

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
                response = await self._complete_chat_request(
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
            else:
                return await self._apply_refusal_fallback(
                    response,
                    request,
                    effective_model,
                    retry_config=retry_config,
                    **current_kwargs,
                )

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

        # Add cache_control if enabled. The system prompt is the most stable
        # part of the request (rebuilt from on-disk bundle content, not from
        # per-turn conversation), so it optionally gets the extended 1h TTL --
        # see `cache_stable_region_ttl_1h` (opt-in, default off).
        if self.enable_prompt_caching:
            cache_control: dict[str, Any] = {"type": "ephemeral"}
            if self.cache_stable_region_ttl_1h:
                cache_control["ttl"] = "1h"
            block["cache_control"] = cache_control

        return [block]

    def _merge_extra_request_params(self, params: dict[str, Any]) -> None:
        """Merge config ``extra_request_params`` into *params*, user-wins-loudly.

        Known typed params go onto the typed surface; everything else goes
        into extra_body, where the API will see it on the wire without the
        SDK ever inspecting it. Whoever sets this owns the consequences.

        Must run BEFORE ``_route_wire_only_params`` (see the call site in
        ``_complete_chat_request``): a user-supplied ``temperature`` or
        ``speed`` would otherwise land back on the typed SDK surface,
        reproducing the exact "got an unexpected keyword argument" retry
        bug ``_route_wire_only_params`` exists to prevent.
        """
        if not self.extra_request_params:
            return
        for key, value in self.extra_request_params.items():
            if key in _TYPED_REQUEST_PARAMS:
                if (
                    key in params
                    and params[key] != value
                    and key not in self._extra_params_warned_keys
                ):
                    self._extra_params_warned_keys.add(key)
                    logger.warning(
                        "[PROVIDER] extra_request_params overrides provider-computed "
                        "'%s' (%r -> %r).",
                        key,
                        params[key],
                        value,
                    )
                params[key] = value
            else:
                extra_body = dict(params.get("extra_body") or {})
                extra_body[key] = value  # user-wins: overwrite, NOT setdefault
                params["extra_body"] = extra_body

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

        # Determine ephemeral status BEFORE conversion -- Message.metadata is
        # only available on the original Message objects; _convert_messages
        # discards unrecognized keys when it rebuilds Anthropic-format dicts.
        unstable_suffix_len, has_ephemeral_signal = self._unstable_suffix_length(
            conversation
        )

        # Track how many of the 4 Anthropic cache breakpoints are used, so we
        # never exceed the API's hard limit (a 5th cache_control is a request
        # error, not a soft failure).
        breakpoints_used = 0

        # Format system messages as content block array (required for caching)
        system_blocks = self._format_system_with_cache(system_msgs)
        if system_blocks and self.enable_prompt_caching:
            breakpoints_used += 1

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

        # Apply up to 2 rolling cache breakpoints over STABLE conversation
        # content (never on the trailing ephemeral messages identified
        # above). Budget is whatever remains of the 4-breakpoint API limit
        # after system (0 or 1) and tools (0 or 1, applied below) -- tools
        # hasn't run yet at this point, so reserve its slot conservatively.
        _tools_will_use_a_slot = bool(request.tools and self.enable_prompt_caching)
        conversation_budget = (
            4 - breakpoints_used - (1 if _tools_will_use_a_slot else 0)
        )

        # Before placing anything, measure the unstable tail rather than
        # relying solely on the orchestrator declaring it. Fingerprints must
        # be taken here -- after `all_messages` is assembled, BEFORE any
        # cache_control is stamped onto it.
        observed_state = "disabled"
        if self.enable_prompt_caching and self.cache_infer_stability_from_history:
            observed_len, observed_state = self._observed_unstable_suffix_length(
                all_messages, system_blocks
            )
            if observed_len is not None:
                if observed_len > unstable_suffix_len:
                    logger.debug(
                        "[PROVIDER] Prompt caching: observed unstable suffix of "
                        "%d message(s) (declared via metadata: %d) -- using the "
                        "larger, more conservative value.",
                        observed_len,
                        unstable_suffix_len,
                    )
                unstable_suffix_len = max(unstable_suffix_len, observed_len)
                # An observation IS a signal: it answers the same question
                # metadata would have, from evidence this provider gathered
                # itself. Without this, a deployment that never populates
                # Message.metadata stays permanently in the skip path.
                has_ephemeral_signal = True

        if observed_state == "no_shared_prefix":
            # Placing a breakpoint would burn a 1.25x cache WRITE every turn
            # for a prefix that can never be read back.
            logger.warning(
                "[PROVIDER] Prompt caching: this request shares no leading "
                "message with the previous request in the same conversation, "
                "so no cached prefix can ever match (Anthropic matches from "
                "the start of the request). Skipping conversation-region "
                "cache breakpoints. Most likely cause: a leading context/"
                "developer message regenerated per request with volatile "
                "content (timestamp, git status, session id)."
            )
            conversation_breakpoints_used = 0
        else:
            all_messages, conversation_breakpoints_used = (
                self._apply_conversation_cache_control(
                    all_messages,
                    unstable_suffix_len,
                    has_ephemeral_signal,
                    conversation_budget,
                )
            )
        breakpoints_used += conversation_breakpoints_used
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
            tools, tool_breakpoint_used = self._apply_tool_cache_control(tools)
            params["tools"] = tools
            if tool_breakpoint_used:
                breakpoints_used += 1
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

        # An EXPLICITLY requested thinking budget — kwargs first, then config.
        # Captured here, before any resolution, for two reasons:
        #   1. it is what the budget chain below now resolves from, so an
        #      explicit config value outranks the effort→budget ladder
        #      (the ladder is a derived default; config is caller intent); and
        #   2. the silent-discard guard after the thinking block compares it
        #      against what actually reached the wire.
        # Before this, config `thinking_budget_tokens` sat BELOW the effort
        # ladder in the chain, and the ladder always produced a value whenever
        # any reasoning_effort was set — so the config key was accepted without
        # complaint and then discarded, and the only budgets reachable from
        # config were {4096 (effort: low), <model default>}.
        requested_budget_source: str | None = None
        requested_budget_raw: Any = None
        if kwargs.get("thinking_budget_tokens") is not None:
            requested_budget_source = "kwargs"
            requested_budget_raw = kwargs["thinking_budget_tokens"]
        elif self.config.get("thinking_budget_tokens") is not None:
            requested_budget_source = "config"
            requested_budget_raw = self.config["thinking_budget_tokens"]

        requested_budget: int | None = None
        if requested_budget_source is not None:
            try:
                requested_budget = int(requested_budget_raw)
            except (TypeError, ValueError):
                # Fail soft, log loudly — the same policy the numeric config
                # helpers use at construction. A typo must not kill every
                # request with a ValueError from int().
                logger.warning(
                    "[PROVIDER] Ignoring invalid %s 'thinking_budget_tokens'=%r "
                    "(expected an integer) — falling back to the resolved default.",
                    requested_budget_source,
                    requested_budget_raw,
                )
                requested_budget_source = None
                requested_budget = None

        # Enable extended thinking if requested (equivalent to OpenAI's reasoning)
        #
        # Precedence chain (highest to lowest):
        #   1. kwargs["extended_thinking"]   — explicit per-request override
        #   2. config["extended_thinking"]   — explicit session-level override
        #   3. request.reasoning_effort      — portable kernel interface (Phase 2)
        #   4. config["reasoning_effort"]    — session-level effort default
        #
        # kwargs["extended_thinking"]=False can disable thinking even when
        # reasoning_effort is set (explicit opt-out); config["extended_thinking"]
        # is the same opt-in/opt-out one level down. config["extended_thinking"]
        # exists so a config-only caller can turn thinking on WITHOUT also
        # choosing an effort: before it, `thinking_budget_tokens` could never be
        # read at all on that path (thinking was never enabled), which is the
        # fifth silently-inert configuration this key had.
        thinking_enabled = bool(kwargs.get("extended_thinking"))
        config_extended_thinking: bool | None = None
        if "extended_thinking" in self.config:
            config_extended_thinking = self._config_bool(
                self.config["extended_thinking"]
            )

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
        #
        # IMPORTANT — this is NOT a complete chain: output_config.effort is a
        # *second*, independently-gated field (see the output_config block
        # below), not merely a side effect of resolving reasoning_effort here.
        # On models with supports_output_config, output_config.effort IS the
        # thinking control surface, so kwargs["extended_thinking"]=False (an
        # explicit "no reasoning on this call" opt-out) is honored there too:
        # an ambient/ resolved reasoning_effort is NOT applied to
        # output_config when the caller explicitly opted out of thinking,
        # unless the caller ALSO passed an explicit kwargs["effort"]
        # override (a deliberate output_config-only request that wins
        # regardless of the opt-out). See the output_config block for the
        # exact condition.
        if reasoning_effort is None:
            # Canonical config key first ("reasoning_effort", matching the
            # kernel's portable request.reasoning_effort), then the legacy
            # "effort" alias. When both are set the canonical key wins (a
            # one-time warning is emitted in __init__).
            config_key = "reasoning_effort"
            config_effort = self.config.get("reasoning_effort")
            if config_effort is None:
                config_key = "effort"
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
                        "[PROVIDER] Ignoring invalid config '%s'=%r (valid values: %s)",
                        config_key,
                        config_effort,
                        ", ".join(valid_efforts),
                    )

        if "extended_thinking" not in kwargs:
            if config_extended_thinking is not None:
                # An explicit config opt-in/opt-out outranks the effort
                # implication below, mirroring how kwargs["extended_thinking"]
                # outranks it. Unset (the default) leaves the implication
                # untouched — see the elif.
                thinking_enabled = config_extended_thinking
            elif reasoning_effort is not None:
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

        # reasoning_effort is now fully resolved, and thinking_enabled is now
        # final (kwargs["extended_thinking"]=False forces it off above;
        # request_caps.supports_thinking=False forces it off just above too).
        # Warn here — covering BOTH the request path and the config path
        # (PR #84's intent) — when the caller asked for more than "high" on a
        # model that has no output_config support. This must run after
        # thinking_enabled is final: when thinking is disabled (explicit
        # extended_thinking=False, or a model with no thinking support at
        # all, e.g. Haiku), reasoning_effort is not applied anywhere —
        # output_config.effort is skipped by the inverse of this same
        # capability check below, and the effort→thinking ladder is skipped
        # entirely. In that case "resolves identically to 'high'" would be
        # false; nothing is applied at all, so warning would be misleading.
        # Only warn when thinking_enabled is True, i.e. the effort ladder
        # below actually runs and collapses "xhigh"/"max" to "high".
        if (
            reasoning_effort in ("xhigh", "max")
            and not request_caps.supports_output_config
            and thinking_enabled
        ):
            logger.warning(
                "[PROVIDER] reasoning_effort=%r has no effect on %s (no output_config "
                "support) — resolves identically to 'high'. Supported efforts for this "
                "model: %s",
                reasoning_effort,
                effective_model,
                request_caps.supported_efforts,
            )

        if thinking_enabled:
            # Fable 5: thinking is always on. Never inject a thinking
            # param — the API handles it implicitly. Sending {type:disabled} causes
            # an HTTP 400. Set resolved_thinking_type for downstream use (beta headers).
            if request_caps.thinking_always_on:
                resolved_thinking_type = "adaptive"
            else:
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

                # Resolve budget: explicit (kwargs > config) > reasoning_effort
                #                 > model default
                #
                # `requested_budget` is the caller's EXPLICIT ask, resolved
                # above. It now outranks the effort→budget ladder, which was
                # previously between kwargs and config and therefore shadowed
                # config on every request that set any reasoning_effort. The
                # ladder is a derived default (for every effort except "low" it
                # simply restates request_caps.default_thinking_budget); an
                # explicitly configured number is caller intent, and explicit
                # beats derived.
                #
                # Byte-identical on the default path: with no explicit budget
                # anywhere, `requested_budget` is None and this collapses to
                # `effort_budget or request_caps.default_thinking_budget`,
                # exactly as before.
                budget_tokens = (
                    requested_budget
                    or effort_budget
                    or request_caps.default_thinking_budget
                )
                budget_tokens = max(1024, int(budget_tokens))
                max_budget_tokens = (
                    model_ceiling
                    if params.get("tools")
                    else max(1024, model_ceiling - 1)
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
                if (
                    thinking_type == "adaptive"
                    and request_caps.supports_adaptive_thinking
                ):
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

        # ------------------------------------------------------------------
        # Silent-discard guard for `thinking_budget_tokens`
        # ------------------------------------------------------------------
        # An explicitly requested budget that does not reach
        # thinking.budget_tokens must SAY SO. The defect this closes is not the
        # precedence order — it is the silence: the key was accepted without
        # complaint and then dropped, exactly the way a discarded `effort` was
        # before it got a loader guard. Runs once, after thinking is fully
        # resolved, so it covers every way the value can fail to land:
        # thinking off, model can't think, adaptive mode (where the API forbids
        # budget_tokens outright), and clamping to the model's limits.
        #
        # Fires ONLY when the caller explicitly asked for a budget, so the
        # default path stays byte-identical and silent.
        if requested_budget_source is not None and requested_budget is not None:
            wire_thinking = params.get("thinking")
            sent_budget = (
                wire_thinking.get("budget_tokens")
                if isinstance(wire_thinking, dict)
                else None
            )
            if sent_budget != requested_budget:
                if not thinking_enabled and not request_caps.supports_thinking:
                    reason = (
                        f"model {params['model']} does not support extended "
                        f"thinking, so no thinking budget is sent at all"
                    )
                elif not thinking_enabled:
                    reason = (
                        "extended thinking is not enabled, so the budget is "
                        "never read — set `reasoning_effort` (low|medium|high|"
                        "xhigh|max) or `extended_thinking: true` to turn it on"
                    )
                elif request_caps.thinking_always_on:
                    reason = (
                        f"{params['model']} always thinks and manages its own "
                        "budget — the API rejects a thinking param on this "
                        "model, so no budget can be sent and this value is "
                        "not used for anything"
                    )
                elif resolved_thinking_type == "adaptive":
                    reason = (
                        "thinking.type='adaptive' — the API forbids "
                        "budget_tokens in adaptive mode (the model manages its "
                        "own budget); the value only feeds max_tokens sizing "
                        f"(resolved max_tokens={params.get('max_tokens')}). "
                        "Set `thinking_type: enabled` to send an explicit budget"
                    )
                else:
                    reason = (
                        "clamped to this model's limits (minimum 1024, maximum "
                        f"{model_ceiling if params.get('tools') else max(1024, model_ceiling - 1)})"
                    )
                logger.warning(
                    "[PROVIDER] %s 'thinking_budget_tokens'=%s did not reach the "
                    "wire: sent %s. Reason: %s.",
                    requested_budget_source,
                    requested_budget,
                    sent_budget if sent_budget is not None else "no thinking budget",
                    reason,
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
        #
        # kwargs["extended_thinking"]=False is an explicit, per-call "no
        # reasoning on this call" opt-out (see thinking_enabled above). On
        # supports_output_config models, output_config.effort IS the
        # thinking control surface — so silently applying an ambient/
        # resolved reasoning_effort here would reintroduce reasoning the
        # caller explicitly turned off, defeating the opt-out. An explicit
        # kwargs["effort"] still wins even when the caller opted out of
        # thinking: it's a deliberate, per-call output_config-only override
        # (not an ambient default), matching the existing precedence note
        # below.
        # config["extended_thinking"]=false is the same explicit opt-out one
        # level down (kwargs still wins). Unset config leaves this False, so
        # the default path is unchanged.
        explicit_thinking_opt_out = (
            kwargs["extended_thinking"] is False
            if "extended_thinking" in kwargs
            else config_extended_thinking is False
        )
        explicit_effort_override = "effort" in kwargs
        if (
            request_caps.supports_output_config
            and reasoning_effort is not None
            and not (explicit_thinking_opt_out and not explicit_effort_override)
        ):
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
            request_caps=request_caps,
            tools_present=bool(params.get("tools")),
            resolved_thinking_type=resolved_thinking_type,
            has_task_budget=has_task_budget,
            fast_mode=fast_mode_enabled,
            tools=params.get("tools"),
        )
        if request_beta_headers:
            extra_headers = dict(params.get("extra_headers", {}))
            extra_headers["anthropic-beta"] = ",".join(request_beta_headers)
            params["extra_headers"] = extra_headers

        # The documented escape hatch, merged LAST -- after every provider-computed
        # value, and BEFORE _route_wire_only_params so that a user-supplied
        # `temperature`/`speed` is relocated to extra_body by the same router that
        # handles the provider's own.
        self._merge_extra_request_params(params)

        # Move wire-only params off the typed SDK surface. Must be the last
        # mutation of `params` before the call -- everything above may still
        # add or overwrite the keys this relocates.
        _route_wire_only_params(params)

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
                            async with self.client.messages.stream(**params) as stream:
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
                                            if (
                                                btype == "tool_use"
                                                and block is not None
                                            ):
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
                                            text = getattr(delta, "thinking", "") or ""
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
                    # Use with_raw_response to access headers.
                    #
                    # `timeout=` is passed explicitly, not merged into `params`,
                    # so it reaches only this call -- the streaming branch above
                    # takes the client-level timeout and the SDK guard below
                    # does not apply to it.
                    #
                    # The SDK's Messages resource guards non-streaming calls:
                    #
                    #   if not stream and not is_given(timeout) and
                    #      self._client.timeout == DEFAULT_TIMEOUT:
                    #       timeout = self._client._calculate_nonstreaming_timeout(...)
                    #
                    # ...which raises "Streaming is required for operations that
                    # may take longer than 10 minutes" whenever `max_tokens`
                    # exceeds 21,333 -- and `self.max_tokens` defaults to the
                    # model's full output ceiling (64k-128k), so the estimate
                    # refuses ordinary calls before any HTTP request is made.
                    #
                    # Setting the client-level timeout is NOT sufficient to skip
                    # it: with the default `self.timeout` of 600.0, the client
                    # timeout is value-equal to the SDK's own DEFAULT_TIMEOUT
                    # (read/write/pool 600, connect 5.0), so that comparison
                    # stays true. `is_given(timeout)` is checked first, so the
                    # per-request timeout is what reliably skips the estimate.
                    #
                    # This is not evading a safety check. The guard exists to
                    # bound a request it cannot time; we already bound this one
                    # with the same value, on the line directly below.
                    raw_response = await asyncio.wait_for(
                        self.client.messages.with_raw_response.create(
                            **params, timeout=self.timeout
                        ),
                        timeout=self.timeout,
                    )
                    response = await raw_response.parse()
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
                if _is_context_overflow(raw_msg):
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
                    # challenge pages the SDK can't parse as JSON, so e.body is
                    # the raw HTML str (never None) and content-type is text/html.
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
                # GAP-016: the Anthropic SDK's own APIConnectionError carries a
                # fixed, generic message ("Connection error.") regardless of
                # *why* the underlying httpx/httpcore call failed -- DNS
                # failure, TLS failure, a malformed base_url producing
                # `UnsupportedProtocol`, etc. all look identical to a user or
                # to logs, and are indistinguishable from real transient
                # network flakiness. The SDK chains the real exception via
                # `raise APIConnectionError(...) from err`, so it's available
                # on `__cause__` -- surface it instead of silently dropping it,
                # so "Connection error." becomes something a user can actually
                # act on (e.g. "caused by UnsupportedProtocol: Request URL is
                # missing an 'http://' or 'https://' protocol" directly names
                # a misconfigured base_url instead of looking like the network
                # is down).
                cause = e.__cause__
                if cause is not None:
                    # Redact before interpolating: this path is generic, so ANY
                    # exception with a __cause__ gets its str() spliced into a
                    # message that reaches logs and user-facing output. A
                    # base_url carrying embedded basic-auth
                    # (https://user:pass@proxy.internal/) shows up verbatim in
                    # httpx/httpcore's own exception text (e.g. the URL is
                    # quoted back in "Request URL is missing/invalid ..."), so
                    # redact_secrets() alone isn't enough -- it only redacts
                    # dict values under a sensitive key, not credentials
                    # embedded inside a plain string. Strip URL userinfo first,
                    # then apply the same redact_secrets() treatment the raw
                    # request/response payloads already get elsewhere in this
                    # file.
                    cause_text = redact_secrets(_redact_url_credentials(str(cause)))
                    # Compare on the redacted text -- comparing on the raw text
                    # would suppress the suffix whenever the unredacted string
                    # happened to appear, which is not the question being asked.
                    #
                    # `cause_text` is checked for truthiness explicitly: an
                    # exception with an empty str() ("" in anything is True)
                    # would otherwise silently drop the whole suffix, taking the
                    # useful *type name* with it -- the one piece of diagnostic
                    # value such a cause still has.
                    if not cause_text or cause_text not in error_msg:
                        error_msg = f"{error_msg} (caused by {type(cause).__name__}: {cause_text})"
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
                # Model-native tool - pass through, minus the function-tool-only
                # fields.
                #
                # A native tool's schema is fixed server-side by Anthropic, so it
                # accepts only its own keys (`type`, `name`, and per-tool config
                # like `display_width_px`). `parameters` and `description` are
                # function-tool concepts and the API rejects them outright:
                #
                #   invalid_request_error
                #   tools.0.computer_20251124.parameters: Extra inputs are not permitted
                #
                # These two keys are hard to avoid upstream: `ToolSpec.parameters`
                # is a required dict (it rejects `None`), so any caller building a
                # native ToolSpec is forced to carry a value that must not reach
                # the wire. Dropping them here - where we already know the tool is
                # native - is the one place that knowledge lives, rather than
                # requiring every caller to know which keys are illegal.
                if hasattr(tool, "model_dump"):
                    native = tool.model_dump(exclude_none=True)
                    for function_only_key in ("parameters", "description"):
                        native.pop(function_only_key, None)
                    anthropic_tools.append(native)
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
    ) -> tuple[list[dict[str, Any]], bool]:
        """Add cache_control to the last *function* tool definition.

        Per Anthropic spec: a cache breakpoint on the last tool creates a
        checkpoint for the entire tool list.

        Model-native tools (``web_search_20250305``, ``computer_20251124``,
        ...) are server-side definitions whose schema is fixed by Anthropic
        and does not accept ``cache_control``. Stamping one is rejected by
        the API, so the breakpoint goes on the last tool that can carry it -
        a function tool. Because the breakpoint still covers everything
        declared before it, and native tools are appended after the
        function tools, the caching benefit is unchanged.

        This mirrors what the native web-search path already does when it
        appends its own tool.

        Args:
            tools: List of Anthropic-formatted tool definitions

        Returns:
            Tuple of (same list, mutated in place with cache_control on the
            last function tool if caching is enabled and a function tool is
            present; whether a breakpoint was actually placed). The caller
            needs the boolean to keep an accurate running count against the
            4-breakpoint API hard limit.
        """
        if not tools or not self.enable_prompt_caching:
            return tools, False

        # The tool list is part of the same "stable region" as the system
        # prompt (rarely changes turn-to-turn within a session), so it shares
        # the opt-in extended TTL setting.
        cache_control: dict[str, Any] = {"type": "ephemeral"}
        if self.cache_stable_region_ttl_1h:
            cache_control["ttl"] = "1h"

        # Walk backwards to the last tool that can carry a cache breakpoint.
        # A tool is a function tool when it has no "type" (the classic shape)
        # or explicitly declares type == "function".
        for tool in reversed(tools):
            tool_type = tool.get("type")
            if tool_type is None or tool_type == "function":
                tool["cache_control"] = dict(cache_control)
                return tools, True
        # No function tool present (native tools only) - nothing can carry the
        # breakpoint. Return unstamped rather than sending a request the API
        # will reject.
        return tools, False

    @staticmethod
    def _is_tool_use_message(msg: dict[str, Any]) -> bool:
        """True if an assistant message's content contains a tool_use block."""
        content = msg.get("content")
        if not isinstance(content, list):
            return False
        return any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in content
        )

    @staticmethod
    def _is_tool_result_message(msg: dict[str, Any]) -> bool:
        """True if a message's content is (only) tool_result blocks.

        This is exactly the shape ``_convert_messages`` produces when it
        batches consecutive ``role: tool`` messages into a single Anthropic
        user message (see ``_convert_messages``).
        """
        content = msg.get("content")
        if not isinstance(content, list) or not content:
            return False
        return all(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        )

    @staticmethod
    def _stamps_empty_text_block(msg: dict[str, Any]) -> bool:
        """True if stamping ``msg`` would put cache_control on empty text.

        Mirrors ``_stamp_last_block``'s two branches, because that method is
        what decides which block actually receives the marker:

        * list content -> the LAST block. Unsafe only when that block is a
          text block whose text is empty or whitespace-only. A trailing
          ``tool_use`` / ``tool_result`` / image block is fine.
        * string content -> the whole string is turned INTO a text block, so
          an empty or whitespace-only string is unsafe.

        Anthropic rejects the request outright when a cache breakpoint lands
        on an empty text block:

            messages.N.content.0.text: cache_control cannot be set for
            empty text blocks

        Whitespace-only counts as empty on their side too, hence ``.strip()``.
        """
        content = msg.get("content")
        if isinstance(content, list):
            if not content:
                # Nothing to stamp; _stamp_last_block is a no-op here.
                return False
            last = content[-1]
            if not isinstance(last, dict) or last.get("type") != "text":
                return False
            return not (last.get("text") or "").strip()
        if isinstance(content, str):
            return not content.strip()
        return False

    @staticmethod
    def _stamps_uncacheable_block(msg: dict[str, Any]) -> bool:
        """True if stamping ``msg`` would put cache_control on a block type
        Anthropic forbids it on.

        Mirrors ``_stamp_last_block``'s list branch: the marker lands on the
        LAST content block. Anthropic permits ``cache_control`` on text,
        tool_use, tool_result, image and document blocks -- but NOT on
        ``thinking`` or ``redacted_thinking`` blocks. When the chosen stable
        message ends with a thinking block (an interleaved-thinking step, or a
        turn whose only stored payload is its thinking block -> content ==
        ``[thinking]``), a breakpoint there makes Anthropic reject the ENTIRE
        request with:

            messages.N.content.0.thinking.cache_control:
            Extra inputs are not permitted

        Only list content can offend: ``_stamp_last_block``'s string branch
        always synthesises a ``text`` block, which is cacheable.
        """
        content = msg.get("content")
        if isinstance(content, list) and content:
            last = content[-1]
            return isinstance(last, dict) and last.get("type") in (
                "thinking",
                "redacted_thinking",
            )
        return False

    def _last_safe_breakpoint_index(
        self, messages: list[dict[str, Any]], start_idx: int
    ) -> int | None:
        """Walk backward from ``start_idx`` to a message boundary that does
        not split a tool_use/tool_result pair.

        A cache_control marker never removes or reorders messages -- it only
        marks a prefix boundary -- so it cannot literally "break" a pair.
        But ending a cached checkpoint precisely between a tool_use and its
        own tool_result is a confusing, easy-to-misread boundary and is
        avoided defensively here: if ``messages[idx]`` is an assistant
        message containing a tool_use and the very next message is that
        tool_use's tool_result batch, the candidate is rejected and the
        search continues one message earlier.

        Returns the safe index, or ``None`` if no safe index exists at or
        before ``start_idx`` (e.g. everything from 0..start_idx is one
        unresolved tool round).
        """
        idx = start_idx
        while idx >= 0:
            msg = messages[idx]
            splits_pair = (
                idx + 1 < len(messages)
                and self._is_tool_use_message(msg)
                and self._is_tool_result_message(messages[idx + 1])
            )
            # A content-less turn (e.g. an OpenAI-format assistant message
            # replayed as ``content: null`` and defaulted to ``""`` upstream)
            # would be stamped as an EMPTY text block, which Anthropic rejects
            # with a 400. Walk past it exactly as we walk past a split pair.
            #
            # A turn ending in a ``thinking`` / ``redacted_thinking`` block is
            # rejected the same way (``...thinking.cache_control: Extra inputs
            # are not permitted``), because ``_stamp_last_block`` would mark
            # that trailing thinking block. Walk past it too, to a message
            # whose last block can legally carry a cache breakpoint.
            if (
                not splits_pair
                and not self._stamps_empty_text_block(msg)
                and not self._stamps_uncacheable_block(msg)
            ):
                return idx
            idx -= 1
        return None

    def _unstable_suffix_length(self, conversation: list[Message]) -> tuple[int, bool]:
        """How many trailing entries must be excluded from breakpoint eligibility.

        A message is UNSTABLE when it is marked ephemeral but not persisted --
        i.e. regenerated per request, so any cached prefix containing it can
        never be reproduced. Everything from the last unstable message to the
        end is excluded, whether or not the unstable message is itself last
        (the pre-user reminder block introduced by the system-reminder
        redesign is not: it sits BEFORE the real user message, not after it).

        Ephemeral AND persisted is STABLE: it was written into canonical
        history via context.add_message and is byte-frozen from then on
        (the persist-mode reminder block, and the orchestrator's own
        budget-warning message, both qualify).

        This is a strict generalization of the prior
        ``_count_trailing_ephemeral_messages`` (any ephemeral message,
        regardless of position, disqualified the tail): that method's
        behavior is reproduced exactly whenever ephemeral content only ever
        appears at the true tail (today's shape, and the redesign's own
        tail-injection-mode shape), and additionally handles a leading
        ephemeral-but-unstable block correctly, which the old walk could not
        express at all.

        The role == "tool" hard stop is retained verbatim from
        _count_trailing_ephemeral_messages: _convert_messages batches
        consecutive tool messages many-to-one, which breaks the 1:1 index
        correspondence a count-from-the-end relies on. Stopping early is safe
        here because view-only (unstable) injections only ever exist for the
        CURRENT request -- they are never in canonical history, so none can be
        buried behind an earlier tool batch.

        Returns:
            (excluded, has_any_metadata_signal). ``excluded`` is a count from
            the end (never an absolute index -- ``all_messages`` combines
            context-prefix messages with converted conversation messages, so
            pre-conversion indices do not map 1:1 onto it).
            ``has_any_metadata_signal`` tells the caller whether *any*
            message in the conversation carries a metadata dict at all --
            i.e. whether "not unstable" is a trustworthy negative, or simply
            means nothing in this deployment ever populates the field.
        """
        has_any_metadata_signal = any(bool(m.metadata) for m in conversation)

        excluded = 0
        for walked, msg in enumerate(reversed(conversation), start=1):
            if msg.role == "tool":
                break
            md = msg.metadata or {}
            if md.get("ephemeral") and not md.get("persisted"):
                excluded = walked
        return excluded, has_any_metadata_signal

    @staticmethod
    def _message_fingerprint(msg: dict[str, Any]) -> str:
        """Content digest of one Anthropic-format message.

        ``cache_control`` keys are stripped at every depth so a fingerprint
        is invariant under breakpoint placement: the *same* message must
        hash identically whether or not this request happened to stamp it.
        Without that, every fingerprint comparison would report a spurious
        difference exactly at the breakpoint, which is the one position the
        comparison exists to reason about.
        """

        def _strip(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: _strip(v) for k, v in obj.items() if k != "cache_control"
                }
            if isinstance(obj, list):
                return [_strip(v) for v in obj]
            return obj

        canonical = json.dumps(_strip(msg), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _conversation_key(
        system_blocks: list[dict[str, Any]] | None, fingerprints: list[str]
    ) -> str:
        """Identity of a conversation, for fingerprint bookkeeping only.

        Derived from the system prompt plus the FIRST message, both of which
        are fixed for the life of a conversation while everything after them
        grows. No session id is available at this layer, and inventing a
        dependency on one would tie caching to an orchestrator contract --
        the exact coupling that broke conversation caching in the first
        place.

        A deployment whose first message is itself regenerated per request
        gets a new key every turn and therefore never accumulates an
        observation. That is not a silent loss: a volatile first message
        means no cached prefix can EVER match, since Anthropic matches from
        the start of the request. Such a deployment has nothing to lose here
        and is reported by the `no_shared_prefix` diagnostic when the key
        does hold still long enough to compare.
        """
        system_text = ""
        if system_blocks:
            system_text = str(system_blocks[0].get("text", ""))
        head = fingerprints[0] if fingerprints else ""
        digest = hashlib.sha256(
            f"{hashlib.sha256(system_text.encode('utf-8')).hexdigest()}:{head}".encode()
        )
        return digest.hexdigest()[:32]

    def _observed_unstable_suffix_length(
        self,
        all_messages: list[dict[str, Any]],
        system_blocks: list[dict[str, Any]] | None,
    ) -> tuple[int | None, str]:
        """Measure the unstable tail by comparing this request to the last one.

        The metadata contract (`_unstable_suffix_length`) asks the
        orchestrator to DECLARE which trailing messages are regenerated per
        request. When nothing populates `Message.metadata`, that question has
        no answer and conversation caching is skipped entirely -- measured at
        a 6.8% hit ratio over 164 calls on a real coding-agent node visit.

        This answers the same question by observation instead. Fingerprint
        every message; on the next request in the same conversation, walk the
        two fingerprint lists to their longest common prefix. Whatever the
        previous request had BEYOND that point did not survive into this one,
        so ``len(previous) - lcp`` is a measured upper bound on how many
        trailing messages are regenerated rather than appended.

        Why a count from the end transfers correctly to the *current*
        (longer) request: the estimator describes the orchestrator's
        behavior, not a fixed index. If it drops/rewrites k trailing entries
        each turn, then excluding k entries from this request's tail lands
        the breakpoint on content that the next request will still carry,
        byte-identical -- which is precisely the condition for a cache hit.
        Worked shapes:

        * append-only  -> prev [a,b,c], now [a,b,c,d,e]: lcp 3, observed 0,
          breakpoint on the true tail. Optimal, and correct.
        * tail reminder -> prev [a,b,c,u,R], now [a,b,c,u,A,T,R']: lcp 4,
          observed 1, breakpoint at T -- present in the next request.
        * pre-user reminder -> prev [a,b,c,R,u], now [a,b,c,u,A,T,R',u']:
          lcp 3, observed 2, breakpoint at T. Over-estimates by one (u moved
          rather than vanished); over-estimating only ever costs cacheable
          tokens, never a hit.

        Returns:
            ``(observed_length, state)``. State is one of:

            * ``"none"``   -- no prior request for this conversation, so no
              observation exists yet (length is None). Caller falls back to
              whatever the metadata contract said.
            * ``"ok"``     -- length is a measured unstable-suffix count.
            * ``"no_shared_prefix"`` -- the previous and current requests
              share NO leading message. Nothing in this conversation can ever
              produce a cache hit, so the caller must place no conversation
              breakpoint at all: every one would be a cache WRITE (billed at
              1.25x) that is never read.
        """
        fingerprints = [self._message_fingerprint(m) for m in all_messages]
        key = self._conversation_key(system_blocks, fingerprints)
        entry = self._prefix_fingerprints.get(key)

        if entry is None:
            self._prefix_fingerprints[key] = (fingerprints, None)
            self._prefix_fingerprints.move_to_end(key)
            while len(self._prefix_fingerprints) > _MAX_TRACKED_CONVERSATIONS:
                self._prefix_fingerprints.popitem(last=False)
            return None, "none"

        previous, previous_observed = entry

        if previous == fingerprints:
            # Byte-identical request. This is a retry or a fallback re-issue
            # of the SAME turn, not evidence that the orchestrator appends
            # stably -- concluding "observed 0" here would place a breakpoint
            # on a tail that the next real turn may well regenerate. Reuse
            # the last real observation and leave the stored list alone.
            self._prefix_fingerprints.move_to_end(key)
            return (
                previous_observed,
                "ok" if previous_observed is not None else "none",
            )

        lcp = 0
        for old, new in zip(previous, fingerprints):
            if old != new:
                break
            lcp += 1

        if lcp == 0:
            # Not even message 0 survived. Anthropic matches a cached prefix
            # from the very start of the request, so no breakpoint anywhere
            # in this conversation can ever be read back.
            self._prefix_fingerprints[key] = (fingerprints, previous_observed)
            self._prefix_fingerprints.move_to_end(key)
            return None, "no_shared_prefix"

        observed = max(0, len(previous) - lcp)
        self._prefix_fingerprints[key] = (fingerprints, observed)
        self._prefix_fingerprints.move_to_end(key)
        while len(self._prefix_fingerprints) > _MAX_TRACKED_CONVERSATIONS:
            self._prefix_fingerprints.popitem(last=False)
        return observed, "ok"

    def _apply_conversation_cache_control(
        self,
        all_messages: list[dict[str, Any]],
        unstable_suffix_len: int,
        has_ephemeral_signal: bool,
        remaining_budget: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Place up to 2 rolling cache breakpoints over stable conversation
        content, per Anthropic's documented multi-turn caching pattern.

        This replaces the old single "cache_control on messages[-1]" logic.
        That approach silently broke the moment the orchestrator started
        appending an ephemeral, regenerated-per-turn tail message (a
        `<system-reminder>` block with a live timestamp/git-status) after
        every ``context.get_messages_for_request()`` call: the cached
        prefix always ended in content that could never be reproduced on the
        next request, so it was written every turn and never re-read.
        Measured impact: cache_read frozen at a single constant for the
        whole session in 22/31 (71%) sampled sessions; aggregate
        write:read ratio 5.96x.

        The fix has two parts:

        1. Never place a breakpoint on a message known to be unstable
           (ephemeral and not persisted -- regenerated per request) --
           walk backward past ``unstable_suffix_len`` messages first. Unlike
           the prior trailing-only walk, this correctly excludes an unstable
           message wherever it sits in the eligible window, not only when
           it is literally last (see ``_unstable_suffix_length``).
        2. Use the *two* breakpoints Anthropic's docs describe for
           multi-turn conversations: one at the current stable boundary
           ("primary") and one right before the previous real user turn
           began ("secondary"). Because each new turn's primary breakpoint
           becomes the point right before the *next* turn's most recent user
           message, next turn's secondary lands exactly on this turn's
           primary -- giving a cache hit even as the conversation grows.

        Args:
            all_messages: Anthropic-formatted message array (context-prefix
                messages followed by conversation messages), mutated in
                place.
            unstable_suffix_len: number of trailing entries of the
                conversation region excluded from breakpoint eligibility
                (see ``_unstable_suffix_length``).
            has_ephemeral_signal: whether ephemeral status could be
                determined at all for this request (see above). If False,
                "not unstable" cannot be trusted as a real signal, and
                placing breakpoints on unverified content would risk
                repeating the exact bug this method exists to fix.
            remaining_budget: cache breakpoints left before hitting
                Anthropic's 4-breakpoint hard limit (system + tools may
                already have consumed some).

        Returns:
            (all_messages, breakpoints_used) -- never places more
            breakpoints than ``remaining_budget`` allows.
        """
        if not all_messages or not self.enable_prompt_caching or remaining_budget <= 0:
            return all_messages, 0

        if len(all_messages) < 2:
            # A single-message request has no conversation region at all: no
            # recurring stable prefix exists for a later request to hit, so
            # both a breakpoint and the "no metadata" warning below would be
            # vacuous here. This is the normal shape of one-shot utility
            # provider calls that bypass the main conversation entirely --
            # e.g. hooks-session-naming's `_generate_name`, which sends a
            # single user message with no history and no metadata on every
            # turn. Without this guard, that call fires the
            # has_ephemeral_signal warning below on every single turn of
            # every session, for a "problem" that was never actually
            # cacheable in the first place. Silent return (not even at
            # `debug`): there is nothing informative to report about a
            # request that was never a caching candidate. Do not remove this
            # thinking it's redundant with `has_ephemeral_signal` -- that
            # check is for genuine multi-message requests missing metadata,
            # a real signal this method must keep surfacing loudly.
            return all_messages, 0

        if not has_ephemeral_signal:
            # No message anywhere in this request carries a `metadata` dict,
            # so there is no basis for distinguishing stable history from a
            # regenerated-per-turn tail. Rather than silently reverting to
            # the old (measurably broken) "cache_control on messages[-1]"
            # placement, this is loud and skips conversation-region caching
            # for this call. See amplifier_module_loop_streaming's ephemeral
            # injection append sites (around the `inject_context` handling
            # in the main orchestrator loop) -- the appended message dict
            # must include `metadata={"ephemeral": True}` for this method to
            # ever place a conversation-region breakpoint.
            logger.warning(
                "[PROVIDER] Prompt caching: no message in this request carries "
                "a `metadata` dict, so ephemeral (regenerated-per-turn) "
                "messages cannot be distinguished from stable history. "
                "Skipping the conversation-region cache breakpoint(s) rather "
                "than risking a breakpoint on unstable content -- the "
                "orchestrator/context manager must stamp "
                "metadata={'ephemeral': True} on ephemeral injections for "
                "conversation-region prompt caching to take effect."
            )
            return all_messages, 0

        eligible_upper = len(all_messages) - unstable_suffix_len
        if eligible_upper <= 0:
            logger.warning(
                "[PROVIDER] Prompt caching: every message in this request is "
                "marked ephemeral -- no stable content available for a "
                "conversation-region cache breakpoint this turn."
            )
            return all_messages, 0

        cache_control: dict[str, Any] = {"type": "ephemeral"}

        primary_idx = self._last_safe_breakpoint_index(all_messages, eligible_upper - 1)
        if primary_idx is None:
            logger.warning(
                "[PROVIDER] Prompt caching: could not find a stable message "
                "boundary that neither splits a tool_use/tool_result pair nor "
                "lands on an empty text block -- skipping conversation-region "
                "cache breakpoint(s)."
            )
            return all_messages, 0

        breakpoints_used = 0
        self._stamp_last_block(all_messages[primary_idx], cache_control)
        breakpoints_used += 1

        if remaining_budget >= 2:
            secondary_idx = self._find_rolling_secondary_index(
                all_messages, primary_idx, eligible_upper
            )
            if secondary_idx is not None and secondary_idx != primary_idx:
                self._stamp_last_block(all_messages[secondary_idx], cache_control)
                breakpoints_used += 1

        return all_messages, breakpoints_used

    def _find_rolling_secondary_index(
        self,
        all_messages: list[dict[str, Any]],
        primary_idx: int,
        eligible_upper: int,
    ) -> int | None:
        """Find the rolling second breakpoint: the stable message right
        before the most recent *real* user turn began.

        A "real" user turn is a user message that is not purely a
        tool_result batch (i.e. an actual new user query, not tool output
        being fed back). Placing the second breakpoint one message before
        that turn's start means: next turn, once a new user message and its
        round-trip are appended, this same index becomes the boundary
        "right before the [new] most recent real user turn" that the
        primary breakpoint would have used last time -- so one of the two
        breakpoints keeps landing on a previously-cached boundary as the
        conversation grows, per Anthropic's documented multi-turn caching
        pattern.

        Deliberately left alone under the system-reminder redesign's
        pre-user reminder block (a stable, persisted, `role="user"` message
        written immediately before the turn's real user message -- see
        ``_unstable_suffix_length``). Verified empirically (see
        tests/test_prompt_cache_breakpoints.py's T-W5-04 test), not just
        derived on paper -- the actual mechanism is subtler than "the block
        becomes the previous turn's primary":

        At iteration 1 of a fresh turn N+1 (request ends `[..., block N+1,
        user N+1]`, no assistant reply yet), `primary_idx` IS `user N+1`'s
        own index. This method's search starts AT `primary_idx` itself, and
        since that message is `role="user"` and not a tool_result batch, it
        matches on the very first loop iteration -- `last_user_turn_idx`
        resolves to `user N+1`'s own index (not the block's), and the
        secondary is `safe(last_user_turn_idx - 1)`, which lands ON `block
        N+1` (it sits directly before `user N+1`).

        Once the model replies (iteration 2+: `[..., block N+1, user N+1,
        assistant N+1(tool_use), tool_result, ...]`), `primary_idx` advances
        into the tool_result batch. The search now walks PAST the
        tool_result (excluded: it IS a tool_result batch) and past the
        assistant message, and matches `user N+1` the same way -- so the
        secondary lands on `block N+1` AGAIN, at the same position, across
        every iteration of the SAME turn. That is the real rolling-overlap
        property this reminder-block shape gets from this method, unchanged:
        a consistent secondary target across a turn's own tool-loop
        iterations. (A cross-TURN overlap -- turn N+1's secondary matching
        turn N's OWN primary from the request that generated it -- does NOT
        hold for this shape: that hypothetical prior request also ends in
        `[..., block N, user N]` with no assistant reply yet, so its own
        primary is `user N`'s index, not `block N`'s -- there is no shared
        position between the two calls in that comparison. This is a
        pre-existing property of "primary lands on a user-role message
        when nothing follows it yet", not something the system-reminder
        redesign changes.) Do not "fix" this to skip reminder blocks --
        skipping them here would break the WITHIN-turn overlap that
        genuinely exists.
        """
        last_user_turn_idx: int | None = None
        for idx in range(min(primary_idx, eligible_upper - 1), -1, -1):
            msg = all_messages[idx]
            if msg.get("role") == "user" and not self._is_tool_result_message(msg):
                last_user_turn_idx = idx
                break

        if last_user_turn_idx is None or last_user_turn_idx == 0:
            # No *earlier* real user turn exists. This is the normal shape of
            # an agentic tool loop: one instruction at index 0, then N rounds
            # of assistant(tool_use)/user(tool_result) with no further user
            # input. It does NOT mean there is no stable earlier boundary --
            # every completed tool round before the primary is permanently
            # frozen, and is exactly the content the next request needs to hit.
            #
            # Returning None here (the previous behaviour) left only the single
            # advancing primary, which can never overlap itself: every request
            # wrote a new prefix at the 1.25x write premium and read back
            # nothing, for the entire run. That is precisely the pre-two-
            # breakpoint failure mode this method exists to fix -- it was still
            # live for every sub-agent delegation, /goal run, recipe step, and
            # the tool-heavy first turn of every session, because none of those
            # shapes contain a second real user turn.
            #
            # Lag one complete tool round behind the primary instead. In the
            # canonical shape this branch handles, a round is two messages --
            # the assistant tool_use and the single batched user tool_result
            # that answers it (parallel calls fan out as multiple blocks
            # inside that one message, not as extra messages). So
            # primary_idx - 2 places this request's secondary precisely where
            # the previous request's primary sat -- the same rolling overlap
            # the real-user-turn path above achieves.
            #
            # The -2 is a heuristic starting point, not an invariant: a
            # caller that interleaves narration or splits tool_results across
            # messages shifts the stride. _last_safe_breakpoint_index absorbs
            # that -- it walks further back from this index if the boundary
            # would split a tool_use/tool_result pair or stamp an empty text
            # block. A mis-stride costs a cache read, never a malformed
            # request; the worst case degrades to today's single-breakpoint
            # behaviour rather than breaking the call.
            if primary_idx < 2:
                return None  # no completed round behind the primary yet
            return self._last_safe_breakpoint_index(all_messages, primary_idx - 2)

        return self._last_safe_breakpoint_index(all_messages, last_user_turn_idx - 1)

    @staticmethod
    def _stamp_last_block(msg: dict[str, Any], cache_control: dict[str, Any]) -> None:
        """Add cache_control to the last content block of a message,
        converting string content to a block array first if needed."""
        content = msg.get("content")
        if isinstance(content, list) and content:
            content[-1]["cache_control"] = dict(cache_control)
        elif isinstance(content, str):
            msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": dict(cache_control),
                }
            ]

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
            else:
                # Unknown block type (e.g. 'fallback' from Fable 5-class
                # models) — skip gracefully rather than crashing.
                logger.debug(
                    "[PROVIDER] Skipping unknown content block type: %s", block.type
                )
                continue

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

        # Anthropic's usage object may carry a per-TTL cache-write split under
        # `usage.cache_creation` (`.ephemeral_5m_input_tokens` /
        # `.ephemeral_1h_input_tokens`), in addition to the aggregate
        # `cache_creation_input_tokens` count. When present, this lets
        # compute_cost() bill 1h writes at their real 2x rate instead of the
        # 1.25x 5-minute rate. The isinstance guards are a graceful fallback
        # for response shapes (or test doubles) that don't carry real ints
        # here -- treated the same as "split absent".
        cache_creation_ttl_split = getattr(response.usage, "cache_creation", None)
        cache_creation_5m = getattr(
            cache_creation_ttl_split, "ephemeral_5m_input_tokens", None
        )
        cache_creation_1h = getattr(
            cache_creation_ttl_split, "ephemeral_1h_input_tokens", None
        )
        if not isinstance(cache_creation_5m, int):
            cache_creation_5m = None
        if not isinstance(cache_creation_1h, int):
            cache_creation_1h = None

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
            cache_creation_5m_input_tokens=cache_creation_5m,
            cache_creation_1h_input_tokens=cache_creation_1h,
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
        """Close the underlying Anthropic client to prevent resource leaks.

        Resets ``self._client`` to ``None`` so the ``client`` property's
        lazy-init contract still holds after close(): that property only
        constructs a client when ``self._client is None``, so leaving a
        closed client in place would make every subsequent call reuse it
        and fail permanently with "Cannot send a request, as the client
        has been closed." Clearing it lets the next use lazily rebuild a
        fresh client, and makes close() idempotent.

        The close is HARD BOUNDED at ``close_timeout`` seconds (config key;
        default 5.0). ``AsyncAnthropic.close()`` awaits
        ``httpx.AsyncClient.aclose()``, which has no deadline of its own: on
        a half-closed (CLOSE-WAIT) connection it can block forever. Session
        cleanup runs inside the ``finally`` that PRECEDES a CLI command's
        return, so an unbounded close here does not merely leak a socket --
        it swallows the result of a completed run (recipes-8sr: 28 minutes,
        two anthropic:443 sockets in CLOSE-WAIT, process asleep in the
        await). On timeout the httpx client is abandoned, with a WARNING;
        the process exit reclaims the socket.

        There is no faster escape than abandoning it: anthropic 1.4.0
        exposes only the async ``close()``, and neither
        ``httpx.AsyncClient`` (0.28.1) nor ``httpcore.AsyncConnectionPool``
        exposes a synchronous close of an async transport.
        """
        client = self._client
        if client is None:
            return
        # Hand off and clear the slot FIRST: the lazy-init contract and
        # idempotency must hold even on the timeout path, where we never
        # learn whether the close finished.
        self._client = None
        close_task = asyncio.ensure_future(client.close())
        close_task.add_done_callback(_retrieve_task_exception)
        try:
            # shield: preserves the pre-existing contract that cancelling
            # our CALLER does not cancel a close already in flight.
            # wait_for: bounds it. Shield's outer future is a plain Future,
            # so cancelling it completes immediately -- this returns within
            # the timeout even though the inner aclose() ignores deadlines
            # and cancellation alike.
            await asyncio.wait_for(
                asyncio.shield(close_task), timeout=self._close_timeout
            )
        except asyncio.CancelledError:
            # Caller cancelled: leave the shielded close running, exactly as
            # before this bound existed.
            pass
        except TimeoutError:
            # Request cancellation and walk away. Deliberately NOT awaited:
            # awaiting a call that ignores cancellation would reintroduce
            # the unbounded wait this bound exists to prevent.
            close_task.cancel()
            logger.warning(
                "[PROVIDER] %s HTTP client close exceeded %.1fs "
                "(half-closed/CLOSE-WAIT connection?) -- abandoning the httpx "
                "client for provider instance %s id=0x%x. The socket is "
                "reclaimed at process exit. Raise the `close_timeout` config "
                "key if this is a false alarm.",
                self.api_label,
                self._close_timeout,
                self.name,
                id(self),
            )
