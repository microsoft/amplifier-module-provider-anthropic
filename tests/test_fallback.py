"""Tests for temporary overload downgrade behavior."""

import asyncio
import json
import time
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
from anthropic._exceptions import OverloadedError as AnthropicOverloadedError
import pytest

import amplifier_module_provider_anthropic as anthropic_module
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import DummyResponse, FakeCoordinator


@pytest.fixture(autouse=True)
def clear_fallback_windows():
    anthropic_module._clear_fallback_windows()
    yield
    anthropic_module._clear_fallback_windows()


def _make_provider(default_model: str, **config_overrides) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "default_model": default_model,
            "max_retries": 3,
            "min_retry_delay": 0.01,
            "max_retry_delay": 60.0,
            "retry_jitter": False,
            **config_overrides,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _request_with_max_output_tokens(max_output_tokens: int) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="Hello")],
        max_output_tokens=max_output_tokens,
    )


def _make_raw_success(model: str) -> MagicMock:
    raw_mock = MagicMock()
    raw_mock.parse.return_value = DummyResponse(model=model)
    raw_mock.headers = {}
    return raw_mock


def _make_sdk_overloaded_error(
    retry_after: float | None = None,
) -> AnthropicOverloadedError:
    mock_response = MagicMock()
    mock_response.status_code = 529
    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["retry-after"] = str(retry_after)
    mock_response.headers = headers
    return AnthropicOverloadedError("overloaded", response=mock_response, body=None)


def _make_sdk_server_error() -> anthropic.InternalServerError:
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.headers = {}
    return anthropic.InternalServerError(
        "server error", response=mock_response, body=None
    )


def _make_sdk_rate_limit_overloaded_error() -> anthropic.RateLimitError:
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    body = {"error": {"type": "overloaded_error", "message": "Overloaded"}}
    return anthropic.RateLimitError("overloaded", response=mock_response, body=body)


def _runtime_model_info(
    *,
    max_input_tokens: int | None = None,
    max_tokens: int | None = None,
    supports_thinking: bool | None = None,
    supports_adaptive_thinking: bool | None = None,
) -> anthropic_module._RuntimeModelInfo:
    return anthropic_module._RuntimeModelInfo(
        max_input_tokens=max_input_tokens,
        max_tokens=max_tokens,
        supports_thinking=supports_thinking,
        supports_adaptive_thinking=supports_adaptive_thinking,
    )


class TestTemporaryFallbackOnOverload:
    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_529_opus_overload_falls_back_to_sonnet(self, mock_sleep):
        provider = _make_provider(
            "claude-opus-4-6",
            fallback_on_overload=True,
            fallback_retry_count=1,
            fallback_cooldown_seconds=300,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_overloaded_error(),
                _make_sdk_overloaded_error(),
                _make_raw_success("claude-sonnet-4-6"),
            ]
        )

        result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        models = [
            call.kwargs["model"]
            for call in provider.client.messages.with_raw_response.create.await_args_list
        ]
        assert models == [
            "claude-opus-4-6",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        ]

        fake_coord = cast(FakeCoordinator, provider.coordinator)
        open_events = [e for e in fake_coord.hooks.events if e[0] == "provider:fallback_open"]
        assert len(open_events) == 1
        assert open_events[0][1]["requested_model"] == "claude-opus-4-6"
        assert open_events[0][1]["fallback_model"] == "claude-sonnet-4-6"

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_fallback_clamps_max_tokens_to_lower_model_ceiling(self, mock_sleep):
        provider = _make_provider(
            "claude-opus-4-6",
            fallback_on_overload=True,
            fallback_retry_count=1,
            fallback_cooldown_seconds=300,
        )
        provider._get_runtime_model_info = AsyncMock(
            side_effect=[
                _runtime_model_info(max_input_tokens=1_000_000, max_tokens=128_000),
                _runtime_model_info(max_input_tokens=1_000_000, max_tokens=64_000),
            ]
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_overloaded_error(),
                _make_sdk_overloaded_error(),
                _make_raw_success("claude-sonnet-4-6"),
            ]
        )

        result = asyncio.run(
            provider.complete(_request_with_max_output_tokens(120_000))
        )

        assert result is not None
        max_tokens = [
            call.kwargs["max_tokens"]
            for call in provider.client.messages.with_raw_response.create.await_args_list
        ]
        assert max_tokens == [120_000, 120_000, 64_000]

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_fallback_clamps_thinking_budget_for_lower_model(self, mock_sleep):
        provider = _make_provider(
            "claude-sonnet-4-6",
            fallback_on_overload=True,
            fallback_retry_count=1,
            fallback_cooldown_seconds=300,
        )
        provider._get_runtime_model_info = AsyncMock(
            side_effect=[
                _runtime_model_info(
                    max_input_tokens=1_000_000,
                    max_tokens=64_000,
                    supports_thinking=True,
                    supports_adaptive_thinking=True,
                ),
                _runtime_model_info(
                    max_input_tokens=200_000,
                    max_tokens=64_000,
                    supports_thinking=True,
                    supports_adaptive_thinking=False,
                ),
            ]
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_overloaded_error(),
                _make_sdk_overloaded_error(),
                _make_raw_success("claude-haiku-4-5"),
            ]
        )

        result = asyncio.run(
            provider.complete(
                _simple_request(),
                extended_thinking=True,
                thinking_budget_tokens=120_000,
            )
        )

        assert result is not None
        fallback_call = provider.client.messages.with_raw_response.create.await_args_list[
            -1
        ]
        assert fallback_call.kwargs["model"] == "claude-haiku-4-5"
        assert fallback_call.kwargs["max_tokens"] == 64_000
        assert fallback_call.kwargs["thinking"]["type"] == "enabled"
        assert fallback_call.kwargs["thinking"]["budget_tokens"] == 63_999

    def test_enable_1m_context_does_not_become_global_beta_header(self):
        provider = _make_provider("claude-sonnet-4-6", enable_1m_context=True)
        assert anthropic_module.BETA_HEADER_1M_CONTEXT not in provider._beta_headers

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_sonnet_46_adds_context_beta_header_when_enabled(self, mock_sleep):
        provider = _make_provider("claude-sonnet-4-6", enable_1m_context=True)
        provider._get_runtime_model_info = AsyncMock(
            return_value=_runtime_model_info(
                max_input_tokens=1_000_000,
                max_tokens=64_000,
            )
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_success("claude-sonnet-4-6")
        )

        result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        call_kwargs = provider.client.messages.with_raw_response.create.await_args.kwargs
        beta_header = call_kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert anthropic_module.BETA_HEADER_1M_CONTEXT in beta_header

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_haiku_fallback_does_not_inherit_context_beta_header(self, mock_sleep):
        provider = _make_provider(
            "claude-sonnet-4-6",
            enable_1m_context=True,
            fallback_on_overload=True,
            fallback_retry_count=1,
            fallback_cooldown_seconds=300,
        )
        provider._get_runtime_model_info = AsyncMock(
            side_effect=[
                _runtime_model_info(max_input_tokens=1_000_000, max_tokens=64_000),
                _runtime_model_info(max_input_tokens=200_000, max_tokens=64_000),
            ]
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_overloaded_error(),
                _make_sdk_overloaded_error(),
                _make_raw_success("claude-haiku-4-5"),
            ]
        )

        result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        first_call = provider.client.messages.with_raw_response.create.await_args_list[0]
        fallback_call = provider.client.messages.with_raw_response.create.await_args_list[
            -1
        ]
        first_header = first_call.kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        fallback_header = fallback_call.kwargs.get("extra_headers", {}).get(
            "anthropic-beta", ""
        )
        assert anthropic_module.BETA_HEADER_1M_CONTEXT in first_header
        assert anthropic_module.BETA_HEADER_1M_CONTEXT not in fallback_header

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_sonnet_45_still_adds_context_beta_header_when_enabled(self, mock_sleep):
        provider = _make_provider("claude-sonnet-4-5", enable_1m_context=True)
        provider._get_runtime_model_info = AsyncMock(
            return_value=_runtime_model_info(
                max_input_tokens=200_000,
                max_tokens=64_000,
            )
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_success("claude-sonnet-4-5")
        )

        result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        call_kwargs = provider.client.messages.with_raw_response.create.await_args.kwargs
        beta_header = call_kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert anthropic_module.BETA_HEADER_1M_CONTEXT in beta_header

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_persisted_breaker_state_is_opt_in(self, mock_sleep, tmp_path):
        state_path = tmp_path / "anthropic-fallback-state.json"
        provider = _make_provider(
            "claude-opus-4-6",
            fallback_on_overload=True,
            persist_fallback_state=False,
            fallback_state_path=str(state_path),
            fallback_retry_count=1,
            fallback_cooldown_seconds=300,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_overloaded_error(),
                _make_sdk_overloaded_error(),
                _make_raw_success("claude-sonnet-4-6"),
            ]
        )

        result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        assert not state_path.exists()

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_persisted_breaker_state_is_used_by_fresh_provider(
        self, mock_sleep, tmp_path
    ):
        state_path = tmp_path / "anthropic-fallback-state.json"
        opening_provider = _make_provider(
            "claude-opus-4-6",
            fallback_on_overload=True,
            persist_fallback_state=True,
            fallback_state_path=str(state_path),
            fallback_retry_count=1,
            fallback_cooldown_seconds=300,
        )
        opening_provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_overloaded_error(),
                _make_sdk_overloaded_error(),
                _make_raw_success("claude-sonnet-4-6"),
            ]
        )

        first_result = asyncio.run(opening_provider.complete(_simple_request()))

        assert first_result is not None
        assert state_path.exists()

        persisted = json.loads(state_path.read_text())
        assert persisted["version"] == 1
        assert persisted["windows"]["opus"]["fallback_model"] == "claude-sonnet-4-6"

        anthropic_module._clear_fallback_windows()

        fresh_provider = _make_provider(
            "claude-opus-4-6",
            fallback_on_overload=True,
            persist_fallback_state=True,
            fallback_state_path=str(state_path),
        )
        fresh_provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_success("claude-sonnet-4-6")
        )

        second_result = asyncio.run(fresh_provider.complete(_simple_request()))

        assert second_result is not None
        models = [
            call.kwargs["model"]
            for call in fresh_provider.client.messages.with_raw_response.create.await_args_list
        ]
        assert models == ["claude-sonnet-4-6"]

    def test_expired_persisted_breaker_state_is_ignored(self, tmp_path):
        state_path = tmp_path / "anthropic-fallback-state.json"
        state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "updated_at": time.time(),
                    "updated_by_pid": 1234,
                    "windows": {
                        "opus": {
                            "requested_model": "claude-opus-4-6",
                            "fallback_model": "claude-sonnet-4-6",
                            "opened_at": time.time() - 600,
                            "until": time.time() - 1,
                            "opened_by_pid": 1234,
                            "error_type": "ProviderUnavailableError",
                            "error_message": "overloaded",
                        }
                    },
                }
            )
        )

        provider = _make_provider(
            "claude-opus-4-6",
            fallback_on_overload=True,
            persist_fallback_state=True,
            fallback_state_path=str(state_path),
        )
        provider._read_shared_fallback_state()

        assert anthropic_module._get_active_fallback_window("opus") is None

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_active_breaker_routes_future_requests_to_fallback_model(self, mock_sleep):
        anthropic_module._set_fallback_window(
            "opus",
            anthropic_module._FallbackWindow(
                requested_model="claude-opus-4-6",
                fallback_model="claude-sonnet-4-6",
                opened_at=time.time(),
                until=time.time() + 300,
                opened_by_pid=1234,
                error_type="ProviderUnavailableError",
                error_message="overloaded",
            ),
        )

        provider = _make_provider(
            "claude-opus-4-6",
            fallback_on_overload=True,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            return_value=_make_raw_success("claude-sonnet-4-6")
        )

        result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        models = [
            call.kwargs["model"]
            for call in provider.client.messages.with_raw_response.create.await_args_list
        ]
        assert models == ["claude-sonnet-4-6"]

        fake_coord = cast(FakeCoordinator, provider.coordinator)
        active_events = [
            e for e in fake_coord.hooks.events if e[0] == "provider:fallback_active"
        ]
        assert len(active_events) == 1
        assert active_events[0][1]["requested_model"] == "claude-opus-4-6"
        assert active_events[0][1]["effective_model"] == "claude-sonnet-4-6"

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_non_overload_errors_keep_full_retry_budget_on_same_model(
        self, mock_sleep
    ):
        provider = _make_provider(
            "claude-opus-4-6",
            fallback_on_overload=True,
            fallback_retry_count=1,
            max_retries=3,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_server_error(),
                _make_sdk_server_error(),
                _make_raw_success("claude-opus-4-6"),
            ]
        )

        result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        models = [
            call.kwargs["model"]
            for call in provider.client.messages.with_raw_response.create.await_args_list
        ]
        assert models == [
            "claude-opus-4-6",
            "claude-opus-4-6",
            "claude-opus-4-6",
        ]

        fake_coord = cast(FakeCoordinator, provider.coordinator)
        open_events = [e for e in fake_coord.hooks.events if e[0] == "provider:fallback_open"]
        assert open_events == []

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_sonnet_overload_falls_back_to_haiku(self, mock_sleep):
        provider = _make_provider(
            "claude-sonnet-4-6",
            fallback_on_overload=True,
            fallback_retry_count=1,
            fallback_cooldown_seconds=300,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_overloaded_error(),
                _make_sdk_overloaded_error(),
                _make_raw_success("claude-haiku-4-5"),
            ]
        )

        result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        models = [
            call.kwargs["model"]
            for call in provider.client.messages.with_raw_response.create.await_args_list
        ]
        assert models == [
            "claude-sonnet-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ]

    @patch("asyncio.sleep", new_callable=AsyncMock)
    def test_429_overloaded_body_also_triggers_fallback(self, mock_sleep):
        provider = _make_provider(
            "claude-opus-4-6",
            fallback_on_overload=True,
            fallback_retry_count=1,
            fallback_cooldown_seconds=300,
        )
        provider.client.messages.with_raw_response.create = AsyncMock(
            side_effect=[
                _make_sdk_rate_limit_overloaded_error(),
                _make_sdk_rate_limit_overloaded_error(),
                _make_raw_success("claude-sonnet-4-6"),
            ]
        )

        result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        models = [
            call.kwargs["model"]
            for call in provider.client.messages.with_raw_response.create.await_args_list
        ]
        assert models == [
            "claude-opus-4-6",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        ]
