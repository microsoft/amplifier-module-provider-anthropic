"""inference_geo threads from config/options onto the wire and into cost."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import DummyResponse, FakeCoordinator


def _make_provider(model: str = "claude-opus-5-5", **overrides: Any) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="x",
        config={
            "default_model": model,
            "max_retries": 0,
            "use_streaming": False,
            **overrides,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _run(provider: AnthropicProvider, **opts):
    response = DummyResponse(model=provider.default_model)
    response.usage.input_tokens = 1_000_000
    response.usage.output_tokens = 1_000_000
    raw_response = MagicMock()
    raw_response.parse = AsyncMock(return_value=response)
    raw_response.headers = {}
    create = AsyncMock(return_value=raw_response)
    provider.client.messages.with_raw_response.create = create
    request = ChatRequest(messages=[Message(role="user", content="hi")])
    result = asyncio.run(provider.complete(request, **opts))
    return create.call_args.kwargs, result


def test_config_inference_geo_us_reaches_wire_and_cost():
    provider = _make_provider(inference_geo="us")
    params, result = _run(provider)
    assert params["inference_geo"] == "us"
    assert result.usage.cost_usd == Decimal("26.40")


def test_no_inference_geo_configured_omits_wire_param():
    provider = _make_provider()
    params, result = _run(provider)
    assert "inference_geo" not in params
    assert result.usage.cost_usd == Decimal("24.00")


def test_per_request_option_overrides_config():
    provider = _make_provider(inference_geo="us")
    params, _ = _run(provider, inference_geo=None)
    assert params.get("inference_geo") is None
