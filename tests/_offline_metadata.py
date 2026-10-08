"""Synthetic, unspecified Models metadata for messages-only unit mocks.

Explicit SDK transports and explicit Models mocks remain authoritative. This
fixture does not qualify vendor metadata: null limits/capabilities deliberately
leave the provider's existing static fallback policy in charge.
"""

from datetime import datetime, timezone
from functools import wraps
from unittest.mock import Mock

import httpx2
from anthropic._base_client import AsyncHttpxClientWrapper
from anthropic.resources.models import AsyncModels
from anthropic.types import ModelInfo


def _messages_are_mocked(client):
    messages = client.messages
    methods = (
        messages.create,
        messages.stream,
        messages.count_tokens,
        messages.with_raw_response.create,
    )
    return isinstance(messages, Mock) or any(
        isinstance(method, Mock)
        or not getattr(method, "__module__", "").startswith("anthropic.")
        for method in methods
    )


def install_metadata_stub(monkeypatch):
    """Patch only the cold SDK interface left open by messages-only mocks."""
    original = AsyncModels.retrieve
    calls = []

    @wraps(original)
    async def retrieve(resource, model_id, *args, **kwargs):
        client = resource._client
        http_client = client._client
        transport = http_client._transport_for_url(
            client.base_url.join(f"models/{model_id}")
        )
        if (
            type(http_client) is AsyncHttpxClientWrapper
            and type(transport) is httpx2.AsyncHTTPTransport
            and _messages_are_mocked(client)
        ):
            calls.append(model_id)
            return ModelInfo(
                id=model_id,
                type="model",
                display_name="Synthetic offline metadata (unspecified capabilities)",
                created_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
                lifecycle="active",  # Required by SDK 1.12; synthetic, not discovery.
                capabilities=None,
                max_input_tokens=None,
                max_tokens=None,
            )
        return await original(resource, model_id, *args, **kwargs)

    monkeypatch.setattr(AsyncModels, "retrieve", retrieve)
    return calls
