"""Synthetic, unspecified Models metadata for messages-only unit mocks.

Explicit SDK transports and explicit Models mocks remain authoritative. This
fixture does not qualify vendor metadata: null limits/capabilities deliberately
leave the provider's existing static fallback policy in charge.
"""

from datetime import datetime, timezone
from functools import wraps
from unittest.mock import Mock
from weakref import WeakKeyDictionary

import httpx2
from anthropic._base_client import AsyncAPIClient, AsyncHttpxClientWrapper
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
    original_init = AsyncAPIClient.__init__
    original = AsyncModels.retrieve
    default_clients = WeakKeyDictionary()
    calls = []

    @wraps(original_init)
    def initialize(client, *args, **kwargs):
        original_init(client, *args, **kwargs)
        if kwargs.get("http_client") is None:
            http_client = client._client
            default_clients[client] = (
                http_client,
                http_client._transport,
                dict(http_client._mounts),
            )

    @wraps(original)
    async def retrieve(resource, model_id, *args, **kwargs):
        client = resource._client
        http_client = client._client
        defaults = default_clients.get(client)
        transport = http_client._transport_for_url(
            client.base_url.join(f"models/{model_id}")
        )
        if (
            defaults is not None
            and http_client is defaults[0]
            and http_client._transport is defaults[1]
            and http_client._mounts == defaults[2]
            and type(http_client) is AsyncHttpxClientWrapper
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

    # Class equality alone cannot distinguish an explicit same-type override.
    monkeypatch.setattr(AsyncAPIClient, "__init__", initialize)
    monkeypatch.setattr(AsyncModels, "retrieve", retrieve)
    return calls
