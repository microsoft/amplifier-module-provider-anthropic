"""Regression tests: cost_usd Decimal must not crash JSON serialization at emission boundary.

PR #52 ("feat(m2): provider-anthropic cost stamping") introduced cost_usd as a
``decimal.Decimal`` on the ``Usage`` model.  Python's built-in json module cannot
serialize ``Decimal``, so any downstream ``json.dumps()`` on the ``llm:response``
event dict (JSONL logger, redaction layer, etc.) would raise::

    TypeError: Object of type Decimal is not JSON serializable

This produced 44 of 90 failures in the production matrix run on 2026-05-08.

Root causes fixed
-----------------
* Line 2686 (now 2687): ``_event_usage["cost_usd"]`` emission boundary — Decimal
  converted to ``str`` before being placed in the event dict.
* Line 363: ``session.cost`` contributor lambda — Decimal converted to ``str``
  before being returned from the capability.

Tests
-----
(a) llm:response event is JSON-serializable when model has known rates (main regression)
(b) cost_usd field in the event is a str, not a Decimal, for a known model
(c) cost_usd field in the event is None for an unknown model (null in JSON)
(d) Full JSON round-trip: str value survives json.dumps / json.loads unchanged
(e) Usage model stores Decimal internally — fix must be at emission boundary, not storage
"""

import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace
from typing import cast

import pytest
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_anthropic import AnthropicProvider

from tests._helpers import FakeCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider() -> AnthropicProvider:
    """Minimal provider wired to a FakeCoordinator so hook events are captured."""
    provider = AnthropicProvider(
        api_key="test-key",
        config={"use_streaming": False, "max_retries": 0},
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _make_raw_response(model: str = "claude-sonnet-4-5-20250929") -> MagicMock:
    """Build a minimal mock Anthropic raw-response object (no cache tokens)."""
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Hi")],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        stop_reason="end_turn",
        model=model,
    )
    # MagicMock (not AsyncMock) — raw.parse() is called synchronously in the provider
    raw = MagicMock()
    raw.parse.return_value = response
    raw.headers = {}
    return raw


# ---------------------------------------------------------------------------
# (a) Main regression: llm:response event is JSON-serializable for known model
# ---------------------------------------------------------------------------


def test_llm_response_event_is_json_serializable_known_model():
    """Known model → compute_cost() returns Decimal → event MUST be json.dumps-able.

    This is the exact scenario that crashed 44/90 runs after PR #52 merged.
    ``claude-sonnet-4-5-20250929`` is registered in ``_RATES``, so
    ``compute_cost()`` returns a non-None ``Decimal``.  Before the fix,
    ``json.dumps()`` on the emitted event dict raised ``TypeError``.
    """
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response("claude-sonnet-4-5-20250929")
    )

    asyncio.run(provider.complete(_simple_request()))

    hooks = provider.coordinator.hooks  # type: ignore[attr-defined]
    llm_event = hooks.payload_for("llm:response")
    assert llm_event is not None, "No llm:response event was emitted"

    # Must not raise TypeError: Object of type Decimal is not JSON serializable
    serialized = json.dumps(llm_event)
    assert serialized  # non-empty


# ---------------------------------------------------------------------------
# (b) cost_usd field is str (not Decimal) in the emitted event for known model
# ---------------------------------------------------------------------------


def test_llm_response_event_cost_usd_is_str_for_known_model():
    """cost_usd in the event dict must be str, not Decimal, for a known model."""
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response("claude-sonnet-4-5-20250929")
    )

    asyncio.run(provider.complete(_simple_request()))

    hooks = provider.coordinator.hooks  # type: ignore[attr-defined]
    llm_event = hooks.payload_for("llm:response")
    assert llm_event is not None

    cost = llm_event["usage"]["cost_usd"]
    assert isinstance(cost, str), (
        f"cost_usd at emission boundary must be str, got {type(cost).__name__}: {cost!r}"
    )
    # The str must round-trip to a positive Decimal (full precision preserved)
    assert Decimal(cost) > 0, f"Parsed cost_usd should be > 0, got {cost!r}"


# ---------------------------------------------------------------------------
# (c) cost_usd is None (JSON null) for unknown model — regression for None case
# ---------------------------------------------------------------------------


def test_llm_response_event_cost_usd_is_none_for_unknown_model():
    """Unknown model → compute_cost() returns None → event has cost_usd: null.

    Before the fix, None serialized fine (→ JSON null).  Verify this still works
    and that we haven't accidentally stringified None into the literal "None".
    """
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response("claude-does-not-exist-9999")
    )

    asyncio.run(provider.complete(_simple_request()))

    hooks = provider.coordinator.hooks  # type: ignore[attr-defined]
    llm_event = hooks.payload_for("llm:response")
    assert llm_event is not None

    cost = llm_event["usage"]["cost_usd"]
    assert cost is None, (
        f"cost_usd should be None for unknown model, got {cost!r}"
    )

    # Must still be JSON-serializable (None → null in JSON)
    parsed = json.loads(json.dumps(llm_event))
    assert parsed["usage"]["cost_usd"] is None


# ---------------------------------------------------------------------------
# (d) Full JSON round-trip: value survives json.dumps/json.loads unchanged
# ---------------------------------------------------------------------------


def test_llm_response_event_cost_usd_round_trips_through_json():
    """The str cost_usd must survive a full JSON round-trip with no data loss."""
    provider = _make_provider()
    provider.client.messages.with_raw_response.create = AsyncMock(
        return_value=_make_raw_response("claude-sonnet-4-5-20250929")
    )

    asyncio.run(provider.complete(_simple_request()))

    hooks = provider.coordinator.hooks  # type: ignore[attr-defined]
    llm_event = hooks.payload_for("llm:response")
    assert llm_event is not None

    original_cost = llm_event["usage"]["cost_usd"]
    parsed = json.loads(json.dumps(llm_event))
    assert parsed["usage"]["cost_usd"] == original_cost, (
        "cost_usd value must survive JSON round-trip unchanged"
    )


# ---------------------------------------------------------------------------
# (e) Usage model stores Decimal internally — fix is at emission boundary
# ---------------------------------------------------------------------------


def test_usage_model_direct_access_is_decimal_but_model_dump_is_str():
    """result.usage.cost_usd is Decimal via direct attribute access, but
    model_dump() always stringifies it — even in plain (non-JSON) mode.

    The Usage model (amplifier_core.message_models.Usage) carries a
    `field_serializer("cost_usd", when_used="always")` that converts
    Decimal -> str on every model_dump() call, not just mode="json". This
    is deliberate (amplifier-core commit 91fa469, "required for M2 cost
    stamping (#73)") so that Decimal can never leak through a plain
    model_dump() anywhere downstream.

    Direct attribute access (`result.usage.cost_usd`) bypasses serialization
    entirely and keeps full Decimal precision, which is what internal
    arithmetic (e.g. cost accumulation in _add_cost) relies on.
    """
    from amplifier_module_provider_anthropic._cost import compute_cost

    provider = _make_provider()

    # Build a minimal raw response for a known model
    raw_response = SimpleNamespace(
        content=[],
        usage=SimpleNamespace(
            input_tokens=1_000,
            output_tokens=500,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        stop_reason="end_turn",
        model="claude-sonnet-4-5-20250929",
    )

    result = provider._convert_to_chat_response(raw_response)

    # Internal storage: still a Decimal (correct for precision)
    assert isinstance(result.usage.cost_usd, Decimal), (
        "Internal cost_usd must remain Decimal; conversion only at emission boundary"
    )

    # model_dump() always stringifies cost_usd via the shared Usage model's
    # field_serializer(when_used="always") — in BOTH plain and JSON mode.
    # Only direct attribute access (above) preserves Decimal.
    usage_dict = result.usage.model_dump()
    assert isinstance(usage_dict.get("cost_usd"), str), (
        "model_dump() must stringify cost_usd (Usage.serialize_cost_usd runs "
        "with when_used='always'); only direct attribute access stays Decimal"
    )
