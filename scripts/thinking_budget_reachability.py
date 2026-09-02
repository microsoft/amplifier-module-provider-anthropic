#!/usr/bin/env python3
"""Reachability matrix for `thinking_budget_tokens` — NO API SPEND.

Drives the real provider with a mocked transport and reads back the params it
would have put on the wire. Answers one question per cell: does an explicitly
configured `thinking_budget_tokens` reach `thinking.budget_tokens`, or is it
discarded?

Usage:
    python scripts/thinking_budget_reachability.py            # human table
    python scripts/thinking_budget_reachability.py --json OUT # machine-readable

Re-run this against any commit to compare. The `default-*` cells set NO
`thinking_budget_tokens` anywhere and exist to prove the default path is
unchanged (byte-identity): their `params` blob must be identical across
commits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amplifier_core import ModuleCoordinator  # noqa: E402
from amplifier_core.message_models import ChatRequest, Message  # noqa: E402

from amplifier_module_provider_anthropic import AnthropicProvider  # noqa: E402

HAIKU = "claude-haiku-4-5-20251001"
HAIKU_35 = "claude-haiku-3-5-20250929"  # supports_thinking=False
SONNET = "claude-sonnet-4-5-20250929"
ADAPTIVE = "claude-sonnet-4-6"  # supports_adaptive_thinking=True
ALWAYS_ON = "claude-fable-5-1"  # thinking_always_on=True


class _FakeHooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class _FakeCoordinator:
    def __init__(self) -> None:
        self.hooks = _FakeHooks()


class _DummyResponse:
    def __init__(self) -> None:
        self.content: list = []
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.stop_reason = "end_turn"
        self.model = HAIKU


class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def _wire_params(
    *,
    model: str,
    config: dict[str, Any],
    reasoning_effort: str | None,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Build one request and return (wire params, warnings emitted)."""
    full_config: dict[str, Any] = {
        "use_streaming": False,
        "max_retries": 0,
        "default_model": model,
        **config,
    }
    handler = _CapturingHandler()
    provider_logger = logging.getLogger("amplifier_module_provider_anthropic")
    provider_logger.addHandler(handler)
    try:
        provider = AnthropicProvider(api_key="test-key", config=full_config)
        provider.coordinator = cast(ModuleCoordinator, _FakeCoordinator())

        raw = MagicMock()
        raw.parse = AsyncMock(return_value=_DummyResponse())
        raw.headers = {}
        create = AsyncMock(return_value=raw)
        provider.client.messages.with_raw_response.create = create

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            reasoning_effort=reasoning_effort,
        )

        async def _drive() -> None:
            try:
                await provider.complete(request, **kwargs)
            finally:
                # Close inside the loop that opened the client, or the SDK's
                # transport teardown raises "Event loop is closed" at exit.
                await provider.close()

        asyncio.run(_drive())

        _, call_kwargs = create.call_args
        params = dict(call_kwargs)
        extra_body = params.pop("extra_body", None) or {}
        for key, value in extra_body.items():
            params.setdefault(key, value)
        params.pop("messages", None)
        return params, list(handler.records)
    finally:
        provider_logger.removeHandler(handler)


# (id, model, config, request.reasoning_effort, complete() kwargs)
CELLS: list[tuple[str, str, dict[str, Any], str | None, dict[str, Any]]] = [
    # --- the four silently-discarded configurations reported by lane d0q ---
    (
        "cfg-budget+request-effort-high",
        HAIKU,
        {"thinking_budget_tokens": 8000},
        "high",
        {},
    ),
    (
        "cfg-budget-64k+request-effort-high",
        HAIKU,
        {"thinking_budget_tokens": 64000},
        "high",
        {},
    ),
    (
        "cfg-budget+request-effort-low",
        HAIKU,
        {"thinking_budget_tokens": 8000},
        "low",
        {},
    ),
    (
        "cfg-budget+cfg-effort-low",
        HAIKU,
        {"thinking_budget_tokens": 8000, "reasoning_effort": "low"},
        None,
        {},
    ),
    # --- the fifth: thinking never turned on, so the budget is never read ---
    ("cfg-budget+no-effort", HAIKU, {"thinking_budget_tokens": 8000}, None, {}),
    (
        "cfg-budget+cfg-extended-thinking",
        HAIKU,
        {"thinking_budget_tokens": 8000, "extended_thinking": True},
        None,
        {},
    ),
    # --- kwargs must still outrank config ---
    (
        "kwargs-budget-beats-cfg-budget",
        HAIKU,
        {"thinking_budget_tokens": 8000},
        "high",
        {"thinking_budget_tokens": 16000},
    ),
    # --- an adaptive-thinking model: budget sizes max_tokens, not thinking ---
    (
        "cfg-budget-sonnet-effort-high",
        SONNET,
        {"thinking_budget_tokens": 8000},
        "high",
        {},
    ),
    (
        "cfg-budget-adaptive-model",
        ADAPTIVE,
        {"thinking_budget_tokens": 8000},
        "high",
        {},
    ),
    # --- a model that always thinks and rejects a thinking param entirely ---
    (
        "cfg-budget-always-on-model",
        ALWAYS_ON,
        {"thinking_budget_tokens": 8000},
        "high",
        {},
    ),
    # --- a model with no thinking support at all ---
    (
        "cfg-budget-no-thinking-model",
        HAIKU_35,
        {"thinking_budget_tokens": 8000},
        "high",
        {},
    ),
    # --- invalid / out-of-range explicit values ---
    ("cfg-budget-zero", HAIKU, {"thinking_budget_tokens": 0}, "high", {}),
    (
        "cfg-budget-not-a-number",
        HAIKU,
        {"thinking_budget_tokens": "not-a-number"},
        "high",
        {},
    ),
    # --- DEFAULT PATH: no thinking_budget_tokens anywhere. Must not move. ---
    ("default-haiku-effort-high", HAIKU, {}, "high", {}),
    ("default-haiku-effort-low", HAIKU, {}, "low", {}),
    ("default-haiku-effort-medium", HAIKU, {}, "medium", {}),
    ("default-haiku-effort-xhigh", HAIKU, {}, "xhigh", {}),
    ("default-haiku-no-effort", HAIKU, {}, None, {}),
    ("default-haiku-cfg-effort-high", HAIKU, {"reasoning_effort": "high"}, None, {}),
    ("default-sonnet-effort-high", SONNET, {}, "high", {}),
    ("default-sonnet-effort-low", SONNET, {}, "low", {}),
    ("default-sonnet-no-effort", SONNET, {}, None, {}),
    ("default-adaptive-effort-high", ADAPTIVE, {}, "high", {}),
    ("default-adaptive-effort-low", ADAPTIVE, {}, "low", {}),
    ("default-always-on-effort-high", ALWAYS_ON, {}, "high", {}),
    ("default-haiku35-effort-high", HAIKU_35, {}, "high", {}),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", dest="json_out", help="write full matrix to this path"
    )
    args = parser.parse_args()

    logging.disable(logging.INFO)  # keep INFO noise out; WARNING still captured

    results = []
    for cell_id, model, config, effort, kwargs in CELLS:
        params, warnings = _wire_params(
            model=model, config=config, reasoning_effort=effort, kwargs=kwargs
        )
        thinking = params.get("thinking")
        asked = kwargs.get(
            "thinking_budget_tokens", config.get("thinking_budget_tokens")
        )
        sent = thinking.get("budget_tokens") if isinstance(thinking, dict) else None
        budget_warnings = [w for w in warnings if "thinking_budget_tokens" in w]
        results.append(
            {
                "id": cell_id,
                "model": model,
                "config": config,
                "request_reasoning_effort": effort,
                "kwargs": kwargs,
                "asked_budget": asked,
                "wire_thinking": thinking,
                "wire_budget_tokens": sent,
                "wire_max_tokens": params.get("max_tokens"),
                "honored": asked is not None and sent == asked,
                "warned": bool(budget_warnings),
                "warnings": budget_warnings,
                "params": params,
            }
        )

    width = max(len(r["id"]) for r in results)
    print(f"{'cell'.ljust(width)}  asked     sent      max_tok   verdict")
    print("-" * (width + 46))
    for r in results:
        if r["asked_budget"] is None:
            verdict = "default path (must be byte-identical across commits)"
        elif r["honored"]:
            verdict = "HONORED"
        elif r["warned"]:
            verdict = "discarded, but WARNED"
        else:
            verdict = "SILENTLY DISCARDED  <-- defect"
        print(
            f"{r['id'].ljust(width)}  "
            f"{str(r['asked_budget']).ljust(8)}  "
            f"{str(r['wire_budget_tokens']).ljust(8)}  "
            f"{str(r['wire_max_tokens']).ljust(8)}  "
            f"{verdict}"
        )

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str)
        )
        print(f"\nwrote {args.json_out}")

    silent = [
        r
        for r in results
        if r["asked_budget"] is not None and not r["honored"] and not r["warned"]
    ]
    return 1 if silent else 0


if __name__ == "__main__":
    raise SystemExit(main())
