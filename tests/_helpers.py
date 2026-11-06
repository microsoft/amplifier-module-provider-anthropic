"""Shared test helpers for amplifier-module-provider-anthropic tests.

Contains only the classes that are truly universal across all test files:
  - FakeHooks
  - FakeCoordinator
  - DummyResponse

Per-file helpers (_make_provider, _make_raw_mock, _get_api_params) are
intentionally kept local — each file has a different signature.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock


def models_page(models):
    """Raw/parsed page pair for unit tests; real SDK coverage is separate."""
    data = [{"lifecycle": "active", **vars(model)} for model in models]
    return SimpleNamespace(
        json=AsyncMock(return_value={
            "data": data, "has_more": False,
            "first_id": models[0].id if models else None,
            "last_id": models[-1].id if models else None,
        }),
        parse=AsyncMock(return_value=SimpleNamespace(data=models)),
    )


class FakeHooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))

    def emitted_names(self) -> list[str]:
        return [name for name, _ in self.events]

    def payload_for(self, event_name: str) -> dict | None:
        for name, payload in self.events:
            if name == event_name:
                return payload
        return None


class FakeCoordinator:
    def __init__(self) -> None:
        self.hooks = FakeHooks()


class DummyResponse:
    """Minimal Anthropic API response stub."""

    def __init__(
        self,
        content: list | None = None,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> None:
        self.content = content if content is not None else []
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.stop_reason = "end_turn"
        self.model = model
