"""Optional, logical-call-local, payload-free observations (not a liveness timer)."""

import math
import time
from typing import Any

from anthropic import Timeout


def _seconds(value: Any, *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        return None
    return float(value)


def effective_limits(
    elapsed: Any, sdk_timeout: Any, *, phase_override: bool = False
) -> dict[str, Any]:
    """Report the actual scalar elapsed policy separately from SDK phase policy."""
    elapsed_seconds = _seconds(elapsed, positive=True)
    if isinstance(sdk_timeout, Timeout):
        phases = {name: _seconds(getattr(sdk_timeout, name)) for name in (
            "connect", "pool", "read", "write"
        )}
    elif sdk_timeout is None or isinstance(sdk_timeout, (int, float)):
        phases = {name: _seconds(sdk_timeout) for name in (
            "connect", "pool", "read", "write"
        )}
    else:
        phases = dict.fromkeys(("connect", "pool", "read", "write"))
    mode = "elapsed" if elapsed_seconds is not None else "none"
    if mode == "none" and (
        phases["read"] is not None or phases["write"] is not None
        or (phase_override and isinstance(sdk_timeout, Timeout))
    ):
        mode = "phase"
    return {
        "mode": mode,
        "elapsed_seconds": elapsed_seconds,
        **{f"{name}_seconds": value for name, value in phases.items()},
    }


class RequestObservation:
    """No tasks, timers, raw events, identifiers, or response content retained."""

    def __init__(self, hooks: Any, limits: dict[str, Any] | None = None):
        self.hooks = hooks
        self.limits = limits if limits is not None else {}
        self.attempt = 0
        self.last_activity: float | None = None
        self.pending: tuple[int, dict[str, Any]] | None = None
        # Latest generation boundary in this logical call; never payload/IDs.
        self.phase = "not_dispatched"
        self.usage: dict[str, Any] | None = None

    async def _emit(
        self, observation: str, attempt: int, limits: dict[str, Any]
    ) -> None:
        emit = getattr(self.hooks, "emit", None)
        if not callable(emit):
            return
        try:
            await emit("llm:progress", {
                "version": 1,
                "observation": observation,
                "attempt": attempt,
                "limits": dict(limits),
            })
        except Exception:
            # Optional observation must not turn success into a failed request.
            # Active-call cancellation propagates. Terminal flush runs in an
            # owned child so an internally cancelled hook cannot impersonate Stop.
            pass

    async def started(self, limits: dict[str, Any] | None = None) -> None:
        self.attempt += 1
        if limits is not None:
            self.limits = dict(limits)
        # A physical dispatch does not reset logical-call pacing or discard
        # the last actual activity from a previous attempt.
        await self._emit("attempt_started", self.attempt, self.limits)

    async def activity(self) -> None:
        self.pending = (self.attempt, dict(self.limits))
        now = time.monotonic()
        if self.last_activity is None or now - self.last_activity >= 1.0:
            await self.flush()

    async def flush(self) -> None:
        # Terminal boundary retains the latest ACTUAL observation, even within
        # the throttle window. There is no periodic/silent-wait publication.
        if self.pending is not None:
            attempt, limits = self.pending
            self.pending = None
            self.last_activity = time.monotonic()
            await self._emit("response_activity", attempt, limits)