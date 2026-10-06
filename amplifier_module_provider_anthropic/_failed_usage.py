"""Usage-only measurements from consumed SDK events, never failed content."""

from typing import Any

from ._cost import compute_cost


_COUNTERS = (
    "input_tokens", "output_tokens", "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


class FailedUsage:
    """One physical attempt. Missing/defaulted fields are not measured zero."""

    def __init__(self, model: str):
        self.model = model
        self.values: dict[str, Any] = {}
        self.complete = False
        self.invalid = False
        self.cost_callback_state = "not_invoked"

    def contribute(self, callback, cost) -> None:
        """Ownership starts at invocation, not conversion entry.

        A raising callback may already have committed externally. Never redeliver
        it automatically; its contribution outcome remains explicitly unknown.
        """
        if cost is None or self.cost_callback_state != "not_invoked":
            return
        self.cost_callback_state = "unknown"
        callback(cost)
        self.cost_callback_state = "returned"

    def capture(self, usage: Any, *, complete: bool = False) -> None:
        # exclude_unset is essential: SDK model defaults are not wire evidence.
        dump = getattr(usage, "model_dump", None)
        if not callable(dump):
            return
        raw = dump(exclude_unset=True)
        if not isinstance(raw, dict):
            return
        self.complete = self.complete or complete
        counters = {key: raw[key] for key in _COUNTERS if key in raw}
        if "cache_creation" in raw and raw["cache_creation"] is not None:
            split = raw["cache_creation"]
            if isinstance(split, dict):
                for ttl in ("5m", "1h"):
                    key = f"ephemeral_{ttl}_input_tokens"
                    if key in split:
                        counters[f"cache_creation_{ttl}_input_tokens"] = split[key]
            else:
                self.invalid = True
        for key, value in counters.items():
            if type(value) is int and 0 <= value <= 2**63 - 1:
                self.values[key] = value
            else:
                self.values.pop(key, None)
                self.invalid = True
        for key, allowed in (
            ("speed", ("standard", "fast")),
            ("service_tier", ("standard",)),
        ):
            if key in raw:
                if raw[key] in allowed:
                    self.values[key] = raw[key]
                else:
                    self.values.pop(key, None)
                    self.invalid = True

    def settlement(self) -> tuple[dict[str, Any], Any]:
        values = self.values
        cost = None
        # A partial stream counter is a snapshot, not a complete bill. The
        # existing calculator cannot price missing tokens or unknown tiers.
        if (
            self.complete and not self.invalid
            and all(key in values for key in _COUNTERS)
            and values.get("service_tier") == "standard"
            and values.get("speed") in ("standard", "fast")
        ):
            five = values.get("cache_creation_5m_input_tokens")
            hour = values.get("cache_creation_1h_input_tokens")
            writes = values["cache_creation_input_tokens"]
            split_known = five is not None and hour is not None and five + hour == writes
            if split_known or (writes == 0 and five is None and hour is None):
                cost = compute_cost(
                    self.model, **{key: values[key] for key in _COUNTERS},
                    cache_creation_5m_input_tokens=five,
                    cache_creation_1h_input_tokens=hour,
                    speed=values["speed"],
                )
        return {
            **values,
            "input_tokens": values.get("input_tokens"),
            "output_tokens": values.get("output_tokens"),
            "usage_complete": self.complete,
            "cost_usd": str(cost) if cost is not None else None,
            "cost_callback_state": self.cost_callback_state,
        }, cost