"""Pure provider-local selectors; no SDK, credentials or inference policy."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from fnmatch import fnmatchcase


_FAMILIES = ("opus", "sonnet", "haiku", "fable")
LIFECYCLES = ("active", "deprecated", "retired")
_DATE = re.compile(r"-(\d{4})-?(\d{2})-?(\d{2})$")


def split_snapshot(model_id: str) -> tuple[str, str]:
    """Separate a real compact or ISO calendar date, never a numeric version."""
    match = _DATE.search(model_id)
    if match:
        try:
            date(*map(int, match.groups()))
        except ValueError:
            pass
        else:
            return model_id[:match.start()], "".join(match.groups())
    return model_id, ""


@dataclass(frozen=True)
class ModelIdentity:
    family: str
    version: tuple[int, ...]
    preview: bool
    snapshot: str
    alias: int

    @property
    def rank(self) -> tuple:
        return self.version, self.alias, self.snapshot


def model_identity(model_id: str) -> ModelIdentity | None:
    """Parse established Claude spellings, without guessing unknown families."""
    base, snapshot = split_snapshot(model_id.casefold())
    alias = 0 if snapshot else 2
    if base.endswith("-latest"):
        base = base[:-7]
        alias = 1
    preview = base.endswith("-preview")
    if preview:
        base = base[:-8]
    families = "|".join((*_FAMILIES, "mythos"))
    match = re.fullmatch(rf"claude-({families})-(\d+(?:-\d{{1,2}})*)", base)
    if match:
        family, version = match.groups()
    else:
        match = re.fullmatch(rf"claude-(\d+(?:-\d{{1,2}})*)-({families})", base)
        if not match:
            return None
        version, family = match.groups()
    parts = tuple(map(int, version.split("-")))
    # 5 and 5-0 describe the same numeric version.
    while len(parts) > 1 and parts[-1] == 0:
        parts = parts[:-1]
    return ModelIdentity(family, parts, preview, snapshot, alias)


def latest_models(ids: list[str], *, previews: bool = True) -> list[str]:
    """Latest stable per supported tier plus its distinct numerically newer preview."""
    groups: dict[tuple[str, bool], tuple[ModelIdentity, str]] = {}
    for model_id in ids:
        identity = model_identity(model_id)
        if identity is None or identity.family not in _FAMILIES:
            continue
        key = identity.family, identity.preview
        previous = groups.get(key)
        if previous is None or (identity.rank, model_id) > (previous[0].rank, previous[1]):
            groups[key] = identity, model_id
    selected = []
    for family in _FAMILIES:
        stable = groups.get((family, False))
        preview = groups.get((family, True))
        if stable:
            selected.append(stable[1])
        if previews and preview and (not stable or preview[0].version > stable[0].version):
            selected.append(preview[1])
    return selected


@dataclass(frozen=True)
class ModelFilters:
    """Immutable policy snapshot. Explicit mappings replace every default."""

    include: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    hide_dated: bool = False
    latest_per_family: bool = False
    lifecycle: tuple[str, ...] = LIFECYCLES

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> "ModelFilters":
        if not isinstance(value, Mapping):
            raise TypeError("model filters must be a mapping")
        if set(value) - {"include", "exclude", "hide_dated", "latest_per_family", "lifecycle"}:
            raise ValueError("unknown model filter key")

        def patterns(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = value.get(key, default)
            if not isinstance(raw, (list, tuple)) or any(
                not isinstance(pattern, str) or not pattern.strip() for pattern in raw
            ):
                raise ValueError(f"model filters {key} must be a list of nonblank strings")
            return tuple(raw)

        for key in ("hide_dated", "latest_per_family"):
            if not isinstance(value.get(key, False), bool):
                raise TypeError(f"model filters {key} must be a boolean")
        lifecycle = patterns("lifecycle", LIFECYCLES)
        if not lifecycle or len(lifecycle) > 3 or len(set(lifecycle)) != len(lifecycle) or any(
            stage not in LIFECYCLES for stage in lifecycle
        ):
            raise ValueError("model filters lifecycle must contain 1..3 distinct known stages")
        return cls(
            patterns("include", ("*",)), patterns("exclude", ()),
            value.get("hide_dated", False), value.get("latest_per_family", False),
            lifecycle,
        )

    def select(self, ids: list[str]) -> list[str]:
        selected = {
            model_id for model_id in ids
            if any(fnmatchcase(model_id.casefold(), pattern.casefold()) for pattern in self.include)
            and not any(fnmatchcase(model_id.casefold(), pattern.casefold()) for pattern in self.exclude)
        }
        if self.hide_dated:
            aliases = {model_id.casefold() for model_id in selected}
            selected = {
                model_id for model_id in selected
                if not (
                    (parts := split_snapshot(model_id.casefold()))[1]
                    and (parts[0] in aliases or parts[0] + "-latest" in aliases)
                )
            }
        if self.latest_per_family:
            selected = set(latest_models(list(selected)))
        return sorted(selected)

    def as_dict(self) -> dict:
        return {
            "include": list(self.include), "exclude": list(self.exclude),
            "hide_dated": self.hide_dated, "latest_per_family": self.latest_per_family,
            "lifecycle": list(self.lifecycle),
        }


DEFAULT_FILTERS = ModelFilters(
    hide_dated=True, latest_per_family=True, lifecycle=("active",)
)


def selector_schema() -> dict:
    """Fresh optional-extension schema; no mutable policy references escape."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "include": {
                "type": "array", "items": {"type": "string", "minLength": 1},
                "default": ["*"],
                "description": "Case-insensitive OS-independent globs; [] includes none.",
            },
            "exclude": {
                "type": "array", "items": {"type": "string", "minLength": 1},
                "default": [],
                "description": "Case-insensitive OS-independent globs; exclusion wins.",
            },
            "hide_dated": {
                "type": "boolean", "default": False,
                "description": "Hide YYYYMMDD/ YYYY-MM-DD snapshots only when a selected matching alias exists.",
            },
            "latest_per_family": {
                "type": "boolean", "default": False,
                "description": "Latest Opus/Sonnet/Haiku/Fable per tier plus a distinct newer preview; unknown spellings excluded.",
            },
            "lifecycle": {
                "type": "array", "items": {"type": "string", "enum": list(LIFECYCLES)},
                "minItems": 1, "maxItems": 3, "uniqueItems": True,
                "default": list(LIFECYCLES),
                "description": "Native service lifecycle filter; {} explicitly requests all stages, including retired.",
            },
        },
    }