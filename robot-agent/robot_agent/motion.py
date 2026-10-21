from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml


@dataclass(frozen=True)
class PoseTarget:
    name: str
    frame_id: str
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class DockTarget:
    name: str
    dock_id: str


@dataclass(frozen=True)
class MotionResult:
    success: bool
    message: str
    data: dict[str, Any] | None = None


class MotionGateway(Protocol):
    @property
    def active(self) -> bool: ...

    def health(self) -> dict[str, Any]: ...

    def navigate(self, target: PoseTarget, timeout_seconds: float) -> MotionResult: ...

    def dock(self, target: DockTarget, timeout_seconds: float) -> MotionResult: ...

    def cancel(self, *, reason: str, emergency: bool = False) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class MotionCatalog:
    version: int
    locations: dict[str, PoseTarget]
    docks: dict[str, DockTarget]

    @classmethod
    def empty(cls) -> "MotionCatalog":
        return cls(version=1, locations={}, docks={})

    @classmethod
    def load(cls, path: str | Path) -> "MotionCatalog":
        catalog_path = Path(path)
        if not catalog_path.exists():
            return cls.empty()
        raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("navigation catalog must be a YAML object")
        version = raw.get("version", 1)
        if not isinstance(version, int) or version < 1:
            raise ValueError("navigation catalog version must be a positive integer")

        locations_raw = raw.get("locations", {})
        docks_raw = raw.get("docks", {})
        if not isinstance(locations_raw, dict) or not isinstance(docks_raw, dict):
            raise ValueError("navigation catalog locations and docks must be objects")

        locations: dict[str, PoseTarget] = {}
        for name, value in locations_raw.items():
            cls._validate_name(name, "location")
            if not isinstance(value, dict):
                raise ValueError(f"location {name!r} must be an object")
            locations[name] = PoseTarget(
                name=name,
                frame_id=cls._required_string(value, "frame_id", name),
                x=cls._finite_float(value, "x", name),
                y=cls._finite_float(value, "y", name),
                yaw=cls._finite_float(value, "yaw", name),
            )

        docks: dict[str, DockTarget] = {}
        for name, value in docks_raw.items():
            cls._validate_name(name, "dock")
            if not isinstance(value, dict):
                raise ValueError(f"dock {name!r} must be an object")
            docks[name] = DockTarget(
                name=name,
                dock_id=cls._required_string(value, "dock_id", name),
            )
        return cls(version=version, locations=locations, docks=docks)

    def public_view(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "locations": sorted(self.locations),
            "docks": sorted(self.docks),
        }

    @staticmethod
    def _validate_name(name: Any, kind: str) -> None:
        if not isinstance(name, str) or not name or not all(
            character.isalnum() or character in {"_", "-"} for character in name
        ):
            raise ValueError(f"{kind} names may contain only letters, digits, '_' and '-'")

    @staticmethod
    def _required_string(value: dict[str, Any], field: str, name: str) -> str:
        result = value.get(field)
        if not isinstance(result, str) or not result.strip():
            raise ValueError(f"catalog target {name!r} requires non-empty {field}")
        return result.strip()

    @staticmethod
    def _finite_float(value: dict[str, Any], field: str, name: str) -> float:
        import math

        result = value.get(field)
        if isinstance(result, bool) or not isinstance(result, (int, float)):
            raise ValueError(f"catalog target {name!r} requires numeric {field}")
        result = float(result)
        if not math.isfinite(result):
            raise ValueError(f"catalog target {name!r} has non-finite {field}")
        return result


class UnavailableMotionGateway:
    def __init__(self, reason: str = "motion backend is disabled"):
        self.reason = reason

    @property
    def active(self) -> bool:
        return False

    def health(self) -> dict[str, Any]:
        return {"available": False, "backend": "unavailable", "reason": self.reason}

    def navigate(self, target: PoseTarget, timeout_seconds: float) -> MotionResult:
        return MotionResult(False, self.reason, {"target": target.name})

    def dock(self, target: DockTarget, timeout_seconds: float) -> MotionResult:
        return MotionResult(False, self.reason, {"target": target.name})

    def cancel(self, *, reason: str, emergency: bool = False) -> None:
        return None

    def close(self) -> None:
        return None


__all__ = [
    "DockTarget",
    "MotionCatalog",
    "MotionGateway",
    "MotionResult",
    "PoseTarget",
    "UnavailableMotionGateway",
]
