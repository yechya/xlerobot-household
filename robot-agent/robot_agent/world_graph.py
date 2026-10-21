from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class WorldGraphError(ValueError):
    pass


ENTITY_TYPES = frozenset({"room", "surface", "receptacle", "zone"})
EVENT_TYPES = frozenset(
    {
        "checkpoint",
        "entity_upsert",
        "entity_remove",
        "object_upsert",
        "object_remove",
        "observation_fusion",
        "semantic_map_sync",
        "robot_update",
        "task_result",
    }
)
ZERO_HASH = "0" * 64
_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SpatialEntity:
    entity_id: str
    label: str
    entity_type: str
    parent_id: str | None = None
    frame_id: str = "map"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    confidence: float = 1.0
    last_seen: float = field(default_factory=time.time)
    source: str = "manual"
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entity_id or not self.label or self.entity_type not in ENTITY_TYPES:
            raise WorldGraphError("spatial entity identity or type is invalid")
        if not self.frame_id or not self.source:
            raise WorldGraphError("spatial entity frame and source must be non-empty")
        if not all(math.isfinite(value) for value in (self.x, self.y, self.z, self.last_seen)):
            raise WorldGraphError("spatial entity values must be finite")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise WorldGraphError("spatial entity confidence must be in [0, 1]")
        object.__setattr__(
            self,
            "properties",
            _normalized_json_object(self.properties, "spatial entity properties"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "entity_type": self.entity_type,
            "parent_id": self.parent_id,
            "frame_id": self.frame_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "confidence": self.confidence,
            "last_seen": self.last_seen,
            "source": self.source,
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SpatialEntity:
        return cls(
            entity_id=str(raw["entity_id"]),
            label=str(raw["label"]),
            entity_type=str(raw["entity_type"]),
            parent_id=(str(raw["parent_id"]) if raw.get("parent_id") is not None else None),
            frame_id=str(raw.get("frame_id", "map")),
            x=float(raw.get("x", 0.0)),
            y=float(raw.get("y", 0.0)),
            z=float(raw.get("z", 0.0)),
            confidence=float(raw.get("confidence", 1.0)),
            last_seen=float(raw.get("last_seen", time.time())),
            source=str(raw.get("source", "legacy")),
            properties=raw.get("properties", {}),
        )


@dataclass(frozen=True)
class WorldEvent:
    sequence: int
    event_id: str
    recorded_at: float
    event_type: str
    source: str
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        event_type: str,
        source: str,
        payload: dict[str, Any],
        previous_hash: str,
        recorded_at: float | None = None,
        event_id: str | None = None,
    ) -> WorldEvent:
        event = cls(
            sequence=sequence,
            event_id=event_id or f"world-{uuid.uuid4().hex}",
            recorded_at=time.time() if recorded_at is None else recorded_at,
            event_type=event_type,
            source=source,
            payload=_normalized_json_object(
                payload,
                "world event payload",
                max_bytes=256 * 1024,
            ),
            previous_hash=previous_hash,
            event_hash="",
        )
        event.validate(include_hash=False)
        return cls(**{**event.__dict__, "event_hash": event.calculate_hash()})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WorldEvent:
        if set(raw) != {
            "sequence",
            "event_id",
            "recorded_at",
            "event_type",
            "source",
            "payload",
            "previous_hash",
            "event_hash",
        }:
            raise WorldGraphError("world event has missing or unexpected fields")
        event = cls(
            sequence=raw["sequence"],
            event_id=raw["event_id"],
            recorded_at=raw["recorded_at"],
            event_type=raw["event_type"],
            source=raw["source"],
            payload=_normalized_json_object(
                raw["payload"],
                "world event payload",
                max_bytes=256 * 1024,
            ),
            previous_hash=raw["previous_hash"],
            event_hash=raw["event_hash"],
        )
        event.validate(include_hash=True)
        return event

    def validate(self, *, include_hash: bool) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            raise WorldGraphError("world event sequence must be positive")
        if not isinstance(self.event_id, str) or not self.event_id or len(self.event_id) > 128:
            raise WorldGraphError("world event ID is invalid")
        if (
            isinstance(self.recorded_at, bool)
            or not isinstance(self.recorded_at, (int, float))
            or not math.isfinite(self.recorded_at)
            or self.recorded_at <= 0
        ):
            raise WorldGraphError("world event timestamp is invalid")
        if self.event_type not in EVENT_TYPES:
            raise WorldGraphError(f"unsupported world event type: {self.event_type!r}")
        if not isinstance(self.source, str) or not self.source.strip() or len(self.source) > 256:
            raise WorldGraphError("world event source is invalid")
        _normalized_json_object(
            self.payload,
            "world event payload",
            max_bytes=256 * 1024,
        )
        if not isinstance(self.previous_hash, str) or not _HASH.fullmatch(self.previous_hash):
            raise WorldGraphError("world event previous hash is invalid")
        if include_hash:
            if not isinstance(self.event_hash, str) or not _HASH.fullmatch(self.event_hash):
                raise WorldGraphError("world event hash is invalid")
            if self.event_hash != self.calculate_hash():
                raise WorldGraphError("world event hash does not match content")

    def calculate_hash(self) -> str:
        encoded = json.dumps(
            self._hash_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "recorded_at": self.recorded_at,
            "event_type": self.event_type,
            "source": self.source,
            "payload": _normalized_json_object(
                self.payload,
                "world event payload",
                max_bytes=256 * 1024,
            ),
            "previous_hash": self.previous_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "event_hash": self.event_hash}


class WorldEventJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[WorldEvent]:
        if not self.path.exists():
            return []
        events: list[WorldEvent] = []
        previous_hash = ZERO_HASH
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    event = WorldEvent.from_dict(raw)
                except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                    raise WorldGraphError(
                        f"invalid world event at line {line_number}: {exc}"
                    ) from exc
                expected_sequence = len(events) + 1
                if event.sequence != expected_sequence or event.previous_hash != previous_hash:
                    raise WorldGraphError(
                        f"world event chain is broken at sequence {event.sequence}"
                    )
                events.append(event)
                previous_hash = event.event_hash
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return events

    def append(self, event: WorldEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            self.path,
            os.O_RDWR | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                current = self._last_event(handle)
                expected_sequence = 1 if current is None else current.sequence + 1
                expected_hash = ZERO_HASH if current is None else current.event_hash
                if event.sequence != expected_sequence or event.previous_hash != expected_hash:
                    raise WorldGraphError(
                        "world event journal changed concurrently; reload before writing"
                    )
                handle.seek(0, os.SEEK_END)
                handle.write(
                    json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o600)
        except Exception:
            if not self.path.exists():
                raise
            self.path.chmod(0o600)
            raise

    @staticmethod
    def _last_event(handle: Any) -> WorldEvent | None:
        handle.seek(0)
        last = ""
        for line in handle:
            if line.strip():
                last = line
        if not last:
            return None
        try:
            return WorldEvent.from_dict(json.loads(last))
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            raise WorldGraphError("world event journal has an invalid final record") from exc


def validate_event_chain(events: Iterable[WorldEvent]) -> tuple[WorldEvent, ...]:
    values = tuple(events)
    previous_hash = ZERO_HASH
    for expected_sequence, event in enumerate(values, start=1):
        event.validate(include_hash=True)
        if event.sequence != expected_sequence or event.previous_hash != previous_hash:
            raise WorldGraphError("world event chain is not contiguous")
        previous_hash = event.event_hash
    return values


def _normalized_json_object(
    value: Any,
    label: str,
    *,
    max_bytes: int = 64 * 1024,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorldGraphError(f"{label} must be an object")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise WorldGraphError(f"{label} must be JSON-safe") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise WorldGraphError(f"{label} is too large")
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise WorldGraphError(f"{label} must be an object")
    return normalized


__all__ = [
    "ENTITY_TYPES",
    "EVENT_TYPES",
    "SpatialEntity",
    "WorldEvent",
    "WorldEventJournal",
    "WorldGraphError",
    "ZERO_HASH",
    "validate_event_chain",
]
