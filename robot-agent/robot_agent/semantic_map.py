from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .world_graph import SpatialEntity, WorldGraphError

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class SemanticMapError(ValueError):
    pass


@dataclass(frozen=True)
class SemanticMapSpec:
    map_id: str
    planning_frame: str
    surveyed_at: float
    entities: tuple[SpatialEntity, ...]
    fingerprint: str

    @property
    def source(self) -> str:
        return f"semantic_map:{self.map_id}"

    @classmethod
    def load(cls, path: str | Path) -> SemanticMapSpec:
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        expected = {
            "version",
            "approved",
            "map_id",
            "planning_frame",
            "surveyed_at",
            "entities",
        }
        if not isinstance(raw, dict) or set(raw) != expected or raw.get("version") != 1:
            raise SemanticMapError("semantic map has missing or unexpected fields")
        if raw["approved"] is not True:
            raise SemanticMapError("semantic map must be approved")
        map_id = _identifier(raw["map_id"], "map_id")
        planning_frame = _identifier(raw["planning_frame"], "planning_frame")
        surveyed_at = _positive_float(raw["surveyed_at"], "surveyed_at")
        entities_raw = raw["entities"]
        if not isinstance(entities_raw, list) or not 1 <= len(entities_raw) <= 512:
            raise SemanticMapError("semantic map must contain 1..512 entities")
        source = f"semantic_map:{map_id}"
        entities = tuple(
            _parse_entity(value, index, planning_frame, surveyed_at, source)
            for index, value in enumerate(entities_raw)
        )
        entity_ids = [entity.entity_id for entity in entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise SemanticMapError("semantic map entity IDs must be unique")
        payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
        spec = cls(
            map_id=map_id,
            planning_frame=planning_frame,
            surveyed_at=surveyed_at,
            entities=entities,
            fingerprint=hashlib.sha256(payload).hexdigest(),
        )
        try:
            from .world_model import WorldModel

            WorldModel().apply_semantic_map_sync(
                spec.entities,
                map_id=spec.map_id,
                map_fingerprint=spec.fingerprint,
                surveyed_at=spec.surveyed_at,
                planning_frame=spec.planning_frame,
                source=spec.source,
            )
        except WorldGraphError as exc:
            raise SemanticMapError(str(exc)) from exc
        return spec


def sync_semantic_map(model: Any, spec: SemanticMapSpec) -> dict[str, Any]:
    try:
        event = model.apply_semantic_map_sync(
            spec.entities,
            map_id=spec.map_id,
            map_fingerprint=spec.fingerprint,
            surveyed_at=spec.surveyed_at,
            planning_frame=spec.planning_frame,
            source=spec.source,
        )
    except WorldGraphError as exc:
        raise SemanticMapError(str(exc)) from exc
    return {
        "available": True,
        "map_id": spec.map_id,
        "planning_frame": spec.planning_frame,
        "surveyed_at": spec.surveyed_at,
        "fingerprint": spec.fingerprint,
        "entity_count": len(spec.entities),
        "revision": model.revision,
        "event_hash": event.event_hash if event is not None else None,
        "changed": event is not None,
    }


def _parse_entity(
    raw: Any,
    index: int,
    planning_frame: str,
    surveyed_at: float,
    source: str,
) -> SpatialEntity:
    expected = {
        "entity_id",
        "label",
        "entity_type",
        "parent_id",
        "position",
        "confidence",
        "properties",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise SemanticMapError(f"semantic map entity {index} has invalid fields")
    position = raw["position"]
    if not isinstance(position, list) or len(position) != 3:
        raise SemanticMapError(f"semantic map entity {index} position must have 3 values")
    try:
        return SpatialEntity(
            entity_id=_identifier(raw["entity_id"], f"entities[{index}].entity_id"),
            label=_text(raw["label"], f"entities[{index}].label", 256),
            entity_type=_text(raw["entity_type"], f"entities[{index}].entity_type", 32),
            parent_id=(
                _identifier(raw["parent_id"], f"entities[{index}].parent_id")
                if raw["parent_id"] is not None
                else None
            ),
            frame_id=planning_frame,
            x=_finite_float(position[0], f"entities[{index}].position[0]"),
            y=_finite_float(position[1], f"entities[{index}].position[1]"),
            z=_finite_float(position[2], f"entities[{index}].position[2]"),
            confidence=_bounded_float(
                raw["confidence"], f"entities[{index}].confidence", 0.0, 1.0
            ),
            last_seen=surveyed_at,
            source=source,
            properties=raw["properties"],
        )
    except WorldGraphError as exc:
        raise SemanticMapError(f"semantic map entity {index} is invalid: {exc}") from exc


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise SemanticMapError(f"{label} is invalid")
    return value


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise SemanticMapError(f"{label} is invalid")
    return value.strip()


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticMapError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SemanticMapError(f"{label} must be finite")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0:
        raise SemanticMapError(f"{label} must be positive")
    return result


def _bounded_float(value: Any, label: str, minimum: float, maximum: float) -> float:
    result = _finite_float(value, label)
    if not minimum <= result <= maximum:
        raise SemanticMapError(f"{label} must be in [{minimum}, {maximum}]")
    return result


__all__ = ["SemanticMapError", "SemanticMapSpec", "sync_semantic_map"]
