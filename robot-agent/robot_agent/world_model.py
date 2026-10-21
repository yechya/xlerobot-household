from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .world_graph import (
    ZERO_HASH,
    SpatialEntity,
    WorldEvent,
    WorldEventJournal,
    WorldGraphError,
    validate_event_chain,
)

logger = logging.getLogger("world_model")

_RE_GRIPPER_HOLDING = re.compile(r"^gripper_holding\(([^)]+)\)$")
_RE_OBJECT_VISIBLE = re.compile(r"^object_visible\(([^)]+)\)$")
_RE_OBJECT_REACHABLE = re.compile(r"^object_reachable\(([^)]+)\)$")
_RE_OBJECT_NEAR = re.compile(r"^object_near\(([^)]+),([^)]+)\)$")
_RE_OBJECT_ON = re.compile(r"^object_on\(([^)]+),([^)]+)\)$")
_RE_OBJECT_IN = re.compile(r"^object_in\(([^)]+),([^)]+)\)$")
_RE_ROBOT_IN = re.compile(r"^robot_in\(([^)]+)\)$")

_VISIBLE_CONFIDENCE_THRESHOLD = 0.5
_REACHABLE_DISTANCE_THRESHOLD = 0.35
_SPATIAL_NEAR_THRESHOLD = 0.05
_CONFIDENCE_HALF_LIFE_SECONDS = 60.0
_CONFIDENCE_MIN_THRESHOLD = 0.1


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    label: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    confidence: float = 1.0
    state: str = "on_table"
    last_seen: float = field(default_factory=time.time)
    properties: dict[str, Any] = field(default_factory=dict)
    frame_id: str = "base_link"
    room_id: str | None = None
    parent_id: str | None = None
    source: str = "manual"
    observation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.object_id or not self.label or not self.frame_id or not self.source:
            raise WorldGraphError("object identity, frame and source must be non-empty")
        values = (self.x, self.y, self.z, self.confidence, self.last_seen)
        if not all(math.isfinite(value) for value in values):
            raise WorldGraphError("object values must be finite")
        if not 0 <= self.confidence <= 1:
            raise WorldGraphError("object confidence must be in [0, 1]")
        object.__setattr__(
            self,
            "properties",
            _normalized_json(self.properties, "object properties"),
        )

    def distance_from_origin(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "confidence": self.confidence,
            "state": self.state,
            "last_seen": self.last_seen,
            "properties": dict(self.properties),
            "frame_id": self.frame_id,
            "room_id": self.room_id,
            "parent_id": self.parent_id,
            "source": self.source,
            "observation_id": self.observation_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ObjectState:
        return cls(
            object_id=str(raw["object_id"]),
            label=str(raw["label"]),
            x=float(raw.get("x", 0.0)),
            y=float(raw.get("y", 0.0)),
            z=float(raw.get("z", 0.0)),
            confidence=float(raw.get("confidence", 1.0)),
            state=str(raw.get("state", "unknown")),
            last_seen=float(raw.get("last_seen", time.time())),
            properties=dict(raw.get("properties", {})),
            frame_id=str(raw.get("frame_id", "base_link")),
            room_id=(str(raw["room_id"]) if raw.get("room_id") is not None else None),
            parent_id=(str(raw["parent_id"]) if raw.get("parent_id") is not None else None),
            source=str(raw.get("source", "legacy")),
            observation_id=(
                str(raw["observation_id"])
                if raw.get("observation_id") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class RobotState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    gripper_state: str = "empty"
    gripper_holding: str | None = None
    joint_positions: list[float] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    frame_id: str = "map"
    room_id: str | None = None
    source: str = "robot_state"

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.theta, self.last_updated, *self.joint_positions)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise WorldGraphError("robot state values must be finite")
        if self.gripper_state not in {"empty", "holding", "unknown"}:
            raise WorldGraphError("robot gripper state is invalid")
        if not self.frame_id or not self.source:
            raise WorldGraphError("robot frame and source must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "gripper_state": self.gripper_state,
            "gripper_holding": self.gripper_holding,
            "joint_positions": list(self.joint_positions),
            "last_updated": self.last_updated,
            "frame_id": self.frame_id,
            "room_id": self.room_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RobotState:
        return cls(
            x=float(raw.get("x", 0.0)),
            y=float(raw.get("y", 0.0)),
            theta=float(raw.get("theta", 0.0)),
            gripper_state=str(raw.get("gripper_state", "empty")),
            gripper_holding=raw.get("gripper_holding"),
            joint_positions=[float(value) for value in raw.get("joint_positions", [])],
            last_updated=float(raw.get("last_updated", time.time())),
            frame_id=str(raw.get("frame_id", "map")),
            room_id=(str(raw["room_id"]) if raw.get("room_id") is not None else None),
            source=str(raw.get("source", "legacy")),
        )


@dataclass(frozen=True)
class WorldDiff:
    timestamp: float
    skill_name: str
    pre_state: dict[str, Any]
    post_state: dict[str, Any]
    success: bool
    message: str = ""


class WorldModel:
    """Event-backed instance graph for household places, objects and robot state."""

    def __init__(self, persistence_path: str | Path | None = None):
        self._entities: dict[str, SpatialEntity] = {}
        self._objects: dict[str, ObjectState] = {}
        self._robot_state = RobotState()
        self._diff_history: list[WorldDiff] = []
        self._events: list[WorldEvent] = []
        self._persistence_path = Path(persistence_path) if persistence_path else None
        self._journal = (
            WorldEventJournal(self._event_path(self._persistence_path))
            if self._persistence_path
            else None
        )
        self._load()

    @staticmethod
    def _event_path(snapshot_path: Path) -> Path:
        return snapshot_path.with_suffix(snapshot_path.suffix + ".events.jsonl")

    @classmethod
    def from_events(cls, events: Iterable[WorldEvent]) -> WorldModel:
        model = cls()
        values = validate_event_chain(events)
        for event in values:
            model._apply_event(event)
            model._events.append(event)
        model._validate_graph()
        return model

    def add_entity(self, entity: SpatialEntity, *, source: str | None = None) -> None:
        if source is not None:
            entity = replace(entity, source=source)
        self._validate_entity_relation(entity)
        self._commit(
            "entity_upsert",
            source or entity.source,
            {"entity": entity.to_dict()},
        )

    def update_entity(
        self,
        entity_id: str,
        *,
        source: str = "entity_update",
        **kwargs: Any,
    ) -> None:
        old = self._entities.get(entity_id)
        if old is None:
            raise WorldGraphError(f"spatial entity does not exist: {entity_id}")
        values = {**old.to_dict(), **kwargs, "entity_id": entity_id, "source": source}
        entity = SpatialEntity.from_dict(values)
        self._validate_entity_relation(entity)
        self._commit("entity_upsert", source, {"entity": entity.to_dict()})

    def remove_entity(self, entity_id: str, *, source: str = "entity_remove") -> None:
        if entity_id not in self._entities:
            return
        if any(value.parent_id == entity_id for value in self._entities.values()):
            raise WorldGraphError("cannot remove a spatial entity with child entities")
        if any(
            value.parent_id == entity_id or value.room_id == entity_id
            for value in self._objects.values()
        ):
            raise WorldGraphError("cannot remove a spatial entity containing objects")
        if self._robot_state.room_id == entity_id:
            raise WorldGraphError("cannot remove the robot's current room")
        self._commit("entity_remove", source, {"entity_id": entity_id})

    def get_entity(self, entity_id: str) -> SpatialEntity | None:
        entity = self._entities.get(entity_id)
        return SpatialEntity.from_dict(entity.to_dict()) if entity else None

    def get_entities(self, entity_type: str | None = None) -> list[SpatialEntity]:
        values = [SpatialEntity.from_dict(value.to_dict()) for value in self._entities.values()]
        if entity_type is None:
            return values
        return [value for value in values if value.entity_type == entity_type]

    def add_object(self, obj: ObjectState, *, source: str | None = None) -> None:
        if source is not None:
            obj = replace(obj, source=source)
        obj = self._normalize_object_relation(obj)
        self._validate_object_relation(obj)
        self._commit("object_upsert", source or obj.source, {"object": obj.to_dict()})

    def update_object(
        self,
        object_id: str,
        *,
        source: str = "object_update",
        **kwargs: Any,
    ) -> None:
        old = self._objects.get(object_id)
        if old is None:
            logger.warning("[WorldModel] object does not exist: %s", object_id)
            return
        values = {**old.to_dict(), **kwargs, "object_id": object_id, "source": source}
        obj = self._normalize_object_relation(ObjectState.from_dict(values))
        self._validate_object_relation(obj)
        self._commit("object_upsert", source, {"object": obj.to_dict()})

    def remove_object(self, object_id: str, *, source: str = "object_remove") -> None:
        if object_id in self._objects:
            self._commit("object_remove", source, {"object_id": object_id})

    def get_object(self, object_id: str) -> ObjectState | None:
        obj = self._objects.get(object_id)
        return ObjectState.from_dict(obj.to_dict()) if obj else None

    def get_objects_by_label(self, label: str) -> list[ObjectState]:
        target = label.lower()
        return [
            ObjectState.from_dict(obj.to_dict())
            for obj in self._objects.values()
            if target in obj.label.lower()
        ]

    def get_all_objects(self) -> list[ObjectState]:
        return [ObjectState.from_dict(obj.to_dict()) for obj in self._objects.values()]

    def update_robot_state(self, *, source: str = "robot_state", **kwargs: Any) -> None:
        values = {
            **self._robot_state.to_dict(),
            **kwargs,
            "source": source,
            "last_updated": kwargs.get("last_updated", time.time()),
        }
        state = RobotState.from_dict(values)
        if state.room_id is not None:
            room = self._entities.get(state.room_id)
            if room is None or room.entity_type != "room":
                raise WorldGraphError("robot room_id must reference a room entity")
        self._commit("robot_update", source, {"robot_state": state.to_dict()})

    def get_robot_state(self) -> RobotState:
        return RobotState.from_dict(self._robot_state.to_dict())

    def gripper_grasp(self, object_id: str) -> None:
        self.update_robot_state(
            source="gripper",
            gripper_state="holding",
            gripper_holding=object_id,
        )
        if object_id in self._objects:
            self.update_object(object_id, source="gripper", state="grasped", parent_id=None)

    def gripper_release(self) -> None:
        holding = self._robot_state.gripper_holding
        self.update_robot_state(
            source="gripper",
            gripper_state="empty",
            gripper_holding=None,
        )
        if holding and holding in self._objects:
            self.update_object(holding, source="gripper", state="placed")

    def check_precondition(self, predicate: str) -> bool:
        pred = predicate.strip().lower()
        if pred == "gripper_empty":
            return self._robot_state.gripper_state == "empty"
        match = _RE_GRIPPER_HOLDING.match(pred)
        if match:
            return (
                self._robot_state.gripper_state == "holding"
                and self._robot_state.gripper_holding == match.group(1)
            )
        match = _RE_OBJECT_VISIBLE.match(pred)
        if match:
            objects = self.get_objects_by_label(match.group(1))
            return bool(objects) and max(objects, key=lambda obj: obj.last_seen).confidence >= 0.5
        match = _RE_OBJECT_REACHABLE.match(pred)
        if match:
            objects = self.get_objects_by_label(match.group(1))
            if not objects:
                return False
            latest = max(objects, key=lambda obj: obj.last_seen)
            return (
                latest.frame_id in {"base_link", "base_footprint"}
                and latest.distance_from_origin() <= _REACHABLE_DISTANCE_THRESHOLD
            )
        match = _RE_OBJECT_ON.match(pred)
        if match:
            return self._object_has_parent(match.group(1), match.group(2), "surface")
        match = _RE_OBJECT_IN.match(pred)
        if match:
            return self._object_has_parent(match.group(1), match.group(2), "receptacle")
        match = _RE_OBJECT_NEAR.match(pred)
        if match:
            left = self.get_objects_by_label(match.group(1))
            right = self.get_objects_by_label(match.group(2))
            if not left or not right or left[0].frame_id != right[0].frame_id:
                return False
            distance = math.sqrt((left[0].x - right[0].x) ** 2 + (left[0].y - right[0].y) ** 2)
            return distance <= _SPATIAL_NEAR_THRESHOLD
        match = _RE_ROBOT_IN.match(pred)
        if match:
            return self._entity_matches(self._robot_state.room_id, match.group(1), "room")
        logger.warning("[WorldModel] unknown predicate: %s", predicate)
        return False

    def check_preconditions(self, preconditions: list[str]) -> tuple[bool, str | None]:
        for predicate in preconditions:
            if not self.check_precondition(predicate):
                return False, predicate
        return True, None

    def apply_effects(self, effects: dict[str, str]) -> None:
        position = effects.get("robot_position")
        if position:
            x, y, theta = position.split(",")
            self.update_robot_state(x=float(x), y=float(y), theta=float(theta))

    def update_from_vlm(
        self,
        vlm_response: str,
        detections: list[dict[str, Any]] | None = None,
        *,
        source: str = "vlm_detection",
        captured_at: float | None = None,
        observation_id: str | None = None,
    ) -> None:
        if not detections:
            logger.debug("[WorldModel] ignored unstructured VLM text: %s", vlm_response[:100])
            return
        observed_at = time.time() if captured_at is None else captured_at
        batch_id = observation_id or f"obs-{uuid.uuid4().hex}"
        for index, detection in enumerate(detections):
            label = str(detection["label"])
            instance = detection.get("instance_id", detection.get("id", index))
            object_id = str(detection.get("object_id", f"{label}_{instance}"))
            obj = ObjectState(
                object_id=object_id,
                label=label,
                x=float(detection.get("x", 0.0)),
                y=float(detection.get("y", 0.0)),
                z=float(detection.get("z", 0.0)),
                confidence=float(detection.get("confidence", 1.0)),
                state=str(detection.get("state", "on_table")),
                last_seen=observed_at,
                properties=dict(detection.get("properties", {})),
                frame_id=str(detection.get("frame_id", "base_link")),
                room_id=detection.get("room_id"),
                parent_id=detection.get("parent_id"),
                source=source,
                observation_id=batch_id,
            )
            self.add_object(obj, source=source)

    def apply_observation_fusion(
        self,
        objects: Iterable[ObjectState],
        *,
        observation_id: str,
        captured_at: float,
        fusion_fingerprint: str,
        source: str,
        ignored_instance_ids: Iterable[str] = (),
    ) -> WorldEvent:
        values = tuple(objects)
        if not 1 <= len(values) <= 128:
            raise WorldGraphError("observation fusion must contain 1..128 objects")
        if not observation_id or len(observation_id) > 128:
            raise WorldGraphError("observation fusion ID is invalid")
        if not math.isfinite(captured_at) or captured_at <= 0:
            raise WorldGraphError("observation fusion timestamp is invalid")
        if len(fusion_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fusion_fingerprint
        ):
            raise WorldGraphError("observation fusion fingerprint must be SHA-256")
        if any(
            event.event_type == "observation_fusion"
            and event.payload.get("observation_id") == observation_id
            for event in self._events
        ):
            raise WorldGraphError("observation fusion ID has already been committed")
        object_ids = [obj.object_id for obj in values]
        if len(object_ids) != len(set(object_ids)):
            raise WorldGraphError("observation fusion object IDs must be unique")
        normalized: list[ObjectState] = []
        for obj in values:
            if obj.observation_id != observation_id:
                raise WorldGraphError("fused object observation ID does not match batch")
            value = self._normalize_object_relation(obj)
            self._validate_object_relation(value)
            normalized.append(value)
        ignored = tuple(str(value) for value in ignored_instance_ids)
        if len(ignored) > 128 or any(not value or len(value) > 128 for value in ignored):
            raise WorldGraphError("ignored fusion instance IDs are invalid")
        self._commit(
            "observation_fusion",
            source,
            {
                "observation_id": observation_id,
                "captured_at": captured_at,
                "fusion_fingerprint": fusion_fingerprint,
                "objects": [obj.to_dict() for obj in normalized],
                "ignored_instance_ids": list(ignored),
            },
        )
        return WorldEvent.from_dict(self._events[-1].to_dict())

    def apply_semantic_map_sync(
        self,
        entities: Iterable[SpatialEntity],
        *,
        map_id: str,
        map_fingerprint: str,
        surveyed_at: float,
        planning_frame: str,
        source: str,
    ) -> WorldEvent | None:
        values = tuple(entities)
        if not 1 <= len(values) <= 512:
            raise WorldGraphError("semantic map sync must contain 1..512 entities")
        if not map_id or len(map_id) > 128 or source != f"semantic_map:{map_id}":
            raise WorldGraphError("semantic map identity or source is invalid")
        if not planning_frame or len(planning_frame) > 128:
            raise WorldGraphError("semantic map planning frame is invalid")
        if not math.isfinite(surveyed_at) or surveyed_at <= 0:
            raise WorldGraphError("semantic map surveyed_at is invalid")
        if len(map_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in map_fingerprint
        ):
            raise WorldGraphError("semantic map fingerprint must be SHA-256")
        if any(
            event.event_type == "semantic_map_sync"
            and event.payload.get("map_id") == map_id
            and event.payload.get("map_fingerprint") == map_fingerprint
            for event in self._events
        ):
            return None
        entity_ids = [entity.entity_id for entity in values]
        if len(entity_ids) != len(set(entity_ids)):
            raise WorldGraphError("semantic map entity IDs must be unique")
        merged = dict(self._entities)
        for entity in values:
            if entity.source != source or entity.frame_id != planning_frame:
                raise WorldGraphError("semantic map entity provenance does not match map")
            current = merged.get(entity.entity_id)
            if current is not None and current.source != source:
                raise WorldGraphError(
                    f"semantic map cannot overwrite non-map entity: {entity.entity_id}"
                )
            merged[entity.entity_id] = entity
        candidate = WorldModel()
        candidate._entities = merged
        candidate._objects = dict(self._objects)
        candidate._robot_state = self._robot_state
        candidate._validate_graph()
        self._commit(
            "semantic_map_sync",
            source,
            {
                "map_id": map_id,
                "map_fingerprint": map_fingerprint,
                "surveyed_at": surveyed_at,
                "planning_frame": planning_frame,
                "entities": [entity.to_dict() for entity in values],
            },
        )
        return WorldEvent.from_dict(self._events[-1].to_dict())

    def record_diff(
        self,
        skill_name: str,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
        success: bool,
        message: str = "",
    ) -> None:
        payload = {
            "skill_name": skill_name,
            "pre_state": dict(pre_state),
            "post_state": dict(post_state),
            "success": bool(success),
            "message": message,
        }
        self._commit("task_result", f"task:{skill_name}", payload)

    def get_diff_history(self) -> list[WorldDiff]:
        return list(self._diff_history)

    def get_events(self, *, since_sequence: int = 0) -> list[WorldEvent]:
        return [
            WorldEvent.from_dict(event.to_dict())
            for event in self._events
            if event.sequence > since_sequence
        ]

    def reset(self, *, source: str = "operator_reset") -> None:
        self._commit(
            "checkpoint",
            source,
            {
                "entities": {},
                "objects": {},
                "robot_state": RobotState(source=source).to_dict(),
            },
        )

    @property
    def revision(self) -> int:
        return len(self._events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "revision": self.revision,
            "last_event_hash": self._events[-1].event_hash if self._events else ZERO_HASH,
            "entities": [entity.to_dict() for entity in self._entities.values()],
            "objects": [obj.to_dict() for obj in self._objects.values()],
            "robot_state": self._robot_state.to_dict(),
            "diff_history_count": len(self._diff_history),
        }

    def summary(self) -> str:
        lines = ["=== 世界状态 ==="]
        room = self._robot_state.room_id or "unknown"
        lines.append(f"机器人: room={room} gripper={self._robot_state.gripper_state}")
        if self._robot_state.gripper_holding:
            lines.append(f"  抓着: {self._robot_state.gripper_holding}")
        lines.append(f"空间实体 ({len(self._entities)}), 物品 ({len(self._objects)}):")
        for obj in sorted(self._objects.values(), key=lambda value: value.label):
            parent = obj.parent_id or "unassigned"
            lines.append(
                f"  - {obj.label}: parent={parent} room={obj.room_id or 'unknown'} "
                f"confidence={obj.confidence:.2f}"
            )
        return "\n".join(lines)

    def decay_confidence(self, elapsed_seconds: float | None = None) -> dict[str, Any]:
        if elapsed_seconds is not None and (
            not math.isfinite(elapsed_seconds) or elapsed_seconds < 0
        ):
            raise WorldGraphError("elapsed_seconds must be finite and non-negative")
        if not self._objects:
            return {"decayed": [], "removed": [], "elapsed_seconds": 0}
        now = time.time()
        decayed: list[str] = []
        removed: list[str] = []
        for object_id in list(self._objects):
            obj = self._objects[object_id]
            elapsed = elapsed_seconds if elapsed_seconds is not None else now - obj.last_seen
            if elapsed <= 0:
                continue
            confidence = obj.confidence * 0.5 ** (elapsed / _CONFIDENCE_HALF_LIFE_SECONDS)
            if confidence < _CONFIDENCE_MIN_THRESHOLD:
                self.remove_object(object_id, source="confidence_decay")
                removed.append(object_id)
            else:
                self.update_object(
                    object_id,
                    source="confidence_decay",
                    confidence=confidence,
                    last_seen=obj.last_seen,
                )
                decayed.append(object_id)
        return {
            "decayed": decayed,
            "removed": removed,
            "elapsed_seconds": elapsed_seconds if elapsed_seconds is not None else "per-object",
        }

    def decay_confidence_to_now(self) -> dict[str, Any]:
        return self.decay_confidence()

    def reset_confidence(self, object_id: str, confidence: float = 1.0) -> None:
        if object_id in self._objects:
            self.update_object(
                object_id,
                source="observation_refresh",
                confidence=confidence,
                last_seen=time.time(),
            )

    def _object_has_parent(self, label: str, parent: str, entity_type: str) -> bool:
        for obj in self.get_objects_by_label(label):
            if self._entity_matches(obj.parent_id, parent, entity_type):
                return True
            if (
                entity_type == "surface"
                and obj.state == "on_table"
                and parent in {"table", "桌面"}
            ):
                return True
            legacy = str(obj.properties.get("placed_on", "")).lower()
            if entity_type == "surface" and parent.lower() in legacy:
                return True
        return False

    def _entity_matches(self, entity_id: str | None, target: str, entity_type: str) -> bool:
        if entity_id is None:
            return False
        entity = self._entities.get(entity_id)
        return bool(
            entity
            and entity.entity_type == entity_type
            and (
                _normalize_symbol(entity.entity_id) == _normalize_symbol(target)
                or _normalize_symbol(target) in _normalize_symbol(entity.label)
            )
        )

    def _validate_entity_relation(self, entity: SpatialEntity) -> None:
        if entity.entity_type == "room" and entity.parent_id is not None:
            raise WorldGraphError("room entities cannot have a parent")
        if entity.entity_type != "room" and entity.parent_id is None:
            raise WorldGraphError("non-room spatial entities require a parent")
        if entity.parent_id == entity.entity_id:
            raise WorldGraphError("spatial entity relationship contains a self-cycle")
        if entity.parent_id is None:
            return
        parent = self._entities.get(entity.parent_id)
        if parent is None:
            raise WorldGraphError("spatial entity parent does not exist")
        allowed_parents = {
            "surface": {"room"},
            "zone": {"room"},
            "receptacle": {"room", "surface"},
        }
        if parent.entity_type not in allowed_parents[entity.entity_type]:
            raise WorldGraphError("spatial entity has an incompatible parent type")
        cursor: SpatialEntity | None = parent
        seen: set[str] = set()
        while cursor is not None:
            if cursor.entity_id == entity.entity_id or cursor.entity_id in seen:
                raise WorldGraphError("spatial entity relationship contains a cycle")
            seen.add(cursor.entity_id)
            cursor = self._entities.get(cursor.parent_id) if cursor.parent_id else None

    def _normalize_object_relation(self, obj: ObjectState) -> ObjectState:
        if obj.parent_id is None:
            return obj
        inherited_room = self._entity_room(obj.parent_id)
        if inherited_room is None:
            return obj
        if obj.room_id is not None and obj.room_id != inherited_room:
            raise WorldGraphError("object room conflicts with its parent entity")
        return obj if obj.room_id == inherited_room else replace(obj, room_id=inherited_room)

    def _entity_room(self, entity_id: str) -> str | None:
        entity = self._entities.get(entity_id)
        seen: set[str] = set()
        while entity is not None and entity.entity_id not in seen:
            seen.add(entity.entity_id)
            if entity.entity_type == "room":
                return entity.entity_id
            entity = self._entities.get(entity.parent_id) if entity.parent_id else None
        return None

    def _validate_object_relation(self, obj: ObjectState) -> None:
        if obj.parent_id is not None:
            parent = self._entities.get(obj.parent_id)
            if parent is None:
                raise WorldGraphError("object parent_id does not exist")
            if parent.entity_type not in {"surface", "receptacle"}:
                raise WorldGraphError("object parent must be a surface or receptacle")
        if obj.room_id is not None:
            room = self._entities.get(obj.room_id)
            if room is None or room.entity_type != "room":
                raise WorldGraphError("object room_id must reference a room entity")
        inherited_room = self._entity_room(obj.parent_id) if obj.parent_id else None
        if inherited_room is not None and obj.room_id != inherited_room:
            raise WorldGraphError("object room does not match its parent entity")

    def _validate_graph(self) -> None:
        for entity in self._entities.values():
            self._validate_entity_relation(entity)
        for obj in self._objects.values():
            self._validate_object_relation(obj)
        if self._robot_state.room_id is not None:
            room = self._entities.get(self._robot_state.room_id)
            if room is None or room.entity_type != "room":
                raise WorldGraphError("robot room does not reference a room entity")

    def _commit(self, event_type: str, source: str, payload: dict[str, Any]) -> None:
        if not self._events and event_type != "checkpoint":
            self._commit(
                "checkpoint",
                "world_init",
                {
                    "entities": {
                        key: value.to_dict() for key, value in self._entities.items()
                    },
                    "objects": {
                        key: value.to_dict() for key, value in self._objects.items()
                    },
                    "robot_state": self._robot_state.to_dict(),
                },
            )
        previous_hash = self._events[-1].event_hash if self._events else ZERO_HASH
        event = WorldEvent.create(
            sequence=len(self._events) + 1,
            event_type=event_type,
            source=source,
            payload=payload,
            previous_hash=previous_hash,
        )
        if self._journal is not None:
            self._journal.append(event)
        self._apply_event(event)
        self._events.append(event)
        self._persist()

    def _apply_event(self, event: WorldEvent) -> None:
        payload = event.payload
        if event.event_type == "checkpoint":
            self._entities = {
                key: SpatialEntity.from_dict(value)
                for key, value in payload.get("entities", {}).items()
            }
            self._objects = {
                key: ObjectState.from_dict(value)
                for key, value in payload.get("objects", {}).items()
            }
            self._robot_state = RobotState.from_dict(payload.get("robot_state", {}))
            self._diff_history = []
            self._validate_graph()
            return
        if event.event_type == "entity_upsert":
            entity = SpatialEntity.from_dict(payload["entity"])
            self._entities[entity.entity_id] = entity
        elif event.event_type == "entity_remove":
            self._entities.pop(str(payload["entity_id"]), None)
        elif event.event_type == "object_upsert":
            obj = ObjectState.from_dict(payload["object"])
            self._objects[obj.object_id] = obj
        elif event.event_type == "object_remove":
            self._objects.pop(str(payload["object_id"]), None)
        elif event.event_type == "observation_fusion":
            for raw in payload["objects"]:
                obj = ObjectState.from_dict(raw)
                self._objects[obj.object_id] = obj
            self._validate_graph()
        elif event.event_type == "semantic_map_sync":
            for raw in payload["entities"]:
                entity = SpatialEntity.from_dict(raw)
                self._entities[entity.entity_id] = entity
            self._validate_graph()
        elif event.event_type == "robot_update":
            self._robot_state = RobotState.from_dict(payload["robot_state"])
        elif event.event_type == "task_result":
            self._diff_history.append(
                WorldDiff(
                    timestamp=event.recorded_at,
                    skill_name=str(payload["skill_name"]),
                    pre_state=dict(payload["pre_state"]),
                    post_state=dict(payload["post_state"]),
                    success=bool(payload["success"]),
                    message=str(payload.get("message", "")),
                )
            )
            self._diff_history = self._diff_history[-100:]

    def _load(self) -> None:
        if self._journal is not None and self._journal.path.exists():
            events = self._journal.load()
            if events:
                for event in events:
                    self._apply_event(event)
                self._events = events
                self._validate_graph()
                return
        if self._persistence_path is None or not self._persistence_path.exists():
            return
        raw = yaml.safe_load(self._persistence_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise WorldGraphError("world model snapshot must be an object")
        entities_raw = raw.get("entities", {})
        objects_raw = raw.get("objects", {})
        if not isinstance(entities_raw, dict) or not isinstance(objects_raw, dict):
            raise WorldGraphError("world model snapshot collections must be objects")
        self._entities = {
            str(key): SpatialEntity.from_dict(value) for key, value in entities_raw.items()
        }
        self._objects = {
            str(key): ObjectState.from_dict(value) for key, value in objects_raw.items()
        }
        self._robot_state = RobotState.from_dict(raw.get("robot_state", {}))
        self._validate_graph()
        self._bootstrap_snapshot()

    def _bootstrap_snapshot(self) -> None:
        payload = {
            "entities": {key: value.to_dict() for key, value in self._entities.items()},
            "objects": {key: value.to_dict() for key, value in self._objects.items()},
            "robot_state": self._robot_state.to_dict(),
        }
        self._commit("checkpoint", "snapshot_migration", payload)

    def _persist(self) -> None:
        if self._persistence_path is None:
            return
        data = {
            "schema_version": 2,
            "revision": self.revision,
            "last_event_hash": self._events[-1].event_hash if self._events else ZERO_HASH,
            "entities": {key: value.to_dict() for key, value in self._entities.items()},
            "objects": {key: value.to_dict() for key, value in self._objects.items()},
            "robot_state": self._robot_state.to_dict(),
        }
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self._persistence_path.with_name(
            f".{self._persistence_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle, allow_unicode=True, default_flow_style=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self._persistence_path)
            self._persistence_path.chmod(0o600)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def _normalized_json(value: dict[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorldGraphError(f"{label} must be an object")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise WorldGraphError(f"{label} must be JSON-safe") from exc
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise WorldGraphError(f"{label} must be an object")
    return normalized


def _normalize_symbol(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


__all__ = [
    "ObjectState",
    "RobotState",
    "SpatialEntity",
    "WorldDiff",
    "WorldEvent",
    "WorldGraphError",
    "WorldModel",
]
