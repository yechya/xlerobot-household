from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .missions import MissionCatalog
from .motion import MotionCatalog
from .semantic_map import SemanticMapSpec
from .table_docking import TableCatalog
from .task_catalog import TaskCatalog


class DeploymentError(ValueError):
    pass


@dataclass(frozen=True)
class DeploymentReport:
    fingerprint: str
    motion_fingerprint: str
    manipulation_fingerprint: str
    planning_frame: str | None
    semantic_map_id: str | None
    task_count: int
    mission_count: int
    location_count: int
    dock_count: int
    semantic_navigation_bindings: int

    def public_view(self) -> dict[str, Any]:
        return {
            "available": True,
            "fingerprint": self.fingerprint,
            "motion_fingerprint": self.motion_fingerprint,
            "manipulation_fingerprint": self.manipulation_fingerprint,
            "planning_frame": self.planning_frame,
            "semantic_map_id": self.semantic_map_id,
            "task_count": self.task_count,
            "mission_count": self.mission_count,
            "location_count": self.location_count,
            "dock_count": self.dock_count,
            "semantic_navigation_bindings": self.semantic_navigation_bindings,
        }


def validate_deployment(
    *,
    task_catalog: TaskCatalog,
    mission_catalog: MissionCatalog,
    motion_catalog: MotionCatalog,
    semantic_map: SemanticMapSpec | None,
    table_catalog: TableCatalog | None = None,
) -> DeploymentReport:
    table_catalog = table_catalog or TableCatalog.empty()
    planning_frame = semantic_map.planning_frame if semantic_map is not None else None
    if semantic_map is not None:
        mismatched = sorted(
            target.name
            for target in motion_catalog.locations.values()
            if target.frame_id != semantic_map.planning_frame
        )
        if mismatched:
            raise DeploymentError(
                "navigation locations do not use the semantic map frame: "
                + ", ".join(mismatched)
            )
    _validate_missions(task_catalog, mission_catalog, motion_catalog, table_catalog)
    binding_count = _validate_semantic_navigation_bindings(
        semantic_map,
        motion_catalog,
    )
    motion_payload = {
        "schema_version": 1,
        "motion_catalog": _motion_payload(motion_catalog),
        "semantic_map_fingerprint": (
            semantic_map.fingerprint if semantic_map is not None else None
        ),
    }
    manipulation_payload = {
        "schema_version": 1,
        "task_catalog": task_catalog.public_view(),
        "semantic_map_fingerprint": (
            semantic_map.fingerprint if semantic_map is not None else None
        ),
    }
    motion_fingerprint = _digest(motion_payload)
    manipulation_fingerprint = _digest(manipulation_payload)
    payload = {
        "schema_version": 1,
        "mission_catalog": mission_catalog.public_view(),
        "table_catalog": table_catalog.public_view(),
        "motion_fingerprint": motion_fingerprint,
        "manipulation_fingerprint": manipulation_fingerprint,
    }
    return DeploymentReport(
        fingerprint=_digest(payload),
        motion_fingerprint=motion_fingerprint,
        manipulation_fingerprint=manipulation_fingerprint,
        planning_frame=planning_frame,
        semantic_map_id=semantic_map.map_id if semantic_map is not None else None,
        task_count=len(task_catalog.tasks),
        mission_count=len(mission_catalog.missions),
        location_count=len(motion_catalog.locations),
        dock_count=len(motion_catalog.docks),
        semantic_navigation_bindings=binding_count,
    )


def _validate_missions(
    task_catalog: TaskCatalog,
    mission_catalog: MissionCatalog,
    motion_catalog: MotionCatalog,
    table_catalog: TableCatalog,
) -> None:
    for mission_id, mission in sorted(mission_catalog.missions.items()):
        for action in mission.actions:
            if action.action == "task":
                available = action.target in task_catalog.tasks
            elif action.action == "navigate":
                available = action.target in motion_catalog.locations
            elif action.action == "dock_table":
                available = action.target in table_catalog.tables
            elif action.action == "observe":
                available = True
            else:
                available = action.target in motion_catalog.docks
            if not available:
                raise DeploymentError(
                    f"mission {mission_id!r} references unknown "
                    f"{action.action} target {action.target!r}"
                )


def _validate_semantic_navigation_bindings(
    semantic_map: SemanticMapSpec | None,
    motion_catalog: MotionCatalog,
) -> int:
    if semantic_map is None:
        return 0
    count = 0
    for entity in semantic_map.entities:
        raw = entity.properties.get("navigation_binding")
        if raw is None:
            continue
        count += 1
        if entity.entity_type != "zone":
            raise DeploymentError(
                f"semantic entity {entity.entity_id!r} navigation binding requires a zone"
            )
        if not isinstance(raw, dict):
            raise DeploymentError(
                f"semantic entity {entity.entity_id!r} navigation binding must be an object"
            )
        action = raw.get("action")
        target_name = raw.get("target")
        if action == "navigate":
            if set(raw) != {"action", "target", "max_distance_m"}:
                raise DeploymentError(
                    f"semantic entity {entity.entity_id!r} navigate binding has invalid fields"
                )
            target = motion_catalog.locations.get(_target_name(target_name, entity.entity_id))
            if target is None:
                raise DeploymentError(
                    f"semantic entity {entity.entity_id!r} references unknown navigate target"
                )
            maximum = _distance(raw.get("max_distance_m"), entity.entity_id)
            distance = math.hypot(entity.x - target.x, entity.y - target.y)
            if distance > maximum:
                raise DeploymentError(
                    f"semantic entity {entity.entity_id!r} is {distance:.3f}m from its "
                    f"navigation target, exceeding {maximum:.3f}m"
                )
        elif action == "dock":
            if set(raw) != {"action", "target"}:
                raise DeploymentError(
                    f"semantic entity {entity.entity_id!r} dock binding has invalid fields"
                )
            target = _target_name(target_name, entity.entity_id)
            if target not in motion_catalog.docks:
                raise DeploymentError(
                    f"semantic entity {entity.entity_id!r} references unknown dock target"
                )
        else:
            raise DeploymentError(
                f"semantic entity {entity.entity_id!r} navigation action is invalid"
            )
    return count


def _target_name(value: Any, entity_id: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise DeploymentError(
            f"semantic entity {entity_id!r} navigation target is invalid"
        )
    return value


def _distance(value: Any, entity_id: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= 5.0
    ):
        raise DeploymentError(
            f"semantic entity {entity_id!r} max_distance_m must be in (0, 5]"
        )
    return float(value)


def _motion_payload(catalog: MotionCatalog) -> dict[str, Any]:
    return {
        "version": catalog.version,
        "locations": {
            name: {
                "frame_id": target.frame_id,
                "x": target.x,
                "y": target.y,
                "yaw": target.yaw,
            }
            for name, target in sorted(catalog.locations.items())
        },
        "docks": {
            name: {"dock_id": target.dock_id}
            for name, target in sorted(catalog.docks.items())
        },
    }


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["DeploymentError", "DeploymentReport", "validate_deployment"]
