from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_agent.deployment import DeploymentError, validate_deployment
from robot_agent.missions import MissionCatalog
from robot_agent.motion import DockTarget, MotionCatalog, PoseTarget
from robot_agent.semantic_map import SemanticMapSpec
from robot_agent.table_docking import TableCatalog, TableDockingConfig, TableProfile
from robot_agent.task_catalog import TaskCatalog


def _semantic_map(tmp_path: Path, *, target: str = "living_room_service") -> SemanticMapSpec:
    path = tmp_path / "semantic-map.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "approved": True,
                "map_id": "home_v1",
                "planning_frame": "map",
                "surveyed_at": 1_700_000_000.0,
                "entities": [
                    {
                        "entity_id": "living_room",
                        "label": "Living room",
                        "entity_type": "room",
                        "parent_id": None,
                        "position": [0.0, 0.0, 0.0],
                        "confidence": 1.0,
                        "properties": {},
                    },
                    {
                        "entity_id": "service_zone",
                        "label": "Living-room service zone",
                        "entity_type": "zone",
                        "parent_id": "living_room",
                        "position": [1.0, 2.0, 0.0],
                        "confidence": 1.0,
                        "properties": {
                            "navigation_binding": {
                                "action": "navigate",
                                "target": target,
                                "max_distance_m": 0.25,
                            }
                        },
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return SemanticMapSpec.load(path)


def _motion_catalog(*, frame: str = "map", x: float = 1.0) -> MotionCatalog:
    return MotionCatalog(
        version=1,
        locations={
            "living_room_service": PoseTarget(
                "living_room_service", frame, x, 2.0, 0.0
            )
        },
        docks={"charging_dock": DockTarget("charging_dock", "dock-1")},
    )


def _mission_catalog(tmp_path: Path, action: str, target: str) -> MissionCatalog:
    path = tmp_path / f"mission-{action}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "missions": {
                    "home_service": {
                        "name": "Home service",
                        "description": "One bounded deployment action",
                        "aliases": ["home service"],
                        "requires_observation": False,
                        "max_runtime_seconds": 120,
                        "max_node_visits": 4,
                        "graph": {
                            "id": "action",
                            "type": "action",
                            "action": action,
                            "target": target,
                            "timeout_seconds": 60,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return MissionCatalog.load(path)


def test_deployment_validates_catalogs_and_produces_stable_fingerprint(tmp_path: Path):
    values = {
        "task_catalog": TaskCatalog.default(),
        "mission_catalog": _mission_catalog(
            tmp_path, "navigate", "living_room_service"
        ),
        "motion_catalog": _motion_catalog(),
        "semantic_map": _semantic_map(tmp_path),
    }

    first = validate_deployment(**values)
    second = validate_deployment(**values)

    assert first == second
    assert first.semantic_navigation_bindings == 1
    assert len(first.fingerprint) == 64


def test_deployment_rejects_cross_frame_configs(tmp_path: Path):
    with pytest.raises(DeploymentError, match="do not use the semantic map frame"):
        validate_deployment(
            task_catalog=TaskCatalog.default(),
            mission_catalog=MissionCatalog.empty(),
            motion_catalog=_motion_catalog(frame="odom"),
            semantic_map=_semantic_map(tmp_path),
        )


@pytest.mark.parametrize(
    ("action", "target"),
    [
        ("navigate", "missing_location"),
        ("dock", "missing_dock"),
        ("task", "missing_task"),
        ("dock_table", "missing_table"),
    ],
)
def test_deployment_rejects_unknown_mission_targets(
    tmp_path: Path,
    action: str,
    target: str,
):
    with pytest.raises(DeploymentError, match="references unknown"):
        validate_deployment(
            task_catalog=TaskCatalog.default(),
            mission_catalog=_mission_catalog(tmp_path, action, target),
            motion_catalog=_motion_catalog(),
            semantic_map=None,
        )


def test_deployment_rejects_unknown_or_distant_semantic_navigation_target(
    tmp_path: Path,
):
    with pytest.raises(DeploymentError, match="unknown navigate target"):
        validate_deployment(
            task_catalog=TaskCatalog.default(),
            mission_catalog=MissionCatalog.empty(),
            motion_catalog=_motion_catalog(),
            semantic_map=_semantic_map(tmp_path, target="missing"),
        )

    with pytest.raises(DeploymentError, match="exceeding"):
        validate_deployment(
            task_catalog=TaskCatalog.default(),
            mission_catalog=MissionCatalog.empty(),
            motion_catalog=_motion_catalog(x=2.0),
            semantic_map=_semantic_map(tmp_path),
        )


def test_deployment_fingerprint_changes_with_navigation_calibration(tmp_path: Path):
    common = {
        "task_catalog": TaskCatalog.default(),
        "mission_catalog": MissionCatalog.empty(),
        "semantic_map": None,
    }
    first = validate_deployment(motion_catalog=_motion_catalog(x=1.0), **common)
    second = validate_deployment(motion_catalog=_motion_catalog(x=1.1), **common)
    assert first.fingerprint != second.fingerprint
    assert first.motion_fingerprint != second.motion_fingerprint
    assert first.manipulation_fingerprint == second.manipulation_fingerprint


def test_domain_fingerprints_do_not_invalidate_unrelated_hil(tmp_path: Path):
    common = {
        "task_catalog": TaskCatalog.default(),
        "motion_catalog": _motion_catalog(),
        "semantic_map": _semantic_map(tmp_path),
    }
    empty = validate_deployment(mission_catalog=MissionCatalog.empty(), **common)
    with_mission = validate_deployment(
        mission_catalog=_mission_catalog(tmp_path, "navigate", "living_room_service"),
        **common,
    )
    assert empty.fingerprint != with_mission.fingerprint
    assert empty.motion_fingerprint == with_mission.motion_fingerprint
    assert empty.manipulation_fingerprint == with_mission.manipulation_fingerprint


def _table_catalog(*, standoff_m: float = 0.45) -> TableCatalog:
    return TableCatalog(
        version=1,
        tables={
            "cabinet_edge": TableProfile(
                name="cabinet_edge",
                docking=TableDockingConfig(standoff_m=standoff_m),
            )
        },
    )


def test_deployment_accepts_and_fingerprints_table_docking_profile(tmp_path: Path):
    mission = _mission_catalog(tmp_path, "dock_table", "cabinet_edge")
    first = validate_deployment(
        task_catalog=TaskCatalog.default(),
        mission_catalog=mission,
        motion_catalog=_motion_catalog(),
        semantic_map=None,
        table_catalog=_table_catalog(standoff_m=0.45),
    )
    second = validate_deployment(
        task_catalog=TaskCatalog.default(),
        mission_catalog=mission,
        motion_catalog=_motion_catalog(),
        semantic_map=None,
        table_catalog=_table_catalog(standoff_m=0.55),
    )
    assert first.fingerprint != second.fingerprint
