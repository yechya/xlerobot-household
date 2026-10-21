from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_agent.config import RuntimeConfig
from robot_agent.semantic_map import SemanticMapError, SemanticMapSpec, sync_semantic_map
from robot_agent.service import RobotRuntimeServer
from robot_agent.world_graph import SpatialEntity
from robot_agent.world_model import ObjectState, WorldModel


def _raw_map() -> dict:
    return {
        "version": 1,
        "approved": True,
        "map_id": "home_v1",
        "planning_frame": "map",
        "surveyed_at": 1_700_000_000.0,
        "entities": [
            {
                "entity_id": "remote_tray",
                "label": "Remote tray",
                "entity_type": "receptacle",
                "parent_id": "living_room_table",
                "position": [1.2, 0.4, 0.78],
                "confidence": 1.0,
                "properties": {"capacity": "small"},
            },
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
                "entity_id": "living_room_table",
                "label": "Living-room table",
                "entity_type": "surface",
                "parent_id": "living_room",
                "position": [1.2, 0.4, 0.75],
                "confidence": 1.0,
                "properties": {},
            },
            {
                "entity_id": "table_docking_zone",
                "label": "Table docking zone",
                "entity_type": "zone",
                "parent_id": "living_room",
                "position": [0.7, 0.4, 0.0],
                "confidence": 1.0,
                "properties": {
                    "navigation_binding": {
                        "action": "navigate",
                        "target": "living_room_service",
                        "max_distance_m": 0.25,
                    }
                },
            },
        ],
    }


def _write_map(path: Path, raw: dict | None = None) -> SemanticMapSpec:
    path.write_text(yaml.safe_dump(raw or _raw_map(), sort_keys=False), encoding="utf-8")
    return SemanticMapSpec.load(path)


def _config(tmp_path: Path, map_path: Path) -> RuntimeConfig:
    state = tmp_path / "state"
    navigation = tmp_path / "navigation.yaml"
    navigation.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "locations": {
                    "living_room_service": {
                        "frame_id": "map",
                        "x": 0.7,
                        "y": 0.4,
                        "yaw": 0.0,
                    }
                },
                "docks": {},
            }
        ),
        encoding="utf-8",
    )
    return RuntimeConfig(
        state_dir=state,
        runtime_dir=tmp_path / "run",
        socket_path=tmp_path / "run" / "robotd.sock",
        token_store_path=state / "tokens.json",
        audit_log_path=state / "audit.jsonl",
        world_model_path=state / "world.yaml",
        token_cache_path=state / "token.json",
        require_fresh_observation=False,
        observation_ttl_seconds=30,
        shutdown_grace_seconds=0.2,
        emergency_stop_command=(),
        navigation_catalog_path=navigation,
        semantic_map_config_path=map_path,
    )


def test_semantic_map_loads_reviewed_topology_and_syncs_atomically(tmp_path: Path):
    spec = _write_map(tmp_path / "semantic-map.yaml")
    model = WorldModel()

    result = sync_semantic_map(model, spec)

    assert result["changed"] is True
    assert model.get_entity("remote_tray").parent_id == "living_room_table"
    events = [event for event in model.get_events() if event.event_type == "semantic_map_sync"]
    assert len(events) == 1
    assert len(events[0].payload["entities"]) == 4


def test_semantic_map_rejects_missing_parent_and_cycles(tmp_path: Path):
    missing = _raw_map()
    missing["entities"][0]["parent_id"] = "missing_surface"
    with pytest.raises(SemanticMapError, match="parent does not exist"):
        _write_map(tmp_path / "missing.yaml", missing)

    cycle = _raw_map()
    cycle["entities"][1]["entity_type"] = "surface"
    cycle["entities"][1]["parent_id"] = "living_room_table"
    with pytest.raises(SemanticMapError, match="cycle"):
        _write_map(tmp_path / "cycle.yaml", cycle)


def test_semantic_map_restart_is_idempotent_and_changed_map_adds_one_event(tmp_path: Path):
    path = tmp_path / "semantic-map.yaml"
    spec = _write_map(path)
    world_path = tmp_path / "world.yaml"
    model = WorldModel(world_path)
    sync_semantic_map(model, spec)
    revision = model.revision

    unchanged = sync_semantic_map(WorldModel(world_path), SemanticMapSpec.load(path))
    assert unchanged["changed"] is False
    assert unchanged["revision"] == revision

    changed = _raw_map()
    changed["entities"][2]["position"] = [1.3, 0.4, 0.75]
    updated_spec = _write_map(path, changed)
    updated = sync_semantic_map(WorldModel(world_path), updated_spec)
    assert updated["changed"] is True
    assert updated["revision"] == revision + 1


def test_semantic_map_rejects_manual_conflicts_and_invalidates_existing_relations(
    tmp_path: Path,
):
    spec = _write_map(tmp_path / "semantic-map.yaml")
    manual = WorldModel()
    manual.add_entity(SpatialEntity("living_room", "Manual room", "room"))
    with pytest.raises(SemanticMapError, match="cannot overwrite non-map entity"):
        sync_semantic_map(manual, spec)

    model = WorldModel()
    sync_semantic_map(model, spec)
    model.add_object(ObjectState("remote-1", "remote", parent_id="remote_tray"))
    changed = _raw_map()
    changed["entities"][0]["entity_type"] = "zone"
    changed["entities"][0]["parent_id"] = "living_room"
    changed_spec = _write_map(tmp_path / "changed.yaml", changed)
    with pytest.raises(SemanticMapError, match="object parent"):
        sync_semantic_map(model, changed_spec)


def test_semantic_map_event_replays_and_fusion_relations_inherit_room(tmp_path: Path):
    spec = _write_map(tmp_path / "semantic-map.yaml")
    model = WorldModel()
    sync_semantic_map(model, spec)
    model.add_object(ObjectState("remote-1", "remote", parent_id="remote_tray"))

    replayed = WorldModel.from_events(model.get_events())
    assert replayed.to_dict() == model.to_dict()
    assert replayed.get_object("remote-1").room_id == "living_room"


def test_robotd_syncs_semantic_map_before_serving_and_exposes_fingerprint(tmp_path: Path):
    map_path = tmp_path / "semantic-map.yaml"
    spec = _write_map(map_path)
    config = _config(tmp_path, map_path)

    server = RobotRuntimeServer(config)
    try:
        capabilities = server.dispatch({"command": "capabilities", "params": {}})
        state = server.dispatch({"command": "world_state", "params": {}})
    finally:
        server.server_close()

    assert capabilities["semantic_map"]["available"] is True
    assert capabilities["semantic_map"]["fingerprint"] == spec.fingerprint
    assert capabilities["deployment"]["semantic_navigation_bindings"] == 1
    assert len(state["data"]["entities"]) == 4


def test_robotd_rejects_semantic_map_conflicting_with_existing_world(tmp_path: Path):
    map_path = tmp_path / "semantic-map.yaml"
    _write_map(map_path)
    config = _config(tmp_path, map_path)
    model = WorldModel(config.world_model_path)
    model.add_entity(SpatialEntity("living_room", "Manual room", "room"))

    with pytest.raises(RuntimeError, match="semantic map is invalid"):
        RobotRuntimeServer(config)
