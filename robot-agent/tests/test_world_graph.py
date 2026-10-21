from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_agent.config import RuntimeConfig
from robot_agent.service import RobotRuntimeServer
from robot_agent.world_graph import SpatialEntity, WorldGraphError, ZERO_HASH
from robot_agent.world_model import ObjectState, WorldModel


def _semantic_world(path: Path | None = None) -> WorldModel:
    model = WorldModel(path)
    model.add_entity(
        SpatialEntity("kitchen", "Kitchen", "room", source="semantic_map")
    )
    model.add_entity(
        SpatialEntity(
            "table-1",
            "Kitchen table",
            "surface",
            parent_id="kitchen",
            source="semantic_map",
        )
    )
    model.add_entity(
        SpatialEntity(
            "tray-1",
            "Remote tray",
            "receptacle",
            parent_id="table-1",
            source="rgbd:head",
        )
    )
    return model


def test_instance_graph_tracks_rooms_parents_sources_and_predicates():
    model = _semantic_world()
    model.add_object(
        ObjectState(
            "remote-7",
            "remote",
            parent_id="tray-1",
            confidence=0.94,
            source="rgbd:head",
            observation_id="frame-42",
        )
    )
    model.update_robot_state(room_id="kitchen", source="amcl")

    remote = model.get_object("remote-7")
    assert remote.room_id == "kitchen"
    assert remote.parent_id == "tray-1"
    assert remote.source == "rgbd:head"
    assert remote.observation_id == "frame-42"
    assert model.check_precondition("object_in(remote,tray-1)") is True
    assert model.check_precondition("object_in(remote,remote_tray)") is True
    assert model.check_precondition("robot_in(kitchen)") is True


def test_event_journal_replays_snapshot_and_task_history(tmp_path: Path):
    path = tmp_path / "world.yaml"
    model = _semantic_world(path)
    model.add_object(
        ObjectState("bottle-1", "bottle", parent_id="table-1", source="rgbd:head")
    )
    model.record_diff("place_bottle", {"held": True}, {"placed": True}, True, "ok")

    events = model.get_events()
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[0].previous_hash == ZERO_HASH
    assert all(
        event.previous_hash == events[index - 1].event_hash
        for index, event in enumerate(events[1:], start=1)
    )

    reloaded = WorldModel(path)
    replayed = WorldModel.from_events(events)
    assert reloaded.to_dict() == replayed.to_dict() == model.to_dict()
    assert reloaded.get_diff_history()[0].skill_name == "place_bottle"
    event_path = path.with_suffix(".yaml.events.jsonl")
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert oct(event_path.stat().st_mode & 0o777) == "0o600"


def test_legacy_snapshot_migrates_to_checkpoint_event(tmp_path: Path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "objects": {
                    "cup-1": {
                        "object_id": "cup-1",
                        "label": "cup",
                        "confidence": 0.8,
                    }
                },
                "robot_state": {"gripper_state": "empty"},
            }
        ),
        encoding="utf-8",
    )

    model = WorldModel(path)
    assert model.get_object("cup-1").source == "legacy"
    assert model.get_events()[0].event_type == "checkpoint"
    assert model.get_events()[0].source == "snapshot_migration"
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["schema_version"] == 2


def test_event_hash_tampering_fails_closed(tmp_path: Path):
    path = tmp_path / "world.yaml"
    model = _semantic_world(path)
    event_path = path.with_suffix(".yaml.events.jsonl")
    lines = event_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["payload"]["entity"]["label"] = "Tampered room"
    lines[1] = json.dumps(tampered)
    event_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(WorldGraphError, match="hash does not match"):
        WorldModel(path)


def test_stale_writer_is_rejected_instead_of_overwriting_new_events(tmp_path: Path):
    path = tmp_path / "world.yaml"
    first = WorldModel(path)
    stale = WorldModel(path)
    first.add_entity(SpatialEntity("kitchen", "Kitchen", "room"))

    with pytest.raises(WorldGraphError, match="changed concurrently"):
        stale.add_entity(SpatialEntity("living", "Living room", "room"))

    reloaded = WorldModel(path)
    assert reloaded.get_entity("kitchen") is not None
    assert reloaded.get_entity("living") is None


def test_invalid_relations_and_cycles_are_rejected():
    model = _semantic_world()
    with pytest.raises(WorldGraphError, match="incompatible parent"):
        model.add_entity(
            SpatialEntity(
                "bad-surface",
                "Bad surface",
                "surface",
                parent_id="tray-1",
            )
        )
    with pytest.raises(WorldGraphError, match="cycle"):
        model.update_entity("table-1", parent_id="table-1")
    with pytest.raises(WorldGraphError, match="conflicts"):
        model.add_entity(SpatialEntity("living", "Living room", "room"))
        model.add_object(
            ObjectState(
                "remote-1",
                "remote",
                parent_id="tray-1",
                room_id="living",
            )
        )


def test_structured_detections_keep_observation_provenance_and_text_is_ignored():
    model = WorldModel()
    model.update_from_vlm("a bottle may be visible")
    assert model.revision == 0
    model.update_from_vlm(
        "structured result",
        [
            {
                "instance_id": "17",
                "label": "bottle",
                "x": 0.2,
                "confidence": 0.88,
                "frame_id": "base_link",
            }
        ],
        source="detector:grounded-sam",
        captured_at=100.0,
        observation_id="rgbd-frame-100",
    )
    bottle = model.get_object("bottle_17")
    assert bottle.source == "detector:grounded-sam"
    assert bottle.last_seen == 100.0
    assert bottle.observation_id == "rgbd-frame-100"


def test_returned_instances_and_events_cannot_mutate_authoritative_state():
    model = _semantic_world()
    model.add_object(
        ObjectState(
            "remote-1",
            "remote",
            parent_id="tray-1",
            properties={"tags": ["electronics"]},
        )
    )
    returned = model.get_object("remote-1")
    event = model.get_events()[-1]
    returned.properties["tags"].append("tampered")
    event.payload["object"]["label"] = "tampered"

    assert model.get_object("remote-1").properties == {"tags": ["electronics"]}
    assert model.get_events()[-1].payload["object"]["label"] == "remote"


def test_reset_is_replayable_and_preserves_revision(tmp_path: Path):
    path = tmp_path / "world.yaml"
    model = _semantic_world(path)
    previous_revision = model.revision
    model.reset(source="operator")
    assert model.revision == previous_revision + 1
    assert model.get_entities() == []
    assert model.get_all_objects() == []

    reloaded = WorldModel(path)
    assert reloaded.revision == model.revision
    assert reloaded.get_entities() == []


def test_robotd_exposes_bounded_read_only_world_pages(tmp_path: Path):
    world_path = tmp_path / "state" / "world.yaml"
    model = _semantic_world(world_path)
    config = RuntimeConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        socket_path=tmp_path / "run" / "robotd.sock",
        token_store_path=tmp_path / "state" / "tokens.json",
        audit_log_path=tmp_path / "state" / "audit.jsonl",
        world_model_path=world_path,
        token_cache_path=tmp_path / "state" / "token.json",
        require_fresh_observation=False,
        observation_ttl_seconds=30,
        shutdown_grace_seconds=0.2,
        emergency_stop_command=(),
    )
    server = RobotRuntimeServer(config)
    try:
        state = server.dispatch({"command": "world_state", "params": {}})
        events = server.dispatch(
            {
                "command": "world_events",
                "params": {"since_sequence": 1, "limit": 2},
            }
        )
    finally:
        server.server_close()
    assert state["success"] is True
    assert state["data"]["revision"] == model.revision
    assert len(events["data"]["events"]) == 2
    assert events["data"]["events"][0]["sequence"] == 2
    with pytest.raises(ValueError, match="limit"):
        server.robot_daemon.world_events(limit=1001)
