from __future__ import annotations

from pathlib import Path

import yaml

from robot_agent import contract
from robot_agent.execution import WorkerResult
from robot_agent.missions import MissionCatalog
from robot_agent.motion import MotionCatalog, MotionResult, PoseTarget
from robot_agent.robotd import RobotDaemon
from robot_agent.safety import (
    FileSafetyInterlock,
    SafetyPolicy,
    SafetySnapshot,
    UnavailableSafetyInterlock,
    write_safety_snapshot,
)


def _policy(tmp_path: Path) -> SafetyPolicy:
    path = tmp_path / "safety-policy.yaml"
    path.write_text(
        """
version: 1
max_age_seconds: 1.0
allowed_source_prefixes: [safety/approved/]
require_physical_estop: true
deny_when_privacy_active: true
motion: {min_person_distance_m: 0.8, min_pet_distance_m: 0.5}
manipulation:
  min_person_distance_m: 1.2
  min_pet_distance_m: 0.8
  require_clear_workspace: true
""".strip(),
        encoding="utf-8",
    )
    return SafetyPolicy.load(path)


def _snapshot(sequence: int, *, privacy: bool = False) -> SafetySnapshot:
    return SafetySnapshot.create(
        sequence=sequence,
        observed_at=100.0,
        source="safety/approved/v1",
        sensor_health="ok",
        physical_estop_available=True,
        emergency_stop_pressed=False,
        person_count=0,
        nearest_person_m=None,
        pet_count=0,
        nearest_pet_m=None,
        person_in_workspace=False,
        pet_in_workspace=False,
        privacy_zone_active=privacy,
    )


class ImmediateExecutor:
    active = False

    def __init__(self):
        self.calls = []

    def execute(self, task_id, args, token, timeout_seconds):
        self.calls.append(task_id)
        return WorkerResult(True, "done", {"steps": []})

    def cancel(self, *, reason, emergency=False):
        return None


def _arm_task(daemon: RobotDaemon):
    return daemon.request_arm(
        "pick_up_bottle",
        {},
        operator_confirmation="CONFIRM pick_up_bottle",
    )


def test_missing_interlock_refuses_arm_with_structured_reason(tmp_path: Path):
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        real_executor=ImmediateExecutor(),
        safety_interlock=UnavailableSafetyInterlock("detector offline"),
        require_safety_interlock=True,
    )
    result = _arm_task(daemon)
    assert result.success is False
    assert result.refusal.reason_code == contract.SAFETY_INTERLOCK_BLOCKED
    assert result.refusal.details["operation"] == "manipulation"
    assert daemon.status()["safety_interlock"]["required"] is True


def test_unsafe_execute_does_not_consume_token_and_new_safe_sequence_recovers(
    tmp_path: Path,
):
    snapshot_path = tmp_path / "safety.json"
    write_safety_snapshot(snapshot_path, _snapshot(1))
    executor = ImmediateExecutor()
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        real_executor=executor,
        safety_interlock=FileSafetyInterlock(snapshot_path, _policy(tmp_path)),
        require_safety_interlock=True,
        clock=lambda: 100.2,
    )
    armed = _arm_task(daemon)
    token = armed.data["token_secret"]
    write_safety_snapshot(snapshot_path, _snapshot(2, privacy=True))
    blocked = daemon.execute("pick_up_bottle", {}, token=token)
    assert blocked.refusal.reason_code == contract.SAFETY_INTERLOCK_BLOCKED
    assert executor.calls == []

    write_safety_snapshot(snapshot_path, _snapshot(3))
    executed = daemon.execute("pick_up_bottle", {}, token=token)
    assert executed.success is True
    assert executor.calls == ["pick_up_bottle"]
    assert executed.data["safety_decisions"]["manipulation"]["snapshot_sequence"] == 3


def _motion_mission(tmp_path: Path) -> MissionCatalog:
    path = tmp_path / "missions.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "missions": {
                    "patrol": {
                        "name": "Patrol",
                        "description": "Visit two named locations",
                        "aliases": ["巡逻"],
                        "requires_observation": False,
                        "max_runtime_seconds": 120,
                        "max_node_visits": 8,
                        "graph": {
                            "id": "root",
                            "type": "sequence",
                            "children": [
                                {
                                    "id": "first",
                                    "type": "action",
                                    "action": "navigate",
                                    "target": "first",
                                    "timeout_seconds": 30,
                                },
                                {
                                    "id": "second",
                                    "type": "action",
                                    "action": "navigate",
                                    "target": "second",
                                    "timeout_seconds": 30,
                                },
                            ],
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return MissionCatalog.load(path)


class SnapshotChangingGateway:
    active = False

    def __init__(self, snapshot_path: Path):
        self.snapshot_path = snapshot_path
        self.calls = []

    def health(self):
        return {"available": True}

    def navigate(self, target, timeout_seconds):
        self.calls.append(target.name)
        if target.name == "first":
            write_safety_snapshot(self.snapshot_path, _snapshot(2, privacy=True))
        return MotionResult(True, "arrived")

    def dock(self, target, timeout_seconds):
        raise AssertionError("dock is not expected")

    def cancel(self, *, reason, emergency=False):
        return None

    def close(self):
        return None


def test_mission_rechecks_interlock_between_nodes(tmp_path: Path):
    snapshot_path = tmp_path / "safety.json"
    write_safety_snapshot(snapshot_path, _snapshot(1))
    gateway = SnapshotChangingGateway(snapshot_path)
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        motion_gateway=gateway,
        motion_catalog=MotionCatalog(
            version=1,
            locations={
                "first": PoseTarget("first", "map", 0.0, 0.0, 0.0),
                "second": PoseTarget("second", "map", 1.0, 0.0, 0.0),
            },
            docks={},
        ),
        mission_catalog=_motion_mission(tmp_path),
        safety_interlock=FileSafetyInterlock(snapshot_path, _policy(tmp_path)),
        require_safety_interlock=True,
        clock=lambda: 100.2,
    )
    armed = daemon.request_mission_arm(
        "patrol",
        operator_confirmation="CONFIRM mission:patrol",
    )
    result = daemon.execute_mission("patrol", token=armed.data["token_secret"])
    assert result.success is False
    assert gateway.calls == ["first"]
    failed_steps = [step for step in result.data["steps"] if not step["success"]]
    assert any("privacy" in step["message"] for step in failed_steps)
