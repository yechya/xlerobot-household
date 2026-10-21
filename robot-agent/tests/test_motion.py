from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path

import pytest

from robot_agent import contract
from robot_agent.config import RuntimeConfig
from robot_agent.episodes import EpisodeRecorder, load_episodes
from robot_agent.motion import DockTarget, MotionCatalog, MotionResult, PoseTarget
from robot_agent.robotd import RobotDaemon
from robot_agent.service import RobotRuntimeServer
from robot_agent.watchdog import HeartbeatState


class FakeMotionGateway:
    def __init__(self, blocking: bool = False):
        self.calls: list = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.blocking = blocking
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def health(self) -> dict:
        return {"available": True, "backend": "fake", "active": self.active}

    def navigate(self, target: PoseTarget, timeout_seconds: float) -> MotionResult:
        self.calls.append(("navigate", target.name, timeout_seconds))
        self._active = True
        self.started.set()
        if self.blocking:
            self.release.wait(timeout=10)
        self._active = False
        return MotionResult(True, "arrived", {"target": target.name})

    def dock(self, target: DockTarget, timeout_seconds: float) -> MotionResult:
        self.calls.append(("dock", target.name, timeout_seconds))
        return MotionResult(True, "docked", {"target": target.name})

    def cancel(self, *, reason: str, emergency: bool = False) -> None:
        self.calls.append(("cancel", reason, emergency))
        self._active = False
        self.release.set()

    def close(self) -> None:
        self.calls.append(("close",))


@pytest.fixture
def catalog() -> MotionCatalog:
    return MotionCatalog(
        version=1,
        locations={
            "kitchen": PoseTarget("kitchen", "map", 1.0, 2.0, 0.5),
            "living_room": PoseTarget("living_room", "map", -1.0, 0.0, 0.0),
        },
        docks={"charger": DockTarget("charger", "dock_1")},
    )


def arm_motion(daemon: RobotDaemon, operation: str, target: str) -> str:
    task_id = f"motion:{operation}:{target}"
    result = daemon.request_motion_arm(
        operation,
        target,
        operator_confirmation=f"CONFIRM {task_id}",
    )
    assert result.success, result.message
    return result.data["token_secret"]


def test_catalog_loads_named_targets_without_exposing_coordinates(tmp_path: Path):
    path = tmp_path / "navigation.yaml"
    path.write_text(
        """
version: 2
locations:
  table_staging: {frame_id: map, x: 1.0, y: -0.5, yaw: 1.57}
docks:
  charger: {dock_id: dock_alpha}
""".strip(),
        encoding="utf-8",
    )
    catalog = MotionCatalog.load(path)
    assert catalog.locations["table_staging"].frame_id == "map"
    assert catalog.docks["charger"].dock_id == "dock_alpha"
    assert catalog.public_view() == {
        "version": 2,
        "locations": ["table_staging"],
        "docks": ["charger"],
    }


def test_unknown_motion_target_is_refused(tmp_path: Path, catalog: MotionCatalog):
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        motion_gateway=FakeMotionGateway(),
        motion_catalog=catalog,
    )
    result = daemon.request_motion_arm(
        "navigate",
        "bedroom",
        operator_confirmation="CONFIRM motion:navigate:bedroom",
    )
    assert result.success is False
    assert result.refusal.reason_code == contract.TASK_NOT_ALLOWLISTED


def test_motion_token_is_bound_to_operation_and_target(tmp_path: Path, catalog: MotionCatalog):
    gateway = FakeMotionGateway()
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        motion_gateway=gateway,
        motion_catalog=catalog,
    )
    token = arm_motion(daemon, "navigate", "kitchen")
    mismatch = daemon.execute_motion("navigate", "living_room", token)
    result = daemon.execute_motion("navigate", "kitchen", token)
    assert mismatch.refusal.reason_code == contract.INVALID_TOKEN
    assert result.success is True
    assert result.state == contract.DRY_RUN
    assert gateway.calls == [("navigate", "kitchen", 300.0)]


def test_motion_token_is_bound_to_calibrated_target_and_deployment(
    tmp_path: Path,
    catalog: MotionCatalog,
):
    gateway = FakeMotionGateway()
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        motion_gateway=gateway,
        motion_catalog=catalog,
        deployment_fingerprint="d" * 64,
    )
    token = arm_motion(daemon, "navigate", "kitchen")
    assert daemon.status()["active_execution_binding"]["deployment_fingerprint"] == "d" * 64
    catalog.locations["kitchen"] = PoseTarget("kitchen", "map", 1.1, 2.0, 0.5)

    result = daemon.execute_motion("navigate", "kitchen", token)

    assert result.refusal.reason_code == contract.INVALID_TOKEN
    assert gateway.calls == []


def test_named_motion_is_recorded_in_unified_episode_format(
    tmp_path: Path,
    catalog: MotionCatalog,
):
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        motion_gateway=FakeMotionGateway(),
        motion_catalog=catalog,
        episode_recorder=EpisodeRecorder(tmp_path / "episodes"),
        clock=lambda: 1000.0,
    )
    token = arm_motion(daemon, "navigate", "kitchen")
    result = daemon.execute_motion("navigate", "kitchen", token)
    manifest = load_episodes(tmp_path / "episodes")[0]
    assert result.data["episode_id"] == manifest.episode_id
    assert manifest.executor == "named_motion"
    assert manifest.execution_binding["resolved_target"]["name"] == "kitchen"


def test_stop_cancels_active_motion_from_another_thread(tmp_path: Path, catalog: MotionCatalog):
    gateway = FakeMotionGateway(blocking=True)
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        motion_gateway=gateway,
        motion_catalog=catalog,
    )
    token = arm_motion(daemon, "navigate", "kitchen")
    results: list = []
    worker = threading.Thread(
        target=lambda: results.append(daemon.execute_motion("navigate", "kitchen", token))
    )
    worker.start()
    assert gateway.started.wait(timeout=5)
    stopped = daemon.stop()
    worker.join(timeout=5)
    assert stopped.state == contract.DISABLED
    assert ("cancel", "operator_stop", False) in gateway.calls
    assert not worker.is_alive()
    assert results[0].state == contract.DISABLED


def test_socket_dispatch_exposes_named_motion_flow(tmp_path: Path, catalog: MotionCatalog):
    config = RuntimeConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        socket_path=tmp_path / "run" / "robotd.sock",
        token_store_path=tmp_path / "state" / "tokens.json",
        audit_log_path=tmp_path / "state" / "audit.jsonl",
        world_model_path=tmp_path / "state" / "world.yaml",
        token_cache_path=tmp_path / "state" / "armed-token.json",
        require_fresh_observation=False,
        observation_ttl_seconds=30,
        shutdown_grace_seconds=0.2,
        emergency_stop_command=(),
    )
    gateway = FakeMotionGateway()
    server = RobotRuntimeServer(config, motion_gateway=gateway, motion_catalog=catalog)
    try:
        listed = server.dispatch({"command": "motion_catalog", "params": {}})
        armed = server.dispatch(
            {
                "command": "motion_arm",
                "params": {
                    "operation": "dock",
                    "target": "charger",
                    "operator_confirmation": "CONFIRM motion:dock:charger",
                },
            }
        )
        executed = server.dispatch(
            {
                "command": "motion_execute",
                "params": {
                    "operation": "dock",
                    "target": "charger",
                    "token": armed["data"]["token_secret"],
                },
            }
        )
    finally:
        server.server_close()
    assert listed["data"]["docks"] == ["charger"]
    assert armed["success"] is True
    assert executed["success"] is True
    assert executed["boundary"] == "standalone-robotd"


def test_ros_gateway_module_import_does_not_require_ros(monkeypatch):
    blocked = {"rclpy", "geometry_msgs", "nav2_msgs", "std_msgs"}
    for name in list(sys.modules):
        if name.split(".")[0] in blocked:
            monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("robot_agent.ros2_gateway")
    with pytest.raises(RuntimeError, match="requires rclpy"):
        module.Ros2MotionGateway()


def test_heartbeat_state_trips_once_and_recovers():
    state = HeartbeatState(timeout_seconds=0.5, last_heartbeat_at=10.0)
    assert state.check(10.5) == (False, False)
    assert state.check(10.51) == (True, True)
    assert state.check(11.0) == (True, False)
    state.heartbeat(11.0)
    assert state.check(11.1) == (False, False)
