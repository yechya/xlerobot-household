from __future__ import annotations

import threading
from pathlib import Path

import yaml

from robot_agent import contract
from robot_agent.agent import ExecutionResult
from robot_agent.episodes import EpisodeRecorder, load_episodes
from robot_agent.execution import WorkerResult
from robot_agent.missions import MissionCatalog
from robot_agent.motion import DockTarget, MotionCatalog, MotionResult, PoseTarget
from robot_agent.robotd import RobotDaemon
from robot_agent.config import RuntimeConfig
from robot_agent.cli import _build_parser, _request_for_args
from robot_agent.service import RobotRuntimeServer
from robot_agent.skill_flow import TaskSafetyMetadata
from robot_agent.table_docking import (
    DockingResult,
    TableCatalog,
    TableDockingConfig,
    TableProfile,
)


def _mission_catalog(tmp_path: Path, *, motion_only: bool = False) -> MissionCatalog:
    children = [
        {
            "id": "navigate",
            "type": "action",
            "action": "navigate",
            "target": "kitchen",
            "timeout_seconds": 60,
        },
        {
            "id": "dock",
            "type": "action",
            "action": "dock",
            "target": "table_dock",
            "timeout_seconds": 30,
        },
    ]
    if not motion_only:
        children.append(
            {
                "id": "pick",
                "type": "action",
                "action": "task",
                "target": "pick_remote",
                "timeout_seconds": 120,
            }
        )
    path = tmp_path / "missions.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "missions": {
                    "fetch_remote": {
                        "name": "Fetch remote",
                        "description": "Navigate, dock and pick the remote",
                        "aliases": ["拿遥控器"],
                        "requires_observation": False,
                        "max_runtime_seconds": 300,
                        "max_node_visits": 16,
                        "graph": {"id": "root", "type": "sequence", "children": children},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return MissionCatalog.load(path)


def _motion_catalog() -> MotionCatalog:
    return MotionCatalog(
        version=3,
        locations={"kitchen": PoseTarget("kitchen", "map", 1.0, 2.0, 0.0)},
        docks={"table_dock": DockTarget("table_dock", "dock-table")},
    )


class MissionGateway:
    def __init__(self, *, blocking: bool = False):
        self.calls = []
        self.blocking = blocking
        self.started = threading.Event()
        self.release = threading.Event()
        self._active = False

    @property
    def active(self):
        return self._active

    def health(self):
        return {"available": True, "active": self._active}

    def navigate(self, target, timeout_seconds):
        self.calls.append(("navigate", target.name))
        self._active = True
        self.started.set()
        if self.blocking:
            self.release.wait(timeout=3)
        self._active = False
        return MotionResult(not self.blocking, "arrived" if not self.blocking else "canceled")

    def dock(self, target, timeout_seconds):
        self.calls.append(("dock", target.name))
        return MotionResult(True, "docked")

    def cancel(self, *, reason, emergency=False):
        self.calls.append(("cancel", reason, emergency))
        self._active = False
        self.release.set()

    def close(self):
        return None


class MissionTaskExecutor:
    def __init__(self):
        self.calls = []

    @property
    def active(self):
        return False

    def execute(self, task_id, args, token, timeout_seconds):
        self.calls.append((task_id, dict(args), timeout_seconds))
        return WorkerResult(
            True,
            "picked",
            {
                "steps": [
                    {"action": "task_backend_execute", "success": True, "data": {}}
                ]
            },
        )

    def cancel(self, *, reason, emergency=False):
        self.calls.append(("cancel", reason, emergency))


def _metadata(task_id: str):
    if task_id != "pick_remote":
        return None
    return TaskSafetyMetadata(
        task_id=task_id,
        name="Pick remote",
        aliases=("pick remote",),
        args={},
        requires_observation=False,
        requires_token=True,
        max_runtime_seconds=180,
        allowed_modes=tuple(contract.RUNTIME_STATES),
        stop_conditions=("operator_stop_requested",),
    )


def test_one_token_runs_mobile_manipulation_and_records_each_level(tmp_path: Path):
    gateway = MissionGateway()
    executor = MissionTaskExecutor()
    binding = {
        "task_fingerprint": "a" * 64,
        "task_executor": "learned_policy",
        "policy_fingerprint": "b" * 64,
    }
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        motion_gateway=gateway,
        motion_catalog=_motion_catalog(),
        real_executor=executor,
        mission_catalog=_mission_catalog(tmp_path),
        task_metadata_resolver=_metadata,
        task_binding_resolver=lambda _task_id: dict(binding),
        episode_recorder=EpisodeRecorder(tmp_path / "episodes"),
    )
    armed = daemon.request_mission_arm(
        "fetch_remote",
        operator_confirmation="CONFIRM mission:fetch_remote",
    )
    result = daemon.execute_mission(
        "fetch_remote",
        token=armed.data["token_secret"],
    )
    assert result.success is True
    assert gateway.calls == [("navigate", "kitchen"), ("dock", "table_dock")]
    assert executor.calls[0][0] == "pick_remote"
    assert daemon.state == contract.DRY_RUN
    manifests = load_episodes(tmp_path / "episodes")
    assert len(manifests) == 4
    assert sorted(manifest.executor for manifest in manifests) == [
        "learned_policy",
        "mission_graph",
        "named_motion",
        "named_motion",
    ]


def test_child_binding_change_after_arm_invalidates_mission_token(tmp_path: Path):
    binding = {
        "task_fingerprint": "a" * 64,
        "task_executor": "learned_policy",
        "policy_fingerprint": "b" * 64,
    }
    gateway = MissionGateway()
    executor = MissionTaskExecutor()
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        motion_gateway=gateway,
        motion_catalog=_motion_catalog(),
        real_executor=executor,
        mission_catalog=_mission_catalog(tmp_path),
        task_metadata_resolver=_metadata,
        task_binding_resolver=lambda _task_id: dict(binding),
    )
    armed = daemon.request_mission_arm(
        "fetch_remote",
        operator_confirmation="CONFIRM mission:fetch_remote",
    )
    binding["policy_fingerprint"] = "c" * 64
    result = daemon.execute_mission("fetch_remote", token=armed.data["token_secret"])
    assert result.success is False
    assert result.refusal.reason_code == contract.INVALID_TOKEN
    assert gateway.calls == []
    assert executor.calls == []


def test_stop_during_navigation_prevents_later_mission_nodes(tmp_path: Path):
    gateway = MissionGateway(blocking=True)
    executor = MissionTaskExecutor()
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        motion_gateway=gateway,
        motion_catalog=_motion_catalog(),
        real_executor=executor,
        mission_catalog=_mission_catalog(tmp_path),
        task_metadata_resolver=_metadata,
        task_binding_resolver=lambda _task_id: {
            "task_fingerprint": "a" * 64,
            "task_executor": "learned_policy",
        },
    )
    armed = daemon.request_mission_arm(
        "fetch_remote",
        operator_confirmation="CONFIRM mission:fetch_remote",
    )
    results = []
    worker = threading.Thread(
        target=lambda: results.append(
            daemon.execute_mission("fetch_remote", token=armed.data["token_secret"])
        )
    )
    worker.start()
    assert gateway.started.wait(timeout=1)
    daemon.stop()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert results[0].success is False
    assert results[0].state == contract.DISABLED
    assert executor.calls == [("cancel", "operator_stop", False)]
    assert not any(call[0] == "dock" for call in gateway.calls)


def test_robotd_service_exposes_mission_catalog_alias_and_one_token_flow(tmp_path: Path):
    _mission_catalog(tmp_path, motion_only=True)
    gateway = MissionGateway()
    config = RuntimeConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        socket_path=tmp_path / "run" / "robotd.sock",
        token_store_path=tmp_path / "state" / "tokens.json",
        audit_log_path=tmp_path / "state" / "audit.jsonl",
        world_model_path=tmp_path / "state" / "world.yaml",
        token_cache_path=tmp_path / "state" / "token.json",
        require_fresh_observation=False,
        observation_ttl_seconds=30,
        shutdown_grace_seconds=0.2,
        emergency_stop_command=(),
        mission_catalog_path=tmp_path / "missions.yaml",
    )
    server = RobotRuntimeServer(
        config,
        motion_gateway=gateway,
        motion_catalog=_motion_catalog(),
    )
    try:
        capabilities = server.dispatch({"command": "capabilities", "params": {}})
        planned = server.dispatch(
            {"command": "plan", "params": {"goal": "请帮我拿遥控器"}}
        )
        dry_run = server.dispatch(
            {
                "command": "mission_dry_run",
                "params": {"mission_id": "fetch_remote"},
            }
        )
        armed = server.dispatch(
            {
                "command": "mission_arm",
                "params": {
                    "mission_id": "fetch_remote",
                    "operator_confirmation": "CONFIRM mission:fetch_remote",
                },
            }
        )
        executed = server.dispatch(
            {
                "command": "mission_execute",
                "params": {
                    "mission_id": "fetch_remote",
                    "token": armed["data"]["token_secret"],
                },
            }
        )
    finally:
        server.server_close()
    assert "fetch_remote" in capabilities["mission_catalog"]["missions"]
    assert planned["data"]["mission_id"] == "fetch_remote"
    assert dry_run["success"] is True
    assert armed["success"] is True
    assert executed["success"] is True


def test_robotctl_mission_commands_use_mission_scoped_token_cache(tmp_path: Path):
    config = RuntimeConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        socket_path=tmp_path / "run" / "robotd.sock",
        token_store_path=tmp_path / "state" / "tokens.json",
        audit_log_path=tmp_path / "state" / "audit.jsonl",
        world_model_path=tmp_path / "state" / "world.yaml",
        token_cache_path=tmp_path / "state" / "token.json",
        require_fresh_observation=False,
        observation_ttl_seconds=30,
        shutdown_grace_seconds=0.2,
        emergency_stop_command=(),
    )
    parser = _build_parser()
    arm_args = parser.parse_args(
        [
            "arm-mission",
            "fetch_remote",
            "--confirm",
            "CONFIRM mission:fetch_remote",
        ]
    )
    command, params = _request_for_args(arm_args, config)
    assert command == "mission_arm"
    assert params["mission_id"] == "fetch_remote"

    config.token_cache_path.parent.mkdir(parents=True)
    config.token_cache_path.write_text(
        '{"task_id":"mission:fetch_remote","token":"mission-token"}',
        encoding="utf-8",
    )
    config.token_cache_path.chmod(0o600)
    run_args = parser.parse_args(["run-mission", "fetch_remote", "--real"])
    command, params = _request_for_args(run_args, config)
    assert command == "mission_execute"
    assert params["token"] == "mission-token"


def _table_mission_catalog(tmp_path: Path) -> MissionCatalog:
    path = tmp_path / "table-missions.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "missions": {
                    "approach_cabinet": {
                        "name": "Approach cabinet",
                        "description": "Visually dock to the cabinet edge",
                        "aliases": ["靠柜子"],
                        "requires_observation": False,
                        "max_runtime_seconds": 180,
                        "max_node_visits": 8,
                        "graph": {
                            "id": "root",
                            "type": "sequence",
                            "children": [
                                {
                                    "id": "dock",
                                    "type": "action",
                                    "action": "dock_table",
                                    "target": "cabinet_edge",
                                    "timeout_seconds": 90,
                                }
                            ],
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return MissionCatalog.load(path)


def _table_catalog(*, standoff_m: float = 0.45) -> TableCatalog:
    return TableCatalog(
        version=1,
        tables={
            "cabinet_edge": TableProfile(
                "cabinet_edge",
                TableDockingConfig(standoff_m=standoff_m),
            )
        },
    )


def test_table_docking_mission_action_runs_and_records_episode(tmp_path: Path):
    calls = []

    def runner(profile, remaining_seconds):
        calls.append((profile.name, remaining_seconds))
        return DockingResult(True, "docked at edge", {"observations": []})

    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        motion_gateway=MissionGateway(),
        motion_catalog=_motion_catalog(),
        mission_catalog=_table_mission_catalog(tmp_path),
        table_catalog=_table_catalog(),
        table_docking_runner=runner,
        task_metadata_resolver=_metadata,
        episode_recorder=EpisodeRecorder(tmp_path / "episodes"),
    )
    armed = daemon.request_mission_arm(
        "approach_cabinet",
        operator_confirmation="CONFIRM mission:approach_cabinet",
    )
    assert armed.success is True
    result = daemon.execute_mission(
        "approach_cabinet",
        token=armed.data["token_secret"],
    )
    assert result.success is True
    assert calls and calls[0][0] == "cabinet_edge"
    manifests = load_episodes(tmp_path / "episodes")
    assert sorted(manifest.executor for manifest in manifests) == [
        "mission_graph",
        "table_docking",
    ]


def test_table_profile_change_after_arm_invalidates_mission_token(tmp_path: Path):
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        motion_gateway=MissionGateway(),
        motion_catalog=_motion_catalog(),
        mission_catalog=_table_mission_catalog(tmp_path),
        table_catalog=_table_catalog(standoff_m=0.45),
        table_docking_runner=lambda profile, remaining: DockingResult(
            True, "docked"
        ),
        task_metadata_resolver=_metadata,
    )
    armed = daemon.request_mission_arm(
        "approach_cabinet",
        operator_confirmation="CONFIRM mission:approach_cabinet",
    )
    assert armed.success is True
    daemon._table_catalog = _table_catalog(standoff_m=0.55)
    result = daemon.execute_mission(
        "approach_cabinet",
        token=armed.data["token_secret"],
    )
    assert result.success is False
    assert result.state == contract.ARMED_REAL


def _retry_recovery_catalog(tmp_path: Path) -> MissionCatalog:
    path = tmp_path / "retry-missions.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "missions": {
                    "fold_with_recovery": {
                        "name": "Fold with recovery",
                        "description": "Retry the fold once after a fresh observation",
                        "aliases": ["叠布并恢复"],
                        "requires_observation": False,
                        "max_runtime_seconds": 600,
                        "max_node_visits": 16,
                        "graph": {
                            "id": "root",
                            "type": "retry",
                            "max_attempts": 2,
                            "recovery": {
                                "action": "observe",
                                "target": "workspace_check",
                                "timeout_seconds": 45,
                            },
                            "child": {
                                "id": "fold",
                                "type": "action",
                                "action": "task",
                                "target": "fold_simple",
                                "timeout_seconds": 360,
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return MissionCatalog.load(path)


def test_retry_runs_recovery_observation_between_task_attempts(tmp_path: Path):
    events: list[tuple[str, str]] = []

    class FlakyExecutor:
        def __init__(self):
            self.attempts = 0

        @property
        def active(self):
            return False

        def execute(self, task_id, args, token, timeout_seconds):
            self.attempts += 1
            events.append(("task", f"{task_id}#{self.attempts}"))
            success = self.attempts >= 2
            return WorkerResult(success, "folded" if success else "gripper missed")

        def cancel(self, *, reason, emergency=False):
            events.append(("cancel", reason))

    class FakeAgent:
        def observe(self, instruction):
            events.append(("observe", instruction[:40]))
            return ExecutionResult(True, "workspace safe", {"observe_only": True})

    def metadata(task_id: str):
        if task_id != "fold_simple":
            return None
        return TaskSafetyMetadata(
            task_id=task_id,
            name="Fold",
            aliases=("fold cloth",),
            args={},
            requires_observation=False,
            requires_token=True,
            max_runtime_seconds=360,
            allowed_modes=tuple(contract.RUNTIME_STATES),
            stop_conditions=("operator_stop_requested",),
        )

    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        motion_gateway=MissionGateway(),
        motion_catalog=_motion_catalog(),
        real_executor=FlakyExecutor(),
        mission_catalog=_retry_recovery_catalog(tmp_path),
        task_metadata_resolver=metadata,
        task_binding_resolver=lambda _task_id: {
            "task_fingerprint": "a" * 64,
            "task_executor": "learned_policy",
        },
        agent_factory=lambda *, dry_run: FakeAgent(),
    )
    armed = daemon.request_mission_arm(
        "fold_with_recovery",
        operator_confirmation="CONFIRM mission:fold_with_recovery",
    )
    assert armed.success is True
    result = daemon.execute_mission(
        "fold_with_recovery",
        token=armed.data["token_secret"],
    )
    assert result.success is True
    assert [event[0] for event in events] == ["task", "observe", "task"]
    assert events[0][1] == "fold_simple#1"
    assert events[2][1] == "fold_simple#2"
    assert "Recovery observation" in events[1][1]
    trace_types = [step["type"] for step in result.data["steps"]]
    assert trace_types == ["action", "recovery", "action", "retry"]
