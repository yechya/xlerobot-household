from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from robot_agent import contract
from robot_agent.audit import JsonlAuditWriter, audit_event_to_dict
from robot_agent.client import RobotClient
from robot_agent.config import RuntimeConfig
from robot_agent.episodes import EpisodeRecorder, load_episodes
from robot_agent.execution import ProcessExecutionSupervisor, WorkerResult
from robot_agent.robotd import RobotDaemon
from robot_agent.service import RobotRuntimeServer


class FakeAgentResult:
    def __init__(self, success: bool = True, message: str = "ok", data: dict | None = None):
        self.success = success
        self.message = message
        self.data = data or {}


class PipelineDryRunAgent:
    def __init__(self, calls: list[str]):
        self.calls = calls

    def execute_dry_run_pipeline(self, task_id: str):
        self.calls.append(task_id)
        return FakeAgentResult(
            message="pipeline dummy complete",
            data={"execution_mode": "pipeline_dummy", "task_id": task_id},
        )


class FakeAgent:
    def __init__(self, dry_run: bool, calls: list):
        self.dry_run = dry_run
        self.calls = calls

    def execute(self, task_id: str) -> FakeAgentResult:
        self.calls.append(("execute", self.dry_run, task_id))
        return FakeAgentResult(data={"task_id": task_id})

    def run_complex(self, goal: str) -> FakeAgentResult:
        self.calls.append(("run_complex", self.dry_run, goal))
        return FakeAgentResult(data={"goal": goal, "dry_run": True})

    def observe(self, instruction: str) -> FakeAgentResult:
        self.calls.append(("observe", self.dry_run, instruction))
        return FakeAgentResult(message="real observation", data={"instruction": instruction})

    def plan_goal(self, goal: str) -> list[dict]:
        return [{"id": "task_1", "skill": "pick_up_bottle", "goal": goal}]


def fake_agent_factory(calls: list):
    def factory(*, dry_run: bool) -> FakeAgent:
        calls.append(("factory", dry_run))
        return FakeAgent(dry_run, calls)

    return factory


class ImmediateExecutor:
    active = False

    def __init__(self):
        self.calls: list = []

    def execute(self, task_id: str, args: dict, token: str, timeout_seconds: int) -> WorkerResult:
        self.calls.append((task_id, args, timeout_seconds))
        return WorkerResult(True, "executed", {"task_id": task_id})

    def cancel(self, *, reason: str, emergency: bool = False) -> None:
        self.calls.append(("cancel", reason, emergency))


class BlockingExecutor:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled: tuple[str, bool] | None = None

    @property
    def active(self) -> bool:
        return self.started.is_set() and not self.release.is_set()

    def execute(self, task_id: str, args: dict, token: str, timeout_seconds: int) -> WorkerResult:
        self.started.set()
        self.release.wait(timeout=5)
        return WorkerResult(False, "cancelled")

    def cancel(self, *, reason: str, emergency: bool = False) -> None:
        self.cancelled = (reason, emergency)
        self.release.set()


def arm(daemon: RobotDaemon, task_id: str = "pick_up_bottle") -> str:
    result = daemon.request_arm(
        task_id,
        {},
        operator_confirmation=f"CONFIRM {task_id}",
    )
    assert result.success, result.message
    return result.data["token_secret"]


def make_config(tmp_path: Path, require_observation: bool = False) -> RuntimeConfig:
    return RuntimeConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        socket_path=tmp_path / "run" / "robotd.sock",
        token_store_path=tmp_path / "state" / "tokens.json",
        audit_log_path=tmp_path / "state" / "audit.jsonl",
        world_model_path=tmp_path / "state" / "world.yaml",
        token_cache_path=tmp_path / "state" / "armed-token.json",
        require_fresh_observation=require_observation,
        observation_ttl_seconds=30,
        shutdown_grace_seconds=0.2,
        emergency_stop_command=(),
    )


def test_runtime_config_has_no_claude_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOT_AGENT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ROBOT_AGENT_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("ROBOT_AGENT_MISSION_CATALOG", str(tmp_path / "missions.yaml"))
    monkeypatch.setenv("ROBOT_AGENT_SAFETY_POLICY", str(tmp_path / "safety.yaml"))
    monkeypatch.setenv("ROBOT_AGENT_SAFETY_SNAPSHOT", str(tmp_path / "safety.json"))
    monkeypatch.setenv(
        "ROBOT_AGENT_SEMANTIC_MAP_CONFIG",
        str(tmp_path / "semantic-map.yaml"),
    )
    config = RuntimeConfig.from_env()
    assert ".claude" not in str(config)
    assert config.socket_path == tmp_path / "run" / "robotd.sock"
    assert config.episode_dir == tmp_path / "state" / "episodes"
    assert config.require_policy_benchmark is True
    assert config.require_policy_benchmark_evidence is True
    assert config.mission_catalog_path == tmp_path / "missions.yaml"
    assert config.semantic_map_config_path == tmp_path / "semantic-map.yaml"
    assert config.require_safety_interlock is True
    assert config.safety_policy_path == tmp_path / "safety.yaml"
    assert config.safety_snapshot_path == tmp_path / "safety.json"
    assert config.experiment_mode == "rgb_pre_post_vla"
    assert config.dataset_recording_config_path is None


def test_runtime_config_rejects_unknown_experiment_mode(monkeypatch):
    monkeypatch.setenv("ROBOT_AGENT_EXPERIMENT_MODE", "unknown")
    with pytest.raises(ValueError, match="ROBOT_AGENT_EXPERIMENT_MODE"):
        RuntimeConfig.from_env()


def test_observation_is_real_and_required_before_arming(tmp_path):
    calls: list = []
    daemon = RobotDaemon(
        agent_factory=fake_agent_factory(calls),
        token_store_path=tmp_path / "tokens.json",
        real_executor=ImmediateExecutor(),
        require_fresh_observation=True,
        clock=lambda: 1000.0,
    )

    refused = daemon.request_arm(
        "pick_up_bottle",
        {},
        operator_confirmation="CONFIRM pick_up_bottle",
    )
    assert refused.refusal.reason_code == contract.OBSERVATION_REQUIRED

    observed = daemon.observe("check table")
    token = arm(daemon)
    assert observed.success is True
    assert observed.data["observe_only"] is True
    assert ("factory", False) in calls
    assert token.startswith("robotd-tok-")


def test_execute_consumes_token_and_returns_to_dry_run(tmp_path):
    executor = ImmediateExecutor()
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        real_executor=executor,
        session_id="test-session",
    )
    token = arm(daemon)

    result = daemon.execute("pick_up_bottle", {}, token=token)
    reused = daemon.execute("pick_up_bottle", {}, token=token)

    assert result.success is True
    assert result.state == contract.DRY_RUN
    assert daemon.status()["active_token_id"] is None
    assert reused.success is False
    assert reused.refusal.reason_code == contract.HARDWARE_NOT_ARMED
    assert executor.calls == [("pick_up_bottle", {}, 360)]


def test_real_execution_writes_hashed_episode_manifest(tmp_path):
    executor = ImmediateExecutor()
    recorder = EpisodeRecorder(tmp_path / "episodes")
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        real_executor=executor,
        episode_recorder=recorder,
        session_id="episode-session",
        clock=lambda: 1000.0,
    )
    token = arm(daemon)
    result = daemon.execute("pick_up_bottle", {}, token=token)
    manifests = load_episodes(tmp_path / "episodes")
    assert result.success is True
    assert result.data["episode_id"] == manifests[0].episode_id
    assert result.data["episode_manifest_sha256"] == manifests[0].manifest_sha256
    assert manifests[0].task_id == "pick_up_bottle"
    assert manifests[0].terminal_state == contract.DRY_RUN


def test_real_execution_records_experiment_mode_without_changing_binding(tmp_path):
    binding = {"policy_fingerprint": "a" * 64}
    recorder = EpisodeRecorder(tmp_path / "episodes")
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        real_executor=ImmediateExecutor(),
        episode_recorder=recorder,
        task_binding_resolver=lambda _task_id: dict(binding),
        experiment_mode="rgb_precondition_vla",
        session_id="experiment-session",
        clock=lambda: 1000.0,
    )
    token = arm(daemon)
    result = daemon.execute("pick_up_bottle", {}, token=token)
    manifest = load_episodes(tmp_path / "episodes")[0]
    assert result.success is True
    assert manifest.execution_binding["policy_fingerprint"] == binding["policy_fingerprint"]
    assert manifest.execution_binding["experiment_mode"] == "rgb_precondition_vla"
    assert daemon.active_execution_binding == {}


def test_policy_binding_is_token_bound_and_passed_to_executor(tmp_path):
    binding = {
        "policy_id": "pick-v1",
        "policy_fingerprint": "a" * 64,
    }
    executor = ImmediateExecutor()
    events = []
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        real_executor=executor,
        task_binding_resolver=lambda _task_id: dict(binding),
        audit_writer=events.append,
    )
    token = arm(daemon)
    assert daemon.status()["active_execution_binding"] == binding
    result = daemon.execute("pick_up_bottle", {}, token=token)
    assert result.success is True
    assert executor.calls == [("pick_up_bottle", binding, 360)]
    assert daemon.status()["active_execution_binding"] == {}
    assert any(event.details.get("execution_binding") == binding for event in events)


def test_policy_change_after_arm_invalidates_token(tmp_path):
    binding = {"policy_id": "pick-v1", "policy_fingerprint": "a" * 64}
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        real_executor=ImmediateExecutor(),
        task_binding_resolver=lambda _task_id: dict(binding),
    )
    token = arm(daemon)
    binding["policy_fingerprint"] = "b" * 64
    result = daemon.execute("pick_up_bottle", {}, token=token)
    assert result.success is False
    assert result.refusal.reason_code == contract.INVALID_TOKEN


def test_execute_rechecks_observation_freshness_without_consuming_token(tmp_path):
    now = [1000.0]
    calls: list = []
    executor = ImmediateExecutor()
    daemon = RobotDaemon(
        agent_factory=fake_agent_factory(calls),
        token_store_path=tmp_path / "tokens.json",
        real_executor=executor,
        require_fresh_observation=True,
        observation_ttl_seconds=30,
        clock=lambda: now[0],
    )
    daemon.observe("check table")
    token = arm(daemon)
    now[0] = 1031.0

    stale = daemon.execute("pick_up_bottle", {}, token=token)
    daemon.observe("check table again")
    executed = daemon.execute("pick_up_bottle", {}, token=token)

    assert stale.refusal.reason_code == contract.OBSERVATION_REQUIRED
    assert executed.success is True


def test_latched_estop_requires_explicit_reset_confirmation(tmp_path):
    daemon = RobotDaemon(token_store_path=tmp_path / "tokens.json")
    daemon.estop()
    refused = daemon.reset()
    reset = daemon.reset(operator_confirmation="RESET ESTOP")
    assert refused.success is False
    assert refused.state == contract.ESTOP
    assert reset.success is True
    assert reset.state == contract.DRY_RUN


def test_stop_cancels_active_execution_from_another_thread(tmp_path):
    executor = BlockingExecutor()
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        real_executor=executor,
    )
    token = arm(daemon)
    results: list = []
    worker = threading.Thread(
        target=lambda: results.append(daemon.execute("pick_up_bottle", {}, token=token))
    )
    worker.start()
    assert executor.started.wait(timeout=2)

    stopped = daemon.stop()
    worker.join(timeout=2)

    assert stopped.success is True
    assert stopped.state == contract.DISABLED
    assert executor.cancelled == ("operator_stop", False)
    assert not worker.is_alive()
    assert results[0].state == contract.DISABLED


def test_estop_is_latched_and_calls_emergency_cancel(tmp_path):
    executor = ImmediateExecutor()
    daemon = RobotDaemon(token_store_path=tmp_path / "tokens.json", real_executor=executor)
    result = daemon.estop()
    assert result.state == contract.ESTOP
    assert executor.calls == [("cancel", "emergency_stop", True)]


def test_jsonl_audit_is_private_and_redacts_tokens(tmp_path):
    events = []
    writer = JsonlAuditWriter(tmp_path / "audit.jsonl")
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        audit_writer=lambda event: (events.append(event), writer(event)),
    )
    token = arm(daemon)
    daemon.execute("pick_up_bottle", {}, token="wrong-token")

    audit_path = tmp_path / "audit.jsonl"
    content = audit_path.read_text(encoding="utf-8")
    assert token not in content
    assert "wrong-token" not in content
    assert oct(audit_path.stat().st_mode & 0o777) == "0o600"
    assert all("token_secret" not in json.dumps(audit_event_to_dict(event)) for event in events)


def test_unix_socket_service_supports_multiple_client_commands(tmp_path):
    config = make_config(tmp_path)
    server = RobotRuntimeServer(config)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    client = RobotClient(config.socket_path, timeout=2)
    try:
        status = client.call("status")
        dry_run = client.call("dry_run", goal="把瓶子放进筐里")
        missing = client.call("execute", task_id="pick_up_bottle", token=None)
        capabilities = client.call("capabilities")
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert status["success"] is True
    assert status["boundary"] == "standalone-robotd"
    assert dry_run["success"] is False
    assert dry_run["data"]["agent_data"]["execution_mode"] == "pipeline_dummy"
    assert missing["refusal"]["reason_code"] == contract.MISSING_TOKEN
    assert "pick_up_bottle" in capabilities["tasks"]
    assert not config.socket_path.exists()


def test_dry_run_can_use_real_task_pipeline_with_dummy_final_backend(tmp_path):
    calls: list[str] = []
    daemon = RobotDaemon(
        token_store_path=tmp_path / "tokens.json",
        dry_run_agent_factory=lambda: PipelineDryRunAgent(calls),
    )

    result = daemon.dry_run("pick_up_bottle")

    assert result.success is True
    assert calls == ["pick_up_bottle"]
    assert result.data["agent_data"]["execution_mode"] == "pipeline_dummy"


def test_second_daemon_cannot_bind_same_socket(tmp_path):
    config = make_config(tmp_path)
    first = RobotRuntimeServer(config)
    try:
        with pytest.raises(RuntimeError, match="already listening"):
            RobotRuntimeServer(config)
    finally:
        first.server_close()


def test_robotctl_cli_is_only_a_socket_client(tmp_path):
    config = make_config(tmp_path)
    server = RobotRuntimeServer(config)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "robot_agent.cli",
                "--socket",
                str(config.socket_path),
                "status",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["boundary"] == "standalone-robotd"
    assert payload["state"] == contract.DRY_RUN


def _sleeping_process(ready: multiprocessing.synchronize.Event) -> None:
    os.setsid()
    ready.set()
    time.sleep(30)


def _successful_spawn_worker(
    task_id: str,
    execution_binding: dict,
    token: str,
    world_path: str,
    policy_registry_path: str,
    task_catalog_path: str,
    connection,
) -> None:
    os.setsid()
    connection.send(
        {
            "success": True,
            "message": "spawned",
            "data": {
                "task_id": task_id,
                "world_path": world_path,
                "execution_binding": execution_binding,
                "policy_registry_path": policy_registry_path,
                "task_catalog_path": task_catalog_path,
            },
        }
    )
    connection.close()


def test_process_supervisor_cancel_terminates_process_group(tmp_path):
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    process = context.Process(target=_sleeping_process, args=(ready,))
    process.start()
    assert ready.wait(timeout=2)
    supervisor = ProcessExecutionSupervisor(
        world_model_path=tmp_path / "world.yaml",
        shutdown_grace_seconds=0.2,
    )
    supervisor._process = process

    supervisor.cancel(reason="test_stop")

    assert not process.is_alive()


def test_process_supervisor_executes_with_spawn_context(tmp_path):
    supervisor = ProcessExecutionSupervisor(
        world_model_path=tmp_path / "world.yaml",
        shutdown_grace_seconds=0.2,
        worker_target=_successful_spawn_worker,
    )
    result = supervisor.execute("pick_up_bottle", {}, "secret", 2)
    assert result.success is True
    assert result.data["task_id"] == "pick_up_bottle"
    assert supervisor.active is False
