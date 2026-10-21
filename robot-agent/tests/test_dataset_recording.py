from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

from robot_agent.config import RuntimeConfig
from robot_agent.dataset_recording import (
    DatasetRecordingConfig,
    DatasetRecordingError,
    DatasetRecordingSupervisor,
)
from robot_agent.policies import CameraSpec, PolicyRegistry, RobotSpec
from robot_agent.service import RobotRuntimeServer


class FakeProcess:
    def __init__(self, pid: int = 4242):
        self.pid = pid
        self.returncode: int | None = None
        self._finished = threading.Event()

    def wait(self) -> int:
        self._finished.wait(timeout=5)
        return -1 if self.returncode is None else self.returncode

    def finish(self, returncode: int) -> None:
        self.returncode = returncode
        self._finished.set()


class FakePopenFactory:
    def __init__(self):
        self.calls: list[tuple[list[str], dict]] = []
        self.process = FakeProcess()

    def __call__(self, command: list[str], **kwargs):
        self.calls.append((list(command), dict(kwargs)))
        kwargs["stdout"].write(b"fake lerobot recorder started\n")
        kwargs["stdout"].flush()
        return self.process


class FakeCameraRunner:
    def __init__(self, *, resume_returncode: int = 0):
        self.calls: list[list[str]] = []
        self.resume_returncode = resume_returncode

    def __call__(self, command: list[str], **_kwargs):
        self.calls.append(list(command))
        returncode = self.resume_returncode if command[-2] == "start" else 0
        return subprocess.CompletedProcess(command, returncode, "", "camera command failed")


class BlockingPauseCameraRunner(FakeCameraRunner):
    def __init__(self):
        super().__init__()
        self.pause_entered = threading.Event()
        self.pause_release = threading.Event()

    def __call__(self, command: list[str], **kwargs):
        if command[-2] == "stop":
            self.pause_entered.set()
            self.pause_release.wait(timeout=2)
        return super().__call__(command, **kwargs)


def _make_devices(tmp_path: Path) -> dict[str, Path]:
    devices = {
        "follower_left": tmp_path / "ttyACM0",
        "follower_right": tmp_path / "ttyACM1",
        "leader_left": tmp_path / "ttyACM2",
        "leader_right": tmp_path / "ttyACM3",
        "head_cam": tmp_path / "video2",
        "left_cam": tmp_path / "video4",
        "right_cam": tmp_path / "video0",
    }
    for path in devices.values():
        path.touch()
    return devices


def _make_registry(devices: dict[str, Path]) -> PolicyRegistry:
    return PolicyRegistry(
        version=1,
        robot=RobotSpec(
            robot_type="bi_so100_follower",
            robot_id="bimanual_follower0",
            left_arm_port=str(devices["follower_left"]),
            right_arm_port=str(devices["follower_right"]),
            dataset_owner="usrname1",
            cameras=(
                CameraSpec("head_cam", "opencv", str(devices["head_cam"]), 640, 480, 15),
                CameraSpec("left_cam", "opencv", str(devices["left_cam"]), 640, 480, 15),
                CameraSpec("right_cam", "opencv", str(devices["right_cam"]), 640, 480, 15),
            ),
        ),
        default_policies={},
        policies={},
    )


def _make_config(tmp_path: Path, devices: dict[str, Path]) -> DatasetRecordingConfig:
    executable = tmp_path / "bin" / "lerobot-record"
    executable.parent.mkdir()
    executable.touch(mode=0o700)
    executable.chmod(0o700)
    workspace = tmp_path / "lerobot"
    workspace.mkdir()
    return DatasetRecordingConfig(
        enabled=True,
        executable=executable,
        workspace=workspace,
        storage_root=tmp_path / "datasets",
        teleoperator_type="bi_so100_leader",
        teleoperator_id="bimanual_leader0",
        left_arm_port=str(devices["leader_left"]),
        right_arm_port=str(devices["leader_right"]),
        fps=30,
        default_num_episodes=10,
        default_episode_time_s=30,
        default_reset_time_s=15,
        max_num_episodes=100,
        max_episode_time_s=600,
        max_reset_time_s=300,
        camera_pause_command=("systemctl", "--user", "stop", "robot-camera.service"),
        camera_resume_command=("systemctl", "--user", "start", "robot-camera.service"),
        camera_command_timeout_seconds=2,
        stop_grace_seconds=0.2,
        log_tail_lines=20,
    )


def _request(*, resume: bool = False) -> dict:
    return {
        "dataset_name": "clear_table_v2",
        "task": "clear the table with both arms",
        "num_episodes": 5,
        "episode_time_s": 20,
        "reset_time_s": 8,
        "resume": resume,
    }


def _write_info(
    dataset_root: Path,
    *,
    episodes: int = 3,
    frames: int = 900,
    fps: int = 30,
) -> None:
    info = {
        "robot_type": "bi_so100_follower",
        "total_episodes": episodes,
        "total_frames": frames,
        "fps": fps,
        "features": {
            f"observation.images.{name}": {"shape": [480, 640, 3]}
            for name in ("head_cam", "left_cam", "right_cam")
        },
    }
    metadata = dataset_root / "meta"
    metadata.mkdir(parents=True)
    (metadata / "info.json").write_text(json.dumps(info), encoding="utf-8")


def _wait_for_state(supervisor: DatasetRecordingSupervisor, state: str) -> dict:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = supervisor.status()["data"]
        if status["state"] == state:
            return status
        time.sleep(0.01)
    raise AssertionError(f"recording did not reach state {state!r}")


def test_config_loader_rejects_unknown_fields(tmp_path: Path):
    raw = yaml.safe_load(
        (Path(__file__).parents[1] / "config" / "dataset-recording.example.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["unexpected"] = True
    path = tmp_path / "recording.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(DatasetRecordingError, match="unknown fields"):
        DatasetRecordingConfig.load(path)


def test_plan_builds_exact_structured_lerobot_command(tmp_path: Path):
    devices = _make_devices(tmp_path)
    config = _make_config(tmp_path, devices)
    registry = _make_registry(devices)
    supervisor = DatasetRecordingSupervisor(
        config=config,
        policy_registry=registry,
        state_dir=tmp_path / "state",
    )

    plan = supervisor.build_plan(_request(), check_devices=True)

    assert plan.repo_id == "usrname1/clear_table_v2"
    assert plan.dataset_root == config.storage_root / "usrname1" / "clear_table_v2"
    assert plan.command[0] == str(config.executable)
    assert f"--robot.left_arm_port={devices['follower_left']}" in plan.command
    assert f"--teleop.right_arm_port={devices['leader_right']}" in plan.command
    assert "--dataset.fps=30" in plan.command
    assert "--dataset.num_episodes=5" in plan.command
    assert "--resume=false" in plan.command
    camera_arg = next(value for value in plan.command if value.startswith("--robot.cameras="))
    cameras = json.loads(camera_arg.split("=", 1)[1])
    assert set(cameras) == {"head_cam", "left_cam", "right_cam"}
    assert cameras["head_cam"]["index_or_path"] == str(devices["head_cam"])


def test_new_recording_never_overwrites_and_resume_validates_metadata(tmp_path: Path):
    devices = _make_devices(tmp_path)
    config = _make_config(tmp_path, devices)
    registry = _make_registry(devices)
    supervisor = DatasetRecordingSupervisor(
        config=config,
        policy_registry=registry,
        state_dir=tmp_path / "state",
    )
    dataset_root = config.storage_root / "usrname1" / "clear_table_v2"
    _write_info(dataset_root)

    with pytest.raises(DatasetRecordingError, match="already exists"):
        supervisor.build_plan(_request(), check_devices=True)

    resumed = supervisor.build_plan(_request(resume=True), check_devices=True)
    assert resumed.initial_episodes == 3
    assert resumed.initial_frames == 900
    assert "--resume=true" in resumed.command

    (dataset_root / "meta" / "info.json").unlink()
    with pytest.raises(DatasetRecordingError, match="metadata is missing"):
        supervisor.build_plan(_request(resume=True), check_devices=True)


def test_validation_rejects_serial_device_owned_by_another_process(tmp_path: Path):
    devices = _make_devices(tmp_path)
    supervisor = DatasetRecordingSupervisor(
        config=_make_config(tmp_path, devices),
        policy_registry=_make_registry(devices),
        state_dir=tmp_path / "state",
    )

    with devices["leader_left"].open("rb"):
        result = supervisor.validate(_request())

    assert result["success"] is False
    assert "recording devices are already in use" in result["message"]
    assert str(devices["leader_left"]) in result["message"]
    assert f"pid {os.getpid()}" in result["message"]


def test_start_completion_reports_progress_and_restores_camera(tmp_path: Path):
    devices = _make_devices(tmp_path)
    popen = FakePopenFactory()
    cameras = FakeCameraRunner()
    supervisor = DatasetRecordingSupervisor(
        config=_make_config(tmp_path, devices),
        policy_registry=_make_registry(devices),
        state_dir=tmp_path / "state",
        popen_factory=popen,
        command_runner=cameras,
    )

    started = supervisor.start(
        _request(),
        operator_confirmation="CONFIRM RECORD DATASET",
    )
    assert started["success"] is True
    assert started["data"]["active"] is True
    assert cameras.calls[0][-2:] == ["stop", "robot-camera.service"]
    assert popen.calls[0][1]["start_new_session"] is True
    assert "shell" not in popen.calls[0][1]

    dataset_root = Path(started["data"]["dataset_path"])
    _write_info(dataset_root, episodes=2, frames=600)
    running = supervisor.status()["data"]
    assert running["recorded_episodes"] == 2
    assert running["progress_percent"] == 40.0
    assert "fake lerobot recorder started" in running["log_tail"]

    popen.process.finish(0)
    completed = _wait_for_state(supervisor, "completed")
    assert completed["returncode"] == 0
    assert completed["camera_service"]["restored"] is True
    assert cameras.calls[-1][-2:] == ["start", "robot-camera.service"]


def test_stop_terminates_process_group_and_restores_camera(tmp_path: Path):
    devices = _make_devices(tmp_path)
    popen = FakePopenFactory()
    cameras = FakeCameraRunner()
    signals: list[tuple[int, int]] = []

    def signal_process_group(pid: int, process_signal: int) -> None:
        signals.append((pid, process_signal))
        popen.process.finish(-process_signal)

    supervisor = DatasetRecordingSupervisor(
        config=_make_config(tmp_path, devices),
        policy_registry=_make_registry(devices),
        state_dir=tmp_path / "state",
        popen_factory=popen,
        command_runner=cameras,
        process_group_signaler=signal_process_group,
    )
    supervisor.start(_request(), operator_confirmation="CONFIRM RECORD DATASET")

    stopped = supervisor.stop(reason="test_stop")

    assert stopped["success"] is True
    assert stopped["data"]["state"] == "stopped"
    assert signals == [(popen.process.pid, signal.SIGTERM)]
    assert cameras.calls[-1][-2:] == ["start", "robot-camera.service"]


def test_stop_cancels_start_before_lerobot_process_launch(tmp_path: Path):
    devices = _make_devices(tmp_path)
    popen = FakePopenFactory()
    cameras = BlockingPauseCameraRunner()
    supervisor = DatasetRecordingSupervisor(
        config=_make_config(tmp_path, devices),
        policy_registry=_make_registry(devices),
        state_dir=tmp_path / "state",
        popen_factory=popen,
        command_runner=cameras,
    )
    results: dict[str, dict] = {}
    start_thread = threading.Thread(
        target=lambda: results.setdefault(
            "start",
            supervisor.start(_request(), operator_confirmation="CONFIRM RECORD DATASET"),
        )
    )
    start_thread.start()
    assert cameras.pause_entered.wait(timeout=1)
    stop_thread = threading.Thread(
        target=lambda: results.setdefault("stop", supervisor.stop(reason="race_stop"))
    )
    stop_thread.start()
    cameras.pause_release.set()
    start_thread.join(timeout=2)
    stop_thread.join(timeout=2)

    assert results["start"]["success"] is False
    assert results["stop"]["success"] is True
    assert popen.calls == []
    assert [command[-2] for command in cameras.calls] == ["stop", "start"]


def test_recorder_failure_restores_camera_and_surfaces_exit_code(tmp_path: Path):
    devices = _make_devices(tmp_path)
    popen = FakePopenFactory()
    cameras = FakeCameraRunner()
    supervisor = DatasetRecordingSupervisor(
        config=_make_config(tmp_path, devices),
        policy_registry=_make_registry(devices),
        state_dir=tmp_path / "state",
        popen_factory=popen,
        command_runner=cameras,
    )
    supervisor.start(_request(), operator_confirmation="CONFIRM RECORD DATASET")

    popen.process.finish(7)
    failed = _wait_for_state(supervisor, "failed")

    assert failed["returncode"] == 7
    assert failed["camera_service"]["restored"] is True
    assert cameras.calls[-1][-2:] == ["start", "robot-camera.service"]


class FakeRecordingSupervisor:
    def __init__(self):
        self.active = False
        self.calls: list[tuple[str, object]] = []

    def capabilities(self) -> dict:
        return {"configured": True, "enabled": True, "available": True}

    def status(self) -> dict:
        return {
            "success": True,
            "message": "recording status",
            "data": {"active": self.active, "state": "recording" if self.active else "idle"},
        }

    def validate(self, params: dict) -> dict:
        self.calls.append(("validate", dict(params)))
        return {"success": True, "message": "valid", "data": dict(params)}

    def start(self, params: dict, *, operator_confirmation: str) -> dict:
        self.calls.append(("start", (dict(params), operator_confirmation)))
        self.active = True
        return {"success": True, "message": "started", "data": {"active": True}}

    def stop(self, *, reason: str) -> dict:
        self.calls.append(("stop", reason))
        self.active = False
        return {"success": True, "message": "stopped", "data": {"active": False}}

    def close(self) -> None:
        self.calls.append(("close", None))


def _runtime_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
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


def test_runtime_dispatches_recording_and_blocks_tasks_while_active(tmp_path: Path):
    recording = FakeRecordingSupervisor()
    server = RobotRuntimeServer(
        _runtime_config(tmp_path),
        dataset_recording_supervisor=recording,
    )
    try:
        started = server.dispatch(
            {
                "id": "start",
                "command": "dataset_recording_start",
                "params": {
                    **_request(),
                    "operator_confirmation": "CONFIRM RECORD DATASET",
                },
            }
        )
        blocked = server.dispatch(
            {"id": "dry", "command": "dry_run", "params": {"goal": "fold_simple"}}
        )
        status = server.dispatch(
            {"id": "status", "command": "dataset_recording_status", "params": {}}
        )
        stopped = server.dispatch(
            {"id": "stop", "command": "dataset_recording_stop", "params": {}}
        )

        assert started["success"] is True
        assert recording.calls[0][0] == "start"
        assert blocked["success"] is False
        assert "dataset recording is active" in blocked["message"]
        assert status["data"]["active"] is True
        assert stopped["success"] is True
    finally:
        server.server_close()
