from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from robot_agent.camera_mapping import (
    CameraMappingConfig,
    CameraMappingError,
    CameraMappingManager,
)


def _write_policy(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "robot": {
                    "type": "bi_so100_follower",
                    "id": "bimanual_follower0",
                    "left_arm_port": "/dev/ttyACM0",
                    "right_arm_port": "/dev/ttyACM1",
                    "dataset_owner": "usrname1",
                    "cameras": {
                        "head_cam": {"type": "opencv", "device": 2, "width": 640, "height": 480, "fps": 15},
                        "left_cam": {"type": "opencv", "device": 4, "width": 640, "height": 480, "fps": 15},
                        "right_cam": {"type": "opencv", "device": 0, "width": 640, "height": 480, "fps": 15},
                    },
                },
                "task_defaults": {},
                "policies": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_env(path: Path) -> None:
    path.write_text(
        "ROBOT_AGENT_HEAD_CAMERA_DEVICE=/dev/video2\n"
        "ROBOT_AGENT_LEFT_CAMERA_DEVICE=/dev/video4\n"
        "ROBOT_AGENT_RIGHT_CAMERA_DEVICE=/dev/video0\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[CameraMappingManager, list[list[str]]]:
    env_file = tmp_path / "robot-agent.env"
    policy = tmp_path / "policies.yaml"
    _write_env(env_file)
    _write_policy(policy)
    for index in (0, 2, 4, 6):
        device = tmp_path / f"video{index}"
        device.touch()
        monkeypatch.setattr(Path, "exists", lambda self, _old=Path.exists: True if str(self).startswith("/dev/video") else _old(self))
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    manager = CameraMappingManager(
        CameraMappingConfig(
            env_file=env_file,
            policy_registry_path=policy,
            restart_camera_command=("systemctl", "--user", "restart", "robot-camera.service"),
            restart_robotd_command=("systemctl", "--user", "restart", "robotd-software.service"),
        ),
        command_runner=runner,
    )
    return manager, calls


def test_camera_mapping_updates_env_and_policy_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager, calls = _manager(tmp_path, monkeypatch)

    result = manager.apply(
        {"head_cam": "/dev/video6", "left_cam": "/dev/video2", "right_cam": "/dev/video4"},
        operator_confirmation="CONFIRM CAMERA MAPPING",
        restart_services=True,
    )

    assert result["success"] is True
    assert calls == [
        ["systemctl", "--user", "restart", "robot-camera.service"],
        ["systemctl", "--user", "restart", "robotd-software.service"],
    ]
    env_content = manager.config.env_file.read_text(encoding="utf-8")
    assert "ROBOT_AGENT_HEAD_CAMERA_DEVICE=/dev/video6" in env_content
    assert "ROBOT_AGENT_LEFT_CAMERA_DEVICE=/dev/video2" in env_content
    policy = yaml.safe_load(manager.config.policy_registry_path.read_text(encoding="utf-8"))
    assert policy["robot"]["cameras"]["head_cam"]["device"] == 6
    assert policy["robot"]["cameras"]["left_cam"]["device"] == 2
    assert policy["robot"]["cameras"]["right_cam"]["device"] == 4


def test_camera_mapping_rejects_duplicate_or_unsafe_devices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager, _calls = _manager(tmp_path, monkeypatch)

    with pytest.raises(CameraMappingError, match="unique"):
        manager.apply(
            {"head_cam": "/dev/video2", "left_cam": "/dev/video2", "right_cam": "/dev/video4"},
            operator_confirmation="CONFIRM CAMERA MAPPING",
            restart_services=False,
        )
    with pytest.raises(CameraMappingError, match="/dev/videoN"):
        manager.apply(
            {"head_cam": "/tmp/video2", "left_cam": "/dev/video4", "right_cam": "/dev/video0"},
            operator_confirmation="CONFIRM CAMERA MAPPING",
            restart_services=False,
        )


def test_camera_mapping_requires_exact_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manager, _calls = _manager(tmp_path, monkeypatch)

    with pytest.raises(CameraMappingError, match="CONFIRM CAMERA MAPPING"):
        manager.apply(
            {"head_cam": "/dev/video2", "left_cam": "/dev/video4", "right_cam": "/dev/video0"},
            operator_confirmation="wrong",
            restart_services=False,
        )
