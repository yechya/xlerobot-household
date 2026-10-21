from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_software_deployment_script_creates_dry_run_profile(tmp_path: Path):
    repository = Path(__file__).parents[1]
    script = repository / "deploy" / "install-software.sh"
    prefix = tmp_path / "prefix"
    config_dir = tmp_path / "config"
    state_dir = tmp_path / "state"
    runtime_dir = tmp_path / "runtime"

    result = subprocess.run(
        [
            str(script),
            "--skip-install",
            "--no-service",
            "--prefix",
            str(prefix),
            "--config-dir",
            str(config_dir),
            "--state-dir",
            str(state_dir),
            "--runtime-dir",
            str(runtime_dir),
        ],
        cwd=repository,
        env={**os.environ, "HOME": str(tmp_path / "home")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    env_file = config_dir / "robot-agent.env"
    content = env_file.read_text(encoding="utf-8")
    assert "ROBOT_AGENT_MOTION_BACKEND=disabled" in content
    assert "ROBOT_AGENT_REQUIRE_POLICY_BENCHMARK=false" in content
    assert "ROBOT_AGENT_REQUIRE_SAFETY_INTERLOCK=false" in content
    assert "ROBOT_AGENT_DATASET_RECORDING_CONFIG=" in content
    assert oct(env_file.stat().st_mode & 0o777) == "0o600"
    assert state_dir.is_dir()
    assert runtime_dir.is_dir()


def test_robotd_user_unit_allows_default_lerobot_dataset_storage():
    repository = Path(__file__).parents[1]
    unit = (repository / "deploy" / "robotd-software.service.in").read_text(
        encoding="utf-8"
    )

    assert "%h/.cache/huggingface/lerobot" in unit


def test_production_unit_allows_only_confirmed_rgb_devices():
    repository = Path(__file__).parents[1]
    unit = (repository / "deploy" / "robotd.service").read_text(encoding="utf-8")
    camera_rules = {
        line
        for line in unit.splitlines()
        if line.startswith("DeviceAllow=/dev/video")
    }

    assert camera_rules == {
        "DeviceAllow=/dev/video0 rw",
        "DeviceAllow=/dev/video2 rw",
        "DeviceAllow=/dev/video4 rw",
    }


def test_software_deployment_script_rejects_enable_without_service(tmp_path: Path):
    repository = Path(__file__).parents[1]
    result = subprocess.run(
        [
            str(repository / "deploy" / "install-software.sh"),
            "--enable",
            "--no-service",
            "--skip-install",
            "--prefix",
            str(tmp_path / "prefix"),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "cannot be combined" in result.stderr


def test_software_deployment_script_rejects_camera_without_service(tmp_path: Path):
    repository = Path(__file__).parents[1]
    result = subprocess.run(
        [
            str(repository / "deploy" / "install-software.sh"),
            "--with-camera",
            "--no-service",
            "--skip-install",
            "--prefix",
            str(tmp_path / "prefix"),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "cannot be combined" in result.stderr
