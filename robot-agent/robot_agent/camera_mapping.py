from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .camera_rig import RGB_CAMERA_DEVICE_ENV, RGB_CAMERA_NAMES


class CameraMappingError(ValueError):
    pass


_VIDEO_DEVICE_PATTERN = re.compile(r"/dev/video([0-9]{1,3})")


@dataclass(frozen=True)
class CameraMappingConfig:
    env_file: Path
    policy_registry_path: Path | None
    restart_camera_command: tuple[str, ...] = (
        "systemctl",
        "--user",
        "restart",
        "robot-camera.service",
    )
    restart_robotd_command: tuple[str, ...] = (
        "systemctl",
        "--user",
        "restart",
        "robotd-software.service",
    )
    command_timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> CameraMappingConfig:
        configured_env = os.environ.get("ROBOT_AGENT_ENV_FILE", "").strip()
        if configured_env:
            env_file = Path(configured_env).expanduser()
        else:
            config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            env_file = config_home / "robot-agent" / "robot-agent.env"
        policy_registry = os.environ.get("ROBOT_AGENT_POLICY_REGISTRY", "").strip()
        return cls(
            env_file=env_file,
            policy_registry_path=Path(policy_registry).expanduser() if policy_registry else None,
        )


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class CameraMappingManager:
    def __init__(
        self,
        config: CameraMappingConfig | None = None,
        *,
        command_runner: CommandRunner | None = None,
    ):
        self.config = config or CameraMappingConfig.from_env()
        self._command_runner = command_runner or subprocess.run

    def status(self) -> dict[str, Any]:
        env_values = self._read_env_values()
        policy_values = self._read_policy_values()
        available_devices = _available_video_devices()
        cameras: dict[str, dict[str, Any]] = {}
        for name in RGB_CAMERA_NAMES:
            configured = policy_values.get(name) or env_values.get(name) or ""
            cameras[name] = {
                "device": configured,
                "env_device": env_values.get(name, ""),
                "policy_device": policy_values.get(name, ""),
                "exists": bool(configured and Path(configured).exists()),
                "consistent": bool(
                    env_values.get(name)
                    and policy_values.get(name)
                    and env_values.get(name) == policy_values.get(name)
                ),
            }
        return {
            "success": True,
            "configured": self.config.env_file.exists()
            and self.config.policy_registry_path is not None
            and self.config.policy_registry_path.exists(),
            "env_file": str(self.config.env_file),
            "policy_registry_path": str(self.config.policy_registry_path or ""),
            "available_devices": available_devices,
            "cameras": cameras,
        }

    def validate(self, raw_mapping: dict[str, Any]) -> dict[str, Any]:
        mapping = _normalize_mapping(raw_mapping, require_exists=True)
        return {
            "success": True,
            "message": "camera mapping is valid; no files were changed",
            "mapping": mapping,
        }

    def apply(
        self,
        raw_mapping: dict[str, Any],
        *,
        operator_confirmation: str,
        restart_services: bool,
    ) -> dict[str, Any]:
        if operator_confirmation != "CONFIRM CAMERA MAPPING":
            raise CameraMappingError(
                "operator confirmation must be exactly 'CONFIRM CAMERA MAPPING'"
            )
        mapping = _normalize_mapping(raw_mapping, require_exists=True)
        self._write_env_mapping(mapping)
        self._write_policy_mapping(mapping)
        restart_results: list[dict[str, Any]] = []
        if restart_services:
            restart_results.append(
                self._run_fixed_command("restart_camera", self.config.restart_camera_command)
            )
            restart_results.append(
                self._run_fixed_command("restart_robotd", self.config.restart_robotd_command)
            )
        return {
            "success": True,
            "message": (
                "camera mapping saved and services restarted"
                if restart_services
                else "camera mapping saved; restart services to apply it"
            ),
            "mapping": mapping,
            "restart_results": restart_results,
            "status": self.status(),
        }

    def _read_env_values(self) -> dict[str, str]:
        try:
            lines = self.config.env_file.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return {}
        values: dict[str, str] = {}
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            for name, variable in RGB_CAMERA_DEVICE_ENV.items():
                if key == variable:
                    values[name] = value.strip().strip('"\'')
        return values

    def _read_policy_values(self) -> dict[str, str]:
        path = self.config.policy_registry_path
        if path is None:
            return {}
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            return {}
        cameras = raw.get("robot", {}).get("cameras", {}) if isinstance(raw, dict) else {}
        if not isinstance(cameras, dict):
            return {}
        values: dict[str, str] = {}
        for name in RGB_CAMERA_NAMES:
            camera = cameras.get(name)
            if isinstance(camera, dict) and "device" in camera:
                values[name] = _normalize_device(camera["device"])
        return values

    def _write_env_mapping(self, mapping: dict[str, str]) -> None:
        path = self.config.env_file
        if not path.exists() or path.is_symlink() or not path.is_file():
            raise CameraMappingError(f"env file is unavailable: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        output: list[str] = []
        seen: set[str] = set()
        variable_to_device = {
            RGB_CAMERA_DEVICE_ENV[name]: mapping[name] for name in RGB_CAMERA_NAMES
        }
        for line in lines:
            stripped = line.strip()
            key = stripped.split("=", 1)[0] if "=" in stripped else ""
            if key in variable_to_device and not stripped.startswith("#"):
                output.append(f"{key}={variable_to_device[key]}")
                seen.add(key)
            else:
                output.append(line)
        for name in RGB_CAMERA_NAMES:
            variable = RGB_CAMERA_DEVICE_ENV[name]
            if variable not in seen:
                output.append(f"{variable}={mapping[name]}")
        _atomic_write(path, "\n".join(output) + "\n")

    def _write_policy_mapping(self, mapping: dict[str, str]) -> None:
        path = self.config.policy_registry_path
        if path is None or not path.exists() or path.is_symlink() or not path.is_file():
            raise CameraMappingError(f"policy registry is unavailable: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise CameraMappingError("policy registry must be a YAML object")
        robot = raw.get("robot")
        cameras = robot.get("cameras") if isinstance(robot, dict) else None
        if not isinstance(cameras, dict) or set(cameras) != set(RGB_CAMERA_NAMES):
            raise CameraMappingError(
                "policy registry must contain exactly head_cam, left_cam, and right_cam"
            )
        for name in RGB_CAMERA_NAMES:
            camera = cameras.get(name)
            if not isinstance(camera, dict):
                raise CameraMappingError(f"policy camera {name} must be an object")
            camera["device"] = int(mapping[name].removeprefix("/dev/video"))
        _atomic_write(path, yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))

    def _run_fixed_command(self, label: str, command: tuple[str, ...]) -> dict[str, Any]:
        try:
            result = self._command_runner(
                list(command),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.config.command_timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CameraMappingError(f"{label} failed: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-1000:]
            raise CameraMappingError(f"{label} exited with {result.returncode}: {detail}")
        return {"label": label, "command": list(command), "returncode": result.returncode}


def _normalize_mapping(raw_mapping: dict[str, Any], *, require_exists: bool) -> dict[str, str]:
    if set(raw_mapping) != set(RGB_CAMERA_NAMES):
        raise CameraMappingError("mapping must contain exactly head_cam, left_cam, and right_cam")
    mapping = {name: _normalize_device(raw_mapping[name]) for name in RGB_CAMERA_NAMES}
    if len(set(mapping.values())) != len(RGB_CAMERA_NAMES):
        raise CameraMappingError("camera devices must be unique")
    if require_exists:
        missing = [device for device in mapping.values() if not Path(device).exists()]
        if missing:
            raise CameraMappingError("camera devices are unavailable: " + ", ".join(missing))
    return mapping


def _normalize_device(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = f"/dev/video{value}"
    if not isinstance(value, str):
        raise CameraMappingError("camera device must be /dev/videoN")
    value = value.strip()
    match = _VIDEO_DEVICE_PATTERN.fullmatch(value)
    if not match:
        raise CameraMappingError("camera device must be /dev/videoN")
    return f"/dev/video{int(match.group(1))}"


def _available_video_devices() -> list[dict[str, Any]]:
    devices = []
    for path in sorted(Path("/dev").glob("video*"), key=lambda item: item.name):
        if _VIDEO_DEVICE_PATTERN.fullmatch(str(path)):
            devices.append({"device": str(path), "exists": path.exists()})
    return devices


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode or 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(mode or 0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = ["CameraMappingConfig", "CameraMappingError", "CameraMappingManager"]
