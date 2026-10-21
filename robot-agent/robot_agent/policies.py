from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .camera_rig import RGB_CAMERA_NAME_SET


class PolicyRegistryError(ValueError):
    pass


RGB_CAMERA_FIELDS = frozenset({"type", "device", "width", "height", "fps"})


@dataclass(frozen=True)
class CameraSpec:
    name: str
    camera_type: str
    device: str | int
    width: int
    height: int
    fps: int

    def to_lerobot(self) -> dict[str, Any]:
        return {
            "type": self.camera_type,
            "index_or_path": self.device,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }


@dataclass(frozen=True)
class RobotSpec:
    robot_type: str
    robot_id: str
    left_arm_port: str
    right_arm_port: str
    dataset_owner: str
    cameras: tuple[CameraSpec, ...]


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    task_id: str
    family: str
    version: str
    description: str
    executable: Path
    executable_sha256: str
    workspace: Path
    runtime_manifest: Path
    runtime_sha256: str
    model_path: Path
    model_file: str
    model_sha256: str
    timeout_seconds: int
    enabled: bool
    extra_args: tuple[str, ...]

    def fingerprint(self) -> str:
        payload = {
            "policy_id": self.policy_id,
            "task_id": self.task_id,
            "family": self.family,
            "version": self.version,
            "description": self.description,
            "executable": str(self.executable),
            "executable_sha256": self.executable_sha256,
            "workspace": str(self.workspace),
            "runtime_manifest": str(self.runtime_manifest),
            "runtime_sha256": self.runtime_sha256,
            "model_path": str(self.model_path),
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
            "timeout_seconds": self.timeout_seconds,
            "extra_args": list(self.extra_args),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

class PolicyRegistry:
    def __init__(
        self,
        *,
        version: int,
        robot: RobotSpec,
        default_policies: dict[str, str],
        policies: dict[str, PolicySpec],
        file_hasher: Callable[[Path], str] | None = None,
    ):
        self.version = version
        self.robot = robot
        self.default_policies = dict(default_policies)
        self.policies = dict(policies)
        self._file_hasher = file_hasher or sha256_file
        self._verified_artifacts: dict[tuple[str, int, int], str] = {}
        self._validate_links()

    @classmethod
    def load(cls, path: str | Path) -> PolicyRegistry:
        registry_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise PolicyRegistryError("policy registry must be a YAML object")
        if raw.get("version") != 1:
            raise PolicyRegistryError("policy registry version must be 1")
        robot = _parse_robot(raw.get("robot"))
        defaults = raw.get("task_defaults", {})
        policies_raw = raw.get("policies", {})
        if not isinstance(defaults, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in defaults.items()
        ):
            raise PolicyRegistryError("task_defaults must map task IDs to policy IDs")
        if not isinstance(policies_raw, dict) or not policies_raw:
            raise PolicyRegistryError("policies must be a non-empty object")
        policies = {
            policy_id: _parse_policy(policy_id, value, registry_path.parent)
            for policy_id, value in policies_raw.items()
        }
        return cls(
            version=1,
            robot=robot,
            default_policies=defaults,
            policies=policies,
        )

    def resolve(
        self,
        task_id: str,
        policy_id: str | None = None,
        *,
        verify_artifact: bool = True,
    ) -> PolicySpec:
        selected_id = policy_id or self.default_policies.get(task_id)
        if not selected_id:
            raise PolicyRegistryError(f"no default policy is configured for task {task_id!r}")
        policy = self.policies.get(selected_id)
        if policy is None or policy.task_id != task_id:
            raise PolicyRegistryError(
                f"policy {selected_id!r} is not registered for task {task_id!r}"
            )
        if not policy.enabled:
            raise PolicyRegistryError(f"policy {selected_id!r} is disabled")
        self._validate_runtime_paths(policy)
        if verify_artifact:
            self._verify_artifact(policy)
        return policy

    def resolve_binding(self, task_id: str) -> dict[str, str]:
        policy = self.resolve(task_id)
        return {
            "policy_id": policy.policy_id,
            "policy_fingerprint": self.policy_fingerprint(policy),
        }

    def validate_binding(self, task_id: str, binding: dict[str, Any]) -> PolicySpec:
        policy_id = binding.get("policy_id")
        fingerprint = binding.get("policy_fingerprint")
        if not isinstance(policy_id, str) or not isinstance(fingerprint, str):
            raise PolicyRegistryError("execution binding is missing policy identity")
        policy = self.resolve(task_id, policy_id)
        if self.policy_fingerprint(policy) != fingerprint:
            raise PolicyRegistryError("policy binding changed after authorization")
        return policy

    def policy_fingerprint(self, policy: PolicySpec) -> str:
        payload = {
            "policy_fingerprint": policy.fingerprint(),
            "robot": {
                "type": self.robot.robot_type,
                "id": self.robot.robot_id,
                "left_arm_port": self.robot.left_arm_port,
                "right_arm_port": self.robot.right_arm_port,
                "dataset_owner": self.robot.dataset_owner,
                "cameras": [
                    {
                        "name": camera.name,
                        **camera.to_lerobot(),
                    }
                    for camera in self.robot.cameras
                ],
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def build_command(self, policy: PolicySpec, run_id: str | None = None) -> list[str]:
        identifier = _safe_identifier(run_id or uuid.uuid4().hex)
        cameras = {
            camera.name: camera.to_lerobot()
            for camera in self.robot.cameras
        }
        return [
            str(policy.executable),
            f"--robot.type={self.robot.robot_type}",
            f"--robot.left_arm_port={self.robot.left_arm_port}",
            f"--robot.right_arm_port={self.robot.right_arm_port}",
            f"--robot.id={self.robot.robot_id}",
            f"--robot.cameras={json.dumps(cameras, separators=(',', ':'))}",
            "--display_data=false",
            f"--dataset.repo_id={self.dataset_repo_id(policy, identifier)}",
            f"--dataset.single_task={policy.description}",
            "--dataset.push_to_hub=false",
            f"--policy.path={policy.model_path}",
            f"--dataset.episode_time_s={policy.timeout_seconds}",
            *policy.extra_args,
        ]

    def dataset_repo_id(self, policy: PolicySpec, run_id: str) -> str:
        identifier = _safe_identifier(run_id)
        return f"{self.robot.dataset_owner}/eval_{policy.task_id}_{identifier}"

    def public_view(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task_defaults": dict(self.default_policies),
            "policies": {
                policy_id: {
                    "task_id": policy.task_id,
                    "family": policy.family,
                    "version": policy.version,
                    "enabled": policy.enabled,
                    "fingerprint": self.policy_fingerprint(policy),
                }
                for policy_id, policy in sorted(self.policies.items())
            },
        }

    def readiness(self, *, verify_artifacts: bool = True) -> dict[str, Any]:
        tasks: dict[str, Any] = {}
        ready = True
        for task_id, policy_id in sorted(self.default_policies.items()):
            try:
                policy = self.resolve(
                    task_id,
                    policy_id,
                    verify_artifact=verify_artifacts,
                )
                tasks[task_id] = {
                    "ready": True,
                    "policy_id": policy.policy_id,
                    "fingerprint": self.policy_fingerprint(policy),
                }
            except (OSError, PolicyRegistryError) as exc:
                ready = False
                tasks[task_id] = {
                    "ready": False,
                    "policy_id": policy_id,
                    "error": str(exc),
                }
        return {"ready": ready and bool(tasks), "tasks": tasks}

    def _validate_links(self) -> None:
        for task_id, policy_id in self.default_policies.items():
            policy = self.policies.get(policy_id)
            if policy is None:
                raise PolicyRegistryError(f"default policy {policy_id!r} does not exist")
            if policy.task_id != task_id:
                raise PolicyRegistryError(f"default policy {policy_id!r} belongs to another task")

    @staticmethod
    def _validate_runtime_paths(policy: PolicySpec) -> None:
        if not policy.workspace.is_dir():
            raise PolicyRegistryError(f"policy workspace does not exist: {policy.workspace}")
        if not policy.executable.is_file() or not os.access(policy.executable, os.X_OK):
            raise PolicyRegistryError(f"policy executable is not executable: {policy.executable}")
        if not policy.runtime_manifest.is_file():
            raise PolicyRegistryError(
                f"policy runtime manifest does not exist: {policy.runtime_manifest}"
            )
        if not policy.model_path.is_dir():
            raise PolicyRegistryError(f"policy model path does not exist: {policy.model_path}")

    def _verify_artifact(self, policy: PolicySpec) -> None:
        artifact = policy.model_path / policy.model_file
        if not artifact.is_file():
            raise PolicyRegistryError(f"policy artifact does not exist: {artifact}")
        self._verify_file(policy, artifact, policy.model_sha256, "model artifact")
        self._verify_file(
            policy,
            policy.executable,
            policy.executable_sha256,
            "executable",
        )
        self._verify_file(
            policy,
            policy.runtime_manifest,
            policy.runtime_sha256,
            "runtime manifest",
        )

    def _verify_file(
        self,
        policy: PolicySpec,
        path: Path,
        expected: str,
        label: str,
    ) -> None:
        stat = path.stat()
        cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
        actual = self._verified_artifacts.get(cache_key)
        if actual is None:
            actual = self._file_hasher(path)
            self._verified_artifacts[cache_key] = actual
        if actual != expected:
            raise PolicyRegistryError(f"policy {label} SHA-256 mismatch: {policy.policy_id}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_robot(raw: Any) -> RobotSpec:
    if not isinstance(raw, dict):
        raise PolicyRegistryError("robot policy configuration must be an object")
    cameras_raw = raw.get("cameras")
    if not isinstance(cameras_raw, dict) or not cameras_raw:
        raise PolicyRegistryError("robot cameras must be a non-empty object")
    if set(cameras_raw) != RGB_CAMERA_NAME_SET:
        raise PolicyRegistryError(
            "robot cameras must contain exactly head_cam, left_cam, and right_cam"
        )
    cameras = tuple(
        _parse_camera(name, value)
        for name, value in sorted(cameras_raw.items())
    )
    return RobotSpec(
        robot_type=_required_string(raw, "type", "robot"),
        robot_id=_required_string(raw, "id", "robot"),
        left_arm_port=_required_string(raw, "left_arm_port", "robot"),
        right_arm_port=_required_string(raw, "right_arm_port", "robot"),
        dataset_owner=_safe_identifier(_required_string(raw, "dataset_owner", "robot")),
        cameras=cameras,
    )


def _parse_camera(name: Any, raw: Any) -> CameraSpec:
    if not isinstance(name, str) or not name or not isinstance(raw, dict):
        raise PolicyRegistryError("camera entries require a name and object")
    if set(raw) != RGB_CAMERA_FIELDS:
        raise PolicyRegistryError(
            f"camera {name!r} must contain only type, device, width, height, and fps"
        )
    camera_type = _required_string(raw, "type", f"camera {name}")
    if camera_type != "opencv":
        raise PolicyRegistryError(f"camera {name!r} must use the opencv RGB backend")
    device = raw.get("device")
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    if not isinstance(device, (str, int)) or isinstance(device, bool):
        raise PolicyRegistryError(f"camera {name!r} requires a device")
    return CameraSpec(
        name=name,
        camera_type=camera_type,
        device=device,
        width=_positive_int(raw, "width", f"camera {name}"),
        height=_positive_int(raw, "height", f"camera {name}"),
        fps=_positive_int(raw, "fps", f"camera {name}"),
    )


def _parse_policy(policy_id: Any, raw: Any, base: Path) -> PolicySpec:
    if not isinstance(policy_id, str) or not policy_id or not isinstance(raw, dict):
        raise PolicyRegistryError("policy entries require an ID and object")
    model_sha256 = _required_string(raw, "model_sha256", f"policy {policy_id}").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", model_sha256):
        raise PolicyRegistryError(f"policy {policy_id!r} model_sha256 must be 64 hex characters")
    extra_args = raw.get("extra_args", [])
    if not isinstance(extra_args, list) or not all(isinstance(value, str) for value in extra_args):
        raise PolicyRegistryError(f"policy {policy_id!r} extra_args must be a string list")
    reserved_prefixes = ("--robot.", "--policy.", "--dataset.", "--display_data")
    if any(argument.startswith(reserved_prefixes) for argument in extra_args):
        raise PolicyRegistryError(f"policy {policy_id!r} extra_args override reserved arguments")
    executable_sha256 = _sha256_string(raw, "executable_sha256", policy_id)
    runtime_sha256 = _sha256_string(raw, "runtime_sha256", policy_id)
    return PolicySpec(
        policy_id=policy_id,
        task_id=_required_string(raw, "task_id", f"policy {policy_id}"),
        family=_required_string(raw, "family", f"policy {policy_id}"),
        version=_required_string(raw, "version", f"policy {policy_id}"),
        description=_required_string(raw, "description", f"policy {policy_id}"),
        executable=_resolve_path(raw, "executable", base, policy_id),
        executable_sha256=executable_sha256,
        workspace=_resolve_path(raw, "workspace", base, policy_id),
        runtime_manifest=_resolve_path(raw, "runtime_manifest", base, policy_id),
        runtime_sha256=runtime_sha256,
        model_path=_resolve_path(raw, "model_path", base, policy_id),
        model_file=_required_string(raw, "model_file", f"policy {policy_id}"),
        model_sha256=model_sha256,
        timeout_seconds=_positive_int(raw, "timeout_seconds", f"policy {policy_id}"),
        enabled=raw.get("enabled") is True,
        extra_args=tuple(extra_args),
    )


def _resolve_path(raw: dict[str, Any], key: str, base: Path, policy_id: str) -> Path:
    value = Path(_required_string(raw, key, f"policy {policy_id}")).expanduser()
    return (base / value).resolve() if not value.is_absolute() else value.resolve()


def _sha256_string(raw: dict[str, Any], key: str, policy_id: str) -> str:
    value = _required_string(raw, key, f"policy {policy_id}").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PolicyRegistryError(f"policy {policy_id!r} {key} must be 64 hex characters")
    return value


def _required_string(raw: dict[str, Any], key: str, owner: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PolicyRegistryError(f"{owner} requires non-empty {key}")
    return value.strip()


def _positive_int(raw: dict[str, Any], key: str, owner: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PolicyRegistryError(f"{owner} requires positive integer {key}")
    return value


def _safe_identifier(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", value).strip("._-")
    if not sanitized:
        raise PolicyRegistryError("identifier contains no safe characters")
    return sanitized[:80]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate learned-policy deployment registry")
    parser.add_argument("registry")
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="validate registry structure without checking runtime paths or model hashes",
    )
    args = parser.parse_args()
    try:
        registry = PolicyRegistry.load(args.registry)
        if args.schema_only:
            payload = {"valid": True, "ready": False, **registry.public_view()}
            exit_code = 0
        else:
            payload = registry.readiness(verify_artifacts=True)
            payload["valid"] = True
            exit_code = 0 if payload["ready"] else 1
    except (OSError, PolicyRegistryError, yaml.YAMLError) as exc:
        payload = {"valid": False, "ready": False, "error": str(exc)}
        exit_code = 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()


__all__ = [
    "CameraSpec",
    "PolicyRegistry",
    "PolicyRegistryError",
    "PolicySpec",
    "RobotSpec",
    "main",
    "sha256_file",
]
