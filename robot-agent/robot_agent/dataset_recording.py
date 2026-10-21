from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .policies import PolicyRegistry


class DatasetRecordingError(ValueError):
    pass


_DATASET_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}")
_ACTIVE_STATES = frozenset({"starting", "recording", "stopping"})
_TOP_LEVEL_FIELDS = frozenset(
    {
        "version",
        "enabled",
        "executable",
        "workspace",
        "storage_root",
        "teleoperator",
        "defaults",
        "limits",
        "camera_service",
        "stop_grace_seconds",
        "log_tail_lines",
    }
)


@dataclass(frozen=True)
class DatasetRecordingConfig:
    enabled: bool
    executable: Path
    workspace: Path
    storage_root: Path
    teleoperator_type: str
    teleoperator_id: str
    left_arm_port: str
    right_arm_port: str
    fps: int
    default_num_episodes: int
    default_episode_time_s: float
    default_reset_time_s: float
    max_num_episodes: int
    max_episode_time_s: float
    max_reset_time_s: float
    camera_pause_command: tuple[str, ...]
    camera_resume_command: tuple[str, ...]
    camera_command_timeout_seconds: float
    stop_grace_seconds: float
    log_tail_lines: int

    @classmethod
    def load(cls, path: str | Path) -> DatasetRecordingConfig:
        source = Path(path).expanduser().resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise DatasetRecordingError("dataset recording config must be a YAML object")
        _require_exact_fields(raw, _TOP_LEVEL_FIELDS, "dataset recording config")
        if raw.get("version") != 1:
            raise DatasetRecordingError("dataset recording config version must be 1")
        if not isinstance(raw.get("enabled"), bool):
            raise DatasetRecordingError("enabled must be a boolean")

        teleoperator = _required_object(raw, "teleoperator")
        _require_exact_fields(
            teleoperator,
            frozenset({"type", "id", "left_arm_port", "right_arm_port"}),
            "teleoperator",
        )
        defaults = _required_object(raw, "defaults")
        _require_exact_fields(
            defaults,
            frozenset({"fps", "num_episodes", "episode_time_s", "reset_time_s"}),
            "defaults",
        )
        limits = _required_object(raw, "limits")
        _require_exact_fields(
            limits,
            frozenset({"num_episodes", "episode_time_s", "reset_time_s"}),
            "limits",
        )
        camera_service = _required_object(raw, "camera_service")
        _require_exact_fields(
            camera_service,
            frozenset({"pause_command", "resume_command", "command_timeout_seconds"}),
            "camera_service",
        )

        config = cls(
            enabled=raw["enabled"],
            executable=_config_path(raw, "executable", source.parent),
            workspace=_config_path(raw, "workspace", source.parent),
            storage_root=_config_path(raw, "storage_root", source.parent),
            teleoperator_type=_required_string(teleoperator, "type", "teleoperator"),
            teleoperator_id=_required_string(teleoperator, "id", "teleoperator"),
            left_arm_port=_required_string(teleoperator, "left_arm_port", "teleoperator"),
            right_arm_port=_required_string(teleoperator, "right_arm_port", "teleoperator"),
            fps=_positive_int(defaults, "fps", "defaults"),
            default_num_episodes=_positive_int(defaults, "num_episodes", "defaults"),
            default_episode_time_s=_positive_number(defaults, "episode_time_s", "defaults"),
            default_reset_time_s=_non_negative_number(defaults, "reset_time_s", "defaults"),
            max_num_episodes=_positive_int(limits, "num_episodes", "limits"),
            max_episode_time_s=_positive_number(limits, "episode_time_s", "limits"),
            max_reset_time_s=_non_negative_number(limits, "reset_time_s", "limits"),
            camera_pause_command=_command(camera_service, "pause_command"),
            camera_resume_command=_command(camera_service, "resume_command"),
            camera_command_timeout_seconds=_positive_number(
                camera_service,
                "command_timeout_seconds",
                "camera_service",
            ),
            stop_grace_seconds=_positive_number(
                raw,
                "stop_grace_seconds",
                "dataset recording config",
            ),
            log_tail_lines=_positive_int(raw, "log_tail_lines", "dataset recording config"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.teleoperator_type != "bi_so100_leader":
            raise DatasetRecordingError("teleoperator.type must be bi_so100_leader")
        if self.default_num_episodes > self.max_num_episodes:
            raise DatasetRecordingError("default num_episodes exceeds its configured limit")
        if self.default_episode_time_s > self.max_episode_time_s:
            raise DatasetRecordingError("default episode_time_s exceeds its configured limit")
        if self.default_reset_time_s > self.max_reset_time_s:
            raise DatasetRecordingError("default reset_time_s exceeds its configured limit")
        if self.fps > 120:
            raise DatasetRecordingError("defaults.fps must not exceed 120")
        if self.log_tail_lines > 500:
            raise DatasetRecordingError("log_tail_lines must not exceed 500")
        serial_ports = {
            self.left_arm_port,
            self.right_arm_port,
        }
        if len(serial_ports) != 2:
            raise DatasetRecordingError("teleoperator arm ports must be different")

    def defaults_view(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "num_episodes": self.default_num_episodes,
            "episode_time_s": self.default_episode_time_s,
            "reset_time_s": self.default_reset_time_s,
        }

    def limits_view(self) -> dict[str, Any]:
        return {
            "num_episodes": self.max_num_episodes,
            "episode_time_s": self.max_episode_time_s,
            "reset_time_s": self.max_reset_time_s,
        }


@dataclass(frozen=True)
class DatasetRecordingRequest:
    dataset_name: str
    task: str
    num_episodes: int
    episode_time_s: float
    reset_time_s: float
    resume: bool

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        config: DatasetRecordingConfig,
    ) -> DatasetRecordingRequest:
        allowed = frozenset(
            {
                "dataset_name",
                "task",
                "num_episodes",
                "episode_time_s",
                "reset_time_s",
                "resume",
            }
        )
        extra = set(raw) - allowed
        if extra:
            raise DatasetRecordingError(
                "unknown dataset recording fields: " + ", ".join(sorted(extra))
            )
        dataset_name = raw.get("dataset_name")
        if not isinstance(dataset_name, str) or not _DATASET_NAME_PATTERN.fullmatch(
            dataset_name.strip()
        ):
            raise DatasetRecordingError(
                "dataset_name must be 1-80 lowercase letters, digits, dots, dashes, or underscores"
            )
        task = raw.get("task")
        if not isinstance(task, str) or not task.strip():
            raise DatasetRecordingError("task must be a non-empty string")
        task = task.strip()
        if len(task) > 240 or any(ord(character) < 32 for character in task):
            raise DatasetRecordingError("task must be at most 240 printable characters")

        num_episodes = raw.get("num_episodes", config.default_num_episodes)
        episode_time_s = raw.get("episode_time_s", config.default_episode_time_s)
        reset_time_s = raw.get("reset_time_s", config.default_reset_time_s)
        resume = raw.get("resume", False)
        if isinstance(num_episodes, bool) or not isinstance(num_episodes, int):
            raise DatasetRecordingError("num_episodes must be an integer")
        if not 1 <= num_episodes <= config.max_num_episodes:
            raise DatasetRecordingError(
                f"num_episodes must be between 1 and {config.max_num_episodes}"
            )
        episode_time_s = _request_number(episode_time_s, "episode_time_s")
        reset_time_s = _request_number(reset_time_s, "reset_time_s", allow_zero=True)
        if episode_time_s > config.max_episode_time_s:
            raise DatasetRecordingError(
                f"episode_time_s must not exceed {config.max_episode_time_s:g}"
            )
        if reset_time_s > config.max_reset_time_s:
            raise DatasetRecordingError(
                f"reset_time_s must not exceed {config.max_reset_time_s:g}"
            )
        if not isinstance(resume, bool):
            raise DatasetRecordingError("resume must be a boolean")
        return cls(
            dataset_name=dataset_name.strip(),
            task=task,
            num_episodes=num_episodes,
            episode_time_s=episode_time_s,
            reset_time_s=reset_time_s,
            resume=resume,
        )


@dataclass(frozen=True)
class DatasetRecordingPlan:
    request: DatasetRecordingRequest
    repo_id: str
    dataset_root: Path
    command: tuple[str, ...]
    initial_episodes: int
    initial_frames: int

    def public_view(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "dataset_name": self.request.dataset_name,
            "task": self.request.task,
            "dataset_path": str(self.dataset_root),
            "fps": _command_value(self.command, "--dataset.fps="),
            "num_episodes": self.request.num_episodes,
            "episode_time_s": self.request.episode_time_s,
            "reset_time_s": self.request.reset_time_s,
            "resume": self.request.resume,
            "initial_episodes": self.initial_episodes,
            "initial_frames": self.initial_frames,
            "argv": list(self.command),
        }


@dataclass
class _RecordingSession:
    session_id: str
    plan: DatasetRecordingPlan
    process: Any
    log_path: Path
    log_handle: Any
    started_at: float
    state: str = "recording"
    message: str = "dataset recording is running"
    ended_at: float | None = None
    returncode: int | None = None
    stop_requested: bool = False
    camera_paused: bool = True
    camera_restored: bool = False
    camera_error: str = ""
    latest_episodes: int = 0
    latest_frames: int = 0
    progress_error: str = ""
    finished_event: threading.Event | None = None


PopenFactory = Callable[..., Any]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ProcessGroupSignaler = Callable[[int, int], None]


class DatasetRecordingSupervisor:
    def __init__(
        self,
        *,
        config: DatasetRecordingConfig | None,
        policy_registry: PolicyRegistry | None,
        state_dir: str | Path,
        unavailable_reason: str = "",
        popen_factory: PopenFactory | None = None,
        command_runner: CommandRunner | None = None,
        process_group_signaler: ProcessGroupSignaler | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self._config = config
        self._policy_registry = policy_registry
        self._state_dir = Path(state_dir).expanduser()
        self._unavailable_reason = unavailable_reason
        self._popen_factory = popen_factory or subprocess.Popen
        self._command_runner = command_runner or subprocess.run
        self._process_group_signaler = process_group_signaler or os.killpg
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._session: _RecordingSession | None = None
        self._starting = False
        self._start_cancelled = False
        self._start_finished = threading.Event()
        self._start_finished.set()
        self._last_state = "idle" if config is not None else "unavailable"
        self._last_message = unavailable_reason or "dataset recorder is idle"

    @property
    def active(self) -> bool:
        with self._lock:
            return self._starting or bool(
                self._session is not None and self._session.state in _ACTIVE_STATES
            )

    def capabilities(self) -> dict[str, Any]:
        config = self._config
        registry = self._policy_registry
        return {
            "configured": config is not None,
            "enabled": bool(config is not None and config.enabled),
            "available": bool(config is not None and config.enabled and registry is not None),
            "reason": self._unavailable_reason,
            "defaults": config.defaults_view() if config is not None else {},
            "limits": config.limits_view() if config is not None else {},
            "dataset_owner": registry.robot.dataset_owner if registry is not None else "",
            "robot_type": registry.robot.robot_type if registry is not None else "",
            "teleoperator_type": config.teleoperator_type if config is not None else "",
            "camera_names": (
                [camera.name for camera in registry.robot.cameras]
                if registry is not None
                else []
            ),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            data = self._status_data_locked()
        return {
            "success": True,
            "message": str(data["message"]),
            "data": data,
        }

    def validate(self, raw_request: dict[str, Any]) -> dict[str, Any]:
        try:
            plan = self.build_plan(raw_request, check_devices=True)
        except (DatasetRecordingError, OSError, json.JSONDecodeError) as exc:
            return self._failure(str(exc))
        return {
            "success": True,
            "message": "dataset recording request is valid; no hardware command was sent",
            "data": {
                **plan.public_view(),
                "dry_run": True,
                "hardware_command_sent": False,
            },
        }

    def build_plan(
        self,
        raw_request: dict[str, Any],
        *,
        check_devices: bool,
    ) -> DatasetRecordingPlan:
        config, registry = self._require_available()
        request = DatasetRecordingRequest.from_mapping(raw_request, config)
        self._validate_runtime(config, registry, check_devices=check_devices)

        storage_root = config.storage_root.resolve()
        dataset_root = (
            storage_root / registry.robot.dataset_owner / request.dataset_name
        ).resolve()
        try:
            dataset_root.relative_to(storage_root)
        except ValueError as exc:
            raise DatasetRecordingError("dataset path escapes storage_root") from exc
        repo_id = f"{registry.robot.dataset_owner}/{request.dataset_name}"
        initial_episodes = 0
        initial_frames = 0
        if request.resume:
            initial_episodes, initial_frames = self._validate_resume_dataset(
                dataset_root,
                registry,
                config.fps,
            )
        elif dataset_root.exists() or dataset_root.is_symlink():
            raise DatasetRecordingError(
                f"dataset already exists; enable resume to append: {dataset_root}"
            )

        cameras = {
            camera.name: camera.to_lerobot()
            for camera in registry.robot.cameras
        }
        command = (
            str(config.executable),
            f"--robot.type={registry.robot.robot_type}",
            f"--robot.left_arm_port={registry.robot.left_arm_port}",
            f"--robot.right_arm_port={registry.robot.right_arm_port}",
            f"--robot.id={registry.robot.robot_id}",
            f"--robot.cameras={json.dumps(cameras, separators=(',', ':'))}",
            f"--teleop.type={config.teleoperator_type}",
            f"--teleop.left_arm_port={config.left_arm_port}",
            f"--teleop.right_arm_port={config.right_arm_port}",
            f"--teleop.id={config.teleoperator_id}",
            "--display_data=false",
            "--play_sounds=false",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={dataset_root}",
            f"--dataset.single_task={request.task}",
            f"--dataset.fps={config.fps}",
            f"--dataset.episode_time_s={request.episode_time_s:g}",
            f"--dataset.reset_time_s={request.reset_time_s:g}",
            f"--dataset.num_episodes={request.num_episodes}",
            "--dataset.video=true",
            "--dataset.push_to_hub=false",
            f"--resume={str(request.resume).lower()}",
        )
        return DatasetRecordingPlan(
            request=request,
            repo_id=repo_id,
            dataset_root=dataset_root,
            command=command,
            initial_episodes=initial_episodes,
            initial_frames=initial_frames,
        )

    def start(
        self,
        raw_request: dict[str, Any],
        *,
        operator_confirmation: str,
    ) -> dict[str, Any]:
        if operator_confirmation != "CONFIRM RECORD DATASET":
            return self._failure(
                "operator confirmation must be exactly 'CONFIRM RECORD DATASET'"
            )
        with self._lock:
            if self._starting or (
                self._session is not None and self._session.state in _ACTIVE_STATES
            ):
                return self._failure("another dataset recording is already active")
            self._starting = True
            self._start_cancelled = False
            self._start_finished.clear()
            self._last_state = "starting"
            self._last_message = "validating dataset recording request"

        camera_paused = False
        log_handle: Any | None = None
        try:
            plan = self.build_plan(raw_request, check_devices=True)
            config, _registry = self._require_available()
            config.storage_root.mkdir(parents=True, exist_ok=True)
            self._run_camera_command(config.camera_pause_command, "pause")
            camera_paused = True
            busy_devices = _device_owners(_recording_device_paths(config, _registry))
            if busy_devices:
                raise DatasetRecordingError(_busy_devices_message(busy_devices))
            with self._lock:
                if self._start_cancelled:
                    raise DatasetRecordingError("dataset recording start was cancelled")

            session_id = f"recording-{uuid.uuid4().hex}"
            log_dir = self._state_dir / "dataset-recording"
            log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            log_dir.chmod(0o700)
            log_path = log_dir / f"{plan.request.dataset_name}-{session_id[-12:]}.log"
            log_handle = log_path.open("xb")
            log_path.chmod(0o600)
            header = {
                "session_id": session_id,
                "repo_id": plan.repo_id,
                "dataset_path": str(plan.dataset_root),
                "started_at": self._clock(),
                "argv": list(plan.command),
            }
            log_handle.write((json.dumps(header, ensure_ascii=False) + "\n").encode("utf-8"))
            log_handle.flush()
            process = self._popen_factory(
                list(plan.command),
                cwd=config.workspace,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            session = _RecordingSession(
                session_id=session_id,
                plan=plan,
                process=process,
                log_path=log_path,
                log_handle=log_handle,
                started_at=self._clock(),
                latest_episodes=plan.initial_episodes,
                latest_frames=plan.initial_frames,
                finished_event=threading.Event(),
            )
            with self._lock:
                self._session = session
                self._starting = False
                self._last_state = session.state
                self._last_message = session.message
                self._start_finished.set()
            threading.Thread(
                target=self._monitor,
                args=(session,),
                name=f"dataset-recording-{session_id[-8:]}",
                daemon=True,
            ).start()
            return {
                "success": True,
                "message": "dataset recording started; camera preview is paused",
                "data": self.status()["data"],
            }
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            if log_handle is not None:
                log_handle.close()
            restore_error = ""
            if camera_paused and self._config is not None:
                try:
                    self._run_camera_command(self._config.camera_resume_command, "resume")
                except DatasetRecordingError as restore_exc:
                    restore_error = f"; camera restore failed: {restore_exc}"
            with self._lock:
                self._starting = False
                self._last_state = "failed"
                self._last_message = f"dataset recording failed to start: {exc}{restore_error}"
                self._start_finished.set()
            return self._failure(self._last_message)

    def stop(self, *, reason: str = "operator_stop") -> dict[str, Any]:
        with self._lock:
            if self._starting:
                self._start_cancelled = True
                start_finished = self._start_finished
                wait_timeout = (
                    self._config.camera_command_timeout_seconds
                    + self._config.stop_grace_seconds
                    if self._config is not None
                    else 10.0
                )
            else:
                start_finished = None
                wait_timeout = 0.0
        if start_finished is not None:
            if not start_finished.wait(wait_timeout):
                return self._failure(
                    "dataset recording start did not cancel within the configured timeout"
                )
            return self.stop(reason=reason)

        with self._lock:
            session = self._session
            if session is None or session.state not in _ACTIVE_STATES:
                return {
                    "success": True,
                    "message": "no dataset recording is active",
                    "data": self._status_data_locked(),
                }
            session.stop_requested = True
            session.state = "stopping"
            session.message = f"stopping dataset recording: {reason}"
            process = session.process
            finished_event = session.finished_event
            grace = self._config.stop_grace_seconds if self._config is not None else 5.0

        try:
            self._process_group_signaler(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            with self._lock:
                session.message = f"failed to signal dataset recorder: {exc}"
        if finished_event is not None and not finished_event.wait(grace):
            try:
                self._process_group_signaler(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                with self._lock:
                    session.message = f"failed to kill dataset recorder: {exc}"
            finished_event.wait(grace)
        with self._lock:
            data = self._status_data_locked()
            still_active = bool(data["active"])
        return {
            "success": not still_active,
            "message": (
                "dataset recording stopped and camera preview restored"
                if not still_active
                else "dataset recorder did not stop within the configured timeout"
            ),
            "data": data,
        }

    def close(self) -> None:
        self.stop(reason="robotd_shutdown")

    def _monitor(self, session: _RecordingSession) -> None:
        try:
            returncode = int(session.process.wait())
        except Exception:
            returncode = -1
        camera_error = ""
        try:
            if self._config is not None:
                self._run_camera_command(self._config.camera_resume_command, "resume")
        except DatasetRecordingError as exc:
            camera_error = str(exc)

        with self._lock:
            session.returncode = returncode
            session.ended_at = self._clock()
            session.camera_restored = not camera_error
            session.camera_error = camera_error
            if camera_error:
                session.state = "failed"
                session.message = f"camera service could not be restored: {camera_error}"
            elif session.stop_requested:
                session.state = "stopped"
                session.message = "dataset recording stopped by operator"
            elif returncode == 0:
                session.state = "completed"
                session.message = "dataset recording completed"
            else:
                session.state = "failed"
                session.message = f"dataset recorder exited with code {returncode}"
            self._refresh_progress_locked(session)
            self._last_state = session.state
            self._last_message = session.message
            try:
                session.log_handle.close()
            finally:
                if session.finished_event is not None:
                    session.finished_event.set()

    def _run_camera_command(self, command: tuple[str, ...], action: str) -> None:
        if self._config is None:
            raise DatasetRecordingError("dataset recording is not configured")
        try:
            result = self._command_runner(
                list(command),
                capture_output=True,
                text=True,
                check=False,
                timeout=self._config.camera_command_timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DatasetRecordingError(f"camera service {action} command failed: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-1000:]
            suffix = f": {detail}" if detail else ""
            raise DatasetRecordingError(
                f"camera service {action} command exited with {result.returncode}{suffix}"
            )

    def _status_data_locked(self) -> dict[str, Any]:
        session = self._session
        if session is None:
            return {
                **self.capabilities(),
                "active": self._starting,
                "state": "starting" if self._starting else self._last_state,
                "message": self._last_message,
                "session_id": "",
                "repo_id": "",
                "dataset_name": "",
                "task": "",
                "dataset_path": "",
                "target_episodes": 0,
                "initial_episodes": 0,
                "recorded_episodes": 0,
                "total_episodes": 0,
                "total_frames": 0,
                "progress_percent": 0.0,
                "started_at": None,
                "ended_at": None,
                "returncode": None,
                "log_path": "",
                "log_tail": "",
                "camera_service": {
                    "paused": False,
                    "restored": False,
                    "error": "",
                },
            }
        self._refresh_progress_locked(session)
        recorded_episodes = max(
            0,
            session.latest_episodes - session.plan.initial_episodes,
        )
        target = session.plan.request.num_episodes
        return {
            **self.capabilities(),
            "active": session.state in _ACTIVE_STATES,
            "state": session.state,
            "message": session.message,
            "session_id": session.session_id,
            "repo_id": session.plan.repo_id,
            "dataset_name": session.plan.request.dataset_name,
            "task": session.plan.request.task,
            "dataset_path": str(session.plan.dataset_root),
            "target_episodes": target,
            "initial_episodes": session.plan.initial_episodes,
            "recorded_episodes": recorded_episodes,
            "total_episodes": session.latest_episodes,
            "total_frames": session.latest_frames,
            "progress_percent": round(min(recorded_episodes / target, 1.0) * 100, 1),
            "progress_error": session.progress_error,
            "started_at": session.started_at,
            "ended_at": session.ended_at,
            "returncode": session.returncode,
            "log_path": str(session.log_path),
            "log_tail": _read_log_tail(session.log_path, self._config.log_tail_lines),
            "camera_service": {
                "paused": session.camera_paused and not session.camera_restored,
                "restored": session.camera_restored,
                "error": session.camera_error,
            },
        }

    def _refresh_progress_locked(self, session: _RecordingSession) -> None:
        try:
            episodes, frames = _read_dataset_totals(session.plan.dataset_root)
        except (DatasetRecordingError, OSError, json.JSONDecodeError) as exc:
            if (session.plan.dataset_root / "meta" / "info.json").exists():
                session.progress_error = str(exc)
            return
        session.latest_episodes = episodes
        session.latest_frames = frames
        session.progress_error = ""

    def _require_available(self) -> tuple[DatasetRecordingConfig, PolicyRegistry]:
        if self._config is None:
            raise DatasetRecordingError(
                self._unavailable_reason or "dataset recording is not configured"
            )
        if not self._config.enabled:
            raise DatasetRecordingError("dataset recording is disabled by configuration")
        if self._policy_registry is None:
            raise DatasetRecordingError(
                self._unavailable_reason or "policy registry is unavailable"
            )
        return self._config, self._policy_registry

    @staticmethod
    def _validate_runtime(
        config: DatasetRecordingConfig,
        registry: PolicyRegistry,
        *,
        check_devices: bool,
    ) -> None:
        if registry.robot.robot_type != "bi_so100_follower":
            raise DatasetRecordingError("recording requires robot.type bi_so100_follower")
        if not config.executable.is_file() or not os.access(config.executable, os.X_OK):
            raise DatasetRecordingError(
                f"lerobot-record executable is unavailable: {config.executable}"
            )
        if not config.workspace.is_dir():
            raise DatasetRecordingError(f"LeRobot workspace is unavailable: {config.workspace}")
        serial_ports = {
            registry.robot.left_arm_port,
            registry.robot.right_arm_port,
            config.left_arm_port,
            config.right_arm_port,
        }
        if len(serial_ports) != 4:
            raise DatasetRecordingError("leader and follower arm ports must all be different")
        camera_devices = [_camera_device_path(camera.device) for camera in registry.robot.cameras]
        if len({str(path) for path in camera_devices}) != len(camera_devices):
            raise DatasetRecordingError("camera devices must be unique")
        if not check_devices:
            return
        missing = [
            str(path)
            for path in [
                Path(registry.robot.left_arm_port),
                Path(registry.robot.right_arm_port),
                Path(config.left_arm_port),
                Path(config.right_arm_port),
                *camera_devices,
            ]
            if not path.exists()
        ]
        if missing:
            raise DatasetRecordingError("recording devices are unavailable: " + ", ".join(missing))
        busy_serial_devices = _device_owners(_serial_device_paths(config, registry))
        if busy_serial_devices:
            raise DatasetRecordingError(_busy_devices_message(busy_serial_devices))

    @staticmethod
    def _validate_resume_dataset(
        dataset_root: Path,
        registry: PolicyRegistry,
        fps: int,
    ) -> tuple[int, int]:
        if dataset_root.is_symlink() or not dataset_root.is_dir():
            raise DatasetRecordingError(f"resume dataset does not exist: {dataset_root}")
        episodes, frames, info = _read_dataset_info(dataset_root)
        if info.get("robot_type") != registry.robot.robot_type:
            raise DatasetRecordingError("resume dataset robot_type does not match the registry")
        if info.get("fps") != fps:
            raise DatasetRecordingError("resume dataset fps does not match recording configuration")
        features = info.get("features")
        if not isinstance(features, dict):
            raise DatasetRecordingError("resume dataset features are invalid")
        for camera in registry.robot.cameras:
            feature = features.get(f"observation.images.{camera.name}")
            if not isinstance(feature, dict) or feature.get("shape") != [
                camera.height,
                camera.width,
                3,
            ]:
                raise DatasetRecordingError(
                    f"resume dataset camera feature does not match {camera.name}"
                )
        return episodes, frames

    def _failure(self, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "message": message,
            "data": {
                "active": self.active,
                "state": self._last_state,
                "configured": self._config is not None,
                "enabled": bool(self._config is not None and self._config.enabled),
            },
        }


def _read_dataset_totals(dataset_root: Path) -> tuple[int, int]:
    episodes, frames, _info = _read_dataset_info(dataset_root)
    return episodes, frames


def _read_dataset_info(dataset_root: Path) -> tuple[int, int, dict[str, Any]]:
    info_path = dataset_root / "meta" / "info.json"
    try:
        stat_result = info_path.lstat()
    except FileNotFoundError as exc:
        raise DatasetRecordingError(f"dataset metadata is missing: {info_path}") from exc
    if info_path.is_symlink() or not info_path.is_file() or stat_result.st_size > 2 * 1024 * 1024:
        raise DatasetRecordingError("dataset metadata must be a bounded regular file")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise DatasetRecordingError("dataset metadata must be a JSON object")
    episodes = info.get("total_episodes")
    frames = info.get("total_frames")
    if (
        isinstance(episodes, bool)
        or not isinstance(episodes, int)
        or episodes < 0
        or isinstance(frames, bool)
        or not isinstance(frames, int)
        or frames < 0
    ):
        raise DatasetRecordingError("dataset episode/frame totals are invalid")
    return episodes, frames, info


def _read_log_tail(path: Path, lines: int) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            return ""
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 64 * 1024))
            content = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(content.splitlines()[-lines:])[-16000:]


def _camera_device_path(device: str | int) -> Path:
    return Path(f"/dev/video{device}") if isinstance(device, int) else Path(device)


def _serial_device_paths(
    config: DatasetRecordingConfig,
    registry: PolicyRegistry,
) -> tuple[Path, ...]:
    return (
        Path(registry.robot.left_arm_port),
        Path(registry.robot.right_arm_port),
        Path(config.left_arm_port),
        Path(config.right_arm_port),
    )


def _recording_device_paths(
    config: DatasetRecordingConfig,
    registry: PolicyRegistry,
) -> tuple[Path, ...]:
    return (
        *_serial_device_paths(config, registry),
        *(_camera_device_path(camera.device) for camera in registry.robot.cameras),
    )


def _device_owners(paths: tuple[Path, ...]) -> dict[str, list[dict[str, Any]]]:
    targets: dict[tuple[int, int], str] = {}
    for path in paths:
        try:
            info = path.stat()
        except OSError:
            continue
        targets[(info.st_dev, info.st_ino)] = str(path)
    if not targets or not Path("/proc").is_dir():
        return {}

    owners: dict[str, dict[int, str]] = {}
    try:
        processes = list(Path("/proc").iterdir())
    except OSError:
        return {}
    for process_dir in processes:
        if not process_dir.name.isdigit():
            continue
        try:
            file_descriptors = list((process_dir / "fd").iterdir())
        except OSError:
            continue
        process_id = int(process_dir.name)
        process_name = _process_name(process_dir)
        for descriptor in file_descriptors:
            try:
                info = descriptor.stat()
            except OSError:
                continue
            path = targets.get((info.st_dev, info.st_ino))
            if path is not None:
                owners.setdefault(path, {})[process_id] = process_name
    return {
        path: [
            {"pid": process_id, "command": command}
            for process_id, command in sorted(processes.items())
        ]
        for path, processes in sorted(owners.items())
    }


def _process_name(process_dir: Path) -> str:
    try:
        name = (process_dir / "comm").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)[:64] or "unknown"


def _busy_devices_message(owners: dict[str, list[dict[str, Any]]]) -> str:
    details = []
    for path, processes in owners.items():
        process_text = ", ".join(
            f"pid {process['pid']} {process['command']}" for process in processes
        )
        details.append(f"{path} ({process_text})")
    return "recording devices are already in use: " + "; ".join(details)


def _command_value(command: tuple[str, ...], prefix: str) -> int | str:
    for argument in command:
        if argument.startswith(prefix):
            value = argument.removeprefix(prefix)
            return int(value) if value.isdigit() else value
    return ""


def _required_object(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise DatasetRecordingError(f"{key} must be an object")
    return value


def _required_string(raw: dict[str, Any], key: str, owner: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetRecordingError(f"{owner}.{key} must be a non-empty string")
    return value.strip()


def _config_path(raw: dict[str, Any], key: str, base: Path) -> Path:
    value = Path(_required_string(raw, key, "dataset recording config")).expanduser()
    return (base / value).resolve() if not value.is_absolute() else value.resolve()


def _positive_int(raw: dict[str, Any], key: str, owner: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetRecordingError(f"{owner}.{key} must be a positive integer")
    return value


def _positive_number(raw: dict[str, Any], key: str, owner: str) -> float:
    value = raw.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise DatasetRecordingError(f"{owner}.{key} must be a positive number")
    return float(value)


def _non_negative_number(raw: dict[str, Any], key: str, owner: str) -> float:
    value = raw.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise DatasetRecordingError(f"{owner}.{key} must be a non-negative number")
    return float(value)


def _request_number(value: Any, label: str, *, allow_zero: bool = False) -> float:
    minimum = 0 if allow_zero else 0.000001
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise DatasetRecordingError(f"{label} must be a {qualifier} number")
    return float(value)


def _command(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(argument, str) and argument for argument in value)
    ):
        raise DatasetRecordingError(f"camera_service.{key} must be a non-empty string list")
    return tuple(value)


def _require_exact_fields(raw: dict[str, Any], expected: frozenset[str], owner: str) -> None:
    missing = expected - set(raw)
    extra = set(raw) - expected
    if missing:
        raise DatasetRecordingError(f"{owner} is missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise DatasetRecordingError(f"{owner} has unknown fields: {', '.join(sorted(extra))}")


__all__ = [
    "DatasetRecordingConfig",
    "DatasetRecordingError",
    "DatasetRecordingPlan",
    "DatasetRecordingRequest",
    "DatasetRecordingSupervisor",
]
