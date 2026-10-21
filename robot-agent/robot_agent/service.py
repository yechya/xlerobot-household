from __future__ import annotations

import getpass
import json
import logging
import os
import signal
import socket
import socketserver
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .agent import DummyTaskBackend, RobotAgent
from .audit import JsonlAuditWriter
from .benchmarks import BenchmarkError, BenchmarkReport, load_report
from .config import RuntimeConfig
from .dataset_recording import (
    DatasetRecordingConfig,
    DatasetRecordingError,
    DatasetRecordingSupervisor,
)
from .deployment import DeploymentError, validate_deployment
from .episodes import EpisodeContext, EpisodeError, EpisodeRecorder
from .execution import ProcessExecutionSupervisor
from .missions import MissionCatalog, MissionError
from .table_docking import TableCatalog, TableDockingError
from .motion import MotionCatalog, MotionGateway, UnavailableMotionGateway
from .policies import PolicyRegistry, PolicyRegistryError
from .robotd import RobotDaemon
from .safety import (
    FileSafetyInterlock,
    SafetyError,
    SafetyPolicy,
    UnavailableSafetyInterlock,
)
from .semantic_map import SemanticMapError, SemanticMapSpec, sync_semantic_map
from .serialization import to_jsonable
from .skill_flow import TaskSafetyMetadata
from .task_catalog import TaskCatalog, TaskDefinition
from .world_model import WorldModel

logger = logging.getLogger("robot_agent.service")
MAX_REQUEST_BYTES = 1024 * 1024
_RECORDING_GATED_COMMANDS = frozenset(
    {
        "plan",
        "visual_plan_inspect",
        "visual_plan_arm",
        "visual_plan_execute",
        "observe",
        "dry_run",
        "arm",
        "execute",
        "reset",
        "motion_dry_run",
        "motion_arm",
        "motion_execute",
        "mission_dry_run",
        "mission_arm",
        "mission_execute",
        "dataset_recording_start",
    }
)


class RuntimeRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        if not raw:
            return
        if len(raw) > MAX_REQUEST_BYTES:
            self._write({"success": False, "error": "request is too large"})
            return
        try:
            request = json.loads(raw.decode("utf-8"))
            response = self.server.dispatch(request)  # type: ignore[attr-defined]
        except (UnicodeDecodeError, json.JSONDecodeError):
            response = {"success": False, "error": "invalid JSON request"}
        except ValueError as exc:
            response = {
                "success": False,
                "boundary": "standalone-robotd",
                "error": str(exc),
            }
        except Exception as exc:
            logger.exception("robotd request failed")
            response = {
                "success": False,
                "error": f"internal robotd error: {type(exc).__name__}",
            }
        self._write(response)

    def _write(self, response: dict[str, Any]) -> None:
        payload = json.dumps(
            to_jsonable(response),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        self.wfile.write(payload)


class RobotRuntimeServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    block_on_close = True

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        motion_gateway: MotionGateway | None = None,
        motion_catalog: MotionCatalog | None = None,
        dataset_recording_supervisor: Any | None = None,
    ):
        self.config = config
        self.config.ensure_directories()
        self.task_catalog = (
            TaskCatalog.load(config.task_catalog_path)
            if config.task_catalog_path
            else TaskCatalog.default()
        )
        try:
            self.mission_catalog = (
                MissionCatalog.load(config.mission_catalog_path)
                if config.mission_catalog_path
                else MissionCatalog.default()
            )
        except (OSError, MissionError, yaml.YAMLError) as exc:
            raise RuntimeError(f"mission catalog is invalid: {exc}") from exc
        try:
            self.table_catalog = (
                TableCatalog.load(config.table_catalog_path)
                if config.table_catalog_path
                else TableCatalog.empty()
            )
        except (OSError, TableDockingError, yaml.YAMLError) as exc:
            raise RuntimeError(f"table catalog is invalid: {exc}") from exc
        try:
            self.semantic_map_spec = (
                SemanticMapSpec.load(config.semantic_map_config_path)
                if config.semantic_map_config_path
                else None
            )
        except (OSError, SemanticMapError, yaml.YAMLError) as exc:
            raise RuntimeError(f"semantic map is invalid: {exc}") from exc

        catalog = motion_catalog or MotionCatalog.load(config.navigation_catalog_path)
        try:
            self.deployment_report = validate_deployment(
                task_catalog=self.task_catalog,
                mission_catalog=self.mission_catalog,
                motion_catalog=catalog,
                semantic_map=self.semantic_map_spec,
                table_catalog=self.table_catalog,
            )
        except DeploymentError as exc:
            raise RuntimeError(f"deployment catalogs are inconsistent: {exc}") from exc
        try:
            self.semantic_map_status = (
                sync_semantic_map(
                    WorldModel(config.world_model_path), self.semantic_map_spec
                )
                if self.semantic_map_spec is not None
                else {"available": False}
            )
        except SemanticMapError as exc:
            raise RuntimeError(f"semantic map is invalid: {exc}") from exc

        supervisor = ProcessExecutionSupervisor(
            world_model_path=config.world_model_path,
            policy_registry_path=config.policy_registry_path,
            task_catalog_path=config.task_catalog_path,
            emergency_stop_command=config.emergency_stop_command,
            shutdown_grace_seconds=config.shutdown_grace_seconds,
        )
        gateway = motion_gateway or self._build_motion_gateway(
            config,
            self.deployment_report.motion_fingerprint,
        )
        self.policy_registry = self._load_policy_registry(config.policy_registry_path)
        self.dataset_recording = (
            dataset_recording_supervisor
            if dataset_recording_supervisor is not None
            else self._build_dataset_recording_supervisor(config, self.policy_registry)
        )
        self._dataset_operation_lock = threading.RLock()
        self.promotion_reports = self._load_promotion_reports(config.benchmark_report_path)
        episode_recorder = self._build_episode_recorder(config)
        safety_interlock = self._build_safety_interlock(config)
        dry_run_agent_factory = lambda: RobotAgent(
            dry_run=False,
            persistence_path=config.world_model_path,
            task_catalog=self.task_catalog,
            task_backend=DummyTaskBackend(),
        )
        daemon = RobotDaemon(
            agent_factory=lambda *, dry_run: RobotAgent(
                dry_run=dry_run,
                persistence_path=config.world_model_path,
                task_catalog=self.task_catalog,
            ),
            token_store_path=config.token_store_path,
            audit_writer=JsonlAuditWriter(config.audit_log_path),
            session_id=f"robotd-{uuid.uuid4().hex}",
            session_owner=getpass.getuser(),
            real_executor=supervisor,
            require_fresh_observation=config.require_fresh_observation,
            observation_ttl_seconds=config.observation_ttl_seconds,
            motion_gateway=gateway,
            motion_catalog=catalog,
            navigation_timeout_seconds=config.navigation_timeout_seconds,
            docking_timeout_seconds=config.docking_timeout_seconds,
            task_binding_resolver=self._resolve_policy_binding,
            task_metadata_resolver=self._task_safety_metadata,
            episode_recorder=episode_recorder,
            mission_catalog=self.mission_catalog,
            table_catalog=self.table_catalog,
            safety_interlock=safety_interlock,
            require_safety_interlock=config.require_safety_interlock,
            deployment_fingerprint=self.deployment_report.fingerprint,
            experiment_mode=config.experiment_mode,
            task_catalog=self.task_catalog,
            dry_run_agent_factory=dry_run_agent_factory,
        )
        self.robot_daemon = daemon
        self._prepare_socket_path(config.socket_path)
        super().__init__(str(config.socket_path), RuntimeRequestHandler)
        os.chmod(config.socket_path, 0o600)

    @staticmethod
    def _build_motion_gateway(
        config: RuntimeConfig,
        target_fingerprint: str = "",
    ) -> MotionGateway:
        if config.motion_backend in {"", "disabled", "none"}:
            return UnavailableMotionGateway("motion backend is disabled by configuration")
        if config.motion_backend != "ros2":
            return UnavailableMotionGateway(
                f"unknown motion backend: {config.motion_backend}"
            )
        gate_error = RobotRuntimeServer._motion_gate_error(
            config,
            target_fingerprint=target_fingerprint,
        )
        if gate_error:
            return UnavailableMotionGateway(gate_error)
        try:
            from .ros2_gateway import Ros2MotionGateway

            return Ros2MotionGateway(
                navigate_action=config.navigate_action,
                dock_action=config.dock_action,
                cmd_vel_topic=config.cmd_vel_topic,
                heartbeat_topic=config.heartbeat_topic,
            )
        except Exception as exc:
            logger.exception("ROS 2 motion gateway is unavailable")
            return UnavailableMotionGateway(str(exc))

    @staticmethod
    def _load_policy_registry(path: Path | None) -> PolicyRegistry | None:
        if path is None:
            return None
        try:
            return PolicyRegistry.load(path)
        except (OSError, PolicyRegistryError, yaml.YAMLError) as exc:
            logger.error("policy registry is unavailable: %s", exc)
            return None

    @staticmethod
    def _build_dataset_recording_supervisor(
        config: RuntimeConfig,
        policy_registry: PolicyRegistry | None,
    ) -> DatasetRecordingSupervisor:
        path = config.dataset_recording_config_path
        if path is None:
            return DatasetRecordingSupervisor(
                config=None,
                policy_registry=policy_registry,
                state_dir=config.state_dir,
                unavailable_reason=(
                    "ROBOT_AGENT_DATASET_RECORDING_CONFIG is not configured"
                ),
            )
        try:
            recording_config = DatasetRecordingConfig.load(path)
        except (OSError, DatasetRecordingError, yaml.YAMLError) as exc:
            logger.error("dataset recording configuration is unavailable: %s", exc)
            return DatasetRecordingSupervisor(
                config=None,
                policy_registry=policy_registry,
                state_dir=config.state_dir,
                unavailable_reason=f"dataset recording config is invalid: {exc}",
            )
        reason = "" if policy_registry is not None else "policy registry is unavailable"
        return DatasetRecordingSupervisor(
            config=recording_config,
            policy_registry=policy_registry,
            state_dir=config.state_dir,
            unavailable_reason=reason,
        )

    @staticmethod
    def _load_promotion_reports(path: Path | None) -> dict[str, BenchmarkReport]:
        if path is None:
            return {}
        try:
            candidates = sorted(path.glob("*.json")) if path.is_dir() else [path]
            reports: dict[str, BenchmarkReport] = {}
            for candidate in candidates:
                report = load_report(candidate)
                if report.task_id in reports:
                    raise BenchmarkError(
                        f"multiple benchmark reports cover task {report.task_id!r}"
                    )
                reports[report.task_id] = report
            return reports
        except (OSError, BenchmarkError) as exc:
            logger.error("benchmark promotion reports are unavailable: %s", exc)
            return {}

    @staticmethod
    def _build_episode_recorder(config: RuntimeConfig) -> EpisodeRecorder | None:
        if config.episode_dir is None:
            return None
        try:
            context = (
                EpisodeContext.load(config.episode_context_path)
                if config.episode_context_path is not None
                else EpisodeContext.empty()
            )
            return EpisodeRecorder(config.episode_dir, context=context)
        except (OSError, EpisodeError) as exc:
            raise RuntimeError(f"episode recorder configuration is invalid: {exc}") from exc

    @staticmethod
    def _build_safety_interlock(config: RuntimeConfig) -> Any:
        if not config.require_safety_interlock:
            return UnavailableSafetyInterlock("safety interlock is disabled by configuration")
        if config.safety_policy_path is None or config.safety_snapshot_path is None:
            return UnavailableSafetyInterlock(
                "ROBOT_AGENT_SAFETY_POLICY and ROBOT_AGENT_SAFETY_SNAPSHOT are required"
            )
        try:
            return FileSafetyInterlock(
                config.safety_snapshot_path,
                SafetyPolicy.load(config.safety_policy_path),
            )
        except (OSError, SafetyError, yaml.YAMLError) as exc:
            return UnavailableSafetyInterlock(f"safety interlock is invalid: {exc}")

    def _resolve_policy_binding(self, task_id: str) -> dict[str, Any]:
        task = self.task_catalog.get(task_id)
        if task is None:
            raise PolicyRegistryError(f"task {task_id!r} is not present in the task catalog")
        if task.executor == "table_docking":
            return {
                "task_fingerprint": task.fingerprint(),
                "task_executor": task.executor,
                "deployment_fingerprint": self.deployment_report.fingerprint,
            }
        if task.executor != "learned_policy":
            raise PolicyRegistryError(
                f"task executor {task.executor!r} is unavailable on the current hardware"
            )
        if self.policy_registry is None:
            raise PolicyRegistryError("ROBOT_AGENT_POLICY_REGISTRY is not configured or invalid")

        binding = {
            "task_fingerprint": task.fingerprint(),
            "task_executor": task.executor,
            "deployment_fingerprint": self.deployment_report.fingerprint,
            **self.policy_registry.resolve_binding(task_id),
        }
        if self.config.require_policy_benchmark:
            promotion_report = self.promotion_reports.get(task_id)
            if promotion_report is None:
                raise PolicyRegistryError(
                    "a valid ROBOT_AGENT_BENCHMARK_REPORT is required for learned policy"
                )
            if self.config.benchmark_report_max_age_seconds <= 0:
                raise PolicyRegistryError("benchmark report maximum age must be positive")
            if (
                self.config.require_policy_benchmark_evidence
                and not promotion_report.required_evidence_kinds
            ):
                raise PolicyRegistryError(
                    "learned policy benchmark must bind required episode evidence"
                )
            if promotion_report.binding_equals.get("policy_fingerprint") != binding.get(
                "policy_fingerprint"
            ):
                raise PolicyRegistryError(
                    "learned policy benchmark must bind the exact policy_fingerprint"
                )
            try:
                promotion_report.require_candidate(
                    task_id=task_id,
                    task_fingerprint=task.fingerprint(),
                    execution_binding=binding,
                    max_age_seconds=self.config.benchmark_report_max_age_seconds,
                )
            except BenchmarkError as exc:
                raise PolicyRegistryError(f"learned policy benchmark gate failed: {exc}") from exc
            binding.update(
                {
                    "benchmark_id": promotion_report.benchmark_id,
                    "benchmark_fingerprint": promotion_report.benchmark_fingerprint,
                    "benchmark_report_sha256": promotion_report.report_sha256,
                    "benchmark_evidence_kinds": list(
                        promotion_report.required_evidence_kinds
                    ),
                }
            )
        return binding

    def _task_safety_metadata(self, task_id: str) -> TaskSafetyMetadata | None:
        task = self.task_catalog.get(task_id)
        if task is None:
            return None
        return _task_safety_metadata(task)

    @staticmethod
    def _motion_gate_error(
        config: RuntimeConfig,
        *,
        target_fingerprint: str = "",
    ) -> str | None:
        if config.bringup_manifest_path is None:
            return "ROBOT_AGENT_BRINGUP_MANIFEST is required for the ROS 2 backend"
        if config.hil_report_path is None:
            return "ROBOT_AGENT_HIL_REPORT is required for the ROS 2 backend"
        try:
            from .bringup import BringupManifest

            errors = BringupManifest.load(config.bringup_manifest_path).validate()
            if errors:
                return "bringup preflight failed: " + "; ".join(errors)
            if target_fingerprint:
                from .acceptance import load_acceptance_report

                report = load_acceptance_report(config.hil_report_path).require(
                    report_type="nav2_hil",
                    target_fingerprint=target_fingerprint,
                )
            else:
                report = json.loads(config.hil_report_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict) or report.get("passed") is not True:
                return "HIL acceptance report has not passed"
            scenarios = report.get("scenarios")
            required = {"heartbeat_loss", "robotd_killed", "nav2_killed", "sensor_loss"}
            passed = {
                item.get("scenario")
                for item in scenarios
                if isinstance(item, dict) and item.get("passed") is True
            } if isinstance(scenarios, list) else set()
            missing = sorted(required - passed)
            if missing:
                return "HIL report is missing passing scenarios: " + ", ".join(missing)
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            return f"motion safety gate could not be verified: {exc}"
        return None

    @staticmethod
    def _prepare_socket_path(path: Path) -> None:
        if not path.exists():
            return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.2)
                probe.connect(str(path))
        except OSError:
            path.unlink(missing_ok=True)
            return
        raise RuntimeError(f"robotd is already listening on {path}")

    def dispatch(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            return self._error("request must be an object")
        request_id = str(request.get("id", ""))[:128].replace("\n", " ").replace("\r", " ")
        command = request.get("command")
        params = request.get("params", {})
        if not isinstance(command, str) or not command:
            return self._error("command must be a non-empty string", request_id)
        if not isinstance(params, dict):
            return self._error("params must be an object", request_id)

        if command in _RECORDING_GATED_COMMANDS:
            with self._dataset_operation_lock:
                return self._dispatch_command(command, params, request_id)
        return self._dispatch_command(command, params, request_id)

    def _dispatch_command(
        self,
        command: str,
        params: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:

        logger.info("robotd command=%r request_id=%r", command, request_id)
        result: Any
        if self.dataset_recording.active and command in _RECORDING_GATED_COMMANDS:
            result = {
                "success": False,
                "state": self.robot_daemon.status()["state"],
                "message": (
                    "dataset recording is active; stop recording before running "
                    f"{command}"
                ),
                "data": {"dataset_recording": self.dataset_recording.status()["data"]},
            }
        elif command in {"health", "status"}:
            result = {
                "success": True,
                **self.robot_daemon.status(),
                "dataset_recording": self.dataset_recording.status()["data"],
            }
        elif command == "capabilities":
            result = {
                "success": True,
                "tasks": {
                    task_id: asdict(_task_safety_metadata(task))
                    for task_id, task in self.task_catalog.tasks.items()
                },
                "task_catalog": self.task_catalog.public_view(),
                "mission_catalog": self.mission_catalog.public_view(),
                "table_catalog": self.table_catalog.public_view(),
                "policy_registry": (
                    self.policy_registry.public_view()
                    if self.policy_registry is not None
                    else {"available": False}
                ),
                "benchmark_gate": {
                    "required": self.config.require_policy_benchmark,
                    "available": bool(self.promotion_reports),
                    "reports": {
                        task_id: {
                            "benchmark_id": report.benchmark_id,
                            "passed": report.passed,
                            "report_sha256": report.report_sha256,
                        }
                        for task_id, report in sorted(self.promotion_reports.items())
                    },
                },
                "semantic_map": dict(self.semantic_map_status),
                "deployment": self.deployment_report.public_view(),
                "dataset_recording": self.dataset_recording.capabilities(),
            }
        elif command == "dataset_recording_status":
            result = self.dataset_recording.status()
        elif command == "dataset_recording_validate":
            result = self.dataset_recording.validate(dict(params))
        elif command == "dataset_recording_start":
            recording_params = dict(params)
            confirmation = self._string(
                recording_params,
                "operator_confirmation",
                default="",
            )
            recording_params.pop("operator_confirmation", None)
            daemon_status = self.robot_daemon.status()
            if daemon_status["state"] != "DRY_RUN" or daemon_status["hardware_active"]:
                result = {
                    "success": False,
                    "state": daemon_status["state"],
                    "message": (
                        "dataset recording requires robotd DRY_RUN with no active hardware task"
                    ),
                    "data": {"dataset_recording": self.dataset_recording.status()["data"]},
                }
            else:
                result = self.dataset_recording.start(
                    recording_params,
                    operator_confirmation=confirmation,
                )
        elif command == "dataset_recording_stop":
            result = self.dataset_recording.stop(reason="operator_recording_stop")
        elif command == "motion_catalog":
            result = self.robot_daemon.motion_catalog()
        elif command == "mission_catalog":
            result = self.robot_daemon.mission_catalog()
        elif command == "mission_dry_run":
            result = self.robot_daemon.dry_run_mission(
                self._string(params, "mission_id")
            )
        elif command == "mission_arm":
            result = self.robot_daemon.request_mission_arm(
                self._string(params, "mission_id"),
                operator_confirmation=self._string(params, "operator_confirmation"),
            )
        elif command == "mission_execute":
            result = self.robot_daemon.execute_mission(
                self._string(params, "mission_id"),
                token=self._optional_string(params, "token"),
            )
        elif command == "world_state":
            result = self.robot_daemon.world_state()
        elif command == "world_events":
            result = self.robot_daemon.world_events(
                since_sequence=self._integer(params, "since_sequence", default=0),
                limit=self._integer(params, "limit", default=100),
            )
        elif command == "motion_dry_run":
            result = self.robot_daemon.dry_run_motion(
                self._string(params, "operation"),
                self._string(params, "target"),
            )
        elif command == "motion_arm":
            result = self.robot_daemon.request_motion_arm(
                self._string(params, "operation"),
                self._string(params, "target"),
                operator_confirmation=self._string(params, "operator_confirmation"),
            )
        elif command == "motion_execute":
            result = self.robot_daemon.execute_motion(
                self._string(params, "operation"),
                self._string(params, "target"),
                token=self._optional_string(params, "token"),
            )
        elif command == "plan":
            result = self.robot_daemon.plan(self._string(params, "goal"))
        elif command == "visual_plan_inspect":
            result = self.robot_daemon.inspect_visual_plan(self._string(params, "goal"))
        elif command == "visual_plan_arm":
            result = self.robot_daemon.request_visual_plan_arm(
                self._string(params, "plan_id"),
                operator_confirmation=self._string(params, "operator_confirmation"),
            )
        elif command == "visual_plan_execute":
            result = self.robot_daemon.execute_visual_plan(
                self._string(params, "plan_id"),
                token=self._optional_string(params, "token"),
            )
        elif command == "dry_run":
            result = self.robot_daemon.dry_run(self._string(params, "goal"))
        elif command == "observe":
            result = self.robot_daemon.observe(
                self._string(params, "instruction", default="描述当前环境")
            )
        elif command == "arm":
            result = self.robot_daemon.request_arm(
                self._string(params, "task_id"),
                {},
                operator_confirmation=self._string(params, "operator_confirmation"),
            )
        elif command == "execute":
            result = self.robot_daemon.execute(
                self._string(params, "task_id"),
                {},
                token=self._optional_string(params, "token"),
            )
        elif command == "pause":
            recording = self.dataset_recording.stop(reason="operator_pause")
            result = self.robot_daemon.pause()
            result.data["dataset_recording"] = recording.get("data", {})
        elif command == "stop":
            recording = self.dataset_recording.stop(reason="operator_stop")
            result = self.robot_daemon.stop()
            result.data["dataset_recording"] = recording.get("data", {})
        elif command == "estop":
            recording = self.dataset_recording.stop(reason="emergency_stop")
            result = self.robot_daemon.estop()
            result.data["dataset_recording"] = recording.get("data", {})
        elif command == "reset":
            result = self.robot_daemon.reset(
                operator_confirmation=self._string(
                    params,
                    "operator_confirmation",
                    default="",
                )
            )
        else:
            return self._error(f"unknown command: {command}", request_id)

        response = to_jsonable(result)
        if not isinstance(response, dict):
            response = {"success": False, "error": "unexpected command result"}
        response["request_id"] = request_id
        response["boundary"] = "standalone-robotd"
        return response

    @staticmethod
    def _string(params: dict[str, Any], key: str, default: str | None = None) -> str:
        value = params.get(key, default)
        if not isinstance(value, str) or (default is None and not value.strip()):
            raise ValueError(f"{key} must be a non-empty string")
        return value

    @staticmethod
    def _optional_string(params: dict[str, Any], key: str) -> str | None:
        value = params.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value

    @staticmethod
    def _integer(params: dict[str, Any], key: str, default: int) -> int:
        value = params.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        return value

    @staticmethod
    def _error(message: str, request_id: str = "") -> dict[str, Any]:
        return {
            "success": False,
            "request_id": request_id,
            "boundary": "standalone-robotd",
            "error": message,
        }

    def server_close(self) -> None:
        try:
            self.dataset_recording.close()
            self.robot_daemon.estop()
            self.robot_daemon.close()
            super().server_close()
        finally:
            self.config.socket_path.unlink(missing_ok=True)


def _task_safety_metadata(task: TaskDefinition) -> TaskSafetyMetadata:
    return TaskSafetyMetadata(
        task_id=task.task_id,
        name=task.name,
        aliases=task.aliases,
        args=dict(task.args),
        requires_observation=task.requires_observation,
        requires_token=task.requires_token,
        max_runtime_seconds=task.max_runtime_seconds,
        allowed_modes=task.allowed_modes,
        stop_conditions=task.stop_conditions,
    )



def main() -> None:
    logging.basicConfig(
        level=os.environ.get("ROBOT_AGENT_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    os.umask(0o077)
    config = RuntimeConfig.from_env()
    server = RobotRuntimeServer(config)

    def request_shutdown(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    logger.info("robotd listening on %s", config.socket_path)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()


__all__ = ["RobotRuntimeServer", "RuntimeRequestHandler", "main"]
