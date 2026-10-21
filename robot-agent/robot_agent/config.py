from __future__ import annotations

import os
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path


_EXPERIMENT_MODES = frozenset(
    {"direct_vla", "rgb_precondition_vla", "rgb_pre_post_vla"}
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_command(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    return tuple(shlex.split(raw)) if raw else ()


def _state_home() -> Path:
    configured = os.environ.get("ROBOT_AGENT_STATE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state_home).expanduser() if xdg_state_home else Path.home() / ".local" / "state"
    return base / "robot-agent"


def _runtime_home() -> Path:
    configured = os.environ.get("ROBOT_AGENT_RUNTIME_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        return Path(xdg_runtime_dir).expanduser() / "robot-agent"
    return Path(tempfile.gettempdir()) / f"robot-agent-{os.getuid()}"


@dataclass(frozen=True)
class RuntimeConfig:
    state_dir: Path
    runtime_dir: Path
    socket_path: Path
    token_store_path: Path
    audit_log_path: Path
    world_model_path: Path
    token_cache_path: Path
    require_fresh_observation: bool
    observation_ttl_seconds: int
    shutdown_grace_seconds: float
    emergency_stop_command: tuple[str, ...]
    motion_backend: str = "disabled"
    navigation_catalog_path: Path = Path("/etc/robot-agent/navigation.yaml")
    navigate_action: str = "navigate_to_pose"
    dock_action: str = "dock_robot"
    cmd_vel_topic: str = "cmd_vel"
    heartbeat_topic: str = "robot_agent/heartbeat"
    navigation_timeout_seconds: float = 300.0
    docking_timeout_seconds: float = 180.0
    bringup_manifest_path: Path | None = None
    hil_report_path: Path | None = None
    policy_registry_path: Path | None = None
    task_catalog_path: Path | None = None
    episode_dir: Path | None = None
    episode_context_path: Path | None = None
    require_policy_benchmark: bool = False
    require_policy_benchmark_evidence: bool = False
    benchmark_report_path: Path | None = None
    benchmark_report_max_age_seconds: float = 30 * 24 * 60 * 60
    mission_catalog_path: Path | None = None
    table_catalog_path: Path | None = None
    require_safety_interlock: bool = False
    safety_policy_path: Path | None = None
    safety_snapshot_path: Path | None = None
    semantic_map_config_path: Path | None = None
    experiment_mode: str = "rgb_pre_post_vla"
    dataset_recording_config_path: Path | None = None

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        state_dir = _state_home()
        runtime_dir = _runtime_home()
        socket_path = Path(
            os.environ.get("ROBOT_AGENT_SOCKET", str(runtime_dir / "robotd.sock"))
        ).expanduser()
        bringup_manifest = os.environ.get("ROBOT_AGENT_BRINGUP_MANIFEST", "").strip()
        hil_report = os.environ.get("ROBOT_AGENT_HIL_REPORT", "").strip()
        policy_registry = os.environ.get("ROBOT_AGENT_POLICY_REGISTRY", "").strip()
        task_catalog = os.environ.get("ROBOT_AGENT_TASK_CATALOG", "").strip()
        episode_context = os.environ.get("ROBOT_AGENT_EPISODE_CONTEXT", "").strip()
        benchmark_report = os.environ.get("ROBOT_AGENT_BENCHMARK_REPORT", "").strip()
        mission_catalog = os.environ.get("ROBOT_AGENT_MISSION_CATALOG", "").strip()
        table_catalog = os.environ.get("ROBOT_AGENT_TABLE_CATALOG", "").strip()
        safety_policy = os.environ.get("ROBOT_AGENT_SAFETY_POLICY", "").strip()
        safety_snapshot = os.environ.get("ROBOT_AGENT_SAFETY_SNAPSHOT", "").strip()
        semantic_map_config = os.environ.get(
            "ROBOT_AGENT_SEMANTIC_MAP_CONFIG", ""
        ).strip()
        dataset_recording_config = os.environ.get(
            "ROBOT_AGENT_DATASET_RECORDING_CONFIG", ""
        ).strip()
        experiment_mode = os.environ.get(
            "ROBOT_AGENT_EXPERIMENT_MODE", "rgb_pre_post_vla"
        ).strip().lower()
        if experiment_mode not in _EXPERIMENT_MODES:
            raise ValueError(
                "ROBOT_AGENT_EXPERIMENT_MODE must be one of "
                "direct_vla, rgb_precondition_vla, rgb_pre_post_vla"
            )
        return cls(
            state_dir=state_dir,
            runtime_dir=runtime_dir,
            socket_path=socket_path,
            token_store_path=state_dir / "tokens.json",
            audit_log_path=state_dir / "audit.jsonl",
            world_model_path=state_dir / "world-model.yaml",
            token_cache_path=state_dir / "armed-token.json",
            require_fresh_observation=_env_bool("ROBOT_AGENT_REQUIRE_OBSERVATION", True),
            observation_ttl_seconds=int(os.environ.get("ROBOT_AGENT_OBSERVATION_TTL", "30")),
            shutdown_grace_seconds=float(os.environ.get("ROBOT_AGENT_SHUTDOWN_GRACE", "2.0")),
            emergency_stop_command=_env_command("ROBOT_AGENT_EMERGENCY_STOP_COMMAND"),
            motion_backend=os.environ.get("ROBOT_AGENT_MOTION_BACKEND", "disabled").strip().lower(),
            navigation_catalog_path=Path(
                os.environ.get(
                    "ROBOT_AGENT_NAVIGATION_CATALOG",
                    "/etc/robot-agent/navigation.yaml",
                )
            ).expanduser(),
            navigate_action=os.environ.get("ROBOT_AGENT_NAVIGATE_ACTION", "navigate_to_pose"),
            dock_action=os.environ.get("ROBOT_AGENT_DOCK_ACTION", "dock_robot"),
            cmd_vel_topic=os.environ.get("ROBOT_AGENT_CMD_VEL_TOPIC", "cmd_vel"),
            heartbeat_topic=os.environ.get(
                "ROBOT_AGENT_HEARTBEAT_TOPIC",
                "robot_agent/heartbeat",
            ),
            navigation_timeout_seconds=float(
                os.environ.get("ROBOT_AGENT_NAVIGATION_TIMEOUT", "300")
            ),
            docking_timeout_seconds=float(
                os.environ.get("ROBOT_AGENT_DOCKING_TIMEOUT", "180")
            ),
            bringup_manifest_path=(
                Path(bringup_manifest).expanduser() if bringup_manifest else None
            ),
            hil_report_path=Path(hil_report).expanduser() if hil_report else None,
            policy_registry_path=(
                Path(policy_registry).expanduser() if policy_registry else None
            ),
            task_catalog_path=Path(task_catalog).expanduser() if task_catalog else None,
            episode_dir=Path(
                os.environ.get("ROBOT_AGENT_EPISODE_DIR", str(state_dir / "episodes"))
            ).expanduser(),
            episode_context_path=(
                Path(episode_context).expanduser() if episode_context else None
            ),
            require_policy_benchmark=_env_bool(
                "ROBOT_AGENT_REQUIRE_POLICY_BENCHMARK",
                True,
            ),
            require_policy_benchmark_evidence=_env_bool(
                "ROBOT_AGENT_REQUIRE_POLICY_EVIDENCE",
                True,
            ),
            benchmark_report_path=(
                Path(benchmark_report).expanduser() if benchmark_report else None
            ),
            benchmark_report_max_age_seconds=float(
                os.environ.get(
                    "ROBOT_AGENT_BENCHMARK_MAX_AGE",
                    str(30 * 24 * 60 * 60),
                )
            ),
            mission_catalog_path=(
                Path(mission_catalog).expanduser() if mission_catalog else None
            ),
            table_catalog_path=(
                Path(table_catalog).expanduser() if table_catalog else None
            ),
            require_safety_interlock=_env_bool(
                "ROBOT_AGENT_REQUIRE_SAFETY_INTERLOCK",
                True,
            ),
            safety_policy_path=(
                Path(safety_policy).expanduser() if safety_policy else None
            ),
            safety_snapshot_path=(
                Path(safety_snapshot).expanduser() if safety_snapshot else None
            ),
            semantic_map_config_path=(
                Path(semantic_map_config).expanduser()
                if semantic_map_config
                else None
            ),
            experiment_mode=experiment_mode,
            dataset_recording_config_path=(
                Path(dataset_recording_config).expanduser()
                if dataset_recording_config
                else None
            ),
        )

    def ensure_directories(self) -> None:
        for directory in (self.state_dir, self.runtime_dir):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)


__all__ = ["RuntimeConfig"]
