from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import math
import os
import secrets
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

try:
    from .audit import build_audit_event, token_hash
    from .contract import (
        ARMED_REAL,
        AWAITING_CONFIRMATION,
        DISABLED,
        DRY_RUN,
        ERROR,
        ESTOP,
        EXECUTING,
        EXPIRED_TOKEN,
        HARDWARE_NOT_ARMED,
        ILLEGAL_STATE_TRANSITION,
        INVALID_TOKEN,
        MISSING_TOKEN,
        OBSERVE_ONLY,
        OBSERVATION_REQUIRED,
        PAUSED,
        SESSION_LOCKED,
        SAFETY_INTERLOCK_BLOCKED,
        STOPPING,
        TASK_NOT_ALLOWLISTED,
        UNSAFE_ARGUMENTS,
        AuditEvent,
        CommandResult,
        Refusal,
        TokenMetadata,
        validate_transition,
    )
    from .skill_flow import get_task_safety_metadata
    from .motion import MotionCatalog, MotionGateway, MotionResult, UnavailableMotionGateway
    from .missions import (
        MissionAction,
        MissionActionResult,
        MissionCatalog,
        MissionDefinition,
        MissionExecutor,
    )
    from .safety import UnavailableSafetyInterlock
    from .table_docking import (
        TableCatalog,
        TableDockingController,
        TableDockingError,
        VlmTableObserver,
        ChassisDockActuator,
    )
    from .visual_task_planner import VisualPlanError, VisualTaskPlan, VisualTaskPlanner
except ImportError:
    _MODULE_DIR = Path(__file__).resolve().parent

    def _load_local_module(module_name: str, filename: str):
        path = _MODULE_DIR / filename
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    _contract = sys.modules.get("robot_agent_contract") or _load_local_module(
        "robot_agent_contract_for_robotd",
        "contract.py",
    )
    _audit = _load_local_module("robot_agent_audit_for_robotd", "audit.py")
    _skill_flow = _load_local_module("robot_agent_skill_flow_for_robotd", "skill_flow.py")

    ARMED_REAL = _contract.ARMED_REAL
    AWAITING_CONFIRMATION = _contract.AWAITING_CONFIRMATION
    DISABLED = _contract.DISABLED
    DRY_RUN = _contract.DRY_RUN
    ERROR = _contract.ERROR
    ESTOP = _contract.ESTOP
    EXECUTING = _contract.EXECUTING
    EXPIRED_TOKEN = _contract.EXPIRED_TOKEN
    HARDWARE_NOT_ARMED = _contract.HARDWARE_NOT_ARMED
    ILLEGAL_STATE_TRANSITION = _contract.ILLEGAL_STATE_TRANSITION
    INVALID_TOKEN = _contract.INVALID_TOKEN
    MISSING_TOKEN = _contract.MISSING_TOKEN
    OBSERVE_ONLY = _contract.OBSERVE_ONLY
    OBSERVATION_REQUIRED = _contract.OBSERVATION_REQUIRED
    PAUSED = _contract.PAUSED
    SESSION_LOCKED = _contract.SESSION_LOCKED
    SAFETY_INTERLOCK_BLOCKED = _contract.SAFETY_INTERLOCK_BLOCKED
    STOPPING = _contract.STOPPING
    TASK_NOT_ALLOWLISTED = _contract.TASK_NOT_ALLOWLISTED
    UNSAFE_ARGUMENTS = _contract.UNSAFE_ARGUMENTS
    AuditEvent = _contract.AuditEvent
    CommandResult = _contract.CommandResult
    Refusal = _contract.Refusal
    TokenMetadata = _contract.TokenMetadata
    validate_transition = _contract.validate_transition
    build_audit_event = _audit.build_audit_event
    token_hash = _audit.token_hash
    get_task_safety_metadata = _skill_flow.get_task_safety_metadata
    _motion = _load_local_module("robot_agent_motion_for_robotd", "motion.py")
    MotionCatalog = _motion.MotionCatalog
    MotionGateway = _motion.MotionGateway
    MotionResult = _motion.MotionResult
    UnavailableMotionGateway = _motion.UnavailableMotionGateway
    _missions = _load_local_module("robot_agent_missions_for_robotd", "missions.py")
    MissionAction = _missions.MissionAction
    MissionActionResult = _missions.MissionActionResult
    MissionCatalog = _missions.MissionCatalog
    MissionDefinition = _missions.MissionDefinition
    MissionExecutor = _missions.MissionExecutor
    _safety = _load_local_module("robot_agent_safety_for_robotd", "safety.py")
    UnavailableSafetyInterlock = _safety.UnavailableSafetyInterlock
    from robot_agent.table_docking import (
        ChassisDockActuator,
        TableCatalog,
        TableDockingController,
        TableDockingError,
        VlmTableObserver,
    )
    from robot_agent.visual_task_planner import VisualPlanError, VisualTaskPlan, VisualTaskPlanner


_EXPERIMENT_MODES = frozenset(
    {"direct_vla", "rgb_precondition_vla", "rgb_pre_post_vla"}
)


AgentFactory = Callable[..., Any]
AuditWriter = Callable[[AuditEvent], None]
Clock = Callable[[], float]
TaskBindingResolver = Callable[[str], dict[str, Any]]
TaskMetadataResolver = Callable[[str], Any | None]


@dataclass
class StoredToken:
    token_id: str
    token_hash: str
    task_id: str
    args: dict[str, Any]
    issued_to: str
    issued_at: float
    expires_at: float
    scopes: list[str]
    session_id: str
    consumed: bool = False

    def metadata(self) -> TokenMetadata:
        return TokenMetadata(
            token_id=self.token_id,
            issued_to=self.issued_to,
            issued_at=self.issued_at,
            expires_at=self.expires_at,
            scopes=list(self.scopes),
            session_id=self.session_id,
        )


class JsonTokenStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    @contextmanager
    def _exclusive_file_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(fd, "rb+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"tokens": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            finally:
                raise
        tmp_path.replace(self.path)

    def save(self, token: StoredToken) -> None:
        with self._lock:
            with self._exclusive_file_lock():
                payload = self._read()
                payload.setdefault("tokens", {})[token.token_id] = asdict(token)
                self._write(payload)

    def find_by_secret(self, token_secret: str) -> StoredToken | None:
        with self._lock:
            with self._exclusive_file_lock():
                wanted_hash = token_hash(token_secret)
                for raw in self._read().get("tokens", {}).values():
                    if raw.get("token_hash") == wanted_hash:
                        return StoredToken(**raw)
                return None

    def mark_consumed(self, token_id: str) -> None:
        with self._lock:
            with self._exclusive_file_lock():
                payload = self._read()
                raw = payload.get("tokens", {}).get(token_id)
                if raw is None:
                    return
                raw["consumed"] = True
                self._write(payload)

    def consume_if_valid(
        self,
        token_secret: str,
        *,
        task_id: str,
        args: dict[str, Any],
        session_id: str,
        active_token_id: str | None,
        now: float,
        state: str,
    ) -> tuple[StoredToken | None, Refusal | None]:
        with self._lock:
            with self._exclusive_file_lock():
                payload = self._read()
                wanted_hash = token_hash(token_secret)
                matched_raw = None
                for raw in payload.get("tokens", {}).values():
                    if raw.get("token_hash") == wanted_hash:
                        matched_raw = raw
                        break
                if matched_raw is None:
                    return None, Refusal(INVALID_TOKEN, "token is unknown", state)
                stored_token = StoredToken(**matched_raw)
                refusal = self._token_refusal(stored_token, task_id, args, session_id, active_token_id, now, state)
                if refusal is not None:
                    return stored_token, refusal
                matched_raw["consumed"] = True
                self._write(payload)
                return stored_token, None

    def revoke_session(self, session_id: str) -> None:
        with self._lock:
            with self._exclusive_file_lock():
                payload = self._read()
                changed = False
                for raw in payload.get("tokens", {}).values():
                    if raw.get("session_id") == session_id and not raw.get("consumed", False):
                        raw["consumed"] = True
                        changed = True
                if changed:
                    self._write(payload)

    def count(self) -> int:
        with self._lock:
            with self._exclusive_file_lock():
                return len(self._read().get("tokens", {}))

    @staticmethod
    def _token_refusal(
        stored_token: StoredToken,
        task_id: str,
        args: dict[str, Any],
        session_id: str,
        active_token_id: str | None,
        now: float,
        state: str,
    ) -> Refusal | None:
        if stored_token.consumed:
            return Refusal(INVALID_TOKEN, "token was already consumed", state)
        if stored_token.expires_at <= now:
            return Refusal(EXPIRED_TOKEN, "token expired", state)
        if stored_token.session_id != session_id:
            return Refusal(INVALID_TOKEN, "token belongs to a different session", state)
        if stored_token.token_id != active_token_id:
            return Refusal(INVALID_TOKEN, "token is not the currently armed token", state)
        if stored_token.task_id != task_id or stored_token.args != args:
            return Refusal(INVALID_TOKEN, "token binding does not match command", state)
        return None


class RobotDaemon:
    def __init__(
        self,
        *,
        agent_factory: AgentFactory | None = None,
        token_store: Any | None = None,
        token_store_path: str | Path | None = None,
        audit_writer: AuditWriter | None = None,
        clock: Clock | None = None,
        session_id: str | None = None,
        session_owner: str = "operator",
        token_ttl_seconds: int = 60,
        real_executor: Any | None = None,
        require_fresh_observation: bool = False,
        observation_ttl_seconds: int = 30,
        motion_gateway: MotionGateway | None = None,
        motion_catalog: MotionCatalog | None = None,
        navigation_timeout_seconds: float = 300.0,
        docking_timeout_seconds: float = 180.0,
        task_binding_resolver: TaskBindingResolver | None = None,
        task_metadata_resolver: TaskMetadataResolver | None = None,
        episode_recorder: Any | None = None,
        mission_catalog: Any | None = None,
        table_catalog: Any | None = None,
        table_docking_runner: Callable[[Any, float], Any] | None = None,
        safety_interlock: Any | None = None,
        require_safety_interlock: bool = False,
        deployment_fingerprint: str = "",
        experiment_mode: str = "rgb_pre_post_vla",
        task_catalog: Any | None = None,
        dry_run_agent_factory: Callable[[], Any] | None = None,
    ):
        self._agent_factory = agent_factory or self._default_agent_factory
        self._clock = clock or self._default_clock
        self._audit_writer = audit_writer
        self._token_ttl_seconds = token_ttl_seconds
        self._real_executor = real_executor
        self._require_fresh_observation = require_fresh_observation
        self._observation_ttl_seconds = observation_ttl_seconds
        self._motion_gateway = motion_gateway or UnavailableMotionGateway()
        self._motion_catalog = motion_catalog or MotionCatalog.empty()
        self._navigation_timeout_seconds = navigation_timeout_seconds
        self._docking_timeout_seconds = docking_timeout_seconds
        self._task_binding_resolver = task_binding_resolver or (lambda _task_id: {})
        self._task_metadata_resolver = task_metadata_resolver or get_task_safety_metadata
        self._episode_recorder = episode_recorder
        self._mission_catalog = mission_catalog or MissionCatalog.empty()
        self._table_catalog = table_catalog or TableCatalog.empty()
        self._table_docking_runner = table_docking_runner
        self._safety_interlock = safety_interlock or UnavailableSafetyInterlock()
        self._require_safety_interlock = require_safety_interlock
        if experiment_mode not in _EXPERIMENT_MODES:
            raise ValueError(
                "experiment_mode must be one of direct_vla, "
                "rgb_precondition_vla, rgb_pre_post_vla"
            )
        self._experiment_mode = experiment_mode
        self._task_catalog = task_catalog
        self._dry_run_agent_factory = dry_run_agent_factory
        if deployment_fingerprint and (
            len(deployment_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in deployment_fingerprint)
        ):
            raise ValueError("deployment_fingerprint must be empty or SHA-256")
        self._deployment_fingerprint = deployment_fingerprint
        self._last_safety_decisions: dict[str, dict[str, Any]] = {}
        self._state_lock = threading.RLock()
        self.session_id = session_id or f"session-{uuid.uuid4().hex}"
        self.session_owner = session_owner
        self.state = DRY_RUN
        self.active_task: str | None = None
        self.active_token_id: str | None = None
        self.active_execution_binding: dict[str, Any] = {}
        self.last_observation_at: float | None = None
        self._visual_plans: dict[str, VisualTaskPlan] = {}

        if token_store is not None:
            self._token_store = token_store
        else:
            store_path = token_store_path or Path(tempfile.gettempdir()) / "robot-agent-robotd-tokens.json"
            self._token_store = JsonTokenStore(store_path)

    @staticmethod
    def _default_clock() -> float:
        import time

        return time.time()

    @staticmethod
    def _default_agent_factory(*, dry_run: bool):
        try:
            from .agent import RobotAgent
        except ImportError:
            from robot_agent.agent import RobotAgent

        return RobotAgent(dry_run=dry_run)

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "state": self.state,
                "session_id": self.session_id,
                "session_owner": self.session_owner,
                "active_task": self.active_task,
                "active_token_id": self.active_token_id,
                "active_execution_binding": dict(self.active_execution_binding),
                "last_observation_at": self.last_observation_at,
                "executor_active": bool(
                    self._real_executor is not None
                    and getattr(self._real_executor, "active", False)
                ),
                "hardware_active": self._hardware_active(),
                "motion": self._motion_gateway.health(),
                "motion_catalog": self._motion_catalog.public_view(),
                "deployment_fingerprint": self._deployment_fingerprint or None,
                "token_count": self._token_count(),
                "visual_plan_count": len(self._visual_plans),
                "safety_interlock": {
                    "required": self._require_safety_interlock,
                    **self._safety_interlock.health(now=self._clock()),
                },
            }

    def motion_catalog(self) -> CommandResult:
        return CommandResult(
            success=True,
            state=self.state,
            message="named motion catalog",
            data=self._motion_catalog.public_view(),
        )

    def mission_catalog(self) -> CommandResult:
        return CommandResult(
            success=True,
            state=self.state,
            message="bounded mission catalog",
            data=self._mission_catalog.public_view(),
        )

    def dry_run_mission(self, mission_id: str) -> CommandResult:
        mission = self._mission_catalog.get(mission_id)
        if mission is None:
            return self._refuse(
                TASK_NOT_ALLOWLISTED,
                f"mission:{mission_id}",
                "mission is not allowlisted",
            )
        try:
            binding = self._resolve_mission_binding(mission)
        except Exception as exc:
            return CommandResult(
                success=False,
                state=self.state,
                message=f"mission dependencies are unavailable: {exc}",
                task_id=f"mission:{mission_id}",
            )
        return CommandResult(
            success=True,
            state=self.state,
            message="mission graph validated; no hardware command was sent",
            task_id=f"mission:{mission_id}",
            data={
                "dry_run": True,
                "mission_id": mission_id,
                "mission_fingerprint": mission.fingerprint(),
                "actions": [
                    {
                        "node_id": action.node_id,
                        "action": action.action,
                        "target": action.target,
                    }
                    for action in mission.actions
                ],
                "binding_fingerprint": self._binding_fingerprint(binding),
            },
        )

    def request_mission_arm(
        self,
        mission_id: str,
        *,
        operator_confirmation: str,
    ) -> CommandResult:
        task_id = f"mission:{mission_id}"
        mission = self._mission_catalog.get(mission_id)
        if mission is None:
            return self._refuse(TASK_NOT_ALLOWLISTED, task_id, "mission is not allowlisted")
        try:
            binding = self._resolve_mission_binding(mission)
        except Exception as exc:
            return self._refuse(
                HARDWARE_NOT_ARMED,
                task_id,
                f"mission dependencies are unavailable: {exc}",
            )
        with self._state_lock:
            if self.state not in {DRY_RUN, OBSERVE_ONLY}:
                return self._refuse(
                    ILLEGAL_STATE_TRANSITION,
                    task_id,
                    f"mission cannot be armed from state {self.state}",
                )
            if self._hardware_active():
                return self._refuse(SESSION_LOCKED, task_id, "another hardware task is active")
            if mission.requires_observation and self._require_fresh_observation:
                if not self._observation_is_fresh():
                    return self._refuse(
                        OBSERVATION_REQUIRED,
                        task_id,
                        "a successful recent observation is required before mission arming",
                    )
            if any(action.action in {"navigate", "dock"} for action in mission.actions):
                health = self._motion_gateway.health()
                if not health.get("available", False):
                    return self._refuse(
                        HARDWARE_NOT_ARMED,
                        task_id,
                        str(health.get("reason", "motion backend is unavailable")),
                    )
            self._last_safety_decisions = {}
            for operation in self._mission_safety_operations(mission):
                safety_refusal = self._safety_refusal(operation)
                if safety_refusal is not None:
                    return self._refuse_refusal(safety_refusal, task_id)
            expected = f"CONFIRM {task_id}"
            if operator_confirmation != expected:
                return self._refuse(
                    HARDWARE_NOT_ARMED,
                    task_id,
                    f"operator confirmation must be exactly {expected!r}",
                )
            refusal = self._transition(AWAITING_CONFIRMATION)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id)
            refusal = self._transition(ARMED_REAL)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id)
            self._revoke_active_tokens()
            token_args = self._bound_token_args({}, binding)
            token_secret, stored_token = self._issue_token(task_id, token_args)
            self.active_task = task_id
            self.active_token_id = stored_token.token_id
            self.active_execution_binding = dict(binding)
            metadata = stored_token.metadata()
            self._audit("accepted", task_id, "operator armed mission", metadata, token_secret)
            return CommandResult(
                success=True,
                state=self.state,
                message="real mission armed",
                task_id=task_id,
                data={
                    "task_id": task_id,
                    "mission_id": mission_id,
                    "mission_fingerprint": mission.fingerprint(),
                    "binding_fingerprint": self._binding_fingerprint(binding),
                    "token": asdict(metadata),
                    "token_secret": token_secret,
                    "ttl_seconds": self._token_ttl_seconds,
                },
            )

    def execute_mission(
        self,
        mission_id: str,
        *,
        token: str | None,
    ) -> CommandResult:
        task_id = f"mission:{mission_id}"
        mission = self._mission_catalog.get(mission_id)
        if mission is None:
            return self._refuse(TASK_NOT_ALLOWLISTED, task_id, "mission is not allowlisted")
        with self._state_lock:
            if not token:
                return self._refuse(MISSING_TOKEN, task_id, "real mission requires a token")
            if self.state != ARMED_REAL:
                return self._refuse(HARDWARE_NOT_ARMED, task_id, "real mission is not armed")
            for operation in self._mission_safety_operations(mission):
                safety_refusal = self._safety_refusal(operation)
                if safety_refusal is not None:
                    return self._refuse_refusal(
                        safety_refusal,
                        task_id,
                        token_value=token,
                    )
            try:
                binding = self._resolve_mission_binding(mission)
            except Exception as exc:
                return self._refuse(
                    HARDWARE_NOT_ARMED,
                    task_id,
                    f"mission dependencies are unavailable: {exc}",
                    token_value=token,
                )
            token_args = self._bound_token_args({}, binding)
            stored_token, refusal = self._consume_valid_token(token, task_id, token_args)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id, token_value=token)
            refusal = self._transition(EXECUTING)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id, token_value=token)
            self.active_task = task_id
            self.active_execution_binding = dict(binding)
            metadata = stored_token.metadata()
            self._audit("accepted", task_id, "starting real mission", metadata, token)
            episode_started_at = self._clock()

        daemon = self

        class Provider:
            def execute(self, action: MissionAction, deadline: float) -> MissionActionResult:
                return daemon._execute_mission_action(action, deadline, binding, token)

        try:
            mission_result = MissionExecutor(
                provider=Provider(),
                cancelled=lambda: self.state != EXECUTING,
            ).execute(mission)
            steps = [
                {
                    "action": f"mission:{entry['node_id']}",
                    **entry,
                }
                for entry in mission_result.trace
            ]
            command = CommandResult(
                success=mission_result.success,
                state=self.state,
                message=(
                    "mission completed"
                    if mission_result.success
                    else mission_result.message
                ),
                task_id=task_id,
                data={
                    "mission_id": mission_id,
                    "mission_fingerprint": mission.fingerprint(),
                    "node_visits": mission_result.node_visits,
                    "steps": steps,
                },
            )
            with self._state_lock:
                if self.state == EXECUTING:
                    self._transition(DRY_RUN)
                elif self.state == STOPPING:
                    self._transition(DISABLED)
                self.active_task = None
                self.active_token_id = None
                self.active_execution_binding = {}
                command.state = self.state
                self._audit(
                    "accepted",
                    task_id,
                    f"mission finished success={command.success}",
                    metadata,
                    token,
                )
            episode_binding = {
                "task_executor": "mission_graph",
                "task_fingerprint": mission.fingerprint(),
                "mission_binding_fingerprint": self._binding_fingerprint(binding),
            }
            self._record_episode(
                command,
                task_id=task_id,
                execution_binding=episode_binding,
                started_at=episode_started_at,
            )
            return command
        except Exception as exc:
            with self._state_lock:
                if self.state not in {ESTOP, PAUSED, STOPPING}:
                    self._transition(ERROR)
                self.active_task = None
                self.active_token_id = None
                self.active_execution_binding = {}
                self._audit("refused", task_id, f"mission failed: {exc}", token_value=token)
            command = CommandResult(
                success=False,
                state=self.state,
                message=f"mission failed: {exc}",
                task_id=task_id,
                data={"error": str(exc)},
            )
            self._record_episode(
                command,
                task_id=task_id,
                execution_binding={
                    "task_executor": "mission_graph",
                    "task_fingerprint": mission.fingerprint(),
                },
                started_at=episode_started_at,
            )
            return command

    def world_state(self) -> CommandResult:
        agent = self._agent_factory(dry_run=True)
        if not hasattr(agent, "get_world_state"):
            return CommandResult(
                success=False,
                state=self.state,
                message="agent does not expose a world model",
            )
        return CommandResult(
            success=True,
            state=self.state,
            message="current world-model snapshot",
            data=agent.get_world_state(),
        )

    def world_events(self, *, since_sequence: int = 0, limit: int = 100) -> CommandResult:
        if (
            isinstance(since_sequence, bool)
            or not isinstance(since_sequence, int)
            or since_sequence < 0
        ):
            raise ValueError("since_sequence must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("world event limit must be in [1, 1000]")
        agent = self._agent_factory(dry_run=True)
        if not hasattr(agent, "get_world_events"):
            return CommandResult(
                success=False,
                state=self.state,
                message="agent does not expose world-model events",
            )
        return CommandResult(
            success=True,
            state=self.state,
            message="replayable world-model events",
            data=agent.get_world_events(since_sequence=since_sequence, limit=limit),
        )

    def dry_run_motion(self, operation: str, target: str) -> CommandResult:
        task_id = self._motion_task_id(operation, target)
        resolved = self._resolve_motion_target(operation, target)
        if resolved is None:
            return self._refuse(
                TASK_NOT_ALLOWLISTED,
                task_id,
                "motion target is not present in the named catalog",
            )
        return CommandResult(
            success=True,
            state=self.state,
            message="named motion validated; no hardware command was sent",
            task_id=task_id,
            data={"dry_run": True, "operation": operation, "target": target},
        )

    def plan(self, goal: str) -> CommandResult:
        mission = self._mission_catalog.get(goal) or self._mission_catalog.match(goal)
        if mission is not None:
            return CommandResult(
                success=True,
                state=self.state,
                message="registered mission plan",
                data={
                    "goal": goal,
                    "mission_id": mission.mission_id,
                    "mission_fingerprint": mission.fingerprint(),
                    "plans": [
                        {
                            "id": action.node_id,
                            "skill": action.action,
                            "goal": action.target,
                        }
                        for action in mission.actions
                    ],
                    "dry_run": True,
                },
            )
        agent = self._agent_factory(dry_run=True)
        if self._should_use_visual_plan(goal, agent):
            return self.inspect_visual_plan(goal, agent=agent)
        plans = agent.plan_goal(goal) if hasattr(agent, "plan_goal") else []
        normalized = [self._plan_to_dict(plan) for plan in plans]
        return CommandResult(
            success=True,
            state=self.state,
            message="dry-run plan generated",
            data={"goal": goal, "plans": normalized, "dry_run": True},
        )

    def inspect_visual_plan(
        self,
        goal: str,
        *,
        agent: Any | None = None,
    ) -> CommandResult:
        """Capture head_cam and produce a catalog-constrained, non-actuating plan."""
        try:
            agent = agent or self._agent_factory(dry_run=True)
            catalog = self._task_catalog or getattr(agent, "task_catalog", None)
            vlm_client = getattr(agent, "vlm", None)
            if catalog is None or vlm_client is None:
                raise VisualPlanError("visual task planner dependencies are unavailable")
            plan = VisualTaskPlanner(catalog, vlm_client).create(
                goal,
                world_context=(
                    agent.world_model.summary()
                    if hasattr(agent, "world_model")
                    else None
                ),
            )
            self._visual_plans[plan.plan_id] = plan
            while len(self._visual_plans) > 32:
                self._visual_plans.pop(next(iter(self._visual_plans)))
            return CommandResult(
                success=True,
                state=self.state,
                message=(
                    "head-camera visual plan generated"
                    if plan.steps
                    else "head-camera could not prove a registered task"
                ),
                task_id=f"plan:{plan.plan_id}",
                data={"visual_plan": plan.to_dict(), "requires_confirmation": bool(plan.steps)},
            )
        except (VisualPlanError, OSError, ValueError) as exc:
            return CommandResult(
                success=False,
                state=self.state,
                message=f"visual plan failed: {exc}",
                task_id="",
                data={"error_type": type(exc).__name__},
            )

    def _should_use_visual_plan(self, goal: str, agent: Any) -> bool:
        """Route only abstract, non-mission goals through head-camera planning."""
        if self._task_catalog is None:
            return False
        normalized_goal = goal.casefold()
        matched_task = None
        matcher = getattr(agent, "match_task", None)
        if callable(matcher):
            try:
                matched_task = matcher(goal)
            except (TypeError, ValueError):
                matched_task = None
        if matched_task is None:
            for task in self._task_catalog.tasks.values():
                phrases = (task.task_id, *task.aliases)
                if any(phrase.casefold() in normalized_goal for phrase in phrases):
                    matched_task = task.task_id
                    break
        if matched_task is None:
            return True
        return self._contains_composite_marker(goal)

    @staticmethod
    def _contains_composite_marker(goal: str) -> bool:
        return any(
            marker in goal
            for marker in ("并", "然后", "再", "同时", "以及", "和", "，", ",", "；", ";")
        )

    def request_visual_plan_arm(
        self,
        plan_id: str,
        *,
        operator_confirmation: str,
    ) -> CommandResult:
        task_id = f"plan:{plan_id}"
        plan = self._visual_plans.get(plan_id)
        if plan is None:
            return self._refuse(TASK_NOT_ALLOWLISTED, task_id, "visual plan is not available")
        try:
            binding = self._visual_plan_binding(plan)
        except (VisualPlanError, OSError, ValueError) as exc:
            return self._refuse(
                HARDWARE_NOT_ARMED,
                task_id,
                f"visual plan is no longer valid: {exc}",
            )
        with self._state_lock:
            if not plan.steps:
                return self._refuse(
                    TASK_NOT_ALLOWLISTED,
                    task_id,
                    "visual plan contains no executable task",
                )
            if self.state not in {DRY_RUN, OBSERVE_ONLY}:
                return self._refuse(
                    ILLEGAL_STATE_TRANSITION,
                    task_id,
                    f"visual plan cannot be armed from state {self.state}",
                )
            if self._hardware_active():
                return self._refuse(SESSION_LOCKED, task_id, "another hardware task is active")
            if self._require_fresh_observation and any(
                self._task_metadata_resolver(step.task_id).requires_observation
                for step in plan.steps
            ) and not self._observation_is_fresh():
                return self._refuse(
                    OBSERVATION_REQUIRED,
                    task_id,
                    "a successful recent full RGB observation is required before plan arming",
                )
            self._last_safety_decisions = {}
            safety_refusal = self._safety_refusal("manipulation")
            if safety_refusal is not None:
                return self._refuse_refusal(safety_refusal, task_id)
            expected_confirmation = f"CONFIRM {task_id}"
            if operator_confirmation != expected_confirmation:
                return self._refuse(
                    HARDWARE_NOT_ARMED,
                    task_id,
                    f"operator confirmation must be exactly {expected_confirmation!r}",
                )
            refusal = self._transition(AWAITING_CONFIRMATION)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id)
            refusal = self._transition(ARMED_REAL)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id)
            self._revoke_active_tokens()
            token_secret, stored_token = self._issue_token(
                task_id,
                self._bound_token_args({}, binding),
            )
            self.active_task = task_id
            self.active_token_id = stored_token.token_id
            self.active_execution_binding = dict(binding)
            metadata = stored_token.metadata()
            self._audit(
                "accepted",
                task_id,
                "operator armed visual task plan",
                metadata,
                token_secret,
            )
            return CommandResult(
                success=True,
                state=self.state,
                message="visual task plan armed",
                task_id=task_id,
                data={
                    "task_id": task_id,
                    "plan_id": plan.plan_id,
                    "plan_sha256": plan.plan_sha256,
                    "task_ids": list(plan.task_ids),
                    "token": asdict(metadata),
                    "token_secret": token_secret,
                    "ttl_seconds": self._token_ttl_seconds,
                },
            )

    def execute_visual_plan(self, plan_id: str, *, token: str | None) -> CommandResult:
        task_id = f"plan:{plan_id}"
        plan = self._visual_plans.get(plan_id)
        if plan is None:
            return self._refuse(TASK_NOT_ALLOWLISTED, task_id, "visual plan is not available")
        try:
            binding = self._visual_plan_binding(plan)
        except (VisualPlanError, OSError, ValueError) as exc:
            return self._refuse(
                HARDWARE_NOT_ARMED,
                task_id,
                f"visual plan is no longer valid: {exc}",
            )
        with self._state_lock:
            if not token:
                return self._refuse(MISSING_TOKEN, task_id, "real visual plan requires a token")
            if self.state != ARMED_REAL:
                return self._refuse(HARDWARE_NOT_ARMED, task_id, "visual plan is not armed")
            stored_token, refusal = self._consume_valid_token(
                token,
                task_id,
                self._bound_token_args({}, binding),
            )
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id, token_value=token)
            refusal = self._transition(EXECUTING)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id, token_value=token)
            self.active_task = task_id
            self.active_execution_binding = dict(binding)
            metadata = stored_token.metadata()
            self._audit("accepted", task_id, "starting visual task plan", metadata, token)
            episode_started_at = self._clock()

        steps: list[dict[str, Any]] = []
        success = True
        message = "visual task plan completed"
        try:
            for step in plan.steps:
                with self._state_lock:
                    if self.state != EXECUTING:
                        success = False
                        message = "visual task plan was canceled"
                        break
                    current_binding = self._visual_plan_binding(plan)
                    if current_binding != binding:
                        success = False
                        message = "visual plan binding changed before task execution"
                        break
                    safety_refusal = self._safety_refusal("manipulation")
                    if safety_refusal is not None:
                        success = False
                        message = safety_refusal.message
                        break
                    task_metadata = self._task_metadata_resolver(step.task_id)
                    child_binding = dict(binding["task_bindings"][step.task_id])
                    child_started_at = self._clock()
                if self._real_executor is not None:
                    worker_result = self._real_executor.execute(
                        step.task_id,
                        child_binding,
                        token,
                        task_metadata.max_runtime_seconds,
                    )
                else:
                    try:
                        from .hardware.authorization import (
                            _make_robotd_hardware_authorization,
                            _robotd_authorized_context,
                        )
                    except ImportError:
                        from robot_agent.hardware.authorization import (
                            _make_robotd_hardware_authorization,
                            _robotd_authorized_context,
                        )

                    authorization = _make_robotd_hardware_authorization(
                        token,
                        step.task_id,
                        child_binding,
                    )
                    with _robotd_authorized_context(authorization):
                        worker_result = self._agent_factory(dry_run=False).execute(step.task_id)
                command = CommandResult(
                    success=bool(worker_result.success),
                    state=EXECUTING,
                    message=str(worker_result.message),
                    task_id=step.task_id,
                    data=dict(worker_result.data or {}),
                )
                self._record_episode(
                    command,
                    task_id=step.task_id,
                    execution_binding=child_binding,
                    started_at=child_started_at,
                )
                steps.append(
                    {
                        "task_id": step.task_id,
                        "success": command.success,
                        "message": command.message,
                        "data": dict(command.data),
                    }
                )
                if not command.success:
                    success = False
                    message = f"visual task {step.task_id} failed"
                    break
        except Exception as exc:
            success = False
            message = f"visual task plan failed: {type(exc).__name__}: {exc}"
        finally:
            command = CommandResult(
                success=success,
                state=self.state,
                message=message,
                task_id=task_id,
                data={
                    "plan_id": plan_id,
                    "plan_sha256": plan.plan_sha256,
                    "task_ids": list(plan.task_ids),
                    "steps": steps,
                },
            )
            with self._state_lock:
                if self.state == EXECUTING:
                    self._transition(DRY_RUN)
                elif self.state == STOPPING:
                    self._transition(DISABLED)
                self.active_task = None
                self.active_token_id = None
                self.active_execution_binding = {}
                command.state = self.state
                self._audit(
                    "accepted" if success else "refused",
                    task_id,
                    f"visual task plan finished success={success}",
                    token_value=token,
                )
            self._record_episode(
                command,
                task_id=task_id,
                execution_binding={
                    "task_executor": "visual_task_plan",
                    "plan_id": plan_id,
                    "plan_sha256": plan.plan_sha256,
                    "task_fingerprints": dict(plan.task_fingerprints),
                },
                started_at=episode_started_at,
            )
        return command

    def _visual_plan_binding(self, plan: VisualTaskPlan) -> dict[str, Any]:
        plan.validate()
        if not plan.steps:
            return {
                "task_executor": "visual_task_plan",
                "plan_id": plan.plan_id,
                "plan_sha256": plan.plan_sha256,
                "task_ids": [],
                "task_bindings": {},
            }
        task_bindings: dict[str, dict[str, Any]] = {}
        for task_id, task_fingerprint in plan.task_fingerprints:
            metadata = self._task_metadata_resolver(task_id)
            if metadata is None:
                raise VisualPlanError(f"planned task is no longer registered: {task_id}")
            if not self._args_are_safe(metadata.args, {}):
                raise VisualPlanError(f"planned task requires explicit arguments: {task_id}")
            binding = dict(self._task_binding_resolver(task_id))
            if binding.get("task_fingerprint") != task_fingerprint:
                raise VisualPlanError(f"task catalog changed after visual planning: {task_id}")
            task_bindings[task_id] = binding
        return {
            "task_executor": "visual_task_plan",
            "plan_id": plan.plan_id,
            "plan_sha256": plan.plan_sha256,
            "task_ids": list(plan.task_ids),
            "task_bindings": task_bindings,
        }

    def dry_run(self, goal_or_task: str) -> CommandResult:
        with self._state_lock:
            if self._real_executor is not None and getattr(self._real_executor, "active", False):
                return self._refuse(
                    SESSION_LOCKED,
                    goal_or_task,
                    "dry-run is unavailable while hardware execution is active",
                )
            self._revoke_active_tokens()
            self._set_state_if_legal(DRY_RUN)
        mission = self._mission_catalog.get(goal_or_task) or self._mission_catalog.match(
            goal_or_task
        )
        if mission is not None:
            return self.dry_run_mission(mission.mission_id)
        agent = (
            self._dry_run_agent_factory()
            if self._dry_run_agent_factory is not None
            else self._agent_factory(dry_run=True)
        )
        matched_task_id = self._dry_run_task_id(goal_or_task, agent)
        if matched_task_id is not None:
            result = self._execute_dry_run_task(agent, matched_task_id)
        elif self._should_use_visual_plan(goal_or_task, agent):
            inspected = self.inspect_visual_plan(goal_or_task, agent=agent)
            if not inspected.success:
                return inspected
            plan_id = inspected.task_id.removeprefix("plan:")
            return self._execute_visual_plan_dry_run(plan_id, agent)
        elif hasattr(agent, "run_complex"):
            result = agent.run_complex(goal_or_task)
        else:
            result = agent.execute(goal_or_task)
        return self._agent_result_to_command(result, task_id=goal_or_task, dry_run=True)

    def _dry_run_task_id(self, goal: str, agent: Any) -> str | None:
        """Resolve exact IDs and non-composite aliases before complex-goal routing."""
        if self._task_metadata_resolver(goal) is not None:
            return goal
        if self._contains_composite_marker(goal):
            return None
        matcher = getattr(agent, "match_task", None)
        if not callable(matcher):
            return None
        try:
            task_id = matcher(goal)
        except (TypeError, ValueError):
            return None
        if not isinstance(task_id, str) or self._task_metadata_resolver(task_id) is None:
            return None
        return task_id

    @staticmethod
    def _execute_dry_run_task(agent: Any, task_id: str) -> Any:
        pipeline = getattr(agent, "execute_dry_run_pipeline", None)
        if callable(pipeline):
            return pipeline(task_id)
        return agent.execute(task_id)

    def _execute_visual_plan_dry_run(self, plan_id: str, agent: Any) -> CommandResult:
        plan = self._visual_plans.get(plan_id)
        task_id = f"plan:{plan_id}"
        if plan is None:
            return self._refuse(TASK_NOT_ALLOWLISTED, task_id, "visual plan is not available")
        try:
            plan.validate()
        except (VisualPlanError, ValueError) as exc:
            return self._refuse(HARDWARE_NOT_ARMED, task_id, f"visual plan is invalid: {exc}")
        if not plan.steps:
            return CommandResult(
                success=False,
                state=self.state,
                message="head-camera could not prove a registered task",
                task_id=task_id,
                data={"dry_run": True, "visual_plan": plan.to_dict(), "steps": []},
            )

        steps: list[dict[str, Any]] = []
        success = True
        for step in plan.steps:
            metadata = self._task_metadata_resolver(step.task_id)
            if metadata is None:
                success = False
                steps.append(
                    {
                        "task_id": step.task_id,
                        "success": False,
                        "message": "planned task is no longer registered",
                    }
                )
                break
            result = self._execute_dry_run_task(agent, step.task_id)
            command = self._agent_result_to_command(
                result,
                task_id=step.task_id,
                dry_run=True,
            )
            steps.append(
                {
                    "task_id": step.task_id,
                    "success": command.success,
                    "message": command.message,
                    "data": dict(command.data),
                }
            )
            if not command.success:
                success = False
                break
        return CommandResult(
            success=success,
            state=self.state,
            message=(
                "dry-run visual plan completed through the real pipeline"
                if success
                else "dry-run visual plan stopped at a failed subtask"
            ),
            task_id=task_id,
            data={
                "dry_run": True,
                "execution_mode": "pipeline_dummy",
                "visual_plan": plan.to_dict(),
                "task_ids": list(plan.task_ids),
                "steps": steps,
            },
        )

    def observe(self, instruction: str = "描述当前环境") -> CommandResult:
        with self._state_lock:
            self._set_state_if_legal(OBSERVE_ONLY)
        agent = self._agent_factory(dry_run=False)
        result = agent.observe(instruction)
        if bool(getattr(result, "success", False)):
            with self._state_lock:
                self.last_observation_at = self._clock()
        command = self._agent_result_to_command(result, dry_run=False)
        command.data["observe_only"] = True
        return command

    def request_arm(
        self,
        task_id: str,
        args: dict[str, Any] | None = None,
        operator_confirmation: str = "",
    ) -> CommandResult:
        args = args or {}
        execution_binding: dict[str, Any] = {}
        metadata_before_lock = self._task_metadata_resolver(task_id)
        if metadata_before_lock is not None and self._args_are_safe(
            metadata_before_lock.args,
            args,
        ):
            try:
                execution_binding = dict(self._task_binding_resolver(task_id))
            except Exception as exc:
                with self._state_lock:
                    return self._refuse(
                        HARDWARE_NOT_ARMED,
                        task_id,
                        f"task execution policy is unavailable: {exc}",
                    )
        token_args = self._bound_token_args(args, execution_binding)
        with self._state_lock:
            metadata = self._task_metadata_resolver(task_id)
            if metadata is None:
                return self._refuse(TASK_NOT_ALLOWLISTED, task_id, "task is not allowlisted")
            if not self._args_are_safe(metadata.args, args):
                return self._refuse(UNSAFE_ARGUMENTS, task_id, "task arguments are not allowlisted")
            if self.state not in metadata.allowed_modes:
                return self._refuse(
                    ILLEGAL_STATE_TRANSITION,
                    task_id,
                    f"task cannot be armed from state {self.state}",
                )
            if self._real_executor is not None and getattr(self._real_executor, "active", False):
                return self._refuse(SESSION_LOCKED, task_id, "another hardware task is active")
            if getattr(self._motion_gateway, "active", False):
                return self._refuse(SESSION_LOCKED, task_id, "a motion task is active")
            if self._require_fresh_observation and metadata.requires_observation:
                observation_age = (
                    None
                    if self.last_observation_at is None
                    else self._clock() - self.last_observation_at
                )
                if observation_age is None or observation_age > self._observation_ttl_seconds:
                    return self._refuse(
                        OBSERVATION_REQUIRED,
                        task_id,
                        "a successful recent observe-only check is required before arming",
                    )

            self._last_safety_decisions = {}
            safety_refusal = self._safety_refusal("manipulation")
            if safety_refusal is not None:
                return self._refuse_refusal(safety_refusal, task_id)

            expected_confirmation = f"CONFIRM {task_id}"
            if operator_confirmation != expected_confirmation:
                return self._refuse(
                    HARDWARE_NOT_ARMED,
                    task_id,
                    f"operator confirmation must be exactly {expected_confirmation!r}",
                )

            refusal = self._transition(AWAITING_CONFIRMATION)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id)
            refusal = self._transition(ARMED_REAL)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id)

            self._revoke_active_tokens()
            token_secret, stored_token = self._issue_token(task_id, token_args)
            token_metadata = stored_token.metadata()
            self.active_task = task_id
            self.active_token_id = stored_token.token_id
            self.active_execution_binding = dict(execution_binding)
            self._audit("accepted", task_id, "operator armed real execution", token_metadata, token_secret)
            return CommandResult(
                success=True,
                state=self.state,
                message="real execution armed for allowlisted task",
                task_id=task_id,
                data={
                    "task_id": task_id,
                    "session_id": self.session_id,
                    "token": {
                        "token_id": token_metadata.token_id,
                        "issued_to": token_metadata.issued_to,
                        "issued_at": token_metadata.issued_at,
                        "expires_at": token_metadata.expires_at,
                        "scopes": list(token_metadata.scopes),
                        "session_id": token_metadata.session_id,
                        "consumed": False,
                    },
                    "token_secret": token_secret,
                    "ttl_seconds": self._token_ttl_seconds,
                    "execution_binding": execution_binding,
                },
            )

    def request_motion_arm(
        self,
        operation: str,
        target: str,
        operator_confirmation: str = "",
    ) -> CommandResult:
        task_id = self._motion_task_id(operation, target)
        args = {"operation": operation, "target": target}
        with self._state_lock:
            resolved_target = self._resolve_motion_target(operation, target)
            if resolved_target is None:
                return self._refuse(
                    TASK_NOT_ALLOWLISTED,
                    task_id,
                    "motion target is not present in the named catalog",
                )
            health = self._motion_gateway.health()
            if not health.get("available", False):
                return self._refuse(
                    HARDWARE_NOT_ARMED,
                    task_id,
                    str(health.get("reason", "motion backend is unavailable")),
                )
            if self.state not in {DRY_RUN, OBSERVE_ONLY}:
                return self._refuse(
                    ILLEGAL_STATE_TRANSITION,
                    task_id,
                    f"motion cannot be armed from state {self.state}",
                )
            if self._hardware_active():
                return self._refuse(SESSION_LOCKED, task_id, "another hardware task is active")
            if self._require_fresh_observation and not self._observation_is_fresh():
                return self._refuse(
                    OBSERVATION_REQUIRED,
                    task_id,
                    "a successful recent observe-only check is required before arming",
                )
            self._last_safety_decisions = {}
            safety_refusal = self._safety_refusal("motion")
            if safety_refusal is not None:
                return self._refuse_refusal(safety_refusal, task_id)
            expected_confirmation = f"CONFIRM {task_id}"
            if operator_confirmation != expected_confirmation:
                return self._refuse(
                    HARDWARE_NOT_ARMED,
                    task_id,
                    f"operator confirmation must be exactly {expected_confirmation!r}",
                )
            refusal = self._transition(AWAITING_CONFIRMATION)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id)
            refusal = self._transition(ARMED_REAL)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id)
            self._revoke_active_tokens()
            execution_binding = self._motion_execution_binding(
                operation,
                target,
                resolved_target,
            )
            token_secret, stored_token = self._issue_token(
                task_id,
                self._bound_token_args(args, execution_binding),
            )
            self.active_task = task_id
            self.active_token_id = stored_token.token_id
            self.active_execution_binding = dict(execution_binding)
            metadata = stored_token.metadata()
            self._audit("accepted", task_id, "operator armed named motion", metadata, token_secret)
            return CommandResult(
                success=True,
                state=self.state,
                message="real motion armed for named target",
                task_id=task_id,
                data={
                    "task_id": task_id,
                    "session_id": self.session_id,
                    "token": asdict(metadata),
                    "token_secret": token_secret,
                    "ttl_seconds": self._token_ttl_seconds,
                    "execution_binding": execution_binding,
                },
            )

    def execute_motion(
        self,
        operation: str,
        target: str,
        token: str | None = None,
    ) -> CommandResult:
        task_id = self._motion_task_id(operation, target)
        args = {"operation": operation, "target": target}
        with self._state_lock:
            resolved_target = self._resolve_motion_target(operation, target)
            if resolved_target is None:
                return self._refuse(
                    TASK_NOT_ALLOWLISTED,
                    task_id,
                    "motion target is not present in the named catalog",
                )
            if not token:
                return self._refuse(
                    MISSING_TOKEN,
                    task_id,
                    "real motion requires an operator token",
                )
            if self.state != ARMED_REAL:
                return self._refuse(HARDWARE_NOT_ARMED, task_id, "real motion is not armed")
            if self._require_fresh_observation and not self._observation_is_fresh():
                return self._refuse(
                    OBSERVATION_REQUIRED,
                    task_id,
                    "the observe-only safety check expired before execution",
                    token_value=token,
                )
            safety_refusal = self._safety_refusal("motion")
            if safety_refusal is not None:
                return self._refuse_refusal(safety_refusal, task_id, token_value=token)
            motion_binding = self._motion_execution_binding(
                operation,
                target,
                resolved_target,
            )
            stored_token, refusal = self._consume_valid_token(
                token,
                task_id,
                self._bound_token_args(args, motion_binding),
            )
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id, token_value=token)
            refusal = self._transition(EXECUTING)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id, token_value=token)
            self.active_task = task_id
            self.active_execution_binding = dict(motion_binding)
            token_metadata = stored_token.metadata()
            self._audit("accepted", task_id, "starting named motion", token_metadata, token)
            episode_started_at = self._clock()

        try:
            if operation == "navigate":
                result = self._motion_gateway.navigate(
                    resolved_target,
                    self._navigation_timeout_seconds,
                )
            else:
                result = self._motion_gateway.dock(
                    resolved_target,
                    self._docking_timeout_seconds,
                )
            command = self._motion_result_to_command(result, task_id)
            with self._state_lock:
                if self.state == EXECUTING:
                    self._transition(DRY_RUN)
                elif self.state == STOPPING:
                    self._transition(DISABLED)
                self.active_task = None
                self.active_token_id = None
                command.state = self.state
                self._audit(
                    "accepted",
                    task_id,
                    f"named motion finished success={command.success}",
                    token_metadata,
                    token,
                )
                self.active_execution_binding = {}
            self._record_episode(
                command,
                task_id=task_id,
                execution_binding=motion_binding,
                started_at=episode_started_at,
            )
            return command
        except Exception as exc:
            with self._state_lock:
                if self.state not in {ESTOP, PAUSED, STOPPING}:
                    self._transition(ERROR)
                self.active_task = None
                self.active_token_id = None
                self._audit("refused", task_id, f"named motion failed: {exc}", token_value=token)
            command = CommandResult(
                success=False,
                state=self.state,
                message=f"named motion failed: {exc}",
                task_id=task_id,
                data={"error": str(exc)},
            )
            self._record_episode(
                command,
                task_id=task_id,
                execution_binding=motion_binding,
                started_at=episode_started_at,
            )
            return command

    def execute(
        self,
        task_id: str,
        args: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> CommandResult:
        args = args or {}
        execution_binding: dict[str, Any] = {}
        with self._state_lock:
            metadata = self._task_metadata_resolver(task_id)
            if metadata is None:
                return self._refuse(TASK_NOT_ALLOWLISTED, task_id, "task is not allowlisted")
            if not self._args_are_safe(metadata.args, args):
                return self._refuse(UNSAFE_ARGUMENTS, task_id, "task arguments are not allowlisted")
            if not token:
                return self._refuse(MISSING_TOKEN, task_id, "real execution requires an operator token")
            if self.state != ARMED_REAL:
                return self._refuse(
                    HARDWARE_NOT_ARMED,
                    task_id,
                    "real execution is not armed",
                    token_value=token,
                )
            if self._require_fresh_observation and metadata.requires_observation:
                observation_age = (
                    None
                    if self.last_observation_at is None
                    else self._clock() - self.last_observation_at
                )
                if observation_age is None or observation_age > self._observation_ttl_seconds:
                    return self._refuse(
                        OBSERVATION_REQUIRED,
                        task_id,
                        "the observe-only safety check expired before execution",
                        token_value=token,
                    )

            safety_refusal = self._safety_refusal("manipulation")
            if safety_refusal is not None:
                return self._refuse_refusal(safety_refusal, task_id, token_value=token)

            try:
                execution_binding = dict(self._task_binding_resolver(task_id))
            except Exception as exc:
                return self._refuse(
                    HARDWARE_NOT_ARMED,
                    task_id,
                    f"task execution policy is unavailable: {exc}",
                    token_value=token,
                )
            token_args = self._bound_token_args(args, execution_binding)

            stored_token, refusal = self._consume_valid_token(token, task_id, token_args)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id, token_value=token)

            refusal = self._transition(EXECUTING)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id, token_value=token)

            self.active_task = task_id
            self.active_execution_binding = dict(execution_binding)
            token_metadata = stored_token.metadata()
            self._audit("accepted", task_id, "starting real execution", token_metadata, token)
            episode_started_at = self._clock()

        try:
            if self._real_executor is not None:
                result = self._real_executor.execute(
                    task_id,
                    execution_binding,
                    token,
                    metadata.max_runtime_seconds,
                )
            else:
                try:
                    from .hardware.authorization import _make_robotd_hardware_authorization, _robotd_authorized_context
                except ImportError:
                    from robot_agent.hardware.authorization import _make_robotd_hardware_authorization, _robotd_authorized_context

                authorization = _make_robotd_hardware_authorization(
                    token,
                    task_id,
                    execution_binding,
                )
                with _robotd_authorized_context(authorization):
                    agent = self._agent_factory(dry_run=False)
                    result = agent.execute(task_id, args)
            command = self._agent_result_to_command(result, task_id=task_id, dry_run=False)
            with self._state_lock:
                if self.state == EXECUTING:
                    self._transition(DRY_RUN)
                elif self.state == STOPPING:
                    self._transition(DISABLED)
                self.active_task = None
                self.active_token_id = None
                command.state = self.state
                self._audit(
                    "accepted",
                    task_id,
                    f"real execution finished success={command.success}",
                    token_metadata,
                    token,
                )
                self.active_execution_binding = {}
            self._record_episode(
                command,
                task_id=task_id,
                execution_binding=execution_binding,
                started_at=episode_started_at,
            )
            return command
        except Exception as exc:
            with self._state_lock:
                if self.state not in {ESTOP, PAUSED, STOPPING}:
                    self._transition(ERROR)
                self.active_task = None
                self.active_token_id = None
                self._audit("refused", task_id, f"real execution failed: {exc}", token_value=token)
                self.active_execution_binding = {}
            command = CommandResult(
                success=False,
                state=self.state,
                message=f"real execution failed: {exc}",
                task_id=task_id,
                data={"error": str(exc)},
            )
            self._record_episode(
                command,
                task_id=task_id,
                execution_binding=execution_binding,
                started_at=episode_started_at,
            )
            return command

    def _record_episode(
        self,
        command: CommandResult,
        *,
        task_id: str,
        execution_binding: dict[str, Any],
        started_at: float,
    ) -> None:
        if self._last_safety_decisions:
            command.data["safety_decisions"] = {
                key: dict(value) for key, value in self._last_safety_decisions.items()
            }
        if self._episode_recorder is None:
            return
        result = {
            "message": command.message,
            "state": command.state,
            **dict(command.data),
        }
        intervention = command.state in {DISABLED, ESTOP, PAUSED, STOPPING}
        try:
            episode_binding = dict(execution_binding)
            episode_binding["experiment_mode"] = self._experiment_mode
            manifest = self._episode_recorder.record(
                task_id=task_id,
                execution_binding=episode_binding,
                started_at=started_at,
                ended_at=self._clock(),
                success=command.success,
                terminal_state=command.state,
                result=result,
                human_intervention=intervention,
            )
            command.data["episode_id"] = manifest.episode_id
            command.data["episode_manifest_sha256"] = manifest.manifest_sha256
        except Exception as exc:
            command.data["episode_record_error"] = f"{type(exc).__name__}: {exc}"

    def _resolve_mission_binding(self, mission: MissionDefinition) -> dict[str, Any]:
        nodes: dict[str, Any] = {}
        for action in mission.actions:
            if action.action == "task":
                metadata = self._task_metadata_resolver(action.target)
                if metadata is None:
                    raise ValueError(f"mission task is not allowlisted: {action.target}")
                task_binding = dict(self._task_binding_resolver(action.target))
                nodes[action.node_id] = {
                    "action": action.action,
                    "target": action.target,
                    "timeout_seconds": action.timeout_seconds,
                    "task_max_runtime_seconds": metadata.max_runtime_seconds,
                    "execution_binding": task_binding,
                }
                continue
            if action.action == "dock_table":
                profile = self._table_catalog.get(action.target)
                if profile is None:
                    raise ValueError(
                        f"mission table profile is not allowlisted: {action.target}"
                    )
                profile_view = profile.public_view()
                nodes[action.node_id] = {
                    "action": action.action,
                    "target": action.target,
                    "timeout_seconds": action.timeout_seconds,
                    "table_catalog_version": self._table_catalog.version,
                    "table_profile": profile_view,
                    "target_fingerprint": self._binding_fingerprint(
                        {
                            "catalog_version": self._table_catalog.version,
                            "profile": profile_view,
                        }
                    ),
                }
                continue
            if action.action == "observe":
                nodes[action.node_id] = {
                    "action": action.action,
                    "target": action.target,
                    "timeout_seconds": action.timeout_seconds,
                }
                continue
            resolved = self._resolve_motion_target(action.action, action.target)
            if resolved is None:
                raise ValueError(
                    f"mission motion target is not allowlisted: {action.action}:{action.target}"
                )
            target_payload = {
                "catalog_version": self._motion_catalog.version,
                "action": action.action,
                "target": asdict(resolved),
            }
            nodes[action.node_id] = {
                "action": action.action,
                "target": action.target,
                "timeout_seconds": action.timeout_seconds,
                "target_fingerprint": self._binding_fingerprint(target_payload),
            }
        return {
            "mission_protocol": 1,
            "mission_id": mission.mission_id,
            "mission_fingerprint": mission.fingerprint(),
            "deployment_fingerprint": self._deployment_fingerprint or None,
            "nodes": nodes,
        }

    def _motion_execution_binding(
        self,
        operation: str,
        target: str,
        resolved_target: Any,
    ) -> dict[str, Any]:
        return {
            "task_executor": "named_motion",
            "operation": operation,
            "target": target,
            "motion_catalog_version": self._motion_catalog.version,
            "resolved_target": asdict(resolved_target),
            "deployment_fingerprint": self._deployment_fingerprint or None,
        }

    def _execute_mission_action(
        self,
        action: MissionAction,
        deadline: float,
        mission_binding: dict[str, Any],
        token: str,
    ) -> MissionActionResult:
        if self.state != EXECUTING:
            return MissionActionResult(False, "mission was canceled")
        if action.action == "task":
            safety_operation = "manipulation"
        elif action.action in {"navigate", "dock", "dock_table"}:
            safety_operation = "motion"
        else:
            safety_operation = ""
        safety_refusal = (
            self._safety_refusal(safety_operation) if safety_operation else None
        )
        if safety_refusal is not None:
            return MissionActionResult(
                False,
                safety_refusal.message,
                {"safety": dict(safety_refusal.details)},
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return MissionActionResult(False, "mission action deadline expired")
        node_binding = mission_binding["nodes"].get(action.node_id)
        if not isinstance(node_binding, dict):
            return MissionActionResult(False, "mission action binding is missing")
        started_at = self._clock()
        if action.action == "task":
            task_binding = dict(node_binding["execution_binding"])
            timeout = min(
                remaining,
                float(node_binding["task_max_runtime_seconds"]),
            )
            if self._real_executor is not None:
                result = self._real_executor.execute(
                    action.target,
                    task_binding,
                    token,
                    max(1, math.ceil(timeout)),
                )
            else:
                try:
                    from .hardware.authorization import (
                        _make_robotd_hardware_authorization,
                        _robotd_authorized_context,
                    )
                except ImportError:
                    from robot_agent.hardware.authorization import (
                        _make_robotd_hardware_authorization,
                        _robotd_authorized_context,
                    )

                authorization = _make_robotd_hardware_authorization(
                    token,
                    action.target,
                    task_binding,
                )
                with _robotd_authorized_context(authorization):
                    result = self._agent_factory(dry_run=False).execute(action.target)
            command = self._agent_result_to_command(
                result,
                task_id=action.target,
                dry_run=False,
            )
            command.state = EXECUTING
            self._record_episode(
                command,
                task_id=action.target,
                execution_binding=task_binding,
                started_at=started_at,
            )
            return MissionActionResult(command.success, command.message, dict(command.data))

        if action.action == "observe":
            agent = self._agent_factory(dry_run=False)
            observed = agent.observe(
                f"Recovery observation '{action.target}': confirm the workspace is "
                "safe and clear, the robot is not touching an unexpected object, "
                "and report the current scene. Do not propose actuation."
            )
            command = CommandResult(
                success=bool(observed.success),
                state=EXECUTING,
                message=str(observed.message),
                task_id=f"observe:{action.target}",
                data=dict(observed.data or {}),
            )
            self._record_episode(
                command,
                task_id=f"observe:{action.target}",
                execution_binding={
                    "task_executor": "recovery_observation",
                    "target": action.target,
                },
                started_at=started_at,
            )
            return MissionActionResult(command.success, command.message, dict(command.data))

        if action.action == "dock_table":
            profile = self._table_catalog.get(action.target)
            if profile is None:
                return MissionActionResult(False, "mission table profile disappeared")
            result = self._run_table_docking(profile, remaining)
            command = CommandResult(
                success=bool(result.success),
                state=EXECUTING,
                message=str(result.message),
                task_id=self._motion_task_id("dock_table", action.target),
                data=dict(result.data or {}),
            )
            self._record_episode(
                command,
                task_id=self._motion_task_id("dock_table", action.target),
                execution_binding={
                    "task_executor": "table_docking",
                    "target": action.target,
                    "target_fingerprint": node_binding["target_fingerprint"],
                },
                started_at=started_at,
            )
            return MissionActionResult(command.success, command.message, dict(command.data))

        resolved = self._resolve_motion_target(action.action, action.target)
        if resolved is None:
            return MissionActionResult(False, "mission motion target disappeared")
        if action.action == "navigate":
            result = self._motion_gateway.navigate(resolved, remaining)
        else:
            result = self._motion_gateway.dock(resolved, remaining)
        command = self._motion_result_to_command(
            result,
            self._motion_task_id(action.action, action.target),
        )
        command.state = EXECUTING
        motion_binding = {
            "task_executor": "named_motion",
            "operation": action.action,
            "target": action.target,
            "target_fingerprint": node_binding["target_fingerprint"],
        }
        self._record_episode(
            command,
            task_id=self._motion_task_id(action.action, action.target),
            execution_binding=motion_binding,
            started_at=started_at,
        )
        return MissionActionResult(command.success, command.message, dict(command.data))

    @staticmethod
    def _binding_fingerprint(value: dict[str, Any]) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _safety_refusal(self, operation: str) -> Refusal | None:
        if not self._require_safety_interlock:
            return None
        decision = self._safety_interlock.check(operation, now=self._clock())
        self._last_safety_decisions[operation] = decision.to_dict()
        if decision.allowed:
            return None
        reasons = "; ".join(decision.reasons) or "safety interlock denied actuation"
        return Refusal(
            reason_code=SAFETY_INTERLOCK_BLOCKED,
            message=reasons,
            state=self.state,
            details=decision.to_dict(),
        )

    @staticmethod
    def _mission_safety_operations(mission: MissionDefinition) -> tuple[str, ...]:
        operations: list[str] = []
        if any(
            action.action in {"navigate", "dock", "dock_table"}
            for action in mission.actions
        ):
            operations.append("motion")
        if any(action.action == "task" for action in mission.actions):
            operations.append("manipulation")
        return tuple(operations)

    def pause(self) -> CommandResult:
        with self._state_lock:
            task_id = self.active_task or ""
            refusal = self._transition(PAUSED)
            if refusal is not None:
                return self._refuse_refusal(refusal, task_id)
        self._cancel_executor("operator_pause")
        self._cancel_motion("operator_pause")
        return CommandResult(success=True, state=self.state, message="paused and actuation halted", task_id=task_id)

    def stop(self) -> CommandResult:
        with self._state_lock:
            task_id = self.active_task or ""
            if validate_transition(self.state, STOPPING) is None:
                self.state = STOPPING
            else:
                self.state = DISABLED
            self._revoke_active_tokens()
        self._cancel_executor("operator_stop")
        self._cancel_motion("operator_stop")
        with self._state_lock:
            if self.state == STOPPING:
                self._transition(DISABLED)
            self.active_task = None
        return CommandResult(success=True, state=self.state, message="stopped", task_id=task_id)

    def estop(self) -> CommandResult:
        with self._state_lock:
            self._revoke_active_tokens()
            self.state = ESTOP
            self.active_task = None
        self._cancel_executor("emergency_stop", emergency=True)
        self._cancel_motion("emergency_stop", emergency=True)
        return CommandResult(success=True, state=self.state, message="emergency stop latched")

    def reset(self, operator_confirmation: str = "") -> CommandResult:
        self._cancel_executor("operator_reset")
        self._cancel_motion("operator_reset")
        with self._state_lock:
            if self.state == ESTOP and operator_confirmation != "RESET ESTOP":
                return self._refuse(
                    HARDWARE_NOT_ARMED,
                    "",
                    "resetting a latched estop requires operator confirmation 'RESET ESTOP'",
                )
            self._revoke_active_tokens()
            if self.state == ESTOP and validate_transition(self.state, DISABLED) is None:
                self.state = DISABLED
            if self.state != DRY_RUN:
                if validate_transition(self.state, DRY_RUN) is None:
                    self.state = DRY_RUN
                elif validate_transition(self.state, DISABLED) is None:
                    self.state = DISABLED
                    if validate_transition(self.state, DRY_RUN) is None:
                        self.state = DRY_RUN
            self.active_task = None
        return CommandResult(success=True, state=self.state, message="robot daemon reset")

    def _issue_token(self, task_id: str, args: dict[str, Any]) -> tuple[str, StoredToken]:
        now = self._clock()
        token_id = f"tok-{uuid.uuid4().hex}"
        token_secret = f"robotd-{token_id}-{secrets.token_urlsafe(24)}"
        stored_token = StoredToken(
            token_id=token_id,
            token_hash=token_hash(token_secret),
            task_id=task_id,
            args=dict(args),
            issued_to=self.session_owner,
            issued_at=now,
            expires_at=now + self._token_ttl_seconds,
            scopes=["robot:execute", f"task:{task_id}"],
            session_id=self.session_id,
        )
        self._token_store.save(stored_token)
        return token_secret, stored_token

    def _consume_valid_token(
        self,
        token_secret: str,
        task_id: str,
        args: dict[str, Any],
    ) -> tuple[StoredToken | None, Refusal | None]:
        consume_if_valid = getattr(self._token_store, "consume_if_valid", None)
        if consume_if_valid is not None:
            return consume_if_valid(
                token_secret,
                task_id=task_id,
                args=args,
                session_id=self.session_id,
                active_token_id=self.active_token_id,
                now=self._clock(),
                state=self.state,
            )
        stored_token = self._token_store.find_by_secret(token_secret)
        if stored_token is None:
            return None, Refusal(INVALID_TOKEN, "token is unknown", self.state)
        if stored_token.consumed:
            return stored_token, Refusal(INVALID_TOKEN, "token was already consumed", self.state)
        if stored_token.expires_at <= self._clock():
            return stored_token, Refusal(EXPIRED_TOKEN, "token expired", self.state)
        if stored_token.session_id != self.session_id:
            return stored_token, Refusal(INVALID_TOKEN, "token belongs to a different session", self.state)
        if stored_token.token_id != self.active_token_id:
            return stored_token, Refusal(INVALID_TOKEN, "token is not the currently armed token", self.state)
        if stored_token.task_id != task_id or stored_token.args != args:
            return stored_token, Refusal(INVALID_TOKEN, "token binding does not match command", self.state)
        self._token_store.mark_consumed(stored_token.token_id)
        return stored_token, None

    def _revoke_active_tokens(self) -> None:
        revoke_session = getattr(self._token_store, "revoke_session", None)
        if revoke_session is not None:
            revoke_session(self.session_id)
        self.active_token_id = None
        self.active_task = None
        self.active_execution_binding = {}

    def _cancel_executor(self, reason: str, emergency: bool = False) -> None:
        if self._real_executor is None:
            return
        cancel = getattr(self._real_executor, "cancel", None)
        if cancel is not None:
            cancel(reason=reason, emergency=emergency)

    def _cancel_motion(self, reason: str, emergency: bool = False) -> None:
        self._motion_gateway.cancel(reason=reason, emergency=emergency)

    def close(self) -> None:
        self._motion_gateway.close()

    def _hardware_active(self) -> bool:
        return bool(
            (self._real_executor is not None and getattr(self._real_executor, "active", False))
            or getattr(self._motion_gateway, "active", False)
        )

    def _observation_is_fresh(self) -> bool:
        return bool(
            self.last_observation_at is not None
            and self._clock() - self.last_observation_at <= self._observation_ttl_seconds
        )

    def _run_table_docking(self, profile: Any, remaining_seconds: float) -> Any:
        if self._table_docking_runner is not None:
            return self._table_docking_runner(profile, remaining_seconds)
        config = profile.docking
        if remaining_seconds < config.deadline_seconds:
            config = replace(
                config,
                deadline_seconds=max(1.0, remaining_seconds),
            )
        controller = TableDockingController(
            observer=VlmTableObserver(),
            actuator=ChassisDockActuator(),
            config=config,
            cancelled=lambda: self.state != EXECUTING,
        )
        return controller.run()

    def _resolve_motion_target(self, operation: str, target: str) -> Any | None:
        if operation == "navigate":
            return self._motion_catalog.locations.get(target)
        if operation == "dock":
            return self._motion_catalog.docks.get(target)
        return None

    @staticmethod
    def _motion_task_id(operation: str, target: str) -> str:
        return f"motion:{operation}:{target}"

    def _motion_result_to_command(self, result: MotionResult, task_id: str) -> CommandResult:
        return CommandResult(
            success=result.success,
            state=self.state,
            message=result.message,
            task_id=task_id,
            data=dict(result.data or {}),
        )

    def _transition(self, target_state: str) -> Refusal | None:
        refusal = validate_transition(self.state, target_state)
        if refusal is not None:
            return refusal
        self.state = target_state
        return None

    def _set_state_if_legal(self, target_state: str) -> None:
        if self.state == target_state:
            return
        if validate_transition(self.state, target_state) is None:
            self.state = target_state

    def _refuse(
        self,
        reason_code: str,
        task_id: str,
        message: str,
        *,
        token_value: str = "",
    ) -> CommandResult:
        refusal = Refusal(reason_code=reason_code, message=message, state=self.state)
        return self._refuse_refusal(refusal, task_id, token_value=token_value)

    def _refuse_refusal(
        self,
        refusal: Refusal,
        task_id: str,
        *,
        token_value: str = "",
    ) -> CommandResult:
        self._audit("refused", task_id, refusal.message, refusal=refusal, token_value=token_value)
        return CommandResult(
            success=False,
            state=self.state,
            message=refusal.message,
            task_id=task_id,
            refusal=refusal,
        )

    def _audit(
        self,
        decision: str,
        task_id: str,
        message: str,
        token: TokenMetadata | None = None,
        token_value: str = "",
        refusal: Refusal | None = None,
    ) -> None:
        if self._audit_writer is None:
            return
        event = build_audit_event(
            decision=decision,
            state=self.state,
            command_id=f"cmd-{uuid.uuid4().hex}",
            task_id=task_id,
            actor=self.session_owner,
            message=message,
            details={
                "session_id": self.session_id,
                "active_task": self.active_task,
                "execution_binding": dict(self.active_execution_binding),
            },
            refusal=refusal,
            token=token,
            token_value=token_value,
            timestamp=self._clock(),
        )
        self._audit_writer(event)

    def _agent_result_to_command(
        self,
        result: Any,
        *,
        task_id: str = "",
        dry_run: bool,
    ) -> CommandResult:
        return CommandResult(
            success=bool(getattr(result, "success", False)),
            state=self.state,
            message=str(getattr(result, "message", "")),
            task_id=task_id,
            data={
                "dry_run": dry_run,
                "agent_data": dict(getattr(result, "data", {}) or {}),
            },
        )

    @staticmethod
    def _plan_to_dict(plan: Any) -> dict[str, Any]:
        if hasattr(plan, "__dict__"):
            return dict(plan.__dict__)
        if isinstance(plan, dict):
            return dict(plan)
        return {"plan": str(plan)}

    @staticmethod
    def _args_are_safe(allowed_args: dict[str, Any], requested_args: dict[str, Any]) -> bool:
        return dict(requested_args) == dict(allowed_args)

    @staticmethod
    def _bound_token_args(
        requested_args: dict[str, Any],
        execution_binding: dict[str, Any],
    ) -> dict[str, Any]:
        if not execution_binding:
            return dict(requested_args)
        return {
            "requested_args": dict(requested_args),
            "execution_binding": dict(execution_binding),
        }

    def _token_count(self) -> int:
        count = getattr(self._token_store, "count", None)
        if count is None:
            return 0
        return int(count())


__all__ = ["JsonTokenStore", "RobotDaemon", "StoredToken"]
