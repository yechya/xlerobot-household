from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DISABLED = "DISABLED"
DRY_RUN = "DRY_RUN"
OBSERVE_ONLY = "OBSERVE_ONLY"
AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
ARMED_REAL = "ARMED_REAL"
EXECUTING = "EXECUTING"
PAUSED = "PAUSED"
STOPPING = "STOPPING"
ESTOP = "ESTOP"
ERROR = "ERROR"

RUNTIME_STATES: tuple[str, ...] = (
    DISABLED,
    DRY_RUN,
    OBSERVE_ONLY,
    AWAITING_CONFIRMATION,
    ARMED_REAL,
    EXECUTING,
    PAUSED,
    STOPPING,
    ESTOP,
    ERROR,
)


MISSING_TOKEN = "missing_token"
INVALID_TOKEN = "invalid_token"
EXPIRED_TOKEN = "expired_token"
TASK_NOT_ALLOWLISTED = "task_not_allowlisted"
ILLEGAL_STATE_TRANSITION = "illegal_state_transition"
HARDWARE_NOT_ARMED = "hardware_not_armed"
SESSION_LOCKED = "session_locked"
UNSAFE_ARGUMENTS = "unsafe_arguments"
LEGACY_ENTRYPOINT_DISABLED = "legacy_entrypoint_disabled"
OBSERVATION_REQUIRED = "observation_required"
SAFETY_INTERLOCK_BLOCKED = "safety_interlock_blocked"

REFUSAL_REASON_CODES: tuple[str, ...] = (
    MISSING_TOKEN,
    INVALID_TOKEN,
    EXPIRED_TOKEN,
    TASK_NOT_ALLOWLISTED,
    ILLEGAL_STATE_TRANSITION,
    HARDWARE_NOT_ARMED,
    SESSION_LOCKED,
    UNSAFE_ARGUMENTS,
    LEGACY_ENTRYPOINT_DISABLED,
    OBSERVATION_REQUIRED,
    SAFETY_INTERLOCK_BLOCKED,
)


LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    DISABLED: frozenset({DRY_RUN, OBSERVE_ONLY}),
    DRY_RUN: frozenset({DRY_RUN, OBSERVE_ONLY, AWAITING_CONFIRMATION, DISABLED, ERROR}),
    OBSERVE_ONLY: frozenset({DRY_RUN, AWAITING_CONFIRMATION, DISABLED, ERROR}),
    AWAITING_CONFIRMATION: frozenset({ARMED_REAL, DRY_RUN, DISABLED, ERROR}),
    ARMED_REAL: frozenset({EXECUTING, PAUSED, STOPPING, DRY_RUN, ESTOP, ERROR}),
    EXECUTING: frozenset({PAUSED, STOPPING, ARMED_REAL, DRY_RUN, ESTOP, ERROR}),
    PAUSED: frozenset({EXECUTING, ARMED_REAL, STOPPING, DRY_RUN, DISABLED, ESTOP, ERROR}),
    STOPPING: frozenset({ARMED_REAL, DRY_RUN, DISABLED, ESTOP, ERROR}),
    ESTOP: frozenset({DISABLED}),
    ERROR: frozenset({DISABLED, DRY_RUN}),
}


@dataclass
class RobotCommand:
    command_id: str
    task_id: str
    requested_by: str = ""
    state: str = DRY_RUN
    arguments: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = True
    requires_hardware: bool = False


@dataclass
class Refusal:
    reason_code: str
    message: str = ""
    state: str = ""
    command_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenMetadata:
    token_id: str
    issued_to: str
    issued_at: float
    expires_at: float
    scopes: list[str] = field(default_factory=list)
    session_id: str = ""


@dataclass
class CommandResult:
    success: bool
    state: str
    message: str = ""
    command_id: str = ""
    task_id: str = ""
    refusal: Refusal | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    event_type: str
    state: str
    command_id: str = ""
    task_id: str = ""
    actor: str = ""
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


def is_known_state(state: str) -> bool:
    return state in RUNTIME_STATES


def is_refusal_reason(reason_code: str) -> bool:
    return reason_code in REFUSAL_REASON_CODES


def is_legal_transition(from_state: str, to_state: str) -> bool:
    return to_state in LEGAL_TRANSITIONS.get(from_state, frozenset())


def validate_transition(from_state: str, to_state: str) -> Refusal | None:
    if is_legal_transition(from_state, to_state):
        return None
    return Refusal(
        reason_code=ILLEGAL_STATE_TRANSITION,
        message=f"Illegal robot-control state transition: {from_state} -> {to_state}",
        state=from_state,
        details={"from_state": from_state, "to_state": to_state},
    )
