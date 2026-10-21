from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

try:
    from ..contract import HARDWARE_NOT_ARMED, MISSING_TOKEN, OBSERVE_ONLY
except ImportError:
    from robot_agent.contract import HARDWARE_NOT_ARMED, MISSING_TOKEN, OBSERVE_ONLY


_CAPABILITY_SENTINEL = object()
_AUTHORIZATION_CONTEXT: ContextVar[object | None] = ContextVar(
    "robotd_hardware_authorization",
    default=None,
)


@dataclass(frozen=True)
class HardwareAuthorizationRefusal:
    reason_code: str
    message: str
    action: str
    task_id: str = ""

    def to_result(self, **details: object) -> dict:
        payload = {
            "success": False,
            "error": self.message,
            "reason_code": self.reason_code,
            "refusal": {
                "reason_code": self.reason_code,
                "message": self.message,
            },
            "action": self.action,
        }
        if self.task_id:
            payload["task_id"] = self.task_id
            payload["refusal"]["task_id"] = self.task_id
        payload.update(details)
        return payload


@dataclass(frozen=True)
class _RobotdHardwareAuthorization:
    token: str
    task_id: str
    execution_binding: dict[str, Any]
    sentinel: object


def _authorization_token() -> str:
    authorization = _AUTHORIZATION_CONTEXT.get()
    if not _is_valid_robotd_authorization(authorization):
        return ""
    return authorization.token


def _authorized_task_id() -> str:
    authorization = _AUTHORIZATION_CONTEXT.get()
    if not _is_valid_robotd_authorization(authorization):
        return ""
    return authorization.task_id


def _authorized_execution_binding() -> dict[str, Any]:
    authorization = _AUTHORIZATION_CONTEXT.get()
    if not _is_valid_robotd_authorization(authorization):
        return {}
    return dict(authorization.execution_binding)


def _is_valid_robotd_authorization(authorization: object | None) -> bool:
    return (
        isinstance(authorization, _RobotdHardwareAuthorization)
        and authorization.sentinel is _CAPABILITY_SENTINEL
        and bool(authorization.token)
        and bool(authorization.task_id)
    )


def _make_robotd_hardware_authorization(
    token: str,
    task_id: str,
    execution_binding: dict[str, Any] | None = None,
) -> _RobotdHardwareAuthorization:
    return _RobotdHardwareAuthorization(
        token=token,
        task_id=task_id,
        execution_binding=dict(execution_binding or {}),
        sentinel=_CAPABILITY_SENTINEL,
    )


def validate_actuation_authorized(
    action: str,
    task_id: str = "",
) -> HardwareAuthorizationRefusal | None:
    if not _authorization_token():
        return HardwareAuthorizationRefusal(
            reason_code=MISSING_TOKEN,
            message="hardware actuation requires a robotd authorization token",
            action=action,
            task_id=task_id,
        )

    authorized_task_id = _authorized_task_id()
    if not authorized_task_id:
        return HardwareAuthorizationRefusal(
            reason_code=HARDWARE_NOT_ARMED,
            message="hardware actuation requires a robotd authorized task marker",
            action=action,
            task_id=task_id,
        )
    if task_id and authorized_task_id != task_id:
        return HardwareAuthorizationRefusal(
            reason_code=HARDWARE_NOT_ARMED,
            message=f"robotd authorization is for {authorized_task_id!r}, not {task_id!r}",
            action=action,
            task_id=task_id,
        )
    return None


@contextmanager
def _robotd_authorized_context(authorization: _RobotdHardwareAuthorization) -> Iterator[None]:
    if not _is_valid_robotd_authorization(authorization):
        raise ValueError("invalid robotd hardware authorization")
    context_marker = _AUTHORIZATION_CONTEXT.set(authorization)
    try:
        yield
    finally:
        _AUTHORIZATION_CONTEXT.reset(context_marker)


def is_observation_only_state(state: str) -> bool:
    return state == OBSERVE_ONLY
