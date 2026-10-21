from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VelocityCommand:
    x: float
    y: float
    yaw: float

    @classmethod
    def zero(cls) -> "VelocityCommand":
        return cls(0.0, 0.0, 0.0)

    def is_zero(self) -> bool:
        return self.x == 0.0 and self.y == 0.0 and self.yaw == 0.0


@dataclass(frozen=True)
class BridgeLimits:
    max_linear_velocity: float
    max_angular_velocity: float
    command_timeout_seconds: float = 0.25
    observation_timeout_seconds: float = 0.25
    max_odometry_step_seconds: float = 0.1

    def __post_init__(self) -> None:
        for name, value in (
            ("max_linear_velocity", self.max_linear_velocity),
            ("max_angular_velocity", self.max_angular_velocity),
            ("command_timeout_seconds", self.command_timeout_seconds),
            ("observation_timeout_seconds", self.observation_timeout_seconds),
            ("max_odometry_step_seconds", self.max_odometry_step_seconds),
        ):
            if not _finite_number(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.command_timeout_seconds > 0.5:
            raise ValueError("command_timeout_seconds must not exceed the host watchdog")


class CommandLimiter:
    def __init__(self, limits: BridgeLimits):
        self.limits = limits
        self._latest = VelocityCommand.zero()
        self._last_command_at: float | None = None

    def update(self, x: Any, y: Any, yaw: Any, now: float) -> VelocityCommand:
        if not all(_finite_number(value) for value in (x, y, yaw, now)):
            self._latest = VelocityCommand.zero()
            self._last_command_at = None
            raise ValueError("velocity command and timestamp must be finite numbers")
        x_value = float(x)
        y_value = float(y)
        magnitude = math.hypot(x_value, y_value)
        if magnitude > self.limits.max_linear_velocity:
            scale = self.limits.max_linear_velocity / magnitude
            x_value *= scale
            y_value *= scale
        yaw_value = max(
            -self.limits.max_angular_velocity,
            min(self.limits.max_angular_velocity, float(yaw)),
        )
        self._latest = VelocityCommand(x_value, y_value, yaw_value)
        self._last_command_at = float(now)
        return self._latest

    def command_for_send(self, now: float) -> tuple[VelocityCommand, bool]:
        if not _finite_number(now):
            return VelocityCommand.zero(), True
        expired = (
            self._last_command_at is None
            or float(now) - self._last_command_at > self.limits.command_timeout_seconds
        )
        return (VelocityCommand.zero(), True) if expired else (self._latest, False)


@dataclass
class OdometryState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    velocity: VelocityCommand = VelocityCommand(0.0, 0.0, 0.0)
    last_observation_at: float | None = None
    stale: bool = True


class OdometryIntegrator:
    def __init__(self, limits: BridgeLimits):
        self.limits = limits
        self.state = OdometryState()

    def update(self, x: Any, y: Any, yaw: Any, now: float) -> OdometryState:
        if not all(_finite_number(value) for value in (x, y, yaw, now)):
            raise ValueError("odometry velocity and timestamp must be finite numbers")
        velocity = VelocityCommand(float(x), float(y), float(yaw))
        timestamp = float(now)
        previous_at = self.state.last_observation_at
        if previous_at is not None:
            delta = timestamp - previous_at
            if delta < 0:
                raise ValueError("odometry timestamps must be monotonic")
            if delta <= self.limits.observation_timeout_seconds:
                delta = min(delta, self.limits.max_odometry_step_seconds)
                cos_yaw = math.cos(self.state.yaw)
                sin_yaw = math.sin(self.state.yaw)
                self.state.x += (velocity.x * cos_yaw - velocity.y * sin_yaw) * delta
                self.state.y += (velocity.x * sin_yaw + velocity.y * cos_yaw) * delta
                self.state.yaw = normalize_angle(self.state.yaw + velocity.yaw * delta)
        self.state.velocity = velocity
        self.state.last_observation_at = timestamp
        self.state.stale = False
        return self.state

    def snapshot(self, now: float) -> OdometryState:
        if not _finite_number(now):
            self.state.stale = True
            self.state.velocity = VelocityCommand.zero()
            return self.state
        self.state.stale = bool(
            self.state.last_observation_at is None
            or float(now) - self.state.last_observation_at
            > self.limits.observation_timeout_seconds
        )
        if self.state.stale:
            self.state.velocity = VelocityCommand.zero()
        return self.state


def command_to_host(command: VelocityCommand) -> dict[str, float]:
    return {
        "x.vel": command.x,
        "y.vel": command.y,
        "theta.vel": math.degrees(command.yaw),
    }


def observation_from_host(payload: Any) -> VelocityCommand:
    if not isinstance(payload, dict):
        raise ValueError("host observation must be a JSON object")
    values = [payload.get("x.vel"), payload.get("y.vel"), payload.get("theta.vel")]
    if not all(_finite_number(value) for value in values):
        raise ValueError("host observation requires finite x.vel, y.vel and theta.vel")
    return VelocityCommand(float(values[0]), float(values[1]), math.radians(float(values[2])))


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


__all__ = [
    "BridgeLimits",
    "CommandLimiter",
    "OdometryIntegrator",
    "OdometryState",
    "VelocityCommand",
    "command_to_host",
    "normalize_angle",
    "observation_from_host",
]
