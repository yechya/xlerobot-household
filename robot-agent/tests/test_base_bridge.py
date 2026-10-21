from __future__ import annotations

import math

import pytest

from robot_agent.base_bridge import (
    BridgeLimits,
    CommandLimiter,
    OdometryIntegrator,
    VelocityCommand,
    command_to_host,
    observation_from_host,
)


@pytest.fixture
def limits() -> BridgeLimits:
    return BridgeLimits(
        max_linear_velocity=0.15,
        max_angular_velocity=0.5,
        command_timeout_seconds=0.25,
        observation_timeout_seconds=0.25,
        max_odometry_step_seconds=0.1,
    )


def test_holonomic_command_is_vector_limited_and_yaw_clamped(limits: BridgeLimits):
    limiter = CommandLimiter(limits)
    command = limiter.update(0.15, 0.15, 1.0, now=10.0)
    assert math.hypot(command.x, command.y) == pytest.approx(0.15)
    assert command.x == pytest.approx(command.y)
    assert command.yaw == 0.5


def test_command_timeout_returns_zero_before_host_watchdog(limits: BridgeLimits):
    limiter = CommandLimiter(limits)
    limiter.update(0.1, 0.0, 0.2, now=1.0)
    current, expired = limiter.command_for_send(1.25)
    stale, stale_expired = limiter.command_for_send(1.251)
    assert current == VelocityCommand(0.1, 0.0, 0.2)
    assert expired is False
    assert stale.is_zero()
    assert stale_expired is True


def test_non_finite_command_fails_closed(limits: BridgeLimits):
    limiter = CommandLimiter(limits)
    limiter.update(0.1, 0.0, 0.0, now=1.0)
    with pytest.raises(ValueError, match="finite"):
        limiter.update(float("nan"), 0.0, 0.0, now=1.1)
    command, expired = limiter.command_for_send(1.1)
    assert command.is_zero()
    assert expired is True


def test_host_protocol_converts_ros_radians_to_degrees():
    payload = command_to_host(VelocityCommand(0.1, -0.05, math.pi / 2))
    restored = observation_from_host(payload)
    assert payload == {"x.vel": 0.1, "y.vel": -0.05, "theta.vel": 90.0}
    assert restored.yaw == pytest.approx(math.pi / 2)


def test_odometry_integrates_body_velocity_in_world_frame(limits: BridgeLimits):
    integrator = OdometryIntegrator(limits)
    integrator.update(1.0, 0.0, math.pi / 2, now=0.0)
    first = integrator.update(1.0, 0.0, math.pi / 2, now=0.1)
    assert first.x == pytest.approx(0.1)
    second = integrator.update(1.0, 0.0, 0.0, now=0.2)
    assert second.x == pytest.approx(0.1 + 0.1 * math.cos(math.pi / 20))
    assert second.y == pytest.approx(0.1 * math.sin(math.pi / 20))


def test_observation_gap_does_not_integrate_unbounded_distance(limits: BridgeLimits):
    integrator = OdometryIntegrator(limits)
    integrator.update(0.1, 0.0, 0.0, now=0.0)
    state = integrator.update(0.1, 0.0, 0.0, now=1.0)
    assert state.x == 0.0
    stale = integrator.snapshot(1.251)
    assert stale.stale is True
    assert stale.velocity.is_zero()
