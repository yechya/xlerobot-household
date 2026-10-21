from __future__ import annotations

from collections.abc import Callable

import pytest

from robot_agent.ros_node_lifecycle import run_ros_node


class ExternalShutdown(Exception):
    pass


class FakeNode:
    def __init__(self, calls: list[str]):
        self.calls = calls

    def destroy_node(self) -> None:
        self.calls.append("destroy")


class FakeRclpy:
    def __init__(
        self,
        calls: list[str],
        *,
        spin_effect: Callable[[], None] | None = None,
        context_ok: bool = True,
    ):
        self.calls = calls
        self.spin_effect = spin_effect
        self.context_ok = context_ok

    def init(self, *, args: None) -> None:
        self.calls.append("init")

    def spin(self, node: FakeNode) -> None:
        self.calls.append("spin")
        if self.spin_effect is not None:
            self.spin_effect()

    def ok(self) -> bool:
        self.calls.append("ok")
        return self.context_ok

    def shutdown(self) -> None:
        self.calls.append("shutdown")


def _raise(exception: BaseException) -> Callable[[], None]:
    def raise_exception() -> None:
        raise exception

    return raise_exception


def test_external_shutdown_destroys_node_without_second_context_shutdown():
    calls: list[str] = []
    rclpy = FakeRclpy(
        calls,
        spin_effect=_raise(ExternalShutdown()),
        context_ok=False,
    )

    run_ros_node(rclpy, lambda: FakeNode(calls), ExternalShutdown)

    assert calls == ["init", "spin", "destroy", "ok"]


@pytest.mark.parametrize("exception", [KeyboardInterrupt(), ExternalShutdown()])
def test_operator_shutdown_is_a_clean_exit(exception: BaseException):
    calls: list[str] = []
    rclpy = FakeRclpy(calls, spin_effect=_raise(exception))

    run_ros_node(rclpy, lambda: FakeNode(calls), ExternalShutdown)

    assert calls == ["init", "spin", "destroy", "ok", "shutdown"]


def test_normal_spin_return_destroys_node_before_context_shutdown():
    calls: list[str] = []
    rclpy = FakeRclpy(calls)

    run_ros_node(rclpy, lambda: FakeNode(calls), ExternalShutdown)

    assert calls == ["init", "spin", "destroy", "ok", "shutdown"]


def test_node_construction_error_still_closes_live_context():
    calls: list[str] = []
    rclpy = FakeRclpy(calls)

    def fail_to_construct() -> FakeNode:
        raise RuntimeError("invalid monitor configuration")

    with pytest.raises(RuntimeError, match="invalid monitor configuration"):
        run_ros_node(rclpy, fail_to_construct, ExternalShutdown)

    assert calls == ["init", "ok", "shutdown"]
