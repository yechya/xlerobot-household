from __future__ import annotations

from collections.abc import Callable
from typing import Any


def run_ros_node(
    rclpy: Any,
    node_factory: Callable[[], Any],
    external_shutdown_exception: type[BaseException],
) -> None:
    node = None
    try:
        rclpy.init(args=None)
        node = node_factory()
        rclpy.spin(node)
    except (KeyboardInterrupt, external_shutdown_exception):
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


__all__ = ["run_ros_node"]
