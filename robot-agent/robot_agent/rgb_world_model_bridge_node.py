from __future__ import annotations

from pathlib import Path
from typing import Any

from .rgb_world_model_bridge import RgbWorldModelBridge
from .ros_node_lifecycle import run_ros_node


def main() -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from xlerobot_manipulation_msgs.msg import SegmentationFrame
    except ImportError as exc:
        raise SystemExit(
            "RGB world-model bridge requires ROS 2 rclpy and xlerobot manipulation messages"
        ) from exc

    class RgbWorldModelBridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("rgb_world_model_bridge")
            self.declare_parameter("world_model_path", "")
            self.declare_parameter("segmentation_topic", "/xlerobot/segmentation")
            self.declare_parameter("max_input_age_seconds", 0.5)
            self.declare_parameter("source", "rgb_segmentation_bridge")
            configured = str(self.get_parameter("world_model_path").value).strip()
            if not configured:
                raise RuntimeError(
                    "world_model_path must be explicitly configured; bridge is disabled by default"
                )
            path = Path(configured).expanduser()
            if not path.is_absolute():
                raise RuntimeError("world_model_path must be an absolute file path")
            self.bridge = RgbWorldModelBridge(
                path,
                max_input_age_seconds=float(
                    self.get_parameter("max_input_age_seconds").value
                ),
                source=str(self.get_parameter("source").value),
            )
            topic = str(self.get_parameter("segmentation_topic").value).strip()
            if not topic.startswith("/"):
                raise RuntimeError("segmentation_topic must be an absolute ROS topic")
            self.subscription = self.create_subscription(
                SegmentationFrame,
                topic,
                self._on_frame,
                qos_profile_sensor_data,
            )
            self.get_logger().info(
                f"RGB world-model bridge ready topic={topic} path={path}"
            )

        def _on_frame(self, message: Any) -> None:
            result = self.bridge.handle_frame(message)
            if result.success:
                self.get_logger().debug(result.message)
            else:
                self.get_logger().error(result.message)

    run_ros_node(rclpy, RgbWorldModelBridgeNode, KeyboardInterrupt)


if __name__ == "__main__":
    main()


__all__ = ["main"]
