from __future__ import annotations

from pathlib import Path
from typing import Any

from .segmentation_backends import (
    SegmentationBackendError,
    SegmentationBackendSpec,
    image_request_from_ros,
    load_backend,
    run_inference,
)


def ros_time_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000


def populate_segmentation_frame(
    message: Any,
    image_message: Any,
    predictions: tuple[Any, ...],
    instance_type: Any,
    *,
    source: str,
) -> Any:
    message.header = image_message.header
    message.image_width = int(image_message.width)
    message.image_height = int(image_message.height)
    instances = []
    for prediction in predictions:
        instance = instance_type()
        instance.instance_id = prediction.instance_id
        instance.label = prediction.label
        instance.confidence = prediction.confidence
        instance.x_offset = prediction.x_offset
        instance.y_offset = prediction.y_offset
        instance.width = prediction.width
        instance.height = prediction.height
        instance.mask = list(prediction.mask)
        instance.source = source
        instances.append(instance)
    message.instances = instances
    return message


def main() -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from xlerobot_manipulation_msgs.msg import InstanceMask, SegmentationFrame
    except ImportError as exc:
        raise SystemExit(
            "segmentation adapter requires rclpy, sensor_msgs and manipulation messages"
        ) from exc

    class SegmentationAdapterNode(Node):
        def __init__(self) -> None:
            super().__init__("segmentation_adapter")
            self.declare_parameter("backend_config_path", "")
            configured = str(self.get_parameter("backend_config_path").value)
            path = Path(configured).expanduser()
            if not path.is_absolute() or not path.is_file():
                raise RuntimeError("backend_config_path must be an existing absolute file")
            self.spec = SegmentationBackendSpec.load(path)
            self.backend = load_backend(self.spec)
            self.publisher = self.create_publisher(
                SegmentationFrame,
                self.spec.output_topic,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Image,
                self.spec.image_topic,
                self._on_image,
                qos_profile_sensor_data,
            )
            self.get_logger().info(
                f"segmentation backend ready id={self.spec.backend_id} "
                f"fingerprint={self.spec.fingerprint}"
            )

        def _on_image(self, message: Any) -> None:
            try:
                captured_at = ros_time_to_seconds(message.header.stamp)
                now = self.get_clock().now().nanoseconds / 1_000_000_000
                age = now - captured_at
                if age < 0 or age > self.spec.max_input_age_seconds:
                    raise SegmentationBackendError(
                        "segmentation input image is stale or from the future"
                    )
                request = image_request_from_ros(
                    message,
                    self.spec,
                    captured_at=captured_at,
                )
                predictions = run_inference(self.backend, request, self.spec)
                source = f"{self.spec.source_id}/{self.spec.backend_id}"
                self.publisher.publish(
                    populate_segmentation_frame(
                        SegmentationFrame(),
                        message,
                        predictions,
                        InstanceMask,
                        source=source,
                    )
                )
            except (SegmentationBackendError, TypeError, ValueError) as exc:
                self.get_logger().error(f"segmentation frame rejected: {exc}")

    rclpy.init(args=None)
    node = SegmentationAdapterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


__all__ = ["main", "populate_segmentation_frame"]
