from __future__ import annotations

import json
import math
import time
from typing import Any

from .base_bridge import (
    BridgeLimits,
    CommandLimiter,
    OdometryIntegrator,
    VelocityCommand,
    command_to_host,
    observation_from_host,
)


def main() -> None:
    try:
        import rclpy
        import zmq
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
        from geometry_msgs.msg import TransformStamped, Twist
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from tf2_ros import TransformBroadcaster
    except ImportError as exc:
        raise SystemExit(
            "xlerobot base bridge requires ROS 2 messages, tf2_ros and pyzmq"
        ) from exc

    class BaseBridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("base_bridge")
            self._declare_parameters()
            if not self.get_parameter("calibration_approved").value:
                raise RuntimeError("base bridge calibration_approved must be true")
            self.limits = BridgeLimits(
                max_linear_velocity=float(self.get_parameter("max_linear_velocity").value),
                max_angular_velocity=float(self.get_parameter("max_angular_velocity").value),
                command_timeout_seconds=float(
                    self.get_parameter("command_timeout_seconds").value
                ),
                observation_timeout_seconds=float(
                    self.get_parameter("observation_timeout_seconds").value
                ),
                max_odometry_step_seconds=float(
                    self.get_parameter("max_odometry_step_seconds").value
                ),
            )
            self.command_limiter = CommandLimiter(self.limits)
            self.odometry = OdometryIntegrator(self.limits)
            self.command_expired = True
            self.invalid_command_count = 0
            self.invalid_observation_count = 0
            self._cycle = 0

            self.base_frame = str(self.get_parameter("base_frame").value)
            self.odom_frame = str(self.get_parameter("odom_frame").value)
            self.publish_tf = bool(self.get_parameter("publish_tf").value)
            self.pose_covariance = _covariance(
                self.get_parameter("pose_covariance_diagonal").value
            )
            self.twist_covariance = _covariance(
                self.get_parameter("twist_covariance_diagonal").value
            )
            command_topic = str(self.get_parameter("command_topic").value)
            odom_topic = str(self.get_parameter("odom_topic").value)
            diagnostics_topic = str(self.get_parameter("diagnostics_topic").value)
            self.create_subscription(Twist, command_topic, self._on_command, 10)
            self.odom_publisher = self.create_publisher(Odometry, odom_topic, 20)
            self.diagnostics_publisher = self.create_publisher(
                DiagnosticArray,
                diagnostics_topic,
                10,
            )
            self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

            remote_ip = str(self.get_parameter("remote_ip").value)
            command_port = int(self.get_parameter("command_port").value)
            observation_port = int(self.get_parameter("observation_port").value)
            self.zmq_context = zmq.Context()
            self.command_socket = self.zmq_context.socket(zmq.PUSH)
            self.command_socket.setsockopt(zmq.CONFLATE, 1)
            self.command_socket.setsockopt(zmq.LINGER, 0)
            self.command_socket.connect(f"tcp://{remote_ip}:{command_port}")
            self.observation_socket = self.zmq_context.socket(zmq.PULL)
            self.observation_socket.setsockopt(zmq.CONFLATE, 1)
            self.observation_socket.setsockopt(zmq.LINGER, 0)
            self.observation_socket.connect(f"tcp://{remote_ip}:{observation_port}")

            stream_rate = float(self.get_parameter("stream_rate_hz").value)
            self.timer = self.create_timer(1.0 / stream_rate, self._on_timer)
            self.get_logger().info(
                f"base bridge ready for {remote_ip}:{command_port}/{observation_port}"
            )

        def _declare_parameters(self) -> None:
            self.declare_parameter("calibration_approved", False)
            self.declare_parameter("remote_ip", "127.0.0.1")
            self.declare_parameter("command_port", 5555)
            self.declare_parameter("observation_port", 5556)
            self.declare_parameter("command_topic", "/xlerobot/base/cmd_vel_safe")
            self.declare_parameter("odom_topic", "/odom")
            self.declare_parameter("diagnostics_topic", "/diagnostics")
            self.declare_parameter("base_frame", "base_link")
            self.declare_parameter("odom_frame", "odom")
            self.declare_parameter("publish_tf", True)
            self.declare_parameter("stream_rate_hz", 30.0)
            self.declare_parameter("max_linear_velocity", 0.15)
            self.declare_parameter("max_angular_velocity", 0.5)
            self.declare_parameter("command_timeout_seconds", 0.25)
            self.declare_parameter("observation_timeout_seconds", 0.25)
            self.declare_parameter("max_odometry_step_seconds", 0.1)
            self.declare_parameter(
                "pose_covariance_diagonal",
                [0.05, 0.05, 999.0, 999.0, 999.0, 0.1],
            )
            self.declare_parameter(
                "twist_covariance_diagonal",
                [0.02, 0.02, 999.0, 999.0, 999.0, 0.05],
            )

        def _on_command(self, message: Any) -> None:
            try:
                self.command_limiter.update(
                    message.linear.x,
                    message.linear.y,
                    message.angular.z,
                    time.monotonic(),
                )
            except ValueError as exc:
                self.invalid_command_count += 1
                self.get_logger().error(f"unsafe velocity command rejected: {exc}")

        def _on_timer(self) -> None:
            now = time.monotonic()
            command, self.command_expired = self.command_limiter.command_for_send(now)
            self.command_socket.send_string(
                json.dumps(command_to_host(command), separators=(",", ":")),
                flags=zmq.NOBLOCK,
            )
            latest = None
            while True:
                try:
                    latest = self.observation_socket.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
            if latest is not None:
                self._handle_observation(latest, now)
            self.odometry.snapshot(now)
            self._cycle += 1
            if self._cycle % 15 == 0:
                self._publish_diagnostics()

        def _handle_observation(self, raw: str, now: float) -> None:
            try:
                observed = observation_from_host(json.loads(raw))
                state = self.odometry.update(observed.x, observed.y, observed.yaw, now)
            except (ValueError, json.JSONDecodeError) as exc:
                self.invalid_observation_count += 1
                self.get_logger().error(f"invalid host observation rejected: {exc}")
                return
            stamp = self.get_clock().now().to_msg()
            orientation_z = math.sin(state.yaw / 2.0)
            orientation_w = math.cos(state.yaw / 2.0)
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = self.odom_frame
            odom.child_frame_id = self.base_frame
            odom.pose.pose.position.x = state.x
            odom.pose.pose.position.y = state.y
            odom.pose.pose.orientation.z = orientation_z
            odom.pose.pose.orientation.w = orientation_w
            odom.pose.covariance = self.pose_covariance
            odom.twist.twist.linear.x = observed.x
            odom.twist.twist.linear.y = observed.y
            odom.twist.twist.angular.z = observed.yaw
            odom.twist.covariance = self.twist_covariance
            self.odom_publisher.publish(odom)
            if self.tf_broadcaster:
                transform = TransformStamped()
                transform.header.stamp = stamp
                transform.header.frame_id = self.odom_frame
                transform.child_frame_id = self.base_frame
                transform.transform.translation.x = state.x
                transform.transform.translation.y = state.y
                transform.transform.rotation.z = orientation_z
                transform.transform.rotation.w = orientation_w
                self.tf_broadcaster.sendTransform(transform)

        def _publish_diagnostics(self) -> None:
            state = self.odometry.state
            status = DiagnosticStatus()
            status.name = "xlerobot/base_bridge"
            status.hardware_id = "xlerobot-zmq-host"
            if state.stale:
                status.level = DiagnosticStatus.ERROR
                status.message = "host observation is stale"
            elif self.command_expired:
                status.level = DiagnosticStatus.WARN
                status.message = "command stale; streaming zero velocity"
            else:
                status.level = DiagnosticStatus.OK
                status.message = "bridge streaming"
            status.values = [
                KeyValue(key="command_expired", value=str(self.command_expired).lower()),
                KeyValue(key="observation_stale", value=str(state.stale).lower()),
                KeyValue(key="invalid_commands", value=str(self.invalid_command_count)),
                KeyValue(
                    key="invalid_observations",
                    value=str(self.invalid_observation_count),
                ),
            ]
            message = DiagnosticArray()
            message.header.stamp = self.get_clock().now().to_msg()
            message.status = [status]
            self.diagnostics_publisher.publish(message)

        def stop_and_close(self) -> None:
            zero = json.dumps(command_to_host(VelocityCommand.zero()))
            for _ in range(3):
                try:
                    self.command_socket.send_string(zero, flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
            self.command_socket.close(linger=0)
            self.observation_socket.close(linger=0)
            self.zmq_context.term()

    def _covariance(diagonal: Any) -> list[float]:
        if not isinstance(diagonal, (list, tuple)) or len(diagonal) != 6:
            raise ValueError("covariance diagonal must contain six values")
        covariance = [0.0] * 36
        for index, value in enumerate(diagonal):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError("covariance values must be non-negative numbers")
            covariance[index * 6 + index] = float(value)
        return covariance

    rclpy.init(args=None)
    node = BaseBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.stop_and_close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


__all__ = ["main"]
