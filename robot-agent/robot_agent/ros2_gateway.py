from __future__ import annotations

import math
import threading
import time
from typing import Any

from .motion import DockTarget, MotionResult, PoseTarget


class Ros2MotionGateway:
    def __init__(
        self,
        *,
        navigate_action: str = "navigate_to_pose",
        dock_action: str = "dock_robot",
        cmd_vel_topic: str = "cmd_vel",
        heartbeat_topic: str = "robot_agent/heartbeat",
    ):
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from nav2_msgs.action import NavigateToPose
            from rclpy.action import ActionClient
            from rclpy.executors import MultiThreadedExecutor
            from std_msgs.msg import Empty
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 motion backend requires rclpy, Nav2 and ROS messages"
            ) from exc

        self._rclpy = rclpy
        self._twist_type = Twist
        self._navigate_type = NavigateToPose
        self._empty_type = Empty
        self._action_client_type = ActionClient
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self._node = rclpy.create_node("robot_agent_motion_gateway")
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._navigate_client = ActionClient(self._node, NavigateToPose, navigate_action)
        self._dock_client = self._build_dock_client(dock_action)
        self._cmd_vel = self._node.create_publisher(Twist, cmd_vel_topic, 10)
        self._heartbeat = self._node.create_publisher(Empty, heartbeat_topic, 10)
        self._heartbeat_timer = self._node.create_timer(0.1, self._publish_heartbeat)
        self._lock = threading.RLock()
        self._active_goal: Any | None = None
        self._active = False
        self._closed = False
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def health(self) -> dict[str, Any]:
        return {
            "available": not self._closed,
            "backend": "ros2_nav2",
            "navigate_server": self._navigate_client.server_is_ready(),
            "dock_server": bool(self._dock_client and self._dock_client.server_is_ready()),
            "active": self.active,
        }

    def navigate(self, target: PoseTarget, timeout_seconds: float) -> MotionResult:
        goal = self._navigate_type.Goal()
        goal.pose.header.frame_id = target.frame_id
        goal.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.pose.position.x = target.x
        goal.pose.pose.position.y = target.y
        goal.pose.pose.orientation.z = math.sin(target.yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(target.yaw / 2.0)
        return self._execute_action(
            self._navigate_client,
            goal,
            timeout_seconds,
            operation="navigate",
            target=target.name,
        )

    def dock(self, target: DockTarget, timeout_seconds: float) -> MotionResult:
        if self._dock_client is None:
            return MotionResult(False, "Open Navigation docking messages are not installed")
        goal = self._dock_goal_type.Goal()
        goal.dock_id = target.dock_id
        if hasattr(goal, "navigate_to_staging_pose"):
            goal.navigate_to_staging_pose = True
        return self._execute_action(
            self._dock_client,
            goal,
            timeout_seconds,
            operation="dock",
            target=target.name,
        )

    def cancel(self, *, reason: str, emergency: bool = False) -> None:
        with self._lock:
            goal = self._active_goal
        if goal is not None:
            try:
                goal.cancel_goal_async()
            except Exception:
                pass
        self._publish_zero_twist()

    def close(self) -> None:
        if self._closed:
            return
        self.cancel(reason="gateway_close", emergency=True)
        self._closed = True
        self._executor.shutdown(timeout_sec=2.0)
        self._thread.join(timeout=2.0)
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def _build_dock_client(self, action_name: str) -> Any | None:
        try:
            from opennav_docking_msgs.action import DockRobot
        except ImportError:
            self._dock_goal_type = None
            return None
        self._dock_goal_type = DockRobot
        return self._action_client_type(self._node, DockRobot, action_name)

    def _execute_action(
        self,
        client: Any,
        goal: Any,
        timeout_seconds: float,
        *,
        operation: str,
        target: str,
    ) -> MotionResult:
        with self._lock:
            if self._active:
                return MotionResult(False, "another motion command is active")
            self._active = True
        deadline = time.monotonic() + timeout_seconds
        try:
            if not client.wait_for_server(timeout_sec=min(5.0, timeout_seconds)):
                return MotionResult(False, f"{operation} action server is unavailable")
            goal_handle = self._wait_future(client.send_goal_async(goal), deadline)
            if goal_handle is None:
                return MotionResult(False, f"{operation} goal request timed out")
            if not goal_handle.accepted:
                return MotionResult(False, f"{operation} goal was rejected")
            with self._lock:
                self._active_goal = goal_handle
            response = self._wait_future(goal_handle.get_result_async(), deadline)
            if response is None:
                goal_handle.cancel_goal_async()
                self._publish_zero_twist()
                return MotionResult(False, f"{operation} timed out")
            status = int(response.status)
            success = status == 4
            return MotionResult(
                success,
                f"{operation} {'completed' if success else 'failed'} with status {status}",
                {"operation": operation, "target": target, "status": status},
            )
        finally:
            with self._lock:
                self._active_goal = None
                self._active = False

    @staticmethod
    def _wait_future(future: Any, deadline: float) -> Any | None:
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        remaining = max(0.0, deadline - time.monotonic())
        if not completed.wait(remaining):
            return None
        try:
            return future.result()
        except Exception:
            return None

    def _publish_heartbeat(self) -> None:
        self._heartbeat.publish(self._empty_type())

    def _publish_zero_twist(self) -> None:
        for _ in range(3):
            self._cmd_vel.publish(self._twist_type())


__all__ = ["Ros2MotionGateway"]
