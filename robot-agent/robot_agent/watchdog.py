from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("robot_agent.watchdog")


@dataclass
class HeartbeatState:
    timeout_seconds: float
    last_heartbeat_at: float
    tripped: bool = False

    def heartbeat(self, now: float) -> None:
        self.last_heartbeat_at = now
        self.tripped = False

    def check(self, now: float) -> tuple[bool, bool]:
        expired = now - self.last_heartbeat_at > self.timeout_seconds
        newly_tripped = expired and not self.tripped
        self.tripped = expired
        return expired, newly_tripped


def main() -> None:
    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.node import Node
        from std_msgs.msg import Empty
        from std_srvs.srv import Trigger
    except ImportError as exc:
        raise SystemExit("robot-watchdog requires ROS 2 rclpy and standard messages") from exc

    heartbeat_topic = os.environ.get(
        "ROBOT_AGENT_HEARTBEAT_TOPIC",
        "robot_agent/heartbeat",
    )
    cmd_vel_topic = os.environ.get("ROBOT_AGENT_CMD_VEL_TOPIC", "cmd_vel")
    estop_service = os.environ.get("ROBOT_AGENT_WATCHDOG_ESTOP_SERVICE", "").strip()
    timeout_seconds = float(os.environ.get("ROBOT_AGENT_WATCHDOG_TIMEOUT", "0.5"))
    check_period = min(0.1, max(0.02, timeout_seconds / 4.0))

    class WatchdogNode(Node):
        def __init__(self) -> None:
            super().__init__("robot_agent_heartbeat_watchdog")
            self.state = HeartbeatState(timeout_seconds, time.monotonic())
            self.velocity_publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
            self.create_subscription(Empty, heartbeat_topic, self.on_heartbeat, 10)
            self.estop_client = (
                self.create_client(Trigger, estop_service) if estop_service else None
            )
            self.create_timer(check_period, self.on_timer)

        def on_heartbeat(self, _message: Any) -> None:
            self.state.heartbeat(time.monotonic())

        def on_timer(self) -> None:
            expired, newly_tripped = self.state.check(time.monotonic())
            if not expired:
                return
            self.velocity_publisher.publish(Twist())
            if newly_tripped:
                self.get_logger().error("robot-agent heartbeat expired; commanding zero velocity")
                if self.estop_client and self.estop_client.service_is_ready():
                    self.estop_client.call_async(Trigger.Request())

    logging.basicConfig(level=os.environ.get("ROBOT_AGENT_LOG_LEVEL", "INFO").upper())
    rclpy.init(args=None)
    node = WatchdogNode()
    try:
        rclpy.spin(node)
    finally:
        node.velocity_publisher.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


__all__ = ["HeartbeatState", "main"]
