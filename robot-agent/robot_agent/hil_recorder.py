from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .hil import HilEvent


class JsonlEventWriter:
    def __init__(
        self,
        path: str | Path,
        scenario_id: str,
        scenario: str,
        *,
        exclusive: bool = False,
    ):
        self.path = Path(path)
        self.scenario_id = scenario_id
        self.scenario = scenario
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_WRONLY | os.O_CREAT
        flags |= os.O_EXCL if exclusive else os.O_APPEND
        descriptor = os.open(self.path, flags, 0o600)
        self._handle = os.fdopen(
            descriptor,
            "w" if exclusive else "a",
            encoding="utf-8",
        )

    def write(self, kind: str, data: dict[str, Any] | None = None) -> None:
        self.write_event(
            HilEvent(
                scenario_id=self.scenario_id,
                scenario=self.scenario,
                kind=kind,
                timestamp=time.monotonic(),
                data=data or {},
            )
        )

    def write_event(self, event: HilEvent) -> None:
        if event.scenario_id != self.scenario_id or event.scenario != self.scenario:
            raise ValueError("HIL event does not match the bound recorder scenario")
        payload = {
            "scenario_id": event.scenario_id,
            "scenario": event.scenario,
            "kind": event.kind,
            "timestamp": event.timestamp,
            "data": event.data,
        }
        self._handle.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record passive ROS 2 HIL safety evidence")
    parser.add_argument("output")
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--odom-topic", default="odom")
    parser.add_argument("--fault-topic", default="robot_agent/hil/fault")
    parser.add_argument("--collision-topic", default="robot_agent/hil/collision")
    parser.add_argument(
        "--action-status-topic",
        default="navigate_to_pose/_action/status",
    )
    args = parser.parse_args()

    try:
        import rclpy
        from action_msgs.msg import GoalStatus, GoalStatusArray
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from std_msgs.msg import Bool, String
    except ImportError as exc:
        raise SystemExit("robot-hil-record requires ROS 2 and Nav2 message packages") from exc

    writer = JsonlEventWriter(args.output, args.scenario_id, args.scenario)
    terminal_names = {
        GoalStatus.STATUS_SUCCEEDED: "succeeded",
        GoalStatus.STATUS_CANCELED: "canceled",
        GoalStatus.STATUS_ABORTED: "aborted",
    }

    class HilRecorderNode(Node):
        def __init__(self) -> None:
            super().__init__("robot_agent_hil_recorder")
            self.seen_terminal: set[tuple[bytes, int]] = set()
            self.create_subscription(Odometry, args.odom_topic, self.on_odom, 50)
            self.create_subscription(String, args.fault_topic, self.on_fault, 10)
            self.create_subscription(Bool, args.collision_topic, self.on_collision, 10)
            self.create_subscription(
                GoalStatusArray,
                args.action_status_topic,
                self.on_action_status,
                10,
            )

        def on_odom(self, message: Any) -> None:
            linear = message.twist.twist.linear
            angular = message.twist.twist.angular
            writer.write(
                "velocity",
                {
                    "linear_speed": math.hypot(linear.x, linear.y),
                    "angular_speed": abs(angular.z),
                },
            )

        def on_fault(self, message: Any) -> None:
            if message.data in {args.scenario_id, args.scenario, "fault"}:
                writer.write("fault", {"marker": message.data})

        def on_collision(self, message: Any) -> None:
            if message.data:
                writer.write("collision", {"detected": True})

        def on_action_status(self, message: Any) -> None:
            for status in message.status_list:
                name = terminal_names.get(status.status)
                key = (bytes(status.goal_info.goal_id.uuid), int(status.status))
                if name and key not in self.seen_terminal:
                    self.seen_terminal.add(key)
                    writer.write("action_terminal", {"status": name})

    rclpy.init(args=None)
    node = HilRecorderNode()
    writer.write("scenario_start")
    try:
        rclpy.spin(node)
    finally:
        writer.write("scenario_end")
        node.destroy_node()
        rclpy.shutdown()
        writer.close()


if __name__ == "__main__":
    main()


__all__ = ["JsonlEventWriter", "main"]
