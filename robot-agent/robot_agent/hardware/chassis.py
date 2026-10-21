"""Chassis Client - 底盘控制客户端"""

import os
import subprocess
import logging
import math
import shlex
from typing import Optional

from .authorization import validate_actuation_authorized

logger = logging.getLogger("hardware.chassis")

class ChassisClient:
    """底盘控制客户端"""

    # 方向映射
    DIRECTIONS = ["forward", "backward", "left", "right", "rotate_left", "rotate_right"]

    # 速度映射
    SPEEDS = ["slow", "medium", "fast"]

    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self.command = tuple(shlex.split(os.environ.get("ROBOT_AGENT_CHASSIS_COMMAND", "")))
        self.max_distance = float(os.environ.get("ROBOT_AGENT_MAX_CHASSIS_DISTANCE", "0.30"))
        self.max_angle = float(os.environ.get("ROBOT_AGENT_MAX_CHASSIS_ANGLE", "45"))

    def move(
        self,
        direction: str,
        distance: Optional[float] = None,
        angle: Optional[float] = None,
        speed: str = "medium",
        observe: Optional[str] = None
    ) -> dict:
        """
        控制底盘移动。

        Args:
            direction: 移动方向
            distance: 移动距离（米）
            angle: 旋转角度（度）
            speed: 速度等级
            observe: VLM 监控指令（迭代移动）

        Returns:
            移动结果字典
        """
        if direction not in self.DIRECTIONS:
            return {
                "success": False,
                "error": f"无效方向: {direction}",
                "valid_directions": self.DIRECTIONS
            }

        if speed not in self.SPEEDS:
            return {
                "success": False,
                "error": f"无效速度: {speed}",
                "valid_speeds": self.SPEEDS
            }

        if direction.startswith("rotate_"):
            if angle is None or distance is not None:
                return {"success": False, "error": "rotation requires angle only", "direction": direction}
            if not math.isfinite(angle) or not 0 < angle <= self.max_angle:
                return {
                    "success": False,
                    "error": f"angle must be within (0, {self.max_angle}] degrees",
                    "direction": direction,
                }
        else:
            if distance is None or angle is not None:
                return {"success": False, "error": "linear movement requires distance only", "direction": direction}
            if not math.isfinite(distance) or not 0 < distance <= self.max_distance:
                return {
                    "success": False,
                    "error": f"distance must be within (0, {self.max_distance}] meters",
                    "direction": direction,
                }

        authorization_refusal = validate_actuation_authorized("chassis.move")
        if authorization_refusal is not None:
            logger.warning(
                "[ChassisClient] refused unauthorized actuation: %s",
                authorization_refusal.reason_code,
            )
            return authorization_refusal.to_result(direction=direction, speed=speed)

        if not self.command:
            return {
                "success": False,
                "error": "ROBOT_AGENT_CHASSIS_COMMAND is not configured",
                "direction": direction,
            }

        logger.info(f"[ChassisClient] move: {direction}, dist={distance}, angle={angle}, speed={speed}")

        try:
            cmd = [*self.command, direction]

            if distance is not None:
                cmd.append(str(distance))
            elif angle is not None:
                cmd.append(str(angle))
            else:
                # 无距离/角度参数，拒绝执行
                logger.warning("[ChassisClient] 未指定距离或角度，拒绝移动")
                return {
                    "success": False,
                    "error": "必须指定 distance 或 angle 参数",
                    "direction": direction
                }

            cmd.append(speed)

            if observe:
                cmd.extend(["--observe", observe])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr[-500:] if result.stderr else "",
                "direction": direction
            }

        except subprocess.TimeoutExpired:
            logger.error("[ChassisClient] 底盘移动超时")
            return {
                "success": False,
                "error": f"移动超时（{self.timeout}秒）",
                "direction": direction
            }
        except Exception as e:
            logger.error(f"[ChassisClient] 底盘移动异常: {e}")
            return {
                "success": False,
                "error": str(e),
                "direction": direction
            }
