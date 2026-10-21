from __future__ import annotations

import argparse
import ipaddress
import json
import math
import xml.etree.ElementTree as element_tree
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BringupManifest:
    path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "BringupManifest":
        manifest_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("bringup manifest must be a YAML object")
        return cls(manifest_path, raw)

    def resolve_file(self, key: str) -> Path:
        files = self.data.get("files", {})
        if not isinstance(files, dict) or not isinstance(files.get(key), str):
            raise ValueError(f"files.{key} must be a path")
        configured = Path(files[key]).expanduser()
        if not configured.is_absolute():
            configured = self.path.parent / configured
        return configured.resolve()

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.data.get("version") != 1:
            errors.append("version must be 1")
        self._validate_calibration(errors)
        self._validate_robot(errors)
        self._validate_topics(errors)
        self._validate_safety(errors)
        self._validate_files(errors)
        return errors

    def _validate_calibration(self, errors: list[str]) -> None:
        calibration = self.data.get("calibration", {})
        if not isinstance(calibration, dict) or calibration.get("approved") is not True:
            errors.append("calibration.approved must be true after physical calibration review")
            return
        for key in ("reviewer", "date"):
            if not isinstance(calibration.get(key), str) or not calibration[key].strip():
                errors.append(f"calibration.{key} must be recorded")

    def _validate_robot(self, errors: list[str]) -> None:
        robot = self.data.get("robot", {})
        if not isinstance(robot, dict):
            errors.append("robot must be an object")
            return
        if robot.get("holonomic") is not True:
            errors.append("robot.holonomic must be true for the XLeRobot three-omniwheel base")
        for key in ("base_frame", "odom_frame", "map_frame"):
            if not isinstance(robot.get(key), str) or not robot[key].strip():
                errors.append(f"robot.{key} must be a non-empty frame name")
        for key in ("wheel_radius", "base_radius"):
            if not _is_positive_number(robot.get(key)):
                errors.append(f"robot.{key} must be a calibrated positive number")

    def _validate_topics(self, errors: list[str]) -> None:
        topics = self.data.get("topics", {})
        if not isinstance(topics, dict):
            errors.append("topics must be an object")
            return
        for key in ("cmd_vel", "odom", "scan", "imu"):
            value = topics.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"topics.{key} must be a non-empty ROS topic")

    def _validate_safety(self, errors: list[str]) -> None:
        safety = self.data.get("safety", {})
        if not isinstance(safety, dict):
            errors.append("safety must be an object")
            return
        watchdog = safety.get("hardware_watchdog_seconds")
        if not _is_positive_number(watchdog) or float(watchdog) > 0.5:
            errors.append("safety.hardware_watchdog_seconds must be in (0, 0.5]")
        linear = safety.get("max_linear_velocity")
        angular = safety.get("max_angular_velocity")
        if not _is_positive_number(linear) or float(linear) > 0.2:
            errors.append("commissioning max_linear_velocity must be in (0, 0.2] m/s")
        if not _is_positive_number(angular) or float(angular) > 0.6:
            errors.append("commissioning max_angular_velocity must be in (0, 0.6] rad/s")

    def _validate_files(self, errors: list[str]) -> None:
        resolved: dict[str, Path] = {}
        for key in ("urdf", "nav2_params", "base_bridge_params", "map"):
            try:
                path = self.resolve_file(key)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            resolved[key] = path
            if not path.is_file():
                errors.append(f"files.{key} does not exist: {path}")
        if "urdf" in resolved and resolved["urdf"].is_file():
            self._validate_urdf(resolved["urdf"], errors)
        if "map" in resolved and resolved["map"].is_file():
            self._validate_map(resolved["map"], errors)
        if "nav2_params" in resolved and resolved["nav2_params"].is_file():
            self._validate_nav2(resolved["nav2_params"], errors)
        if "base_bridge_params" in resolved and resolved["base_bridge_params"].is_file():
            self._validate_base_bridge(resolved["base_bridge_params"], errors)

    def _validate_urdf(self, path: Path, errors: list[str]) -> None:
        if path.suffix == ".xacro":
            if "<robot" not in path.read_text(encoding="utf-8"):
                errors.append("xacro file does not contain a robot element")
            return
        try:
            root = element_tree.parse(path).getroot()
        except element_tree.ParseError as exc:
            errors.append(f"URDF is invalid XML: {exc}")
            return
        if root.tag != "robot":
            errors.append("URDF root element must be robot")
            return
        base_frame = self.data.get("robot", {}).get("base_frame")
        links = {link.get("name") for link in root.findall("link")}
        if base_frame not in links:
            errors.append(f"URDF does not contain configured base frame {base_frame!r}")

    @staticmethod
    def _validate_map(path: Path, errors: list[str]) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict) or not isinstance(raw.get("image"), str):
            errors.append("map YAML must contain an image path")
            return
        image = Path(raw["image"])
        if not image.is_absolute():
            image = path.parent / image
        if not image.is_file():
            errors.append(f"map image does not exist: {image.resolve()}")

    def _validate_nav2(self, path: Path, errors: list[str]) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            errors.append("Nav2 parameters must be a YAML object")
            return
        controller = _ros_parameters(raw, "controller_server", errors)
        local_costmap = _nested_ros_parameters(raw, "local_costmap", errors)
        global_costmap = _nested_ros_parameters(raw, "global_costmap", errors)
        collision_monitor = _ros_parameters(raw, "collision_monitor", errors)
        if controller:
            self._validate_controller(controller, errors)
        if local_costmap:
            self._validate_costmap("local_costmap", local_costmap, errors)
        if global_costmap:
            self._validate_costmap("global_costmap", global_costmap, errors)
        if collision_monitor:
            sources = collision_monitor.get("observation_sources")
            if not isinstance(sources, list) or not sources:
                errors.append("collision_monitor must define observation_sources")
            if collision_monitor.get("enabled", True) is not True:
                errors.append("collision_monitor must be enabled")
            polygons = collision_monitor.get("polygons")
            if not isinstance(polygons, list) or not polygons:
                errors.append("collision_monitor must define a stop polygon")
            else:
                polygon = collision_monitor.get(polygons[0], {})
                points = polygon.get("points") if isinstance(polygon, dict) else None
                if isinstance(points, str):
                    points = yaml.safe_load(points)
                if not isinstance(points, list) or len(points) < 3:
                    errors.append("collision_monitor stop polygon must be calibrated")
            topics = self.data.get("topics", {})
            if collision_monitor.get("cmd_vel_out_topic") != topics.get("cmd_vel"):
                errors.append("collision_monitor must own the configured cmd_vel output topic")

    def _validate_base_bridge(self, path: Path, errors: list[str]) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        params = _ros_parameters(raw, "base_bridge", errors)
        if not params:
            return
        if params.get("calibration_approved") is not True:
            errors.append("base_bridge calibration_approved must be true")
        topics = self.data.get("topics", {})
        robot = self.data.get("robot", {})
        safety = self.data.get("safety", {})
        expected = {
            "command_topic": topics.get("cmd_vel"),
            "odom_topic": topics.get("odom"),
            "base_frame": robot.get("base_frame"),
            "odom_frame": robot.get("odom_frame"),
        }
        for key, value in expected.items():
            if params.get(key) != value:
                errors.append(f"base_bridge {key} must match bringup manifest")
        if not isinstance(params.get("remote_ip"), str) or not params["remote_ip"].strip():
            errors.append("base_bridge remote_ip must be configured")
        elif not _is_loopback_host(params["remote_ip"]):
            errors.append("base_bridge remote_ip must remain loopback until ZMQ is authenticated")
        for key in ("command_port", "observation_port"):
            value = params.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 65536:
                errors.append(f"base_bridge {key} must be a valid TCP port")
        stream_rate = params.get("stream_rate_hz")
        if not _is_positive_number(stream_rate) or float(stream_rate) < 20.0:
            errors.append("base_bridge stream_rate_hz must be at least 20Hz")
        command_timeout = params.get("command_timeout_seconds")
        watchdog = safety.get("hardware_watchdog_seconds")
        if (
            not _is_positive_number(command_timeout)
            or float(command_timeout) > 0.25
            or (_is_positive_number(watchdog) and float(command_timeout) >= float(watchdog))
        ):
            errors.append("base_bridge command timeout must be <=250ms and below host watchdog")
        limits = {
            "max_linear_velocity": safety.get("max_linear_velocity"),
            "max_angular_velocity": safety.get("max_angular_velocity"),
        }
        for key, limit in limits.items():
            value = params.get(key)
            if not _is_positive_number(value):
                errors.append(f"base_bridge {key} must be positive")
            elif _is_positive_number(limit) and float(value) > float(limit):
                errors.append(f"base_bridge {key} exceeds manifest safety limit")
        for key in ("pose_covariance_diagonal", "twist_covariance_diagonal"):
            value = params.get(key)
            if not isinstance(value, list) or len(value) != 6 or not all(
                _is_non_negative_number(item) for item in value
            ):
                errors.append(f"base_bridge {key} must contain six non-negative values")

    def _validate_controller(self, controller: dict[str, Any], errors: list[str]) -> None:
        plugins = controller.get("controller_plugins")
        if not isinstance(plugins, list) or not plugins:
            errors.append("controller_server must define controller_plugins")
            return
        if controller.get("cmd_vel_topic") != "cmd_vel_nav":
            errors.append("controller_server must publish to cmd_vel_nav behind collision_monitor")
        plugin = controller.get(plugins[0], {})
        if not isinstance(plugin, dict):
            errors.append("configured controller plugin parameters are missing")
            return
        safety = self.data.get("safety", {})
        limits = {
            "max_vel_x": safety.get("max_linear_velocity"),
            "max_vel_y": safety.get("max_linear_velocity"),
            "max_vel_theta": safety.get("max_angular_velocity"),
        }
        for key, limit in limits.items():
            value = plugin.get(key)
            if not _is_positive_number(value):
                errors.append(f"controller {key} must be positive")
            elif _is_positive_number(limit) and float(value) > float(limit):
                errors.append(f"controller {key} exceeds manifest safety limit")
        if not _is_negative_number(plugin.get("min_vel_y")):
            errors.append("holonomic controller min_vel_y must be negative")

    @staticmethod
    def _validate_costmap(
        name: str,
        costmap: dict[str, Any],
        errors: list[str],
    ) -> None:
        footprint = costmap.get("footprint")
        if isinstance(footprint, str):
            try:
                footprint = yaml.safe_load(footprint)
            except yaml.YAMLError:
                footprint = None
        if not isinstance(footprint, list) or len(footprint) < 3:
            errors.append(f"{name} requires a calibrated polygon footprint")
        plugins = costmap.get("plugins")
        if not isinstance(plugins, list):
            errors.append(f"{name} must define plugins")
            return
        if not any("obstacle" in name or "voxel" in name for name in plugins):
            errors.append(f"{name} requires an obstacle or voxel layer")
        if not any("inflation" in name for name in plugins):
            errors.append(f"{name} requires an inflation layer")


def _ros_parameters(raw: dict[str, Any], node: str, errors: list[str]) -> dict[str, Any]:
    value = raw.get(node, {})
    params = value.get("ros__parameters", {}) if isinstance(value, dict) else {}
    if not isinstance(params, dict) or not params:
        errors.append(f"Nav2 parameters are missing {node}.ros__parameters")
        return {}
    return params


def _nested_ros_parameters(
    raw: dict[str, Any],
    node: str,
    errors: list[str],
) -> dict[str, Any]:
    outer = raw.get(node, {})
    inner = outer.get(node, {}) if isinstance(outer, dict) else {}
    params = inner.get("ros__parameters", {}) if isinstance(inner, dict) else {}
    if not isinstance(params, dict) or not params:
        errors.append(f"Nav2 parameters are missing {node}.{node}.ros__parameters")
        return {}
    return params


def _is_positive_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value > 0
    )


def _is_negative_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value < 0
    )


def _is_non_negative_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and value >= 0
    )


def _is_loopback_host(value: str) -> bool:
    if value.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value.strip()).is_loopback
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate calibrated XLeRobot ROS 2 bringup")
    parser.add_argument("manifest")
    args = parser.parse_args()
    try:
        manifest = BringupManifest.load(args.manifest)
        errors = manifest.validate()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors = [str(exc)]
    payload = {"valid": not errors, "errors": errors}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()


__all__ = ["BringupManifest", "main"]
