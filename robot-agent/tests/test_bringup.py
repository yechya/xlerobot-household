from __future__ import annotations

import json
from pathlib import Path

import yaml

from robot_agent.acceptance import DeploymentAcceptanceReport, write_acceptance_report
from robot_agent.bringup import BringupManifest
from robot_agent.config import RuntimeConfig
from robot_agent.service import RobotRuntimeServer


def write_valid_assets(tmp_path: Path) -> Path:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="xlerobot"><link name="base_link"/></robot>',
        encoding="utf-8",
    )
    image = tmp_path / "map.pgm"
    image.write_bytes(b"P5\n1 1\n255\n\x00")
    map_yaml = tmp_path / "map.yaml"
    map_yaml.write_text(
        "image: map.pgm\nresolution: 0.05\norigin: [0, 0, 0]\n",
        encoding="utf-8",
    )
    footprint = [[-0.2, -0.15], [0.2, -0.15], [0.2, 0.15], [-0.2, 0.15]]
    nav2 = {
        "controller_server": {
            "ros__parameters": {
                "cmd_vel_topic": "cmd_vel_nav",
                "controller_plugins": ["FollowPath"],
                "FollowPath": {
                    "max_vel_x": 0.15,
                    "max_vel_y": 0.15,
                    "min_vel_y": -0.15,
                    "max_vel_theta": 0.5,
                },
            }
        },
        "local_costmap": {
            "local_costmap": {
                "ros__parameters": {
                    "footprint": footprint,
                    "plugins": ["voxel_layer", "inflation_layer"],
                }
            }
        },
        "global_costmap": {
            "global_costmap": {
                "ros__parameters": {
                    "footprint": footprint,
                    "plugins": ["obstacle_layer", "inflation_layer"],
                }
            }
        },
        "collision_monitor": {
            "ros__parameters": {
                "enabled": True,
                "cmd_vel_out_topic": "/cmd_vel",
                "observation_sources": ["scan"],
                "polygons": ["StopPolygon"],
                "StopPolygon": {"points": footprint, "action_type": "stop"},
            }
        },
    }
    nav2_yaml = tmp_path / "nav2.yaml"
    nav2_yaml.write_text(yaml.safe_dump(nav2), encoding="utf-8")
    bridge = {
        "base_bridge": {
            "ros__parameters": {
                "calibration_approved": True,
                "remote_ip": "127.0.0.1",
                "command_port": 5555,
                "observation_port": 5556,
                "command_topic": "/cmd_vel",
                "odom_topic": "/odom",
                "base_frame": "base_link",
                "odom_frame": "odom",
                "stream_rate_hz": 30.0,
                "max_linear_velocity": 0.15,
                "max_angular_velocity": 0.5,
                "command_timeout_seconds": 0.25,
                "pose_covariance_diagonal": [0.05, 0.05, 999.0, 999.0, 999.0, 0.1],
                "twist_covariance_diagonal": [0.02, 0.02, 999.0, 999.0, 999.0, 0.05],
            }
        }
    }
    bridge_yaml = tmp_path / "base-bridge.yaml"
    bridge_yaml.write_text(yaml.safe_dump(bridge), encoding="utf-8")
    manifest = {
        "version": 1,
        "calibration": {"approved": True, "reviewer": "operator", "date": "2026-07-14"},
        "robot": {
            "holonomic": True,
            "base_frame": "base_link",
            "odom_frame": "odom",
            "map_frame": "map",
            "wheel_radius": 0.05,
            "base_radius": 0.125,
        },
        "files": {
            "urdf": urdf.name,
            "nav2_params": nav2_yaml.name,
            "base_bridge_params": bridge_yaml.name,
            "map": map_yaml.name,
        },
        "topics": {
            "cmd_vel": "/cmd_vel",
            "odom": "/odom",
            "scan": "/scan",
            "imu": "/imu/data",
        },
        "safety": {
            "hardware_watchdog_seconds": 0.5,
            "max_linear_velocity": 0.15,
            "max_angular_velocity": 0.5,
        },
    }
    path = tmp_path / "bringup.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def test_calibrated_holonomic_bringup_passes_preflight(tmp_path: Path):
    manifest = BringupManifest.load(write_valid_assets(tmp_path))
    assert manifest.validate() == []
    assert manifest.resolve_file("urdf") == tmp_path / "robot.urdf"


def test_preflight_rejects_unapproved_and_unsafe_config(tmp_path: Path):
    path = write_valid_assets(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["calibration"]["approved"] = False
    raw["robot"]["holonomic"] = False
    raw["safety"]["max_linear_velocity"] = 0.5
    raw["safety"]["hardware_watchdog_seconds"] = 1.0
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    errors = BringupManifest.load(path).validate()
    assert any("calibration.approved" in error for error in errors)
    assert any("three-omniwheel" in error for error in errors)
    assert any("max_linear_velocity" in error for error in errors)
    assert any("hardware_watchdog" in error for error in errors)


def test_preflight_rejects_missing_collision_and_costmap_geometry(tmp_path: Path):
    path = write_valid_assets(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    nav2_path = tmp_path / raw["files"]["nav2_params"]
    nav2 = yaml.safe_load(nav2_path.read_text(encoding="utf-8"))
    nav2["local_costmap"]["local_costmap"]["ros__parameters"]["footprint"] = []
    nav2["collision_monitor"]["ros__parameters"]["StopPolygon"]["points"] = []
    nav2_path.write_text(yaml.safe_dump(nav2), encoding="utf-8")
    errors = BringupManifest.load(path).validate()
    assert any("local_costmap requires" in error for error in errors)
    assert any("stop polygon" in error for error in errors)


def test_preflight_rejects_unapproved_or_slow_base_bridge(tmp_path: Path):
    path = write_valid_assets(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    bridge_path = tmp_path / raw["files"]["base_bridge_params"]
    bridge = yaml.safe_load(bridge_path.read_text(encoding="utf-8"))
    params = bridge["base_bridge"]["ros__parameters"]
    params["calibration_approved"] = False
    params["stream_rate_hz"] = 10.0
    params["command_timeout_seconds"] = 0.5
    params["remote_ip"] = "192.168.1.10"
    bridge_path.write_text(yaml.safe_dump(bridge), encoding="utf-8")
    errors = BringupManifest.load(path).validate()
    assert any("calibration_approved" in error for error in errors)
    assert any("at least 20Hz" in error for error in errors)
    assert any("below host watchdog" in error for error in errors)
    assert any("loopback" in error for error in errors)


def runtime_config(tmp_path: Path, manifest: Path | None, report: Path | None) -> RuntimeConfig:
    return RuntimeConfig(
        state_dir=tmp_path / "state",
        runtime_dir=tmp_path / "run",
        socket_path=tmp_path / "run" / "robotd.sock",
        token_store_path=tmp_path / "state" / "tokens.json",
        audit_log_path=tmp_path / "state" / "audit.jsonl",
        world_model_path=tmp_path / "state" / "world.yaml",
        token_cache_path=tmp_path / "state" / "token.json",
        require_fresh_observation=True,
        observation_ttl_seconds=30,
        shutdown_grace_seconds=0.2,
        emergency_stop_command=(),
        motion_backend="ros2",
        bringup_manifest_path=manifest,
        hil_report_path=report,
    )


def test_ros2_backend_gate_requires_preflight_and_complete_hil_report(tmp_path: Path):
    missing = RobotRuntimeServer._motion_gate_error(runtime_config(tmp_path, None, None))
    assert "BRINGUP_MANIFEST" in missing

    manifest = write_valid_assets(tmp_path)
    report = tmp_path / "hil-report.json"
    scenarios = [
        {"scenario": name, "passed": True}
        for name in ("heartbeat_loss", "robotd_killed", "nav2_killed", "sensor_loss")
    ]
    report.write_text(
        json.dumps({"passed": True, "scenarios": scenarios}),
        encoding="utf-8",
    )
    assert RobotRuntimeServer._motion_gate_error(
        runtime_config(tmp_path, manifest, report)
    ) is None

    scenarios.pop()
    report.write_text(
        json.dumps({"passed": True, "scenarios": scenarios}),
        encoding="utf-8",
    )
    error = RobotRuntimeServer._motion_gate_error(runtime_config(tmp_path, manifest, report))
    assert "sensor_loss" in error


def test_ros2_backend_gate_requires_matching_deployment_acceptance(tmp_path: Path):
    manifest = write_valid_assets(tmp_path)
    report = tmp_path / "hil-acceptance.json"
    scenarios = [
        {"scenario": name, "passed": True}
        for name in ("heartbeat_loss", "robotd_killed", "nav2_killed", "sensor_loss")
    ]
    write_acceptance_report(
        report,
        DeploymentAcceptanceReport.create(
            report_id="nav2-home-v1",
            report_type="nav2_hil",
            target_fingerprint="a" * 64,
            generated_at=100.0,
            valid_until=4_000_000_000.0,
            payload={"passed": True, "scenarios": scenarios},
        ),
    )
    config = runtime_config(tmp_path, manifest, report)
    assert RobotRuntimeServer._motion_gate_error(
        config,
        target_fingerprint="a" * 64,
    ) is None
    error = RobotRuntimeServer._motion_gate_error(
        config,
        target_fingerprint="b" * 64,
    )
    assert "fingerprint does not match" in error
