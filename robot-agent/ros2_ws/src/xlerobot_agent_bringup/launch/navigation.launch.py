from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node

from robot_agent.bringup import BringupManifest


def _launch(context):
    manifest_path = LaunchConfiguration("manifest").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time")
    manifest = BringupManifest.load(manifest_path)
    errors = manifest.validate()
    if errors:
        raise RuntimeError("bringup preflight failed: " + "; ".join(errors))

    urdf = manifest.resolve_file("urdf")
    nav2_params = manifest.resolve_file("nav2_params")
    base_bridge_params = manifest.resolve_file("base_bridge_params")
    map_yaml = manifest.resolve_file("map")
    robot_description = (
        Command(["xacro ", str(urdf)])
        if urdf.suffix == ".xacro"
        else urdf.read_text(encoding="utf-8")
    )
    nav2_launch = (
        Path(get_package_share_directory("nav2_bringup"))
        / "launch"
        / "bringup_launch.py"
    )
    return [
        Node(
            package="xlerobot_agent_bringup",
            executable="base_bridge",
            name="base_bridge",
            output="screen",
            parameters=[str(base_bridge_params)],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description, "use_sim_time": use_sim_time}],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(nav2_launch)),
            launch_arguments={
                "map": str(map_yaml),
                "params_file": str(nav2_params),
                "use_sim_time": use_sim_time,
                "autostart": "true",
                "use_composition": "false",
            }.items(),
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("manifest", description="Calibrated bringup manifest YAML"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            OpaqueFunction(function=_launch),
        ]
    )
