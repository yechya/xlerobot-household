#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${ROBOT_AGENT_ROS2_CI_IMAGE:-xlerobot-agent-ros2-humble-ci}"
ros_image="${ROBOT_AGENT_ROS2_BASE_IMAGE:-ros:humble-ros-base}"

docker build \
  --build-arg "ROS_IMAGE=${ros_image}" \
  --file "${repo_root}/tools/ros2-humble-ci.Dockerfile" \
  --tag "${image}" \
  "${repo_root}"

docker run --rm \
  --volume "${repo_root}:/repo:ro" \
  --volume "${repo_root}/ros2_ws/src:/ws/src:ro" \
  "${image}" \
  bash -lc \
  "source /opt/ros/humble/setup.bash \
    && export PYTHONPATH=/repo:\${PYTHONPATH:-} \
    && colcon build --event-handlers console_direct+ --packages-select xlerobot_manipulation_msgs xlerobot_moveit_tasks xlerobot_agent_bringup \
    && source install/setup.bash \
    && { timeout 3s ros2 run xlerobot_moveit_tasks manipulation_server --ros-args \
      -p \"executor_plugins:=['xlerobot::tasks::RigidPickPlace','xlerobot::tasks::VisualServo','xlerobot::tasks::GuardedCartesian','xlerobot::tasks::SustainedPull']\"; \
      status=\$?; test \${status} -eq 0 -o \${status} -eq 124; } \
    && { timeout 3s ros2 run xlerobot_agent_bringup joint_stop_monitor --ros-args \
      -p \"joint_names:=['left_joint','right_joint']\"; \
      status=\$?; test \${status} -eq 0 -o \${status} -eq 124; } \
    && { timeout 3s ros2 run xlerobot_agent_bringup controller_zero_monitor --ros-args \
      --params-file /repo/ros2_ws/src/xlerobot_agent_bringup/config/controller-zero-monitor.example.yaml; \
      status=\$?; test \${status} -eq 0 -o \${status} -eq 124; } \
    && { timeout 3s ros2 run xlerobot_agent_bringup manipulation_hil_process_probe \
      --scenario-id smoke-process-loss --scenario robotd_killed --pid 1; \
      status=\$?; test \${status} -eq 0 -o \${status} -eq 124; } \
    && { timeout 3s ros2 run xlerobot_agent_bringup manipulation_hil_fault_marker \
      --scenario-id smoke-scene-loss --scenario scene_loss; \
      status=\$?; test \${status} -eq 0 -o \${status} -eq 124; } \
    && rm -f /tmp/xlerobot-hil-session-smoke.jsonl \
    && { timeout 3s ros2 launch xlerobot_agent_bringup manipulation_hil_session.launch.py \
      scenario_id:=smoke-scene-loss scenario:=scene_loss \
      output:=/tmp/xlerobot-hil-session-smoke.jsonl; \
      status=\$?; test \${status} -eq 124; } \
    && test -f /tmp/xlerobot-hil-session-smoke.jsonl"
