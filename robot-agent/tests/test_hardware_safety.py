from __future__ import annotations

from robot_agent.hardware.chassis import ChassisClient
from robot_agent.hardware.lerobot import LeRobotClient


def test_lerobot_is_default_deny_without_robotd_context():
    client = LeRobotClient()
    result = client.execute_task("pick_up_bottle")
    assert result["success"] is False
    assert result["reason_code"] == "missing_token"


def test_chassis_is_default_deny_without_robotd_context(monkeypatch):
    monkeypatch.setenv("ROBOT_AGENT_CHASSIS_COMMAND", "/bin/false")
    client = ChassisClient()
    result = client.move("forward", distance=0.1, speed="slow")
    assert result["success"] is False
    assert result["reason_code"] == "missing_token"


def test_chassis_rejects_unbounded_or_mismatched_motion(monkeypatch):
    monkeypatch.setenv("ROBOT_AGENT_CHASSIS_COMMAND", "/bin/false")
    client = ChassisClient()
    assert "within" in client.move("forward", distance=10.0)["error"]
    assert "angle only" in client.move("rotate_left", distance=0.1)["error"]
    assert "distance only" in client.move("left", angle=20)["error"]


def test_source_tree_has_no_claude_runtime_dependency():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "robot_agent").rglob("*.py")
    )
    assert ".claude" not in source
    assert "mcp.server" not in source
    assert "ROBOT_AGENT_TASK_RUNNER" not in source
    assert "outputs/train" not in (root / "robot_agent" / "skill_flow.py").read_text(
        encoding="utf-8"
    )
    agent_source = (root / "robot_agent" / "agent.py").read_text(encoding="utf-8")
    assert "SmartExecutor(" not in agent_source
    assert "ChassisNavigator(" not in agent_source
    assert "direction_to_command" not in agent_source
    assert "ChassisClient" not in agent_source
    assert "def move(" not in agent_source


def test_product_package_does_not_export_legacy_vlm_motion_api():
    import robot_agent

    assert "VLMParser" not in robot_agent.__all__
    assert "ChassisNavigator" not in robot_agent.__all__
    assert "SmartExecutionResult" not in robot_agent.__all__


def test_legacy_vlm_motion_sources_are_removed():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "robot_agent"
    for filename in (
        "smart_executor.py",
        "vlm_parser.py",
        "chassis_navigator.py",
        "smart_executor_helpers.py",
    ):
        assert not (root / filename).exists()


def test_product_agent_import_does_not_load_legacy_motion_modules():
    import subprocess
    import sys

    script = """
import sys
import robot_agent.agent
legacy = {
    'robot_agent.smart_executor',
    'robot_agent.vlm_parser',
    'robot_agent.chassis_navigator',
    'robot_agent.smart_executor_helpers',
}
loaded = sorted(legacy.intersection(sys.modules))
raise SystemExit('loaded legacy modules: ' + ', '.join(loaded) if loaded else 0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
