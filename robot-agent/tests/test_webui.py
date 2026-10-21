from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

from robot_agent.webui import RobotWebUIServer


class FakeRobotClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def call(self, command: str, **params):
        self.calls.append((command, dict(params)))
        if command == "status":
            return {"success": True, "state": "DRY_RUN", "visual_plan_count": 0}
        if command == "capabilities":
            return {"success": True, "task_catalog": {"tasks": {}}}
        if command == "dataset_recording_status":
            return {
                "success": True,
                "message": "dataset recorder is idle",
                "data": {
                    "configured": True,
                    "enabled": True,
                    "available": True,
                    "active": False,
                    "state": "idle",
                },
            }
        return {
            "success": True,
            "message": "ok",
            "data": {"token_secret": "secret-for-browser"},
        }


class FakeCameraMapping:
    def __init__(self):
        self.applied: list[tuple[dict, str, bool]] = []

    def status(self):
        return {
            "success": True,
            "configured": True,
            "env_file": "/tmp/robot-agent.env",
            "policy_registry_path": "/tmp/policies.yaml",
            "available_devices": [
                {"device": "/dev/video0", "exists": True},
                {"device": "/dev/video2", "exists": True},
                {"device": "/dev/video4", "exists": True},
            ],
            "cameras": {
                "head_cam": {
                    "device": "/dev/video2",
                    "env_device": "/dev/video2",
                    "policy_device": "/dev/video2",
                    "exists": True,
                    "consistent": True,
                },
                "left_cam": {
                    "device": "/dev/video4",
                    "env_device": "/dev/video4",
                    "policy_device": "/dev/video4",
                    "exists": True,
                    "consistent": True,
                },
                "right_cam": {
                    "device": "/dev/video0",
                    "env_device": "/dev/video0",
                    "policy_device": "/dev/video0",
                    "exists": True,
                    "consistent": True,
                },
            },
        }

    def apply(self, mapping: dict, *, operator_confirmation: str, restart_services: bool):
        self.applied.append((dict(mapping), operator_confirmation, restart_services))
        return {
            "success": True,
            "message": "camera mapping saved",
            "mapping": dict(mapping),
            "restart_results": [],
            "status": self.status(),
        }


def _request(port: int, method: str, path: str, body: dict | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    content = response.read()
    connection.close()
    return response.status, content


def test_webui_serves_dashboard_and_forwards_allowlisted_commands(tmp_path: Path):
    fake_client = FakeRobotClient()
    fake_camera_mapping = FakeCameraMapping()
    server = RobotWebUIServer(
        ("127.0.0.1", 0),
        socket_path=tmp_path / "robotd.sock",
        robot_client=fake_client,
        camera_mapping=fake_camera_mapping,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        status, html = _request(port, "GET", "/")
        assert status == 200
        assert b"Robotd Control Desk" in html
        assert b"response-detail" in html
        assert b"recording-form" in html
        assert b"camera-mapping-form" in html
        status, script = _request(port, "GET", "/app.js")
        assert status == 200
        assert b"recordingRequestValidated" in script
        assert b"camera_mapping_apply" in script

        status, payload = _request(port, "GET", "/api/status")
        assert status == 200
        assert json.loads(payload)["state"] == "DRY_RUN"

        status, payload = _request(port, "GET", "/api/recording")
        assert status == 200
        assert json.loads(payload)["data"]["state"] == "idle"
        assert fake_client.calls[-1] == ("dataset_recording_status", {})

        status, payload = _request(port, "GET", "/api/camera-mapping")
        assert status == 200
        assert json.loads(payload)["cameras"]["head_cam"]["device"] == "/dev/video2"

        camera_mapping_params = {
            "mapping": {
                "head_cam": "/dev/video2",
                "left_cam": "/dev/video4",
                "right_cam": "/dev/video0",
            },
            "operator_confirmation": "CONFIRM CAMERA MAPPING",
            "restart_services": False,
        }
        status, _ = _request(
            port,
            "POST",
            "/api/camera-mapping",
            camera_mapping_params,
        )
        assert status == 200
        assert fake_camera_mapping.applied == [
            (
                camera_mapping_params["mapping"],
                "CONFIRM CAMERA MAPPING",
                False,
            )
        ]
        assert fake_client.calls[-1] == ("dataset_recording_status", {})

        status, payload = _request(
            port,
            "POST",
            "/api/command",
            {"command": "arm", "params": {"task_id": "pick_up_bottle"}},
        )
        assert status == 200
        assert json.loads(payload)["data"]["token_secret"] == "secret-for-browser"
        assert fake_client.calls[-1] == ("arm", {"task_id": "pick_up_bottle"})

        recording_params = {
            "dataset_name": "clear_table_v2",
            "task": "clear the table",
            "num_episodes": 5,
            "episode_time_s": 20,
            "reset_time_s": 8,
            "resume": False,
            "operator_confirmation": "CONFIRM RECORD DATASET",
        }
        status, _ = _request(
            port,
            "POST",
            "/api/command",
            {"command": "dataset_recording_start", "params": recording_params},
        )
        assert status == 200
        assert fake_client.calls[-1] == ("dataset_recording_start", recording_params)

        status, _ = _request(
            port,
            "POST",
            "/api/command",
            {"command": "arbitrary_shell", "params": {}},
        )
        assert status == 400
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_webui_reports_same_origin_camera_proxy_urls(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "ROBOT_AGENT_HEAD_CAMERA_SNAPSHOT_URL",
        "http://127.0.0.1:8766/cameras/head_cam/snapshot",
    )
    monkeypatch.setenv(
        "ROBOT_AGENT_LEFT_CAMERA_SNAPSHOT_URL",
        "http://127.0.0.1:8766/cameras/left_cam/snapshot",
    )
    monkeypatch.setenv(
        "ROBOT_AGENT_RIGHT_CAMERA_SNAPSHOT_URL",
        "http://127.0.0.1:8766/cameras/right_cam/snapshot",
    )
    monkeypatch.setenv(
        "ROBOT_AGENT_HEAD_CAMERA_MJPEG_URL",
        "http://127.0.0.1:8766/cameras/head_cam/stream",
    )
    monkeypatch.setenv(
        "ROBOT_AGENT_LEFT_CAMERA_MJPEG_URL",
        "http://127.0.0.1:8766/cameras/left_cam/stream",
    )
    monkeypatch.setenv(
        "ROBOT_AGENT_RIGHT_CAMERA_MJPEG_URL",
        "http://127.0.0.1:8766/cameras/right_cam/stream",
    )
    server = RobotWebUIServer(
        ("127.0.0.1", 0),
        socket_path=tmp_path / "robotd.sock",
        robot_client=FakeRobotClient(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = _request(server.server_address[1], "GET", "/api/cameras")
        response = json.loads(payload)
        assert status == 200
        assert response["enabled"] is True
        assert response["cameras"]["head_cam"]["stream_url"] == "/api/cameras/head_cam/stream"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
