from __future__ import annotations

import http.client
import json
import threading

from robot_agent.camera_rig import DEFAULT_RGB_CAMERA_DEVICES, device_paths_from_env
from robot_agent.camera_server import CameraHTTPServer, CameraRig, extract_jpegs


def test_extract_jpegs_handles_chunks_and_noise():
    buffer = bytearray(b"noise\xff\xd8first")
    assert extract_jpegs(buffer) == []
    buffer.extend(b"\xff\xd9between\xff\xd8second\xff\xd9tail")
    assert extract_jpegs(buffer) == [b"\xff\xd8first\xff\xd9", b"\xff\xd8second\xff\xd9"]
    assert bytes(buffer) == b""


def test_device_paths_keep_the_three_confirmed_roles(monkeypatch):
    for variable in (
        "ROBOT_AGENT_HEAD_CAMERA_DEVICE",
        "ROBOT_AGENT_LEFT_CAMERA_DEVICE",
        "ROBOT_AGENT_RIGHT_CAMERA_DEVICE",
    ):
        monkeypatch.delenv(variable, raising=False)
    assert device_paths_from_env() == DEFAULT_RGB_CAMERA_DEVICES


def test_default_device_mapping_matches_current_replugged_roles():
    assert DEFAULT_RGB_CAMERA_DEVICES == {
        "head_cam": "/dev/video2",
        "left_cam": "/dev/video4",
        "right_cam": "/dev/video0",
    }


class _FakeCapture:
    def __init__(self, frame: bytes):
        self.frame = frame

    def status(self):
        return {"device": "fake", "available": True, "sequence": 1, "age_seconds": 0, "error": ""}

    def snapshot(self, timeout=5.0):
        del timeout
        return 1, self.frame, 123.0

    def wait_for_next(self, sequence, timeout=5.0):
        del sequence, timeout
        raise RuntimeError("stream test stops after first frame")


def test_camera_http_server_serves_health_and_snapshot():
    frame = b"\xff\xd8fake\xff\xd9"
    rig = CameraRig({name: _FakeCapture(frame) for name in ("head_cam", "left_cam", "right_cam")})
    server = CameraHTTPServer(("127.0.0.1", 0), rig)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2)
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["cameras"]["head_cam"]["available"] is True
        connection.close()

        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2)
        connection.request("GET", "/cameras/head_cam/snapshot")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/jpeg"
        assert response.read() == frame
        connection.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
