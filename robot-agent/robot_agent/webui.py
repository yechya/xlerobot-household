"""Local browser control surface for the standalone robotd runtime."""

from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .camera_mapping import CameraMappingError, CameraMappingManager
from .camera_rig import RGB_CAMERA_NAMES, mjpeg_urls_from_env, snapshot_urls_from_env
from .client import RobotClient, RobotClientError
from .config import RuntimeConfig

logger = logging.getLogger("robot_agent.webui")
MAX_BODY_BYTES = 1024 * 1024
MAX_CAMERA_SNAPSHOT_BYTES = 12 * 1024 * 1024

_COMMANDS: dict[str, frozenset[str]] = {
    "read": frozenset(
        {"status", "health", "capabilities", "dataset_recording_status"}
    ),
    "control": frozenset(
        {
            "plan",
            "visual_plan_inspect",
            "visual_plan_arm",
            "visual_plan_execute",
            "observe",
            "dry_run",
            "arm",
            "execute",
            "pause",
            "stop",
            "estop",
            "reset",
            "dataset_recording_validate",
            "dataset_recording_start",
            "dataset_recording_stop",
        }
    ),
}
ALLOWED_COMMANDS = _COMMANDS["read"] | _COMMANDS["control"]


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _redact_tokens(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[held in browser memory]"
            if key == "token_secret"
            else _redact_tokens(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_tokens(item) for item in value]
    return value


class RobotWebUIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        socket_path: str | Path,
        timeout: float = 30.0,
        robot_client: RobotClient | None = None,
        camera_mapping: CameraMappingManager | None = None,
    ):
        self.robot_client = robot_client or RobotClient(socket_path, timeout=timeout)
        self.camera_mapping = camera_mapping or CameraMappingManager()
        super().__init__(server_address, RobotWebUIHandler)


class RobotWebUIHandler(BaseHTTPRequestHandler):
    server: RobotWebUIServer

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/status":
            self._call_robotd("status", {}, redact=False)
            return
        if path == "/api/capabilities":
            self._call_robotd("capabilities", {}, redact=False)
            return
        if path == "/api/recording":
            self._call_robotd("dataset_recording_status", {}, redact=False)
            return
        if path == "/api/cameras":
            self._camera_config()
            return
        if path == "/api/camera-mapping":
            self._camera_mapping_status()
            return
        camera_path = self._camera_path(path)
        if camera_path is not None:
            name, kind = camera_path
            self._proxy_camera(name, kind)
            return
        if path in {"/", "/index.html"}:
            self._serve_asset("index.html", "text/html; charset=utf-8")
            return
        if path == "/app.js":
            self._serve_asset("app.js", "text/javascript; charset=utf-8")
            return
        if path == "/styles.css":
            self._serve_asset("styles.css", "text/css; charset=utf-8")
            return
        self._send_json({"success": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/camera-mapping":
            self._apply_camera_mapping()
            return
        if path != "/api/command":
            self._send_json({"success": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
            command = payload.get("command")
            params = payload.get("params", {})
            if not isinstance(command, str) or command not in ALLOWED_COMMANDS:
                raise ValueError("command is not available in the web control surface")
            if not isinstance(params, dict):
                raise ValueError("params must be an object")
        except ValueError as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._call_robotd(command, params, redact=False)

    def _call_robotd(self, command: str, params: dict[str, Any], *, redact: bool) -> None:
        try:
            response = self.server.robot_client.call(command, **params)
        except RobotClientError as exc:
            self._send_json(
                {"success": False, "error": str(exc)},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        self._send_json(_redact_tokens(response) if redact else response, HTTPStatus.OK)

    def _camera_mapping_status(self) -> None:
        self._send_json(self.server.camera_mapping.status(), HTTPStatus.OK)

    def _apply_camera_mapping(self) -> None:
        try:
            payload = self._read_json()
            mapping = payload.get("mapping")
            if not isinstance(mapping, dict):
                raise ValueError("mapping must be an object")
            confirmation = payload.get("operator_confirmation", "")
            if not isinstance(confirmation, str):
                raise ValueError("operator_confirmation must be a string")
            restart_services = payload.get("restart_services", True)
            if not isinstance(restart_services, bool):
                raise ValueError("restart_services must be a boolean")
            recording = self.server.robot_client.call("dataset_recording_status")
            recording_data = recording.get("data", {}) if isinstance(recording, dict) else {}
            if isinstance(recording_data, dict) and recording_data.get("active") is True:
                raise CameraMappingError("dataset recording is active; stop it before remapping cameras")
            response = self.server.camera_mapping.apply(
                mapping,
                operator_confirmation=confirmation,
                restart_services=restart_services,
            )
        except (ValueError, CameraMappingError, RobotClientError) as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(response, HTTPStatus.OK)

    @staticmethod
    def _camera_path(path: str) -> tuple[str, str] | None:
        parts = path.rstrip("/").split("/")
        if len(parts) != 5 or parts[1:3] != ["api", "cameras"]:
            return None
        if parts[3] not in RGB_CAMERA_NAMES or parts[4] not in {"snapshot", "stream"}:
            return None
        return parts[3], parts[4]

    def _camera_config(self) -> None:
        snapshots = snapshot_urls_from_env()
        streams = mjpeg_urls_from_env()
        cameras = {
            name: {
                "configured": bool(snapshots[name] and streams[name]),
                "snapshot_url": f"/api/cameras/{name}/snapshot" if snapshots[name] else "",
                "stream_url": f"/api/cameras/{name}/stream" if streams[name] else "",
            }
            for name in RGB_CAMERA_NAMES
        }
        self._send_json(
            {
                "success": True,
                "enabled": all(item["configured"] for item in cameras.values()),
                "cameras": cameras,
            },
            HTTPStatus.OK,
        )

    def _proxy_camera(self, name: str, kind: str) -> None:
        urls = snapshot_urls_from_env() if kind == "snapshot" else mjpeg_urls_from_env()
        url = urls[name]
        if not url:
            self._send_json(
                {"success": False, "error": f"{name} {kind} URL is not configured"},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        try:
            upstream = urllib.request.urlopen(
                urllib.request.Request(url, headers={"Cache-Control": "no-cache"}),
                timeout=5,
            )
        except urllib.error.HTTPError as exc:
            self._send_json(
                {"success": False, "error": f"camera service returned HTTP {exc.code}"},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._send_json(
                {"success": False, "error": f"camera service unavailable: {exc}"},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        with upstream:
            if kind == "snapshot":
                self._proxy_snapshot(upstream)
            else:
                self._proxy_stream(upstream)

    def _proxy_snapshot(self, upstream: Any) -> None:
        content = upstream.read(MAX_CAMERA_SNAPSHOT_BYTES + 1)
        if len(content) > MAX_CAMERA_SNAPSHOT_BYTES:
            self._send_json(
                {"success": False, "error": "camera snapshot is too large"},
                HTTPStatus.BAD_GATEWAY,
            )
            return
        content_type = upstream.headers.get("Content-Type", "image/jpeg")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _proxy_stream(self, upstream: Any) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            upstream.headers.get(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            ),
        )
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            while True:
                chunk = upstream.read(64 * 1024)
                if not chunk:
                    return
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def _read_json(self) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(length_header)
        except ValueError as exc:
            raise ValueError("Content-Length is invalid") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _serve_asset(self, name: str, content_type: str) -> None:
        try:
            content = files("robot_agent").joinpath("webui", name).read_bytes()
        except (FileNotFoundError, ModuleNotFoundError):
            self._send_json(
                {"success": False, "error": "web UI assets are not installed"},
                HTTPStatus.NOT_FOUND,
            )
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: Any, status: HTTPStatus) -> None:
        content = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format_string: str, *args: Any) -> None:
        logger.info("web %s", format_string % args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local robotd browser control surface")
    parser.add_argument(
        "--host",
        default=os.environ.get("ROBOT_WEB_HOST", "127.0.0.1"),
        help="HTTP bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ROBOT_WEB_PORT", "8765")),
        help="HTTP port (default: 8765)",
    )
    parser.add_argument("--socket", help="robotd Unix socket path")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("ROBOT_WEB_TIMEOUT", "30")),
        help="robotd request timeout in seconds",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("ROBOT_AGENT_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    socket_path = (
        Path(args.socket).expanduser()
        if args.socket
        else RuntimeConfig.from_env().socket_path
    )
    server = RobotWebUIServer(
        (args.host, args.port),
        socket_path=socket_path,
        timeout=args.timeout,
    )
    logger.info("robot web UI listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()


__all__ = ["ALLOWED_COMMANDS", "RobotWebUIHandler", "RobotWebUIServer", "main"]
