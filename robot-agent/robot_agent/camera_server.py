"""V4L2 RGB camera service for snapshots and browser MJPEG previews."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .camera_rig import RGB_CAMERA_NAMES, device_paths_from_env

logger = logging.getLogger("robot_agent.camera_server")
JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
MAX_BUFFER_BYTES = 8 * 1024 * 1024


class CameraCaptureError(RuntimeError):
    """Raised when a camera cannot provide a frame."""


def extract_jpegs(buffer: bytearray) -> list[bytes]:
    """Extract complete JPEG images from an incremental byte buffer."""
    frames: list[bytes] = []
    while True:
        start = buffer.find(JPEG_START)
        if start < 0:
            if buffer.endswith(JPEG_START[:1]):
                buffer[:] = JPEG_START[:1]
            else:
                buffer.clear()
            break
        if start:
            del buffer[:start]
        end = buffer.find(JPEG_END, len(JPEG_START))
        if end < 0:
            break
        frames.append(bytes(buffer[: end + len(JPEG_END)]))
        del buffer[: end + len(JPEG_END)]
    if len(buffer) > MAX_BUFFER_BYTES:
        del buffer[:-MAX_BUFFER_BYTES]
    return frames


@dataclass(frozen=True)
class CameraSpec:
    name: str
    device: str


class CameraCapture:
    """Maintain one latest-frame buffer backed by a restartable ffmpeg process."""

    def __init__(
        self,
        spec: CameraSpec,
        *,
        ffmpeg: str = "ffmpeg",
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        input_format: str = "mjpeg",
        quality: int = 5,
        restart_delay: float = 1.0,
    ) -> None:
        self.spec = spec
        self.ffmpeg = ffmpeg
        self.width = width
        self.height = height
        self.fps = fps
        self.input_format = input_format
        self.quality = quality
        self.restart_delay = restart_delay
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._frame = b""
        self._sequence = 0
        self._captured_at = 0.0
        self._error = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"camera-{self.spec.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._terminate_process()
        if self._thread:
            self._thread.join(timeout=3.0)
        with self._condition:
            self._condition.notify_all()

    def snapshot(self, timeout: float = 5.0) -> tuple[int, bytes, float]:
        sequence, frame, captured_at = self._wait_for_frame(timeout)
        return sequence, frame, captured_at

    def wait_for_next(
        self,
        sequence: int,
        timeout: float = 5.0,
    ) -> tuple[int, bytes, float]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._sequence <= sequence and not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._sequence > sequence:
                return self._sequence, self._frame, self._captured_at
            if self._frame:
                return self._sequence, self._frame, self._captured_at
            raise CameraCaptureError(self._error or f"{self.spec.name} has no frame")

    def status(self) -> dict[str, Any]:
        with self._condition:
            captured_at = self._captured_at
            sequence = self._sequence
            error = self._error
            has_frame = bool(self._frame)
        return {
            "device": self.spec.device,
            "available": has_frame,
            "sequence": sequence,
            "age_seconds": round(max(0.0, time.time() - captured_at), 3)
            if captured_at
            else None,
            "error": error,
        }

    def _wait_for_frame(self, timeout: float) -> tuple[int, bytes, float]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._frame and not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._frame:
                return self._sequence, self._frame, self._captured_at
            raise CameraCaptureError(self._error or f"{self.spec.name} has no frame")

    def _command(self) -> list[str]:
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
        ]
        if self.input_format:
            command.extend(["-input_format", self.input_format])
        command.extend(
            [
                "-video_size",
                f"{self.width}x{self.height}",
                "-framerate",
                str(self.fps),
                "-i",
                self.spec.device,
                "-f",
                "mjpeg",
                "-q:v",
                str(self.quality),
                "pipe:1",
            ]
        )
        return command

    def _capture_loop(self) -> None:
        if not shutil.which(self.ffmpeg) and not Path(self.ffmpeg).exists():
            self._set_error(f"ffmpeg executable not found: {self.ffmpeg}")
            return
        while not self._stop_event.is_set():
            process: subprocess.Popen[bytes] | None = None
            try:
                process = subprocess.Popen(
                    self._command(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
                with self._condition:
                    self._process = process
                    self._error = ""
                self._read_process(process)
            except OSError as exc:
                self._set_error(str(exc))
            finally:
                if process is not None:
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                with self._condition:
                    if self._process is process:
                        self._process = None
                if not self._stop_event.is_set():
                    self._stop_event.wait(self.restart_delay)

    def _read_process(self, process: subprocess.Popen[bytes]) -> None:
        if process.stdout is None:
            raise CameraCaptureError("ffmpeg stdout is unavailable")
        buffer = bytearray()
        while not self._stop_event.is_set():
            chunk = process.stdout.read(64 * 1024)
            if not chunk:
                break
            buffer.extend(chunk)
            for frame in extract_jpegs(buffer):
                with self._condition:
                    self._frame = frame
                    self._sequence += 1
                    self._captured_at = time.time()
                    self._error = ""
                    self._condition.notify_all()
        if not self._stop_event.is_set() and process.poll() not in (None, 0):
            self._set_error(f"ffmpeg exited with status {process.returncode}")

    def _set_error(self, error: str) -> None:
        with self._condition:
            self._error = error
            self._condition.notify_all()

    def _terminate_process(self) -> None:
        with self._condition:
            process = self._process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()


class CameraRig:
    """Own the fixed three-camera RGB capture set."""

    def __init__(self, captures: dict[str, CameraCapture]) -> None:
        if set(captures) != set(RGB_CAMERA_NAMES):
            raise ValueError("camera rig must contain exactly the three RGB camera roles")
        self.captures = captures

    @classmethod
    def from_env(cls) -> "CameraRig":
        devices = device_paths_from_env()
        ffmpeg = os.environ.get("ROBOT_AGENT_CAMERA_FFMPEG", "ffmpeg").strip() or "ffmpeg"
        width = _positive_int_env("ROBOT_AGENT_CAMERA_WIDTH", 640)
        height = _positive_int_env("ROBOT_AGENT_CAMERA_HEIGHT", 480)
        fps = _positive_int_env("ROBOT_AGENT_CAMERA_FPS", 15)
        quality = max(2, min(31, _positive_int_env("ROBOT_AGENT_CAMERA_JPEG_QUALITY", 5)))
        input_format = os.environ.get("ROBOT_AGENT_CAMERA_INPUT_FORMAT", "mjpeg").strip()
        restart_delay = max(0.1, float(os.environ.get("ROBOT_AGENT_CAMERA_RESTART_DELAY", "1")))
        return cls(
            {
                name: CameraCapture(
                    CameraSpec(name, devices[name]),
                    ffmpeg=ffmpeg,
                    width=width,
                    height=height,
                    fps=fps,
                    input_format=input_format,
                    quality=quality,
                    restart_delay=restart_delay,
                )
                for name in RGB_CAMERA_NAMES
            }
        )

    def start(self) -> None:
        for capture in self.captures.values():
            capture.start()

    def stop(self) -> None:
        for capture in self.captures.values():
            capture.stop()

    def capture(self, name: str) -> CameraCapture:
        try:
            return self.captures[name]
        except KeyError as exc:
            raise KeyError(f"unknown camera: {name}") from exc

    def status(self) -> dict[str, Any]:
        return {name: capture.status() for name, capture in self.captures.items()}


class CameraHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], rig: CameraRig):
        self.rig = rig
        super().__init__(address, CameraHTTPHandler)


class CameraHTTPHandler(BaseHTTPRequestHandler):
    server: CameraHTTPServer

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path == "/health":
            self._send_json({"success": True, "cameras": self.server.rig.status()})
            return
        parts = path.split("/")
        if len(parts) == 4 and parts[1] == "cameras" and parts[3] in {"snapshot", "stream"}:
            name = parts[2]
            if name not in RGB_CAMERA_NAMES:
                self._send_json({"success": False, "error": "unknown camera"}, HTTPStatus.NOT_FOUND)
                return
            if parts[3] == "snapshot":
                self._serve_snapshot(name)
            else:
                self._serve_stream(name)
            return
        self._send_json({"success": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

    def _serve_snapshot(self, name: str) -> None:
        try:
            sequence, frame, captured_at = self.server.rig.capture(name).snapshot()
        except CameraCaptureError as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Camera-Frame-Sequence", str(sequence))
        self.send_header("X-Camera-Captured-At", str(captured_at))
        self.end_headers()
        self.wfile.write(frame)

    def _serve_stream(self, name: str) -> None:
        capture = self.server.rig.capture(name)
        try:
            sequence, _frame, _captured_at = capture.snapshot(timeout=5.0)
        except CameraCaptureError as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                sequence, frame, _captured_at = capture.wait_for_next(sequence, timeout=5.0)
                if sequence <= 0:
                    continue
                self.wfile.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                    + frame
                    + b"\r\n"
                )
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, CameraCaptureError):
            return

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format_string: str, *args: Any) -> None:
        logger.info("camera web %s", format_string % args)


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the three RGB V4L2 cameras")
    parser.add_argument("--host", default=os.environ.get("ROBOT_AGENT_CAMERA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ROBOT_AGENT_CAMERA_PORT", "8766")))
    return parser


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("ROBOT_AGENT_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    rig = CameraRig.from_env()
    rig.start()
    server = CameraHTTPServer((args.host, args.port), rig)
    logger.info("robot-camera listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        rig.stop()


__all__ = [
    "CameraCapture",
    "CameraCaptureError",
    "CameraHTTPServer",
    "CameraRig",
    "CameraSpec",
    "extract_jpegs",
    "main",
]


if __name__ == "__main__":
    main()
