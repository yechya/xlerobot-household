from __future__ import annotations

import json
import socket
import uuid
from pathlib import Path
from typing import Any


class RobotClientError(RuntimeError):
    pass


class RobotClient:
    def __init__(self, socket_path: str | Path, timeout: float = 30.0):
        self.socket_path = str(socket_path)
        self.timeout = timeout

    def call(self, command: str, **params: Any) -> dict[str, Any]:
        request = {
            "id": f"req-{uuid.uuid4().hex}",
            "command": command,
            "params": params,
        }
        payload = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout)
                connection.connect(self.socket_path)
                connection.sendall(payload)
                chunks: list[bytes] = []
                while True:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
        except (OSError, TimeoutError) as exc:
            raise RobotClientError(f"cannot reach robotd at {self.socket_path}: {exc}") from exc

        raw = b"".join(chunks).split(b"\n", 1)[0]
        if not raw:
            raise RobotClientError("robotd returned an empty response")
        try:
            response = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RobotClientError("robotd returned an invalid response") from exc
        if not isinstance(response, dict):
            raise RobotClientError("robotd response must be an object")
        return response


__all__ = ["RobotClient", "RobotClientError"]

