from __future__ import annotations

import math
import os
import re
import signal
import stat
import subprocess
from pathlib import Path
from typing import Any

from robot_agent.episode_replay_runtime import (
    ReplayHandlerResult,
    ReplayInvocation,
    ReplayProducedArtifact,
)

_EXTENSION = re.compile(r"^\.[A-Za-z0-9]{1,10}$")
_PASSTHROUGH_ENV = {"HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "PATH"}


class VideoClipError(ValueError):
    pass


class TerminalVideoClipHandler:
    def __init__(
        self,
        *,
        executable: Path,
        input_kind: str,
        before_terminal_seconds: float,
        output_extension: str,
        video_codec: str,
        timeout_seconds: float,
    ) -> None:
        self.executable = executable
        self.input_kind = input_kind
        self.before_terminal_seconds = before_terminal_seconds
        self.output_extension = output_extension
        self.video_codec = video_codec
        self.timeout_seconds = timeout_seconds

    def clip_window(self, invocation: ReplayInvocation) -> tuple[float, float]:
        artifacts = [
            artifact for artifact in invocation.artifacts if artifact.kind == self.input_kind
        ]
        if len(artifacts) != 1:
            raise VideoClipError("video clip requires exactly one configured input artifact")
        artifact = artifacts[0]
        artifact_duration = artifact.captured_at_end - artifact.captured_at_start
        terminal_offset = min(
            max(invocation.ended_at - artifact.captured_at_start, 0.0),
            artifact_duration,
        )
        start = max(0.0, terminal_offset - self.before_terminal_seconds)
        duration = terminal_offset - start
        if not math.isfinite(duration) or duration < 0.01:
            raise VideoClipError("episode terminal time is outside usable video evidence")
        return start, duration

    def build_command(
        self,
        invocation: ReplayInvocation,
        source: Path,
        destination: Path,
    ) -> tuple[str, ...]:
        start, duration = self.clip_window(invocation)
        command = [
            str(self.executable),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-ss",
            format(start, ".12g"),
            "-t",
            format(duration, ".12g"),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            self.video_codec,
        ]
        if self.video_codec == "libx264":
            command.extend(["-preset", "veryfast", "-crf", "20", "-movflags", "+faststart"])
        command.extend(["-n", str(destination)])
        return tuple(command)

    def execute(self, invocation: ReplayInvocation) -> ReplayHandlerResult:
        artifacts = [
            artifact for artifact in invocation.artifacts if artifact.kind == self.input_kind
        ]
        if len(artifacts) != 1:
            raise VideoClipError("video clip requires exactly one configured input artifact")
        artifact = artifacts[0]
        if not artifact.mime_type.startswith("video/"):
            raise VideoClipError("video clip input artifact must have a video MIME type")
        destination = invocation.output_dir / f"terminal_failure{self.output_extension}"
        command = self.build_command(invocation, artifact.source, destination)
        environment = {
            key: value for key, value in os.environ.items() if key in _PASSTHROUGH_ENV
        }
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
            raise VideoClipError("terminal video clip exceeded timeout") from exc
        _write_bounded(invocation.output_dir / "ffmpeg.stdout.log", stdout)
        _write_bounded(invocation.output_dir / "ffmpeg.stderr.log", stderr)
        if process.returncode != 0:
            destination.unlink(missing_ok=True)
            raise VideoClipError(f"ffmpeg exited with status {process.returncode}")
        if destination.is_symlink() or not destination.is_file() or destination.stat().st_size <= 0:
            raise VideoClipError("ffmpeg did not create a regular non-empty clip")
        destination.chmod(0o600)
        start, duration = self.clip_window(invocation)
        return ReplayHandlerResult(
            metadata={
                "input_artifact_id": artifact.artifact_id,
                "terminal_timestamp": invocation.ended_at,
                "clip_start_offset_seconds": start,
                "clip_duration_seconds": duration,
                "video_codec": self.video_codec,
            },
            produced_artifacts=(
                ReplayProducedArtifact(
                    artifact_id="terminal_failure_clip",
                    kind="failure_video_clip",
                    relative_path=destination.name,
                    mime_type="video/mp4",
                    captured_at_start=artifact.captured_at_start + start,
                    captured_at_end=artifact.captured_at_start + start + duration,
                ),
            ),
        )


def create_handler(
    *,
    options: dict[str, Any],
    artifacts: tuple[Path, ...],
) -> TerminalVideoClipHandler:
    expected = {
        "input_kind",
        "before_terminal_seconds",
        "output_extension",
        "video_codec",
        "timeout_seconds",
    }
    if set(options) != expected:
        raise VideoClipError("video clip handler options are invalid")
    if len(artifacts) != 1:
        raise VideoClipError("video clip handler requires one bound ffmpeg executable")
    executable = artifacts[0]
    try:
        info = executable.lstat()
    except OSError as exc:
        raise VideoClipError("bound ffmpeg executable cannot be read") from exc
    if (
        executable.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or not os.access(executable, os.X_OK)
    ):
        raise VideoClipError("bound ffmpeg executable must be a regular executable file")
    input_kind = _identifier(options["input_kind"], "input_kind")
    extension = options["output_extension"]
    if not isinstance(extension, str) or not _EXTENSION.fullmatch(extension):
        raise VideoClipError("output_extension is invalid")
    codec = options["video_codec"]
    if codec not in {"copy", "libx264"}:
        raise VideoClipError("video_codec must be copy or libx264")
    return TerminalVideoClipHandler(
        executable=executable,
        input_kind=input_kind,
        before_terminal_seconds=_bounded_float(
            options["before_terminal_seconds"],
            "before_terminal_seconds",
            0.1,
            300.0,
        ),
        output_extension=extension.lower(),
        video_codec=codec,
        timeout_seconds=_bounded_float(
            options["timeout_seconds"], "timeout_seconds", 0.1, 3600.0
        ),
    )


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value
    ):
        raise VideoClipError(f"{label} is not a safe identifier")
    return value


def _bounded_float(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VideoClipError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise VideoClipError(f"{label} must be in [{minimum}, {maximum}]")
    return parsed


def _write_bounded(path: Path, content: bytes) -> None:
    if len(content) > 8 * 1024 * 1024:
        content = content[: 8 * 1024 * 1024]
    if not content:
        content = b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


__all__ = ["TerminalVideoClipHandler", "VideoClipError", "create_handler"]
