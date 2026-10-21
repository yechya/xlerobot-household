from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest
from xlerobot_replay_ffmpeg.handler import VideoClipError, create_handler

from robot_agent.episode_replay_runtime import ReplayInputArtifact, ReplayInvocation


def _executable(tmp_path: Path, *, exit_code: int = 0) -> Path:
    path = tmp_path / "ffmpeg-reviewed"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "destination = pathlib.Path(sys.argv[-1])\n"
        f"exit_code = {exit_code}\n"
        "if exit_code == 0:\n"
        "    destination.write_bytes(b'encoded failure clip')\n"
        "print(json.dumps(sys.argv[1:]))\n"
        "raise SystemExit(exit_code)\n",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def _options(**changes):
    options = {
        "input_kind": "rgb_video",
        "before_terminal_seconds": 2.0,
        "output_extension": ".mp4",
        "video_codec": "libx264",
        "timeout_seconds": 10.0,
    }
    options.update(changes)
    return options


def _invocation(tmp_path: Path, *, ended_at: float = 12.0) -> ReplayInvocation:
    source = tmp_path / "head.mp4"
    source.write_bytes(b"reviewed video")
    output = tmp_path / "output"
    output.mkdir()
    return ReplayInvocation(
        runtime_id="clip-run-v1",
        replay_id="learned-failures-v1",
        episode_id="episode-one",
        route_id="learned_candidate_review",
        task_id="fold_simple",
        executor="learned_policy",
        success=False,
        failure_stage="task_backend_execute",
        outcome_code="FAILED_STAGE",
        started_at=10.0,
        ended_at=ended_at,
        output_dir=output,
        artifacts=(
            ReplayInputArtifact(
                artifact_id="head_rgb",
                kind="rgb_video",
                source=source,
                mime_type="video/mp4",
                sha256="a" * 64,
                size_bytes=source.stat().st_size,
                captured_at_start=9.0,
                captured_at_end=13.0,
            ),
        ),
    )


def test_video_clip_uses_terminal_timestamp_and_bound_ffmpeg(tmp_path: Path):
    handler = create_handler(options=_options(), artifacts=(_executable(tmp_path),))
    invocation = _invocation(tmp_path)
    result = handler.execute(invocation)
    clip = invocation.output_dir / "terminal_failure.mp4"
    argv = json.loads(
        (invocation.output_dir / "ffmpeg.stdout.log").read_text(encoding="utf-8")
    )

    assert argv[argv.index("-ss") + 1] == "1"
    assert argv[argv.index("-t") + 1] == "2"
    assert argv[argv.index("-c:v") + 1] == "libx264"
    assert argv[-1] == str(clip)
    assert clip.read_bytes() == b"encoded failure clip"
    assert stat.S_IMODE(clip.stat().st_mode) == 0o600
    assert result.metadata["terminal_timestamp"] == 12.0
    assert result.metadata["clip_start_offset_seconds"] == 1.0
    assert result.metadata["clip_duration_seconds"] == 2.0
    assert result.produced_artifacts[0].kind == "failure_video_clip"


def test_video_clip_clamps_window_to_available_evidence(tmp_path: Path):
    handler = create_handler(
        options=_options(before_terminal_seconds=30.0),
        artifacts=(_executable(tmp_path),),
    )
    assert handler.clip_window(_invocation(tmp_path)) == (0.0, 3.0)


def test_video_clip_rejects_terminal_outside_video_and_bad_codec(tmp_path: Path):
    handler = create_handler(options=_options(), artifacts=(_executable(tmp_path),))
    with pytest.raises(VideoClipError, match="outside usable"):
        handler.execute(_invocation(tmp_path, ended_at=8.0))
    with pytest.raises(VideoClipError, match="copy or libx264"):
        create_handler(
            options=_options(video_codec="arbitrary"),
            artifacts=(_executable(tmp_path),),
        )


def test_video_clip_preserves_ffmpeg_failure_diagnostics(tmp_path: Path):
    handler = create_handler(
        options=_options(),
        artifacts=(_executable(tmp_path, exit_code=9),),
    )
    invocation = _invocation(tmp_path)
    with pytest.raises(VideoClipError, match="status 9"):
        handler.execute(invocation)
    assert (invocation.output_dir / "ffmpeg.stdout.log").is_file()
    assert (invocation.output_dir / "ffmpeg.stderr.log").is_file()
    assert not (invocation.output_dir / "terminal_failure.mp4").exists()


def test_ffmpeg_plugin_is_independent_entry_point_distribution():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "plugins/xlerobot_replay_ffmpeg/pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert 'name = "xlerobot-replay-ffmpeg"' in pyproject
    assert '[project.entry-points."xlerobot.replay_handlers"]' in pyproject
    assert "terminal_video_clip_v1" in pyproject
