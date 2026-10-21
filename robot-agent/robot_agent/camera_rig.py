from __future__ import annotations

import os
from collections.abc import Mapping

RGB_CAMERA_NAMES = ("head_cam", "left_cam", "right_cam")
RGB_CAMERA_NAME_SET = frozenset(RGB_CAMERA_NAMES)
RGB_CAMERA_DEVICE_ENV = {
    "head_cam": "ROBOT_AGENT_HEAD_CAMERA_DEVICE",
    "left_cam": "ROBOT_AGENT_LEFT_CAMERA_DEVICE",
    "right_cam": "ROBOT_AGENT_RIGHT_CAMERA_DEVICE",
}
DEFAULT_RGB_CAMERA_DEVICES = {
    "head_cam": "/dev/video2",
    "left_cam": "/dev/video4",
    "right_cam": "/dev/video0",
}
RGB_CAMERA_SNAPSHOT_ENV = {
    "head_cam": "ROBOT_AGENT_HEAD_CAMERA_SNAPSHOT_URL",
    "left_cam": "ROBOT_AGENT_LEFT_CAMERA_SNAPSHOT_URL",
    "right_cam": "ROBOT_AGENT_RIGHT_CAMERA_SNAPSHOT_URL",
}
RGB_CAMERA_MJPEG_ENV = {
    "head_cam": "ROBOT_AGENT_HEAD_CAMERA_MJPEG_URL",
    "left_cam": "ROBOT_AGENT_LEFT_CAMERA_MJPEG_URL",
    "right_cam": "ROBOT_AGENT_RIGHT_CAMERA_MJPEG_URL",
}


class CameraRigError(ValueError):
    pass


def snapshot_urls_from_env() -> dict[str, str]:
    return {
        name: os.environ.get(variable, "").strip()
        for name, variable in RGB_CAMERA_SNAPSHOT_ENV.items()
    }


def mjpeg_urls_from_env() -> dict[str, str]:
    return {
        name: os.environ.get(variable, "").strip()
        for name, variable in RGB_CAMERA_MJPEG_ENV.items()
    }


def device_paths_from_env() -> dict[str, str]:
    """Return the reviewed three-camera RGB device mapping."""
    return {
        name: os.environ.get(variable, DEFAULT_RGB_CAMERA_DEVICES[name]).strip()
        for name, variable in RGB_CAMERA_DEVICE_ENV.items()
    }


def validate_snapshot_urls(urls: Mapping[str, str]) -> dict[str, str]:
    if set(urls) != RGB_CAMERA_NAME_SET:
        raise CameraRigError(
            "camera URLs must contain exactly head_cam, left_cam, and right_cam"
        )
    normalized = {name: str(urls[name]).strip() for name in RGB_CAMERA_NAMES}
    missing = [name for name, url in normalized.items() if not url]
    if missing:
        raise CameraRigError("camera snapshot URLs are missing: " + ", ".join(missing))
    return normalized


__all__ = [
    "CameraRigError",
    "RGB_CAMERA_NAMES",
    "RGB_CAMERA_NAME_SET",
    "RGB_CAMERA_DEVICE_ENV",
    "DEFAULT_RGB_CAMERA_DEVICES",
    "RGB_CAMERA_SNAPSHOT_ENV",
    "RGB_CAMERA_MJPEG_ENV",
    "device_paths_from_env",
    "mjpeg_urls_from_env",
    "snapshot_urls_from_env",
    "validate_snapshot_urls",
]
