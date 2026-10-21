from __future__ import annotations

import base64

import pytest

from robot_agent.camera_rig import RGB_CAMERA_NAMES
from robot_agent.hardware.camera import CameraClient
from robot_agent.hardware.vlm import VLMClient, parse_observation_report


def _set_camera_urls(monkeypatch) -> dict[str, str]:
    urls = {
        "head_cam": "http://camera/head",
        "left_cam": "http://camera/left",
        "right_cam": "http://camera/right",
    }
    monkeypatch.setenv("ROBOT_AGENT_HEAD_CAMERA_SNAPSHOT_URL", urls["head_cam"])
    monkeypatch.setenv("ROBOT_AGENT_LEFT_CAMERA_SNAPSHOT_URL", urls["left_cam"])
    monkeypatch.setenv("ROBOT_AGENT_RIGHT_CAMERA_SNAPSHOT_URL", urls["right_cam"])
    return urls


def test_vlm_observation_binds_all_three_rgb_views(monkeypatch):
    urls = _set_camera_urls(monkeypatch)
    images = {name: f"rgb-{name}".encode() for name in RGB_CAMERA_NAMES}
    requested: list[str] = []
    posted: list[dict] = []
    client = VLMClient(api_token="token")

    def read_snapshot(url: str, _timeout: int) -> tuple[bytes, str]:
        requested.append(url)
        name = next(name for name, candidate in urls.items() if candidate == url)
        return images[name], "image/jpeg"

    monkeypatch.setattr(client, "_read_snapshot", read_snapshot)
    monkeypatch.setattr(
        client,
        "_post_chat",
        lambda messages, _timeout: posted.extend(messages) or "three-view response",
    )

    observation = client.observe_evidence("inspect the workspace")

    assert observation.success is True
    assert observation.response == "three-view response"
    assert set(requested) == set(urls.values())
    assert tuple(name for name, _digest in observation.camera_sha256) == RGB_CAMERA_NAMES
    assert len(observation.snapshot_sha256) == 64
    content = posted[0]["content"]
    assert [item["text"] for item in content if item["type"] == "text"] == [
        "inspect the workspace",
        "camera_view=head_cam",
        "camera_view=left_cam",
        "camera_view=right_cam",
    ]
    encoded_images = [
        item["image_url"]["url"].split(",", 1)[1]
        for item in content
        if item["type"] == "image_url"
    ]
    assert [base64.b64decode(value) for value in encoded_images] == [
        images[name] for name in RGB_CAMERA_NAMES
    ]


def test_generic_observe_uses_structured_prompt(monkeypatch):
    _set_camera_urls(monkeypatch)
    client = VLMClient(api_token="token")
    posted: list[dict] = []
    response = (
        '{"schema_version":1,"scene_summary":"桌面可见一个瓶子",'
        '"lighting":"clear","camera_quality":"clear","objects":[],'
        '"hazards":[],"task_candidates":["pick_up_bottle"],"next_action":"proceed"}'
    )

    monkeypatch.setattr(client, "_read_snapshot", lambda url, _timeout: (b"rgb", "image/jpeg"))
    monkeypatch.setattr(
        client,
        "_post_chat",
        lambda messages, _timeout: posted.extend(messages) or response,
    )

    result = client.observe("检查桌面", registered_task_ids=("pick_up_bottle",))

    assert result == response
    prompt = posted[0]["content"][0]["text"]
    assert '"schema_version": 1' in prompt
    assert "不要建议 turn_on_light" in prompt
    assert "pick_up_bottle" in prompt


def test_observation_schema_rejects_free_form_actions():
    with pytest.raises(ValueError, match="observation JSON is invalid"):
        parse_observation_report("建议执行 turn_on_light()")


def test_observation_schema_rejects_unregistered_task():
    response = (
        '{"schema_version":1,"scene_summary":"可见桌面","lighting":"clear",'
        '"camera_quality":"clear","objects":[],"hazards":[],'
        '"task_candidates":["invented_task"],"next_action":"proceed"}'
    )
    with pytest.raises(ValueError, match="unregistered"):
        parse_observation_report(response, registered_task_ids=("fold_simple",))


def test_observation_schema_accepts_complete_json_fence():
    response = """```json
{"schema_version":1,"scene_summary":"可见桌面","lighting":"clear","camera_quality":"clear","objects":[],"hazards":[],"task_candidates":[],"next_action":"none"}
```"""
    report = parse_observation_report(response)
    assert report["next_action"] == "none"


def test_vlm_head_planning_sends_only_head_rgb(monkeypatch):
    urls = _set_camera_urls(monkeypatch)
    monkeypatch.delenv("ROBOT_AGENT_LEFT_CAMERA_SNAPSHOT_URL")
    monkeypatch.delenv("ROBOT_AGENT_RIGHT_CAMERA_SNAPSHOT_URL")
    requested: list[str] = []
    posted: list[dict] = []
    client = VLMClient(api_token="token")

    def read_snapshot(url: str, _timeout: int) -> tuple[bytes, str]:
        requested.append(url)
        assert url == urls["head_cam"]
        return b"head-rgb", "image/jpeg"

    monkeypatch.setattr(client, "_read_snapshot", read_snapshot)
    monkeypatch.setattr(
        client,
        "_post_chat",
        lambda messages, _timeout: posted.extend(messages) or '{"steps":[]}',
    )

    observation = client.generate_head_camera("decompose the household task", timeout=7)

    assert observation.success is True
    assert requested == [urls["head_cam"]]
    assert observation.camera_sha256 == (("head_cam", observation.snapshot_sha256),)
    content = posted[0]["content"]
    assert [item["text"] for item in content if item["type"] == "text"] == [
        "decompose the household task",
        "camera_view=head_cam",
    ]
    encoded_image = next(
        item["image_url"]["url"].split(",", 1)[1]
        for item in content
        if item["type"] == "image_url"
    )
    assert base64.b64decode(encoded_image) == b"head-rgb"


def test_vlm_observation_fails_when_one_rgb_url_is_missing(monkeypatch):
    _set_camera_urls(monkeypatch)
    monkeypatch.delenv("ROBOT_AGENT_RIGHT_CAMERA_SNAPSHOT_URL")
    client = VLMClient(api_token="token")

    observation = client.observe_evidence("inspect")

    assert observation.success is False
    assert "right_cam" in observation.error


def test_vlm_observation_fails_when_one_camera_capture_fails(monkeypatch):
    urls = _set_camera_urls(monkeypatch)
    client = VLMClient(api_token="token")

    def read_snapshot(url: str, _timeout: int) -> tuple[bytes, str]:
        if url == urls["left_cam"]:
            raise RuntimeError("left camera unavailable")
        return b"rgb", "image/jpeg"

    monkeypatch.setattr(client, "_read_snapshot", read_snapshot)
    observation = client.observe_evidence("inspect")

    assert observation.success is False
    assert observation.camera_sha256 == ()
    assert "left camera unavailable" in observation.error


def test_camera_client_reports_the_same_three_rgb_roles(monkeypatch):
    urls = _set_camera_urls(monkeypatch)

    assert CameraClient().get_camera_urls()["snapshots"] == urls


def test_vlm_client_requires_explicit_endpoint_model_and_token(monkeypatch):
    for name in ("VLM_API_URL", "VLM_MODEL", "VLM_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="VLM_API_URL"):
        VLMClient()._post_chat([], 1)

    monkeypatch.setenv("VLM_API_URL", "http://localhost:8000/v1/chat/completions")
    with pytest.raises(RuntimeError, match="VLM_MODEL"):
        VLMClient()._post_chat([], 1)

    monkeypatch.setenv("VLM_MODEL", "local-model")
    with pytest.raises(RuntimeError, match="VLM_API_TOKEN"):
        VLMClient()._post_chat([], 1)


def test_vlm_max_tokens_is_configurable(monkeypatch):
    monkeypatch.setenv("VLM_MAX_TOKENS", "8192")
    assert VLMClient(api_token="token").max_tokens == 8192


@pytest.mark.parametrize("value", ["0", "255", "16385", "not-a-number"])
def test_vlm_max_tokens_rejects_unsafe_values(monkeypatch, value):
    monkeypatch.setenv("VLM_MAX_TOKENS", value)
    with pytest.raises(ValueError, match="VLM_MAX_TOKENS"):
        VLMClient(api_token="token")
