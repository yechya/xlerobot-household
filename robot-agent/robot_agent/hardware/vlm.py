"""OpenAI-compatible VLM client used by the standalone runtime."""

import base64
import hashlib
import json
import logging
import math
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..camera_rig import (
    RGB_CAMERA_NAMES,
    snapshot_urls_from_env,
    validate_snapshot_urls,
)

logger = logging.getLogger("hardware.vlm")
OBSERVATION_SCHEMA_VERSION = 1
_OBSERVATION_LIGHTING = frozenset({"clear", "dim", "unknown"})
_OBSERVATION_CAMERA_QUALITY = frozenset(
    {"clear", "dim", "blurred", "blocked", "mixed", "unknown"}
)
_OBSERVATION_NEXT_ACTION = frozenset(
    {"proceed", "improve_view", "operator_help", "none"}
)
_OBSERVATION_POSITIONS = frozenset({"left", "center", "right", "unknown"})
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


@dataclass(frozen=True)
class VLMObservation:
    response: str
    snapshot_sha256: str
    captured_at: float
    model: str
    error: str = ""
    camera_sha256: tuple[tuple[str, str], ...] = ()

    @property
    def success(self) -> bool:
        return not self.error and bool(self.snapshot_sha256)


def build_observation_prompt(
    instruction: str,
    registered_task_ids: tuple[str, ...] = (),
) -> str:
    """Build the strict schema prompt for generic full-RGB observations."""
    task_ids = json.dumps(sorted(set(registered_task_ids)), ensure_ascii=False)
    return f"""你是居家服务机器人视觉观察器。你会看到 head_cam、left_cam、right_cam 三张同步的普通 RGB 图像。

观察指令：{instruction.strip()}
已注册任务 ID（只能逐字选择；不能创造新任务）：{task_ids}

只返回一个严格 JSON 对象，不要 Markdown、解释文字、函数名、API 调用、坐标、距离、动作脚本或安全保证：
{{
  "schema_version": 1,
  "scene_summary": "不超过 160 字的可见场景摘要",
  "lighting": "clear|dim|unknown",
  "camera_quality": "clear|dim|blurred|blocked|mixed|unknown",
  "objects": [
    {{"label":"物品名","count":1,"position":"left|center|right|unknown","state":"可见状态","confidence":0.0}}
  ],
  "hazards": ["仅填写图像明确可见的风险；没有则为空数组"],
  "task_candidates": ["仅填写上面已注册且图像/指令支持的 task_id"],
  "next_action": "proceed|improve_view|operator_help|none"
}}

规则：
1. 所有字段必须存在；没有证据就用 unknown、空数组或 confidence=0。
2. 普通 RGB 不能证明深度、精确距离、力/接触、可达性或人员安全；不要猜测。
3. 画面昏暗、模糊或被遮挡时，next_action 必须为 improve_view 或 operator_help，task_candidates 必须为空。
4. 不要建议 turn_on_light、move_closer、tilt_head、pick_up_trash 等函数或未注册动作；只返回结构化事实。
5. task_candidates 最多 8 个，objects 最多 32 个，hazards 最多 8 个。"""


def parse_observation_report(
    response: str,
    *,
    registered_task_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    """Validate the generic observation schema before planning consumes it."""
    try:
        payload = _parse_json_object(response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"observation JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("observation response must be a JSON object")
    expected = {
        "schema_version",
        "scene_summary",
        "lighting",
        "camera_quality",
        "objects",
        "hazards",
        "task_candidates",
        "next_action",
    }
    if set(payload) != expected:
        raise ValueError("observation object has missing or unexpected fields")
    if payload["schema_version"] != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("observation schema_version must be 1")
    scene_summary = payload["scene_summary"]
    if not isinstance(scene_summary, str) or not 1 <= len(scene_summary.strip()) <= 512:
        raise ValueError("scene_summary must be 1..512 characters")
    lighting = payload["lighting"]
    if lighting not in _OBSERVATION_LIGHTING:
        raise ValueError("lighting is invalid")
    camera_quality = payload["camera_quality"]
    if camera_quality not in _OBSERVATION_CAMERA_QUALITY:
        raise ValueError("camera_quality is invalid")
    next_action = payload["next_action"]
    if next_action not in _OBSERVATION_NEXT_ACTION:
        raise ValueError("next_action is invalid")
    objects = payload["objects"]
    if not isinstance(objects, list) or len(objects) > 32:
        raise ValueError("objects must contain 0..32 items")
    normalized_objects: list[dict[str, object]] = []
    for item in objects:
        if not isinstance(item, dict) or set(item) != {
            "label",
            "count",
            "position",
            "state",
            "confidence",
        }:
            raise ValueError("object has missing or unexpected fields")
        label = item["label"]
        count = item["count"]
        position = item["position"]
        state = item["state"]
        confidence = item["confidence"]
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 80:
            raise ValueError("object label is invalid")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 32:
            raise ValueError("object count must be an integer in [1, 32]")
        if position not in _OBSERVATION_POSITIONS:
            raise ValueError("object position is invalid")
        if not isinstance(state, str) or not 1 <= len(state.strip()) <= 120:
            raise ValueError("object state is invalid")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError("object confidence must be in [0, 1]")
        normalized_objects.append(
            {
                "label": label.strip(),
                "count": count,
                "position": position,
                "state": state.strip(),
                "confidence": float(confidence),
            }
        )
    hazards = payload["hazards"]
    if (
        not isinstance(hazards, list)
        or len(hazards) > 8
        or not all(isinstance(value, str) and 0 < len(value.strip()) <= 120 for value in hazards)
    ):
        raise ValueError("hazards must contain 0..8 concise strings")
    candidates = payload["task_candidates"]
    if not isinstance(candidates, list) or len(candidates) > 8:
        raise ValueError("task_candidates must contain 0..8 items")
    if not all(isinstance(value, str) and _SAFE_TASK_ID.fullmatch(value) for value in candidates):
        raise ValueError("task_candidates contains an invalid task ID")
    if len(candidates) != len(set(candidates)):
        raise ValueError("task_candidates must not contain duplicates")
    allowed = set(registered_task_ids)
    unknown = sorted(set(candidates) - allowed) if allowed else []
    if unknown:
        raise ValueError(f"task_candidates contains unregistered IDs: {unknown}")
    if next_action in {"improve_view", "operator_help"} and candidates:
        raise ValueError("blocked observation cannot contain task_candidates")
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "scene_summary": scene_summary.strip(),
        "lighting": lighting,
        "camera_quality": camera_quality,
        "objects": normalized_objects,
        "hazards": [value.strip() for value in hazards],
        "task_candidates": list(candidates),
        "next_action": next_action,
    }


def _parse_json_object(response: str) -> object:
    """Parse plain JSON or a complete JSON code fence, without accepting prose."""
    text = response.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            raise first_error
        return json.loads(match.group(1))

# 方向性描述的 prompt 模板
DIRECTIONAL_OBSERVATION_PROMPT = """请描述画面中主要物品相对于机器人的位置关系。

请回答以下问题（用一句话描述）：
1. 物品在机器人的哪个方向？（左侧/右侧/正前方/偏左/偏右）
2. 物品距离机器人大致多远？（很近/中等距离/较远/很远）
3. 物品是否在机器人的可及范围内？（在范围内/范围外/不确定）

注意：
- 不需要估计具体米数，只需要定性描述
- 如果视野内没有物品，请说明"未检测到物品"
"""


class VLMClient:
    """VLM 客户端，封装 VLM API 调用"""

    def __init__(self, api_url: str | None = None, api_token: str | None = None):
        self.api_url = (api_url or os.environ.get("VLM_API_URL", "")).strip()
        self.api_token = api_token or os.environ.get("VLM_API_TOKEN", "")
        self.model = os.environ.get("VLM_MODEL", "").strip()
        self.timeout = int(os.environ.get("VLM_TIMEOUT", "90"))
        try:
            self.max_tokens = int(os.environ.get("VLM_MAX_TOKENS", "2048"))
        except ValueError as exc:
            raise ValueError("VLM_MAX_TOKENS must be an integer") from exc
        if not 256 <= self.max_tokens <= 16384:
            raise ValueError("VLM_MAX_TOKENS must be in [256, 16384]")
        self.camera_snapshot_urls = snapshot_urls_from_env()

    def _post_chat(self, messages: list[dict], timeout: int) -> str:
        if not self.api_url:
            raise RuntimeError("VLM_API_URL is not configured")
        if not self.model:
            raise RuntimeError("VLM_MODEL is not configured")
        if not self.api_token:
            raise RuntimeError("VLM_API_TOKEN is not configured")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result["choices"][0]["message"]["content"])

    @staticmethod
    def _read_snapshot(url: str, timeout: int) -> tuple[bytes, str]:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            image = response.read()
            content_type = response.headers.get_content_type() or "image/jpeg"
        if not image:
            raise RuntimeError("camera snapshot is empty")
        return image, content_type

    def _capture_rgb_rig(
        self,
        camera_urls: dict[str, str] | None = None,
    ) -> tuple[tuple[str, bytes, str], ...]:
        urls = validate_snapshot_urls(camera_urls or self.camera_snapshot_urls)
        with ThreadPoolExecutor(max_workers=len(RGB_CAMERA_NAMES)) as executor:
            captures = tuple(
                executor.map(
                    lambda name: (name, *self._read_snapshot(urls[name], self.timeout)),
                    RGB_CAMERA_NAMES,
                )
            )
        return captures

    @staticmethod
    def _rig_digest(captures: tuple[tuple[str, bytes, str], ...]) -> str:
        digest = hashlib.sha256()
        for name, image, _content_type in captures:
            digest.update(name.encode("ascii"))
            digest.update(b"\0")
            digest.update(image)
        return digest.hexdigest()

    def observe(
        self,
        instruction: str,
        camera_urls: dict[str, str] | None = None,
        *,
        registered_task_ids: tuple[str, ...] = (),
    ) -> str:
        """
        执行视觉观察。

        Args:
            instruction: 观察指令
            camera_urls: 三路 RGB 快照 URL（可选，默认读取环境配置）

        Returns:
            VLM 的文本响应
        """
        prompt = build_observation_prompt(instruction, registered_task_ids)
        return self.observe_evidence(prompt, camera_urls).response

    def observe_evidence(
        self,
        instruction: str,
        camera_urls: dict[str, str] | None = None,
    ) -> VLMObservation:
        """Capture the three-camera RGB rig and return bound visual evidence."""
        logger.info("[VLMClient] evidence observation request")

        try:
            captures = self._capture_rgb_rig(camera_urls)
            content: list[dict] = [{"type": "text", "text": instruction}]
            camera_digests: list[tuple[str, str]] = []
            for name, image, content_type in captures:
                image_data = base64.b64encode(image).decode("ascii")
                camera_digests.append((name, hashlib.sha256(image).hexdigest()))
                content.extend(
                    [
                        {"type": "text", "text": f"camera_view={name}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{image_data}",
                            },
                        },
                    ]
                )
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            response = self._post_chat(messages, self.timeout)
            return VLMObservation(
                response=response,
                snapshot_sha256=self._rig_digest(captures),
                captured_at=time.time(),
                model=self.model,
                camera_sha256=tuple(camera_digests),
            )
        except TimeoutError:
            logger.error("[VLMClient] observation timed out")
            return VLMObservation(
                response="观察超时",
                snapshot_sha256="",
                captured_at=time.time(),
                model=self.model,
                error="observation timed out",
            )
        except Exception as exc:
            logger.error("[VLMClient] observation failed: %s", exc)
            return VLMObservation(
                response=f"观察失败: {exc}",
                snapshot_sha256="",
                captured_at=time.time(),
                model=self.model,
                error=str(exc),
            )

    def generate_head_camera(
        self,
        instruction: str,
        *,
        timeout: int | None = None,
    ) -> VLMObservation:
        """Generate structured text from the verified head-camera snapshot only."""
        captured_at = time.time()
        try:
            head_url = self.camera_snapshot_urls.get("head_cam", "").strip()
            if not head_url:
                raise RuntimeError("head camera snapshot URL is not configured")
            image, content_type = self._read_snapshot(
                head_url,
                self.timeout if timeout is None else timeout,
            )
            image_digest = hashlib.sha256(image).hexdigest()
            image_data = base64.b64encode(image).decode("ascii")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "text", "text": "camera_view=head_cam"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{image_data}",
                            },
                        },
                    ],
                }
            ]
            response = self._post_chat(
                messages,
                self.timeout if timeout is None else timeout,
            )
            return VLMObservation(
                response=response,
                snapshot_sha256=image_digest,
                captured_at=captured_at,
                model=self.model,
                camera_sha256=(("head_cam", image_digest),),
            )
        except TimeoutError:
            logger.error("[VLMClient] head-camera planning timed out")
            return VLMObservation(
                response="",
                snapshot_sha256="",
                captured_at=captured_at,
                model=self.model,
                error="head-camera planning timed out",
            )
        except Exception as exc:
            logger.error("[VLMClient] head-camera planning failed: %s", exc)
            return VLMObservation(
                response="",
                snapshot_sha256="",
                captured_at=captured_at,
                model=self.model,
                error=str(exc),
            )

    def observe_with_direction(self) -> str:
        """
        执行方向性视觉观察。

        VLM 返回方向性描述（左右远近），用于迭代式底盘调整。

        Returns:
            VLM 的文本响应（方向性描述）
        """
        return self.observe(DIRECTIONAL_OBSERVATION_PROMPT)

    def parse_position(self, response: str) -> dict:
        """
        从 VLM 响应中解析物品位置信息。

        Args:
            response: VLM 的文本响应

        Returns:
            解析后的位置信息字典
        """
        # 简单实现：返回原始响应供上层解析
        return {"raw_response": response}

    def generate(self, prompt: str, timeout: int = 30) -> str:
        """
        通用的 LLM 文本生成接口（供 GoalPlanner 使用）。

        Args:
            prompt: 输入提示词
            timeout: 超时时间（秒）

        Returns:
            LLM 的文本响应
        """
        logger.info("[VLMClient] text generation request: %s characters", len(prompt))
        try:
            return self._post_chat([{"role": "user", "content": prompt}], timeout)
        except Exception as e:
            logger.error("[VLMClient] generation failed: %s", e)
            raise

    def verify_completion(self, task: str, response: str) -> bool:
        """
        验证任务是否完成。

        Args:
            task: 任务描述
            response: VLM 响应

        Returns:
            True 如果任务完成
        """
        normalized = response.strip().lower()
        negative_markers = (
            "未完成",
            "没有完成",
            "失败",
            "not completed",
            "not done",
            "incomplete",
            "failed",
        )
        if any(marker in normalized for marker in negative_markers):
            return False
        positive_markers = ("任务已完成", "已经完成", "completed", "task is done")
        return any(marker in normalized for marker in positive_markers)
