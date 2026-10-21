from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from .hardware.vlm import VLMObservation
from .task_catalog import TaskCatalog


class VisualPlanError(ValueError):
    pass


class HeadCameraPlanner(Protocol):
    def generate_head_camera(
        self,
        instruction: str,
        *,
        timeout: int | None = None,
    ) -> VLMObservation: ...


@dataclass(frozen=True)
class VisualPlanStep:
    task_id: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class VisualTaskPlan:
    plan_id: str
    goal: str
    steps: tuple[VisualPlanStep, ...]
    task_fingerprints: tuple[tuple[str, str], ...]
    captured_at: float
    camera_snapshot_sha256: str
    model: str
    plan_sha256: str

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(step.task_id for step in self.steps)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "task_fingerprints": [
                {"task_id": task_id, "fingerprint": fingerprint}
                for task_id, fingerprint in self.task_fingerprints
            ],
            "captured_at": self.captured_at,
            "camera_snapshot_sha256": self.camera_snapshot_sha256,
            "model": self.model,
        }

    def validate(self, *, include_hash: bool = True) -> None:
        if not re.fullmatch(r"plan-[a-f0-9]{32}", self.plan_id):
            raise VisualPlanError("visual plan ID is invalid")
        if (
            not isinstance(self.goal, str)
            or not 0 < len(self.goal.strip()) <= 1024
        ):
            raise VisualPlanError("visual plan goal is invalid")
        if not 0 <= len(self.steps) <= 16:
            raise VisualPlanError("visual plan must contain 0..16 steps")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise VisualPlanError("visual plan task IDs must be unique")
        if tuple(task_id for task_id, _ in self.task_fingerprints) != self.task_ids:
            raise VisualPlanError("visual plan task fingerprints do not match steps")
        for task_id, fingerprint in self.task_fingerprints:
            _identifier(task_id, "visual plan fingerprint task ID")
            _sha(fingerprint, "visual plan task fingerprint")
        if not math.isfinite(self.captured_at) or self.captured_at <= 0:
            raise VisualPlanError("visual plan capture timestamp is invalid")
        _sha(self.camera_snapshot_sha256, "camera_snapshot_sha256")
        if not isinstance(self.model, str) or not 0 < len(self.model) <= 128:
            raise VisualPlanError("visual plan model is invalid")
        for step in self.steps:
            _identifier(step.task_id, "visual plan task ID")
            if (
                not math.isfinite(step.confidence)
                or not 0.5 <= step.confidence <= 1.0
            ):
                raise VisualPlanError("visual plan confidence must be in [0.5, 1]")
            if not isinstance(step.reason, str) or not 0 < len(step.reason) <= 512:
                raise VisualPlanError("visual plan reason is invalid")
        if include_hash:
            _sha(self.plan_sha256, "plan_sha256")
            if self.plan_sha256 != _digest(self._hash_payload()):
                raise VisualPlanError("visual plan hash does not match content")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._hash_payload(),
            "task_ids": list(self.task_ids),
            "plan_sha256": self.plan_sha256,
        }


class VisualTaskPlanner:
    """Use one head RGB view to select an ordered subset of catalog tasks."""

    def __init__(self, task_catalog: TaskCatalog, vlm_client: HeadCameraPlanner):
        self.task_catalog = task_catalog
        self.vlm_client = vlm_client

    def create(
        self,
        goal: str,
        *,
        world_context: dict[str, Any] | None = None,
    ) -> VisualTaskPlan:
        if not isinstance(goal, str) or not 0 < len(goal.strip()) <= 1024:
            raise VisualPlanError("visual planning goal must be 1..1024 characters")
        prompt = self._prompt(goal.strip(), world_context)
        observation = self.vlm_client.generate_head_camera(prompt, timeout=60)
        if not observation.success:
            raise VisualPlanError(observation.error or "head-camera planning failed")
        steps = self._parse(observation.response)
        plan = VisualTaskPlan(
            plan_id=f"plan-{uuid.uuid4().hex}",
            goal=goal.strip(),
            steps=steps,
            task_fingerprints=tuple(
                (step.task_id, self.task_catalog.tasks[step.task_id].fingerprint())
                for step in steps
            ),
            captured_at=observation.captured_at,
            camera_snapshot_sha256=observation.snapshot_sha256,
            model=observation.model,
            plan_sha256="",
        )
        plan.validate(include_hash=False)
        return VisualTaskPlan(
            **{
                **plan.__dict__,
                "plan_sha256": _digest(plan._hash_payload()),
            }
        )

    def _prompt(self, goal: str, world_context: dict[str, Any] | None) -> str:
        catalog = [
            {
                "task_id": task.task_id,
                "name": task.name,
                "description": task.description,
                "preconditions": [
                    requirement.requirement_id
                    for requirement in task.manipulation.preconditions
                ],
                "postconditions": [
                    requirement.requirement_id
                    for requirement in task.manipulation.postconditions
                ],
            }
            for task in sorted(
                self.task_catalog.tasks.values(),
                key=lambda value: value.task_id,
            )
        ]
        context = world_context or {}
        context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)[:8192]
        catalog_json = json.dumps(catalog, ensure_ascii=False, sort_keys=True)
        return f"""你是居家机器人任务分解器，会看到一张 head_cam RGB 图像。

用户抽象指令：{goal}

当前世界模型摘要（仅作辅助，不得据此创造新能力）：
{context_json}

已注册任务目录（只能逐字选择其中的 task_id）：
{catalog_json}

请根据图像和指令，选择需要按顺序执行的已注册任务。只返回严格 JSON：
{{"steps":[{{"task_id":"registered_id","confidence":0.85,"reason":"简短依据"}}]}}

规则：
1. 只能选择目录中的 task_id，不得生成坐标、参数、脚本或新任务。
2. steps 按执行顺序排列，不得重复，最多 16 项。
3. confidence 必须是 0.5 到 1 之间的数字；不确定时返回空 steps。
4. reason 只能说明图像和指令依据，不能宣称已经执行或保证安全。
5. 复杂流程只有目录中存在对应任务时才能选择。
6. 只输出一个 JSON 对象，不要 Markdown 代码块、解释文字、函数名或其他字段。
7. 图像昏暗、模糊、遮挡，或无法确认目标时必须返回 {{"steps":[]}}。
"""

    def _parse(self, response: str) -> tuple[VisualPlanStep, ...]:
        try:
            raw = _parse_json_object(response)
        except json.JSONDecodeError as exc:
            raise VisualPlanError(f"visual plan JSON is invalid: {exc}") from exc
        if not isinstance(raw, dict) or set(raw) != {"steps"}:
            raise VisualPlanError("visual plan must contain only steps")
        values = raw["steps"]
        if not isinstance(values, list) or not 0 <= len(values) <= 16:
            raise VisualPlanError("visual plan steps must contain 0..16 items")
        steps: list[VisualPlanStep] = []
        seen: set[str] = set()
        for value in values:
            if (
                not isinstance(value, dict)
                or set(value) != {"task_id", "confidence", "reason"}
            ):
                raise VisualPlanError("visual plan step has invalid fields")
            task_id = value["task_id"]
            if not isinstance(task_id, str) or task_id not in self.task_catalog.tasks:
                raise VisualPlanError(f"visual plan task is not registered: {task_id!r}")
            if task_id in seen:
                raise VisualPlanError(f"visual plan task is duplicated: {task_id!r}")
            confidence = value["confidence"]
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise VisualPlanError("visual plan confidence must be numeric")
            reason = value["reason"]
            step = VisualPlanStep(task_id, float(confidence), reason)
            step_reason = step.reason
            if not isinstance(step_reason, str) or not 0 < len(step_reason) <= 512:
                raise VisualPlanError("visual plan reason is invalid")
            steps.append(step)
            seen.add(task_id)
        return tuple(steps)


def _parse_json_object(response: str) -> object:
    """Parse plain JSON or one complete JSON fence, without accepting prose."""
    text = response.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            raise first_error
        return json.loads(match.group(1))


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}", value
    ):
        raise VisualPlanError(f"{label} is invalid")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise VisualPlanError(f"{label} must be a SHA-256 digest")
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "VisualPlanError",
    "VisualPlanStep",
    "VisualTaskPlan",
    "VisualTaskPlanner",
]
