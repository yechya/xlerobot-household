from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .skill_flow import (
    SkillExecutionContext,
    SkillExecutionResult,
    SkillInvocation,
)


_IDENTIFIER = r"[A-Za-z0-9_.:-]{1,128}"
_SUPPORTED_PREDICATES = (
    re.compile(r"^gripper_empty$"),
    re.compile(rf"^gripper_holding\({_IDENTIFIER}\)$"),
    re.compile(rf"^object_visible\({_IDENTIFIER}\)$"),
    re.compile(rf"^object_on\({_IDENTIFIER},{_IDENTIFIER}\)$"),
    re.compile(rf"^object_in\({_IDENTIFIER},{_IDENTIFIER}\)$"),
)


class RgbWorldPredicateExecutor:
    """Evaluate reviewed qualitative predicates from the RGB world model."""

    executor_id = "rgb_world_predicate"

    def validate_config(self, config: Mapping[str, Any]) -> tuple[bool, str | None]:
        if set(config) != {"predicate"}:
            return False, "rgb_world_predicate config must contain only predicate"
        predicate = config.get("predicate")
        if not isinstance(predicate, str):
            return False, "rgb_world_predicate predicate must be a string"
        normalized = predicate.strip().lower()
        if not any(pattern.fullmatch(normalized) for pattern in _SUPPORTED_PREDICATES):
            return False, f"unsupported world predicate: {predicate!r}"
        return True, None

    def execute(
        self,
        invocation: SkillInvocation,
        context: SkillExecutionContext,
    ) -> SkillExecutionResult:
        del invocation
        predicate = str(context.config["predicate"]).strip().lower()
        world_model = context.world_model
        checker = getattr(world_model, "check_precondition", None)
        if not callable(checker):
            return SkillExecutionResult(
                success=False,
                message="RGB world model is unavailable",
                error="world_model_unavailable",
                data={"predicate": predicate, "state": "unknown"},
            )
        try:
            passed = bool(checker(predicate))
        except Exception as exc:
            return SkillExecutionResult(
                success=False,
                message=f"RGB world predicate failed: {type(exc).__name__}: {exc}",
                error="world_predicate_error",
                data={"predicate": predicate, "state": "unknown"},
            )
        return SkillExecutionResult(
            success=passed,
            message=(
                f"RGB world predicate passed: {predicate}"
                if passed
                else f"RGB world predicate not proven: {predicate}"
            ),
            error="" if passed else "predicate_not_proven",
            data={
                "predicate": predicate,
                "state": "pass" if passed else "unknown",
                "source": "rgb_world_model",
            },
        )


__all__ = ["RgbWorldPredicateExecutor"]
