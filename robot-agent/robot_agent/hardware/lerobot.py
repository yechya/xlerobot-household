from __future__ import annotations

import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable

from ..policies import PolicyRegistry, PolicyRegistryError
from .authorization import (
    _authorized_execution_binding,
    validate_actuation_authorized,
)

logger = logging.getLogger("hardware.lerobot")
ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


class LeRobotClient:
    """Executes one robotd-bound, integrity-checked learned policy."""

    def __init__(
        self,
        timeout: int | None = None,
        *,
        registry: PolicyRegistry | None = None,
        process_runner: ProcessRunner | None = None,
    ):
        self.timeout = timeout
        self._registry = registry
        self._process_runner = process_runner or subprocess.run

    def _policy_registry(self) -> PolicyRegistry:
        if self._registry is not None:
            return self._registry
        configured = os.environ.get("ROBOT_AGENT_POLICY_REGISTRY", "").strip()
        if not configured:
            raise PolicyRegistryError("ROBOT_AGENT_POLICY_REGISTRY is not configured")
        self._registry = PolicyRegistry.load(Path(configured))
        return self._registry

    def execute_task(self, task_id: str) -> dict[str, Any]:
        authorization_refusal = validate_actuation_authorized("lerobot.execute_task", task_id)
        if authorization_refusal is not None:
            logger.warning(
                "LeRobot actuation refused: %s",
                authorization_refusal.reason_code,
            )
            return authorization_refusal.to_result()

        binding = _authorized_execution_binding()
        if not binding:
            return {
                "success": False,
                "error": "robotd authorization is missing a policy binding",
                "task_id": task_id,
            }
        try:
            registry = self._policy_registry()
            policy = registry.validate_binding(task_id, binding)
            run_id = uuid.uuid4().hex
            command = registry.build_command(policy, run_id=run_id)
            dataset_repo_id = registry.dataset_repo_id(policy, run_id)
        except (OSError, PolicyRegistryError) as exc:
            return {"success": False, "error": str(exc), "task_id": task_id}

        timeout = (
            min(self.timeout, policy.timeout_seconds)
            if self.timeout
            else policy.timeout_seconds
        )
        logger.info(
            "executing task=%s policy=%s fingerprint=%s",
            task_id,
            policy.policy_id,
            binding["policy_fingerprint"][:12],
        )
        try:
            result = self._process_runner(
                command,
                cwd=policy.workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"policy timed out after {timeout} seconds",
                "task_id": task_id,
                "policy_id": policy.policy_id,
            }
        except OSError as exc:
            return {"success": False, "error": str(exc), "task_id": task_id}

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:] if result.stdout else "",
            "stderr": result.stderr[-4000:] if result.stderr else "",
            "task_id": task_id,
            "policy_id": policy.policy_id,
            "policy_family": policy.family,
            "policy_fingerprint": binding["policy_fingerprint"],
            "run_id": run_id,
            "dataset_repo_id": dataset_repo_id,
            "error": "" if result.returncode == 0 else "policy process failed",
        }

    def list_tasks(self) -> list[dict[str, Any]]:
        registry = self._policy_registry()
        return [
            {
                "task_id": task_id,
                "policy_id": policy_id,
                "enabled": registry.policies[policy_id].enabled,
            }
            for task_id, policy_id in sorted(registry.default_policies.items())
        ]


__all__ = ["LeRobotClient"]
