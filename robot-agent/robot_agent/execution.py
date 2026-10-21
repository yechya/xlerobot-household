from __future__ import annotations

import multiprocessing
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorkerResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


def _agent_worker(
    task_id: str,
    execution_binding: dict[str, Any],
    token: str,
    world_model_path: str,
    policy_registry_path: str,
    task_catalog_path: str,
    connection: Any,
) -> None:
    try:
        if hasattr(os, "setsid"):
            os.setsid()
        if policy_registry_path:
            os.environ["ROBOT_AGENT_POLICY_REGISTRY"] = policy_registry_path
        if task_catalog_path:
            os.environ["ROBOT_AGENT_TASK_CATALOG"] = task_catalog_path
        from .agent import RobotAgent
        from .hardware.authorization import (
            _make_robotd_hardware_authorization,
            _robotd_authorized_context,
        )

        authorization = _make_robotd_hardware_authorization(
            token,
            task_id,
            execution_binding,
        )
        with _robotd_authorized_context(authorization):
            result = RobotAgent(
                dry_run=False,
                persistence_path=world_model_path,
            ).execute(task_id)
        connection.send(
            {
                "success": bool(result.success),
                "message": str(result.message),
                "data": dict(result.data or {}),
            }
        )
    except BaseException as exc:
        connection.send(
            {
                "success": False,
                "message": f"worker failed: {type(exc).__name__}: {exc}",
                "data": {"error_type": type(exc).__name__},
            }
        )
    finally:
        connection.close()


class ProcessExecutionSupervisor:
    """Runs one hardware task at a time in a cancellable process group."""

    def __init__(
        self,
        *,
        world_model_path: str | Path,
        policy_registry_path: str | Path | None = None,
        task_catalog_path: str | Path | None = None,
        emergency_stop_command: tuple[str, ...] = (),
        shutdown_grace_seconds: float = 2.0,
        worker_target: Any = _agent_worker,
    ):
        self.world_model_path = str(world_model_path)
        self.policy_registry_path = str(policy_registry_path or "")
        self.task_catalog_path = str(task_catalog_path or "")
        self.emergency_stop_command = tuple(emergency_stop_command)
        self.shutdown_grace_seconds = shutdown_grace_seconds
        self._worker_target = worker_target
        self._lock = threading.RLock()
        self._process: multiprocessing.Process | None = None
        self._cancel_reason = ""

    @property
    def active(self) -> bool:
        with self._lock:
            if self._process is None or self._process.pid is None:
                return False
            return self._process.is_alive()

    def execute(
        self,
        task_id: str,
        args: dict[str, Any],
        token: str,
        timeout_seconds: int,
    ) -> WorkerResult:
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=self._worker_target,
            args=(
                task_id,
                dict(args),
                token,
                self.world_model_path,
                self.policy_registry_path,
                self.task_catalog_path,
                sender,
            ),
            name=f"robot-task-{task_id}",
        )

        with self._lock:
            if self._process is not None and self._process.is_alive():
                receiver.close()
                sender.close()
                return WorkerResult(False, "another hardware task is already active")
            self._cancel_reason = ""
            self._process = process
            try:
                process.start()
            except Exception:
                self._process = None
                receiver.close()
                sender.close()
                raise
        sender.close()

        deadline = time.monotonic() + timeout_seconds
        try:
            while process.is_alive() and time.monotonic() < deadline:
                process.join(timeout=0.1)

            if process.is_alive():
                self.cancel(reason="max_runtime_exceeded", emergency=True)

            process.join(timeout=self.shutdown_grace_seconds)
            if receiver.poll():
                payload = receiver.recv()
                return WorkerResult(
                    success=bool(payload.get("success", False)),
                    message=str(payload.get("message", "")),
                    data=dict(payload.get("data", {}) or {}),
                )

            reason = self._cancel_reason or f"worker exited with code {process.exitcode}"
            return WorkerResult(False, reason, {"exit_code": process.exitcode})
        finally:
            receiver.close()
            with self._lock:
                if self._process is process:
                    self._process = None

    def cancel(self, *, reason: str, emergency: bool = False) -> None:
        with self._lock:
            self._cancel_reason = reason
            process = self._process

        if process is not None and process.pid is not None and process.is_alive():
            self._terminate(process, signal.SIGTERM)
            process.join(timeout=self.shutdown_grace_seconds)
            if process.is_alive():
                self._terminate(process, signal.SIGKILL)
                process.join(timeout=self.shutdown_grace_seconds)

        if emergency or process is not None:
            self._run_emergency_stop_hook()

    @staticmethod
    def _terminate(process: multiprocessing.Process, signal_number: int) -> None:
        try:
            process_group = os.getpgid(process.pid)
            if process_group == process.pid:
                os.killpg(process_group, signal_number)
            else:
                os.kill(process.pid, signal_number)
        except (ProcessLookupError, OSError, TypeError):
            return

    def _run_emergency_stop_hook(self) -> None:
        if not self.emergency_stop_command:
            return
        try:
            subprocess.run(
                list(self.emergency_stop_command),
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return


__all__ = ["ProcessExecutionSupervisor", "WorkerResult"]
