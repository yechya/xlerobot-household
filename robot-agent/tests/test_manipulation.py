from __future__ import annotations

import json

from robot_agent.agent import DummyTaskBackend, RobotAgent
from robot_agent.hardware.authorization import (
    _make_robotd_hardware_authorization,
    _robotd_authorized_context,
)
from robot_agent.hardware.vlm import VLMObservation
from robot_agent.manipulation import ManipulationExecutor, parse_evidence_response
from robot_agent.skill_flow import SkillExecutionResult
from robot_agent.task_catalog import TaskCatalog


class FakeVLM:
    def __init__(self, observations: list[VLMObservation]):
        self.observations = list(observations)
        self.prompts: list[str] = []

    def observe_evidence(self, prompt: str) -> VLMObservation:
        self.prompts.append(prompt)
        return self.observations.pop(0)


class FakeBackend:
    def __init__(self, success: bool = True):
        self.success = success
        self.calls: list[str] = []

    def execute_task(self, task_id: str) -> dict:
        self.calls.append(task_id)
        return {
            "success": self.success,
            "stdout": "executed" if self.success else "",
            "error": "" if self.success else "backend failed",
        }


class FakeSkillRouter:
    def __init__(self):
        self.invocations = []

    def execute_skill(self, invocation):
        self.invocations.append(invocation)
        return SkillExecutionResult(success=True, message="skill completed")


def observation(
    task_id: str,
    phase: str,
    check_ids: list[str],
    *,
    digest: str,
    captured_at: float,
    state: str = "pass",
    confidence: float = 0.95,
) -> VLMObservation:
    payload = {
        "schema_version": 1,
        "task_id": task_id,
        "phase": phase,
        "checks": [
            {
                "id": check_id,
                "state": state,
                "confidence": confidence,
                "evidence": f"visual evidence for {check_id}",
            }
            for check_id in check_ids
        ],
    }
    return VLMObservation(json.dumps(payload), digest, captured_at, "fake-vlm")


def test_pipeline_runs_one_backend_attempt_between_independent_evidence():
    catalog = TaskCatalog.default()
    task = catalog.get("fold_simple")
    pre_ids = [item.requirement_id for item in task.manipulation.preconditions]
    post_ids = [item.requirement_id for item in task.manipulation.postconditions]
    vlm = FakeVLM(
        [
            observation("fold_simple", "precondition", pre_ids, digest="a" * 64, captured_at=1),
            observation("fold_simple", "postcondition", post_ids, digest="b" * 64, captured_at=2),
        ]
    )
    backend = FakeBackend()
    executor = ManipulationExecutor(
        vlm_client=vlm,
        task_backend=backend,
        task_catalog=catalog,
        enforce_authorization_binding=False,
    )
    result = executor.execute("fold_simple")
    assert result.success is True
    assert backend.calls == ["fold_simple"]
    assert [step.action for step in result.steps] == [
        "precondition_evidence",
        "task_backend_execute",
        "postcondition_evidence",
    ]
    assert len(vlm.prompts) == 2


def test_pipeline_passes_params_through_standard_skill_executor():
    catalog = TaskCatalog.default()
    task = catalog.get("fold_simple")
    pre_ids = [item.requirement_id for item in task.manipulation.preconditions]
    post_ids = [item.requirement_id for item in task.manipulation.postconditions]
    router = FakeSkillRouter()
    executor = ManipulationExecutor(
        vlm_client=FakeVLM(
            [
                observation(task.task_id, "precondition", pre_ids, digest="a" * 64, captured_at=1),
                observation(task.task_id, "postcondition", post_ids, digest="b" * 64, captured_at=2),
            ]
        ),
        task_backend=router,
        task_catalog=catalog,
        enforce_authorization_binding=False,
    )

    result = executor.execute(task.task_id, params={"target": "cloth"})

    assert result.success is True
    assert router.invocations[0].skill_id == task.task_id
    assert router.invocations[0].params == {"target": "cloth"}
    assert result.steps[1].message == "skill completed"


def test_failed_precondition_never_calls_backend():
    catalog = TaskCatalog.default()
    task = catalog.get("pick_up_bottle")
    pre_ids = [item.requirement_id for item in task.manipulation.preconditions]
    vlm = FakeVLM(
        [
            observation(
                task.task_id,
                "precondition",
                pre_ids,
                digest="a" * 64,
                captured_at=1,
                state="unknown",
            )
        ]
    )
    backend = FakeBackend()
    result = ManipulationExecutor(
        vlm_client=vlm,
        task_backend=backend,
        task_catalog=catalog,
        enforce_authorization_binding=False,
    ).execute(task.task_id)
    assert result.success is False
    assert backend.calls == []
    assert "preconditions" in result.final_message


def test_identical_or_old_postcondition_snapshot_is_rejected():
    catalog = TaskCatalog.default()
    task = catalog.get("stack_contrast")
    pre_ids = [item.requirement_id for item in task.manipulation.preconditions]
    post_ids = [item.requirement_id for item in task.manipulation.postconditions]
    vlm = FakeVLM(
        [
            observation(task.task_id, "precondition", pre_ids, digest="a" * 64, captured_at=2),
            observation(task.task_id, "postcondition", post_ids, digest="a" * 64, captured_at=2),
        ]
    )
    result = ManipulationExecutor(
        vlm_client=vlm,
        task_backend=FakeBackend(),
        task_catalog=catalog,
        enforce_authorization_binding=False,
    ).execute(task.task_id)
    assert result.success is False
    failures = result.steps[-1].data["failures"]
    assert any("identical" in failure for failure in failures)
    assert any("not newer" in failure for failure in failures)


def test_unstructured_completion_text_is_not_evidence():
    catalog = TaskCatalog.default()
    task = catalog.get("fold_simple")
    raw = VLMObservation("任务已完成", "a" * 64, 1, "fake-vlm")
    report = parse_evidence_response(
        task,
        "postcondition",
        task.manipulation.postconditions,
        raw,
    )
    assert report.passed is False
    assert "invalid evidence response" in report.failures[0]


def test_evidence_parser_accepts_model_results_alias_and_json_fence():
    catalog = TaskCatalog.default()
    task = catalog.get("fold_simple")
    payload = {
        "schema_version": 1,
        "task_id": task.task_id,
        "phase": "precondition",
        "results": [
            {
                "id": requirement.requirement_id,
                "state": "pass",
                "confidence": 0.95,
                "visual_evidence": f"visible {requirement.requirement_id}",
            }
            for requirement in task.manipulation.preconditions
        ],
    }
    report = parse_evidence_response(
        task,
        "precondition",
        task.manipulation.preconditions,
        VLMObservation(
            "```json\n" + json.dumps(payload) + "\n```",
            "a" * 64,
            1,
            "fake-vlm",
        ),
    )
    assert report.passed is True
    assert [check.evidence for check in report.checks]


def test_pipeline_requires_task_contract_binding():
    catalog = TaskCatalog.default()
    task = catalog.get("fold_simple")
    executor = ManipulationExecutor(
        vlm_client=FakeVLM([]),
        task_backend=FakeBackend(),
        task_catalog=catalog,
    )
    wrong = _make_robotd_hardware_authorization(
        "secret",
        task.task_id,
        {"task_fingerprint": "wrong", "task_executor": "learned_policy"},
    )
    with _robotd_authorized_context(wrong):
        result = executor.execute(task.task_id)
    assert result.success is False
    assert result.steps[0].action == "authorization_binding"


def test_pipeline_accepts_exact_task_contract_binding():
    catalog = TaskCatalog.default()
    task = catalog.get("fold_simple")
    pre_ids = [item.requirement_id for item in task.manipulation.preconditions]
    post_ids = [item.requirement_id for item in task.manipulation.postconditions]
    executor = ManipulationExecutor(
        vlm_client=FakeVLM(
            [
                observation(
                    task.task_id,
                    "precondition",
                    pre_ids,
                    digest="a" * 64,
                    captured_at=1,
                ),
                observation(
                    task.task_id,
                    "postcondition",
                    post_ids,
                    digest="b" * 64,
                    captured_at=2,
                ),
            ]
        ),
        task_backend=FakeBackend(),
        task_catalog=catalog,
    )
    authorization = _make_robotd_hardware_authorization(
        "secret",
        task.task_id,
        {
            "task_fingerprint": task.fingerprint(),
            "task_executor": task.executor,
        },
    )
    with _robotd_authorized_context(authorization):
        result = executor.execute(task.task_id)
    assert result.success is True


def test_normal_robot_agent_uses_deterministic_manipulation_pipeline():
    agent = RobotAgent(dry_run=False)
    assert isinstance(agent.manipulation_executor, ManipulationExecutor)
    assert not hasattr(agent.manipulation_executor, "chassis")


def test_robot_agent_dry_run_pipeline_uses_dummy_backend_for_final_step():
    catalog = TaskCatalog.default()
    task = catalog.get("fold_simple")
    pre_ids = [item.requirement_id for item in task.manipulation.preconditions]
    post_ids = [item.requirement_id for item in task.manipulation.postconditions]
    agent = RobotAgent(
        dry_run=False,
        task_catalog=catalog,
        task_backend=DummyTaskBackend(),
    )
    agent.manipulation_executor.observer.vlm = FakeVLM(
        [
            observation(task.task_id, "precondition", pre_ids, digest="a" * 64, captured_at=1),
            observation(task.task_id, "postcondition", post_ids, digest="b" * 64, captured_at=2),
        ]
    )

    result = agent.execute_dry_run_pipeline(task.task_id)

    assert result.success is True
    assert result.data["execution_mode"] == "pipeline_dummy"
    assert result.data["steps"][1]["data"]["dummy_command"] is True


def test_default_bottle_task_uses_original_learned_policy():
    task = TaskCatalog.default().get("pick_up_bottle")
    assert task is not None
    assert task.executor == "learned_policy"
    assert task.executor_config == {}


def test_default_bottle_on_towel_task_uses_original_learned_policy():
    task = TaskCatalog.default().get("pick_up_bottle_towel")
    assert task is not None
    assert task.executor == "learned_policy"
    assert task.executor_config == {}
