"""Phase 3 单元测试

测试目标：
- T11: GoalPlanner 单元测试
- T12: DialogueHistory 单元测试
- T13: ErrorDiagnoser 单元测试
- T14: RobotAgent 集成测试

运行方法：
    cd robot-agent
    python -m pytest tests/test_phase3.py -v
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from robot_agent.dialogue_history import DialogueHistory
from robot_agent.error_diagnoser import ErrorDiagnoser
from robot_agent.execution_types import ExecutionStep, TaskExecutionResult
from robot_agent.goal_planner import GoalPlanner, PlanningError, TaskPlan
from robot_agent.skill_flow import get_allowed_skills
from robot_agent.world_model import WorldModel

# ============================================================================
# T11: GoalPlanner 单元测试
# ============================================================================

class TestGoalPlanner:
    """GoalPlanner 单元测试"""

    def test_decompose_valid_json(self):
        planner = GoalPlanner()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            '{"task_ids":["fold_simple","stack_contrast"]}'
        )
        planner._llm_client = mock_llm
        plans = planner.decompose("把桌子收拾干净")

        assert [plan.skill for plan in plans] == ["fold_simple", "stack_contrast"]
        assert [plan.id for plan in plans] == ["task_1", "task_2"]
        assert all(plan.goal == "把桌子收拾干净" for plan in plans)
        assert all(plan.depends_on == [] for plan in plans)
        assert all(plan.preconditions == [] for plan in plans)

    def test_decompose_invalid_json(self):
        """T11-02: LLM 返回无效 JSON"""
        planner = GoalPlanner()

        invalid_response = "这不是有效的JSON格式"

        mock_llm = MagicMock()
        mock_llm.generate.return_value = invalid_response
        planner._llm_client = mock_llm
        with pytest.raises(PlanningError) as excinfo:
            planner.decompose("把桌子收拾干净")
        assert "JSON" in str(excinfo.value)

    def test_decompose_invalid_skill(self):
        planner = GoalPlanner()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"task_ids":["unknown_skill"]}'
        planner._llm_client = mock_llm
        with pytest.raises(PlanningError) as excinfo:
            planner.decompose("把桌子收拾干净")
        assert "不在允许列表中" in str(excinfo.value)

    def test_generated_dependencies_and_preconditions_are_rejected(self):
        planner = GoalPlanner()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            '{"task_ids":["fold_simple"],"depends_on":["anything"],'
            '"preconditions":["trust me"]}'
        )
        planner._llm_client = mock_llm
        with pytest.raises(PlanningError) as excinfo:
            planner.decompose("把桌子收拾干净")
        assert "只能包含" in str(excinfo.value)

    def test_markdown_wrapped_or_trailing_output_is_rejected(self):
        planner = GoalPlanner()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '```json\n{"task_ids":["fold_simple"]}\n```'
        planner._llm_client = mock_llm
        with pytest.raises(PlanningError):
            planner.decompose("把桌子收拾干净")

    def test_duplicate_task_ids_are_rejected(self):
        planner = GoalPlanner()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            '{"task_ids":["fold_simple","fold_simple"]}'
        )
        planner._llm_client = mock_llm
        with pytest.raises(PlanningError) as excinfo:
            planner.decompose("把桌子收拾干净")
        assert "不得重复" in str(excinfo.value)

    def test_allowed_skills_whitelist(self):
        """验证技能白名单"""
        expected_skills = frozenset([
            "dock_table",
            "fold_simple",
            "stack_contrast",
            "pick_up_bottle_towel",
            "pick_up_bottle",
        ])
        assert GoalPlanner.get_allowed_skills() == expected_skills
        assert GoalPlanner.get_allowed_skills() == get_allowed_skills()

    def test_legacy_unregistered_skills_rejected(self):
        planner = GoalPlanner()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"task_ids":["place_bottle"]}'
        planner._llm_client = mock_llm
        with pytest.raises(PlanningError) as excinfo:
            planner.decompose("整理瓶子")
        assert "不在允许列表中" in str(excinfo.value)

    def test_llm_timeout(self):
        """LLM 超时处理"""
        planner = GoalPlanner()

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = TimeoutError("LLM 超时")
        planner._llm_client = mock_llm
        with pytest.raises(PlanningError) as excinfo:
            planner.decompose("把桌子收拾干净")
        assert "超时" in str(excinfo.value)

    def test_task_plan_status_default(self):
        """TaskPlan 默认状态为 pending"""
        plan = TaskPlan(
            id="task_1",
            skill="fold_simple",
            goal="折叠毛巾"
        )
        assert plan.status == "pending"


# ============================================================================
# T12: DialogueHistory 单元测试
# ============================================================================

class TestDialogueHistory:
    """DialogueHistory 单元测试"""

    def test_add_user_message(self):
        """T12-01: 添加用户消息"""
        history = DialogueHistory(max_entries=100)

        history.add("user", "折叠毛巾")
        assert len(history.entries) == 1
        assert history.entries[0].role == "user"
        assert history.entries[0].content == "折叠毛巾"
        assert history.entries[0].timestamp > 0

    def test_search_keyword(self):
        """T12-02: 关键词搜索"""
        history = DialogueHistory()

        history.add("user", "折叠毛巾")
        history.add("assistant", "开始执行折叠任务")
        history.add("user", "叠放碗碟")
        history.add("assistant", "开始叠碗")

        results = history.search("折叠")
        assert len(results) == 2
        assert "折叠" in results[0].content
        assert "折叠" in results[1].content

        # 搜索不存在的内容
        empty_results = history.search("不存在的关键词")
        assert len(empty_results) == 0

    def test_get_recent_limit(self):
        """T12-03: 获取最近 N 条"""
        history = DialogueHistory()

        for i in range(10):
            history.add("user", f"消息 {i}")

        recent_5 = history.get_recent(5)
        assert len(recent_5) == 5
        # 验证是最新的5条
        assert recent_5[0].content == "消息 5"
        assert recent_5[4].content == "消息 9"

    def test_summarize(self):
        """T12-04: 生成摘要"""
        history = DialogueHistory()

        history.add("user", "折叠毛巾")
        history.add("assistant", "执行中")
        history.add("user", "叠放碗碟", metadata={"task_id": "stack_contrast"})

        summary = history.summarize()
        assert "对话" in summary
        assert "用户消息" in summary
        assert "fold" not in summary  # 不包含技能ID

    def test_max_entries_truncation(self):
        """T12-05: 超过 max_entries 截断"""
        history = DialogueHistory(max_entries=5)

        for i in range(10):
            history.add("user", f"消息 {i}")

        assert len(history.entries) == 5
        # 最早的被截断，最新的是消息 5-9
        assert history.entries[0].content == "消息 5"
        assert history.entries[4].content == "消息 9"

    def test_add_with_metadata(self):
        """带元数据的添加"""
        history = DialogueHistory()

        history.add(
            "assistant",
            "执行成功",
            metadata={"task_id": "fold_simple", "duration": 10.5}
        )

        entry = history.entries[0]
        assert entry.metadata["task_id"] == "fold_simple"
        assert entry.metadata["duration"] == 10.5

    def test_get_user_goals(self):
        """获取所有用户目标"""
        history = DialogueHistory()

        history.add("user", "折叠毛巾")
        history.add("assistant", "执行中")
        history.add("user", "叠放碗碟")

        goals = history.get_user_goals()
        assert len(goals) == 2
        assert "折叠毛巾" in goals
        assert "叠放碗碟" in goals

    def test_get_failed_tasks(self):
        """获取失败任务记录"""
        history = DialogueHistory()

        history.add("user", "折叠毛巾")
        history.add("assistant", "执行成功")
        history.add("user", "放瓶子")
        history.add("assistant", "任务失败：VLM 观察失败")

        failed = history.get_failed_tasks()
        assert len(failed) == 1
        assert failed[0]["goal"] == "放瓶子"
        assert "失败" in failed[0]["result"]

    def test_empty_history(self):
        """空历史处理"""
        history = DialogueHistory()

        assert history.get_recent(5) == []
        assert history.search("test") == []
        assert history.summarize() == "无对话历史"


# ============================================================================
# T13: ErrorDiagnoser 单元测试
# ============================================================================

class TestErrorDiagnoser:
    """ErrorDiagnoser 单元测试"""

    def _make_step(self, action: str, message: str) -> ExecutionStep:
        return ExecutionStep(name="test", action=action, success=False, message=message)

    def test_vlm_observe_failure(self):
        """T13-01: VLM 观察失败"""
        diagnoser = ErrorDiagnoser()
        step = self._make_step("vlm_observe", "VLM 观察失败")
        wm = WorldModel()

        report = diagnoser.diagnose(step, wm, "无法看到目标")

        assert report.error_type == "object_not_visible"
        assert "物品被遮挡" in report.possible_causes
        assert len(report.suggestions) > 0

    def test_chassis_failure(self):
        """T13-02: 底盘移动失败"""
        diagnoser = ErrorDiagnoser()
        step = self._make_step("chassis_move", "底盘移动失败")
        wm = WorldModel()

        report = diagnoser.diagnose(step, wm, "")

        assert report.error_type == "adjustment_failed"
        assert "障碍物阻挡" in report.possible_causes

    def test_vla_failure(self):
        """T13-03: VLA 执行失败"""
        diagnoser = ErrorDiagnoser()
        step = self._make_step("lerobot_execute", "VLA 执行失败")
        wm = WorldModel()

        report = diagnoser.diagnose(step, wm, "")

        assert report.error_type == "task_execution_failed"
        assert "物品位置偏差" in report.possible_causes

    def test_max_iterations(self):
        """T13-04: 最大迭代次数"""
        diagnoser = ErrorDiagnoser()
        step = self._make_step("chassis_navigate", "达到最大迭代次数 8")
        wm = WorldModel()

        report = diagnoser.diagnose(step, wm, "")

        assert report.error_type == "range_not_reached"
        assert "物品距离太远" in report.possible_causes

    def test_unknown_error(self):
        """T13-05: 未知错误"""
        diagnoser = ErrorDiagnoser()
        step = self._make_step("unknown_action", "这是完全未知的错误")
        wm = WorldModel()

        report = diagnoser.diagnose(step, wm, "")

        assert report.error_type == "unknown_error"
        assert report.confidence == 0.0

    def test_regex_pattern_matching(self):
        """正则表达式模式匹配"""
        diagnoser = ErrorDiagnoser()

        # 测试震荡匹配
        step = self._make_step("navigate", "检测到方向震荡")
        report = diagnoser.diagnose(step, WorldModel(), "")
        assert report.error_type == "adjustment_failed"

        # 测试迭代次数（regex "达到.*次" 匹配 "3次调整后仍未进入范围"）
        step = self._make_step("adjust", "达到最大迭代次数 8")
        report = diagnoser.diagnose(step, WorldModel(), "")
        assert report.error_type == "range_not_reached"

    def test_confidence_calculation(self):
        """置信度计算"""
        diagnoser = ErrorDiagnoser()
        step = self._make_step("test", "测试")

        report = diagnoser.diagnose(step, WorldModel(), "")
        # 未知错误置信度为 0
        assert report.confidence == 0.0

    def test_suggestions_deduplication(self):
        """建议去重"""
        diagnoser = ErrorDiagnoser()
        step = self._make_step("test", "")

        # 制造一个有两个原因匹配同一 fixes 的情况
        report = diagnoser.diagnose(step, WorldModel(), "")

        # 验证去重：同一个建议不重复出现
        if len(report.suggestions) > 1:
            seen = set()
            for s in report.suggestions:
                assert s not in seen, "建议重复"
                seen.add(s)

    def test_oscillation_pattern(self):
        """震荡错误模式"""
        diagnoser = ErrorDiagnoser()
        step = self._make_step("adjust", "方向震荡停止")

        report = diagnoser.diagnose(step, WorldModel(), "")
        assert report.error_type == "adjustment_failed"

    def test_gripper_failure(self):
        """夹爪失败"""
        diagnoser = ErrorDiagnoser()
        step = self._make_step("execute", "夹爪无法夹住物品")

        report = diagnoser.diagnose(step, WorldModel(), "")
        assert report.error_type == "task_execution_failed"

    def test_four_error_types(self):
        """T13: 验证支持 4 种以上错误类型"""
        diagnoser = ErrorDiagnoser()

        error_types = [
            ("vlm_observe", "VLM 观察失败", "object_not_visible"),
            ("chassis_move", "底盘移动失败", "adjustment_failed"),
            ("lerobot_execute", "VLA 执行失败", "task_execution_failed"),
            ("navigate", "最大迭代次数", "range_not_reached"),
        ]

        for action, message, expected_type in error_types:
            step = self._make_step(action, message)
            report = diagnoser.diagnose(step, WorldModel(), "")
            assert report.error_type == expected_type, (
                f"Action: {action}, Expected: {expected_type}, "
                f"Got: {report.error_type}"
            )


# ============================================================================
# T14: RobotAgent 集成测试
# ============================================================================

class TestRobotAgent:
    """RobotAgent integration without legacy motion executors."""

    def _mock_agent(self, *, success: bool = True):
        from robot_agent.agent import RobotAgent

        agent = RobotAgent.__new__(RobotAgent)
        agent.dry_run = False
        agent.status = "idle"
        agent.TASK_MAP = {
            "fold_simple": "fold cloth",
            "stack_contrast": "stack bowls",
            "pick_up_bottle_towel": "place bottle on towel",
            "pick_up_bottle": "place bottle in basket",
        }
        agent._instance_task_keywords = {
            task_id: [task_id] for task_id in agent.TASK_MAP
        }
        agent.vlm = MagicMock()
        agent.lerobot = MagicMock()
        agent.camera = MagicMock()
        agent.world_model = WorldModel()
        agent.goal_planner = GoalPlanner(allowed_skills=frozenset(agent.TASK_MAP))
        agent.dialogue_history = DialogueHistory()
        agent.error_diagnoser = ErrorDiagnoser()
        step = ExecutionStep(
            name="fake backend",
            action="task_backend_execute",
            success=success,
            message="completed" if success else "backend execution failed",
        )
        agent.manipulation_executor = MagicMock()
        agent.manipulation_executor.execute.side_effect = (
            lambda task_id: TaskExecutionResult(
            success=success,
            task_id=task_id,
            total_steps=1,
            completed_steps=1,
            steps=[step],
            final_message="completed" if success else "backend execution failed",
            )
        )
        agent.skill_registry = MagicMock()
        return agent

    def test_execute_records_dialogue(self):
        agent = self._mock_agent()
        result = agent.execute("fold_simple")
        assert result.success is True
        assert len(agent.dialogue_history.entries) == 2
        assert agent.dialogue_history.entries[0].role == "user"
        assert agent.dialogue_history.entries[0].content == "fold_simple"
        assert agent.dialogue_history.entries[1].role == "assistant"

    def test_execute_failure_generates_diagnosis(self):
        agent = self._mock_agent(success=False)
        result = agent.execute("fold_simple")
        assert result.success is False
        assert result.data.get("diagnosis") is not None
        diagnosis = result.data["diagnosis"]
        assert "error_type" in diagnosis
        assert "possible_causes" in diagnosis
        assert "suggestions" in diagnosis

    def test_run_complex_goal_decomposition(self):
        agent = self._mock_agent()
        mock_plans = [
            TaskPlan(id="task_1", skill="fold_simple", goal="折叠毛巾"),
            TaskPlan(id="task_2", skill="stack_contrast", goal="叠碗"),
        ]
        with patch.object(agent.goal_planner, "decompose", return_value=mock_plans):
            result = agent.run_complex("把桌子收拾干净")
        assert [plan["skill"] for plan in result.data["plans"]] == [
            "fold_simple",
            "stack_contrast",
        ]

    def test_reset_world(self):
        agent = self._mock_agent()
        agent.dialogue_history.add("user", "测试消息")
        agent.reset_world()
        assert isinstance(agent.world_model, WorldModel)
        assert len(agent.dialogue_history.entries) == 0

    def test_simple_task_direct_execution(self):
        agent = self._mock_agent()
        with patch.object(agent.goal_planner, "decompose") as mock_decompose:
            result = agent.run_complex("fold_simple")
        mock_decompose.assert_not_called()
        assert result.success is True

    def test_is_simple_task_patterns(self):
        from robot_agent.agent import RobotAgent

        simple_patterns = ["折叠", "叠", "放瓶子", "抓取"]
        for pattern in simple_patterns:
            assert any(p in pattern for p in RobotAgent.SIMPLE_TASK_PATTERNS)


# ============================================================================
# 辅助测试
# ============================================================================

class TestPhase3Integration:
    """Phase 3 模块集成测试"""

    def test_all_modules_importable(self):
        """所有 Phase 3 模块可导入"""
        from robot_agent import (
            DiagnosisReport,
            DialogueEntry,
            DialogueHistory,
            ErrorDiagnoser,
            GoalPlanner,
            RobotAgent,
            TaskPlan,
        )

        assert GoalPlanner is not None
        assert DialogueHistory is not None
        assert ErrorDiagnoser is not None
        assert RobotAgent is not None
        assert TaskPlan is not None
        assert DialogueEntry is not None
        assert DiagnosisReport is not None

    def test_dataclass_consistency(self):
        """验证数据类一致性"""
        from robot_agent import DiagnosisReport, DialogueEntry, TaskPlan

        # TaskPlan
        plan = TaskPlan(id="t1", skill="fold", goal="test")
        assert plan.status == "pending"
        assert plan.depends_on == []
        assert plan.preconditions == []

        # DialogueEntry
        entry = DialogueEntry(timestamp=time.time(), role="user", content="test")
        assert entry.metadata == {}

        # DiagnosisReport
        report = DiagnosisReport(
            error_type="test",
            possible_causes=["cause1"],
            suggestions=["fix1"],
            confidence=0.8
        )
        assert report.confidence == 0.8

    def test_all_error_patterns_defined(self):
        """验证所有 4 种错误类型都已定义"""
        expected_types = [
            "object_not_visible",
            "adjustment_failed",
            "task_execution_failed",
            "range_not_reached",
        ]

        # 错误类型定义在 SUBSTRING_PATTERNS 中
        for error_type in expected_types:
            assert error_type in ErrorDiagnoser.SUBSTRING_PATTERNS
            pattern = ErrorDiagnoser.SUBSTRING_PATTERNS[error_type]
            assert "keywords" in pattern
            assert "causes" in pattern
            assert "fixes" in pattern
            assert len(pattern["keywords"]) > 0
            assert len(pattern["causes"]) > 0
            assert len(pattern["fixes"]) > 0


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
