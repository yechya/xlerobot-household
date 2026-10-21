"""ErrorDiagnoser - 错误诊断器

Phase 3: 错误诊断模块

职责：
1. 分析执行失败的原因
2. 生成诊断报告
3. 提供修复建议
"""

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .execution_types import ExecutionStep
    from .world_model import WorldModel

logger = logging.getLogger("robot_agent.error_diagnoser")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DiagnosisReport:
    """诊断报告"""
    error_type: str
    possible_causes: list[str]
    suggestions: list[str]
    confidence: float  # 0.0 - 1.0


# ============================================================================
# ErrorDiagnoser
# ============================================================================

class ErrorDiagnoser:
    """错误诊断器

    根据错误症状匹配错误模式，生成诊断报告和修复建议。

    使用方法：
        diagnoser = ErrorDiagnoser()
        report = diagnoser.diagnose(
            failed_step=step,
            world_model=wm,
            vlm_response=vlm_text
        )
        print(f"错误类型: {report.error_type}")
        print(f"可能原因: {report.possible_causes}")
        print(f"修复建议: {report.suggestions}")
    """

    # 错误模式库（分为子串匹配和正则匹配）
    # 子串模式：精确匹配的关键词（用于快速匹配）
    SUBSTRING_PATTERNS = {
        "object_not_visible": {
            "keywords": ["观察失败", "未检测到", "看不到", "无法识别", "VLM 失败"],
            "causes": ["物品被遮挡", "光照不足", "物品不在视野内"],
            "fixes": ["调整机器人位置", "改善光照", "扩大搜索范围"]
        },
        "adjustment_failed": {
            "keywords": ["底盘失败", "移动失败", "调整超时", "导航失败", "震荡"],
            "causes": ["障碍物阻挡", "地形不平", "电机故障", "VLM 方向描述不一致"],
            "fixes": ["清除障碍物", "检查电机状态", "手动辅助定位", "重新校准 VLM"]
        },
        "task_execution_failed": {
            "keywords": ["VLA 失败", "执行失败", "动作无法完成", "任务执行失败", "夹爪无法"],
            "causes": ["物品位置偏差", "夹爪问题", "任务定义不清晰"],
            "fixes": ["重新观察调整", "检查夹爪", "简化任务目标"]
        },
        "range_not_reached": {
            "keywords": ["最大迭代", "无法到达范围"],
            "causes": ["物品距离太远", "VLM 描述不准确", "底盘移动受限"],
            "fixes": ["手动移动机器人靠近", "重新标定 VLM", "分多次小范围移动"]
        }
    }

    # 正则模式：用于更灵活的匹配（需要 import re）
    REGEX_PATTERNS = {
        "object_not_visible": [
            r"VLM.*失败",
            r"观察.*失败",
            r"未检测到",
            r"看不到",
            r"无法.*识别",
        ],
        "adjustment_failed": [
            r"底盘.*失败",
            r"移动.*失败",
            r"调整超时",
            r"导航失败",
            r"震荡",
        ],
        "task_execution_failed": [
            r"VLA.*失败",
            r"执行.*失败",
            r"动作无法完成",
            r"任务执行失败",
            r"夹爪.*无法",
        ],
        "range_not_reached": [
            r"最大迭代",
            r"无法到达范围",
            r"达到.*次",
        ]
    }

    def diagnose(
        self,
        failed_step: "ExecutionStep",
        world_model: "WorldModel",
        vlm_response: str
    ) -> DiagnosisReport:
        """
        诊断执行失败的原因。

        Args:
            failed_step: 失败的执行步骤
            world_model: 当前世界状态
            vlm_response: 最后的 VLM 响应

        Returns:
            DiagnosisReport
        """
        action = failed_step.action
        message = failed_step.message

        # 匹配错误模式
        matched_pattern = self._match_pattern(action, message)

        # 分析可能原因
        possible_causes = self._analyze_causes(matched_pattern, world_model, vlm_response)

        # 生成建议
        suggestions = self._generate_suggestions(possible_causes)

        report = DiagnosisReport(
            error_type=matched_pattern,
            possible_causes=possible_causes,
            suggestions=suggestions,
            confidence=self._calculate_confidence(possible_causes)
        )

        logger.info(
            f"[ErrorDiagnoser] 诊断结果: {matched_pattern}, "
            f"原因: {possible_causes}, 建议: {suggestions}"
        )

        return report

    def _match_pattern(self, action: str, message: str) -> str:
        """
        匹配错误模式。

        匹配策略：
        1. 首先尝试子串精确匹配（快速）
        2. 然后尝试正则表达式匹配（灵活）
        3. 都没有匹配则返回 unknown_error
        """
        combined_text = f"{action} {message}".lower()

        # 1. 子串精确匹配
        for pattern_name, pattern in self.SUBSTRING_PATTERNS.items():
            for keyword in pattern["keywords"]:
                if keyword.lower() in combined_text:
                    return pattern_name

        # 2. 正则表达式匹配
        for pattern_name, regex_list in self.REGEX_PATTERNS.items():
            for regex in regex_list:
                if re.search(regex, combined_text, re.IGNORECASE):
                    return pattern_name

        return "unknown_error"

    def _analyze_causes(
        self,
        pattern: str,
        world_model: "WorldModel",
        vlm_response: str
    ) -> list[str]:
        """分析可能原因"""
        if pattern == "unknown_error":
            return ["未知错误，需要人工介入"]
        return list(self.SUBSTRING_PATTERNS[pattern]["causes"])

    def _generate_suggestions(self, causes: list[str]) -> list[str]:
        """生成修复建议"""
        suggestions = []
        for cause in causes:
            for _pattern_name, pattern in self.SUBSTRING_PATTERNS.items():
                if cause in pattern["causes"]:
                    suggestions.extend(pattern["fixes"])
        # 去重，保持顺序
        seen = set()
        unique = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    def _calculate_confidence(self, causes: list[str]) -> float:
        """计算诊断置信度"""
        if not causes or causes == ["未知错误，需要人工介入"]:
            return 0.0
        # 原因越多，置信度越低（诊断越不精确）
        return min(1.0, 0.5 + 0.1 * (3 - len(causes)))
