"""DialogueHistory - 对话历史管理器

Phase 3: 对话历史模块

职责：
1. 记录用户对话和系统响应
2. 支持关键词检索
3. 生成会话摘要
"""

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("robot_agent.dialogue_history")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DialogueEntry:
    """对话条目"""
    timestamp: float
    role: str  # "user" | "assistant" | "system"
    content: str
    metadata: dict = field(default_factory=dict)


# ============================================================================
# DialogueHistory
# ============================================================================

class DialogueHistory:
    """对话历史管理器

    记录用户对话和系统响应，支持跨会话理解。

    使用方法：
        history = DialogueHistory()
        history.add("user", "折叠毛巾")
        history.add("assistant", "执行中...")
        recent = history.get_recent(5)
        results = history.search("毛巾")
        summary = history.summarize()
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_entries: int = 1000
    ):
        """
        Args:
            storage_path: 持久化文件路径（可选）
            max_entries: 最大条目数
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.entries: list[DialogueEntry] = []
        self.max_entries = max_entries

    def add(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> None:
        """
        添加对话条目。

        Args:
            role: 角色 ("user" | "assistant" | "system")
            content: 对话内容
            metadata: 额外元数据
        """
        entry = DialogueEntry(
            timestamp=time.time(),
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.entries.append(entry)

        # 限制条目数量
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

        logger.debug(f"[DialogueHistory] 添加条目: {role} - {content[:50]}...")

    def get_recent(self, n: int = 10) -> list[DialogueEntry]:
        """
        获取最近 n 条对话。

        Args:
            n: 条目数量

        Returns:
            DialogueEntry 列表
        """
        return self.entries[-n:] if self.entries else []

    def search(self, query: str, limit: int = 5) -> list[DialogueEntry]:
        """
        关键词搜索对话历史。

        Args:
            query: 搜索关键词
            limit: 返回结果数量上限

        Returns:
            匹配的 DialogueEntry 列表
        """
        results = []
        query_lower = query.lower()
        for entry in reversed(self.entries):
            if query_lower in entry.content.lower():
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def summarize(self) -> str:
        """
        生成会话摘要。

        Returns:
            摘要字符串
        """
        if not self.entries:
            return "无对话历史"

        user_msgs = [e.content for e in self.entries if e.role == "user"]
        task_ids = [e.metadata.get("task_id") for e in self.entries if e.metadata.get("task_id")]

        summary_parts = [
            f"对话 {len(self.entries)} 条",
            f"用户消息 {len(user_msgs)} 条",
        ]
        if task_ids:
            unique_tasks = set(filter(None, task_ids))
            if unique_tasks:
                summary_parts.append(f"执行任务: {', '.join(unique_tasks)}")

        return "，".join(summary_parts)

    def get_user_goals(self) -> list[str]:
        """
        获取所有用户目标。

        Returns:
            用户目标列表
        """
        return [e.content for e in self.entries if e.role == "user"]

    def get_failed_tasks(self) -> list[dict]:
        """
        获取所有失败的任务记录。

        Returns:
            失败任务列表 [{"goal": str, "timestamp": float}, ...]
        """
        failed = []
        for entry in self.entries:
            if entry.role == "assistant" and "失败" in entry.content:
                # 尝试找到对应的用户目标
                for prev in reversed(self.entries[:self.entries.index(entry)]):
                    if prev.role == "user":
                        failed.append({
                            "goal": prev.content,
                            "timestamp": entry.timestamp,
                            "result": entry.content
                        })
                        break
        return failed
