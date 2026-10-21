"""EnvironmentScanner - 全局环境扫描器

Phase 5: 全面环境感知模块

职责：
1. 调用摄像头云台进行全景扫描（多角度拍摄 + VLM 分析）
2. 解析 VLM 响应，提取物体及其方位信息
3. 更新世界模型，维护物品位置和置信度
4. 置信度随时间衰减

使用方式：
    scanner = EnvironmentScanner(world_model)
    result = scanner.full_scan()
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .world_model import WorldModel, ObjectState

logger = logging.getLogger("robot_agent.environment_scanner")

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DetectedObject:
    """检测到的物品"""
    label: str
    direction: str  # left, right, front, back, center
    distance: str   # very_close, close, medium, far, very_far
    confidence: float = 1.0
    pan_angle: int = 0  # 云台平移角度
    tilt_angle: int = 0  # 云台俯仰角度
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "direction": self.direction,
            "distance": self.distance,
            "confidence": self.confidence,
            "pan_angle": self.pan_angle,
            "tilt_angle": self.tilt_angle,
            "properties": self.properties,
        }


@dataclass
class ScanResult:
    """扫描结果"""
    success: bool
    objects: list[DetectedObject] = field(default_factory=list)
    vlm_response: str = ""
    updated_objects: int = 0
    new_objects: int = 0
    message: str = ""


# ============================================================================
# 方位常量
# ============================================================================

# 云台角度到方向的映射
PAN_ANGLE_TO_DIRECTION = {
    -80: "left",
    -40: "left",
    0: "front",
    40: "right",
    80: "right",
}

# 距离关键词到距离级别的映射
DISTANCE_KEYWORDS = {
    "很近": "very_close",
    "非常近": "very_close",
    "较近": "close",
    "中等": "medium",
    "中等距离": "medium",
    "较远": "far",
    "很远": "very_far",
    "非常远": "very_far",
    "very close": "very_close",
    "close": "close",
    "medium": "medium",
    "far": "far",
    "very far": "very_far",
}

# 物体名称标准化映射（中文 → 英文/标准化名称）
OBJECT_LABEL_NORMALIZE = {
    "毛巾": "towel",
    "抹布": "towel",
    "rag": "towel",
    "纸巾": "tissue",
    "碗": "bowl",
    "小碗": "bowl",
    "瓶子": "bottle",
    "水瓶": "bottle",
    "bottle": "bottle",
    "筐": "basket",
    "篮子": "basket",
    "筐子": "basket",
    "盘子": "plate",
    "盒子": "box",
    "杯子": "cup",
    "纸箱": "box",
    "箱子": "box",
    "遥控器": "remote",
    "手机": "phone",
    "胶带": "tape",
    "台灯": "lamp",
    "桌": "table",
    "椅子": "chair",
    "床": "bed",
    "电线": "cable",
    "充电器": "charger",
    "bowl": "bowl",
    "towel": "towel",
    "cup": "cup",
    "plate": "plate",
    "box": "box",
    "bag": "bag",
    "cloth": "cloth",
}


# ============================================================================
# EnvironmentScanner
# ============================================================================

class EnvironmentScanner:
    """
    环境扫描器 - 全面环境感知

    功能：
    1. 全景扫描：控制云台拍摄多角度图像，发给 VLM 分析
    2. 物体解析：从 VLM 响应中提取物体和方位
    3. 世界模型更新：维护物品位置和置信度

    使用方式：
        scanner = EnvironmentScanner(world_model)
        result = scanner.full_scan()
        if result.success:
            print(f"检测到 {len(result.objects)} 个物体")
    """

    def __init__(
        self,
        world_model: WorldModel,
        timeout: int = 300,
        panorama_script: str | None = None,
        lerobot_root: str | None = None,
    ):
        """
        Args:
            world_model: 世界模型实例
            timeout: 扫描超时时间（秒）
            panorama_script: 全景扫描脚本路径（可选，默认使用 skills 目录）
            lerobot_root: LeRobot 根目录（可选）
        """
        self.world_model = world_model
        self.timeout = timeout
        self._last_scan_time: float | None = None
        self._legacy_panorama_script = panorama_script
        self._legacy_lerobot_root = lerobot_root

    def full_scan(self, task: Optional[str] = None) -> ScanResult:
        """
        执行完整的环境扫描。

        流程：
        1. 衰减现有物品的置信度
        2. 调用全景扫描 + VLM 分析
        3. 从 stdout 解析 VLM 响应
        4. 解析物体
        5. 更新世界模型

        Args:
            task: 可选的搜索任务（如 "找右边的筐子"）

        Returns:
            ScanResult
        """
        logger.info("[EnvironmentScanner] 开始全景扫描")

        # 1. 衰减现有物品置信度
        if self._last_scan_time is not None:
            elapsed = time.time() - self._last_scan_time
            decay_result = self.world_model.decay_confidence(elapsed)
            logger.info(f"[EnvironmentScanner] 置信度衰减: {len(decay_result['decayed'])} 衰减, {len(decay_result['removed'])} 移除")
        else:
            decay_result = {"decayed": [], "removed": []}

        # 2. 单次调用全景驱动，同时获取角度清单和 VLM 响应。
        from .hardware.camera import CameraClient
        camera = CameraClient()
        camera_result = camera.panorama_scan(task=task, timeout=self.timeout)

        if not camera_result.get("success"):
            return ScanResult(success=False, message=f"全景扫描失败: {camera_result.get('error', '未知错误')}")

        captures = camera_result.get("captures", [])
        logger.info("[EnvironmentScanner] captured %s panorama views", len(captures))
        stdout = camera_result.get("stdout", "")
        vlm_response = self._extract_vlm_from_stdout(stdout)

        if not vlm_response:
            return ScanResult(success=False, message="未获取到 VLM 响应")

        logger.info(f"[EnvironmentScanner] VLM 响应长度: {len(vlm_response)}")

        # 3. 解析 VLM 响应
        detected_objects = self._parse_vlm_response(vlm_response, captures)

        # 4. 更新世界模型
        updated_count, new_count = self._update_world_model(detected_objects)

        # 记录扫描时间
        self._last_scan_time = time.time()

        result = ScanResult(
            success=True,
            objects=detected_objects,
            vlm_response=vlm_response,
            updated_objects=updated_count,
            new_objects=new_count,
            message=f"检测到 {len(detected_objects)} 个物体，更新 {updated_count}，新增 {new_count}"
        )

        logger.info(f"[EnvironmentScanner] 扫描完成: {result.message}")
        return result

    def _extract_vlm_from_stdout(self, stdout: str) -> str:
        """
        从 camera_panorama.py 的 stdout 中提取 VLM 响应。

        格式: "VLM 响应:\n==========\n内容...\n==========\n"

        Args:
            stdout: camera_panorama.py 的标准输出

        Returns:
            VLM 响应文本，如果提取失败则返回空字符串
        """
        if not stdout:
            return ""

        # 查找 VLM 响应标记
        marker = "VLM 响应"
        marker_pos = stdout.find(marker)

        if marker_pos == -1:
            logger.warning("[EnvironmentScanner] 未在 stdout 中找到 VLM 响应标记")
            return ""

        # 从标记后开始提取内容
        # 格式: "VLM 响应:\n==========\n内容...\n==========\n"
        lines = stdout[marker_pos:].split('\n')
        # 跳过第一行（"VLM 响应:"）和第二行（分隔线）
        # 从第三行开始是实际内容，遇到分隔线或日志行停止
        content_lines = []
        skip_next = 2  # 跳过前两行（标题+分隔线）
        for line in lines:
            if skip_next > 0:
                skip_next -= 1
                continue
            stripped = line.strip()
            # 遇到分隔线（全是=）或日志标记则停止
            if stripped.startswith('[INFO]') or stripped.startswith('[ERROR]'):
                break
            if stripped and all(c in '= \t' for c in stripped):
                # 全是 = 符号的行（结尾分隔线），停止
                break
            content_lines.append(line)

        vlm_response = '\n'.join(content_lines).strip()

        if not vlm_response:
            logger.warning("[EnvironmentScanner] VLM 响应内容为空")
            return ""

        logger.debug(f"[EnvironmentScanner] 提取 VLM 响应: {len(vlm_response)} 字符")
        return vlm_response

    def _parse_vlm_response(self, vlm_response: str, captures: list[dict]) -> list[DetectedObject]:
        """
        解析 VLM 响应，提取物体信息。

        支持多种格式：
        1. 表格格式：| 物体 | 位置 | 最佳视角 |
        2. 列表格式：- 纸巾、胶带、遥控器（左侧小桌）
        3. 段落格式：描述性文本中提取物体名和方向

        Args:
            vlm_response: VLM 响应文本
            captures: 捕获的图像列表（包含角度信息）

        Returns:
            检测到的物体列表
        """
        objects = []

        # 建立角度到方向的映射（用于从表格或角度信息推断方向）
        pan_to_direction = {-80: "left", -40: "left", 0: "front", 40: "right", 80: "right"}
        angle_mentioned = set()
        for cap in captures:
            pan = cap.get("pan", 0)
            if pan in pan_to_direction:
                angle_mentioned.add(pan)

        # 从 VLM 响应中提取物体名
        # 常见物品关键词
        object_keywords = [
            "纸巾", "胶带", "遥控器", "纸箱", "箱子",
            "无人机", "机械臂", "设备", "手机", "被子", "台灯",
            "冰箱", "储物柜", "桌", "椅", "床", "电线", "充电器",
            "bowl", "bottle", "towel", "rag", "cup", "plate",
            "box", "bottle", "bag", "towel", "cloth",
        ]


        # 方向关键词 -> 方向
        direction_keywords = [
            ("左", "left"),
            ("右", "right"),
            ("前", "front"),
            ("后", "back"),
            ("中", "center"),
        ]

        # 方法1: 解析表格格式 "| 物体 | 位置 | 视角 |"
        lines = vlm_response.split('\n')
        for line in lines:
            line = line.strip()
            if not line.startswith('|'):
                continue

            # 解析表格单元格
            parts = [p.strip() for p in line.split('|')]
            parts = [p for p in parts if p]  # 去掉空字符串

            if len(parts) < 2:
                continue

            # 第一列: 物体名
            objects_cell = parts[0]
            # 第二列: 位置描述（含方向）
            position_cell = parts[1] if len(parts) > 1 else ""

            # 从位置列提取方向
            direction = "unknown"
            for kw, dir_val in direction_keywords:
                if kw in position_cell:
                    direction = dir_val
                    break

            # 如果没有方向关键词，尝试从云台角度提取（如 "平移-40°~0°"）
            if direction == "unknown":
                pan_match = re.search(r'平移[（(]([+-]?\d+)[°度]', position_cell)
                if not pan_match:
                    pan_match = re.search(r'([+-]?\d+)[°度]', position_cell)
                if pan_match:
                    pan_val = int(pan_match.group(1))
                    if pan_val in pan_to_direction:
                        direction = pan_to_direction[pan_val]

            # 从位置列提取距离
            distance = "medium"
            for kw, dist_val in DISTANCE_KEYWORDS.items():
                if kw in position_cell:
                    distance = dist_val
                    break

            # 从物体列提取已知物品
            found_in_this_row = set()
            for obj_kw in object_keywords:
                if obj_kw in objects_cell:
                    label = self._normalize_label(obj_kw)
                    if label and label != "unknown":
                        found_in_this_row.add(label)
                        obj = DetectedObject(
                            label=label,
                            direction=direction,
                            distance=distance,
                            confidence=1.0,
                        )
                        objects.append(obj)
                        logger.debug(f"[EnvironmentScanner] 解析物体(表格): {label} -> {direction}")

            # 也从物体列提取逗号/顿号分隔的中文词（仅处理可标准化的物品）
            # 只有 seg 本身是 OBJECT_LABEL_NORMALIZE 的 key 时才添加
            for seg in objects_cell.replace("、", ",").replace("，", ",").split(","):
                seg = seg.strip()
                if not seg or len(seg) > 8:
                    continue
                # 跳过含方向词的片段
                if any(kw in seg for kw, _ in direction_keywords):
                    continue
                # 只有可标准化的物品名才添加
                if seg not in OBJECT_LABEL_NORMALIZE:
                    continue
                label = self._normalize_label(seg)
                if label and label != "unknown" and label not in found_in_this_row:
                    obj = DetectedObject(
                        label=label,
                        direction=direction,
                        distance=distance,
                        confidence=1.0,
                    )
                    objects.append(obj)
                    logger.debug(f"[EnvironmentScanner] 解析物体(seg): {label} -> {direction}")

        # 方法2: 从描述性文本中提取（非表格段落）
        # 匹配物体名+位置模式，要求物体名至少包含容器词的一部分
        position_patterns = [
            # 物体名在前，位置在后（物体名至少2字符，防止"纸"单独匹配"纸箱"）
            (r'([^\s，。，。]{2,}[箱包杯瓶碗盘])[^\s，。，。]*(左[前方后侧]+)', 'left'),
            (r'([^\s，。，。]{2,}[箱包杯瓶碗盘])[^\s，。，。]*(右[前方后侧]+)', 'right'),
            (r'([^\s，。，。]{2,}[箱包杯瓶碗盘])[^\s，。，。]*前', 'front'),
            (r'([^\s，。，。]{2,}[箱包杯瓶碗盘])[^\s，。，。]*后', 'back'),
            # 位置在前，物体名在后
            (r'左[侧前方后]+[^\s，。，。]*([^\s，。，。]{2,}[箱包杯瓶碗盘])', 'left'),
            (r'右[侧前方后]+[^\s，。，。]*([^\s，。，。]{2,}[箱包杯瓶碗盘])', 'right'),
        ]

        for pattern, direction in position_patterns:
            matches = re.findall(pattern, vlm_response)
            for match in matches:
                if isinstance(match, tuple):
                    label = self._normalize_label(match[0] if len(match) > 0 else "")
                else:
                    label = self._normalize_label(match)
                if label and label != "unknown":
                    # 检查是否已存在
                    if not any(o.label == label and o.direction == direction for o in objects):
                        obj = DetectedObject(
                            label=label,
                            direction=direction,
                            distance="medium",
                            confidence=1.0,
                        )
                        objects.append(obj)
                        logger.debug(f"[EnvironmentScanner] 解析物体(文本): {label} → {direction}")

        # 去重：同标签优先保留有方向的，同方向只保留第一个
        seen = set()
        unique_objects = []
        for obj in objects:
            key = (obj.label, obj.direction)
            if key not in seen:
                seen.add(key)
                unique_objects.append(obj)
        # 二次去重：同一标签多个方向时，优先保留有方向的
        best_objects: dict[str, DetectedObject] = {}
        for obj in unique_objects:
            existing = best_objects.get(obj.label)
            if existing is None:
                best_objects[obj.label] = obj
            elif obj.direction != "unknown" and existing.direction == "unknown":
                best_objects[obj.label] = obj
        objects = list(best_objects.values())

        return objects

    def _parse_panorama_result(self, captures: list[dict]) -> list[DetectedObject]:
        """
        从全景扫描结果解析物体。

        调用 camera_panorama 后，VLM 响应已包含在结果中。
        这个方法解析 VLM 响应中的物体信息。

        Args:
            captures: 捕获列表

        Returns:
            检测到的物体列表
        """
        # 这里需要从 captures 中提取 VLM 响应
        # 由于 camera_panorama 脚本已经做了 VLM 分析，
        # 我们需要另一种方式获取解析结果

        # 简化方案：使用已知的角度-方向映射
        objects = []
        for cap in captures:
            pan = cap.get("pan", 0)
            tilt = cap.get("tilt", 0)
            direction = PAN_ANGLE_TO_DIRECTION.get(pan, "front")

            # 这里应该根据图像内容识别物体
            # 暂时返回空列表，后续可以通过 VLM 分析获取
            logger.debug(f"[EnvironmentScanner] 角度 {pan}°/{tilt}° → 方向 {direction}")

        return objects

    def _normalize_direction(self, text: str) -> str:
        """标准化方向文本"""
        text = text.lower()
        if "左" in text and "前" in text:
            return "left"
        if "右" in text and "前" in text:
            return "right"
        if "左" in text and "后" in text:
            return "left"
        if "右" in text and "后" in text:
            return "right"
        if "左" in text:
            return "left"
        if "右" in text:
            return "right"
        if "前" in text:
            return "front"
        if "后" in text:
            return "back"
        if "中" in text:
            return "center"
        return "unknown"

    def _normalize_distance(self, text: str) -> str:
        """标准化距离文本"""
        for keyword, level in DISTANCE_KEYWORDS.items():
            if keyword in text:
                return level
        return "unknown"

    def _normalize_label(self, label: str) -> str:
        """标准化物体标签"""
        label = label.strip()
        # 先检查是否在标准化映射中
        for key, value in OBJECT_LABEL_NORMALIZE.items():
            if key in label:
                return value
        # 如果不在映射中，返回小写版本
        return label.lower()

    def _update_world_model(self, detected_objects: list[DetectedObject]) -> tuple[int, int]:
        """
        更新世界模型。

        Args:
            detected_objects: 检测到的物体列表

        Returns:
            (更新的物体数, 新增的物体数)
        """
        updated = 0
        added = 0

        for obj in detected_objects:
            # 生成物体ID（基于标签）
            base_id = obj.label
            counter = 1
            object_id = base_id
            # 使用公共 API 检查 ID 是否存在
            while self.world_model.get_object(object_id) is not None:
                object_id = f"{base_id}_{counter}"
                counter += 1

            # 计算位置（基于方向和距离的简化估计）
            x, y = self._estimate_position(obj.direction, obj.distance)

            # 检查是否已存在相同标签的物体
            existing = self.world_model.get_objects_by_label(obj.label)

            if existing:
                # 更新已存在的物体
                existing_obj = existing[0]
                self.world_model.update_object(
                    existing_obj.object_id,
                    source="environment_scanner",
                    x=x,
                    y=y,
                    confidence=1.0,  # 重新观察，重置置信度
                    last_seen=time.time(),
                    properties={
                        **existing_obj.properties,
                        "direction": obj.direction,
                        "distance": obj.distance,
                        "pan_angle": obj.pan_angle,
                        "tilt_angle": obj.tilt_angle,
                    }
                )
                updated += 1
                logger.debug(f"[EnvironmentScanner] 更新物体: {obj.label} at ({x:.3f}, {y:.3f})")
            else:
                # 添加新物体
                new_obj = ObjectState(
                    object_id=object_id,
                    label=obj.label,
                    x=x,
                    y=y,
                    confidence=1.0,
                    state="on_table",
                    last_seen=time.time(),
                    source="environment_scanner",
                    properties={
                        "direction": obj.direction,
                        "distance": obj.distance,
                        "pan_angle": obj.pan_angle,
                        "tilt_angle": obj.tilt_angle,
                    }
                )
                self.world_model.add_object(new_obj, source="environment_scanner")
                added += 1
                logger.debug(f"[EnvironmentScanner] 添加物体: {obj.label} at ({x:.3f}, {y:.3f})")

        return updated, added

    def _estimate_position(self, direction: str, distance: str) -> tuple[float, float]:
        """
        基于方向和距离估计物品位置。

        这是一个简化估计，实际位置需要更精确的感知。

        Args:
            direction: 方向
            distance: 距离级别

        Returns:
            (x, y) 位置坐标
        """
        # 距离到米数的映射
        distance_meters = {
            "very_close": 0.1,
            "close": 0.2,
            "medium": 0.35,
            "far": 0.5,
            "very_far": 0.8,
            "unknown": 0.3,
        }

        dist = distance_meters.get(distance, 0.3)

        # 方向到角度的映射
        direction_angles = {
            "left": 90,
            "right": -90,
            "front": 0,
            "back": 180,
            "center": 0,
            "unknown": 0,
        }

        angle = direction_angles.get(direction, 0)
        import math
        # 转换为弧度并计算坐标
        rad = math.radians(angle)
        x = dist * math.sin(rad)
        y = dist * math.cos(rad)

        return x, y

    def get_objects_by_direction(self, direction: str) -> list[ObjectState]:
        """
        获取指定方向的物体。

        Args:
            direction: left, right, front, back, center

        Returns:
            物体列表
        """
        result = []
        for obj in self.world_model.get_all_objects():
            obj_direction = obj.properties.get("direction", "unknown")
            if obj_direction == direction:
                result.append(obj)
        return result

    def get_closest_object(self, direction: Optional[str] = None) -> ObjectState | None:
        """
        获取最近的物体。

        Args:
            direction: 可选，指定方向

        Returns:
            最近的物体，如果没有则返回 None
        """
        candidates = self.world_model.get_all_objects()

        if direction:
            candidates = [obj for obj in candidates
                        if obj.properties.get("direction") == direction]

        if not candidates:
            return None

        # 按距离排序（x^2 + y^2）
        return min(candidates, key=lambda o: o.x**2 + o.y**2)
