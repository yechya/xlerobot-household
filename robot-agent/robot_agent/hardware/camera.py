"""Camera Client - 摄像头客户端"""

import json
import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from ..camera_rig import mjpeg_urls_from_env, snapshot_urls_from_env

logger = logging.getLogger("hardware.camera")

# 日志输出标记（用于从 stdout 中提取信息）
STDOUT_VLM_START = "__VLM_RESPONSE_START__"
STDOUT_VLM_END = "__VLM_RESPONSE_END__"


class CameraClient:
    """摄像头客户端

    职责：调用 camera_panorama.py 脚本，返回原始输出。
    VLM 响应解析由调用方（EnvironmentScanner）负责。
    """

    def __init__(self):
        self.command = tuple(shlex.split(os.environ.get("ROBOT_AGENT_PANORAMA_COMMAND", "")))

    def panorama_scan(self, task: str | None = None, timeout: int = 180) -> dict:
        """
        执行全景扫描。

        Args:
            task: 任务描述
            timeout: 超时时间（秒）

        Returns:
            扫描结果字典，包含 success, stdout, stderr
        """
        logger.info(f"[CameraClient] panorama_scan: task={task}")

        if not self.command:
            return {
                "success": False,
                "error": "ROBOT_AGENT_PANORAMA_COMMAND is not configured",
            }

        manifest_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as manifest:
                manifest_path = Path(manifest.name)
            cmd = [*self.command, "--output-json", str(manifest_path)]
            if task:
                cmd.extend(["--task", task])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            stdout = result.stdout if result.stdout else ""
            stderr = result.stderr if result.stderr else ""

            payload = {
                "success": result.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
            }
            if manifest_path.exists() and manifest_path.stat().st_size:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                payload["captures"] = manifest_data.get("angles", [])
            return payload

        except subprocess.TimeoutExpired:
            logger.error("[CameraClient] 全景扫描超时")
            return {"success": False, "error": "全景扫描超时"}
        except Exception as e:
            logger.error(f"[CameraClient] 全景扫描异常: {e}")
            return {"success": False, "error": str(e)}
        finally:
            if manifest_path is not None:
                manifest_path.unlink(missing_ok=True)

    def get_camera_urls(self) -> dict:
        """获取摄像头 URL"""
        mjpeg_urls = mjpeg_urls_from_env()
        return {
            "snapshots": snapshot_urls_from_env(),
            "mjpeg": mjpeg_urls,
            "head_mjpeg": mjpeg_urls["head_cam"],
        }
