# Pluggable Segmentation Backends

感知模型通过 Python entry point 接入，核心运行时不依赖 Grounded-SAM、YOLO、Torch、CUDA 或特定权重格式。稳定边界是 `xlerobot_manipulation_msgs/SegmentationFrame`；模型只能输出 label、confidence 和二值 ROI mask，不能输出或暗示深度、3D 位姿、接触状态或碰撞安全。

## Plugin Contract

模型发行包在 `pyproject.toml` 注册：

```toml
[project.entry-points."xlerobot.segmentation_backends"]
open_vocab_v1 = "my_segmentation.backend:create_backend"
```

factory 接收审核后的 options 与模型资产绝对路径：

```python
def create_backend(*, options, artifacts):
    return Backend(options, artifacts)

class Backend:
    def infer(self, request):
        return [MaskPrediction(...)]
```

`request` 是不依赖 ROS/OpenCV 的 `SegmentationInput`，包含原始 image bytes、encoding、step、时间、frame 和 `label -> prompt` 查询映射。后端返回 `MaskPrediction`；mask 必须是 ROI 内逐像素 `0/1` 数据。

## Reviewed Configuration

`segmentation-backend.yaml` 严格绑定：

- backend ID、entry point、Python distribution 名称与版本；
- source ID、输入图像 topic、输出 segmentation topic 和 camera frame；
- 任务 label 到模型 prompt 的映射；
- 最低置信度、最大实例数、最大图像大小、输入年龄和推理 deadline；
- 模型权重、词表、TensorRT engine、插件 wheel/lockfile 等全部运行资产的 SHA-256；
- 有界 JSON options。

模板位于 `config/segmentation-backend.example.yaml`。生产配置必须 `approved: true`，且建议把模型 wheel 和依赖 lockfile 一并列入 `artifacts`。加载时 distribution/version、入口唯一性和所有资产摘要必须匹配。

## Runtime Safety

适配节点拒绝：

- 过期、未来时间、错误 camera frame、异常尺寸或超限 image bytes；
- 未声明 label、重复 instance ID、越界 ROI、非二值或尺寸错误 mask；
- 超过实例上限或推理 deadline 的输出；
- 发行包名称、版本、entry point 或资产摘要不匹配。

低于配置置信度的实例会被过滤，provenance 由适配器覆盖为审核 source，模型不能自行伪造。mask 只用于 RGB 观察、跟踪和任务证据，不进入几何运动规划。

## Task Extensibility

新增刚性物体任务时，只增加 task catalog 的 scene role 和 segmentation config 的 query，不修改 `robotd` 或几何节点。manipulation preflight 会计算所有 manipulation task 所需 label；缺少任何 query 都拒绝发布。

```text
new task profile
      + scene roles/labels
      + label -> prompt query
      + geometry benchmark samples
      + postcondition evidence
      = no new end-to-end VLA dataset by default
```

只有模型、几何、导航停靠和夹爪能力均通过门禁后仍稳定失败的接触任务，才进入 learned-policy 数据队列。

仓库提供独立的 loopback HTTP 参考插件，便于把 GPU 模型服务与 ROS/控制环境隔离，详见 [segmentation-http.md](segmentation-http.md)。

模型服务本身也采用 `xlerobot.segmentation_providers` entry point。新增 Grounded-SAM2、SAM2 或其他实现时新增独立 provider wheel 并绑定其版本和资产摘要，不修改核心 adapter、HTTP client 或任务目录。这使模型升级和任务扩展成为两个互不耦合的发布维度。
