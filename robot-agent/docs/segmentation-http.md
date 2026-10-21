# Local HTTP Segmentation Plugin

`plugins/xlerobot_segmentation_http` 是独立客户端 wheel，用于把运行 Grounded-SAM、SAM2、YOLO-Seg 或其他开源模型的本机 GPU 服务接入 `xlerobot.segmentation_backends`。`plugins/xlerobot_segmentation_server` 是与之配套的模型无关服务 wheel；两者都不把 Torch/CUDA 加入机器人核心环境。

```text
ROS segmentation adapter
        |
xlerobot-segmentation-http client
        |  loopback HTTP v1
xlerobot-segmentation-server
        |
xlerobot.segmentation_providers entry point
        |
Grounded-SAM2 / SAM2 / YOLO-Seg / future provider wheel
```

服务端启动时严格绑定 provider distribution/version、模型文件、provider wheel、lockfile、manifest 和 token 摘要。仓库已经提供 `xlerobot-segmentation-grounded-sam2` provider：它逐 label 使用 GroundingDINO 生成候选框，类内 NMS 后交给 SAM2 生成 mask，避免组合 caption 的 phrase 到 catalog label 映射歧义。模型 provider 只接收审核后的 label/prompt 和图像，并返回 `SegmentationInstance`；服务会在发送前拒绝未请求 label、重复 ID、越界 ROI、非法置信度和非二值 mask。

## Trust Boundary

- endpoint 只能是 `http://127.0.0.1:<port>/v1/segment`、`localhost` 或 IPv6 loopback；
- 禁止 HTTPS 远端、userinfo、query、fragment、非固定 path 和 HTTP redirect；
- server manifest 固定 server ID、model SHA-256 与允许 label；
- 可选 bearer token 必须来自已哈希绑定的 artifact；
- plugin wheel、服务 lockfile、模型清单和 token 都应列入 segmentation config artifacts；
- response 必须回显 server ID 与 model digest，且受 content type、字节数、字段和 mask base64 严格限制。

这不是远程模型证明或硬件可信执行。部署者仍需保证 loopback 服务进程、模型文件和容器镜像由受控发布流程启动；不可信本机 root 可伪造任何用户态证据。

## Request V1

```json
{
  "schema_version": 1,
  "server_id": "grounded_sam_local_v1",
  "model_sha256": "<sha256>",
  "captured_at": 10.25,
  "frame_id": "camera_color_optical_frame",
  "width": 640,
  "height": 480,
  "encoding": "rgb8",
  "step": 1920,
  "queries": {"remote": "remote control"},
  "image_base64": "..."
}
```

## Response V1

```json
{
  "schema_version": 1,
  "server_id": "grounded_sam_local_v1",
  "model_sha256": "<sha256>",
  "instances": [
    {
      "instance_id": "remote-1",
      "label": "remote",
      "confidence": 0.91,
      "x_offset": 120,
      "y_offset": 80,
      "width": 90,
      "height": 45,
      "mask_base64": "..."
    }
  ]
}
```

`mask_base64` 解码后必须恰好是 `width * height` 字节，每字节为 `0` 或 `1`。HTTP 插件只解析传输；核心 adapter 会再次检查 label、confidence、ROI、mask、实例数量和推理总时限。

## Build and Configure

```bash
python -m pip wheel plugins/xlerobot_segmentation_http --no-deps -w dist
python -m pip wheel plugins/xlerobot_segmentation_server --no-deps -w dist
python -m pip wheel plugins/xlerobot_segmentation_grounded_sam2 --no-deps -w dist
sha256sum dist/xlerobot_segmentation_http-*.whl \
  dist/xlerobot_segmentation_server-*.whl \
  dist/xlerobot_segmentation_grounded_sam2-*.whl \
  segmentation-server-manifest.json segmentation-server.token segmentation-server.lock
```

按 `config/grounded-sam2-model-bundle.example.json`、`config/segmentation-server.example.json` 与 `config/segmentation-backend.http.example.yaml` 生成审核配置。Grounded-SAM2 使用两套 checkpoint，因此 `model_artifact_index` 指向组合模型 manifest；该 manifest 的摘要就是对外 `model_sha256`，内部再绑定 SAM2 checkpoint、GroundingDINO config/checkpoint 和三个上游完整 Git revision。示例中的全零摘要和 `approved: false` 不能启动。

模型发行包通过独立 entry point 接入，因此替换模型不修改 ROS、HTTP 协议或几何代码：

```toml
[project.entry-points."xlerobot.segmentation_providers"]
grounded_sam2_v1 = "my_provider.backend:create_provider"
```

服务启动后先运行黑盒一致性检查：

```bash
xlerobot-segmentation-conformance \
  --endpoint http://127.0.0.1:8090/v1/segment \
  --manifest segmentation-server-manifest.json \
  --token segmentation-server.token \
  --label cup --prompt "drinking cup" \
  --output segmentation-conformance.json
```

一致性检查验证合法请求、服务/模型身份、label、ROI、置信度、二值 mask 和畸形请求拒绝。它只证明协议实现正确，不证明模型精度；模型服务必须先用三路 RGB 实机样本做离线精度评测和人工复核，不能直接用于真机解锁。

服务端对 provider 推理实行单飞：前一帧仍在运行时，新请求立即返回 busy，避免 SAM2 `set_image` 等有状态 predictor 被并发图像污染。provider 超过服务 deadline 时响应会被拒绝，但 Python 线程无法强制终止任意 CUDA kernel。生产部署必须用 systemd/container watchdog 限制进程资源并在连续超时后重启服务；机器人侧客户端 deadline、stale perception guard 和 planning-scene 阻塞仍是独立保护。
