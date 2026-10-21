# XLeRobot Segmentation Server

This independent wheel runs the reviewed local HTTP v1 boundary between the robot and an
open-vocabulary segmentation model. It contains no Torch, CUDA or model implementation.

Model packages register a provider entry point:

```toml
[project.entry-points."xlerobot.segmentation_providers"]
grounded_sam2_v1 = "my_provider.backend:create_provider"
```

The factory receives `options`, absolute `artifacts`, `server_id`, `model_sha256` and
`allowed_labels`. Its `infer(request)` method returns a list of `SegmentationInstance` values.
The server rejects undeclared labels, duplicate IDs, invalid confidence, out-of-image ROIs and
non-binary masks before they cross the HTTP boundary.

```bash
xlerobot-segmentation-serve --config segmentation-server.json
xlerobot-segmentation-conformance \
  --endpoint http://127.0.0.1:8090/v1/segment \
  --manifest segmentation-server-manifest.json \
  --token segmentation-server.token \
  --label cup --prompt "drinking cup"
```

The example config is default-deny. Replace every digest, set `approved` only after review, and
run conformance plus an RGB-only evaluation on recorded target-camera data before use.
