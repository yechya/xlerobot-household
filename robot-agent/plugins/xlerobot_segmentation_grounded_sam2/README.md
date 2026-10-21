# XLeRobot Grounded-SAM2 Provider

This independent provider combines GroundingDINO boxes with SAM2 image masks. It plugs into
`xlerobot-segmentation-server` through `xlerobot.segmentation_providers:grounded_sam2_v1` and is
never imported by the robot core.

The provider runs GroundingDINO once per declared label. This is deliberately slower than one
combined caption, but preserves an unambiguous `catalog label -> prompt -> prediction` mapping.
Class-local NMS removes duplicate boxes before SAM2, and final confidence is the product of the
detector and mask scores.

## Reviewed Assets

Populate `config/grounded-sam2-model-bundle.example.json` with the real SHA-256 of:

- the SAM2.1 checkpoint;
- the exact GroundingDINO Python model config;
- the GroundingDINO checkpoint;
- full Git commits for Grounded-SAM-2, GroundingDINO and SAM2.

Then hash that bundle manifest. Its digest is the `model_sha256` exposed by the HTTP service. The
server config binds the bundle, both checkpoints, config, provider wheel and complete environment
lockfile independently.

The three inspected upstream repositories use Apache-2.0 at the revisions recorded in the example
bundle. Deployment review must still preserve their notices and verify the terms and provenance of
the selected checkpoint download itself.

## Environment

Build this wheel without pulling GPU dependencies into the core environment:

```bash
uv build plugins/xlerobot_segmentation_grounded_sam2 --out-dir dist
```

Create a dedicated GPU virtual environment or container. Pin the three upstream revisions from the
bundle and generate a complete `uv.lock`/container digest for the deployed CUDA and PyTorch stack.
Do not use floating Git branches in production.

After starting the local service, run `xlerobot-segmentation-conformance`, then the target-camera
three-camera RGB evaluation. Protocol conformance alone is not a model-quality approval.
