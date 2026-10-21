# XLeRobot ROS 2 Workspace

The ROS 2 workspace currently contains only interfaces supported by the known hardware boundary:

- the three-wheel base bridge and navigation bringup scaffolding;
- the RGB segmentation adapter;
- `InstanceMask` and `SegmentationFrame` message contracts.

MoveIt task plugins, RGB-D geometry nodes, force/torque HIL nodes, and manipulation action messages
have been removed. The RGB segmentation output is 2D evidence only and must not be treated as metric
geometry.

The optional `rgb_world_model_bridge` subscribes to `SegmentationFrame` and persists qualitative
RGB observations when started explicitly with an absolute `world_model_path`. It is intentionally
absent from the default navigation launch.
