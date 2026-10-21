# RGB Observation World Model

The world model stores task observations and event history for planning and debugging. With the
current hardware, camera-derived state is qualitative: labels, confidence, view source, timestamp,
room hints, and relationships that are directly visible in RGB.

Unstructured VLM prose does not mutate state. Structured observations must identify their source and
capture time. Metric 3D poses, sizes, collision geometry, and contact state are not accepted as facts
unless a future verified sensor interface provides them.

The world model supports replayable updates and stale-write rejection. It does not authorize robot
motion; task authorization remains bound to the task catalog, policy registry, and fresh RGB
evidence.

## RGB Segmentation Fusion

`rgb_segmentation_world.fuse_segmentation_frame` consumes the existing segmentation message
contract and appends a hashed observation-fusion event. It stores label, confidence, camera frame,
2D ROI, mask digest, and source provenance. It deliberately leaves `x`, `y`, and `z` at zero and
does not infer reachability, distance, or contact. Empty frames do not delete existing objects;
duplicate observation IDs and older frames are rejected.
