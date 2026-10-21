# RGB Policy Data Flywheel

Every real learned-policy run can emit an immutable episode manifest with task, policy binding,
environment tags, result, and attached RGB media hashes. The same format is used for all four
current tasks.

Failure triage should separate observable RGB problems from policy problems:

- camera unavailable, stale image, wrong view, or ambiguous precondition -> system fix;
- model artifact mismatch or runtime failure -> deployment fix;
- valid three-camera observation and repeated task failure -> dataset candidate;
- successful execution with failed visual postcondition -> review before labeling.

The replay runtime supports evidence export and RGB video clipping. Removed RGB-D and rosbag
geometry handlers are not part of this pipeline.

New datasets should be prioritized by repeated failure slices across lighting, placement, object
appearance, and camera occlusion rather than by raw episode count.

For an optional offline value-learning round, export the verified manifests through the Evo-RL
bridge described in [evo-rl-integration.md](evo-rl-integration.md). The bridge keeps Evo-RL out of
the real-time daemon and does not fabricate frame-level policy actions.

## RGB/VLA Comparison

`robot-experiment-evaluate` produces a hashed report for a controlled comparison of:

- `direct_vla`: the existing policy path without RGB pre/post checks;
- `rgb_precondition_vla`: RGB precondition evidence before policy execution;
- `rgb_pre_post_vla`: RGB precondition and postcondition evidence.

Start from [../config/experiment.example.yaml](../config/experiment.example.yaml), replace the task
and policy fingerprints, and evaluate an episode directory:

```bash
robot-experiment-evaluate config/experiment.example.yaml /var/lib/robot-agent/episodes \
  --output /var/lib/robot-agent/experiments/pick-up-bottle.json
```

The runtime labels newly recorded episode manifests with `ROBOT_AGENT_EXPERIMENT_MODE`, which
defaults to `rgb_pre_post_vla`. The label is episode metadata only: it is deliberately excluded
from the robotd authorization token and policy binding. Set it explicitly when collecting a real
ablation arm. A report fails when any configured mode lacks the required sample count or when
there is no real `direct_vla` baseline. Do not manufacture baseline manifests to make an
experiment pass.
