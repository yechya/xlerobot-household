#!/usr/bin/env python3
"""Smoothness metrics for evaluation runs recorded with different action-smoothing windows.

Metrics (servo position units, averaged over all joints and frames):
    command_change_rate  mean |a[t] - a[t-1]|               a = commanded joint position (action)
    joint_acceleration   mean |s[t] - 2 s[t-1] + s[t-2]|    s = measured joint position (observation.state)
    tracking_error       mean |a[t] - s[t]|
    command_second_diff  mean |a[t] - 2 a[t-1] + a[t-2]|    (extra, not used in the paper table)

Two input forms are accepted and can be combined:
    --rrd LABEL PATH            rerun recording written by `lerobot-record --display_data=true`
                                (needs rerun-sdk; the original numbers were computed from these files)
    --episode LABEL DATASET EP  one episode of a LeRobot v3 dataset

Example:
    python smoothing_metrics.py --rrd 1 window_1.rrd --rrd 3 window_3.rrd --rrd 5 window_5.rrd \
        --episode 1 /data/eval_smolvla_pick_up_bottle 0 --out-dir .
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = [f"{side}_{j}" for side in ("left", "right")
          for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


def load_rrd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Commands and measured positions from one rerun recording, aligned on the frame index.

    Rows where any of the 24 scalars is missing are dropped from both arrays together."""
    import rerun as rr  # rerun-sdk 0.22 dataframe API

    rec = rr.dataframe.load_recording(str(path))
    index_names = [c.name for c in rec.schema().index_columns()]
    index = "frame_index" if "frame_index" in index_names else index_names[0]
    paths = [f"/{kind}/{j}" for kind in ("action", "observation/state") for j in JOINTS]
    tbl = rec.view(index=index, contents={p: ["Scalar"] for p in paths}).select().read_all()
    cols = []
    for p in paths:
        name = next(c for c in tbl.column_names if c.startswith(p + ":") and c.endswith(":Scalar"))
        cols.append(np.array([v[0] if v is not None and len(v) else np.nan
                              for v in tbl.column(name).to_pylist()], dtype=np.float64))
    arr = np.stack(cols, axis=1)
    arr = arr[~np.isnan(arr).any(axis=1)]
    return arr[:, :12], arr[:, 12:]


def load_episode(root: Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {root / 'data'}")
    cols = ["action", "observation.state", "episode_index", "frame_index"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df = df[df["episode_index"] == episode].sort_values("frame_index")
    if df.empty:
        raise SystemExit(f"episode {episode} not found in {root}")
    return (np.stack(df["action"].to_numpy()).astype(np.float64),
            np.stack(df["observation.state"].to_numpy()).astype(np.float64))


def metrics(a: np.ndarray, s: np.ndarray) -> dict:
    if a.shape != s.shape or len(a) < 3:
        raise SystemExit(f"need aligned command/state arrays with >= 3 frames, got {a.shape} and {s.shape}")
    return {
        "frames": len(a),
        "command_change_rate": float(np.abs(np.diff(a, axis=0)).mean()),
        "joint_acceleration": float(np.abs(np.diff(s, 2, axis=0)).mean()),
        "tracking_error": float(np.abs(a - s).mean()),
        "command_second_diff": float(np.abs(np.diff(a, 2, axis=0)).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rrd", nargs=2, action="append", default=[], metavar=("WINDOW", "PATH"))
    ap.add_argument("--episode", nargs=3, action="append", default=[], metavar=("WINDOW", "DATASET_DIR", "EPISODE"))
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--prefix", default="smoothing_metrics", help="output file name prefix")
    args = ap.parse_args()
    if not args.rrd and not args.episode:
        ap.error("give at least one --rrd or --episode")

    rows, arrays = [], {}
    for window, path in args.rrd:
        a, s = load_rrd(Path(path))
        arrays[("rrd", window)] = (a, s)
        rows.append({"window": int(window), "source": "rrd", "input": Path(path).name, **metrics(a, s)})
    for window, root, ep in args.episode:
        a, s = load_episode(Path(root), int(ep))
        arrays[("episode", window)] = (a, s)
        rows.append({"window": int(window), "source": "lerobot_dataset",
                     "input": f"{Path(root).name}#episode{ep}", **metrics(a, s)})

    # When the same window is given in both forms, record how far apart the two inputs are.
    for (kind, window), (a, s) in arrays.items():
        if kind != "rrd" or ("episode", window) not in arrays:
            continue
        a2, s2 = arrays[("episode", window)]
        same_shape = a.shape == a2.shape and s.shape == s2.shape
        max_diff = float(max(np.abs(a - a2).max(), np.abs(s - s2).max())) if same_shape else None
        for r in rows:
            if r["window"] == int(window):
                r["rrd_vs_dataset_same_shape"] = same_shape
                r["rrd_vs_dataset_max_abs_diff"] = max_diff

    for r in rows:
        for k in ("command_change_rate", "joint_acceleration", "tracking_error", "command_second_diff"):
            r[k] = round(r[k], 6)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.prefix}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out_dir / f"{args.prefix}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(f"window={r['window']} {r['source']:15s} frames={r['frames']:4d} "
              f"change_rate={r['command_change_rate']:.3f} joint_acc={r['joint_acceleration']:.3f} "
              f"tracking={r['tracking_error']:.3f} max_abs_diff_rrd_vs_dataset={r.get('rrd_vs_dataset_max_abs_diff', '-')}")


if __name__ == "__main__":
    main()
