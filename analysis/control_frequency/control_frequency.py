#!/usr/bin/env python3
"""Effective control frequency of fixed-duration real-robot evaluation recordings.

    effective_hz = recorded_frames / (episodes * duration_s)

The LeRobot record loop writes one frame per control step. With a fixed
recording duration, a slower policy stalls the loop more often and leaves fewer
frames. The duration is the configured ``--dataset.episode_time_s`` of the
recording run; it is not stored in the dataset and has to be passed here.

Example:
    python control_frequency.py \
        --run SmolVLA PyTorch       /data/eval_smolvla_pick_up_bottle_three_py_20s  20 \
        --run SmolVLA "TensorRT FP16" /data/eval_smolvla_pick_up_bottle_three_trt_20s 20 \
        --out-dir .

The first run of each model is the baseline for the improvement column.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def load_frames(root: Path) -> tuple[dict, pd.DataFrame]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {root / 'data'}")
    cols = ["episode_index", "frame_index", "timestamp", "task_index"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    return info, df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=4, action="append", required=True,
                    metavar=("MODEL", "BACKEND", "DATASET_DIR", "DURATION_S"),
                    help="one evaluation recording; repeat for each run")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    rows = []
    baseline_hz: dict[str, float] = {}
    for model, backend, path, duration in args.run:
        root = Path(path)
        info, df = load_frames(root)
        duration_s = float(duration)
        if duration_s <= 0:
            ap.error(f"duration must be positive: {duration}")
        n_frames = len(df)
        n_episodes = df["episode_index"].nunique()
        eff_hz = n_frames / (n_episodes * duration_s)
        if model not in baseline_hz:
            baseline_hz[model] = eff_hz
            improvement = None
        else:
            improvement = eff_hz / baseline_hz[model] - 1.0
        rows.append({
            "model": model,
            "backend": backend,
            "dataset": root.name,
            "episodes": n_episodes,
            "duration_s": duration_s,
            "frames": n_frames,
            "target_fps": info.get("fps"),
            "effective_hz": round(eff_hz, 4),
            "fraction_of_target": round(eff_hz / info["fps"], 4) if info.get("fps") else None,
            "improvement_vs_first_run_of_model": None if improvement is None else round(improvement, 4),
            "last_timestamp_s": round(float(df["timestamp"].max()), 4),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "control_frequency.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    with open(args.out_dir / "control_frequency.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        imp = r["improvement_vs_first_run_of_model"]
        imp_s = "baseline" if imp is None else f"{imp * 100:+.1f}%"
        print(f"{r['model']:8s} {r['backend']:14s} frames={r['frames']:4d} T={r['duration_s']:g}s "
              f"eff={r['effective_hz']:.2f} Hz (target {r['target_fps']} Hz) {imp_s}")


if __name__ == "__main__":
    main()
