#!/usr/bin/env python3
"""Jitter analysis of SmolVLA real-robot evaluation episodes (three-layer error model evidence).

Input: a LeRobot v3 dataset with `action` (commanded joint positions) and
`observation.state` (measured joint positions), 12 joints in the order
left arm 0-5, right arm 6-11 (checked against meta/info.json). Values are servo position
units as stored by LeRobot.

Computed quantities:

1. Correlation (one episode, right arm, joints 6-11):
     e[t]  = mean_j |a[t,j] - s[t,j]|                    tracking error
     d1[t] = mean_j |a[t,j] - a[t-1,j]|                  command change rate, t >= 1
     d2[t] = mean_j |a[t,j] - 2a[t-1,j] + a[t-2,j]|      command second difference, t >= 2
     m[t]  = mean_j a[t,j]                               command amplitude (signed mean)
   Pearson r of e against d1, d2 and m over the frames where both are defined.
2. Stable vs abrupt frames: d1 < threshold (0.5) vs >= threshold; mean and max of e.
3. Phases: the episode split into thirds (n//3 boundaries). Per phase: mean of d1 (d1[0] = 0),
   std of m, mean of e.
4. Chunk start vs end: complete chunks of `chunk_size` frames from frame 0. Within each chunk,
   acc[k] = sum over right-arm joints |a[k+2] - 2a[k+1] + a[k]|, k = 0 .. chunk_size-3.
   Mean of the first `edge_frames` values (they span chunk frames 0..edge_frames+1) vs the last
   `edge_frames` values (frames chunk_size-edge_frames-2 .. chunk_size-1), averaged over chunks.
5. Gripper: fraction of all frames whose right-gripper command is below `fine_threshold`; tracking
   error of the right gripper in both groups.

Example:
    python jitter_analysis.py --dataset /data/eval_smolvla_pick_up_bottle --out-dir . --plot
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

RIGHT = slice(6, 12)
RIGHT_GRIPPER = 11


def load_dataset(root: Path) -> tuple[dict, pd.DataFrame]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {root / 'data'}")
    cols = ["action", "observation.state", "episode_index", "frame_index", "index"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    return info, df.sort_values("index").reset_index(drop=True)


def arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (np.stack(df["action"].to_numpy()).astype(np.float64),
            np.stack(df["observation.state"].to_numpy()).astype(np.float64))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def stat(x: np.ndarray, fn) -> float | None:
    return float(fn(x)) if len(x) else None


def correlation_block(a: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    e = np.abs(a - s)[:, RIGHT].mean(1)
    d1 = np.abs(np.diff(a, axis=0))[:, RIGHT].mean(1)
    d2 = np.abs(np.diff(a, 2, axis=0))[:, RIGHT].mean(1)
    m = a[:, RIGHT].mean(1)
    e1 = e[1:]
    stable = d1 < threshold
    return {
        "frames": int(len(a)),
        "r_error_vs_change_rate": pearson(e1, d1),
        "r_error_vs_second_diff": pearson(e[2:], d2),
        "r_error_vs_amplitude": pearson(e, m),
        "stable_vs_abrupt": {
            "threshold": threshold,
            "stable_frames": int(stable.sum()),
            "stable_error_mean": stat(e1[stable], np.mean),
            "stable_error_max": stat(e1[stable], np.max),
            "abrupt_frames": int((~stable).sum()),
            "abrupt_error_mean": stat(e1[~stable], np.mean),
            "abrupt_error_max": stat(e1[~stable], np.max),
        },
    }


def phase_block(a: np.ndarray, s: np.ndarray) -> list[dict]:
    n = len(a)
    e = np.abs(a - s)[:, RIGHT].mean(1)
    d1 = np.concatenate([[0.0], np.abs(np.diff(a, axis=0))[:, RIGHT].mean(1)])
    m = a[:, RIGHT].mean(1)
    bounds = [0, n // 3, 2 * n // 3, n]
    names = ["coarse (first third)", "transition (middle third)", "fine (last third)"]
    out = []
    for i, name in enumerate(names):
        lo, hi = bounds[i], bounds[i + 1]
        out.append({"phase": name, "frame_start": lo, "frame_end": hi,
                    "change_rate_mean": float(d1[lo:hi].mean()),
                    "amplitude_std": float(m[lo:hi].std()),
                    "tracking_error_mean": float(e[lo:hi].mean())})
    return out


def chunk_block(a: np.ndarray, chunk: int, k: int) -> tuple[dict, list[dict]]:
    rows = []
    for c, start in enumerate(range(0, len(a) - chunk + 1, chunk)):
        acc = np.abs(np.diff(a[start:start + chunk, RIGHT], 2, axis=0)).sum(1)
        rows.append({"chunk": c, "frame_start": start,
                     f"first{k}": float(acc[:k].mean()), f"last{k}": float(acc[-k:].mean())})
    summ = {"complete_chunks": len(rows)}
    if rows:
        summ[f"first{k}_mean"] = float(np.mean([r[f"first{k}"] for r in rows]))
        summ[f"last{k}_mean"] = float(np.mean([r[f"last{k}"] for r in rows]))
        summ["last_vs_first_increase"] = summ[f"last{k}_mean"] / summ[f"first{k}_mean"] - 1.0
    return summ, rows


def gripper_block(a: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    g = a[:, RIGHT_GRIPPER]
    err = np.abs(a[:, RIGHT_GRIPPER] - s[:, RIGHT_GRIPPER])
    fine = g < threshold
    return {"joint": "right_gripper", "threshold": threshold, "frames": int(len(g)),
            "fine_frames": int(fine.sum()), "fine_fraction": float(fine.mean()),
            "fine_error_mean": stat(err[fine], np.mean), "fine_error_max": stat(err[fine], np.max),
            "coarse_frames": int((~fine).sum()),
            "coarse_error_mean": stat(err[~fine], np.mean), "coarse_error_max": stat(err[~fine], np.max)}


def data_checks(df: pd.DataFrame, episodes: dict) -> dict:
    a, s = arrays(df)
    joints = [i for i in range(12) if i not in (5, RIGHT_GRIPPER)]
    grip = [5, RIGHT_GRIPPER]
    return {
        "value_ranges": {
            "command_joints": [float(a[:, joints].min()), float(a[:, joints].max())],
            "measured_joints": [float(s[:, joints].min()), float(s[:, joints].max())],
            "command_grippers": [float(a[:, grip].min()), float(a[:, grip].max())],
            "measured_grippers": [float(s[:, grip].min()), float(s[:, grip].max())],
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"skip {path.name}: no rows")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def rounded(x):
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    return x


def plot_chunks(path: Path, per_episode: dict, k: int, chunk: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = sorted(e for e in per_episode if per_episode[e]["complete_chunks"])
    head = [per_episode[e][f"first{k}_mean"] for e in eps]
    tail = [per_episode[e][f"last{k}_mean"] for e in eps]
    x = np.arange(len(eps))
    w = 0.36
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.bar(x - w / 2, head, w, color="#2a78d6", label=f"first {k} values (chunk frames 0-{k + 1})")
    ax.bar(x + w / 2, tail, w, color="#eb6834", label=f"last {k} values (chunk frames {chunk - k - 2}-{chunk - 1})")
    for i, (h, t) in enumerate(zip(head, tail)):
        ax.text(i + w / 2, t + 0.15, f"{(t / h - 1) * 100:+.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"episode {e}" for e in eps])
    ax.set_ylabel("right-arm command jitter\n(sum |2nd diff|, servo units)")
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.savefig(path, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="LeRobot v3 dataset root")
    ap.add_argument("--corr-episode", type=int, default=0, help="episode for correlation and phase analysis")
    ap.add_argument("--change-threshold", type=float, default=0.5)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--edge-frames", type=int, default=10)
    ap.add_argument("--fine-threshold", type=float, default=20.0)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--plot", action="store_true", help="also write chunk_jitter.png")
    args = ap.parse_args()

    info, df = load_dataset(args.dataset)
    names = [n.removesuffix(".pos") for n in info["features"]["action"]["names"]]
    if len(names) != 12 or not all(n.startswith("right_") for n in names[RIGHT]) or names[RIGHT_GRIPPER] != "right_gripper":
        raise SystemExit(f"unexpected joint order in meta/info.json: {names}")
    episodes = {int(k): g for k, g in df.groupby("episode_index")}
    if args.corr_episode not in episodes:
        raise SystemExit(f"episode {args.corr_episode} not in dataset (episodes: {sorted(episodes)})")
    a0, s0 = arrays(episodes[args.corr_episode])
    a_all, s_all = arrays(df)

    chunk_summary, chunk_rows = {}, []
    for ep, g in episodes.items():
        a_ep = arrays(g)[0]
        summ, rows = chunk_block(a_ep, args.chunk_size, args.edge_frames)
        chunk_summary[ep] = summ
        chunk_rows += [{"episode": ep, **r} for r in rows]

    summary = {
        "dataset": args.dataset.name,
        "dataset_fps": info.get("fps"),
        "episodes": {ep: int(len(g)) for ep, g in episodes.items()},
        "total_frames": int(len(df)),
        "data_checks": data_checks(df, episodes),
        "correlation_episode": args.corr_episode,
        "correlation": correlation_block(a0, s0, args.change_threshold),
        "phases": phase_block(a0, s0),
        "chunk_size": args.chunk_size,
        "edge_frames": args.edge_frames,
        "chunk_edges": chunk_summary,
        "gripper": gripper_block(a_all, s_all, args.fine_threshold),
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "jitter_summary.json").write_text(
        json.dumps(rounded(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    write_csv(out / "chunk_edges.csv", chunk_rows)
    write_csv(out / "phases.csv", summary["phases"])
    if args.plot and chunk_rows:
        plot_chunks(out / "chunk_jitter.png", chunk_summary, args.edge_frames, args.chunk_size)

    c = summary["correlation"]
    print(f"episode {args.corr_episode}: r(error, change rate)={c['r_error_vs_change_rate']:.4f} "
          f"r(error, 2nd diff)={c['r_error_vs_second_diff']:.4f} r(error, amplitude)={c['r_error_vs_amplitude']:.4f}")
    sa = c["stable_vs_abrupt"]
    print(f"stable {sa['stable_frames']} frames err {sa['stable_error_mean']}/{sa['stable_error_max']}; "
          f"abrupt {sa['abrupt_frames']} frames err {sa['abrupt_error_mean']}/{sa['abrupt_error_max']}")
    for p in summary["phases"]:
        print(f"{p['phase']:26s} change={p['change_rate_mean']:.3f} amp_std={p['amplitude_std']:.2f} "
              f"err={p['tracking_error_mean']:.3f}")
    k = args.edge_frames
    for ep, sm in chunk_summary.items():
        if not sm["complete_chunks"]:
            print(f"episode {ep}: no complete chunk")
            continue
        print(f"episode {ep}: {sm['complete_chunks']} chunks first{k}={sm[f'first{k}_mean']:.2f} "
              f"last{k}={sm[f'last{k}_mean']:.2f} ({sm['last_vs_first_increase'] * 100:+.0f}%)")
    gr = summary["gripper"]
    print(f"right gripper command < {gr['threshold']:g}: {gr['fine_frames']}/{gr['frames']} = {gr['fine_fraction'] * 100:.1f}%")
    print("data checks:", json.dumps(rounded(summary["data_checks"])))


if __name__ == "__main__":
    main()
