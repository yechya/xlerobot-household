#!/usr/bin/env python3
"""Summarize a SmolVLA PyTorch-vs-TensorRT real-data verification result.

Input is the JSON written by `scripts/verify_trt_with_real_data.py` from the LeRobot Jetson
integration patch (integrations/lerobot_jetson). That script runs the deployed inference path
on N recorded observations once with PyTorch and once with the TensorRT expert-decoder engine,
and stores per-frame latency and action-difference metrics. This script recomputes the
aggregate numbers from the per-frame records and checks them against the stored summary.

Example:
    python summarize_smolvla_trt.py raw/smolvla_pick_up_bottle_verify_real_data.json --out-dir . --plot
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FIELDS = ["frame", "py_time_ms", "trt_time_ms", "cosine_similarity", "relative_l2", "mean_abs_diff", "max_abs_diff"]


def summarize(res: dict) -> tuple[dict, list[dict]]:
    pf = res["per_frame"]
    if len(pf) < 2:
        raise SystemExit("need at least 2 per-frame records")
    col = {k: np.array([f[k] for f in pf], dtype=np.float64) for k in FIELDS}
    py, trt = col["py_time_ms"].mean(), col["trt_time_ms"].mean()
    out = {
        "frames": len(pf),
        "py_time_ms_mean": py,
        "trt_time_ms_mean": trt,
        "speedup": py / trt,
        "trt_time_ms_frame0": float(col["trt_time_ms"][0]),
        "trt_time_ms_mean_excl_frame0": float(col["trt_time_ms"][1:].mean()),
        "py_time_ms_mean_excl_frame0": float(col["py_time_ms"][1:].mean()),
        "cosine_similarity_mean": float(col["cosine_similarity"].mean()),
        "cosine_similarity_worst": float(col["cosine_similarity"].min()),
        "relative_l2_mean": float(col["relative_l2"].mean()),
        "relative_l2_worst": float(col["relative_l2"].max()),
        "mean_abs_diff_mean": float(col["mean_abs_diff"].mean()),
        "max_abs_diff_worst": float(col["max_abs_diff"].max()),
    }
    stored = res.get("summary", {})
    checks = {
        "py_time_ms": (stored.get("py_time_ms"), py),
        "trt_time_ms": (stored.get("trt_time_ms"), trt),
        "speedup": (stored.get("speedup"), py / trt),
        "cosine_similarity.mean": (stored.get("cosine_similarity", {}).get("mean"), out["cosine_similarity_mean"]),
        "cosine_similarity.worst": (stored.get("cosine_similarity", {}).get("worst"), out["cosine_similarity_worst"]),
        "relative_l2.mean": (stored.get("relative_l2", {}).get("mean"), out["relative_l2_mean"]),
        "relative_l2.worst": (stored.get("relative_l2", {}).get("worst"), out["relative_l2_worst"]),
        "mean_abs_diff.mean": (stored.get("mean_abs_diff", {}).get("mean"), out["mean_abs_diff_mean"]),
    }
    out["matches_stored_summary"] = all(s is not None and abs(s - r) <= 1e-9 * max(1.0, abs(s))
                                        for s, r in checks.values())
    out["config"] = res.get("config", {})
    return out, [{k: f[k] for k in FIELDS} for f in pf]


def plot(path: Path, rows: list[dict], s: dict, trt_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["frame"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.plot(x, [r["py_time_ms"] for r in rows], color="#2a78d6", lw=2, marker="o", ms=4, label="PyTorch")
    ax.plot(x, [r["trt_time_ms"] for r in rows], color="#eb6834", lw=2, marker="s", ms=4,
            label=trt_label)
    top = 1.15 * max(max(r["py_time_ms"] for r in rows), max(r["trt_time_ms"] for r in rows))
    ax.text(x[-1], s["py_time_ms_mean"] + 0.03 * top, f"mean {s['py_time_ms_mean']:.1f} ms", ha="right", fontsize=8)
    ax.text(x[-1], s["trt_time_ms_mean"] + 0.03 * top, f"mean {s['trt_time_ms_mean']:.1f} ms", ha="right", fontsize=8)
    ax.annotate("frame 0 (first engine run)", xy=(0, rows[0]["trt_time_ms"]),
                xytext=(0.08 * len(rows), 0.5 * (s["py_time_ms_mean"] + s["trt_time_ms_mean"])), fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.8))
    ax.set_xlabel("observation index")
    ax.set_ylabel("inference latency / ms")
    ax.set_ylim(0, top)
    ax.set_xticks(range(0, len(x), 2))
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=2)
    fig.savefig(path, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="verify_real_data_results JSON")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--name", default=None, help="output file prefix (default: input file stem)")
    ap.add_argument("--plot", action="store_true", help="also write <name>_latency.png")
    ap.add_argument("--trt-label", default="TensorRT", help="legend label of the TensorRT series")
    args = ap.parse_args()

    res = json.loads(args.results.read_text(encoding="utf-8"))
    summary, rows = summarize(res)
    name = args.name or args.results.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    with open(args.out_dir / f"{name}_per_frame.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    if args.plot:
        plot(args.out_dir / f"{name}_latency.png", rows, summary, args.trt_label)

    s = summary
    print(f"frames={s['frames']} PyTorch {s['py_time_ms_mean']:.1f} ms -> TensorRT {s['trt_time_ms_mean']:.1f} ms "
          f"({s['speedup']:.2f}x); TRT frame 0 = {s['trt_time_ms_frame0']:.0f} ms, "
          f"TRT mean without frame 0 = {s['trt_time_ms_mean_excl_frame0']:.1f} ms")
    print(f"cosine mean {s['cosine_similarity_mean']:.7f} worst {s['cosine_similarity_worst']:.7f}; "
          f"rel L2 mean {s['relative_l2_mean']:.2e} worst {s['relative_l2_worst']:.2e}; "
          f"mean abs diff {s['mean_abs_diff_mean']:.3f}; matches stored summary: {s['matches_stored_summary']}")


if __name__ == "__main__":
    main()
