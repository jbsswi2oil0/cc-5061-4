"""Quantitative comparison: known-shake synthetic + authored short + parallax.

For every bundled clip and smoothing setting we record:
  * cumulative-path RMSE against ground truth (generated clips only);
  * high-frequency path energy before stabilization and measured from the
    re-encoded stabilized video (end-to-end, not just the preview curve);
  * retained field of view and local-stability-sacrifice fraction;
  * reprojection / model residuals and the parallax verdict;
  * moving-object direction preservation for the street clip.

Run:  python scripts/run_benchmark.py
Writes reports/benchmark.json and prints a summary table.
"""

from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stabilizer.metrics import trajectory_rmse_vs_gt  # noqa: E402
from stabilizer.pipeline import render_plan  # noqa: E402
from stabilizer.synth import SAMPLE_INFO, SAMPLE_DIR, generate_all  # noqa: E402
from stabilizer.tracking import analyze_video  # noqa: E402
from stabilizer.videoio import ffmpeg_exe  # noqa: E402

SETTINGS = [
    {"id": "off", "label": "关闭平滑", "smoothness": 0, "min_fov": 0.85},
    {"id": "mid", "label": "标准 55", "smoothness": 55, "min_fov": 0.85},
    {"id": "max", "label": "最强 100", "smoothness": 100, "min_fov": 0.80},
    {"id": "tightfov", "label": "高视野下限 95%", "smoothness": 80, "min_fov": 0.95},
]


def car_trend(video_path: str):
    cap = cv2.VideoCapture(video_path)
    xs = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        b, g, r = fr[:, :, 0], fr[:, :, 1], fr[:, :, 2]
        mask = ((r > 150) & (g < 120) & (b < 120)).astype("uint8")
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        if n > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[k, cv2.CC_STAT_AREA] >= 120:
                xs.append(float(cent[k][0]))
    cap.release()
    if len(xs) < 12:
        return {"detected_frames": len(xs)}
    t = np.arange(len(xs))
    return {
        "detected_frames": len(xs),
        "slope_px_per_frame": round(float(np.polyfit(t, xs, 1)[0]), 3),
        "trend_corr": round(float(np.corrcoef(t, xs)[0, 1]), 3),
    }


def main():
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    if not all(
        os.path.exists(os.path.join(SAMPLE_DIR, SAMPLE_INFO[k]["file"]))
        for k in SAMPLE_INFO
    ):
        generate_all()

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ffmpeg": os.path.basename(ffmpeg_exe()),
        "settings": SETTINGS,
        "clips": {},
    }

    for key, info in SAMPLE_INFO.items():
        video = os.path.join(SAMPLE_DIR, info["file"])
        gt_path = os.path.join(SAMPLE_DIR, info["gt"])
        print(f"\n=== {key}: {info['title']} ===")
        an = analyze_video(video)
        clip_rep = {
            "title": info["title"],
            "description": info["desc"],
            "meta": an.summary(),
            "gt_rmse": None,
            "parallax": None,
            "settings": {},
            "moving_object": None,
        }
        if os.path.exists(gt_path):
            gt = json.load(open(gt_path, encoding="utf-8"))
            gt_mats = [np.array(m) for m in gt["mats"]]
            from stabilizer.geometry import cumulative_trajectory

            est = cumulative_trajectory(an.motions)
            clip_rep["gt_rmse"] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in trajectory_rmse_vs_gt(est, gt_mats).items()
            }

        for st in SETTINGS:
            out_dir = os.path.join(ROOT, "runs", "benchmark", key, st["id"])
            t0 = time.time()
            res = render_plan(
                an,
                out_dir,
                st["smoothness"],
                st["min_fov"],
                verify=True,
            )
            dt = time.time() - t0
            m = res["metrics"]
            v = res.get("verification") or {}
            clip_rep["parallax"] = m["parallax"]
            row = {
                "elapsed_s": round(dt, 1),
                "coverage_violations": m["coverage_violations"],
                "fov": {k: (round(x, 4) if isinstance(x, float) else x)
                        for k, x in m["fov"].items()},
                "orig_hf_index_px": round(m["orig_hf"]["hf_index_px"], 3),
                "target_hf_index_px": round(m["target_hf"]["hf_index_px"], 4),
                "output_hf_index_px": round(
                    v.get("output_hf", {}).get("hf_index_px", float("nan")), 3
                ),
                "hf_reduction_pct": v.get("hf_reduction_pct"),
                "output_angle_d2_deg": round(
                    v.get("output_hf", {})
                    .get("second_diff_rms", {})
                    .get("angle_deg", float("nan")),
                    4,
                ),
                "output_inlier_rmse_px": (
                    round(v.get("analysis", {}).get("mean_inlier_rmse_px"), 3)
                    if v.get("analysis") else None
                ),
            }
            clip_rep["settings"][st["id"]] = row
            print(
                f"  {st['label']:14s} HF {row['orig_hf_index_px']:7.2f} -> "
                f"{row['output_hf_index_px']:6.3f} "
                f"({row['hf_reduction_pct']}%)  "
                f"FOVmin {m['fov']['min_retained_fov']*100:5.1f}%  "
                f"sac {m['fov']['sacrificed_frames']:3d}f  "
                f"viol {m['coverage_violations']}"
            )

        if key == "handheld_scene":
            mid_video = os.path.join(
                ROOT, "runs", "benchmark", key, "mid", "stabilized.mp4"
            )
            clip_rep["moving_object"] = car_trend(mid_video)
            print("  car trend:", clip_rep["moving_object"])

        report["clips"][key] = clip_rep

    out = os.path.join(ROOT, "reports", "benchmark.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
