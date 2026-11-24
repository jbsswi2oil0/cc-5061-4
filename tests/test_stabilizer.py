"""End-to-end checks for the stabilization kernel.

Run:  PYTHONPATH=/workspace python3 tests/test_stabilizer.py

The script uses the bundled synthetic sequences (generated if missing).
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stabilizer.geometry import (  # noqa: E402
    aff_compose,
    apply_aff,
    blend_toward_identity,
    coverage_zoom,
    cumulative_trajectory,
    decompose_similarity,
)
from stabilizer.metrics import (  # noqa: E402
    highfreq_report,
    pose_series_from_mats,
    trajectory_rmse_vs_gt,
)
from stabilizer.pipeline import (  # noqa: E402
    compute_plan,
    iter_warped_frames,
    plan_metrics,
    render_plan,
)
from stabilizer.synth import SAMPLE_DIR, generate_all  # noqa: E402
from stabilizer.tracking import analyze_video  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name} {detail}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


def ensure_samples():
    need = ["planar.mp4", "handheld_scene.mp4", "parallax_layers.mp4"]
    if not all(os.path.exists(os.path.join(SAMPLE_DIR, f)) for f in need):
        print("Generating samples ...")
        generate_all()


def test_geometry():
    print("[geometry]")
    M = aff_compose(12.5, -7.0, 0.03, 1.05)
    x, y, th, s = decompose_similarity(M)
    check("pose roundtrip tx", abs(x - 12.5) < 1e-9)
    check("pose roundtrip ty", abs(y + 7.0) < 1e-9)
    check("pose roundtrip theta", abs(th - 0.03) < 1e-12)
    check("pose roundtrip scale", abs(s - 1.05) < 1e-12)

    C = aff_compose(24, -18, 0.02, 1.0)
    z = coverage_zoom(C, 640, 360, safety=1.5)
    q = apply_aff(np.linalg.inv(
        np.array([[z, 0, 320 * (1 - z)], [0, z, 180 * (1 - z)], [0, 0, 1]])
        @ C
    ), np.array([[0, 0], [640, 0], [640, 360], [0, 360.0]]))
    check("coverage zoom >= 1", z >= 1.0, f"z={z:.3f}")
    check("coverage keeps corners inside",
          q[:, 0].min() >= 1.4 and q[:, 0].max() <= 638.6
          and q[:, 1].min() >= 1.4 and q[:, 1].max() <= 358.6,
          f"min={q.min(0).round(2)}")

    a = 0.4
    Cb = blend_toward_identity(C, a)
    xb, yb, tb, sb = decompose_similarity(Cb)
    check("alpha scales translation", abs(xb - 0.4 * 24) < 1e-9)
    check("alpha scales angle", abs(tb - 0.4 * 0.02) < 1e-12)
    check("alpha=0 identity",
          np.allclose(blend_toward_identity(C, 0.0), np.eye(3)))


def test_planar_accuracy_and_stabilization():
    print("[planar ground truth + stabilization]")
    video = os.path.join(SAMPLE_DIR, "planar.mp4")
    an = analyze_video(video)
    gt = json.load(open(os.path.join(SAMPLE_DIR, "planar_gt.json")))
    gt_mats = [np.array(m) for m in gt["mats"]]
    est = cumulative_trajectory(an.motions)
    rm = trajectory_rmse_vs_gt(est, gt_mats)
    check("trajectory translation RMSE < 2px",
          rm["translation_rmse_px"] < 2.0,
          f"rmse={rm['translation_rmse_px']:.2f}px")
    check("trajectory angle RMSE < 0.1deg",
          rm["angle_rmse_deg"] < 0.1, f"{rm['angle_rmse_deg']:.4f}")
    check("trajectory scale RMSE < 1%",
          rm["scale_rmse_pct"] < 1.0, f"{rm['scale_rmse_pct']:.3f}%")

    out_dir = os.path.join(ROOT, "runs", "test_planar")
    res = render_plan(an, out_dir, 60, 0.85, verify=True)
    m = res["metrics"]
    check("zero coverage violations", m["coverage_violations"] == 0,
          f"violations={m['coverage_violations']}")
    check("FOV never below declared 0.85",
          m["fov"]["min_retained_fov"] >= 0.849,
          f"min={m['fov']['min_retained_fov']:.4f}")
    red = res["verification"]["hf_reduction_pct"]
    check("output HF reduced by >= 80%", red >= 80.0, f"{red}%")
    check("output video exists",
          os.path.getsize(os.path.join(out_dir, "stabilized.mp4")) > 100_000)


def test_tight_fov_sacrifice_and_no_borders():
    print("[tight FOV adjudication + border invariant]")
    video = os.path.join(SAMPLE_DIR, "planar.mp4")
    an = analyze_video(video)
    plan = compute_plan(an, 80, 0.95)
    m = plan_metrics(an, plan)
    check("tight FOV respected",
          float(plan["fov"].min()) >= 0.949,
          f"min fov={plan['fov'].min():.4f} cap={1/plan['z_cap']:.3f}")
    check("some frames locally sacrificed",
          m["fov"]["sacrificed_frames"] > 0,
          f"n={m['fov']['sacrificed_frames']}")
    check("coverage violations zero", m["coverage_violations"] == 0)

    sentinel = (255, 0, 255)
    total_bad = 0
    frames = 0
    for _, fr in iter_warped_frames(
        video, an.width, an.height, plan["K"], border_value=sentinel
    ):
        d = np.abs(fr.astype(np.int16) - np.array(sentinel)).max(axis=2)
        total_bad += int(np.count_nonzero(d <= 12))
        frames += 1
    check("no sentinel border pixels in any frame", total_bad == 0,
          f"bad={total_bad} over {frames} frames")


def test_no_freeze_preserves_intent():
    print("[no-freeze: deliberate pan survives max smoothing]")
    video = os.path.join(SAMPLE_DIR, "planar.mp4")
    an = analyze_video(video)
    out_dir = os.path.join(ROOT, "runs", "test_maxsmooth")
    render_plan(an, out_dir, 100, 0.80, verify=False)
    out_an = analyze_video(os.path.join(out_dir, "stabilized.mp4"))
    pose = pose_series_from_mats(cumulative_trajectory(out_an.motions))
    low = np.convolve(pose[:, 0], np.ones(31) / 31, mode="same")
    span = float(np.percentile(low, 95) - np.percentile(low, 5))

    gt = json.load(open(os.path.join(SAMPLE_DIR, "planar_gt.json")))
    gt_mats = [np.array(x) for x in gt["mats"]]
    gt_rel = [g @ np.linalg.inv(gt_mats[0]) for g in gt_mats]
    gt_x = pose_series_from_mats(gt_rel)[:, 0]
    n = min(len(low), len(gt_x))
    corr = float(np.corrcoef(low[:n], gt_x[:n])[0, 1])
    check("slow pan still spans >= 25px at max smoothness", span >= 25,
          f"span={span:.1f}px")
    check("output low-freq path follows intended pan (corr>=0.6)",
          corr >= 0.6, f"corr={corr:.2f}")

    # frames are not frozen: consecutive output frames still differ
    cap = cv2.VideoCapture(os.path.join(out_dir, "stabilized.mp4"))
    diffs = []
    prev = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append(float(np.mean(cv2.absdiff(g, prev))))
        prev = g
    cap.release()
    check("output frames are not frozen", np.median(diffs) > 0.4,
          f"median absdiff={np.median(diffs):.2f}")


def _red_car_x(frame):
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    mask = ((r > 150) & (g < 120) & (b < 120)).astype(np.uint8)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    if stats[k, cv2.CC_STAT_AREA] < 120:
        return None
    return float(cent[k][0])


def test_moving_object_order():
    print("[temporal order: moving car keeps its direction]")
    src = os.path.join(SAMPLE_DIR, "handheld_scene.mp4")
    an = analyze_video(src)
    out_dir = os.path.join(ROOT, "runs", "test_handheld")
    render_plan(an, out_dir, 55, 0.85, verify=False)
    plan = compute_plan(an, 55, 0.85)
    xs = []
    for _, fr in iter_warped_frames(
        src, an.width, an.height, plan["K"]
    ):
        x = _red_car_x(fr)
        if x is not None:
            xs.append(x)
    check("car detected in many frames", len(xs) > 40, f"n={len(xs)}")
    if len(xs) > 10:
        t = np.arange(len(xs))
        slope = float(np.polyfit(t, xs, 1)[0])
        corr = float(np.corrcoef(t, xs)[0, 1])
        check("car moves monotonically across stabilized frames",
              slope > 1.5 and corr > 0.7,
              f"slope={slope:.2f} corr={corr:.2f}")


def test_parallax_flag():
    print("[parallax residual separation]")
    plan = os.path.join(SAMPLE_DIR, "planar.mp4")
    par = os.path.join(SAMPLE_DIR, "parallax_layers.mp4")
    a0 = analyze_video(plan)
    a1 = analyze_video(par)
    from stabilizer.pipeline import plan_metrics as pm
    lvl0 = pm(a0, compute_plan(a0, 55, 0.85))["parallax"]["level"]
    lvl1 = pm(a1, compute_plan(a1, 55, 0.85))["parallax"]["level"]
    check("planar clip not flagged high", lvl0 in ("low", "medium"), lvl0)
    check("parallax clip flagged high", lvl1 == "high", lvl1)


if __name__ == "__main__":
    ensure_samples()
    test_geometry()
    test_planar_accuracy_and_stabilization()
    test_tight_fov_sacrifice_and_no_borders()
    test_no_freeze_preserves_intent()
    test_moving_object_order()
    test_parallax_flag()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
